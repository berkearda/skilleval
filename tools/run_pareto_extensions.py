"""Pareto extensions: K-sweep + ensemble + beta calibration.

Adds three new lines of curves to the existing v3/v4 Pareto:
  1. CDMEval at K ∈ {50, 150, 200, 300, 400, 500} canonical-training (single-seed
     checkpoints text_conditioned_K{K}.pt). Tests whether higher K closes the
     EmbedLLM gap.
  2. CDMEval (60% train) + EmbedLLM (60% train) logit-space ensemble via
     logistic regression fit on val. Per agent recommendation: highest leverage,
     non-monotone, can actually improve AUC.
  3. CDMEval beta-calibrated (alternative to isotonic that doesn't compress
     range — fixes the "cdmeval_calibrated saturates at 83.5%" problem).

T-035 / T-036 / T-037 audit-first plan. the project log 2026-05-01.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from betacal import BetaCalibration

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"
CKPT_DIR = REPO / "cdm_exploration" / "checkpoints" / "expanded"


def threshold_sweep(preds, R_test, flops, thresholds):
    n_llms, n_test = preds.shape
    out = []
    for t in thresholds:
        correct = 0
        total_cost = 0.0
        picks = []
        n_fallback = 0
        for j in range(n_test):
            p = preds[:, j]
            above = np.where(p > t)[0]
            if len(above) > 0:
                chosen = above[np.argmin(flops[above])]
            else:
                chosen = int(np.argmax(p))
                n_fallback += 1
            picks.append(chosen)
            total_cost += flops[chosen]
            if R_test[chosen, j] > 0:
                correct += 1
        out.append({
            "threshold": float(t),
            "accuracy": correct / n_test,
            "mean_cost": total_cost / n_test,
            "n_unique_picks": int(len(np.unique(picks))),
            "fallback_rate": n_fallback / n_test,
        })
    return out


def cdmeval_predictions(ckpt_path, K, test_idx, n_llms, emb, q_matrix):
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = TextConditionedNet(K, n_llms, text_dim=768)
    if "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        net.load_state_dict(ckpt)  # raw state dict
    net.eval()
    test_emb = torch.tensor(emb[test_idx], dtype=torch.float32)
    test_q = torch.tensor(q_matrix[test_idx], dtype=torch.float32)
    all_ids = torch.arange(n_llms)
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    with torch.no_grad():
        for j in range(len(test_idx)):
            te = test_emb[j].unsqueeze(0).expand(n_llms, -1)
            tq = test_q[j].unsqueeze(0).expand(n_llms, -1)
            preds[:, j] = net(all_ids, te, tq).numpy()
    return preds


def ece(preds, true, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    pf = preds.flatten()
    tf = true.flatten()
    out = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pf >= lo) & (pf < hi) if i < n_bins - 1 else (pf >= lo) & (pf <= hi)
        if mask.sum() == 0:
            continue
        out += (mask.sum() / len(pf)) * abs(pf[mask].mean() - tf[mask].mean())
    return float(out)


def normalize(sweep, strongest_acc, strongest_cost):
    return [{**p,
              "cost_pct": p["mean_cost"] / strongest_cost * 100,
              "acc_pct": p["accuracy"] / strongest_acc * 100}
             for p in sweep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_kswep", action="store_true",
                    help="Skip K-sweep curves (faster reruns)")
    args = ap.parse_args()

    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.utils.device import seed_everything

    seed_everything(42)

    print("=== load data ===", flush=True)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(DATA / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    n_llms, n_items = R.shape
    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    R_test = R[:, test_idx]
    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])

    print(f"  R={R.shape}  test={len(test_idx)}", flush=True)

    # Load existing v4 pareto for thresholds + strongest reference
    v4 = json.load(open(EXP / "v2_pareto_multibaseline_v4.json"))
    thresholds = v4["thresholds"]
    strongest_acc = v4["strongest"]["test_acc"]
    strongest_cost = v4["strongest"]["cost_gflops"]

    new_methods = {}

    # ── (1) K-sweep ──
    if not args.skip_kswep:
        print("\n=== (1) K-sweep CDMEval (canonical 80% train, single seed) ===",
                flush=True)
        for K in [50, 150, 200, 300, 400, 500]:
            ckpt = CKPT_DIR / f"text_conditioned_K{K}.pt"
            qmat_path = DATA / f"qmatrix_v2_K{K}.npy"
            if not ckpt.exists() or not qmat_path.exists():
                print(f"  SKIP K={K}: missing checkpoint or q-matrix", flush=True)
                continue
            print(f"  K={K} ... ", end="", flush=True)
            q_matrix_K = np.load(qmat_path)
            preds = cdmeval_predictions(ckpt, K, test_idx, n_llms, emb, q_matrix_K)
            auc = float(roc_auc_score(R_test.flatten(), preds.flatten()))
            ece_v = ece(preds, R_test)
            sweep = threshold_sweep(preds, R_test, flops, thresholds)
            print(f"AUC={auc:.4f}  ECE={ece_v:.4f}", flush=True)
            new_methods[f"cdmeval_K{K}"] = {
                "sweep": sweep,
                "normalized": normalize(sweep, strongest_acc, strongest_cost),
                "test_auc": auc, "test_ece": ece_v,
            }
    else:
        print("\n=== SKIP K-sweep ===", flush=True)

    # ── (2) Ensemble: CDMEval(60%) + EmbedLLM(60%) ──
    print("\n=== (2) Ensemble: CDMEval(60%) + EmbedLLM(60%) logit stack ===",
            flush=True)
    cdm_npz = np.load(EXP / "v2_cdmeval_calibrated_predictions_s42.npz")
    el_npz = np.load(EXP / "v2_embedllm_calibrated_d232_s42_predictions.npz")
    val_idx_cdm = cdm_npz["val_idx"]
    val_idx_el = el_npz["val_idx"]
    if not np.array_equal(val_idx_cdm, val_idx_el):
        print("  WARNING: val_idx mismatch CDM/EL — EmbedLLM was trained on a "
                "different split. Skipping ensemble; rerun extensions after "
                "EmbedLLM canonical-split retraining completes.", flush=True)
        skip_ensemble = True
    else:
        skip_ensemble = False
    if not skip_ensemble:
        assert np.array_equal(cdm_npz["test_idx"], el_npz["test_idx"]), "test_idx mismatch"

    cdm_val = cdm_npz["val_preds"]   # (n_llms, n_val)
    cdm_test = cdm_npz["test_preds"]
    R_val = cdm_npz["R_val"]
    R_test_60 = cdm_npz["R_test"]

    if skip_ensemble:
        # Save what we have and exit gracefully
        v4["methods"].update(new_methods)
        out_path = EXP / "v2_pareto_multibaseline_v5.json"
        with open(out_path, "w") as f:
            json.dump(v4, f, indent=2)
        print(f"\nwrote {out_path} (without ensemble — waiting on EmbedLLM "
                "rerun)", flush=True)
        # Run beta calibration on CDMEval only and exit
        print("\n=== (3) Beta calibration on CDMEval (60% train) ===", flush=True)
        bc = BetaCalibration(parameters="abm")
        bc.fit(cdm_val.flatten().reshape(-1, 1), R_val.flatten())
        cdm_test_beta = bc.predict(cdm_test.flatten().reshape(-1, 1)).reshape(cdm_test.shape)
        cdm_val_beta = bc.predict(cdm_val.flatten().reshape(-1, 1)).reshape(cdm_val.shape)
        bc_test_auc = float(roc_auc_score(R_test_60.flatten(), cdm_test_beta.flatten()))
        bc_val_ece = ece(cdm_val_beta, R_val)
        bc_test_ece = ece(cdm_test_beta, R_test_60)
        print(f"  CDMEval 60% + beta-cal: val ECE={bc_val_ece:.4f}  "
                f"test ECE={bc_test_ece:.4f}  test AUC={bc_test_auc:.4f}",
                flush=True)
        print(f"  pred range after beta: [{cdm_test_beta.min():.3f}, "
                f"{cdm_test_beta.max():.3f}]", flush=True)
        sweep = threshold_sweep(cdm_test_beta, R_test_60, flops, thresholds)
        new_methods["cdmeval_beta_calibrated"] = {
            "sweep": sweep,
            "normalized": normalize(sweep, strongest_acc, strongest_cost),
            "test_auc": bc_test_auc, "test_ece": bc_test_ece,
            "val_ece": bc_val_ece,
            "pred_range": [float(cdm_test_beta.min()), float(cdm_test_beta.max())],
        }
        v4["methods"].update(new_methods)
        with open(out_path, "w") as f:
            json.dump(v4, f, indent=2)
        print(f"\nrewrote {out_path} (with K-sweep + beta-cal, no ensemble)",
                flush=True)
        # Summary
        print("\n=== Summary of NEW methods (no ensemble) ===", flush=True)
        for name, meta in new_methods.items():
            print(f"  {name:>30}  AUC={meta['test_auc']:.4f}  ECE={meta['test_ece']:.4f}",
                    flush=True)
        return 0

    el_val = el_npz["val_preds"]
    el_test = el_npz["test_preds"]
    # Stack features: (cdm_val.flatten(), el_val.flatten()) → R_val.flatten()
    print(f"  fitting LogisticRegression on val "
            f"({R_val.size:,} pairs, 2 features)...", flush=True)
    X_val = np.stack([cdm_val.flatten(), el_val.flatten()], axis=1)
    y_val = R_val.flatten().astype(int)
    stacker = LogisticRegression(max_iter=200, C=1.0)
    stacker.fit(X_val, y_val)
    print(f"  weights: cdm={stacker.coef_[0, 0]:.3f}, "
            f"el={stacker.coef_[0, 1]:.3f}, b={stacker.intercept_[0]:.3f}",
            flush=True)

    X_test = np.stack([cdm_test.flatten(), el_test.flatten()], axis=1)
    ens_test_preds = stacker.predict_proba(X_test)[:, 1].reshape(cdm_test.shape)
    R_test_60 = cdm_npz["R_test"]
    ens_auc = float(roc_auc_score(R_test_60.flatten(), ens_test_preds.flatten()))
    ens_ece = ece(ens_test_preds, R_test_60)
    print(f"  ensemble: AUC={ens_auc:.4f}  ECE={ens_ece:.4f}", flush=True)
    print(f"  baseline AUC: cdm={float(roc_auc_score(R_test_60.flatten(), cdm_test.flatten())):.4f}  "
            f"el={float(roc_auc_score(R_test_60.flatten(), el_test.flatten())):.4f}",
            flush=True)

    sweep = threshold_sweep(ens_test_preds, R_test_60, flops, thresholds)
    new_methods["ensemble_cdm_embedllm"] = {
        "sweep": sweep,
        "normalized": normalize(sweep, strongest_acc, strongest_cost),
        "test_auc": ens_auc, "test_ece": ens_ece,
        "stacker_weights": {"cdm": float(stacker.coef_[0, 0]),
                              "embedllm": float(stacker.coef_[0, 1]),
                              "intercept": float(stacker.intercept_[0])},
    }

    # ── Constrained ensemble: 0.5/0.5 (interpretability defense) ──
    print("\n=== (2b) Constrained 0.5/0.5 ensemble (interpretability defense) ===",
            flush=True)
    ens_uniform = (cdm_test + el_test) / 2.0
    u_auc = float(roc_auc_score(R_test_60.flatten(), ens_uniform.flatten()))
    u_ece = ece(ens_uniform, R_test_60)
    print(f"  uniform 0.5/0.5: AUC={u_auc:.4f}  ECE={u_ece:.4f}", flush=True)
    sweep = threshold_sweep(ens_uniform, R_test_60, flops, thresholds)
    new_methods["ensemble_uniform_5050"] = {
        "sweep": sweep,
        "normalized": normalize(sweep, strongest_acc, strongest_cost),
        "test_auc": u_auc, "test_ece": u_ece,
    }

    # ── (3) Beta calibration on CDMEval (60% train) ──
    print("\n=== (3) Beta calibration on CDMEval (60% train) ===", flush=True)
    bc = BetaCalibration(parameters="abm")
    bc.fit(cdm_val.flatten().reshape(-1, 1), R_val.flatten())
    cdm_test_beta = bc.predict(cdm_test.flatten().reshape(-1, 1)).reshape(cdm_test.shape)
    cdm_val_beta = bc.predict(cdm_val.flatten().reshape(-1, 1)).reshape(cdm_val.shape)
    bc_test_auc = float(roc_auc_score(R_test_60.flatten(), cdm_test_beta.flatten()))
    bc_val_ece = ece(cdm_val_beta, R_val)
    bc_test_ece = ece(cdm_test_beta, R_test_60)
    print(f"  CDMEval 60% + beta-cal: val ECE={bc_val_ece:.4f}  "
            f"test ECE={bc_test_ece:.4f}  test AUC={bc_test_auc:.4f}",
            flush=True)
    print(f"  pred range after beta: [{cdm_test_beta.min():.3f}, "
            f"{cdm_test_beta.max():.3f}]", flush=True)
    sweep = threshold_sweep(cdm_test_beta, R_test_60, flops, thresholds)
    new_methods["cdmeval_beta_calibrated"] = {
        "sweep": sweep,
        "normalized": normalize(sweep, strongest_acc, strongest_cost),
        "test_auc": bc_test_auc, "test_ece": bc_test_ece,
        "val_ece": bc_val_ece,
        "pred_range": [float(cdm_test_beta.min()), float(cdm_test_beta.max())],
    }

    # ── Save ──
    v4["methods"].update(new_methods)
    out_path = EXP / "v2_pareto_multibaseline_v5.json"
    with open(out_path, "w") as f:
        json.dump(v4, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # ── Summary ──
    print("\n=== Summary of ALL new methods ===", flush=True)
    print(f"  {'method':>30}  {'AUC':>6}  {'ECE':>6}", flush=True)
    for name, d in new_methods.items():
        print(f"  {name:>30}  {d['test_auc']:>6.4f}  {d['test_ece']:>6.4f}",
                flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
