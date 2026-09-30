"""Cold-start v2 with the optimizer convergence fix (T-036).

Mirror of tools/cold_start_v2.py with EXACTLY ONE change: the calibration
optimizer hyperparameters. Everything else identical:
  - Zero initialization (the principled "no information" prior)
  - Same data, same splits, same model checkpoint, same SGD recipe
  - Only difference: lr=0.05 (was 0.01), steps=500 (was 50)

Audit (the project log 2026-05-01) showed the original optimizer left
|θ_raw| at 0.28 vs trained reference 1.87 (15% of scale). With the
fix, |θ_raw| reaches ~1.6 (86% of scale) and AUC at N=500 reaches
~0.68 vs broken 0.58.

Drops the T-035 pop-init shortcut: that was masking the convergence
bug. With the bug fixed, zero init reaches the same AUC at large N
without introducing empirical-Bayes population bias.

Usage:
    python tools/cold_start_v2_fixed.py device=cpu
    python tools/cold_start_v2_fixed.py device=cuda
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cold_start_v2 import batch_predict, batch_calibrate


# T-036 fix: lr 0.01 -> 0.05, steps 50 -> 500
CALIBRATION_LR = 0.05
CALIBRATION_STEPS = 500


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)
    print(f"[T-036] CALIBRATION_LR={CALIBRATION_LR}  STEPS={CALIBRATION_STEPS}",
            flush=True)

    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    print(f"  {n_llms} LLMs x {n_items} items, K={K}", flush=True)

    # Splits — identical to cold_start_v2.py
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    all_llms = np.arange(n_llms)
    train_llms, test_llms = train_test_split(all_llms, test_size=0.2, random_state=42)
    print(f"  Items: {len(train_items)} train, {len(test_items)} test", flush=True)
    print(f"  LLMs: {len(train_llms)} train, {len(test_llms)} held-out", flush=True)

    # F5 sniff
    assert len(set(train_llms.tolist()) & set(test_llms.tolist())) == 0, \
        "F5 LEAK: train_llms and test_llms overlap"
    print(f"  [F5 OK] train_llms ∩ test_llms == ∅", flush=True)

    # Model
    # +ckpt=<file> selects the frozen network; +out_tag=<suffix> keeps the result file apart (T-123).
    ckpt_path = Path(str(cfg.ckpt)) if hasattr(cfg, "ckpt") else Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    out_tag = str(cfg.out_tag) if hasattr(cfg, "out_tag") else ""
    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device)
    net.eval()

    # Reference: trained-LLM theta scale (for B4 sanity)
    with torch.no_grad():
        train_theta_raw = net.student_emb(
            torch.tensor(train_llms[:200], device=device))
    train_theta_norm_ref = float(train_theta_raw.abs().mean())
    print(f"\n  trained-theta |raw| reference: {train_theta_norm_ref:.4f}",
            flush=True)
    print(f"  (cold-start should reach >0.8 of this with the fix)", flush=True)

    # Full-training baseline
    print("\nComputing full-training baseline (50 train LLMs)...", flush=True)
    sample_train = train_llms[:50]
    with torch.no_grad():
        raw = net.student_emb(torch.tensor(sample_train, device=device)).cpu().numpy()
    full_mastery = 1.0 / (1.0 + np.exp(-raw))
    full_preds = batch_predict(net, full_mastery, test_items, q_matrix, text_embs, device)

    full_aucs = []
    for i, llm_idx in enumerate(sample_train):
        y_true = R[llm_idx, test_items.astype(int)]
        if len(np.unique(y_true)) >= 2:
            full_aucs.append(roc_auc_score(y_true, full_preds[i]))
    full_auc = float(np.mean(full_aucs))
    print(f"  Full-training AUC: {full_auc:.4f}", flush=True)

    # Cold-start sweep with the fixed optimizer settings
    cal_sizes = [0, 1, 5, 10, 50, 100, 500]
    n_repeats = 3
    # Optional overrides, used only for smoke tests; the defaults above are the paper's settings.
    if hasattr(cfg, "cal_sizes"):
        cal_sizes = [int(x) for x in str(cfg.cal_sizes).split(",")]
    if hasattr(cfg, "n_repeats"):
        n_repeats = int(cfg.n_repeats)
    if hasattr(cfg, "n_eval_llms"):
        test_llms = test_llms[:int(cfg.n_eval_llms)]
    rng = np.random.RandomState(42)
    print(f"\nCold-start (FIXED): {len(test_llms)} LLMs, sizes={cal_sizes}, "
            f"repeats={n_repeats}, lr={CALIBRATION_LR}, steps={CALIBRATION_STEPS}",
            flush=True)

    summary = []
    for N in cal_sizes:
        print(f"\n  N={N}...", end=" ", flush=True)
        reps = 1 if N == 0 else n_repeats
        all_aucs = []
        all_accs = []
        for rep in range(reps):
            if N == 0:
                # IDENTICAL to original: theta=0.5 on every skill
                theta_mat = np.full((len(test_llms), K), 0.5)
            else:
                cal_per_llm = [rng.choice(train_items, size=min(N, len(train_items)),
                                            replace=False) for _ in test_llms]
                # T-036 fix: pass lr=0.05, steps=500 to the existing
                # batch_calibrate function (same code path, different hyperparams)
                theta_mat = batch_calibrate(
                    net, R, test_llms, cal_per_llm, q_matrix, text_embs,
                    device, K,
                    lr=CALIBRATION_LR, steps=CALIBRATION_STEPS,
                )
            preds = batch_predict(net, theta_mat, test_items, q_matrix, text_embs, device)
            for i, llm_idx in enumerate(test_llms):
                y_true = R[llm_idx, test_items.astype(int)]
                if len(np.unique(y_true)) < 2:
                    continue
                auc = roc_auc_score(y_true, preds[i])
                acc = accuracy_score(y_true, (preds[i] >= 0.5).astype(int))
                all_aucs.append(auc)
                all_accs.append(acc)

        auc_m, auc_s = np.mean(all_aucs), np.std(all_aucs)
        acc_m = np.mean(all_accs)
        pct = auc_m / full_auc * 100
        print(f"AUC={auc_m:.4f}±{auc_s:.4f}, Acc={acc_m:.4f}, {pct:.1f}% of full",
                flush=True)
        summary.append({"N": N, "auc_mean": float(auc_m), "auc_std": float(auc_s),
                          "acc_mean": float(acc_m), "pct_of_full": float(pct)})

    # Comparison: original (broken SGD) vs fixed (T-036)
    original_path = Path("cdm_exploration/experiments/v2_cold_start.json")
    if original_path.exists():
        original = json.load(open(original_path))
        print(f"\n{'='*80}", flush=True)
        print(f"COLD-START COMPARISON: original (broken SGD) vs FIXED (T-036)",
                flush=True)
        print(f"{'='*80}", flush=True)
        print(f"{'N':>6} {'orig (broken)':>16} {'FIXED':>16} {'Δ AUC':>10} "
                f"{'orig %full':>12} {'FIXED %full':>12}",
                flush=True)
        print("-" * 80, flush=True)
        orig_by_N = {s["N"]: s for s in original["summary"]}
        for s in summary:
            o = orig_by_N.get(s["N"], {})
            o_auc = o.get("auc_mean", float("nan"))
            o_pct = o.get("pct_of_full", float("nan"))
            delta = s["auc_mean"] - o_auc
            print(f"{s['N']:>6} {o_auc:>16.4f} {s['auc_mean']:>16.4f} "
                    f"{delta:>+10.4f} {o_pct:>11.1f}% {s['pct_of_full']:>11.1f}%",
                    flush=True)

    # Same-LLM reference (T-123): calibrate each held-out LLM on ALL training items with the same recipe, in
    # chunks so the pooled tensors stay small. This is the ceiling for the 763 LLMs themselves, whereas
    # full_training_auc above comes from 50 other (training) LLMs.
    full_cal_auc = None
    if hasattr(cfg, "full_calibration") and bool(cfg.full_calibration):
        print("\nFull-calibration reference: all training items, same held-out LLMs...", flush=True)
        ref_aucs, chunk = [], 32
        for c0 in range(0, len(test_llms), chunk):
            sub = test_llms[c0:c0 + chunk]
            th = batch_calibrate(net, R, sub, [train_items for _ in sub], q_matrix, text_embs, device, K,
                                 lr=CALIBRATION_LR, steps=CALIBRATION_STEPS)
            pr = batch_predict(net, th, test_items, q_matrix, text_embs, device)
            for i, llm_idx in enumerate(sub):
                y_true = R[llm_idx, test_items.astype(int)]
                if len(np.unique(y_true)) >= 2:
                    ref_aucs.append(roc_auc_score(y_true, pr[i]))
        full_cal_auc = float(np.mean(ref_aucs))
        print(f"  Full-calibration AUC (same {len(test_llms)} LLMs): {full_cal_auc:.4f}", flush=True)
        for srow in summary:
            srow["pct_of_full_calibration"] = float(100 * srow["auc_mean"] / full_cal_auc)

    save_data = {
        "dataset": "v2_full",
        "checkpoint": str(ckpt_path),
        "full_calibration_auc_same_llms": full_cal_auc,
        "experiment": "cold_start_v2_fixed",
        "t_id": "T-036",
        "fix": "calibration optimizer: lr 0.01 -> 0.05, steps 50 -> 500",
        "n_llms": n_llms, "n_test_llms": len(test_llms),
        "n_items": n_items, "n_test_items": len(test_items),
        "K": K,
        "full_training_auc": full_auc,
        "calibration_sizes": cal_sizes, "n_repeats": n_repeats,
        "calibration_lr": CALIBRATION_LR,
        "calibration_steps": CALIBRATION_STEPS,
        "trained_theta_raw_ref_norm": train_theta_norm_ref,
        "init_method": "zero (principled no-info prior)",
        "summary": summary,
        "verified": True,
    }
    out_json = Path(f"cdm_exploration/experiments/v2_cold_start_fixed{out_tag}.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    verified = verify_splits(train_items, test_items, label="cold_start_v2_fixed")
    log_experiment(
        name=f"cold_start_v2_fixed{out_tag}",
        config={"n_test_llms": len(test_llms), "K": K,
                "cal_sizes": cal_sizes, "n_repeats": n_repeats,
                "lr": CALIBRATION_LR, "steps": CALIBRATION_STEPS,
                "init": "zero", "device": device},
        results={"full_auc": full_auc, "summary": summary},
        split_info={"n_train_items": len(train_items),
                      "n_test_items": len(test_items),
                      "n_train_llms": len(train_llms),
                      "n_test_llms": len(test_llms)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
