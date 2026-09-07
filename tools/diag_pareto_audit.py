"""Diagnostic audit of fig:pareto v3 multibaseline predictions.

Re-runs inference for CDMEval / IrtNet d=100/232 / EmbedLLM d=232 / KNN K=10
and answers H2-H6 from the audit:

  H2  scale of P(correct) per method (histogram + percentiles)
  H3  unique LLMs picked per tau (F2 sniff)
  H4  cost distribution of picks per tau (cheapest-LLM trick check)
  H5  per-bench Acc@1 at tau=0.5 cheapest-above-threshold
  H6  calibration: bin pred -> mean actual acc per bin

H1 (F5 leakage) and H7 (genuine better per-cell) are answered by the
existing input-contracts and AUC numbers respectively; not re-run here.

Usage:
    python tools/diag_pareto_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"

BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


def main() -> None:
    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.utils.device import seed_everything
    from tools.run_pareto_multibaseline import (
        cdmeval_predictions,
        embedllm_predictions,
        irtnet_predictions,
        knn_predictions,
        threshold_sweep,
    )

    seed_everything(42)
    print("=== diag_pareto_audit ===\n", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    benches = np.array([it.get("benchmark") for it in items])
    with open(DATA / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    n_llms, n_items = R.shape
    n_skills = q_matrix.shape[1]

    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    R_test = R[:, test_idx]
    test_bench = benches[test_idx]
    n_test = len(test_idx)

    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])

    print(f"  test items: {n_test}, LLMs: {n_llms}", flush=True)
    print(f"  flops range: {flops.min():.0f} – {flops.max():.0f} GFLOPs",
            flush=True)
    print(f"  cheapest LLM idx: {int(np.argmin(flops))} "
            f"({llm_names[int(np.argmin(flops))]})", flush=True)

    print("\n[infer] regenerating predictions ...", flush=True)
    preds = {
        "cdmeval": cdmeval_predictions(test_idx, n_llms, n_skills, emb, q_matrix),
        "irtnet_d100": irtnet_predictions(100, 42, test_idx, n_llms, n_items, emb, "cpu"),
        "irtnet_d232": irtnet_predictions(232, 42, test_idx, n_llms, n_items, emb, "cpu"),
        "embedllm_d232": embedllm_predictions(232, 42, test_idx, n_llms, n_items, emb, "cpu"),
        "knn_k10": knn_predictions(emb, R, train_idx, test_idx, K=10),
    }
    print("  done", flush=True)

    audit = {}

    # ── H2: scale of P(correct) ──
    print("\n=== H2: prediction scale per method ===", flush=True)
    print(f"  {'method':>15}  {'min':>6} {'p10':>6} {'p50':>6} {'p90':>6} {'p99':>6} {'max':>6}",
            flush=True)
    for name, p in preds.items():
        pcts = np.percentile(p.flatten(), [10, 50, 90, 99])
        print(f"  {name:>15}  {p.min():>6.3f} {pcts[0]:>6.3f} {pcts[1]:>6.3f} "
                f"{pcts[2]:>6.3f} {pcts[3]:>6.3f} {p.max():>6.3f}", flush=True)
        audit.setdefault(name, {})["pred_percentiles"] = {
            "p10": float(pcts[0]), "p50": float(pcts[1]),
            "p90": float(pcts[2]), "p99": float(pcts[3]),
        }
        audit[name]["pred_range"] = [float(p.min()), float(p.max())]
        audit[name]["pred_mean"] = float(p.mean())
        audit[name]["pred_std"] = float(p.std())

    # ── H3 + H4: unique picks + cost of picks at fixed taus ──
    print("\n=== H3 + H4: unique routing picks + cost of picks per tau ===",
            flush=True)
    print(f"  {'method':>15}  {'tau':>5}  {'n_unique':>8}  {'cheapest_pick%':>14}  "
            f"{'mean_cost_GFL':>13}  {'fallback%':>10}", flush=True)
    cheapest_idx = int(np.argmin(flops))
    for name, p in preds.items():
        for tau in [0.3, 0.5, 0.7, 0.9]:
            picks = []
            n_fallback = 0
            for j in range(n_test):
                above = np.where(p[:, j] > tau)[0]
                if len(above) > 0:
                    chosen = above[np.argmin(flops[above])]
                else:
                    chosen = int(np.argmax(p[:, j]))
                    n_fallback += 1
                picks.append(int(chosen))
            picks = np.array(picks)
            n_unique = int(len(np.unique(picks)))
            cheapest_frac = float((picks == cheapest_idx).mean())
            mean_cost = float(flops[picks].mean())
            print(f"  {name:>15}  {tau:>5.2f}  {n_unique:>8}  "
                    f"{cheapest_frac * 100:>13.1f}%  {mean_cost:>13.0f}  "
                    f"{n_fallback / n_test * 100:>9.1f}%", flush=True)
            audit[name].setdefault("picks_per_tau", {})[f"tau_{tau}"] = {
                "n_unique": n_unique,
                "cheapest_LLM_pick_fraction": cheapest_frac,
                "mean_pick_cost_gflops": mean_cost,
                "fallback_fraction": n_fallback / n_test,
            }

    # ── H5: per-benchmark Acc@1 at tau=0.5 ──
    print("\n=== H5: per-bench Acc@1 at tau=0.5 (cheapest-above-threshold) ===",
            flush=True)
    tau = 0.5
    print(f"  {'method':>15}  {'MATH':>7}  {'BBH':>7}  {'GPQA':>7}  {'MuSR':>7}  {'IFEval':>7}  {'overall':>8}",
            flush=True)
    for name, p in preds.items():
        per_bench = {}
        # Compute picks at tau=0.5
        picks = np.zeros(n_test, dtype=int)
        for j in range(n_test):
            above = np.where(p[:, j] > tau)[0]
            if len(above) > 0:
                picks[j] = above[np.argmin(flops[above])]
            else:
                picks[j] = int(np.argmax(p[:, j]))
        correct = R_test[picks, np.arange(n_test)]
        for b in BENCHMARKS:
            mask = test_bench == b
            per_bench[b] = float(correct[mask].mean()) if mask.any() else float("nan")
        per_bench["overall"] = float(correct.mean())
        print(f"  {name:>15}  "
                f"{per_bench['MATH']:>7.4f}  "
                f"{per_bench['BBH']:>7.4f}  "
                f"{per_bench['GPQA']:>7.4f}  "
                f"{per_bench['MuSR']:>7.4f}  "
                f"{per_bench['IFEval']:>7.4f}  "
                f"{per_bench['overall']:>8.4f}", flush=True)
        audit[name]["per_bench_at_tau_0.5"] = per_bench

    # ── H6: calibration ──
    print("\n=== H6: calibration (binned pred -> mean actual acc) ===",
            flush=True)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    print(f"  {'method':>15}  " + "  ".join(f"[{bins[i]:.1f}-{bins[i+1]:.1f}]"
                                                for i in range(len(bins) - 1)),
            flush=True)
    for name, p in preds.items():
        pred_flat = p.flatten()
        true_flat = R_test.flatten()
        cal = []
        n_per_bin = []
        for i in range(len(bins) - 1):
            mask = (pred_flat >= bins[i]) & (pred_flat < bins[i + 1])
            if i == len(bins) - 2:
                mask = (pred_flat >= bins[i]) & (pred_flat <= bins[i + 1])
            if mask.sum() == 0:
                cal.append(float("nan"))
                n_per_bin.append(0)
            else:
                cal.append(float(true_flat[mask].mean()))
                n_per_bin.append(int(mask.sum()))
        print(f"  {name:>15}  " + "  ".join(
            f"{c:>9.3f}" if not np.isnan(c) else f"{'   nan':>9}"
            for c in cal), flush=True)
        audit[name]["calibration"] = {
            "bin_edges": bins,
            "actual_acc_per_bin": cal,
            "n_per_bin": n_per_bin,
        }
    # Also report a single calibration metric: ECE (weighted gap between bin
    # mean pred and actual acc, weighted by bin count)
    print(f"\n  {'method':>15}  {'ECE':>6}  {'AUC':>6}", flush=True)
    for name, p in preds.items():
        pred_flat = p.flatten()
        true_flat = R_test.flatten()
        ece = 0.0
        n_total = len(pred_flat)
        for i in range(len(bins) - 1):
            mask = (pred_flat >= bins[i]) & (pred_flat < bins[i + 1])
            if i == len(bins) - 2:
                mask = (pred_flat >= bins[i]) & (pred_flat <= bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_mean_pred = float(pred_flat[mask].mean())
            bin_mean_true = float(true_flat[mask].mean())
            ece += (mask.sum() / n_total) * abs(bin_mean_pred - bin_mean_true)
        auc = float(roc_auc_score(true_flat, pred_flat))
        audit[name]["ECE"] = float(ece)
        audit[name]["AUC"] = auc
        print(f"  {name:>15}  {ece:>6.3f}  {auc:>6.3f}", flush=True)

    # Save
    out_path = EXP / "v2_pareto_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
