"""Per-method sub-audits: per-bench AUC, prediction correlation, picks distribution.

Runs after the main pareto comparison to deepen the audit:
  S1. Per-benchmark AUC + Acc@1 for each method (F8 aggregation-hides-heterogeneity).
  S2. Pairwise prediction correlation matrix (informs ensemble potential —
      if methods are highly correlated, ensembling won't help much).
  S3. Distribution of routing picks at fixed tau values per method (extends H3
      to "what cost band is each method routing to").
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
    from tools.run_pareto_multibaseline import (
        cdmeval_predictions, embedllm_predictions, irtnet_predictions,
        knn_predictions,
    )

    print("=== diag_pareto_subaudits ===\n", flush=True)

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

    print("[infer] generating predictions for all methods ...", flush=True)
    preds = {
        "cdmeval": cdmeval_predictions(test_idx, n_llms, n_skills, emb, q_matrix),
        "irtnet_d100": irtnet_predictions(100, 42, test_idx, n_llms, n_items, emb, "cpu"),
        "irtnet_d232": irtnet_predictions(232, 42, test_idx, n_llms, n_items, emb, "cpu"),
        "embedllm_d232": embedllm_predictions(232, 42, test_idx, n_llms, n_items, emb, "cpu"),
        "knn_k10": knn_predictions(emb, R, train_idx, test_idx, K=10),
    }
    print("  done\n", flush=True)

    audit = {}

    # ── S1: Per-bench AUC + Acc@1 ──
    print("=== S1: Per-bench AUC + argmax Acc@1 ===", flush=True)
    for name, p in preds.items():
        per_bench = {}
        per_bench_auc = {}
        for b in BENCHMARKS:
            mask = test_bench == b
            if mask.sum() == 0:
                continue
            preds_b = p[:, mask]
            R_b = R_test[:, mask]
            try:
                auc = float(roc_auc_score(R_b.flatten(), preds_b.flatten()))
            except ValueError:
                auc = float("nan")
            top1 = np.argmax(preds_b, axis=0)
            acc1 = float(R_b[top1, np.arange(mask.sum())].mean())
            per_bench[b] = acc1
            per_bench_auc[b] = auc
        # Overall
        try:
            auc_all = float(roc_auc_score(R_test.flatten(), p.flatten()))
        except ValueError:
            auc_all = float("nan")
        top1_all = np.argmax(p, axis=0)
        acc1_all = float(R_test[top1_all, np.arange(n_test)].mean())
        audit.setdefault(name, {})["per_bench_auc"] = per_bench_auc
        audit[name]["per_bench_acc1"] = per_bench
        audit[name]["overall_auc"] = auc_all
        audit[name]["overall_acc1"] = acc1_all

    print(f"  {'method':>15}  " + "  ".join(f"{b:>7}" for b in BENCHMARKS) +
            "  overall", flush=True)
    print(f"  {'(AUC)':>15}", flush=True)
    for name, r in audit.items():
        print(f"  {name:>15}  " +
                "  ".join(f"{r['per_bench_auc'][b]:>7.4f}" for b in BENCHMARKS) +
                f"  {r['overall_auc']:>7.4f}", flush=True)
    print(f"\n  {'method':>15}  " + "  ".join(f"{b:>7}" for b in BENCHMARKS) +
            "  overall", flush=True)
    print(f"  {'(Acc@1)':>15}", flush=True)
    for name, r in audit.items():
        print(f"  {name:>15}  " +
                "  ".join(f"{r['per_bench_acc1'][b]:>7.4f}" for b in BENCHMARKS) +
                f"  {r['overall_acc1']:>7.4f}", flush=True)

    # ── S2: Pairwise prediction correlation ──
    print("\n=== S2: Pairwise Pearson correlation of test predictions ===",
            flush=True)
    methods = list(preds.keys())
    corr = np.zeros((len(methods), len(methods)))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            corr[i, j] = float(np.corrcoef(preds[m1].flatten(),
                                              preds[m2].flatten())[0, 1])
    print(f"  {'':>15}  " + "  ".join(f"{m:>10}" for m in methods), flush=True)
    for i, m in enumerate(methods):
        print(f"  {m:>15}  " + "  ".join(f"{corr[i, j]:>10.4f}"
                                              for j in range(len(methods))),
                flush=True)
    audit["pairwise_pred_correlation"] = {
        m1: {m2: float(corr[i, j]) for j, m2 in enumerate(methods)}
        for i, m1 in enumerate(methods)
    }

    # ── S3: Pick cost distribution per tau ──
    print("\n=== S3: Pick cost distribution per tau (cost percentiles of "
            "routed LLMs) ===", flush=True)
    # IrtNet predictions need sigmoid
    preds_for_routing = {}
    for name, p in preds.items():
        if "irtnet" in name:
            preds_for_routing[name] = 1.0 / (1.0 + np.exp(-p))
        else:
            preds_for_routing[name] = p

    for tau in [0.3, 0.5, 0.7, 0.9]:
        print(f"\n  tau={tau}:", flush=True)
        print(f"    {'method':>15}  {'p10_cost':>10}  {'p50_cost':>10}  "
                f"{'p90_cost':>10}  {'unique':>7}  {'fallback%':>10}", flush=True)
        for name, p in preds_for_routing.items():
            picks = []
            n_fallback = 0
            for j in range(n_test):
                above = np.where(p[:, j] > tau)[0]
                if len(above) > 0:
                    picks.append(int(above[np.argmin(flops[above])]))
                else:
                    picks.append(int(np.argmax(p[:, j])))
                    n_fallback += 1
            picks = np.array(picks)
            pick_costs = flops[picks]
            cps = np.percentile(pick_costs, [10, 50, 90])
            print(f"    {name:>15}  {cps[0]:>10.0f}  {cps[1]:>10.0f}  "
                    f"{cps[2]:>10.0f}  {len(np.unique(picks)):>7}  "
                    f"{n_fallback / n_test * 100:>9.1f}%", flush=True)
            audit.setdefault(name, {}).setdefault("pick_cost_per_tau", {})[
                f"tau_{tau}"] = {
                "p10_cost_gflops": float(cps[0]),
                "p50_cost_gflops": float(cps[1]),
                "p90_cost_gflops": float(cps[2]),
                "n_unique_picks": int(len(np.unique(picks))),
                "fallback_fraction": n_fallback / n_test,
            }

    out_path = EXP / "v2_pareto_subaudits.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2, default=float)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
