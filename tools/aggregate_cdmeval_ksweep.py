"""Aggregate CDMEval K-sweep multi-seed results into a single Table-1-ready
summary. Pairs the 3-seed CDMEval rows with the 3-seed IrtNet d-sweep rows
so the asymmetric-ablation gap (T-021) is closed.

Sources read:
  - cdm_exploration/experiments/v2_K{K}_results.json      (legacy seed-42)
  - cdm_exploration/experiments/v2_K{K}_s{seed}_results.json
  - cdm_exploration/experiments/v2_multi_seed.json        (K=100, 5 seeds)
  - cdm_exploration/experiments/v2_irtnet_headtohead_summary.json (IrtNet)

Writes:
  - cdm_exploration/experiments/v2_cdmeval_ksweep_summary.json

Run:
    python3 tools/aggregate_cdmeval_ksweep.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration" / "experiments"
K_VALUES = [50, 100, 150, 200, 300, 400, 500]
EXTRA_SEEDS = (43, 44)


def safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_per_k(K: int) -> dict:
    """Return {seed: results_dict} for a given K, scanning legacy + multi-seed
    files."""
    runs: dict[int, dict] = {}

    legacy = safe_load(EXP / f"v2_K{K}_results.json")
    if legacy is not None:
        seed = legacy.get("seed", 42)
        runs[int(seed)] = legacy

    for seed in EXTRA_SEEDS:
        d = safe_load(EXP / f"v2_K{K}_s{seed}_results.json")
        if d is not None:
            runs[seed] = d

    if K == 100:
        ms = safe_load(EXP / "v2_multi_seed.json")
        if ms is not None and "per_seed" in ms:
            for entry in ms["per_seed"]:
                seed = int(entry.get("seed", -1))
                if seed >= 0 and seed not in runs:
                    runs[seed] = entry

    return runs


def stats_for_metric(runs: dict[int, dict], metric: str) -> dict | None:
    candidate_keys = (
        metric,
        f"acc@{metric}" if metric in {"1", "3", "5", "10"} else None,
        metric.replace("acc@", "routing_acc@"),
    )
    values = []
    for seed in sorted(runs.keys()):
        r = runs[seed]
        for k in candidate_keys:
            if k is None:
                continue
            if k in r:
                values.append(float(r[k]))
                break
    if not values:
        return None
    return {
        "n": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "values": values,
    }


def aggregate() -> dict:
    summary = {"K_values": {}, "_meta": {"k_grid": K_VALUES}}
    for K in K_VALUES:
        runs = collect_per_k(K)
        if not runs:
            continue
        summary["K_values"][K] = {
            "K": K,
            "n_seeds": len(runs),
            "seeds": sorted(runs.keys()),
            "test_auc": stats_for_metric(runs, "test_auc"),
            "acc@1": stats_for_metric(runs, "acc@1"),
            "acc@3": stats_for_metric(runs, "acc@3"),
            "acc@5": stats_for_metric(runs, "acc@5"),
            "acc@10": stats_for_metric(runs, "acc@10"),
        }
    irtnet_path = EXP / "v2_irtnet_headtohead_summary.json"
    if irtnet_path.exists():
        with open(irtnet_path) as f:
            summary["_irtnet_reference"] = json.load(f)
    return summary


def fmt_pm(stats: dict | None, digits: int = 4) -> str:
    if stats is None:
        return "  --   "
    if stats["n"] == 1:
        return f"{stats['mean']:.{digits}f}      (1 seed)"
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def print_cdmeval_table(summary: dict) -> None:
    print()
    print("=" * 96)
    print(f"{'K':>5}  {'n_seeds':>8}  {'Acc@1 (mean ± std)':>22}  "
          f"{'Acc@10 (mean ± std)':>22}  {'Test AUC (mean)':>16}")
    print("-" * 96)
    for K in K_VALUES:
        if K not in summary["K_values"]:
            continue
        row = summary["K_values"][K]
        acc1 = fmt_pm(row["acc@1"])
        acc10 = fmt_pm(row["acc@10"])
        auc = fmt_pm(row["test_auc"])
        print(f"{K:>5}  {row['n_seeds']:>8}  {acc1:>22}  {acc10:>22}  {auc:>16}")
    print("=" * 96)


def print_headtohead(summary: dict) -> None:
    if "_irtnet_reference" not in summary:
        print("\n(no IrtNet summary found — head-to-head not printed)")
        return
    irt = summary["_irtnet_reference"]
    print()
    print("Head-to-head against IrtNet (3-seed mean ± std)")
    print("-" * 96)
    for d_label in sorted(k for k in irt if k.startswith("d_model=")):
        d_block = irt[d_label]
        d_int = d_block["d_model"]
        m = d_block["metrics"]["routing_acc@1"]
        auc = d_block["metrics"]["test_auc"]
        print(f"  IrtNet d={d_int:>4}        Acc@1 = {m['mean']:.4f} ± {m['std']:.4f}    "
              f"AUC = {auc['mean']:.4f}")
    print("-" * 96)
    for K in K_VALUES:
        if K not in summary["K_values"]:
            continue
        row = summary["K_values"][K]
        if row["acc@1"] is None:
            continue
        acc1 = row["acc@1"]
        auc = row["test_auc"]
        marker = "  CDMEval"
        print(f"{marker} K={K:>4}     Acc@1 = {acc1['mean']:.4f}"
              + (f" ± {acc1['std']:.4f}" if acc1['n'] > 1 else f"  (1 seed) ")
              + f"   AUC = {auc['mean']:.4f}"
              + (f" ± {auc['std']:.4f}" if auc['n'] > 1 else ""))
    print("=" * 96)


def main() -> int:
    summary = aggregate()
    out = EXP / "v2_cdmeval_ksweep_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {out}")
    print_cdmeval_table(summary)
    print_headtohead(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
