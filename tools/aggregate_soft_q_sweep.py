"""Aggregate per-tau soft Q NCDM run JSONs into a single sweep summary.

Reads cdm_exploration/experiments/v2_soft_q_tau{tau}_s{seed}.json for each
configured tau (default 0.05, 0.1, 0.3, 1.0, 3.0) and writes
cdm_exploration/experiments/v2_soft_q_sweep_summary.json.

Also reports the binary-Q baseline numbers from the most recent
v2_multi_seed_42.json for an at-a-glance comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_TAUS = [0.05, 0.1, 0.3, 1.0, 3.0]
EXP_DIR = Path("cdm_exploration/experiments")


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"  WARN: could not parse {path}: {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--taus", type=float, nargs="*", default=DEFAULT_TAUS)
    ap.add_argument(
        "--baseline",
        type=str,
        default="v2_multi_seed.json",
        help=(
            "Filename for the binary-Q baseline summary in EXP_DIR. "
            "Defaults to v2_multi_seed.json (multi-seed aggregate); "
            "the seed=42 entry is read from per_seed['42'] inside it."
        ),
    )
    args = ap.parse_args()

    rows = []
    missing = []
    for tau in args.taus:
        path = EXP_DIR / f"v2_soft_q_tau{tau}_s{args.seed}.json"
        d = _load(path)
        if d is None:
            missing.append(str(path))
            continue
        rows.append(
            {
                "tau": tau,
                "path": str(path),
                "test_auc": d.get("test_auc"),
                "test_acc": d.get("test_acc"),
                "test_rmse": d.get("test_rmse"),
                "routing_acc1": d.get("routing_acc1"),
                "strongest_acc1": d.get("strongest_acc1"),
                "random_acc1": d.get("random_acc1"),
                "per_benchmark_acc1": d.get("per_benchmark_acc1", {}),
                "epochs": d.get("epochs"),
                "soft_q_file": d.get("soft_q_file"),
            }
        )

    baseline = _load(EXP_DIR / args.baseline)
    baseline_summary = None
    if baseline is not None:
        # If the file is the multi-seed aggregate, prefer the matching
        # per-seed entry; otherwise treat it as a single-run file.
        per_seed = baseline.get("per_seed")
        if isinstance(per_seed, dict) and str(args.seed) in per_seed:
            seed_entry = per_seed[str(args.seed)]
            baseline_summary = {
                "path": str(EXP_DIR / args.baseline),
                "source": f"per_seed['{args.seed}']",
                "test_auc": seed_entry.get("test_auc"),
                "routing_acc1": seed_entry.get("routing_acc1"),
                "strongest_acc1": seed_entry.get("strongest_acc1"),
                "random_acc1": seed_entry.get("random_acc1"),
                "aggregate_test_auc_mean": baseline.get("test_auc_mean"),
                "aggregate_routing_acc1_mean": baseline.get("routing_acc1_mean"),
            }
        else:
            baseline_summary = {
                "path": str(EXP_DIR / args.baseline),
                "source": "top_level",
                "test_auc": baseline.get("test_auc"),
                "routing_acc1": baseline.get("routing_acc1"),
                "strongest_acc1": baseline.get("strongest_acc1"),
                "random_acc1": baseline.get("random_acc1"),
            }

    summary = {
        "seed": args.seed,
        "taus_requested": args.taus,
        "n_results": len(rows),
        "results": rows,
        "missing": missing,
        "binary_q_baseline": baseline_summary,
    }

    if rows:
        # ASCII table for quick inspection
        print("\n  tau    epochs  test_auc  routing_acc1  strongest")
        print("  ------ ------  --------  ------------  ---------")
        for r in rows:
            print(
                f"  {r['tau']:>5}  {r['epochs']:>6}   "
                f"{r['test_auc']:.4f}     {r['routing_acc1']:.4f}        "
                f"{r['strongest_acc1']:.4f}"
            )
        if baseline_summary and baseline_summary["test_auc"] is not None:
            print(
                f"\n  binary baseline (seed=42): "
                f"AUC={baseline_summary['test_auc']:.4f}  "
                f"Acc@1={baseline_summary['routing_acc1']:.4f}"
            )

    if missing:
        print(f"\n  Missing per-tau files (sweep incomplete): {missing}")

    out = EXP_DIR / "v2_soft_q_sweep_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
