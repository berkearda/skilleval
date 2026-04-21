"""Aggregate IrtNet head-to-head sweep results into a single summary JSON.

Reads all `v2_irtnet_d{d}_s{seed}.json` files written by
`tools/run_irtnet_headtohead.py` and produces:

  * `cdm_exploration/experiments/v2_irtnet_headtohead_summary.json` with
    per-(d_model) mean and std over seeds for AUC, Acc@k, and best-val-acc.
  * A Table-1-ready row printed to stdout.

Run after the SLURM array sweep completes:

    python tools/aggregate_irtnet_results.py

The script aborts if any seed for a given d_model is missing, so that
stale summaries are never written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev


D_MODEL_GRID = [232, 512, 1024]
SEED_GRID = [42, 43, 44]
METRICS = (
    "test_auc",
    "test_acc",
    "routing_acc@1",
    "routing_acc@3",
    "routing_acc@5",
    "routing_acc@10",
    "best_val_acc",
)


def load_run(exp_dir: Path, d_model: int, seed: int) -> dict:
    path = exp_dir / f"v2_irtnet_d{d_model}_s{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run: {path}")
    with open(path) as f:
        return json.load(f)


def aggregate() -> dict:
    exp_dir = Path("cdm_exploration/experiments")
    summary: dict[str, dict] = {}
    checksums = set()
    for d in D_MODEL_GRID:
        per_seed = [load_run(exp_dir, d, s) for s in SEED_GRID]
        checksums.update(run["data_checksum"] for run in per_seed)
        stats = {}
        for m in METRICS:
            vals = [run[m] for run in per_seed]
            stats[m] = {
                "mean": float(mean(vals)),
                "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
        summary[f"d_model={d}"] = {
            "d_model": d,
            "seeds": SEED_GRID,
            "metrics": stats,
            "best_epoch_per_seed": [run["best_epoch"] for run in per_seed],
        }

    if len(checksums) > 1:
        raise RuntimeError(
            f"Data checksum mismatch across runs: {checksums}. All runs must "
            f"have used the same response matrix — aborting.",
        )

    summary["_meta"] = {
        "data_checksum": next(iter(checksums)) if checksums else None,
        "n_runs": len(D_MODEL_GRID) * len(SEED_GRID),
        "d_model_grid": D_MODEL_GRID,
        "seed_grid": SEED_GRID,
    }
    return summary


def print_table(summary: dict) -> None:
    print()
    print("=" * 84)
    print(
        f"{'d_model':>8}  {'AUC (mean±std)':>18}  {'Acc@1 (mean±std)':>20}  {'Acc@10 (mean±std)':>20}",
    )
    print("-" * 84)
    for d in D_MODEL_GRID:
        s = summary[f"d_model={d}"]["metrics"]
        print(
            f"{d:>8}  "
            f"{s['test_auc']['mean']:.4f} ± {s['test_auc']['std']:.4f}   "
            f"{s['routing_acc@1']['mean']:.4f} ± {s['routing_acc@1']['std']:.4f}    "
            f"{s['routing_acc@10']['mean']:.4f} ± {s['routing_acc@10']['std']:.4f}",
        )
    print("=" * 84)


def main() -> int:
    try:
        summary = aggregate()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "Run the full sweep first: sbatch tools/run_irtnet_headtohead.sbatch",
            file=sys.stderr,
        )
        return 1

    out_path = Path("cdm_exploration/experiments/v2_irtnet_headtohead_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {out_path}")
    print_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
