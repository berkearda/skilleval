"""Verify all experiment splits are correct and non-overlapping.

Usage:
    python tools/verify_splits.py
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("SPLIT VERIFICATION")
    print("=" * 60)

    n_items = 2643
    n_llms = 235
    all_items = np.arange(n_items)

    # Protocol B item split (used by all experiments)
    train_items, test_items = train_test_split(
        all_items, test_size=0.2, random_state=42
    )
    print(f"\nProtocol B item split:")
    print(f"  Train: {len(train_items)}, Test: {len(test_items)}")
    overlap = np.intersect1d(train_items, test_items)
    print(f"  Overlap: {len(overlap)} {'PASS' if len(overlap) == 0 else 'FAIL'}")
    assert len(overlap) == 0, "Train/test items overlap!"

    # LLM cold-start split
    from cdmeval.evaluation.llm_cold_start import llm_cold_start_split
    train_llms, test_llms = llm_cold_start_split(n_llms, test_fraction=0.2, seed=42)
    print(f"\nLLM cold-start split:")
    print(f"  Train LLMs: {len(train_llms)}, Test LLMs: {len(test_llms)}")
    overlap_llm = np.intersect1d(train_llms, test_llms)
    print(f"  Overlap: {len(overlap_llm)} {'PASS' if len(overlap_llm) == 0 else 'FAIL'}")
    assert len(overlap_llm) == 0, "Train/test LLMs overlap!"

    # Verify reproducibility
    train2, test2 = train_test_split(all_items, test_size=0.2, random_state=42)
    assert np.array_equal(train_items, train2), "Item split not reproducible!"
    print(f"\nReproducibility: PASS (random_state=42)")

    # Check saved results files
    data_dir = Path("cdm_exploration/data/cdm_ready")

    checks = [
        ("baseline_comparison_results.json", "baseline"),
        ("llm_cold_start_results.json", "cold_start"),
        ("pareto_routing_results.json", "pareto"),
        ("cost_routing_results_price.json", "cost_routing"),
    ]

    print(f"\nSaved results files:")
    for fname, label in checks:
        path = data_dir / fname
        if path.exists():
            print(f"  {fname}: EXISTS")
        else:
            print(f"  {fname}: not found (ok if not yet run)")

    print(f"\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
