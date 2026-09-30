"""Canonical 60/20/20 item split for the calibration experiments (T-033, T-034).

Single source of truth for the train_inner / val / test split used across
CDMEval and EmbedLLM calibrated retraining. Without this central definition,
each script's call to ``train_test_split`` could produce a different val set
(F14 garbage-out caught 2026-05-01: pre-sorting train_idx before the second
split changes the random shuffle and yields a 76% mismatch with the unsorted
version).

Logic:
  1. Canonical 80/20 item split: ``train_test_split(np.arange(n_items),
     test_size=0.2, random_state=42)`` -> train_idx (7618), test_idx (1905).
     Test set is FROZEN: every v2 baseline uses these 1905 items.
  2. Sub-split train_idx (UNSORTED, in the order returned by the first
     train_test_split): ``train_test_split(train_idx, test_size=0.25,
     random_state=42)`` -> train_inner (5713), val (1905). DO NOT SORT
     train_idx before this call -- it changes the random shuffle.
  3. Sort each of train_inner_idx / val_idx / test_idx for downstream use.

Verification: any script that uses these splits should call ``verify_split()``
to confirm the deterministic match.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

CANONICAL_N_ITEMS = 9523
CANONICAL_SEED = 42
SPLIT_JSON_PATH = Path(__file__).resolve().parent.parent.parent / \
    "cdm_exploration" / "data" / "cdm_ready" / "calibration_split.json"


def make_canonical_calibrated_split(n_items: int = CANONICAL_N_ITEMS,
                                       seed: int = CANONICAL_SEED):
    """Return (train_inner_idx, val_idx, test_idx) all sorted ascending.

    Sizes for n_items=9523, seed=42: (5713, 1905, 1905).
    Test set matches the canonical 80/20 split used throughout v2 baselines.
    """
    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=seed)
    # IMPORTANT: do NOT sort train_idx before the next split.
    # The random_state of the inner train_test_split shuffles based on input
    # order, so pre-sorting changes which items end up in val. CDMEval's
    # original calibrated training (T-033) used the unsorted version, and
    # we treat that as canonical for downstream consistency.
    train_inner_idx, val_idx = train_test_split(train_idx, test_size=0.25,
                                                 random_state=seed)
    train_inner_idx = np.sort(train_inner_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    # F5 sniff: no overlap
    assert len(set(train_inner_idx) & set(val_idx)) == 0
    assert len(set(train_inner_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    # Sanity: union covers all items
    assert (len(train_inner_idx) + len(val_idx) + len(test_idx)) == n_items

    return train_inner_idx, val_idx, test_idx


def write_split_json(out_path: Path = SPLIT_JSON_PATH) -> None:
    """Write the canonical split to disk for cross-script verification."""
    train_inner_idx, val_idx, test_idx = make_canonical_calibrated_split()
    with open(out_path, "w") as f:
        json.dump({
            "n_items": CANONICAL_N_ITEMS,
            "seed": CANONICAL_SEED,
            "train_inner_idx": train_inner_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "description": "Canonical 60/20/20 calibration split. Source of "
                            "truth for T-033 / T-034. Reproduce with "
                            "cdmeval.utils.calibration_split.make_canonical_calibrated_split().",
        }, f, indent=2)


def verify_split(train_inner_idx: np.ndarray, val_idx: np.ndarray,
                   test_idx: np.ndarray) -> None:
    """Assert that the supplied splits match the canonical definition."""
    expected_ti, expected_val, expected_test = make_canonical_calibrated_split()
    assert np.array_equal(np.sort(train_inner_idx), expected_ti), \
        "train_inner_idx mismatch with canonical"
    assert np.array_equal(np.sort(val_idx), expected_val), \
        "val_idx mismatch with canonical"
    assert np.array_equal(np.sort(test_idx), expected_test), \
        "test_idx mismatch with canonical"


if __name__ == "__main__":
    ti, val, test = make_canonical_calibrated_split()
    print(f"train_inner: {len(ti)}  val: {len(val)}  test: {len(test)}")
    print(f"val[:10]:  {val[:10].tolist()}")
    print(f"test[:10]: {test[:10].tolist()}")
    write_split_json()
    print(f"wrote {SPLIT_JSON_PATH}")
