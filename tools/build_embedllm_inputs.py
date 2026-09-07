"""Convert CDMEval response matrix + SBERT embeddings into EmbedLLM input format.

EmbedLLM's ``algorithm/mf.py`` expects:
  - ``train.csv``  with columns: model_id, prompt_id, label
  - ``test.csv``   with columns: model_id, prompt_id, label, model_name
  - ``question_embeddings.pth``  torch tensor [num_prompts, 768]

This script writes those three files from
  - ``response_matrix_v2_full.npy``        (3811, 9523) binary
  - ``item_text_embeddings_v2_full.npz``   (9523, 768)
  - ``response_matrix_v2_full_llms.json``  list of 3811 model names

Split: canonical 80/20 item-wise via train_test_split(random_state=42),
identical to every other v2 baseline.

Pre-flight asserts (B2 / failure-mode guards):
  * embeddings shape matches matrix items dim
  * after split, every LLM has >= 1 train and >= 1 test response (the
    EmbedLLM ``CustomDataset`` re-rank at mf.py:55-58 is identity only
    if unique-id sets match between splits)

Usage:
    python tools/build_embedllm_inputs.py [--smoke]

Smoke mode subsamples to 100 LLMs and 200 items so a full mf.py training
run is feasible in <2 min on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
OUT_DIR = REPO / "cdm_exploration" / "data" / "embedllm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Subsample to 100 LLMs × 200 items for B1 smoke test.")
    ap.add_argument("--calibrated", action="store_true",
                    help="Build the 60/20/20 calibrated split: train.csv from "
                          "train_inner_idx (60%), test.csv from val_idx (20%, "
                          "used for calibration). Canonical 20% test items "
                          "evaluated separately at inference time.")
    ap.add_argument("--out_subdir", type=str, default=None,
                    help="Subdirectory under embedllm/ (default: 'smoke' for "
                          "smoke mode, 'calibrated' for calibrated mode, '' "
                          "otherwise).")
    args = ap.parse_args()

    out_subdir = args.out_subdir
    if out_subdir is None:
        if args.smoke:
            out_subdir = "smoke"
        elif args.calibrated:
            out_subdir = "calibrated"
        else:
            out_subdir = ""
    out = OUT_DIR / out_subdir if out_subdir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== build_embedllm_inputs (smoke={args.smoke}) ===", flush=True)
    print(f"  out: {out}", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(DATA / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)

    n_llms, n_items = R.shape
    assert emb.shape == (n_items, 768), \
        f"embedding shape {emb.shape} != ({n_items}, 768)"
    assert len(llm_names) == n_llms

    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    train_idx.sort()
    test_idx.sort()

    if args.calibrated:
        # Use canonical 60/20/20 calibration split (single source of truth).
        # F14 bug 2026-05-01: pre-sorting train_idx before train_test_split
        # gave a 76%-mismatched val from CDMEval's val. Now centralised.
        import sys
        sys.path.insert(0, str(REPO))
        from cdmeval.utils.calibration_split import (
            make_canonical_calibrated_split,
        )
        train_inner_idx, val_idx, _ = make_canonical_calibrated_split(n_items)
        # In the calibrated setup, "train.csv" = train_inner triplets, and
        # "test.csv" = val triplets (so the EmbedLLM training-time eval
        # measures val performance, used downstream for calibration). The
        # canonical test items are NOT in the CSVs; inference happens later.
        train_idx = train_inner_idx
        test_idx = val_idx
        print(f"  CALIBRATED split: train_inner={len(train_inner_idx)}, "
                f"val={len(val_idx)} (used as 'test' for EmbedLLM training)",
                flush=True)
        print(f"  val[:5] (canonical): {val_idx[:5].tolist()}", flush=True)

    if args.smoke:
        # Sample 100 LLMs randomly with seed for repeatability + 200 items
        rng = np.random.default_rng(42)
        llm_sub = rng.choice(n_llms, 100, replace=False)
        llm_sub.sort()
        # Sample 160 train items + 40 test items so split sizes are sensible
        train_idx = rng.choice(train_idx, 160, replace=False)
        train_idx.sort()
        test_idx = rng.choice(test_idx, 40, replace=False)
        test_idx.sort()
        R = R[llm_sub][:, np.concatenate([train_idx, test_idx])]
        emb_sub_idx = np.concatenate([train_idx, test_idx])
        emb_sub_idx_sorted = np.sort(emb_sub_idx)
        # Re-index items 0..199 in the sub-matrix
        old_to_new = {int(o): int(n) for n, o in enumerate(emb_sub_idx_sorted)}
        train_idx = np.array([old_to_new[int(i)] for i in train_idx])
        test_idx = np.array([old_to_new[int(i)] for i in test_idx])
        emb = emb[emb_sub_idx_sorted]
        llm_names = [llm_names[i] for i in llm_sub]
        n_llms, n_items = R.shape
        print(f"  smoke shape: R={R.shape}  emb={emb.shape}", flush=True)

    print(f"  n_llms={n_llms}  n_items={n_items}", flush=True)
    print(f"  train items: {len(train_idx)}, test items: {len(test_idx)}",
            flush=True)

    print("\n[1/3] writing question_embeddings.pth ...", flush=True)
    emb_t = torch.tensor(emb, dtype=torch.float32)
    assert emb_t.shape == (n_items, 768)
    torch.save(emb_t, out / "question_embeddings.pth")
    print(f"  shape: {tuple(emb_t.shape)}  bytes: {emb_t.numel() * 4 / 2**20:.1f} MiB",
            flush=True)

    print("\n[2/3] building train.csv ...", flush=True)
    # Long-format: one row per (LLM, train item)
    R_train = R[:, train_idx]  # (n_llms, n_train)
    train_rows = pd.DataFrame({
        "model_id": np.repeat(np.arange(n_llms), len(train_idx)),
        "prompt_id": np.tile(train_idx, n_llms),
        "label": R_train.flatten().astype(int),
    })
    train_path = out / "train.csv"
    train_rows.to_csv(train_path, index=False)
    print(f"  rows: {len(train_rows):,}", flush=True)
    print(f"  size: {train_path.stat().st_size / 2**20:.1f} MiB", flush=True)

    print("\n[3/3] building test.csv ...", flush=True)
    R_test = R[:, test_idx]
    test_rows = pd.DataFrame({
        "model_id": np.repeat(np.arange(n_llms), len(test_idx)),
        "prompt_id": np.tile(test_idx, n_llms),
        "label": R_test.flatten().astype(int),
        "model_name": np.repeat(np.array(llm_names), len(test_idx)),
    })
    test_path = out / "test.csv"
    test_rows.to_csv(test_path, index=False)
    print(f"  rows: {len(test_rows):,}", flush=True)
    print(f"  size: {test_path.stat().st_size / 2**20:.1f} MiB", flush=True)

    # ── Pre-flight asserts (B2 / F10/F14 guards) ──
    print("\n[asserts] checking input contracts ...", flush=True)
    train_uids = sorted(train_rows["model_id"].unique().tolist())
    test_uids = sorted(test_rows["model_id"].unique().tolist())
    expected_uids = list(range(n_llms))
    assert train_uids == expected_uids, \
        f"train.csv missing model_ids: {set(expected_uids) - set(train_uids)}"
    assert test_uids == expected_uids, \
        f"test.csv missing model_ids: {set(expected_uids) - set(test_uids)}"
    assert set(train_rows["label"].unique()) <= {0, 1}
    assert set(test_rows["label"].unique()) <= {0, 1}
    assert int(train_rows["prompt_id"].max()) < n_items
    assert int(test_rows["prompt_id"].max()) < n_items
    # CustomDataset re-rank (mf.py:55-58) is identity iff unique sets match
    # AND are dense 0..n-1 — both confirmed above.
    print("  train/test cover all model_ids 0..n-1", flush=True)
    print("  labels are binary, prompt_ids in range", flush=True)

    summary = {
        "smoke": args.smoke,
        "n_llms": n_llms,
        "n_items": n_items,
        "n_train_items": int(len(train_idx)),
        "n_test_items": int(len(test_idx)),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "embedding_shape": list(emb_t.shape),
    }
    with open(out / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {out}/build_summary.json", flush=True)
    print("  ok.", flush=True)


if __name__ == "__main__":
    main()
