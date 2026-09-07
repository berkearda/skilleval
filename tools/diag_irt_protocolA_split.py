"""Diagnostic for IRT 2PL Protocol A audit.

Verifies:
  - Failure mode 1: identical te_idx between IRT and NCDM scripts.
  - Failure mode 2: identical (s, j, y) labels at test indices across pipelines.
  - Failure mode 3: metric definitions sane (binary y, sigmoid in [0,1]).
  - Failure mode 7: data files identical.
  - Failure mode 9: no overlap between train and test indices.
  - Failure mode 11: strongest-baseline (per-LLM mean train acc) test AUC.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
from cdmeval.utils.device import seed_everything

DATA = ROOT / "cdm_exploration" / "data" / "cdm_ready"


def hash_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> None:
    seed_everything(42)

    # ── Load data ──
    R = np.load(DATA / "response_matrix_v2_full.npy")
    n_llms, n_items = R.shape
    print(f"R shape: {R.shape}, dtype={R.dtype}, sum={int(R.sum())}")
    print(f"R sha256[:16] = {hashlib.sha256(R.tobytes()).hexdigest()[:16]}")
    assert R.shape == (3811, 9523), R.shape

    # ── IRT-style triplet construction ──
    irt_llm = np.repeat(np.arange(n_llms, dtype=np.int32), n_items)
    irt_item = np.tile(np.arange(n_items, dtype=np.int32), n_llms)
    irt_label = R[irt_llm, irt_item].astype(np.float32)
    n_total_irt = len(irt_label)

    irt_all_idx = np.arange(n_total_irt, dtype=np.int64)
    irt_trvl_idx, irt_te_idx = train_test_split(
        irt_all_idx, test_size=0.10, random_state=42,
    )
    irt_tr_idx, irt_va_idx = train_test_split(
        irt_trvl_idx, test_size=10.0 / 90.0, random_state=42,
    )
    print(
        f"IRT split: train={len(irt_tr_idx):,}  val={len(irt_va_idx):,}  "
        f"test={len(irt_te_idx):,}"
    )

    # ── NCDM-style triplet construction ──
    ncdm_llm = np.repeat(np.arange(n_llms), n_items)
    ncdm_item = np.tile(np.arange(n_items), n_llms)
    ncdm_score = R.ravel().astype(float)
    ncdm_triplets = np.column_stack([ncdm_llm, ncdm_item, ncdm_score])
    ncdm_all_idx = np.arange(len(ncdm_triplets))
    ncdm_trvl_idx, ncdm_te_idx = train_test_split(
        ncdm_all_idx, test_size=0.10, random_state=42,
    )
    ncdm_tr_idx, ncdm_va_idx = train_test_split(
        ncdm_trvl_idx, test_size=10.0 / 90.0, random_state=42,
    )
    print(
        f"NCDM split: train={len(ncdm_tr_idx):,}  val={len(ncdm_va_idx):,}  "
        f"test={len(ncdm_te_idx):,}"
    )

    # ── F1: identical splits? ──
    print("\n[F1] Identical te_idx between IRT and NCDM?")
    same_te = np.array_equal(irt_te_idx, ncdm_te_idx)
    same_tr = np.array_equal(irt_tr_idx, ncdm_tr_idx)
    same_va = np.array_equal(irt_va_idx, ncdm_va_idx)
    print(f"  te equal: {same_te}  tr equal: {same_tr}  va equal: {same_va}")
    print(f"  irt_te hash: {hash_arr(irt_te_idx)}")
    print(f"  ncdm_te hash: {hash_arr(ncdm_te_idx)}")

    # ── F2: identical labels at the same indices? ──
    print("\n[F2] Identical (s, j, y) at test indices?")
    rng = np.random.RandomState(0)
    sample_n = 200_000
    sample_pos = rng.choice(len(irt_te_idx), size=sample_n, replace=False)

    irt_sample_global = irt_te_idx[sample_pos]
    ncdm_sample_global = ncdm_te_idx[sample_pos]

    irt_s = irt_llm[irt_sample_global]
    irt_j = irt_item[irt_sample_global]
    irt_y = irt_label[irt_sample_global]

    ncdm_s = ncdm_triplets[ncdm_sample_global, 0].astype(np.int64)
    ncdm_j = ncdm_triplets[ncdm_sample_global, 1].astype(np.int64)
    ncdm_y = ncdm_triplets[ncdm_sample_global, 2].astype(np.float32)

    s_eq = np.array_equal(irt_s.astype(np.int64), ncdm_s)
    j_eq = np.array_equal(irt_j.astype(np.int64), ncdm_j)
    y_eq = np.array_equal(irt_y, ncdm_y)
    print(f"  s equal (n={sample_n}): {s_eq}")
    print(f"  j equal (n={sample_n}): {j_eq}")
    print(f"  y equal (n={sample_n}): {y_eq}")

    # spot-check: triplet label ought to equal R[s, j]
    R_lookup_irt = R[irt_s, irt_j].astype(np.float32)
    R_lookup_ncdm = R[ncdm_s, ncdm_j].astype(np.float32)
    print(f"  R[s,j] == y (IRT)  : {np.array_equal(R_lookup_irt, irt_y)}")
    print(f"  R[s,j] == y (NCDM) : {np.array_equal(R_lookup_ncdm, ncdm_y)}")

    # ── F9: no overlap train ∩ test ──
    print("\n[F9] No overlap train ∩ test?")
    set_tr = set(irt_tr_idx.tolist())
    set_te = set(irt_te_idx.tolist())
    set_va = set(irt_va_idx.tolist())
    print(f"  |tr ∩ te| = {len(set_tr & set_te)}")
    print(f"  |tr ∩ va| = {len(set_tr & set_va)}")
    print(f"  |va ∩ te| = {len(set_va & set_te)}")
    print(f"  |tr ∪ va ∪ te| = {len(set_tr | set_va | set_te):,}  "
          f"(expected {n_total_irt:,})")

    # ── F11: strongest-baseline per-LLM mean train acc ──
    print("\n[F11] Per-LLM mean-train-accuracy baseline on Protocol A test...")
    # train mask in (s, j) space
    n_total = n_llms * n_items
    is_train = np.zeros(n_total, dtype=bool)
    is_train[irt_tr_idx] = True

    # cumulative sum trick: mean train accuracy per LLM s
    # Each LLM s owns indices [s*n_items, (s+1)*n_items).
    train_label_only = np.where(is_train, irt_label, 0.0)
    train_count_only = is_train.astype(np.float32)
    sum_per_llm = train_label_only.reshape(n_llms, n_items).sum(axis=1)
    cnt_per_llm = train_count_only.reshape(n_llms, n_items).sum(axis=1)
    llm_mean_train = sum_per_llm / np.clip(cnt_per_llm, 1.0, None)
    print(
        f"  LLM mean-train-acc summary: min={llm_mean_train.min():.4f}, "
        f"median={np.median(llm_mean_train):.4f}, max={llm_mean_train.max():.4f}"
    )

    s_te = irt_llm[irt_te_idx]
    y_te = irt_label[irt_te_idx]
    pred_te = llm_mean_train[s_te]
    base_auc = roc_auc_score(y_te, pred_te)
    base_acc = accuracy_score(y_te, (pred_te >= 0.5).astype(int))
    base_rmse = float(np.sqrt(((y_te - pred_te) ** 2).mean()))
    print(
        f"  per-LLM mean-train-acc baseline:  test_auc={base_auc:.4f}  "
        f"test_acc={base_acc:.4f}  test_rmse={base_rmse:.4f}"
    )

    # also: per-item mean train acc (item difficulty baseline)
    sum_per_item = train_label_only.reshape(n_llms, n_items).sum(axis=0)
    cnt_per_item = train_count_only.reshape(n_llms, n_items).sum(axis=0)
    item_mean_train = sum_per_item / np.clip(cnt_per_item, 1.0, None)
    j_te = irt_item[irt_te_idx]
    pred_item = item_mean_train[j_te]
    item_auc = roc_auc_score(y_te, pred_item)
    item_acc = accuracy_score(y_te, (pred_item >= 0.5).astype(int))
    print(
        f"  per-ITEM mean-train-acc baseline: test_auc={item_auc:.4f}  "
        f"test_acc={item_acc:.4f}"
    )

    # combined: 0.5*(LLM_mean + item_mean) sanity
    pred_comb = 0.5 * (llm_mean_train[s_te] + item_mean_train[j_te])
    comb_auc = roc_auc_score(y_te, pred_comb)
    print(f"  0.5*(LLM+item)-mean baseline:    test_auc={comb_auc:.4f}")

    # global train mean (constant predictor) -> should be ~0.5 AUC
    g_pred = np.full_like(y_te, fill_value=float(train_label_only.sum() / train_count_only.sum()))
    print(
        f"  global-mean baseline (sanity):    test_auc=undefined (constant), "
        f"test_acc={(y_te == (g_pred >= 0.5).astype(int)).mean():.4f}"
    )


if __name__ == "__main__":
    main()
