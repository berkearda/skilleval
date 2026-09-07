"""KNN baseline for CDMEval Table 1.

Recipe (matches EmbedLLM Table 4 and IRTNet Table 1 KNN baseline):
  For each test (model, query) pair:
    1. Embed query via SBERT (already cached on disk).
    2. Cosine-retrieve K=10 nearest training queries.
    3. Average that model's correctness on those K (continuous score).
    4. Threshold at 0.5 (= majority vote) for binary call.

Reports the same columns as Table 1 / tab:perbench:
  - Test AUC (uses continuous score)
  - Test Acc (uses binary call)
  - Routing Acc@1, Acc@3, Acc@5, Acc@10 (uses continuous score for ranking)
  - Per-benchmark Acc@1

Usage:
    python3 tools/run_knn_baseline.py [--K 10] [--smoke]
        --smoke runs on the first 100 test items only for verification
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"

BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="run on first 100 test items only")
    args = ap.parse_args()

    print(f"=== KNN baseline (K={args.K}, smoke={args.smoke}) ===\n", flush=True)

    # ─── Load inputs (B2 already verified) ───
    R = np.load(DATA / "response_matrix_v2_full.npy")  # (3811, 9523), binary
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    n_llms, n_items = R.shape
    assert emb.shape == (n_items, 768)
    assert len(items) == n_items
    print(f"R: {R.shape}  emb: {emb.shape}  llms: {n_llms}  items: {n_items}",
          flush=True)

    # Per-item benchmark labels for per-benchmark breakdowns
    item_bench = np.array([it.get("benchmark") for it in items])

    # ─── Canonical 80/20 split (same as Table 1) ───
    all_idx = np.arange(n_items)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=42,
    )
    print(f"train items: {len(train_idx)}, test items: {len(test_idx)}",
          flush=True)
    if args.smoke:
        test_idx = test_idx[:100]
        print(f"  SMOKE: subsampling test to first {len(test_idx)} items",
              flush=True)

    # ─── Cosine similarity: test queries vs train queries ───
    t0 = time.time()
    sim = cosine_similarity(emb[test_idx], emb[train_idx])  # (n_test, n_train)
    print(f"cosine sims computed: shape={sim.shape}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ─── Top-K nearest training items per test item ───
    t0 = time.time()
    K = args.K
    # argpartition gives unsorted top-K, then we sort just those K
    top_k_local = np.argpartition(-sim, K, axis=1)[:, :K]  # indices into train_idx
    # Sort within top-K so neighbour 0 is closest (not strictly needed for averaging)
    for i in range(len(test_idx)):
        order = np.argsort(-sim[i, top_k_local[i]])
        top_k_local[i] = top_k_local[i][order]
    top_k_global = train_idx[top_k_local]  # (n_test, K) -> indices into R columns
    print(f"top-{K} retrieval done ({time.time()-t0:.1f}s)", flush=True)

    # ─── Predict per (model, test item) cell ───
    # For each test item q, look up R[:, top_k_global[q]]: (n_llms, K) of correctness.
    # Mean across K → predicted probability per model. Stack into (n_llms, n_test).
    t0 = time.time()
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    for j, neighbours in enumerate(top_k_global):
        preds[:, j] = R[:, neighbours].mean(axis=1)
    print(f"predictions computed: shape={preds.shape}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ─── Ground truth on test items ───
    R_test = R[:, test_idx]  # (n_llms, n_test)

    # ─── Test AUC (per-cell binary classification with continuous scores) ───
    from sklearn.metrics import roc_auc_score
    flat_pred = preds.flatten()
    flat_true = R_test.flatten()
    test_auc = roc_auc_score(flat_true, flat_pred)
    print(f"\nTest AUC: {test_auc:.4f}", flush=True)

    # ─── Test Acc (binary call via threshold-at-0.5) ───
    flat_pred_bin = (flat_pred > 0.5).astype(np.int8)
    test_acc = float((flat_pred_bin == flat_true).mean())
    print(f"Test Acc (threshold=0.5, == majority vote for K=10): {test_acc:.4f}",
          flush=True)

    # ─── Routing Acc@k ───
    # For each test item, rank LLMs by predicted score; check if any of the
    # top-k actually got it right. Acc@k = fraction of test items where >=1 of
    # top-k LLMs is correct.
    def acc_at_k(k):
        # argsort descending by score per column (test item)
        # preds is (n_llms, n_test); we want top-k LLMs per test item
        # argpartition for speed
        topk_llm = np.argpartition(-preds, k, axis=0)[:k, :]  # (k, n_test)
        # For each test item, check if any topk LLM has gt=1
        hits = 0
        for j in range(preds.shape[1]):
            if R_test[topk_llm[:, j], j].max() > 0:
                hits += 1
        return hits / preds.shape[1]

    routing = {}
    for k in [1, 3, 5, 10]:
        routing[f"acc@{k}"] = acc_at_k(k)
        print(f"Routing Acc@{k}: {routing[f'acc@{k}']:.4f}", flush=True)

    # ─── Per-benchmark Acc@1 ───
    print(f"\nPer-benchmark Acc@1:", flush=True)
    per_bench = {}
    test_bench = item_bench[test_idx]
    for b in BENCHMARKS:
        mask = test_bench == b
        n = int(mask.sum())
        if n == 0:
            continue
        preds_b = preds[:, mask]
        gt_b = R_test[:, mask]
        topk_llm_b = np.argpartition(-preds_b, 1, axis=0)[:1, :]
        hits = sum(int(gt_b[topk_llm_b[0, j], j] > 0) for j in range(preds_b.shape[1]))
        per_bench[b] = {"n_items": n, "knn_acc1": hits / n}
        print(f"  {b:>7}: n={n:>4}  Acc@1={hits/n:.4f}", flush=True)

    # ─── Summary JSON ───
    out = {
        "experiment": "knn_baseline",
        "K_neighbours": args.K,
        "split": ("from sklearn.model_selection import train_test_split; "
                  "train_test_split(np.arange(9523), test_size=0.2, "
                  "random_state=42)"),
        "n_llms": int(n_llms),
        "n_train_items": int(len(train_idx)),
        "n_test_items": int(len(test_idx)),
        "smoke": bool(args.smoke),
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "routing": routing,
        "per_benchmark": per_bench,
        "verified": True,
    }

    out_name = "v2_knn_baseline_smoke.json" if args.smoke else "v2_knn_baseline.json"
    out_path = EXP / out_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
