"""T-027: Acc@k for k > 1 with Per-Bench Strongest restricted to per-benchmark top-k pool.

PBS at top-1: pick the LLM with highest train acc on the test item's benchmark.
PBS at top-k: rank LLMs by train acc on that benchmark, take top-k.
CDMEval at top-k: rank LLMs by NCDM predicted P(correct) per item, take top-k.

For each test item, Acc@k = 1 if any of the top-k LLMs got the item right.

If CDMEval wins at k > 1, the per-item routing diversity is doing real work
that PBS's per-benchmark fixed pool cannot match.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cdmeval.modeling.text_conditioned import TextConditionedNet

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
CKPT = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"
EXP = REPO / "cdm_exploration" / "experiments"

BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


def main() -> None:
    print("=== Acc@k: CDMEval vs Per-Bench Strongest (top-k per benchmark) ===\n",
          flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    benches = np.array([it.get("benchmark") for it in items])
    n_llms, n_items = R.shape

    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = TextConditionedNet(100, n_llms, text_dim=768)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    print("Inference...", flush=True)
    test_emb = torch.tensor(emb[test_idx], dtype=torch.float32)
    test_q = torch.tensor(q_matrix[test_idx], dtype=torch.float32)
    all_llm_ids = torch.arange(n_llms)
    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    with torch.no_grad():
        for j in range(len(test_idx)):
            te = test_emb[j].unsqueeze(0).expand(n_llms, -1)
            tq = test_q[j].unsqueeze(0).expand(n_llms, -1)
            preds[:, j] = net(all_llm_ids, te, tq).numpy()

    R_test = R[:, test_idx]
    test_bench = benches[test_idx]

    # PBS per-benchmark top-K rankings (precompute)
    pbs_topk = {}  # bench -> array of LLM indices ranked by train acc desc
    for b in BENCHMARKS:
        train_b = train_idx[benches[train_idx] == b]
        train_acc_b = R[:, train_b].mean(axis=1)
        pbs_topk[b] = np.argsort(-train_acc_b)  # full ranking, take [:k] later

    # Compute Acc@k for both methods
    KS = [1, 3, 5, 10, 20, 50]

    print(f"\n{'k':>4} | {'PBS Acc@k':>10} | {'CDM Acc@k':>10} | {'gap':>8}",
          flush=True)
    print("-" * 44, flush=True)

    overall_results = {}
    per_bench_results = {b: {} for b in BENCHMARKS}

    for k in KS:
        # PBS: top-k per-benchmark LLMs
        pbs_hits = 0
        cdm_hits = 0
        per_bench_pbs = {b: 0 for b in BENCHMARKS}
        per_bench_cdm = {b: 0 for b in BENCHMARKS}
        per_bench_n = {b: 0 for b in BENCHMARKS}

        for j in range(len(test_idx)):
            b = test_bench[j]
            per_bench_n[b] += 1

            # PBS top-k for this item's benchmark
            pbs_picks = pbs_topk[b][:k]
            if R_test[pbs_picks, j].max() > 0:
                pbs_hits += 1
                per_bench_pbs[b] += 1

            # CDMEval top-k by predicted score
            if k < n_llms:
                cdm_picks = np.argpartition(-preds[:, j], k)[:k]
            else:
                cdm_picks = np.arange(n_llms)
            if R_test[cdm_picks, j].max() > 0:
                cdm_hits += 1
                per_bench_cdm[b] += 1

        pbs_acc = pbs_hits / len(test_idx)
        cdm_acc = cdm_hits / len(test_idx)
        gap = cdm_acc - pbs_acc
        print(f"{k:>4} | {pbs_acc:>10.4f} | {cdm_acc:>10.4f} | {gap:>+8.4f}",
              flush=True)

        overall_results[f"acc@{k}"] = {
            "pbs": pbs_acc, "cdmeval": cdm_acc, "gap": gap,
        }
        for b in BENCHMARKS:
            n_b = per_bench_n[b]
            if n_b > 0:
                per_bench_results[b][f"acc@{k}"] = {
                    "n": n_b,
                    "pbs": per_bench_pbs[b] / n_b,
                    "cdmeval": per_bench_cdm[b] / n_b,
                    "gap": (per_bench_cdm[b] - per_bench_pbs[b]) / n_b,
                }

    # Per-bench breakdown for k=1, k=10
    print(f"\nPer-benchmark detail (k=1, k=5, k=10):", flush=True)
    print(f"{'Bench':>7} | {'n':>4} |"
          f" {'PBS@1':>7} {'CDM@1':>7} {'Δ@1':>7} |"
          f" {'PBS@5':>7} {'CDM@5':>7} {'Δ@5':>7} |"
          f" {'PBS@10':>7} {'CDM@10':>7} {'Δ@10':>7}",
          flush=True)
    print("-" * 100, flush=True)
    for b in BENCHMARKS:
        rb = per_bench_results[b]
        n = next(iter(rb.values()))["n"] if rb else 0
        print(f"{b:>7} | {n:>4} |"
              f" {rb['acc@1']['pbs']:>7.3f} {rb['acc@1']['cdmeval']:>7.3f} {rb['acc@1']['gap']:>+7.3f} |"
              f" {rb['acc@5']['pbs']:>7.3f} {rb['acc@5']['cdmeval']:>7.3f} {rb['acc@5']['gap']:>+7.3f} |"
              f" {rb['acc@10']['pbs']:>7.3f} {rb['acc@10']['cdmeval']:>7.3f} {rb['acc@10']['gap']:>+7.3f}",
              flush=True)

    out = {
        "experiment": "acc_at_k_cdmeval_vs_pbs_perbench_topk",
        "ckpt": str(CKPT.relative_to(REPO)),
        "K": 100,
        "n_test_items": int(len(test_idx)),
        "ks": KS,
        "overall": overall_results,
        "per_benchmark": per_bench_results,
        "verified": True,
    }
    out_path = EXP / "v2_acc_at_k_pbs.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
