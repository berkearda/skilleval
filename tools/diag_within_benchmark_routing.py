"""Diagnostic: does CDMEval add value over Per-Benchmark Strongest WITHIN a benchmark?

For each benchmark, compare:
  (a) Per-Bench Strongest -- pick one LLM per benchmark from train, use for all test items
  (b) CDMEval -- pick a different LLM per item using NCDM predictions

If (b) beats (a) within a benchmark, CDMEval's per-skill structure adds value
beyond knowing the benchmark. If they tie, CDMEval is just an expensive way to
do per-benchmark routing.
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


def main() -> int:
    print("=== Within-benchmark routing comparison ===\n", flush=True)

    # Load data
    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    benches = np.array([it.get("benchmark") for it in items])
    n_llms, n_items = R.shape
    print(f"R: {R.shape}  emb: {emb.shape}", flush=True)

    # Canonical 80/20 split
    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    print(f"train: {len(train_idx)}, test: {len(test_idx)}", flush=True)

    # Load NCDM with K=100 Q-matrix
    K = 100
    q_matrix = np.load(DATA / f"qmatrix_v2_K{K}.npy")
    print(f"Q-matrix K={K} shape: {q_matrix.shape}", flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = TextConditionedNet(K, n_llms, text_dim=768)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"Loaded NCDM checkpoint val_auc={ckpt.get('val_auc')}", flush=True)

    # Run NCDM inference on every test item × every LLM
    print("\nRunning NCDM inference on test set...", flush=True)
    test_emb = torch.tensor(emb[test_idx], dtype=torch.float32)
    test_q = torch.tensor(q_matrix[test_idx], dtype=torch.float32)
    all_llm_ids = torch.arange(n_llms)

    preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    with torch.no_grad():
        for j in range(len(test_idx)):
            te = test_emb[j].unsqueeze(0).expand(n_llms, -1)
            tq = test_q[j].unsqueeze(0).expand(n_llms, -1)
            preds[:, j] = net(all_llm_ids, te, tq).numpy()
    print(f"  predictions shape: {preds.shape}", flush=True)

    R_test = R[:, test_idx]
    test_bench = benches[test_idx]

    # ── Compute Per-Bench Strongest baseline ──
    print("\n=== Per-Benchmark Strongest baseline ===", flush=True)
    pbs_pick = {}
    for b in BENCHMARKS:
        train_b = train_idx[benches[train_idx] == b]
        train_acc = R[:, train_b].mean(axis=1)
        pbs_pick[b] = int(np.argmax(train_acc))
        print(f"  {b:>7}: PBS picks LLM idx={pbs_pick[b]:>4}, train acc={train_acc.max():.4f}",
              flush=True)

    # ── Compute Per-Bench results ──
    print("\n=== Per-benchmark Acc@1 (CDMEval vs Per-Bench Strongest) ===", flush=True)
    print(f"{'Bench':>7} | {'n':>4} | {'PBS Acc@1':>10} | {'CDM Acc@1':>10} | "
          f"{'CDM picks':>10} | {'unique LLMs':>12} | {'gap':>7}", flush=True)
    print("-" * 80, flush=True)

    summary = {}
    for b in BENCHMARKS:
        mask = test_bench == b
        n = int(mask.sum())
        if n == 0:
            continue

        # PBS: always pick pbs_pick[b], measure correctness on test items in this bench
        pbs_llm = pbs_pick[b]
        gt_b = R_test[:, mask]  # (n_llms, n_b)
        pbs_hits = int((gt_b[pbs_llm, :] > 0).sum())
        pbs_acc = pbs_hits / n

        # CDMEval: argmax LLM per item, measure correctness
        preds_b = preds[:, mask]
        cdm_picks = np.argmax(preds_b, axis=0)  # (n_b,) LLM idx per item
        cdm_hits = sum(int(gt_b[cdm_picks[j], j] > 0) for j in range(n))
        cdm_acc = cdm_hits / n

        # How often does CDMEval pick a different LLM than PBS?
        n_diff_pick = int((cdm_picks != pbs_llm).sum())
        n_unique = int(len(np.unique(cdm_picks)))

        gap = cdm_acc - pbs_acc
        summary[b] = {
            "n_items": n,
            "pbs_acc1": pbs_acc,
            "cdm_acc1": cdm_acc,
            "gap_cdm_minus_pbs": gap,
            "cdm_picks_diff_from_pbs": n_diff_pick,
            "cdm_unique_llms_picked": n_unique,
        }
        print(f"{b:>7} | {n:>4} | {pbs_acc:>10.4f} | {cdm_acc:>10.4f} | "
              f"{n_diff_pick:>10} | {n_unique:>12} | {gap:>+7.4f}", flush=True)

    # Aggregated overall numbers
    total_pbs_hits = 0
    total_cdm_hits = 0
    total_n = 0
    for b, s in summary.items():
        total_pbs_hits += int(s["pbs_acc1"] * s["n_items"])
        total_cdm_hits += int(s["cdm_acc1"] * s["n_items"])
        total_n += s["n_items"]
    print("-" * 80, flush=True)
    print(f"{'TOTAL':>7} | {total_n:>4} | {total_pbs_hits/total_n:>10.4f} | "
          f"{total_cdm_hits/total_n:>10.4f} | "
          f"{'':>10} | {'':>12} | {(total_cdm_hits-total_pbs_hits)/total_n:>+7.4f}",
          flush=True)

    # Headline interpretation
    print("\n=== Interpretation ===", flush=True)
    n_wins = sum(1 for s in summary.values() if s["gap_cdm_minus_pbs"] > 0)
    n_loses = sum(1 for s in summary.values() if s["gap_cdm_minus_pbs"] < 0)
    print(f"CDMEval beats PBS on {n_wins}/{len(summary)} benchmarks, "
          f"loses on {n_loses}/{len(summary)}", flush=True)
    print(f"Diversity: CDMEval picks {sum(s['cdm_unique_llms_picked'] for s in summary.values())} "
          f"distinct LLMs across all benchmarks (PBS picks {len(BENCHMARKS)})",
          flush=True)

    out = {
        "experiment": "within_benchmark_routing",
        "ckpt": str(CKPT.relative_to(REPO)),
        "K": K,
        "summary_per_benchmark": summary,
        "overall": {
            "pbs_acc1": total_pbs_hits / total_n,
            "cdm_acc1": total_cdm_hits / total_n,
            "gap": (total_cdm_hits - total_pbs_hits) / total_n,
        },
        "verified": True,
    }
    out_path = EXP / "v2_within_benchmark_routing.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
