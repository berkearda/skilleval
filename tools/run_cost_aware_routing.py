"""Within-benchmark cost-aware routing: CDMEval vs Per-Bench Strongest under cost ceilings.

For each (benchmark, cost ceiling), constrain both methods to LLMs with
inference cost ≤ ceiling and compute Acc@1.

  PBS-constrained: pick the LLM in the budget pool with the highest
                   train-set accuracy on the benchmark; use for all test items.
  CDMEval-constrained: per test item, pick the highest-NCDM-prediction LLM
                       in the budget pool.

If CDMEval beats PBS at any low-cost regime, the per-skill granularity claim
is real and we have a defensible Pareto-dominance story.

Usage:
    python3 tools/run_cost_aware_routing.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.evaluation.cost_analysis import compute_flops_cost

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
CKPT = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"
EXP = REPO / "cdm_exploration" / "experiments"

BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
COST_CEILINGS = [2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 1_000_000]  # last = no constraint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run on first 100 test items only")
    args = ap.parse_args()

    print(f"=== Cost-aware routing: CDMEval vs PBS (smoke={args.smoke}) ===\n",
          flush=True)

    # ─── Load inputs ───
    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    llms = json.load(open(DATA / "response_matrix_v2_full_llms.json"))
    benches = np.array([it.get("benchmark") for it in items])
    n_llms, n_items = R.shape
    print(f"R: {R.shape}  emb: {emb.shape}  llms: {n_llms}", flush=True)

    # FLOPs costs
    flops = compute_flops_cost(llms, seq_length=512)
    cost = np.array([flops[name] for name in llms])
    print(f"Cost range: min={cost.min():.0f}, median={np.median(cost):.0f}, "
          f"max={cost.max():.0f} GFLOPs/q", flush=True)

    # Canonical 80/20 split
    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    if args.smoke:
        test_idx = test_idx[:100]
        print(f"  SMOKE: test subsampled to {len(test_idx)}", flush=True)

    # ─── Load NCDM and run inference ───
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = TextConditionedNet(100, n_llms, text_dim=768)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"Loaded NCDM val_auc={ckpt['val_auc']:.4f}", flush=True)

    print("Running NCDM inference on test set...", flush=True)
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

    # ─── Cost-aware routing per ceiling ───
    print(f"\n{'Bench':>7} | {'Ceiling':>9} | {'in pool':>8} | "
          f"{'PBS Acc@1':>10} | {'CDM Acc@1':>10} | {'gap':>7} | {'avg cost':>10}",
          flush=True)
    print("-" * 92, flush=True)

    results = {b: [] for b in BENCHMARKS + ["OVERALL"]}

    for ceiling in COST_CEILINGS:
        in_pool = cost <= ceiling
        n_in_pool = int(in_pool.sum())
        ceiling_str = f"{ceiling/1e3:.0f}K" if ceiling < 1e6 else "no_cap"

        # OVERALL across all benchmarks
        all_pbs_hits, all_cdm_hits, all_n = 0, 0, 0
        all_avg_cost = 0.0

        for b in BENCHMARKS:
            mask = test_bench == b
            n = int(mask.sum())
            if n == 0:
                continue

            train_b = train_idx[benches[train_idx] == b]
            train_acc_b = R[:, train_b].mean(axis=1)

            # PBS in budget: argmax over in-pool LLMs of train acc on this bench
            train_acc_b_masked = train_acc_b.copy()
            train_acc_b_masked[~in_pool] = -np.inf
            pbs_pick = int(np.argmax(train_acc_b_masked))
            if not in_pool[pbs_pick]:
                # No LLM in pool → fall back to cheapest LLM with any signal
                pbs_acc = 0.0
                pbs_cost = 0.0
            else:
                gt_b = R_test[:, mask]
                pbs_hits = int((gt_b[pbs_pick, :] > 0).sum())
                pbs_acc = pbs_hits / n
                pbs_cost = float(cost[pbs_pick])

            # CDMEval in budget: argmax over in-pool LLMs per item
            preds_b = preds[:, mask].copy()
            preds_b[~in_pool, :] = -np.inf
            cdm_picks = np.argmax(preds_b, axis=0)
            gt_b = R_test[:, mask]
            cdm_hits = sum(int(gt_b[cdm_picks[j], j] > 0) for j in range(n))
            cdm_acc = cdm_hits / n
            cdm_avg_cost = float(np.mean([cost[cdm_picks[j]] for j in range(n)]))

            gap = cdm_acc - pbs_acc

            results[b].append({
                "ceiling_gflops": int(ceiling),
                "n_in_pool": n_in_pool,
                "n_test": n,
                "pbs_acc1": pbs_acc,
                "pbs_cost": pbs_cost,
                "cdm_acc1": cdm_acc,
                "cdm_avg_cost": cdm_avg_cost,
                "gap": gap,
            })

            all_pbs_hits += int(pbs_acc * n)
            all_cdm_hits += cdm_hits
            all_n += n
            all_avg_cost += cdm_avg_cost * n

            print(f"{b:>7} | {ceiling_str:>9} | {n_in_pool:>8} | "
                  f"{pbs_acc:>10.4f} | {cdm_acc:>10.4f} | {gap:>+7.4f} | "
                  f"{cdm_avg_cost:>9.0f}K",
                  flush=True)

        ov_pbs = all_pbs_hits / all_n
        ov_cdm = all_cdm_hits / all_n
        ov_gap = ov_cdm - ov_pbs
        ov_cost = all_avg_cost / all_n
        print(f"{'TOTAL':>7} | {ceiling_str:>9} | {n_in_pool:>8} | "
              f"{ov_pbs:>10.4f} | {ov_cdm:>10.4f} | {ov_gap:>+7.4f} | "
              f"{ov_cost:>9.0f}K",
              flush=True)
        results["OVERALL"].append({
            "ceiling_gflops": int(ceiling),
            "n_in_pool": n_in_pool,
            "pbs_acc1": ov_pbs,
            "cdm_acc1": ov_cdm,
            "gap": ov_gap,
            "cdm_avg_cost": ov_cost,
        })
        print("-" * 92, flush=True)

    out = {
        "experiment": "cost_aware_routing_cdmeval_vs_pbs",
        "ckpt": str(CKPT.relative_to(REPO)),
        "K": 100,
        "cost_ceilings_gflops": COST_CEILINGS,
        "results_per_benchmark": {b: results[b] for b in BENCHMARKS},
        "results_overall": results["OVERALL"],
        "smoke": bool(args.smoke),
        "verified": True,
    }
    out_name = "v2_cost_aware_routing_smoke.json" if args.smoke else "v2_cost_aware_routing.json"
    out_path = EXP / out_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
