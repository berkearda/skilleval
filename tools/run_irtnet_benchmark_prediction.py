"""IrtNet benchmark-level prediction Pearson r.

For each LLM m and benchmark b: predicted mean accuracy = mean over test
items in b of IrtNet(m, item). True mean accuracy = R[m, items in b].mean().
Across 3,811 LLMs, compute Pearson r per benchmark and overall.

Multi-seed (3 seeds) for d=100. Saves to v2_irtnet_benchmark_prediction.json.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_irtnet_headtohead import import_irtnet_model_class

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
CKPT_DIR = REPO / "cdm_exploration" / "checkpoints" / "irtnet"
EXP = REPO / "cdm_exploration" / "experiments"
BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
SEEDS = [42, 43, 44]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"=== IrtNet d={args.d_model} benchmark-level prediction ===\n", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    benches = np.array([it.get("benchmark") for it in items])
    n_llms, n_items = R.shape
    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    if args.smoke:
        test_idx = test_idx[:100]
        print(f"  SMOKE: 100 items", flush=True)
    R_test = R[:, test_idx]
    test_bench = benches[test_idx]

    MoEClassifier = import_irtnet_model_class()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n", flush=True)

    per_seed_results = []

    for seed in SEEDS:
        ck = CKPT_DIR / f"d{args.d_model}_s{seed}.pt"
        if not ck.exists():
            print(f"  MISSING: {ck}", flush=True)
            continue
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        model_hp = {
            "model_embed_dim": cfg.get("model_embed_dim", args.d_model),
            "num_experts": cfg.get("num_experts", 39),
            "top_k_experts": cfg.get("top_k_experts", 39),
            "expert_hidden_dim": cfg.get("expert_hidden_dim", 512),
            "shared_expert_hidden_dim": cfg.get("shared_expert_hidden_dim", 512),
            "expert_output_dim": cfg.get("expert_output_dim", 256),
            "dropout_rate": cfg.get("dropout_rate", 0.5),
            "embedding_noise": cfg.get("embedding_noise", 0.05),
        }
        prompt_embeddings = torch.tensor(emb, dtype=torch.float32)
        model = MoEClassifier(num_models=n_llms, num_prompts=n_items,
                                prompt_embeddings=prompt_embeddings, **model_hp)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)

        all_llm_ids = torch.arange(n_llms, device=device)
        preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
        with torch.no_grad():
            for j, item_idx in enumerate(test_idx):
                p_id = torch.full((n_llms,), int(item_idx), dtype=torch.long,
                                    device=device)
                preds[:, j] = model(all_llm_ids, p_id).cpu().numpy()

        # Per-bench Pearson r between predicted and true per-LLM mean accuracy
        per_bench = {}
        for b in BENCHMARKS:
            mask = test_bench == b
            if mask.sum() == 0:
                continue
            true_acc = R_test[:, mask].mean(axis=1)
            pred_acc = preds[:, mask].mean(axis=1)
            r, _ = pearsonr(true_acc, pred_acc)
            per_bench[b] = float(r)

        # Concatenated overall
        true_all = np.concatenate([R_test[:, test_bench == b].mean(axis=1)
                                      for b in BENCHMARKS])
        pred_all = np.concatenate([preds[:, test_bench == b].mean(axis=1)
                                      for b in BENCHMARKS])
        r_overall, _ = pearsonr(true_all, pred_all)

        result = {"seed": seed, "per_bench": per_bench,
                    "overall_concatenated": float(r_overall)}
        per_seed_results.append(result)
        print(f"  seed {seed}:", flush=True)
        for b in BENCHMARKS:
            print(f"    {b:>7}: r = {per_bench[b]:.4f}", flush=True)
        print(f"    overall: r = {r_overall:.4f}\n", flush=True)

    # Aggregate
    print("=== Multi-seed mean ± std ===", flush=True)
    summary = {"d_model": args.d_model, "n_seeds": len(per_seed_results),
                 "per_seed": per_seed_results, "summary": {}}
    for b in BENCHMARKS:
        vals = [r["per_bench"][b] for r in per_seed_results if b in r["per_bench"]]
        m, s = float(np.mean(vals)), float(np.std(vals))
        summary["summary"][b] = {"mean": m, "std": s}
        print(f"  {b:>7}: {m:.4f} ± {s:.4f}", flush=True)
    overall_vals = [r["overall_concatenated"] for r in per_seed_results]
    summary["summary"]["overall"] = {"mean": float(np.mean(overall_vals)),
                                        "std": float(np.std(overall_vals))}
    print(f"  overall: {np.mean(overall_vals):.4f} ± {np.std(overall_vals):.4f}",
            flush=True)

    out_path = EXP / f"v2_irtnet_d{args.d_model}_benchmark_prediction.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
