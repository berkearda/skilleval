"""Compute per-benchmark Acc@1 for IrtNet checkpoints.

Loads d=100 (or any d) checkpoint, runs inference on test items, splits by
benchmark, computes Acc@1 per benchmark and overall. Multi-seed mean+std.

Usage on Euler:
    python3 tools/run_irtnet_perbench.py --d_model 100 [--smoke]
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
# Reuse the IrtNet model loader from the head-to-head script
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

    print(f"=== IrtNet d={args.d_model} per-benchmark Acc@1 ===\n", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    benches = np.array([it.get("benchmark") for it in items])
    n_llms, n_items = R.shape

    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2,
                                            random_state=42)
    if args.smoke:
        test_idx = test_idx[:100]
        print(f"  SMOKE mode: 100 test items", flush=True)
    R_test = R[:, test_idx]
    test_bench = benches[test_idx]

    # IrtNet config matches what we trained
    MoEClassifier = import_irtnet_model_class()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n", flush=True)

    per_seed_per_bench = {b: [] for b in BENCHMARKS}
    per_seed_overall = []

    for seed in SEEDS:
        ckpt_path = CKPT_DIR / f"d{args.d_model}_s{seed}.pt"
        if not ckpt_path.exists():
            print(f"  MISSING: {ckpt_path}", flush=True)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]

        # Reconstruct model with the correct IrtNet MoEClassifier signature
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

        model = MoEClassifier(
            num_models=n_llms,
            num_prompts=n_items,
            prompt_embeddings=prompt_embeddings,
            **model_hp,
        )
        model.load_state_dict(sd)
        model.eval().to(device)

        # Inference: model takes (model_ids, prompt_ids), both index tensors
        all_llm_ids = torch.arange(n_llms, device=device)

        preds = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
        with torch.no_grad():
            for j, item_idx in enumerate(test_idx):
                p_id = torch.full((n_llms,), int(item_idx), dtype=torch.long,
                                    device=device)
                p = model(all_llm_ids, p_id).cpu().numpy()
                preds[:, j] = p

        # Routing Acc@1: argmax over LLMs per test item
        cdm_picks = np.argmax(preds, axis=0)
        overall_hits = sum(int(R_test[cdm_picks[j], j] > 0) for j in range(len(test_idx)))
        overall_acc = overall_hits / len(test_idx)
        per_seed_overall.append(overall_acc)
        print(f"  seed {seed}: overall Acc@1 = {overall_acc:.4f}", flush=True)

        for b in BENCHMARKS:
            mask = test_bench == b
            n = int(mask.sum())
            if n == 0:
                continue
            preds_b = preds[:, mask]
            picks_b = np.argmax(preds_b, axis=0)
            R_test_b = R_test[:, mask]
            hits_b = sum(int(R_test_b[picks_b[j], j] > 0) for j in range(n))
            acc_b = hits_b / n
            per_seed_per_bench[b].append(acc_b)
            print(f"    {b:>7}: n={n}, Acc@1={acc_b:.4f}", flush=True)
        print(flush=True)

    print(f"\n=== Multi-seed mean ± std (d={args.d_model}, 3 seeds) ===", flush=True)
    print(f"  overall Acc@1: {np.mean(per_seed_overall):.4f} ± {np.std(per_seed_overall):.4f}",
          flush=True)
    summary = {
        "d_model": args.d_model,
        "smoke": args.smoke,
        "n_seeds": len(per_seed_overall),
        "overall_mean": float(np.mean(per_seed_overall)),
        "overall_std": float(np.std(per_seed_overall)),
        "per_bench": {},
    }
    for b in BENCHMARKS:
        if not per_seed_per_bench[b]:
            continue
        v = per_seed_per_bench[b]
        m, s = float(np.mean(v)), float(np.std(v))
        print(f"  {b:>7}: {m:.4f} ± {s:.4f}", flush=True)
        summary["per_bench"][b] = {"mean": m, "std": s, "per_seed": v}

    out_name = f"v2_irtnet_d{args.d_model}_perbench{'_smoke' if args.smoke else ''}.json"
    out_path = EXP / out_name
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
