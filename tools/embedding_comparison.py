"""Compare text embedding models for the text-conditioned NCDM.

Trains Protocol B (exercise-level split) with each embedding model,
evaluates AUC and routing accuracy, and prints a comparison table.

Usage:
    python tools/embedding_comparison.py
    python tools/embedding_comparison.py device=mps model.epochs=15
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


K = 50

# Model-specific prefixes for encoding documents/passages.
# E5 models require "passage: " for documents; others need nothing.
PASSAGE_PREFIX = {
    "intfloat/e5-large-v2": "passage: ",
}


def encode_items(sbert_name: str, texts: list[str], cache_dir: Path) -> np.ndarray:
    """Encode item texts with a given SBERT model. Caches per model."""
    safe_name = sbert_name.replace("/", "__")
    cache_path = cache_dir / f"item_text_embeddings_{safe_name}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        print(f"  Loaded cached embeddings: {data['embeddings'].shape}")
        return data["embeddings"]

    from sentence_transformers import SentenceTransformer

    prefix = PASSAGE_PREFIX.get(sbert_name, "")
    if prefix:
        print(f"  Applying prefix '{prefix}' for {sbert_name}")
        texts = [prefix + t for t in texts]

    print(f"  Encoding {len(texts)} texts with {sbert_name}...")
    model = SentenceTransformer(sbert_name)
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64,
    )

    np.savez(cache_path, embeddings=embeddings)
    print(f"  Saved: {embeddings.shape} -> {cache_path}")
    return embeddings


def routing_eval(net, text_embeddings, q_matrix, response_vals, test_items,
                 n_llms, device):
    """Compute routing acc@1/3/5 and strongest model baseline."""
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    accs_at = {1: 0, 3: 0, 5: 0}
    total = 0

    for item_idx in test_items:
        gt = response_vals[:, int(item_idx)]
        if gt.sum() == 0:
            continue

        emb = torch.tensor(
            text_embeddings[int(item_idx)], dtype=torch.float32, device=device,
        )
        emb_batch = emb.unsqueeze(0).expand(n_llms, -1)
        q_row = torch.tensor(
            q_matrix[int(item_idx)], dtype=torch.float32, device=device,
        )
        q_batch = q_row.unsqueeze(0).expand(n_llms, -1)

        with torch.no_grad():
            preds = net(all_llm_ids, emb_batch, q_batch).cpu().numpy()

        ranking = np.argsort(-preds)
        total += 1
        for k in accs_at:
            if gt[ranking[:k]].sum() > 0:
                accs_at[k] += 1

    best_llm = response_vals.mean(axis=1).argmax()
    strongest = sum(
        response_vals[best_llm, int(i)]
        for i in test_items if response_vals[:, int(i)].sum() > 0
    ) / total

    return {
        "acc@1": accs_at[1] / total,
        "acc@3": accs_at[3] / total,
        "acc@5": accs_at[5] / total,
        "strongest": strongest,
        "n_items": total,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_text_dataloader
    from cdmeval.data.response_matrix import (
        build_triplets, load_q_matrix, load_response_matrix,
    )
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # Load experiment config
    exp_cfg = OmegaConf.load(Path("configs/experiment/embedding_comparison.yaml"))
    embedding_models = exp_cfg.embedding_models

    # ── Load shared data ──
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv",
    )
    q_matrix, skill_cols = load_q_matrix(data_dir / f"q_matrix_hac{K}.csv")
    n_skills = q_matrix.shape[1]
    triplets = build_triplets(response_df)
    response_vals = response_df.values

    with open(data_dir / "skills_extracted.json") as f:
        items_data = json.load(f)
    texts = [item["problem"] for item in items_data]

    print(f"Data: {n_llms} LLMs, {n_items} items, {n_skills} skills")

    # ── Exercise-level split (same random_state=42 as run_experiments.py) ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(
        all_items, test_size=0.2, random_state=42,
    )
    train_val_triplets = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    test_triplets = triplets[np.isin(triplets[:, 1].astype(int), test_items)]
    tidx = np.arange(len(train_val_triplets))
    train_idx, val_idx = train_test_split(tidx, test_size=0.1, random_state=42)
    train_trip = train_val_triplets[train_idx]
    val_trip = train_val_triplets[val_idx]

    print(f"Exercise split: {len(train_items)} train / {len(test_items)} test items")

    bs = cfg.model.batch_size
    results = []

    for emb_cfg in embedding_models:
        sbert_name = emb_cfg.name
        text_dim = emb_cfg.dim

        print("\n" + "=" * 70)
        print(f"EMBEDDING MODEL: {sbert_name}  (dim={text_dim})")
        print("=" * 70)

        # 1. Encode
        text_embeddings = encode_items(sbert_name, texts, data_dir)
        assert text_embeddings.shape[1] == text_dim, (
            f"Expected dim={text_dim}, got {text_embeddings.shape[1]}"
        )

        # 2. Build dataloaders
        seed_everything(42)
        tr_txt = make_text_dataloader(
            train_trip, text_embeddings, q_matrix, bs, shuffle=True,
        )
        va_txt = make_text_dataloader(
            val_trip, text_embeddings, q_matrix, bs, shuffle=False,
        )
        te_txt = make_text_dataloader(
            test_triplets, text_embeddings, q_matrix, bs, shuffle=False,
        )

        # 3. Train
        print(f"\nTraining TextConditionedNet (text_dim={text_dim})...")
        net = TextConditionedNet(n_skills, n_llms, text_dim)
        net = train_text_model(
            net, tr_txt, va_txt,
            epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
        )

        # 4. Evaluate Protocol B
        auc, acc, rmse = eval_text_model(net, te_txt, device)
        print(f"Protocol B: AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}")

        # 5. Routing accuracy
        print("Computing routing accuracy...")
        routing = routing_eval(
            net, text_embeddings, q_matrix, response_vals,
            test_items, n_llms, device,
        )
        print(
            f"Routing: acc@1={routing['acc@1']:.4f}, "
            f"acc@3={routing['acc@3']:.4f}, acc@5={routing['acc@5']:.4f}"
        )

        results.append({
            "model": sbert_name,
            "dim": text_dim,
            "auc": float(auc),
            "acc": float(acc),
            "rmse": float(rmse),
            "routing_acc1": routing["acc@1"],
            "routing_acc3": routing["acc@3"],
            "routing_acc5": routing["acc@5"],
            "strongest": routing["strongest"],
        })

    # ── Comparison table ──
    print("\n" + "=" * 100)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 100)
    header = (
        f"{'Model':<30} {'Dim':>5} {'AUC':>8} {'Acc':>8} {'RMSE':>8} "
        f"{'Rt@1':>8} {'Rt@3':>8} {'Rt@5':>8} {'Strongest':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<30} {r['dim']:>5d} {r['auc']:>8.4f} {r['acc']:>8.4f} "
            f"{r['rmse']:>8.4f} {r['routing_acc1']:>8.4f} {r['routing_acc3']:>8.4f} "
            f"{r['routing_acc5']:>8.4f} {r['strongest']:>10.4f}"
        )

    # Best per metric
    print()
    for metric in ["auc", "routing_acc1", "routing_acc5"]:
        best = max(results, key=lambda r: r[metric])
        label = metric.replace("routing_", "Routing ").replace("auc", "AUC")
        print(f"  Best {label}: {best['model']} ({best[metric]:.4f})")

    # Save
    out_path = data_dir / "embedding_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
