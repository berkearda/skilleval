"""Train text-conditioned NCDM on the expanded dataset (3811 LLMs x 7039 items).

Usage:
    python tools/train_expanded.py device=cpu                    # full dataset
    python tools/train_expanded.py device=cpu '+subset.n_llms=381'  # 10% subset
    python tools/train_expanded.py device=mps model.epochs=5     # quick test
"""

import json
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


def fix_zero_skill_items(q_matrix, skill_embeddings_path, items_data, text_emb_path):
    """Assign items with 0 skills to nearest cluster centroid."""
    zero_mask = q_matrix.sum(axis=1) == 0
    n_zero = zero_mask.sum()
    if n_zero == 0:
        return q_matrix

    print(f"  Fixing {n_zero} items with 0 skills...")

    # Load skill embeddings to get cluster centroids
    data = np.load(skill_embeddings_path, allow_pickle=True)
    skill_embs = data["embeddings"]
    labels = data["labels"]
    K = q_matrix.shape[1]

    # Compute cluster centroids
    centroids = np.zeros((K, skill_embs.shape[1]))
    cluster_ids = sorted(set(labels))
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        if mask.sum() > 0:
            centroids[i] = skill_embs[mask].mean(axis=0)
    # L2 normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-8)

    # Load item text embeddings for the zero-skill items
    text_embs = np.load(text_emb_path)["embeddings"]

    # For each zero-skill item, find nearest centroid
    zero_indices = np.where(zero_mask)[0]
    for idx in zero_indices:
        if idx < len(text_embs):
            item_emb = text_embs[idx]
            item_emb = item_emb / max(np.linalg.norm(item_emb), 1e-8)
            sims = centroids @ item_emb
            best_cluster = np.argmax(sims)
            q_matrix[idx, best_cluster] = 1

    fixed = q_matrix[zero_mask].sum(axis=1)
    print(f"  Fixed: {(fixed > 0).sum()}/{n_zero} now have >= 1 skill")
    return q_matrix


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_text_dataloader
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint, verify_splits

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # ── Load expanded data ──
    # Support v2 full dataset via config overrides:
    #   data.response_matrix, data.qmatrix, data.text_embeddings, data.llm_names, data.items
    print("\nLoading expanded dataset...")
    if hasattr(cfg, "data") and hasattr(cfg.data, "response_matrix"):
        R = np.load(data_dir / cfg.data.response_matrix)
        q_matrix = np.load(data_dir / cfg.data.qmatrix).copy()
        with open(data_dir / cfg.data.llm_names) as f:
            llm_names = json.load(f)
        with open(data_dir / cfg.data.items) as f:
            items_data = json.load(f)
        text_emb_path = data_dir / cfg.data.text_embeddings
    else:
        R = np.load(data_dir / "response_matrix_expanded.npy")
        q_matrix = np.load(data_dir / "qmatrix_expanded.npy").copy()
        with open(data_dir / "response_matrix_expanded_llms.json") as f:
            llm_names = json.load(f)
        with open(data_dir / "response_matrix_expanded_items.json") as f:
            items_data = json.load(f)
        text_emb_path = data_dir / "item_text_embeddings_expanded.npz"

    n_llms, n_items = R.shape
    n_skills = q_matrix.shape[1]
    print(f"  Response matrix: {n_llms} LLMs x {n_items} items")
    print(f"  Q-matrix: {q_matrix.shape}")
    print(f"  Evaluating on N_test test items, N_train train items (see split below)")

    # ── Fix zero-skill items ──
    if not text_emb_path.exists():
        print("\n  Encoding item texts with SBERT...")
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-mpnet-base-v2")
        texts = [it.get("question_preview", "") for it in items_data]
        text_embeddings = sbert.encode(texts, show_progress_bar=True,
                                       normalize_embeddings=True, batch_size=256)
        np.savez(text_emb_path, embeddings=text_embeddings)
        print(f"  Saved: {text_emb_path} ({text_embeddings.shape})")
    else:
        text_embeddings = np.load(text_emb_path)["embeddings"]
        print(f"  Item text embeddings: {text_embeddings.shape}")

    # Only fix if there are zero-skill items
    if (q_matrix.sum(axis=1) == 0).sum() > 0:
        skill_emb_name = cfg.data.skill_embeddings if (hasattr(cfg, "data") and hasattr(cfg.data, "skill_embeddings")) else "skill_embeddings_expanded.npz"
        q_matrix = fix_zero_skill_items(
            q_matrix, data_dir / skill_emb_name,
            items_data, text_emb_path,
        )
    else:
        print(f"  No zero-skill items to fix")
    text_dim = text_embeddings.shape[1]

    # ── Optional LLM subset ──
    if hasattr(cfg, "subset") and hasattr(cfg.subset, "n_llms"):
        n_sub = cfg.subset.n_llms
        rng = np.random.RandomState(42)
        llm_indices = rng.choice(n_llms, size=n_sub, replace=False)
        R = R[llm_indices]
        llm_names = [llm_names[i] for i in llm_indices]
        n_llms = n_sub
        print(f"\n  Using {n_llms}-LLM subset")

    # ── Build triplets ──
    print("\nBuilding triplets...")
    student_ids = np.repeat(np.arange(n_llms), n_items)
    item_ids = np.tile(np.arange(n_items), n_llms)
    scores = R.ravel().astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])
    print(f"  Total triplets: {len(triplets):,}")

    # ── Protocol B split ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  Train items: {len(train_items)}, Test items: {len(test_items)}")

    train_mask = np.isin(triplets[:, 1].astype(int), train_items)
    train_triplets = triplets[train_mask]
    test_triplets = triplets[~train_mask]

    tv_idx = np.arange(len(train_triplets))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)
    print(f"  Train triplets: {len(train_triplets[tr_idx]):,}, Val: {len(train_triplets[va_idx]):,}, Test: {len(test_triplets):,}")

    bs = cfg.model.batch_size
    train_loader = make_text_dataloader(
        train_triplets[tr_idx], text_embeddings, q_matrix, bs, shuffle=True
    )
    val_loader = make_text_dataloader(
        train_triplets[va_idx], text_embeddings, q_matrix, bs, shuffle=False
    )
    test_loader = make_text_dataloader(
        test_triplets, text_embeddings, q_matrix, bs, shuffle=False
    )

    # ── Train ──
    print(f"\nTraining TextConditionedNet ({n_skills} skills, {n_llms} LLMs, {text_dim}d text)...")
    batches = len(train_triplets[tr_idx]) // bs
    print(f"  {batches:,} batches/epoch, {cfg.model.epochs} epochs")

    net = TextConditionedNet(n_skills, n_llms, text_dim)
    net = train_text_model(
        net, train_loader, val_loader,
        epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
    )

    # ── Evaluate ──
    print("\nEvaluating on test items...")
    auc, acc, rmse = eval_text_model(net, test_loader, device)
    print(f"  Test AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}")

    # ── Routing ──
    print("\nRouting evaluation on test items...")
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    accs_at = {1: 0, 3: 0, 5: 0, 10: 0}
    total = len(test_items)

    for item_idx in test_items:
        gt = R[:, int(item_idx)]
        if gt.sum() == 0:
            continue

        emb = torch.tensor(text_embeddings[int(item_idx)], dtype=torch.float32, device=device)
        emb_batch = emb.unsqueeze(0).expand(n_llms, -1)
        q_row = torch.tensor(q_matrix[int(item_idx)], dtype=torch.float32, device=device)
        q_batch = q_row.unsqueeze(0).expand(n_llms, -1)

        with torch.no_grad():
            preds = net(all_llm_ids, emb_batch, q_batch).cpu().numpy()

        ranking = np.argsort(-preds)
        for k in accs_at:
            if gt[ranking[:k]].sum() > 0:
                accs_at[k] += 1

    routing = {f"acc@{k}": accs_at[k] / total for k in accs_at}
    print(f"  Routing: " + ", ".join(f"@{k}={routing[f'acc@{k}']:.4f}" for k in accs_at))

    # Strongest model baseline
    train_acc = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    strongest_correct = sum(R[strongest_idx, int(i)] for i in test_items)
    strongest_acc1 = strongest_correct / total
    print(f"  Strongest model: {llm_names[strongest_idx].split('__')[-1] if '__' in llm_names[strongest_idx] else llm_names[strongest_idx]}")
    print(f"  Strongest Acc@1: {strongest_acc1:.4f}")

    # ── Save checkpoint ──
    ckpt_dir = Path("cdm_exploration/checkpoints/expanded")
    save_checkpoint(
        net, ckpt_dir / "text_conditioned_protocolB.pt",
        config={"K": n_skills, "n_llms": n_llms, "n_items": n_items,
                "epochs": cfg.model.epochs, "lr": cfg.model.lr, "text_dim": text_dim},
        train_items=train_items, test_items=test_items,
        val_auc=auc, epoch=cfg.model.epochs,
    )

    # ── Verify and log ──
    verified = verify_splits(train_items, test_items, label="train_expanded")
    log_experiment(
        name="train_expanded",
        config={"K": n_skills, "n_llms": n_llms, "n_items": n_items,
                "epochs": cfg.model.epochs, "lr": cfg.model.lr, "device": device},
        results={"test_auc": float(auc), "test_acc": float(acc),
                 "test_rmse": float(rmse), **routing,
                 "strongest_acc1": float(strongest_acc1)},
        split_info={"n_train_items": len(train_items),
                    "n_test_items": len(test_items), "n_llms": n_llms},
        verified=verified,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
