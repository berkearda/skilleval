"""Cluster sweep analysis: evaluate NCDM performance across different K values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .clustering import build_q_from_labels, cluster_hac, label_clusters, load_skills


def _compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    q_matrix: np.ndarray,
) -> dict:
    """Compute intrinsic clustering and Q-matrix interpretability metrics."""
    n_clusters = len(set(labels))
    cluster_sizes = np.bincount(labels)
    singletons = int((cluster_sizes == 1).sum())

    sil = float(silhouette_score(embeddings, labels, metric="cosine"))

    # Mean intra-cluster cosine similarity
    intra_sims = []
    for cid in range(n_clusters):
        mask = labels == cid
        if mask.sum() < 2:
            continue
        cluster_embs = embeddings[mask]
        # Cosine similarity matrix (embeddings already L2-normalised)
        sim_matrix = cluster_embs @ cluster_embs.T
        n = sim_matrix.shape[0]
        # Mean of upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(n, k=1)
        intra_sims.append(float(sim_matrix[triu_idx].mean()))
    mean_intra_sim = float(np.mean(intra_sims)) if intra_sims else 0.0

    # Q-matrix interpretability
    row_sums = q_matrix.sum(axis=1)
    col_sums = q_matrix.sum(axis=0)
    mean_skills_per_item = float(row_sums.mean())
    # Fraction of items that use each skill (averaged across skills)
    skill_coverage = float((col_sums / q_matrix.shape[0]).mean())

    return {
        "silhouette": sil,
        "mean_intra_cosine_sim": mean_intra_sim,
        "n_singletons": singletons,
        "mean_cluster_size": float(cluster_sizes.mean()),
        "mean_skills_per_item": mean_skills_per_item,
        "skill_coverage": skill_coverage,
    }


def _compute_routing_accuracy(
    net: torch.nn.Module,
    text_embeddings: np.ndarray,
    q_matrix: np.ndarray,
    response_vals: np.ndarray,
    test_items: np.ndarray,
    n_llms: int,
    device: str,
) -> dict:
    """Compute routing acc@1 and acc@5 on held-out items."""
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    accs_at = {1: 0, 5: 0}
    total = 0

    for item_idx in test_items:
        gt = response_vals[:, int(item_idx)]
        if gt.sum() == 0:
            continue

        emb = torch.tensor(
            text_embeddings[int(item_idx)], dtype=torch.float32, device=device
        )
        emb_batch = emb.unsqueeze(0).expand(n_llms, -1)
        q_row = torch.tensor(
            q_matrix[int(item_idx)], dtype=torch.float32, device=device
        )
        q_batch = q_row.unsqueeze(0).expand(n_llms, -1)

        with torch.no_grad():
            preds = net(all_llm_ids, emb_batch, q_batch).cpu().numpy()

        ranking = np.argsort(-preds)
        total += 1
        for k in accs_at:
            if gt[ranking[:k]].sum() > 0:
                accs_at[k] += 1

    if total == 0:
        return {"routing_acc1": 0.0, "routing_acc5": 0.0}

    return {
        "routing_acc1": accs_at[1] / total,
        "routing_acc5": accs_at[5] / total,
    }


def run_cluster_sweep(
    k_values: list[int],
    skill_embeddings: np.ndarray,
    unique_skills: list[str],
    all_skills: list[list[str]],
    skill_counts,
    response_df: pd.DataFrame,
    text_embeddings: np.ndarray,
    n_llms: int,
    n_items: int,
    llm_names: list[str],
    epochs: int = 15,
    lr: float = 0.002,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full cluster sweep, training ID and text-conditioned NCDMs per K.

    Returns a DataFrame with one row per K value and columns for all metrics.
    """
    from cdmeval.data.dataloader import make_dataloader, make_text_dataloader
    from cdmeval.data.response_matrix import build_triplets
    from cdmeval.evaluation.metrics import eval_id_model, eval_text_model
    from cdmeval.evaluation.training import train_id_model, train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import seed_everything

    triplets = build_triplets(response_df)
    response_vals = response_df.values
    text_dim = text_embeddings.shape[1]

    results = []

    for ki, K in enumerate(k_values):
        print(f"\n{'=' * 70}")
        print(f"  K = {K}  ({ki + 1}/{len(k_values)})")
        print(f"{'=' * 70}")

        seed_everything(seed)

        # ── Cluster ──
        labels = cluster_hac(skill_embeddings, K)
        q_matrix, col_names = build_q_from_labels(labels, unique_skills, all_skills)
        n_skills = q_matrix.shape[1]
        print(f"  Q-matrix: {n_items} items x {n_skills} skills")

        # Clustering metrics
        clust_metrics = _compute_clustering_metrics(
            skill_embeddings, labels, q_matrix
        )
        print(
            f"  Silhouette={clust_metrics['silhouette']:.4f}, "
            f"intra-sim={clust_metrics['mean_intra_cosine_sim']:.4f}, "
            f"singletons={clust_metrics['n_singletons']}, "
            f"skills/item={clust_metrics['mean_skills_per_item']:.2f}"
        )

        # ── Protocol A: ID-based NCDM ──
        print(f"\n  --- Protocol A: ID-based NCDM ---")
        all_idx = np.arange(len(triplets))
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=0.2, random_state=seed
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.1 / 0.8, random_state=seed
        )

        train_loader = make_dataloader(
            triplets[train_idx], q_matrix, batch_size, shuffle=True
        )
        val_loader = make_dataloader(
            triplets[val_idx], q_matrix, batch_size, shuffle=False
        )
        test_loader = make_dataloader(
            triplets[test_idx], q_matrix, batch_size, shuffle=False
        )

        model = train_id_model(
            train_loader, val_loader, n_skills, n_items, n_llms,
            epochs=epochs, lr=lr, device=device,
        )
        auc_a, acc_a, rmse_a = eval_id_model(model, test_loader, device)
        print(f"  Protocol A: AUC={auc_a:.4f}, Acc={acc_a:.4f}, RMSE={rmse_a:.4f}")

        # ── Protocol B: Text-conditioned NCDM (cold-start) ──
        print(f"\n  --- Protocol B: Text-conditioned NCDM (cold-start) ---")
        seed_everything(seed)

        all_items = np.arange(n_items)
        train_items, test_items = train_test_split(
            all_items, test_size=0.2, random_state=seed
        )
        train_val_trip = triplets[
            np.isin(triplets[:, 1].astype(int), train_items)
        ]
        test_trip = triplets[
            np.isin(triplets[:, 1].astype(int), test_items)
        ]
        tv_idx = np.arange(len(train_val_trip))
        tr_idx, vl_idx = train_test_split(
            tv_idx, test_size=0.1, random_state=seed
        )

        train_loader_b = make_text_dataloader(
            train_val_trip[tr_idx], text_embeddings, q_matrix, batch_size, shuffle=True
        )
        val_loader_b = make_text_dataloader(
            train_val_trip[vl_idx], text_embeddings, q_matrix, batch_size, shuffle=False
        )
        test_loader_b = make_text_dataloader(
            test_trip, text_embeddings, q_matrix, batch_size, shuffle=False
        )

        net = TextConditionedNet(n_skills, n_llms, text_dim)
        net = train_text_model(
            net, train_loader_b, val_loader_b,
            epochs=epochs, lr=lr, device=device,
        )
        auc_b, acc_b, rmse_b = eval_text_model(net, test_loader_b, device)
        print(f"  Protocol B: AUC={auc_b:.4f}, Acc={acc_b:.4f}, RMSE={rmse_b:.4f}")

        # Routing accuracy
        routing = _compute_routing_accuracy(
            net, text_embeddings, q_matrix, response_vals,
            test_items, n_llms, device,
        )
        print(f"  Routing: Acc@1={routing['routing_acc1']:.4f}, Acc@5={routing['routing_acc5']:.4f}")

        row = {
            "K": K,
            "n_skills": n_skills,
            # Protocol A (ID-based)
            "proto_a_auc": auc_a,
            "proto_a_acc": acc_a,
            "proto_a_rmse": rmse_a,
            # Protocol B (text-conditioned, cold-start)
            "proto_b_auc": auc_b,
            "proto_b_acc": acc_b,
            "proto_b_rmse": rmse_b,
            # Routing
            **routing,
            # Clustering metrics
            **clust_metrics,
        }
        results.append(row)
        print(f"\n  [K={K}] DONE.")

    return pd.DataFrame(results)
