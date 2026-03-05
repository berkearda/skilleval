"""Compare clustering methods (HDBSCAN vs K-Means vs HAC) for skill taxonomy.

Usage:
    python tools/compare_clustering.py
    python tools/compare_clustering.py device=cpu model.epochs=10
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_dataloader
    from cdmeval.data.response_matrix import build_triplets, load_response_matrix
    from cdmeval.evaluation.metrics import eval_id_model
    from cdmeval.evaluation.training import train_id_model
    from cdmeval.skills.clustering import (
        build_q_from_labels,
        cluster_hac,
        cluster_kmeans,
        cluster_skills_hdbscan,
        load_skills,
    )
    from cdmeval.utils.device import resolve_device

    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # Load skill embeddings
    emb_data = np.load(data_dir / "skill_embeddings.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]
    unique_skills = list(emb_data["skills"])
    print(f"Embeddings: {embeddings.shape}")

    data, all_skills, _, _ = load_skills(data_dir / "skills_extracted.json")

    response_df, n_llms, n_items, _ = load_response_matrix(data_dir / "response_matrix.csv")
    print(f"Response matrix: {n_llms} LLMs x {n_items} items")

    triplets = build_triplets(response_df)
    all_indices = np.arange(len(triplets))
    train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1 / 0.8, random_state=42)
    print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    results = []
    bs = cfg.model.batch_size

    # --- HDBSCAN ---
    print("\n" + "=" * 60 + "\nHDBSCAN\n" + "=" * 60)
    labels, _ = cluster_skills_hdbscan(embeddings)
    n_clusters = len(set(labels))
    sil = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  {n_clusters} clusters, silhouette(cosine)={sil:.4f}")

    q_matrix, _ = build_q_from_labels(labels, unique_skills, all_skills)
    train_loader = make_dataloader(triplets[train_idx], q_matrix, bs, shuffle=True)
    val_loader = make_dataloader(triplets[val_idx], q_matrix, bs, shuffle=False)
    test_loader = make_dataloader(triplets[test_idx], q_matrix, bs, shuffle=False)

    model = train_id_model(
        train_loader, val_loader, q_matrix.shape[1], n_items, n_llms,
        epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
    )
    auc, acc, rmse = eval_id_model(model, test_loader, device)
    print(f"  Test: AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}")
    results.append({
        "method": "HDBSCAN", "n_clusters": n_clusters,
        "silhouette": sil, "test_auc": auc, "test_accuracy": acc, "test_rmse": rmse,
    })

    # --- K-Means and HAC at various K ---
    k_values = [50, 100, 150, 200, 250, 281, 350]

    for k in k_values:
        for method_name, cluster_fn in [("K-Means", cluster_kmeans), ("HAC", cluster_hac)]:
            print(f"\n{'=' * 60}\n{method_name} (K={k})\n{'=' * 60}")
            labels = cluster_fn(embeddings, k)
            n_clusters = len(set(labels))
            sil = silhouette_score(embeddings, labels, metric="cosine")
            print(f"  {n_clusters} clusters, silhouette(cosine)={sil:.4f}")

            q_matrix, _ = build_q_from_labels(labels, unique_skills, all_skills)
            train_loader = make_dataloader(triplets[train_idx], q_matrix, bs, shuffle=True)
            val_loader = make_dataloader(triplets[val_idx], q_matrix, bs, shuffle=False)
            test_loader = make_dataloader(triplets[test_idx], q_matrix, bs, shuffle=False)

            model = train_id_model(
                train_loader, val_loader, q_matrix.shape[1], n_items, n_llms,
                epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
            )
            auc, acc, rmse = eval_id_model(model, test_loader, device)
            print(f"  Test: AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}")
            results.append({
                "method": method_name, "n_clusters": n_clusters,
                "silhouette": sil, "test_auc": auc, "test_accuracy": acc, "test_rmse": rmse,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(data_dir / "clustering_comparison.csv", index=False)
    print(f"\nResults saved to {data_dir / 'clustering_comparison.csv'}")
    print(results_df.to_string(index=False))

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = [
        ("silhouette", "Silhouette Score", axes[0, 0]),
        ("test_auc", "Test AUC", axes[0, 1]),
        ("test_accuracy", "Test Accuracy", axes[1, 0]),
        ("test_rmse", "Test RMSE", axes[1, 1]),
    ]
    styles = {
        "K-Means": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
        "HAC": {"color": "#ff7f0e", "marker": "^", "linestyle": "--"},
        "HDBSCAN": {"color": "#2ca02c", "marker": "D"},
    }
    for col, ylabel, ax in metrics:
        for method in ["K-Means", "HAC"]:
            subset = results_df[results_df["method"] == method].sort_values("n_clusters")
            s = styles[method]
            ax.plot(
                subset["n_clusters"], subset[col],
                marker=s["marker"], linestyle=s["linestyle"],
                color=s["color"], label=method, markersize=6,
            )
        hdb = results_df[results_df["method"] == "HDBSCAN"]
        if not hdb.empty:
            s = styles["HDBSCAN"]
            ax.scatter(
                hdb["n_clusters"].values, hdb[col].values,
                marker=s["marker"], color=s["color"],
                s=100, zorder=5, label="HDBSCAN", edgecolors="black", linewidths=0.8,
            )
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = fig_dir / "fig_clustering_comparison.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
