"""
Compare clustering methods (HDBSCAN vs K-Means vs HAC) for skill taxonomy.

Evaluates both intrinsic clustering quality (silhouette) and downstream
NCDM performance (AUC, accuracy, RMSE) across multiple cluster counts.

Usage:
    python 07_compare_clustering.py
    python 07_compare_clustering.py --device cpu --epochs 10
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from collections import Counter

import umap
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from EduCDM import NCDM

from cdmeval.utils.device import resolve_device
from cdmeval.data.response_matrix import load_response_matrix, build_triplets
from cdmeval.data.dataloader import make_dataloader
from cdmeval.evaluation.metrics import eval_id_model
from cdmeval.evaluation.training import train_id_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "report"

# ---------- Clustering methods ----------

def cluster_kmeans(embeddings, k):
    """K-Means on raw 768-dim embeddings."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


def cluster_hac(embeddings, k):
    """Agglomerative clustering with cosine distance + average linkage."""
    hac = AgglomerativeClustering(
        n_clusters=k, metric="cosine", linkage="average"
    )
    return hac.fit_predict(embeddings)


def cluster_hdbscan(embeddings, mcs_list=(5, 10, 15, 20)):
    """UMAP 5D → HDBSCAN with min_cluster_size sweep. Noise reassigned."""
    reducer = umap.UMAP(n_components=5, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)

    best_score, best_labels, best_mcs = -1, None, None
    print("\n  HDBSCAN sweep:")

    for mcs in mcs_list:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean", min_samples=2)
        labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()

        if n_clusters < 2:
            print(f"    min_cluster_size={mcs}: {n_clusters} clusters (skipped)")
            continue

        mask = labels != -1
        score = silhouette_score(reduced[mask], labels[mask])
        print(f"    min_cluster_size={mcs}: {n_clusters} clusters, {n_noise} noise, silhouette={score:.3f}")

        if score > best_score:
            best_score, best_labels, best_mcs = score, labels.copy(), mcs

    print(f"  Selected min_cluster_size={best_mcs} (silhouette={best_score:.3f})")

    # Reassign noise to nearest cluster
    if (best_labels == -1).any():
        from sklearn.neighbors import NearestCentroid
        mask = best_labels != -1
        clf = NearestCentroid()
        clf.fit(reduced[mask], best_labels[mask])
        noise_idx = np.where(best_labels == -1)[0]
        best_labels[noise_idx] = clf.predict(reduced[noise_idx])
        print(f"  Reassigned {len(noise_idx)} noise points")

    return best_labels


# ---------- Q-matrix building ----------

def build_q_from_labels(labels, unique_skills, all_skills, data):
    """Build Q-matrix from cluster labels."""
    skill_to_cluster = {unique_skills[i]: labels[i] for i in range(len(unique_skills))}
    cluster_ids = sorted(set(labels))
    n_clusters = len(cluster_ids)

    col_names = [f"skill_{cid}" for cid in cluster_ids]

    rows = []
    for item_skills in all_skills:
        row = np.zeros(n_clusters, dtype=int)
        for skill in item_skills:
            cid = skill_to_cluster.get(skill)
            if cid is not None:
                idx = cluster_ids.index(cid)
                row[idx] = 1
        rows.append(row)

    q_matrix = np.array(rows)
    return q_matrix, col_names


# ---------- NCDM training and evaluation ----------

def train_and_evaluate_ncdm(q_matrix, triplets, train_idx, val_idx, test_idx,
                            n_llms, n_items, epochs=15, lr=0.002,
                            batch_size=64, device="cpu"):
    """Build dataloaders, train NCDM, return test metrics."""
    n_skills = q_matrix.shape[1]

    train_loader = make_dataloader(triplets[train_idx], q_matrix, batch_size, shuffle=True)
    val_loader = make_dataloader(triplets[val_idx], q_matrix, batch_size, shuffle=False)
    test_loader = make_dataloader(triplets[test_idx], q_matrix, batch_size, shuffle=False)

    model = train_id_model(train_loader, val_loader, n_skills, n_items, n_llms,
                           epochs=epochs, lr=lr, device=device)

    test_auc, test_acc, test_rmse = eval_id_model(model, test_loader, device)
    return test_auc, test_acc, test_rmse


# ---------- Main comparison loop ----------

def run_comparison(args):
    """Run all clustering configs and evaluate each."""
    emb_data = np.load(DATA_DIR / "skill_embeddings.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]
    unique_skills = list(emb_data["skills"])
    print(f"Embeddings: {embeddings.shape}")

    with open(DATA_DIR / "skills_extracted.json") as f:
        data = json.load(f)
    all_skills = []
    for item in data:
        skills = item.get("skills", [])
        if isinstance(skills, str):
            skills = eval(skills)
        all_skills.append([s.lower().strip() for s in skills])

    response_df, n_llms, n_items, _ = load_response_matrix(DATA_DIR / "response_matrix.csv")
    print(f"Response matrix: {n_llms} LLMs x {n_items} items")

    triplets = build_triplets(response_df)

    # Fixed train/val/test split
    all_indices = np.arange(len(triplets))
    train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1 / 0.8, random_state=42)
    print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    results = []

    # --- HDBSCAN ---
    print("\n" + "="*60)
    print("HDBSCAN")
    print("="*60)
    labels = cluster_hdbscan(embeddings, mcs_list=[5, 10, 15, 20])
    n_clusters = len(set(labels))
    sil = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  {n_clusters} clusters, silhouette(cosine, full)={sil:.4f}")

    q_matrix, _ = build_q_from_labels(labels, unique_skills, all_skills, data)
    print(f"  Q-matrix: {q_matrix.shape}")
    print(f"  Training NCDM...")
    auc, acc, rmse = train_and_evaluate_ncdm(
        q_matrix, triplets, train_idx, val_idx, test_idx,
        n_llms, n_items, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, device=args.device
    )
    print(f"  Test: AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}")
    results.append({
        "method": "HDBSCAN", "n_clusters": n_clusters,
        "silhouette": sil, "test_auc": auc, "test_accuracy": acc, "test_rmse": rmse
    })

    # --- K-Means and HAC at various K ---
    k_values = [50, 100, 150, 200, 250, 281, 350]

    for k in k_values:
        for method_name, cluster_fn in [("K-Means", cluster_kmeans), ("HAC", cluster_hac)]:
            print("\n" + "="*60)
            print(f"{method_name} (K={k})")
            print("="*60)

            labels = cluster_fn(embeddings, k)
            n_clusters = len(set(labels))
            sil = silhouette_score(embeddings, labels, metric="cosine")
            print(f"  {n_clusters} clusters, silhouette(cosine, full)={sil:.4f}")

            q_matrix, _ = build_q_from_labels(labels, unique_skills, all_skills, data)
            print(f"  Q-matrix: {q_matrix.shape}")
            print(f"  Training NCDM...")
            auc, acc, rmse = train_and_evaluate_ncdm(
                q_matrix, triplets, train_idx, val_idx, test_idx,
                n_llms, n_items, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, device=args.device
            )
            print(f"  Test: AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}")
            results.append({
                "method": method_name, "n_clusters": n_clusters,
                "silhouette": sil, "test_auc": auc, "test_accuracy": acc, "test_rmse": rmse
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / "clustering_comparison.csv", index=False)
    print(f"\nResults saved to {DATA_DIR / 'clustering_comparison.csv'}")
    print(results_df.to_string(index=False))

    return results_df


# ---------- Plotting ----------

def plot_comparison(results_df):
    """2x2 comparison figure: silhouette, AUC, accuracy, RMSE vs K."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics = [
        ("silhouette", "Silhouette Score", axes[0, 0], "(a)"),
        ("test_auc", "Test AUC", axes[0, 1], "(b)"),
        ("test_accuracy", "Test Accuracy", axes[1, 0], "(c)"),
        ("test_rmse", "Test RMSE", axes[1, 1], "(d)"),
    ]

    method_styles = {
        "K-Means": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
        "HAC": {"color": "#ff7f0e", "marker": "^", "linestyle": "--"},
        "HDBSCAN": {"color": "#2ca02c", "marker": "D"},
    }

    for col, ylabel, ax, panel_label in metrics:
        for method in ["K-Means", "HAC"]:
            subset = results_df[results_df["method"] == method].sort_values("n_clusters")
            style = method_styles[method]
            ax.plot(subset["n_clusters"], subset[col],
                    marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], label=method, markersize=6)

        hdb = results_df[results_df["method"] == "HDBSCAN"]
        if not hdb.empty:
            style = method_styles["HDBSCAN"]
            ax.scatter(hdb["n_clusters"].values, hdb[col].values,
                       marker=style["marker"], color=style["color"],
                       s=100, zorder=5, label="HDBSCAN", edgecolors="black", linewidths=0.8)

        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{panel_label} {ylabel} vs. Number of Clusters")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "fig_clustering_comparison.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


# ---------- Entry point ----------

def main():
    parser = argparse.ArgumentParser(description="Compare clustering methods for CDM")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps", help="cpu, cuda, or mps")
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    print(f"Device: {args.device}")

    results_df = run_comparison(args)
    plot_comparison(results_df)


if __name__ == "__main__":
    main()
