"""
Cluster extracted skill labels into a taxonomy using embeddings.

Takes the raw LLM-extracted skill strings, embeds them with
sentence-transformers, reduces dimensionality with UMAP, clusters
with HDBSCAN, and produces a binary Q-matrix for CDM fitting.

Usage:
    python 03_cluster_skills.py --input ../data/cdm_ready/skills_extracted.json
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score

OUTPUT_DIR = (Path(__file__).resolve().parent.parent / "data" / "cdm_ready")
FIG_DIR = (Path(__file__).resolve().parent.parent / "figures")


def load_skills(input_path):
    """Load extracted skills JSON. Expected format: list of dicts with 'skills' key."""
    with open(input_path) as f:
        data = json.load(f)

    all_skills = []
    for item in data:
        skills = item.get("skills", [])
        if isinstance(skills, str):
            skills = eval(skills)
        all_skills.append([s.lower().strip() for s in skills])

    unique = sorted(set(s for skills in all_skills for s in skills))
    counts = Counter(s for skills in all_skills for s in skills)
    print(f"Loaded {len(data)} items, {sum(len(s) for s in all_skills)} mentions, {len(unique)} unique skills")
    print(f"Skills/item: mean={np.mean([len(s) for s in all_skills]):.1f}, "
          f"median={np.median([len(s) for s in all_skills]):.0f}")
    return data, all_skills, unique, counts


def embed_skills(unique_skills, model_name="all-mpnet-base-v2"):
    """Embed skill strings with a sentence-transformer model."""
    print(f"Embedding {len(unique_skills)} skills with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(unique_skills, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def cluster_skills(embeddings, unique_skills, min_cluster_range=(5, 10, 15, 20)):
    """
    UMAP + HDBSCAN clustering with silhouette-based min_cluster_size selection.
    Returns cluster labels, the UMAP 2D projection, and the reducer.
    """
    # UMAP for clustering (higher dims for accuracy)
    reducer_cluster = umap.UMAP(n_components=5, metric="cosine", random_state=42)
    reduced = reducer_cluster.fit_transform(embeddings)

    # UMAP for visualization (2D)
    reducer_viz = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords_2d = reducer_viz.fit_transform(embeddings)

    # Sweep min_cluster_size, pick best silhouette
    best_score, best_labels, best_mcs = -1, None, None
    print("\nHDBSCAN sweep:")

    for mcs in min_cluster_range:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean", min_samples=2)
        labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()

        if n_clusters < 2:
            print(f"  min_cluster_size={mcs}: {n_clusters} clusters (skipped)")
            continue

        # Silhouette on non-noise points
        mask = labels != -1
        score = silhouette_score(reduced[mask], labels[mask])
        print(f"  min_cluster_size={mcs}: {n_clusters} clusters, {n_noise} noise, silhouette={score:.3f}")

        if score > best_score:
            best_score, best_labels, best_mcs = score, labels, mcs

    print(f"\nSelected min_cluster_size={best_mcs} (silhouette={best_score:.3f})")

    # Assign noise points to nearest cluster
    if (best_labels == -1).any():
        from sklearn.neighbors import NearestCentroid
        mask = best_labels != -1
        clf = NearestCentroid()
        clf.fit(reduced[mask], best_labels[mask])
        noise_idx = np.where(best_labels == -1)[0]
        best_labels[noise_idx] = clf.predict(reduced[noise_idx])
        print(f"Reassigned {len(noise_idx)} noise points to nearest cluster")

    return best_labels, coords_2d


def label_clusters(unique_skills, labels, counts):
    """Generate a representative label for each cluster based on most frequent members."""
    cluster_ids = sorted(set(labels))
    cluster_labels = {}

    for cid in cluster_ids:
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        # Pick the most frequent skill as the cluster name
        member_counts = [(s, counts.get(s, 0)) for s in members]
        member_counts.sort(key=lambda x: -x[1])
        top = member_counts[0][0]

        # Clean up: title case, truncate if too long
        label = top.replace("_", " ").title()
        if len(label) > 50:
            label = label[:47] + "..."
        cluster_labels[cid] = label

    return cluster_labels


def build_q_matrix(data, all_skills, unique_skills, labels, cluster_labels):
    """
    Map each item's skills to cluster IDs, produce binary Q-matrix.
    Returns DataFrame of shape (n_items, n_clusters).
    """
    skill_to_cluster = {unique_skills[i]: labels[i] for i in range(len(unique_skills))}
    cluster_ids = sorted(set(labels))
    cluster_names = [cluster_labels[c] for c in cluster_ids]

    rows = []
    for item_skills in all_skills:
        row = np.zeros(len(cluster_ids), dtype=int)
        for skill in item_skills:
            cid = skill_to_cluster.get(skill)
            if cid is not None:
                idx = cluster_ids.index(cid)
                row[idx] = 1
        rows.append(row)

    q_matrix = pd.DataFrame(rows, columns=cluster_names)

    # Add item metadata if available
    if data and "item_idx" in data[0]:
        q_matrix.insert(0, "item_idx", [d["item_idx"] for d in data])
    if data and "source" in data[0]:
        q_matrix.insert(1, "source", [d["source"] for d in data])

    return q_matrix


def plot_skill_space(coords_2d, labels, cluster_labels, unique_skills, output_path):
    """UMAP 2D scatter plot of the skill embedding space, colored by cluster."""
    fig, ax = plt.subplots(figsize=(12, 8))

    cluster_ids = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(cluster_ids))

    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[cmap(i)], label=cluster_labels[cid], s=30, alpha=0.7,
        )

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Skill Embedding Space (UMAP 2D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved skill space plot: {output_path}")


def plot_cluster_sizes(q_matrix, cluster_names, output_path):
    """Bar chart of how many items involve each skill cluster."""
    # Sum columns (excluding metadata columns)
    skill_cols = [c for c in q_matrix.columns if c not in ("item_idx", "source")]
    counts = q_matrix[skill_cols].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.3)))
    ax.barh(counts.index, counts.values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(counts))))
    ax.set_xlabel("Number of Items")
    ax.set_title("Items per Skill Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cluster size plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster extracted skills into taxonomy")
    parser.add_argument("--input", required=True, help="Path to extracted skills JSON")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Sentence-transformer model")
    parser.add_argument("--min-cluster-sizes", default="3,5,8,10,15,20",
                        help="Comma-separated min_cluster_size values to sweep")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    data, all_skills, unique_skills, counts = load_skills(args.input)

    # Embed
    embeddings = embed_skills(unique_skills, args.model)

    # Cluster
    mcs_range = [int(x) for x in args.min_cluster_sizes.split(",")]
    labels, coords_2d = cluster_skills(embeddings, unique_skills, mcs_range)

    # Label clusters
    cluster_labels = label_clusters(unique_skills, labels, counts)
    n_clusters = len(set(labels))
    print(f"\nFinal taxonomy ({n_clusters} clusters):")
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        print(f"  [{cid}] {cluster_labels[cid]} ({len(members)} skills)")

    # Q-matrix
    q_matrix = build_q_matrix(data, all_skills, unique_skills, labels, cluster_labels)
    q_matrix.to_csv(OUTPUT_DIR / "q_matrix.csv", index=False)
    print(f"\nQ-matrix: {q_matrix.shape[0]} items x {n_clusters} skills")

    skills_per_item = q_matrix.drop(columns=["item_idx", "source"], errors="ignore").sum(axis=1)
    print(f"Skills/item: mean={skills_per_item.mean():.1f}, min={skills_per_item.min()}, max={skills_per_item.max()}")

    # Save taxonomy mapping
    taxonomy = {}
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        taxonomy[cluster_labels[cid]] = members
    with open(OUTPUT_DIR / "skill_taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    # Save skill-to-cluster mapping
    skill_map = {unique_skills[i]: cluster_labels[labels[i]] for i in range(len(unique_skills))}
    with open(OUTPUT_DIR / "skill_to_cluster.json", "w") as f:
        json.dump(skill_map, f, indent=2)

    # Save embeddings for reproducibility
    np.savez(
        OUTPUT_DIR / "skill_embeddings.npz",
        embeddings=embeddings,
        skills=unique_skills,
        labels=labels,
        coords_2d=coords_2d,
    )

    # Visualize
    plot_skill_space(coords_2d, labels, cluster_labels, unique_skills, FIG_DIR / "skill_space_umap.png")
    plot_cluster_sizes(q_matrix, list(cluster_labels.values()), FIG_DIR / "cluster_sizes.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
