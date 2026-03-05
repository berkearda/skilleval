"""Skill embedding, clustering, and Q-matrix construction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_skills(
    input_path: str | Path,
) -> tuple[list, list[list[str]], list[str], Counter]:
    """Load extracted skills JSON.

    Returns:
        ``(data, all_skills, unique_skills, counts)``
    """
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

    print(
        f"Loaded {len(data)} items, "
        f"{sum(len(s) for s in all_skills)} mentions, "
        f"{len(unique)} unique skills"
    )
    return data, all_skills, unique, counts


def embed_skills(
    unique_skills: list[str], model_name: str = "all-mpnet-base-v2"
) -> np.ndarray:
    """Embed skill strings with a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    print(f"Embedding {len(unique_skills)} skills with {model_name}...")
    model = SentenceTransformer(model_name)
    return model.encode(unique_skills, show_progress_bar=True, normalize_embeddings=True)


# ── Clustering methods ──────────────────────────────────────────────


def cluster_skills_hdbscan(
    embeddings: np.ndarray,
    min_cluster_range: Sequence[int] = (5, 10, 15, 20),
) -> tuple[np.ndarray, np.ndarray]:
    """UMAP + HDBSCAN clustering with silhouette-based selection.

    Returns:
        ``(labels, coords_2d)``
    """
    import hdbscan as hdb
    import umap
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestCentroid

    reducer_cluster = umap.UMAP(n_components=5, metric="cosine", random_state=42)
    reduced = reducer_cluster.fit_transform(embeddings)

    reducer_viz = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords_2d = reducer_viz.fit_transform(embeddings)

    best_score, best_labels, best_mcs = -1, None, None
    print("\nHDBSCAN sweep:")

    for mcs in min_cluster_range:
        clusterer = hdb.HDBSCAN(min_cluster_size=mcs, metric="euclidean", min_samples=2)
        labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()

        if n_clusters < 2:
            print(f"  min_cluster_size={mcs}: {n_clusters} clusters (skipped)")
            continue

        mask = labels != -1
        score = silhouette_score(reduced[mask], labels[mask])
        print(
            f"  min_cluster_size={mcs}: {n_clusters} clusters, "
            f"{n_noise} noise, silhouette={score:.3f}"
        )

        if score > best_score:
            best_score, best_labels, best_mcs = score, labels.copy(), mcs

    print(f"Selected min_cluster_size={best_mcs} (silhouette={best_score:.3f})")

    if (best_labels == -1).any():
        mask = best_labels != -1
        clf = NearestCentroid()
        clf.fit(reduced[mask], best_labels[mask])
        noise_idx = np.where(best_labels == -1)[0]
        best_labels[noise_idx] = clf.predict(reduced[noise_idx])
        print(f"Reassigned {len(noise_idx)} noise points")

    return best_labels, coords_2d


def cluster_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    """K-Means on raw embeddings."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


def cluster_hac(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative clustering with cosine distance + average linkage."""
    from sklearn.cluster import AgglomerativeClustering

    hac = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return hac.fit_predict(embeddings)


# ── Cluster labelling and Q-matrix ──────────────────────────────────


def label_clusters(
    unique_skills: list[str],
    labels: np.ndarray,
    counts: Counter,
) -> dict[int, str]:
    """Generate a representative label for each cluster from the most frequent member."""
    cluster_ids = sorted(set(labels))
    cluster_labels = {}
    for cid in cluster_ids:
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        member_counts = [(s, counts.get(s, 0)) for s in members]
        member_counts.sort(key=lambda x: -x[1])
        top = member_counts[0][0]
        label = top.replace("_", " ").title()
        if len(label) > 50:
            label = label[:47] + "..."
        cluster_labels[cid] = label
    return cluster_labels


def build_q_matrix(
    data: list[dict],
    all_skills: list[list[str]],
    unique_skills: list[str],
    labels: np.ndarray,
    cluster_labels: dict[int, str],
) -> pd.DataFrame:
    """Map each item's skills to cluster IDs, produce binary Q-matrix DataFrame."""
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

    q_df = pd.DataFrame(rows, columns=cluster_names)

    if data and "item_idx" in data[0]:
        q_df.insert(0, "item_idx", [d["item_idx"] for d in data])
    if data and "source" in data[0]:
        q_df.insert(1, "source", [d["source"] for d in data])

    return q_df


def build_q_from_labels(
    labels: np.ndarray,
    unique_skills: list[str],
    all_skills: list[list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Build a numeric Q-matrix from cluster labels (no metadata columns)."""
    skill_to_cluster = {unique_skills[i]: labels[i] for i in range(len(unique_skills))}
    cluster_ids = sorted(set(labels))
    col_names = [f"skill_{cid}" for cid in cluster_ids]

    rows = []
    for item_skills in all_skills:
        row = np.zeros(len(cluster_ids), dtype=int)
        for skill in item_skills:
            cid = skill_to_cluster.get(skill)
            if cid is not None:
                idx = cluster_ids.index(cid)
                row[idx] = 1
        rows.append(row)

    return np.array(rows), col_names
