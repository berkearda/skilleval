"""Cluster expanded skills and build Q-matrix for 7,039-item dataset.

Usage:
    python tools/cluster_expanded.py
    python tools/cluster_expanded.py --k 50
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=50, help="Number of clusters")
    args = parser.parse_args()

    K = args.k
    data_dir = Path("cdm_exploration/data/cdm_ready")

    # ── Load skills ──
    print("Loading expanded skills...")
    with open(data_dir / "skills_extracted_expanded.json") as f:
        items_data = json.load(f)
    print(f"  Items: {len(items_data)}")

    all_skills = []
    for item in items_data:
        skills = item.get("skills", [])
        if isinstance(skills, str):
            import ast
            skills = ast.literal_eval(skills)
        all_skills.append([s.lower().strip() for s in skills])

    unique_skills = sorted(set(s for skills in all_skills for s in skills))
    counts = Counter(s for skills in all_skills for s in skills)
    print(f"  Total mentions: {sum(len(s) for s in all_skills)}")
    print(f"  Unique skills: {len(unique_skills)}")

    # ── Embed skills ──
    print(f"\nEmbedding {len(unique_skills)} skills with all-mpnet-base-v2...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(unique_skills, show_progress_bar=True,
                              normalize_embeddings=True, batch_size=256)
    print(f"  Embeddings: {embeddings.shape}")

    # ── Cluster with HAC ──
    print(f"\nClustering with HAC K={K} (cosine, average linkage)...")
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    hac = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average")
    labels = hac.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  Clusters: {len(set(labels))}, Silhouette: {sil:.4f}")

    # ── Label clusters ──
    cluster_labels = {}
    cluster_members = {}
    for cid in sorted(set(labels)):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        member_counts = [(s, counts.get(s, 0)) for s in members]
        member_counts.sort(key=lambda x: -x[1])
        top = member_counts[0][0]
        label = top.replace("_", " ").title()
        if len(label) > 50:
            label = label[:47] + "..."
        cluster_labels[cid] = label
        cluster_members[cid] = member_counts

    # ── Build Q-matrix ──
    print("\nBuilding Q-matrix...")
    skill_to_cluster = {unique_skills[i]: labels[i] for i in range(len(unique_skills))}
    cluster_ids = sorted(set(labels))

    q_rows = []
    for item_skills in all_skills:
        row = np.zeros(K, dtype=int)
        for skill in item_skills:
            cid = skill_to_cluster.get(skill)
            if cid is not None:
                idx = cluster_ids.index(cid)
                row[idx] = 1
        q_rows.append(row)

    q_matrix = np.array(q_rows)
    print(f"  Q-matrix: {q_matrix.shape}")

    # ── Stats ──
    row_sums = q_matrix.sum(axis=1)
    col_sums = q_matrix.sum(axis=0)
    density = q_matrix.sum() / q_matrix.size

    print(f"\n  Mean skills per item: {row_sums.mean():.2f}")
    print(f"  Mean items per skill: {col_sums.mean():.1f}")
    print(f"  Density: {density:.4f}")
    print(f"  Items with 0 skills: {(row_sums == 0).sum()}")

    # Top/bottom clusters
    print(f"\n  Top 5 largest clusters:")
    sizes = [(cid, len(cluster_members[cid])) for cid in cluster_ids]
    sizes.sort(key=lambda x: -x[1])
    for cid, size in sizes[:5]:
        print(f"    [{cid}] {cluster_labels[cid]} ({size} skills, {col_sums[cluster_ids.index(cid)]} items)")

    print(f"\n  Top 5 smallest clusters:")
    for cid, size in sizes[-5:]:
        print(f"    [{cid}] {cluster_labels[cid]} ({size} skills, {col_sums[cluster_ids.index(cid)]} items)")

    # ── Benchmark distribution across clusters ──
    print("\n  Benchmark distribution across clusters:")
    math_mask = np.array([it.get("benchmark") == "MATH" for it in items_data])
    bbh_mask = np.array([it.get("benchmark") == "BBH" for it in items_data])

    math_q = q_matrix[math_mask]
    bbh_q = q_matrix[bbh_mask]

    math_active = (math_q.sum(axis=0) > 0).sum()
    bbh_active = (bbh_q.sum(axis=0) > 0).sum()
    both_active = ((math_q.sum(axis=0) > 0) & (bbh_q.sum(axis=0) > 0)).sum()
    math_only = math_active - both_active
    bbh_only = bbh_active - both_active

    print(f"    Clusters used by MATH: {math_active}/{K}")
    print(f"    Clusters used by BBH:  {bbh_active}/{K}")
    print(f"    Shared clusters:       {both_active}/{K}")
    print(f"    MATH-only clusters:    {math_only}")
    print(f"    BBH-only clusters:     {bbh_only}")

    # ── Save ──
    print("\nSaving...")
    np.save(data_dir / "qmatrix_expanded.npy", q_matrix)
    print(f"  qmatrix_expanded.npy ({q_matrix.shape})")

    labels_out = {int(k): v for k, v in cluster_labels.items()}
    with open(data_dir / "cluster_labels_expanded.json", "w") as f:
        json.dump(labels_out, f, indent=2)
    print(f"  cluster_labels_expanded.json ({len(labels_out)} clusters)")

    s2c_out = {skill: cluster_labels[int(labels[i])]
               for i, skill in enumerate(unique_skills)}
    with open(data_dir / "skill_clusters_expanded.json", "w") as f:
        json.dump(s2c_out, f, indent=2)
    print(f"  skill_clusters_expanded.json ({len(s2c_out)} skills)")

    # Save embeddings for future use
    np.savez(
        data_dir / "skill_embeddings_expanded.npz",
        embeddings=embeddings,
        skills=np.array(unique_skills, dtype=object),
        labels=labels,
    )
    print(f"  skill_embeddings_expanded.npz")

    # ── Verify and log ──
    from cdmeval.utils.experiment import log_experiment

    log_experiment(
        name="cluster_expanded",
        config={"K": K, "metric": "cosine", "linkage": "average",
                "sbert_model": "all-mpnet-base-v2", "n_unique_skills": len(unique_skills)},
        results={"silhouette": float(sil),
                 "q_matrix_shape": list(q_matrix.shape),
                 "mean_skills_per_item": float(row_sums.mean()),
                 "mean_items_per_skill": float(col_sums.mean()),
                 "density": float(density),
                 "shared_clusters": int(both_active),
                 "math_only_clusters": int(math_only),
                 "bbh_only_clusters": int(bbh_only)},
        split_info={"n_items": len(items_data),
                    "n_math": int(math_mask.sum()),
                    "n_bbh": int(bbh_mask.sum())},
        verified=True,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
