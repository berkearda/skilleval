"""Cluster extracted skill labels into a taxonomy using SBERT + UMAP + HDBSCAN.

Usage:
    python tools/cluster_skills.py
    python tools/cluster_skills.py skills.sbert_model=all-MiniLM-L6-v2
"""

from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.skills.clustering import (
        build_q_matrix,
        cluster_skills_hdbscan,
        embed_skills,
        label_clusters,
        load_skills,
    )

    import json

    cdm_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    cdm_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data, all_skills, unique_skills, counts = load_skills(
        cdm_dir / "skills_extracted.json"
    )

    embeddings = embed_skills(unique_skills, cfg.skills.sbert_model)

    mcs_range = list(cfg.skills.min_cluster_sizes)
    labels, coords_2d = cluster_skills_hdbscan(embeddings, mcs_range)

    cluster_labels = label_clusters(unique_skills, labels, counts)
    n_clusters = len(set(labels))
    print(f"\nFinal taxonomy ({n_clusters} clusters):")
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        print(f"  [{cid}] {cluster_labels[cid]} ({len(members)} skills)")

    q_matrix = build_q_matrix(data, all_skills, unique_skills, labels, cluster_labels)
    q_matrix.to_csv(cdm_dir / "q_matrix.csv", index=False)
    print(f"\nQ-matrix: {q_matrix.shape[0]} items x {n_clusters} skills")

    taxonomy = {}
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        taxonomy[cluster_labels[cid]] = members
    with open(cdm_dir / "skill_taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    skill_map = {unique_skills[i]: cluster_labels[labels[i]] for i in range(len(unique_skills))}
    with open(cdm_dir / "skill_to_cluster.json", "w") as f:
        json.dump(skill_map, f, indent=2)

    np.savez(
        cdm_dir / "skill_embeddings.npz",
        embeddings=embeddings, skills=unique_skills,
        labels=labels, coords_2d=coords_2d,
    )

    print(f"All outputs saved to {cdm_dir}/")


if __name__ == "__main__":
    main()
