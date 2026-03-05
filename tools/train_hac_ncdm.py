"""Re-cluster skills with HAC K=50 and retrain NCDM.

Usage:
    python tools/train_hac_ncdm.py
    python tools/train_hac_ncdm.py device=cpu model.epochs=10
"""

import json
from pathlib import Path

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
    from cdmeval.evaluation.metrics import eval_id_model, extract_mastery_profiles
    from cdmeval.evaluation.training import train_id_model
    from cdmeval.skills.clustering import build_q_matrix, cluster_hac, label_clusters, load_skills
    from cdmeval.utils.device import resolve_device

    data_dir = Path(cfg.paths.cdm_ready)
    model_dir = Path(cfg.paths.models)
    model_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    K = cfg.skills.n_clusters
    print(f"Device: {device}")

    # Load skill embeddings
    emb_data = np.load(data_dir / "skill_embeddings.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]
    unique_skills = list(emb_data["skills"])
    print(f"Skill embeddings: {embeddings.shape}")

    data, all_skills, _, counts = load_skills(data_dir / "skills_extracted.json")

    # HAC clustering
    print(f"\nClustering {len(unique_skills)} skills with HAC K={K}...")
    labels = cluster_hac(embeddings, K)
    n_clusters = len(set(labels))
    sil = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  {n_clusters} clusters, silhouette(cosine)={sil:.4f}")

    cluster_labels = label_clusters(unique_skills, labels, counts)
    print(f"\nSkill taxonomy ({n_clusters} clusters):")
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        print(f"  [{cid}] {cluster_labels[cid]} ({len(members)} skills)")

    q_matrix_df = build_q_matrix(data, all_skills, unique_skills, labels, cluster_labels)

    meta_cols = [c for c in ["item_idx", "source"] if c in q_matrix_df.columns]
    skill_cols = [c for c in q_matrix_df.columns if c not in meta_cols]
    q_matrix = q_matrix_df[skill_cols].values
    n_skills = q_matrix.shape[1]
    print(f"\nQ-matrix: {q_matrix_df.shape[0]} items x {n_skills} skills")

    # Save artifacts
    q_matrix_df.to_csv(data_dir / f"q_matrix_hac{K}.csv", index=False)

    taxonomy = {}
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        taxonomy[cluster_labels[cid]] = members
    with open(data_dir / f"skill_taxonomy_hac{K}.json", "w") as f:
        json.dump(taxonomy, f, indent=2)

    s2c = {unique_skills[i]: cluster_labels[labels[i]] for i in range(len(unique_skills))}
    with open(data_dir / f"skill_to_cluster_hac{K}.json", "w") as f:
        json.dump(s2c, f, indent=2)

    # Load response matrix and train
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    print(f"\nResponse matrix: {n_llms} LLMs x {n_items} items")

    triplets = build_triplets(response_df)
    all_indices = np.arange(len(triplets))
    train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1 / 0.8, random_state=42)
    print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    bs = cfg.model.batch_size
    train_loader = make_dataloader(triplets[train_idx], q_matrix, bs, shuffle=True)
    val_loader = make_dataloader(triplets[val_idx], q_matrix, bs, shuffle=False)
    test_loader = make_dataloader(triplets[test_idx], q_matrix, bs, shuffle=False)

    print(f"\nTraining NCDM: {n_skills} skills, {n_items} items, {n_llms} LLMs")
    model = train_id_model(
        train_loader, val_loader, n_skills, n_items, n_llms,
        epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
    )

    test_auc, test_acc, test_rmse = eval_id_model(model, test_loader, device)
    print(f"\nTest results: AUC={test_auc:.4f}, Accuracy={test_acc:.4f}, RMSE={test_rmse:.4f}")

    model.save(str(model_dir / f"ncdm_hac{K}.pt"))

    mastery = extract_mastery_profiles(model, n_llms, n_skills, device)
    mastery_df = pd.DataFrame(mastery, index=llm_names, columns=skill_cols)
    mastery_df.index.name = "llm"
    mastery_df.to_csv(data_dir / f"skill_mastery_profiles_hac{K}.csv")
    print(f"Skill mastery profiles saved ({n_llms} LLMs x {n_skills} skills)")

    metrics = {
        "method": "HAC", "n_clusters": int(n_clusters), "silhouette": float(sil),
        "test_auc": float(test_auc), "test_accuracy": float(test_acc),
        "test_rmse": float(test_rmse), "n_llms": n_llms, "n_items": n_items,
        "n_skills": n_skills, "epochs": cfg.model.epochs, "lr": cfg.model.lr,
    }
    with open(data_dir / f"ncdm_metrics_hac{K}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: ncdm_metrics_hac{K}.json")


if __name__ == "__main__":
    main()
