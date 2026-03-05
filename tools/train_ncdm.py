"""Train ID-based NCDM and extract skill mastery profiles.

Usage:
    python tools/train_ncdm.py
    python tools/train_ncdm.py device=cpu model.epochs=1
"""

from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from sklearn.model_selection import train_test_split

    from cdmeval.data.dataloader import make_dataloader
    from cdmeval.data.response_matrix import build_triplets, load_q_matrix, load_response_matrix
    from cdmeval.evaluation.metrics import eval_id_model, extract_mastery_profiles
    from cdmeval.evaluation.training import train_id_model
    from cdmeval.utils.device import resolve_device

    data_dir = Path(cfg.paths.cdm_ready)
    model_dir = Path(cfg.paths.models)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    q_matrix, skill_names = load_q_matrix(data_dir / "q_matrix.csv")
    n_skills = q_matrix.shape[1]

    print(f"Response matrix: {n_llms} LLMs x {n_items} items")
    print(f"Q-matrix: {n_items} items x {n_skills} skills")

    triplets = build_triplets(response_df)
    print(f"Total triplets: {len(triplets):,}")

    train_val, test = train_test_split(triplets, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1 / 0.8, random_state=42)
    print(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    bs = cfg.model.batch_size
    train_loader = make_dataloader(train, q_matrix, bs, shuffle=True)
    val_loader = make_dataloader(val, q_matrix, bs, shuffle=False)
    test_loader = make_dataloader(test, q_matrix, bs, shuffle=False)

    print(f"\nTraining NCDM: {n_skills} skills, {n_items} items, {n_llms} LLMs")
    model = train_id_model(
        train_loader, val_loader, n_skills, n_items, n_llms,
        epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
    )

    test_auc, test_acc, test_rmse = eval_id_model(model, test_loader, device)
    print(f"\nTest results: AUC={test_auc:.4f}, Accuracy={test_acc:.4f}, RMSE={test_rmse:.4f}")

    model.save(str(model_dir / "ncdm.pt"))
    print(f"Model saved: {model_dir / 'ncdm.pt'}")

    mastery = extract_mastery_profiles(model, n_llms, n_skills, device)
    mastery_df = pd.DataFrame(mastery, index=llm_names, columns=skill_names)
    mastery_df.index.name = "llm"
    mastery_df.to_csv(data_dir / "skill_mastery_profiles.csv")
    print(f"Skill mastery profiles saved ({n_llms} LLMs x {n_skills} skills)")

    mastery_df["overall"] = mastery_df.mean(axis=1)
    mastery_df = mastery_df.sort_values("overall", ascending=False)
    print(f"\nTop 10 LLMs by average mastery:")
    for llm, row in mastery_df.head(10).iterrows():
        print(f"  {llm}: overall={row['overall']:.3f}")

    metrics = {
        "test_auc": float(test_auc), "test_accuracy": float(test_acc),
        "test_rmse": float(test_rmse), "n_llms": n_llms,
        "n_items": n_items, "n_skills": n_skills,
        "epochs": cfg.model.epochs, "lr": cfg.model.lr,
    }
    pd.Series(metrics).to_json(data_dir / "ncdm_metrics.json")


if __name__ == "__main__":
    main()
