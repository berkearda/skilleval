"""
Fit a Neural Cognitive Diagnosis Model (NCDM) on the response matrix + Q-matrix.

Loads the binary response matrix (LLMs x items) and the Q-matrix
(items x skills), trains NCDM from EduCDM, evaluates performance,
and extracts per-LLM skill mastery profiles.

Usage:
    python 05_fit_ncdm.py
    python 05_fit_ncdm.py --epochs 20 --lr 0.001 --device mps
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from EduCDM import NCDM

from cdmeval.utils.device import resolve_device
from cdmeval.data.response_matrix import load_response_matrix, load_q_matrix, build_triplets
from cdmeval.data.dataloader import make_dataloader
from cdmeval.evaluation.metrics import eval_id_model
from cdmeval.evaluation.training import train_id_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_DIR = DATA_DIR


def extract_mastery_profiles(model, n_llms, n_skills, device="cpu"):
    """Extract the learned skill mastery profile for each LLM."""
    model.ncdm_net.eval()
    model.ncdm_net = model.ncdm_net.to(device)

    with torch.no_grad():
        all_ids = torch.arange(n_llms, device=device)
        raw_emb = model.ncdm_net.student_emb(all_ids)
        mastery = torch.sigmoid(raw_emb).cpu().numpy()

    return mastery  # (n_llms, n_skills)


def main():
    parser = argparse.ArgumentParser(description="Fit Neural CDM on LLM response data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps", help="cpu, cuda, or mps")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    args.device = resolve_device(args.device)
    print(f"Device: {args.device}")

    # Load data
    response_df, n_llms, n_items, llm_names = load_response_matrix(DATA_DIR / "response_matrix.csv")
    q_matrix, skill_names = load_q_matrix(DATA_DIR / "q_matrix.csv")
    n_skills = q_matrix.shape[1]

    print(f"Response matrix: {n_llms} LLMs x {n_items} items")
    print(f"Q-matrix: {n_items} items x {n_skills} skills")

    triplets = build_triplets(response_df)
    print(f"Total triplets: {len(triplets):,} (density: {triplets[:, 2].mean():.3f})")

    # Train/val/test split
    train_val, test = train_test_split(triplets, test_size=args.test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=args.val_ratio / (1 - args.test_ratio),
                                  random_state=42)
    print(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    train_loader = make_dataloader(train, q_matrix, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val, q_matrix, args.batch_size, shuffle=False)
    test_loader = make_dataloader(test, q_matrix, args.batch_size, shuffle=False)

    # Train NCDM
    print(f"\nNCDM: {n_skills} skills, {n_items} items, {n_llms} LLMs")
    model = train_id_model(train_loader, val_loader, n_skills, n_items, n_llms,
                           epochs=args.epochs, lr=args.lr, device=args.device)

    # Final evaluation
    test_auc, test_acc, test_rmse = eval_id_model(model, test_loader, args.device)
    print(f"\nTest results: AUC={test_auc:.4f}, Accuracy={test_acc:.4f}, RMSE={test_rmse:.4f}")

    # Save model
    model_path = MODEL_DIR / "ncdm.pt"
    model.save(str(model_path))
    print(f"Model saved: {model_path}")

    # Extract skill mastery profiles
    mastery = extract_mastery_profiles(model, n_llms, n_skills, args.device)
    mastery_df = pd.DataFrame(mastery, index=llm_names, columns=skill_names)
    mastery_df.index.name = "llm"
    mastery_df.to_csv(OUTPUT_DIR / "skill_mastery_profiles.csv")
    print(f"\nSkill mastery profiles saved ({n_llms} LLMs x {n_skills} skills)")

    # Print top/bottom LLMs
    mastery_df["overall"] = mastery_df.mean(axis=1)
    mastery_df = mastery_df.sort_values("overall", ascending=False)

    print(f"\nTop 10 LLMs by average mastery:")
    for llm, row in mastery_df.head(10).iterrows():
        skills_str = ", ".join(f"{s}={row[s]:.2f}" for s in skill_names[:5])
        print(f"  {llm}: overall={row['overall']:.3f} ({skills_str}...)")

    print(f"\nBottom 10 LLMs:")
    for llm, row in mastery_df.tail(10).iterrows():
        print(f"  {llm}: overall={row['overall']:.3f}")

    # Save evaluation metrics
    metrics = {
        "test_auc": test_auc,
        "test_accuracy": test_acc,
        "test_rmse": test_rmse,
        "n_llms": n_llms,
        "n_items": n_items,
        "n_skills": n_skills,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    pd.Series(metrics).to_json(OUTPUT_DIR / "ncdm_metrics.json")


if __name__ == "__main__":
    main()
