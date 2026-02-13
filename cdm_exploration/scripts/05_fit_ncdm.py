"""
Fit a Neural Cognitive Diagnosis Model (NCDM) on the response matrix + Q-matrix.

Loads the binary response matrix (LLMs x items) and the Q-matrix
(items x skills), trains NCDM from EduCDM, evaluates performance,
and extracts per-LLM skill mastery profiles.

Usage:
    python 05_fit_ncdm.py
    python 05_fit_ncdm.py --epochs 20 --lr 0.001 --device mps
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from pathlib import Path
from tqdm import tqdm

# Add EduCDM to path
REPO_DIR = Path(__file__).resolve().parent.parent / "repos" / "EduCDM"
sys.path.insert(0, str(REPO_DIR))
from EduCDM import NCDM

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_DIR = DATA_DIR


def load_data():
    """Load response matrix and Q-matrix, return aligned arrays."""
    response_df = pd.read_csv(DATA_DIR / "response_matrix.csv", index_col=0)
    q_matrix_df = pd.read_csv(DATA_DIR / "q_matrix.csv")

    # Separate metadata from skill columns in Q-matrix
    meta_cols = [c for c in ["item_idx", "source"] if c in q_matrix_df.columns]
    skill_cols = [c for c in q_matrix_df.columns if c not in meta_cols]
    q_matrix = q_matrix_df[skill_cols].values  # (n_items, n_skills)

    n_llms = response_df.shape[0]
    n_items = response_df.shape[1]
    n_skills = q_matrix.shape[1]

    print(f"Response matrix: {n_llms} LLMs x {n_items} items")
    print(f"Q-matrix: {n_items} items x {n_skills} skills")
    print(f"Skill names: {skill_cols}")

    # Convert response matrix to triplet format (llm_id, item_id, score)
    llm_names = list(response_df.index)
    triplets = []
    for llm_id in range(n_llms):
        row = response_df.iloc[llm_id].values
        for item_id in range(n_items):
            triplets.append((llm_id, item_id, float(row[item_id])))

    triplets = np.array(triplets)
    print(f"Total triplets: {len(triplets):,} (density: {triplets[:, 2].mean():.3f})")

    return triplets, q_matrix, n_llms, n_items, n_skills, skill_cols, llm_names


def make_dataloader(triplets, q_matrix, batch_size=64, shuffle=True):
    """Convert triplets to a PyTorch DataLoader with Q-matrix embeddings."""
    user_ids = torch.tensor(triplets[:, 0], dtype=torch.int64)
    item_ids = torch.tensor(triplets[:, 1], dtype=torch.int64)
    scores = torch.tensor(triplets[:, 2], dtype=torch.float32)

    # Build knowledge embedding per triplet (Q-matrix row for that item)
    knowledge_emb = torch.zeros(len(triplets), q_matrix.shape[1])
    for i in range(len(triplets)):
        knowledge_emb[i] = torch.tensor(q_matrix[int(triplets[i, 1])], dtype=torch.float32)

    dataset = TensorDataset(user_ids, item_ids, knowledge_emb, scores)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def extract_mastery_profiles(model, n_llms, n_skills, device="cpu"):
    """
    Extract the learned skill mastery profile for each LLM.
    The student embedding in NCDM represents latent ability per skill dimension.
    """
    model.ncdm_net.eval()
    model.ncdm_net = model.ncdm_net.to(device)

    with torch.no_grad():
        all_ids = torch.arange(n_llms, device=device)
        raw_emb = model.ncdm_net.student_emb(all_ids)
        mastery = torch.sigmoid(raw_emb).cpu().numpy()

    return mastery  # (n_llms, n_skills)


def evaluate(model, dataloader, device="cpu"):
    """Compute AUC, accuracy, and RMSE on a dataset."""
    model.ncdm_net.eval()
    model.ncdm_net = model.ncdm_net.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, item_id, knowledge_emb, y in dataloader:
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            pred = model.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return auc, acc, rmse


def train_model(model, train_loader, val_loader, epochs=10, lr=0.002, device="cpu"):
    """Training loop with validation monitoring."""
    model.ncdm_net = model.ncdm_net.to(device)
    optimizer = torch.optim.Adam(model.ncdm_net.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.ncdm_net.train()
        losses = []

        for user_id, item_id, knowledge_emb, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_id = user_id.to(device)
            item_id = item_id.to(device)
            knowledge_emb = knowledge_emb.to(device)
            y = y.to(device)

            pred = model.ncdm_net(user_id, item_id, knowledge_emb)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        val_auc, val_acc, val_rmse = evaluate(model, val_loader, device)
        print(f"  loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}, val_rmse={val_rmse:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.ncdm_net.state_dict().items()}

    # Restore best model
    if best_state:
        model.ncdm_net.load_state_dict(best_state)
        print(f"Restored best model (val_auc={best_auc:.4f})")

    return model


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

    # Check device
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"Device: {args.device}")

    # Load data
    triplets, q_matrix, n_llms, n_items, n_skills, skill_names, llm_names = load_data()

    # Train/val/test split (by triplet, stratified isn't necessary here)
    train_val, test = train_test_split(triplets, test_size=args.test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=args.val_ratio / (1 - args.test_ratio), random_state=42)
    print(f"Split: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    train_loader = make_dataloader(train, q_matrix, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val, q_matrix, args.batch_size, shuffle=False)
    test_loader = make_dataloader(test, q_matrix, args.batch_size, shuffle=False)

    # Initialize and train NCDM
    model = NCDM(n_skills, n_items, n_llms)
    print(f"\nNCDM: {n_skills} skills, {n_items} items, {n_llms} LLMs")

    model = train_model(model, train_loader, val_loader,
                        epochs=args.epochs, lr=args.lr, device=args.device)

    # Final evaluation on test set
    test_auc, test_acc, test_rmse = evaluate(model, test_loader, args.device)
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

    # Print top/bottom LLMs by overall mastery
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
