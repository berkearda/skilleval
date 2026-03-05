"""
Re-cluster skills with HAC K=50 (cosine, average linkage) and retrain NCDM.

Best clustering from 07_compare_clustering.py was HAC K=50 (AUC=0.9486).
This script produces the final HAC-50 artifacts: Q-matrix, taxonomy, model,
skill mastery profiles, and evaluation metrics.

Usage:
    python 08_hac_clustering.py
    python 08_hac_clustering.py --device cpu --epochs 15
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Add EduCDM to path
REPO_DIR = Path(__file__).resolve().parent.parent / "repos" / "EduCDM"
sys.path.insert(0, str(REPO_DIR))
from EduCDM import NCDM

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

K = 50  # number of clusters


# ---------- Clustering ----------

def cluster_hac(embeddings, k):
    """Agglomerative clustering with cosine distance + average linkage."""
    hac = AgglomerativeClustering(
        n_clusters=k, metric="cosine", linkage="average"
    )
    return hac.fit_predict(embeddings)


def label_clusters(unique_skills, labels, counts):
    """Generate a representative label for each cluster based on most frequent members."""
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


def build_q_matrix(data, all_skills, unique_skills, labels, cluster_labels):
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

    q_matrix = pd.DataFrame(rows, columns=cluster_names)

    if data and "item_idx" in data[0]:
        q_matrix.insert(0, "item_idx", [d["item_idx"] for d in data])
    if data and "source" in data[0]:
        q_matrix.insert(1, "source", [d["source"] for d in data])

    return q_matrix, skill_to_cluster


# ---------- NCDM helpers ----------

def make_dataloader(triplets, q_matrix, batch_size=64, shuffle=True):
    """Convert triplets to a PyTorch DataLoader with Q-matrix embeddings."""
    user_ids = torch.tensor(triplets[:, 0], dtype=torch.int64)
    item_ids = torch.tensor(triplets[:, 1], dtype=torch.int64)
    scores = torch.tensor(triplets[:, 2], dtype=torch.float32)

    # Vectorized: index Q-matrix by item_id column
    item_indices = triplets[:, 1].astype(int)
    knowledge_emb = torch.tensor(q_matrix[item_indices], dtype=torch.float32)

    dataset = TensorDataset(user_ids, item_ids, knowledge_emb, scores)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def train_model(model, train_loader, val_loader, epochs=15, lr=0.002, device="cpu"):
    """Training loop with validation monitoring and best-model restoration."""
    model.ncdm_net = model.ncdm_net.to(device)
    optimizer = torch.optim.Adam(model.ncdm_net.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.ncdm_net.train()
        losses = []

        for user_id, item_id, knowledge_emb, y in tqdm(
            train_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False
        ):
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
        print(f"    loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}, val_rmse={val_rmse:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.ncdm_net.state_dict().items()}

    if best_state:
        model.ncdm_net.load_state_dict(best_state)
        print(f"  Restored best model (val_auc={best_auc:.4f})")

    return model


def extract_mastery_profiles(model, n_llms, n_skills, device="cpu"):
    """Extract learned skill mastery profile for each LLM."""
    model.ncdm_net.eval()
    model.ncdm_net = model.ncdm_net.to(device)

    with torch.no_grad():
        all_ids = torch.arange(n_llms, device=device)
        raw_emb = model.ncdm_net.student_emb(all_ids)
        mastery = torch.sigmoid(raw_emb).cpu().numpy()

    return mastery  # (n_llms, n_skills)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="HAC K=50 clustering + NCDM baseline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps", help="cpu, cuda, or mps")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"Device: {args.device}")

    # ── 1. Load skill embeddings ──
    emb_data = np.load(DATA_DIR / "skill_embeddings.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]   # (7723, 768)
    unique_skills = list(emb_data["skills"])
    print(f"Skill embeddings: {embeddings.shape}")

    # Skill frequency counts (for cluster labelling)
    with open(DATA_DIR / "skills_extracted.json") as f:
        data = json.load(f)
    all_skills = []
    for item in data:
        skills = item.get("skills", [])
        if isinstance(skills, str):
            skills = eval(skills)
        all_skills.append([s.lower().strip() for s in skills])
    counts = Counter(s for skills in all_skills for s in skills)

    # ── 2. HAC K=50 clustering ──
    print(f"\nClustering {len(unique_skills)} skills with HAC K={K}...")
    labels = cluster_hac(embeddings, K)
    n_clusters = len(set(labels))
    sil = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  {n_clusters} clusters, silhouette(cosine)={sil:.4f}")

    # ── 3. Label clusters and build Q-matrix ──
    cluster_labels = label_clusters(unique_skills, labels, counts)

    print(f"\nSkill taxonomy ({n_clusters} clusters):")
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        print(f"  [{cid}] {cluster_labels[cid]} ({len(members)} skills)")

    q_matrix_df, skill_to_cluster = build_q_matrix(
        data, all_skills, unique_skills, labels, cluster_labels
    )

    # Extract numeric Q-matrix (drop metadata columns)
    meta_cols = [c for c in ["item_idx", "source"] if c in q_matrix_df.columns]
    skill_cols = [c for c in q_matrix_df.columns if c not in meta_cols]
    q_matrix = q_matrix_df[skill_cols].values  # (2643, 50)
    n_skills = q_matrix.shape[1]

    print(f"\nQ-matrix: {q_matrix_df.shape[0]} items x {n_skills} skills")
    skills_per_item = q_matrix.sum(axis=1)
    print(f"Skills/item: mean={skills_per_item.mean():.1f}, min={skills_per_item.min()}, max={skills_per_item.max()}")

    # ── 4. Save clustering artifacts ──
    q_matrix_df.to_csv(DATA_DIR / "q_matrix_hac50.csv", index=False)
    print(f"Saved: q_matrix_hac50.csv")

    taxonomy = {}
    for cid in sorted(cluster_labels):
        members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
        taxonomy[cluster_labels[cid]] = members
    with open(DATA_DIR / "skill_taxonomy_hac50.json", "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Saved: skill_taxonomy_hac50.json")

    s2c = {unique_skills[i]: cluster_labels[labels[i]] for i in range(len(unique_skills))}
    with open(DATA_DIR / "skill_to_cluster_hac50.json", "w") as f:
        json.dump(s2c, f, indent=2)
    print(f"Saved: skill_to_cluster_hac50.json")

    # ── 5. Load response matrix and build triplets ──
    response_df = pd.read_csv(DATA_DIR / "response_matrix.csv", index_col=0)
    n_llms = response_df.shape[0]
    n_items = response_df.shape[1]
    llm_names = list(response_df.index)
    print(f"\nResponse matrix: {n_llms} LLMs x {n_items} items")

    triplets = []
    for llm_id in range(n_llms):
        row = response_df.iloc[llm_id].values
        for item_id in range(n_items):
            triplets.append((llm_id, item_id, float(row[item_id])))
    triplets = np.array(triplets)

    # Standard 80/10/10 split
    all_indices = np.arange(len(triplets))
    train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1 / 0.8, random_state=42)
    print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    train_loader = make_dataloader(triplets[train_idx], q_matrix, args.batch_size, shuffle=True)
    val_loader = make_dataloader(triplets[val_idx], q_matrix, args.batch_size, shuffle=False)
    test_loader = make_dataloader(triplets[test_idx], q_matrix, args.batch_size, shuffle=False)

    # ── 6. Train NCDM ──
    print(f"\nTraining NCDM: {n_skills} skills, {n_items} items, {n_llms} LLMs")
    model = NCDM(n_skills, n_items, n_llms)
    model = train_model(model, train_loader, val_loader,
                        epochs=args.epochs, lr=args.lr, device=args.device)

    # ── 7. Evaluate ──
    test_auc, test_acc, test_rmse = evaluate(model, test_loader, args.device)
    print(f"\nTest results: AUC={test_auc:.4f}, Accuracy={test_acc:.4f}, RMSE={test_rmse:.4f}")

    # ── 8. Save model ──
    model_path = MODEL_DIR / "ncdm_hac50.pt"
    model.save(str(model_path))
    print(f"Model saved: {model_path}")

    # ── 9. Skill mastery profiles ──
    mastery = extract_mastery_profiles(model, n_llms, n_skills, args.device)
    mastery_df = pd.DataFrame(mastery, index=llm_names, columns=skill_cols)
    mastery_df.index.name = "llm"
    mastery_df.to_csv(DATA_DIR / "skill_mastery_profiles_hac50.csv")
    print(f"Skill mastery profiles saved ({n_llms} LLMs x {n_skills} skills)")

    mastery_df["overall"] = mastery_df.mean(axis=1)
    mastery_df = mastery_df.sort_values("overall", ascending=False)
    print(f"\nTop 10 LLMs by average mastery:")
    for llm, row in mastery_df.head(10).iterrows():
        print(f"  {llm}: overall={row['overall']:.3f}")
    print(f"\nBottom 5 LLMs:")
    for llm, row in mastery_df.tail(5).iterrows():
        print(f"  {llm}: overall={row['overall']:.3f}")

    # ── 10. Save metrics ──
    metrics = {
        "method": "HAC",
        "n_clusters": int(n_clusters),
        "silhouette": float(sil),
        "test_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "test_rmse": float(test_rmse),
        "n_llms": n_llms,
        "n_items": n_items,
        "n_skills": n_skills,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    with open(DATA_DIR / "ncdm_metrics_hac50.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: ncdm_metrics_hac50.json")
    print(f"  AUC={test_auc:.4f}  Acc={test_acc:.4f}  RMSE={test_rmse:.4f}")


if __name__ == "__main__":
    main()
