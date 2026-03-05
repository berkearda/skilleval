"""
Text-Conditioned NCDM: replace ID-based item embeddings with SBERT projections.

Motivation: ID-based embeddings can't generalize to unseen items. By deriving
item difficulty from frozen text embeddings (CAIMIRA-style), the model can
predict on new questions given only their text — enabling cold-start routing.

Evaluations:
  A) Standard triplet split (sanity check vs ID-based)
  B) Exercise-level split (cold-start: text model vs ID-based)
  C) Routing demo: rank LLMs for unseen example queries

Usage:
    python 09_text_conditioned_ncdm.py
    python 09_text_conditioned_ncdm.py --device cpu --epochs 15
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Add EduCDM to path
REPO_DIR = Path(__file__).resolve().parent.parent / "repos" / "EduCDM"
sys.path.insert(0, str(REPO_DIR))
from EduCDM import NCDM
from EduCDM.NCDM.NCDM import PosLinear

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

K = 50  # HAC clusters (from step 08)
TEXT_DIM = 768  # all-mpnet-base-v2 output dimension


# ============================================================
# 2A. Pre-compute item text embeddings
# ============================================================

def precompute_item_embeddings():
    """Encode all 2,643 question texts with frozen SBERT → (2643, 768).
    Caches to item_text_embeddings.npz."""
    cache_path = DATA_DIR / "item_text_embeddings.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"Loaded cached item text embeddings: {data['embeddings'].shape}")
        return data["embeddings"]

    from sentence_transformers import SentenceTransformer

    with open(DATA_DIR / "skills_extracted.json") as f:
        items = json.load(f)

    texts = [item["problem"] for item in items]
    print(f"Encoding {len(texts)} item texts with all-mpnet-base-v2...")

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True,
                              batch_size=64)

    np.savez(cache_path, embeddings=embeddings)
    print(f"Saved item text embeddings: {embeddings.shape} → {cache_path}")
    return embeddings


# ============================================================
# 2B. TextConditionedNet
# ============================================================

class TextConditionedNet(nn.Module):
    """NCDM variant that derives item difficulty from text embeddings
    instead of ID-based nn.Embedding lookups.

    Student embeddings remain ID-based (we know all 235 LLMs).
    """

    def __init__(self, knowledge_n, student_n, text_dim=768):
        super().__init__()
        self.knowledge_dim = knowledge_n
        self.stu_dim = knowledge_n
        self.prednet_input_len = knowledge_n
        self.prednet_len1, self.prednet_len2 = 512, 256

        # Student embedding (ID-based — all LLMs are known)
        self.student_emb = nn.Embedding(student_n, self.stu_dim)

        # Text-conditioned item projections (replaces nn.Embedding)
        self.k_difficulty_proj = nn.Linear(text_dim, knowledge_n)
        self.e_difficulty_proj = nn.Linear(text_dim, 1)

        # Prediction sub-net (identical to original NCDM)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # Initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, text_emb, input_knowledge_point):
        """
        Args:
            stu_id: (batch,) LLM index
            text_emb: (batch, 768) frozen SBERT embedding for the question
            input_knowledge_point: (batch, K) Q-matrix row
        """
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty_proj(text_emb))
        e_difficulty = torch.sigmoid(self.e_difficulty_proj(text_emb))

        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


# ============================================================
# 2C. DataLoader helpers
# ============================================================

def make_text_dataloader(triplets, text_embeddings, q_matrix, batch_size=64, shuffle=True):
    """Build DataLoader: (user_id, text_emb, knowledge_emb, score)."""
    user_ids = torch.tensor(triplets[:, 0], dtype=torch.int64)
    scores = torch.tensor(triplets[:, 2], dtype=torch.float32)

    item_indices = triplets[:, 1].astype(int)
    text_embs = torch.tensor(text_embeddings[item_indices], dtype=torch.float32)
    knowledge_embs = torch.tensor(q_matrix[item_indices], dtype=torch.float32)

    dataset = TensorDataset(user_ids, text_embs, knowledge_embs, scores)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_id_dataloader(triplets, q_matrix, batch_size=64, shuffle=True):
    """Build DataLoader for ID-based NCDM: (user_id, item_id, knowledge_emb, score)."""
    user_ids = torch.tensor(triplets[:, 0], dtype=torch.int64)
    item_ids = torch.tensor(triplets[:, 1], dtype=torch.int64)
    scores = torch.tensor(triplets[:, 2], dtype=torch.float32)

    item_indices = triplets[:, 1].astype(int)
    knowledge_embs = torch.tensor(q_matrix[item_indices], dtype=torch.float32)

    dataset = TensorDataset(user_ids, item_ids, knowledge_embs, scores)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# Training / Evaluation
# ============================================================

def train_text_model(net, train_loader, val_loader, epochs=15, lr=0.002, device="cpu"):
    """Train TextConditionedNet with validation early-stop."""
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        net.train()
        losses = []

        for user_id, text_emb, knowledge_emb, y in tqdm(
            train_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False
        ):
            user_id = user_id.to(device)
            text_emb = text_emb.to(device)
            knowledge_emb = knowledge_emb.to(device)
            y = y.to(device)

            pred = net(user_id, text_emb, knowledge_emb)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        val_auc, val_acc, val_rmse = eval_text_model(net, val_loader, device)
        print(f"    loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}, val_rmse={val_rmse:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    if best_state:
        net.load_state_dict(best_state)
        print(f"  Restored best model (val_auc={best_auc:.4f})")

    return net


def eval_text_model(net, dataloader, device="cpu"):
    """Evaluate TextConditionedNet."""
    net.eval()
    net = net.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, text_emb, knowledge_emb, y in dataloader:
            user_id = user_id.to(device)
            text_emb = text_emb.to(device)
            knowledge_emb = knowledge_emb.to(device)
            pred = net(user_id, text_emb, knowledge_emb)
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return auc, acc, rmse


def train_id_model(train_loader, val_loader, n_skills, n_items, n_llms,
                   epochs=15, lr=0.002, device="cpu"):
    """Train standard ID-based NCDM, return model."""
    model = NCDM(n_skills, n_items, n_llms)
    model.ncdm_net = model.ncdm_net.to(device)
    optimizer = torch.optim.Adam(model.ncdm_net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

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
        val_auc, val_acc, val_rmse = eval_id_model(model, val_loader, device)
        print(f"    loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}, val_rmse={val_rmse:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.ncdm_net.state_dict().items()}

    if best_state:
        model.ncdm_net.load_state_dict(best_state)
        print(f"  Restored best model (val_auc={best_auc:.4f})")

    return model


def eval_id_model(model, dataloader, device="cpu"):
    """Evaluate standard ID-based NCDM."""
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


# ============================================================
# 2D. Splits
# ============================================================

def standard_triplet_split(triplets):
    """80/10/10 random triplet split (same as 05/07)."""
    all_idx = np.arange(len(triplets))
    train_val_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1 / 0.8, random_state=42)
    return triplets[train_idx], triplets[val_idx], triplets[test_idx]


def exercise_level_split(triplets, n_items, holdout_ratio=0.2):
    """Hold out 20% of items entirely. All triplets for held-out items → test.
    Remaining 80% items split 90/10 train/val by triplet."""
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=holdout_ratio, random_state=42)
    train_items_set = set(train_items)
    test_items_set = set(test_items)

    train_val_triplets = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    test_triplets = triplets[np.isin(triplets[:, 1].astype(int), test_items)]

    # Split train_val into train/val
    all_idx = np.arange(len(train_val_triplets))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.1, random_state=42)

    return (train_val_triplets[train_idx], train_val_triplets[val_idx],
            test_triplets, train_items, test_items)


# ============================================================
# 2E. Routing demonstration
# ============================================================

def routing_demo(net, text_embeddings, q_matrix, llm_names, n_items, device="cpu"):
    """Demonstrate routing: for novel queries, predict P(correct) for all LLMs."""
    print("\n" + "=" * 70)
    print("ROUTING DEMONSTRATION")
    print("=" * 70)

    # Example new queries (not in the dataset)
    example_queries = [
        "What is the derivative of x^3 * sin(x)? Use the product rule and show your work.",
        "A train leaves Chicago at 8am traveling at 60mph. Another train leaves New York "
        "at 9am traveling at 80mph toward Chicago. If Chicago and New York are 790 miles "
        "apart, at what time do the trains meet?",
        "Explain the difference between a stack and a queue data structure. "
        "Give a real-world example of each.",
        "Translate the following Python function to Rust, maintaining the same logic: "
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    ]

    # Encode queries with SBERT
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-mpnet-base-v2")
    query_embeddings = sbert.encode(example_queries, normalize_embeddings=True)

    # Nearest-neighbor item lookup for Q-matrix approximation
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(text_embeddings)

    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(len(llm_names), device=device)

    for i, query in enumerate(example_queries):
        print(f"\n{'─' * 60}")
        print(f"Query {i+1}: {query[:100]}...")

        # Find nearest item for Q-matrix row
        dists, indices = nn_model.kneighbors(query_embeddings[i:i+1])
        nn_item_idx = indices[0, 0]
        nn_dist = dists[0, 0]
        print(f"  Nearest item: #{nn_item_idx} (cosine dist={nn_dist:.4f})")

        # Prepare inputs
        query_emb_t = torch.tensor(query_embeddings[i], dtype=torch.float32, device=device)
        query_emb_batch = query_emb_t.unsqueeze(0).expand(len(llm_names), -1)
        q_row = torch.tensor(q_matrix[nn_item_idx], dtype=torch.float32, device=device)
        q_row_batch = q_row.unsqueeze(0).expand(len(llm_names), -1)

        with torch.no_grad():
            preds = net(all_llm_ids, query_emb_batch, q_row_batch).cpu().numpy()

        # Rank LLMs
        ranking = np.argsort(-preds)
        print(f"\n  Top 5 LLMs (highest P(correct)):")
        for rank, idx in enumerate(ranking[:5]):
            print(f"    {rank+1}. {llm_names[idx]}: P(correct)={preds[idx]:.4f}")

        print(f"\n  Bottom 3 LLMs (lowest P(correct)):")
        for rank, idx in enumerate(ranking[-3:]):
            print(f"    {len(llm_names)-2+rank}. {llm_names[idx]}: P(correct)={preds[idx]:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Text-Conditioned NCDM + Routing")
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

    # ── Load data ──
    with open(DATA_DIR / "skills_extracted.json") as f:
        items_data = json.load(f)

    response_df = pd.read_csv(DATA_DIR / "response_matrix.csv", index_col=0)
    n_llms = response_df.shape[0]
    n_items = response_df.shape[1]
    llm_names = list(response_df.index)
    print(f"Response matrix: {n_llms} LLMs x {n_items} items")

    # Load HAC-50 Q-matrix from step 08
    q_df = pd.read_csv(DATA_DIR / "q_matrix_hac50.csv")
    meta_cols = [c for c in ["item_idx", "source"] if c in q_df.columns]
    skill_cols = [c for c in q_df.columns if c not in meta_cols]
    q_matrix = q_df[skill_cols].values.astype(float)
    n_skills = q_matrix.shape[1]
    print(f"Q-matrix (HAC-50): {n_items} items x {n_skills} skills")

    # Build triplets
    triplets = []
    for llm_id in range(n_llms):
        row = response_df.iloc[llm_id].values
        for item_id in range(n_items):
            triplets.append((llm_id, item_id, float(row[item_id])))
    triplets = np.array(triplets)
    print(f"Total triplets: {len(triplets):,}")

    # ── Pre-compute item text embeddings ──
    text_embeddings = precompute_item_embeddings()  # (2643, 768)

    # Collect all results
    all_metrics = {}

    # ================================================================
    # PROTOCOL A: Standard triplet split — TextConditionedNet
    # ================================================================
    print("\n" + "=" * 70)
    print("PROTOCOL A: Text-Conditioned NCDM — Standard Triplet Split")
    print("=" * 70)

    train_trip, val_trip, test_trip = standard_triplet_split(triplets)
    print(f"Split: train={len(train_trip):,}, val={len(val_trip):,}, test={len(test_trip):,}")

    train_loader_a = make_text_dataloader(train_trip, text_embeddings, q_matrix,
                                          args.batch_size, shuffle=True)
    val_loader_a = make_text_dataloader(val_trip, text_embeddings, q_matrix,
                                        args.batch_size, shuffle=False)
    test_loader_a = make_text_dataloader(test_trip, text_embeddings, q_matrix,
                                         args.batch_size, shuffle=False)

    net_a = TextConditionedNet(n_skills, n_llms, TEXT_DIM)
    net_a = train_text_model(net_a, train_loader_a, val_loader_a,
                             epochs=args.epochs, lr=args.lr, device=args.device)

    auc_a, acc_a, rmse_a = eval_text_model(net_a, test_loader_a, args.device)
    print(f"\nProtocol A results: AUC={auc_a:.4f}, Acc={acc_a:.4f}, RMSE={rmse_a:.4f}")
    all_metrics["protocol_a_text_standard"] = {
        "model": "TextConditionedNet", "split": "standard_triplet",
        "test_auc": float(auc_a), "test_accuracy": float(acc_a), "test_rmse": float(rmse_a),
    }

    # Save the text-conditioned model (trained on full standard split)
    model_path = MODEL_DIR / "ncdm_text_conditioned.pt"
    torch.save(net_a.state_dict(), str(model_path))
    print(f"Model saved: {model_path}")

    # ================================================================
    # PROTOCOL B: Exercise-level split — TextConditionedNet vs ID-based
    # ================================================================
    print("\n" + "=" * 70)
    print("PROTOCOL B: Exercise-Level Split (Cold-Start)")
    print("=" * 70)

    train_trip_b, val_trip_b, test_trip_b, train_items, test_items = \
        exercise_level_split(triplets, n_items, holdout_ratio=0.2)
    print(f"Train items: {len(train_items)}, Test (held-out) items: {len(test_items)}")
    print(f"Split: train={len(train_trip_b):,}, val={len(val_trip_b):,}, test={len(test_trip_b):,}")

    # --- B1: Text-Conditioned on exercise split ---
    print(f"\n--- B1: TextConditionedNet (cold-start) ---")
    train_loader_b1 = make_text_dataloader(train_trip_b, text_embeddings, q_matrix,
                                           args.batch_size, shuffle=True)
    val_loader_b1 = make_text_dataloader(val_trip_b, text_embeddings, q_matrix,
                                         args.batch_size, shuffle=False)
    test_loader_b1 = make_text_dataloader(test_trip_b, text_embeddings, q_matrix,
                                          args.batch_size, shuffle=False)

    net_b1 = TextConditionedNet(n_skills, n_llms, TEXT_DIM)
    net_b1 = train_text_model(net_b1, train_loader_b1, val_loader_b1,
                              epochs=args.epochs, lr=args.lr, device=args.device)

    auc_b1, acc_b1, rmse_b1 = eval_text_model(net_b1, test_loader_b1, args.device)
    print(f"\nProtocol B1 (Text, cold-start): AUC={auc_b1:.4f}, Acc={acc_b1:.4f}, RMSE={rmse_b1:.4f}")
    all_metrics["protocol_b1_text_exercise"] = {
        "model": "TextConditionedNet", "split": "exercise_level",
        "test_auc": float(auc_b1), "test_accuracy": float(acc_b1), "test_rmse": float(rmse_b1),
        "n_train_items": int(len(train_items)), "n_test_items": int(len(test_items)),
    }

    # --- B2: ID-based NCDM on same exercise split (negative control) ---
    print(f"\n--- B2: ID-based NCDM (cold-start — negative control) ---")
    train_loader_b2 = make_id_dataloader(train_trip_b, q_matrix,
                                         args.batch_size, shuffle=True)
    val_loader_b2 = make_id_dataloader(val_trip_b, q_matrix,
                                       args.batch_size, shuffle=False)
    test_loader_b2 = make_id_dataloader(test_trip_b, q_matrix,
                                        args.batch_size, shuffle=False)

    id_model_b2 = train_id_model(train_loader_b2, val_loader_b2,
                                 n_skills, n_items, n_llms,
                                 epochs=args.epochs, lr=args.lr, device=args.device)

    auc_b2, acc_b2, rmse_b2 = eval_id_model(id_model_b2, test_loader_b2, args.device)
    print(f"\nProtocol B2 (ID-based, cold-start): AUC={auc_b2:.4f}, Acc={acc_b2:.4f}, RMSE={rmse_b2:.4f}")
    all_metrics["protocol_b2_id_exercise"] = {
        "model": "ID-based NCDM", "split": "exercise_level",
        "test_auc": float(auc_b2), "test_accuracy": float(acc_b2), "test_rmse": float(rmse_b2),
        "n_train_items": int(len(train_items)), "n_test_items": int(len(test_items)),
    }

    # ================================================================
    # Summary table
    # ================================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Evaluation':<35} {'Model':<22} {'Split':<18} {'AUC':>8} {'Acc':>8} {'RMSE':>8}")
    print("-" * 100)

    rows = [
        ("A: Sanity check", "TextConditionedNet", "Standard triplet", auc_a, acc_a, rmse_a),
        ("B1: Cold-start (text)", "TextConditionedNet", "Exercise-level", auc_b1, acc_b1, rmse_b1),
        ("B2: Cold-start (ID, neg ctrl)", "ID-based NCDM", "Exercise-level", auc_b2, acc_b2, rmse_b2),
    ]

    # Include HAC-50 baseline if metrics exist
    hac50_path = DATA_DIR / "ncdm_metrics_hac50.json"
    if hac50_path.exists():
        with open(hac50_path) as f:
            hac50 = json.load(f)
        rows.insert(0, (
            "HAC-50 baseline", "ID-based NCDM", "Standard triplet",
            hac50["test_auc"], hac50["test_accuracy"], hac50["test_rmse"]
        ))
        all_metrics["hac50_baseline"] = {
            "model": "ID-based NCDM", "split": "standard_triplet",
            "test_auc": hac50["test_auc"], "test_accuracy": hac50["test_accuracy"],
            "test_rmse": hac50["test_rmse"],
        }

    for label, model_name, split, auc, acc, rmse in rows:
        print(f"{label:<35} {model_name:<22} {split:<18} {auc:>8.4f} {acc:>8.4f} {rmse:>8.4f}")

    print(f"\nCold-start AUC gain: TextConditioned ({auc_b1:.4f}) vs ID-based ({auc_b2:.4f})"
          f" = +{auc_b1 - auc_b2:.4f}")

    # ── Save all metrics ──
    with open(DATA_DIR / "ncdm_text_conditioned_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved: ncdm_text_conditioned_metrics.json")

    # ================================================================
    # 2E. Routing demonstration
    # ================================================================
    routing_demo(net_a, text_embeddings, q_matrix, llm_names, n_items, args.device)


if __name__ == "__main__":
    main()
