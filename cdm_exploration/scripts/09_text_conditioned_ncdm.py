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

import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from EduCDM import NCDM

from cdmeval.utils.device import resolve_device
from cdmeval.data.response_matrix import load_response_matrix, load_q_matrix, build_triplets
from cdmeval.data.dataloader import make_dataloader, make_text_dataloader
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.evaluation.metrics import eval_id_model, eval_text_model
from cdmeval.evaluation.training import train_id_model, train_text_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

K = 50  # HAC clusters (from step 08)
TEXT_DIM = 768  # all-mpnet-base-v2 output dimension


# ============================================================
# Pre-compute item text embeddings
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
# Splits
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

    train_val_triplets = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    test_triplets = triplets[np.isin(triplets[:, 1].astype(int), test_items)]

    all_idx = np.arange(len(train_val_triplets))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.1, random_state=42)

    return (train_val_triplets[train_idx], train_val_triplets[val_idx],
            test_triplets, train_items, test_items)


# ============================================================
# Routing demonstration
# ============================================================

def routing_demo(net, text_embeddings, q_matrix, llm_names, n_items, device="cpu"):
    """Demonstrate routing: for novel queries, predict P(correct) for all LLMs."""
    print("\n" + "=" * 70)
    print("ROUTING DEMONSTRATION")
    print("=" * 70)

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

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-mpnet-base-v2")
    query_embeddings = sbert.encode(example_queries, normalize_embeddings=True)

    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(text_embeddings)

    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(len(llm_names), device=device)

    for i, query in enumerate(example_queries):
        print(f"\n{'─' * 60}")
        print(f"Query {i+1}: {query[:100]}...")

        dists, indices = nn_model.kneighbors(query_embeddings[i:i+1])
        nn_item_idx = indices[0, 0]
        nn_dist = dists[0, 0]
        print(f"  Nearest item: #{nn_item_idx} (cosine dist={nn_dist:.4f})")

        query_emb_t = torch.tensor(query_embeddings[i], dtype=torch.float32, device=device)
        query_emb_batch = query_emb_t.unsqueeze(0).expand(len(llm_names), -1)
        q_row = torch.tensor(q_matrix[nn_item_idx], dtype=torch.float32, device=device)
        q_row_batch = q_row.unsqueeze(0).expand(len(llm_names), -1)

        with torch.no_grad():
            preds = net(all_llm_ids, query_emb_batch, q_row_batch).cpu().numpy()

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
    args.device = resolve_device(args.device)
    print(f"Device: {args.device}")

    # ── Load data ──
    with open(DATA_DIR / "skills_extracted.json") as f:
        items_data = json.load(f)

    response_df, n_llms, n_items, llm_names = load_response_matrix(
        DATA_DIR / "response_matrix.csv"
    )
    print(f"Response matrix: {n_llms} LLMs x {n_items} items")

    q_matrix, skill_cols = load_q_matrix(DATA_DIR / "q_matrix_hac50.csv")
    n_skills = q_matrix.shape[1]
    print(f"Q-matrix (HAC-50): {n_items} items x {n_skills} skills")

    triplets = build_triplets(response_df)
    print(f"Total triplets: {len(triplets):,}")

    # ── Pre-compute item text embeddings ──
    text_embeddings = precompute_item_embeddings()  # (2643, 768)

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

    # --- B2: ID-based NCDM on same exercise split ---
    print(f"\n--- B2: ID-based NCDM (cold-start — negative control) ---")
    train_loader_b2 = make_dataloader(train_trip_b, q_matrix,
                                      args.batch_size, shuffle=True)
    val_loader_b2 = make_dataloader(val_trip_b, q_matrix,
                                    args.batch_size, shuffle=False)
    test_loader_b2 = make_dataloader(test_trip_b, q_matrix,
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
    # Routing demonstration
    # ================================================================
    routing_demo(net_a, text_embeddings, q_matrix, llm_names, n_items, args.device)


if __name__ == "__main__":
    main()
