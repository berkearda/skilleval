"""
Extended Experiments for CDM Evaluation.

Five experiments that deepen our understanding of the text-conditioned NCDM:
  1. Skill Correlation Analysis — inter-skill correlations and PCA
  2. Difficulty-Stratified Evaluation — AUC by item difficulty bucket
  3. Routing Accuracy with Ground Truth — accuracy@k for LLM ranking
  4. Per-Skill AUC Breakdown — skill-level text vs ID comparison
  5. Q-Row Approximation Quality — Jaccard similarity of NN-approximated Q-rows

Usage:
    python 10_extended_experiments.py
    python 10_extended_experiments.py --device mps --epochs 15
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm
from EduCDM import NCDM

from cdmeval.utils.device import resolve_device, seed_everything
from cdmeval.data.response_matrix import load_response_matrix, load_q_matrix, build_triplets
from cdmeval.data.dataloader import make_dataloader, make_text_dataloader
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.evaluation.metrics import eval_id_model, eval_text_model
from cdmeval.evaluation.training import train_id_model, train_text_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cdm_ready"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "report"

K = 50  # HAC clusters
TEXT_DIM = 768  # all-mpnet-base-v2 output dimension


# ============================================================
# Helper: raw predictions
# ============================================================

def exercise_level_split(triplets, n_items, holdout_ratio=0.2):
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=holdout_ratio, random_state=42)
    train_val_triplets = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    test_triplets = triplets[np.isin(triplets[:, 1].astype(int), test_items)]
    all_idx = np.arange(len(train_val_triplets))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.1, random_state=42)
    return (train_val_triplets[train_idx], train_val_triplets[val_idx],
            test_triplets, train_items, test_items)


def predict_text_model_raw(net, triplets, text_embeddings, q_matrix, device="cpu", batch_size=256):
    """Return (y_true, y_pred) arrays for a set of triplets using text model."""
    net.eval()
    net = net.to(device)
    loader = make_text_dataloader(triplets, text_embeddings, q_matrix,
                                  batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, text_emb, knowledge_emb, y in loader:
            pred = net(user_id.to(device), text_emb.to(device), knowledge_emb.to(device))
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred)


def predict_id_model_raw(model, triplets, q_matrix, device="cpu", batch_size=256):
    """Return (y_true, y_pred) arrays for a set of triplets using ID model."""
    model.ncdm_net.eval()
    model.ncdm_net = model.ncdm_net.to(device)
    loader = make_dataloader(triplets, q_matrix, batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, item_id, knowledge_emb, y in loader:
            pred = model.ncdm_net(user_id.to(device), item_id.to(device),
                                  knowledge_emb.to(device))
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred)


# ============================================================
# Experiment 1: Skill Correlation Analysis
# ============================================================

def experiment_1_skill_correlation(data_dir, fig_dir):
    """Analyze inter-skill correlations and PCA on mastery profiles."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Skill Correlation Analysis")
    print("=" * 70)

    mastery_df = pd.read_csv(data_dir / "skill_mastery_profiles_hac50.csv", index_col=0)
    skill_names = list(mastery_df.columns)
    mastery_matrix = mastery_df.values  # (235, 50)
    print(f"Mastery matrix: {mastery_matrix.shape[0]} LLMs x {mastery_matrix.shape[1]} skills")

    # --- Pearson correlation across LLMs ---
    corr_matrix = np.corrcoef(mastery_matrix.T)  # (50, 50)

    # Clustermap heatmap
    fig = plt.figure(figsize=(14, 12))
    plt.close(fig)

    short_names = [s[:25] + "..." if len(s) > 28 else s for s in skill_names]
    corr_df = pd.DataFrame(corr_matrix, index=short_names, columns=short_names)

    g = sns.clustermap(
        corr_df,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        figsize=(14, 12),
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(0.02, 0.82, 0.03, 0.15),
        linewidths=0,
        xticklabels=True, yticklabels=True,
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=5, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=5)
    g.fig.suptitle("Skill Mastery Correlation (50 HAC Clusters, 235 LLMs)", y=1.01, fontsize=13)

    path_corr = fig_dir / "fig_skill_correlation_heatmap.pdf"
    g.savefig(path_corr, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {path_corr}")

    # --- PCA ---
    pca = PCA()
    pca.fit(mastery_matrix)
    explained = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(explained, 0.90) + 1)
    n_95 = int(np.searchsorted(explained, 0.95) + 1)
    print(f"  PCA: {n_90} components for 90% variance, {n_95} for 95%")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(explained) + 1), pca.explained_variance_ratio_,
           alpha=0.5, label="Individual")
    ax.step(range(1, len(explained) + 1), explained, where="mid",
            color="red", linewidth=2, label="Cumulative")
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA on Skill Mastery Profiles (235 LLMs x 50 Skills)")
    ax.legend()
    ax.set_xlim(0.5, 50.5)
    ax.set_ylim(0, 1.05)

    path_pca = fig_dir / "fig_pca_variance.pdf"
    fig.savefig(path_pca, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_pca}")

    # --- Summary stats ---
    upper_tri = corr_matrix[np.triu_indices(50, k=1)]
    results = {
        "n_llms": int(mastery_matrix.shape[0]),
        "n_skills": int(mastery_matrix.shape[1]),
        "correlation_stats": {
            "mean": float(np.mean(upper_tri)),
            "std": float(np.std(upper_tri)),
            "min": float(np.min(upper_tri)),
            "max": float(np.max(upper_tri)),
            "median": float(np.median(upper_tri)),
        },
        "pca": {
            "n_components_90pct": n_90,
            "n_components_95pct": n_95,
            "top5_explained_variance": [float(v) for v in pca.explained_variance_ratio_[:5]],
            "cumulative_explained": [float(v) for v in explained.tolist()],
        },
    }

    path_json = data_dir / "skill_correlation_analysis.json"
    with open(path_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path_json}")

    return results


# ============================================================
# Experiment 2: Difficulty-Stratified Evaluation
# ============================================================

def experiment_2_difficulty_stratified(triplets, text_embeddings, q_matrix,
                                       response_df, n_items, n_llms, n_skills,
                                       args, fig_dir, data_dir):
    """Evaluate text vs ID models stratified by item difficulty."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Difficulty-Stratified Evaluation")
    print("=" * 70)

    seed_everything(42)

    # Exercise-level split
    train_trip, val_trip, test_trip, train_items, test_items = \
        exercise_level_split(triplets, n_items, holdout_ratio=0.2)
    print(f"  Train items: {len(train_items)}, Test items: {len(test_items)}")
    print(f"  Train triplets: {len(train_trip):,}, Val: {len(val_trip):,}, Test: {len(test_trip):,}")

    # Model checkpoint paths
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    text_ckpt = MODEL_DIR / "ncdm_text_conditioned_protocolB.pt"
    id_ckpt = MODEL_DIR / "ncdm_id_protocolB.pt"

    if text_ckpt.exists() and id_ckpt.exists():
        print(f"\n  Loading saved text model from {text_ckpt}")
        net_text = TextConditionedNet(n_skills, n_llms, TEXT_DIM)
        net_text.load_state_dict(torch.load(str(text_ckpt), map_location="cpu"))
        net_text.to(args.device)
        net_text.eval()

        print(f"  Loading saved ID model from {id_ckpt}")
        id_model = NCDM(n_skills, n_items, n_llms)
        id_model.ncdm_net.load_state_dict(torch.load(str(id_ckpt), map_location="cpu"))
        id_model.ncdm_net.to(args.device)
        id_model.ncdm_net.eval()
        print("  Both models loaded from checkpoints.")
    else:
        print("\n  Training TextConditionedNet...")
        train_loader_t = make_text_dataloader(train_trip, text_embeddings, q_matrix, args.batch_size)
        val_loader_t = make_text_dataloader(val_trip, text_embeddings, q_matrix, args.batch_size, shuffle=False)

        net_text = TextConditionedNet(n_skills, n_llms, TEXT_DIM)
        net_text = train_text_model(net_text, train_loader_t, val_loader_t,
                                    epochs=args.epochs, lr=args.lr, device=args.device)

        print("\n  Training ID-based NCDM...")
        train_loader_id = make_dataloader(train_trip, q_matrix, args.batch_size)
        val_loader_id = make_dataloader(val_trip, q_matrix, args.batch_size, shuffle=False)

        id_model = train_id_model(train_loader_id, val_loader_id, n_skills, n_items, n_llms,
                                  epochs=args.epochs, lr=args.lr, device=args.device)

        torch.save(net_text.state_dict(), str(text_ckpt))
        print(f"  Saved text model: {text_ckpt}")
        torch.save(id_model.ncdm_net.state_dict(), str(id_ckpt))
        print(f"  Saved ID model: {id_ckpt}")

    # --- Compute item difficulty ---
    response_matrix = response_df.values
    item_difficulty = {}
    for item_id in test_items:
        item_difficulty[int(item_id)] = float(response_matrix[:, item_id].mean())

    difficulties = np.array([item_difficulty[int(i)] for i in test_items])
    thresholds = np.percentile(difficulties, [33.33, 66.67])

    hard_items = set(test_items[difficulties <= thresholds[0]])
    medium_items = set(test_items[(difficulties > thresholds[0]) & (difficulties <= thresholds[1])])
    easy_items = set(test_items[difficulties > thresholds[1]])

    print(f"  Difficulty buckets: easy={len(easy_items)}, medium={len(medium_items)}, hard={len(hard_items)}")
    print(f"  Thresholds: hard <= {thresholds[0]:.3f}, medium <= {thresholds[1]:.3f}, easy > {thresholds[1]:.3f}")

    # --- Compute AUC per bucket ---
    buckets = {"easy": easy_items, "medium": medium_items, "hard": hard_items}
    results = {"thresholds": [float(thresholds[0]), float(thresholds[1])]}

    for bucket_name, bucket_items in buckets.items():
        mask = np.isin(test_trip[:, 1].astype(int), list(bucket_items))
        bucket_triplets = test_trip[mask]

        if len(bucket_triplets) < 10:
            print(f"    {bucket_name}: too few triplets ({len(bucket_triplets)}), skipping")
            continue

        yt_text, yp_text = predict_text_model_raw(net_text, bucket_triplets,
                                                   text_embeddings, q_matrix, args.device)
        yt_id, yp_id = predict_id_model_raw(id_model, bucket_triplets, q_matrix, args.device)

        if len(np.unique(yt_text)) < 2:
            print(f"    {bucket_name}: only one class in test set, skipping AUC")
            results[bucket_name] = {
                "n_items": len(bucket_items),
                "n_triplets": len(bucket_triplets),
                "mean_difficulty": float(np.mean([item_difficulty[int(i)] for i in bucket_items])),
                "text_auc": None, "id_auc": None,
            }
            continue

        auc_text = roc_auc_score(yt_text, yp_text)
        auc_id = roc_auc_score(yt_id, yp_id)

        results[bucket_name] = {
            "n_items": len(bucket_items),
            "n_triplets": int(len(bucket_triplets)),
            "mean_difficulty": float(np.mean([item_difficulty[int(i)] for i in bucket_items])),
            "text_auc": float(auc_text),
            "id_auc": float(auc_id),
            "auc_gap": float(auc_text - auc_id),
        }
        print(f"    {bucket_name}: text_AUC={auc_text:.4f}, id_AUC={auc_id:.4f}, "
              f"gap={auc_text - auc_id:+.4f} (n={len(bucket_triplets):,})")

    # --- Figure ---
    bucket_names = ["hard", "medium", "easy"]
    text_aucs = [results.get(b, {}).get("text_auc") for b in bucket_names]
    id_aucs = [results.get(b, {}).get("id_auc") for b in bucket_names]

    valid = [(b, t, i) for b, t, i in zip(bucket_names, text_aucs, id_aucs)
             if t is not None and i is not None]

    if valid:
        vb, vt, vi = zip(*valid)
        x = np.arange(len(vb))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - width/2, vt, width, label="Text-Conditioned", color="#2196F3")
        ax.bar(x + width/2, vi, width, label="ID-Based", color="#FF9800")

        ax.set_ylabel("AUC")
        ax.set_title("AUC by Item Difficulty (Protocol B, Exercise Split)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b.capitalize()}\n(n={results[b]['n_items']})" for b in vb])
        ax.legend()
        ax.set_ylim(0.5, 1.0)

        for idx, (t_val, i_val) in enumerate(zip(vt, vi)):
            ax.text(idx - width/2, t_val + 0.005, f"{t_val:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(idx + width/2, i_val + 0.005, f"{i_val:.3f}", ha="center", va="bottom", fontsize=8)

        path_fig = fig_dir / "fig_difficulty_stratified_auc.pdf"
        fig.savefig(path_fig, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {path_fig}")

    path_json = data_dir / "difficulty_stratified_results.json"
    with open(path_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path_json}")

    return results, net_text, id_model, test_trip, test_items, train_items


# ============================================================
# Experiment 3: Routing Accuracy with Ground Truth
# ============================================================

def experiment_3_routing_accuracy(net_text, response_df, test_items, text_embeddings,
                                  q_matrix, n_llms, device, fig_dir, data_dir):
    """Accuracy@k: does top-k contain an LLM that actually got the item right?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Routing Accuracy with Ground Truth")
    print("=" * 70)

    response_matrix = response_df.values
    llm_names = list(response_df.index)
    net_text.eval()
    net_text = net_text.to(device)

    all_llm_ids = torch.arange(n_llms, device=device)

    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(text_embeddings)

    ks = [1, 3, 5, 10]
    item_results = []

    global_acc = response_matrix.mean(axis=1)
    best_global_llm = np.argmax(global_acc)
    print(f"  Global best LLM: {llm_names[best_global_llm]} (acc={global_acc[best_global_llm]:.4f})")

    for item_id in tqdm(test_items, desc="  Routing items"):
        item_id = int(item_id)
        ground_truth = response_matrix[:, item_id]

        text_emb = torch.tensor(text_embeddings[item_id], dtype=torch.float32, device=device)
        text_emb_batch = text_emb.unsqueeze(0).expand(n_llms, -1)
        q_row = torch.tensor(q_matrix[item_id], dtype=torch.float32, device=device)
        q_row_batch = q_row.unsqueeze(0).expand(n_llms, -1)

        with torch.no_grad():
            preds = net_text(all_llm_ids, text_emb_batch, q_row_batch).cpu().numpy()

        ranking = np.argsort(-preds)

        item_res = {"item_id": item_id, "n_correct_llms": int(ground_truth.sum())}
        for k_val in ks:
            top_k = ranking[:k_val]
            hit = int(ground_truth[top_k].max())
            item_res[f"acc_at_{k_val}"] = hit

        n_correct = ground_truth.sum()
        for k_val in ks:
            if n_correct == 0:
                item_res[f"random_at_{k_val}"] = 0.0
            else:
                from scipy.special import comb
                n_total = n_llms
                p_none = comb(n_total - n_correct, k_val, exact=True) / comb(n_total, k_val, exact=True) \
                    if k_val <= n_total - n_correct else 0.0
                item_res[f"random_at_{k_val}"] = 1.0 - p_none

        item_res["strongest_correct"] = int(ground_truth[best_global_llm])
        item_res["oracle"] = int(n_correct > 0)

        item_results.append(item_res)

    # Aggregate
    n_test = len(item_results)
    results = {
        "n_test_items": n_test,
        "n_llms": n_llms,
        "global_best_llm": llm_names[best_global_llm],
    }

    for k_val in ks:
        model_hits = np.mean([r[f"acc_at_{k_val}"] for r in item_results])
        random_hits = np.mean([r[f"random_at_{k_val}"] for r in item_results])
        results[f"text_model_acc_at_{k_val}"] = float(model_hits)
        results[f"random_acc_at_{k_val}"] = float(random_hits)
        print(f"  Accuracy@{k_val}: text_model={model_hits:.4f}, random={random_hits:.4f}")

    strongest_acc = np.mean([r["strongest_correct"] for r in item_results])
    oracle_acc = np.mean([r["oracle"] for r in item_results])
    results["strongest_acc"] = float(strongest_acc)
    results["oracle_acc"] = float(oracle_acc)
    print(f"  Strongest model baseline: {strongest_acc:.4f}")
    print(f"  Oracle: {oracle_acc:.4f}")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5))

    model_vals = [results[f"text_model_acc_at_{k}"] for k in ks]
    random_vals = [results[f"random_acc_at_{k}"] for k in ks]

    ax.plot(ks, model_vals, "o-", linewidth=2, markersize=8, label="Text-Conditioned NCDM", color="#2196F3")
    ax.plot(ks, random_vals, "s--", linewidth=1.5, markersize=6, label="Random Routing", color="#9E9E9E")
    ax.axhline(strongest_acc, color="#FF9800", linestyle=":", linewidth=1.5, label=f"Strongest model")
    ax.axhline(oracle_acc, color="#4CAF50", linestyle="-.", linewidth=1.5, label="Oracle")

    ax.set_xlabel("k (top-k)")
    ax.set_ylabel("Accuracy@k")
    ax.set_title("Routing Accuracy: Does Top-k Contain a Correct LLM?")
    ax.set_xticks(ks)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    path_fig = fig_dir / "fig_routing_accuracy.pdf"
    fig.savefig(path_fig, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_fig}")

    path_json = data_dir / "routing_accuracy_results.json"
    with open(path_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path_json}")

    return results


# ============================================================
# Experiment 4: Per-Skill AUC Breakdown
# ============================================================

def experiment_4_per_skill_breakdown(net_text, id_model, test_trip, text_embeddings,
                                     q_matrix, skill_names, device, fig_dir, data_dir):
    """Per-skill AUC comparison: text-conditioned vs ID-based."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Per-Skill AUC Breakdown")
    print("=" * 70)

    yt_text, yp_text = predict_text_model_raw(net_text, test_trip, text_embeddings,
                                               q_matrix, device)
    yt_id, yp_id = predict_id_model_raw(id_model, test_trip, q_matrix, device)

    item_indices = test_trip[:, 1].astype(int)

    results = {}
    for skill_idx, skill_name in enumerate(skill_names):
        mask = q_matrix[item_indices, skill_idx] == 1
        if mask.sum() < 20:
            continue

        yt_s = yt_text[mask]
        if len(np.unique(yt_s)) < 2:
            continue

        auc_text = roc_auc_score(yt_s, yp_text[mask])
        auc_id = roc_auc_score(yt_id[mask], yp_id[mask])

        results[skill_name] = {
            "skill_idx": skill_idx,
            "n_triplets": int(mask.sum()),
            "n_items": int(len(set(item_indices[mask]))),
            "text_auc": float(auc_text),
            "id_auc": float(auc_id),
            "auc_gap": float(auc_text - auc_id),
        }

    sorted_skills = sorted(results.items(), key=lambda x: x[1]["auc_gap"], reverse=True)

    print(f"\n  Skills evaluated: {len(results)}/{len(skill_names)}")
    print(f"\n  Top 5 skills with largest text model advantage:")
    for name, r in sorted_skills[:5]:
        print(f"    {name[:40]:<42} text={r['text_auc']:.4f}  id={r['id_auc']:.4f}  gap={r['auc_gap']:+.4f}")
    print(f"\n  Bottom 5 skills (ID model advantage):")
    for name, r in sorted_skills[-5:]:
        print(f"    {name[:40]:<42} text={r['text_auc']:.4f}  id={r['id_auc']:.4f}  gap={r['auc_gap']:+.4f}")

    # --- Figure ---
    if sorted_skills:
        names = [s[0] for s in sorted_skills]
        gaps = [s[1]["auc_gap"] for s in sorted_skills]
        text_aucs = [s[1]["text_auc"] for s in sorted_skills]
        id_aucs = [s[1]["id_auc"] for s in sorted_skills]

        short_names = [n[:30] + "..." if len(n) > 33 else n for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(names) * 0.28)),
                                 gridspec_kw={"width_ratios": [1, 1.5]})

        colors = ["#2196F3" if g >= 0 else "#FF9800" for g in gaps]
        axes[0].barh(range(len(gaps)), gaps, color=colors, height=0.7)
        axes[0].set_yticks(range(len(gaps)))
        axes[0].set_yticklabels(short_names, fontsize=6)
        axes[0].set_xlabel("AUC Gap (Text - ID)")
        axes[0].set_title("Text Model Advantage by Skill")
        axes[0].axvline(0, color="black", linewidth=0.5)
        axes[0].invert_yaxis()

        y = np.arange(len(names))
        axes[1].scatter(text_aucs, y, color="#2196F3", marker="o", s=30, label="Text", zorder=3)
        axes[1].scatter(id_aucs, y, color="#FF9800", marker="s", s=30, label="ID", zorder=3)
        for idx in range(len(names)):
            axes[1].plot([text_aucs[idx], id_aucs[idx]], [idx, idx],
                         color="gray", linewidth=0.5, zorder=1)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(short_names, fontsize=6)
        axes[1].set_xlabel("AUC")
        axes[1].set_title("Per-Skill AUC Comparison")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].invert_yaxis()

        fig.tight_layout()
        path_fig = fig_dir / "fig_per_skill_auc_comparison.pdf"
        fig.savefig(path_fig, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {path_fig}")

    # Save JSON
    output = {
        "n_skills_evaluated": len(results),
        "mean_text_auc": float(np.mean([r["text_auc"] for r in results.values()])),
        "mean_id_auc": float(np.mean([r["id_auc"] for r in results.values()])),
        "mean_auc_gap": float(np.mean([r["auc_gap"] for r in results.values()])),
        "per_skill": results,
    }

    path_json = data_dir / "per_skill_auc_results.json"
    with open(path_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path_json}")

    return output


# ============================================================
# Experiment 5: Q-Row Approximation Quality
# ============================================================

def experiment_5_qrow_approximation(net_text, text_embeddings, q_matrix,
                                     test_trip, train_items, test_items,
                                     device, data_dir):
    """Evaluate Q-row approximation via nearest-neighbor lookup."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Q-Row Approximation Quality")
    print("=" * 70)

    train_embeddings = text_embeddings[train_items]
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn_model.fit(train_embeddings)

    test_embeddings = text_embeddings[test_items]
    distances, indices = nn_model.kneighbors(test_embeddings)

    jaccard_scores = []
    exact_matches = 0

    for i, test_item in enumerate(test_items):
        true_q = q_matrix[test_item]
        nn_train_item = train_items[indices[i, 0]]
        approx_q = q_matrix[nn_train_item]

        intersection = np.sum((true_q == 1) & (approx_q == 1))
        union = np.sum((true_q == 1) | (approx_q == 1))
        jaccard = intersection / union if union > 0 else 1.0
        jaccard_scores.append(jaccard)

        if np.array_equal(true_q, approx_q):
            exact_matches += 1

    jaccard_scores = np.array(jaccard_scores)
    print(f"  Jaccard similarity: mean={jaccard_scores.mean():.4f}, "
          f"median={np.median(jaccard_scores):.4f}, "
          f"std={jaccard_scores.std():.4f}")
    print(f"  Exact Q-row matches: {exact_matches}/{len(test_items)} "
          f"({100*exact_matches/len(test_items):.1f}%)")
    print(f"  Cosine distance to NN: mean={distances.mean():.4f}, "
          f"median={np.median(distances):.4f}")

    approx_q_matrix = q_matrix.copy()
    for i, test_item in enumerate(test_items):
        nn_train_item = train_items[indices[i, 0]]
        approx_q_matrix[test_item] = q_matrix[nn_train_item]

    yt_true, yp_true = predict_text_model_raw(net_text, test_trip, text_embeddings,
                                               q_matrix, device)
    yt_nn, yp_nn = predict_text_model_raw(net_text, test_trip, text_embeddings,
                                           approx_q_matrix, device)

    auc_true = roc_auc_score(yt_true, yp_true)
    auc_nn = roc_auc_score(yt_nn, yp_nn)
    acc_true = accuracy_score(yt_true, (yp_true >= 0.5).astype(int))
    acc_nn = accuracy_score(yt_nn, (yp_nn >= 0.5).astype(int))

    print(f"\n  With true Q-rows:   AUC={auc_true:.4f}, Acc={acc_true:.4f}")
    print(f"  With NN Q-rows:     AUC={auc_nn:.4f}, Acc={acc_nn:.4f}")
    print(f"  AUC degradation:    {auc_true - auc_nn:+.4f}")

    results = {
        "n_test_items": len(test_items),
        "n_train_items": len(train_items),
        "jaccard_similarity": {
            "mean": float(jaccard_scores.mean()),
            "median": float(np.median(jaccard_scores)),
            "std": float(jaccard_scores.std()),
            "min": float(jaccard_scores.min()),
            "max": float(jaccard_scores.max()),
        },
        "exact_q_row_matches": exact_matches,
        "exact_match_rate": float(exact_matches / len(test_items)),
        "cosine_distance_to_nn": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "std": float(distances.std()),
        },
        "model_with_true_qrows": {
            "auc": float(auc_true),
            "accuracy": float(acc_true),
        },
        "model_with_nn_qrows": {
            "auc": float(auc_nn),
            "accuracy": float(acc_nn),
        },
        "auc_degradation": float(auc_true - auc_nn),
    }

    path_json = data_dir / "qrow_approximation_results.json"
    with open(path_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path_json}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Extended Experiments for CDM Evaluation")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps", help="cpu, cuda, or mps")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    args.device = resolve_device(args.device)
    print(f"Device: {args.device}")

    # ── Load data ──
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        DATA_DIR / "response_matrix.csv"
    )
    print(f"Response matrix: {n_llms} LLMs x {n_items} items")

    q_matrix, skill_names = load_q_matrix(DATA_DIR / "q_matrix_hac50.csv")
    n_skills = q_matrix.shape[1]
    print(f"Q-matrix (HAC-50): {n_items} items x {n_skills} skills")

    triplets = build_triplets(response_df)
    print(f"Total triplets: {len(triplets):,}")

    # Load text embeddings
    data = np.load(DATA_DIR / "item_text_embeddings.npz")
    text_embeddings = data["embeddings"]
    print(f"Text embeddings: {text_embeddings.shape}")

    # ================================================================
    # Run experiments
    # ================================================================

    summary = {}

    # --- Experiment 1 ---
    res1 = experiment_1_skill_correlation(DATA_DIR, FIG_DIR)
    summary["experiment_1"] = {
        "mean_correlation": res1["correlation_stats"]["mean"],
        "pca_90pct_components": res1["pca"]["n_components_90pct"],
    }

    # --- Experiment 2 (also trains models for 3, 4, 5) ---
    res2, net_text, id_model, test_trip, test_items, train_items = \
        experiment_2_difficulty_stratified(
            triplets, text_embeddings, q_matrix, response_df,
            n_items, n_llms, n_skills, args, FIG_DIR, DATA_DIR
        )
    summary["experiment_2"] = {
        b: {"text_auc": res2[b].get("text_auc"), "id_auc": res2[b].get("id_auc")}
        for b in ["easy", "medium", "hard"] if b in res2
    }

    # --- Experiment 3 ---
    res3 = experiment_3_routing_accuracy(
        net_text, response_df, test_items, text_embeddings,
        q_matrix, n_llms, args.device, FIG_DIR, DATA_DIR
    )
    summary["experiment_3"] = {
        f"acc_at_{k}": res3[f"text_model_acc_at_{k}"] for k in [1, 3, 5, 10]
    }

    # --- Experiment 4 ---
    res4 = experiment_4_per_skill_breakdown(
        net_text, id_model, test_trip, text_embeddings,
        q_matrix, skill_names, args.device, FIG_DIR, DATA_DIR
    )
    summary["experiment_4"] = {
        "mean_text_auc": res4["mean_text_auc"],
        "mean_id_auc": res4["mean_id_auc"],
        "mean_auc_gap": res4["mean_auc_gap"],
    }

    # --- Experiment 5 ---
    res5 = experiment_5_qrow_approximation(
        net_text, text_embeddings, q_matrix,
        test_trip, train_items, test_items,
        args.device, DATA_DIR
    )
    summary["experiment_5"] = {
        "mean_jaccard": res5["jaccard_similarity"]["mean"],
        "auc_with_true_qrows": res5["model_with_true_qrows"]["auc"],
        "auc_with_nn_qrows": res5["model_with_nn_qrows"]["auc"],
        "auc_degradation": res5["auc_degradation"],
    }

    # ================================================================
    # Final Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("EXTENDED EXPERIMENTS SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
