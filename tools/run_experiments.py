"""Run extended experiments for CDM evaluation.

Five experiments:
  1. Skill Correlation Analysis
  2. Difficulty-Stratified Evaluation
  3. Routing Accuracy with Ground Truth
  4. Per-Skill AUC Breakdown
  5. Q-Row Approximation Quality

Usage:
    python tools/run_experiments.py
    python tools/run_experiments.py device=cpu model.epochs=5
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import hydra
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


TEXT_DIM = 768
K = 50


def predict_text_model_raw(net, dataloader, device="cpu"):
    """Return raw (y_true, y_pred) arrays from a text model."""
    net.eval()
    net = net.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, text_emb, knowledge_emb, y in dataloader:
            pred = net(
                user_id.to(device), text_emb.to(device), knowledge_emb.to(device)
            )
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred)


def predict_id_model_raw(model, dataloader, device="cpu"):
    """Return raw (y_true, y_pred) arrays from an ID-based model."""
    model.ncdm_net.eval()
    model.ncdm_net = model.ncdm_net.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for user_id, item_id, knowledge_emb, y in dataloader:
            pred = model.ncdm_net(
                user_id.to(device), item_id.to(device), knowledge_emb.to(device)
            )
            y_pred.extend(pred.cpu().tolist())
            y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_dataloader, make_text_dataloader
    from cdmeval.data.response_matrix import build_triplets, load_q_matrix, load_response_matrix
    from cdmeval.evaluation.metrics import eval_id_model, eval_text_model
    from cdmeval.evaluation.training import train_id_model, train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    model_dir = Path(cfg.paths.models)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # Load data
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    q_matrix, skill_cols = load_q_matrix(data_dir / f"q_matrix_hac{K}.csv")
    n_skills = q_matrix.shape[1]
    triplets = build_triplets(response_df)

    text_emb_data = np.load(data_dir / "item_text_embeddings.npz")
    text_embeddings = text_emb_data["embeddings"]

    bs = cfg.model.batch_size

    # Exercise-level split
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    train_val_triplets = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    test_triplets = triplets[np.isin(triplets[:, 1].astype(int), test_items)]
    tidx = np.arange(len(train_val_triplets))
    train_idx, val_idx = train_test_split(tidx, test_size=0.1, random_state=42)
    train_trip = train_val_triplets[train_idx]
    val_trip = train_val_triplets[val_idx]

    print(f"Data: {n_llms} LLMs, {n_items} items, {n_skills} skills")
    print(f"Exercise split: {len(train_items)} train items, {len(test_items)} test items")

    # Train models
    print("\nTraining text-conditioned model...")
    tr_txt = make_text_dataloader(train_trip, text_embeddings, q_matrix, bs, shuffle=True)
    va_txt = make_text_dataloader(val_trip, text_embeddings, q_matrix, bs, shuffle=False)
    te_txt = make_text_dataloader(test_triplets, text_embeddings, q_matrix, bs, shuffle=False)
    net = TextConditionedNet(n_skills, n_llms, TEXT_DIM)
    net = train_text_model(net, tr_txt, va_txt,
                           epochs=cfg.model.epochs, lr=cfg.model.lr, device=device)

    print("\nTraining ID-based model...")
    tr_id = make_dataloader(train_trip, q_matrix, bs, shuffle=True)
    va_id = make_dataloader(val_trip, q_matrix, bs, shuffle=False)
    te_id = make_dataloader(test_triplets, q_matrix, bs, shuffle=False)
    id_model = train_id_model(tr_id, va_id, n_skills, n_items, n_llms,
                              epochs=cfg.model.epochs, lr=cfg.model.lr, device=device)

    # ── Experiment 1: Skill Correlation Analysis ──
    print("\n" + "=" * 60 + "\nExperiment 1: Skill Correlation Analysis\n" + "=" * 60)
    mastery = pd.read_csv(data_dir / f"skill_mastery_profiles_hac{K}.csv", index_col=0)
    corr = mastery.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.clustermap(corr, cmap="RdBu_r", vmin=-1, vmax=1, figsize=(12, 10))
    plt.savefig(fig_dir / "fig_skill_correlation_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved fig_skill_correlation_heatmap.pdf")

    pca = PCA()
    pca.fit(mastery.values)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumvar, 0.9) + 1)
    print(f"  PCA: {n90} components for 90% variance")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", markersize=3)
    ax.axhline(0.9, ls="--", color="red", label="90%")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative explained variance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_pca_variance.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_pca_variance.pdf")

    with open(data_dir / "skill_correlation_analysis.json", "w") as f:
        json.dump({"n_components_90pct": n90, "top_eigenvalues": pca.explained_variance_ratio_[:10].tolist()}, f, indent=2)

    # ── Experiment 2: Difficulty-Stratified Evaluation ──
    print("\n" + "=" * 60 + "\nExperiment 2: Difficulty-Stratified Evaluation\n" + "=" * 60)
    response_vals = response_df.values
    test_item_difficulty = response_vals[:, test_items.astype(int)].mean(axis=0)
    thresholds = np.percentile(test_item_difficulty, [33.3, 66.7])
    buckets = {"hard": test_items[test_item_difficulty <= thresholds[0]],
               "medium": test_items[(test_item_difficulty > thresholds[0]) & (test_item_difficulty <= thresholds[1])],
               "easy": test_items[test_item_difficulty > thresholds[1]]}

    y_true_txt, y_pred_txt = predict_text_model_raw(net, te_txt, device)
    y_true_id, y_pred_id = predict_id_model_raw(id_model, te_id, device)
    test_item_ids_txt = test_triplets[:, 1].astype(int)

    strat_results = {}
    for bname, bitems in buckets.items():
        mask = np.isin(test_item_ids_txt, bitems)
        if mask.sum() > 0 and len(np.unique(y_true_txt[mask])) > 1:
            auc_t = roc_auc_score(y_true_txt[mask], y_pred_txt[mask])
            auc_i = roc_auc_score(y_true_id[mask], y_pred_id[mask])
            strat_results[bname] = {"text_auc": float(auc_t), "id_auc": float(auc_i), "n_triplets": int(mask.sum())}
            print(f"  {bname}: text AUC={auc_t:.4f}, ID AUC={auc_i:.4f} ({mask.sum()} triplets)")

    fig, ax = plt.subplots(figsize=(6, 4))
    bnames = list(strat_results.keys())
    x = range(len(bnames))
    ax.bar([i - 0.15 for i in x], [strat_results[b]["text_auc"] for b in bnames], 0.3, label="Text", color="#4C72B0")
    ax.bar([i + 0.15 for i in x], [strat_results[b]["id_auc"] for b in bnames], 0.3, label="ID", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bnames)
    ax.set_ylabel("AUC")
    ax.set_title("AUC by Difficulty Bucket (Exercise-Level Split)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_difficulty_stratified_auc.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_difficulty_stratified_auc.pdf")

    with open(data_dir / "difficulty_stratified_results.json", "w") as f:
        json.dump(strat_results, f, indent=2)

    # ── Experiment 3: Routing Accuracy with Ground Truth ──
    print("\n" + "=" * 60 + "\nExperiment 3: Routing Accuracy\n" + "=" * 60)
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    accs_at = {1: 0, 3: 0, 5: 0}
    total = 0

    for item_idx in tqdm(test_items, desc="Routing"):
        gt = response_vals[:, int(item_idx)]
        if gt.sum() == 0:
            continue

        emb = torch.tensor(text_embeddings[int(item_idx)], dtype=torch.float32, device=device)
        emb_batch = emb.unsqueeze(0).expand(n_llms, -1)
        q_row = torch.tensor(q_matrix[int(item_idx)], dtype=torch.float32, device=device)
        q_batch = q_row.unsqueeze(0).expand(n_llms, -1)

        with torch.no_grad():
            preds = net(all_llm_ids, emb_batch, q_batch).cpu().numpy()

        ranking = np.argsort(-preds)
        total += 1
        for k in accs_at:
            if gt[ranking[:k]].sum() > 0:
                accs_at[k] += 1

    routing_results = {f"accuracy@{k}": accs_at[k] / total for k in accs_at}
    routing_results["n_items"] = total
    best_llm = response_vals.mean(axis=1).argmax()
    majority_correct = sum(response_vals[best_llm, int(i)] for i in test_items if response_vals[:, int(i)].sum() > 0)
    routing_results["majority_baseline"] = majority_correct / total
    print(f"  Routing: acc@1={routing_results['accuracy@1']:.4f}, acc@3={routing_results['accuracy@3']:.4f}, acc@5={routing_results['accuracy@5']:.4f}")
    print(f"  Majority baseline: {routing_results['majority_baseline']:.4f}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ks = [1, 3, 5]
    vals = [routing_results[f"accuracy@{k}"] for k in ks]
    ax.bar(range(len(ks)), vals, color="#4C72B0", tick_label=[f"@{k}" for k in ks])
    ax.axhline(routing_results["majority_baseline"], ls="--", color="red", label="Majority")
    ax.set_ylabel("Accuracy")
    ax.set_title("Routing Accuracy (Cold-Start Items)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_routing_accuracy.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_routing_accuracy.pdf")

    with open(data_dir / "routing_accuracy_results.json", "w") as f:
        json.dump(routing_results, f, indent=2)

    # ── Experiment 4: Per-Skill AUC Breakdown ──
    print("\n" + "=" * 60 + "\nExperiment 4: Per-Skill AUC Breakdown\n" + "=" * 60)
    per_skill = {}
    for si, sname in enumerate(skill_cols):
        mask = q_matrix[test_triplets[:, 1].astype(int), si] == 1
        if mask.sum() < 50 or len(np.unique(y_true_txt[mask])) < 2:
            continue
        auc_t = roc_auc_score(y_true_txt[mask], y_pred_txt[mask])
        auc_i = roc_auc_score(y_true_id[mask], y_pred_id[mask])
        per_skill[sname] = {"text_auc": float(auc_t), "id_auc": float(auc_i), "n": int(mask.sum())}

    if per_skill:
        ps_df = pd.DataFrame(per_skill).T
        ps_df["delta"] = ps_df["text_auc"] - ps_df["id_auc"]
        ps_df = ps_df.sort_values("delta", ascending=False)
        print(f"  Top 5 skills (text advantage):")
        for name, row in ps_df.head(5).iterrows():
            print(f"    {name}: delta={row['delta']:.4f}")

        fig, ax = plt.subplots(figsize=(8, max(4, len(ps_df) * 0.25)))
        y_pos = range(len(ps_df))
        ax.barh(y_pos, ps_df["text_auc"], 0.35, label="Text", color="#4C72B0", alpha=0.8)
        ax.barh([y + 0.35 for y in y_pos], ps_df["id_auc"], 0.35, label="ID", color="#DD8452", alpha=0.8)
        ax.set_yticks([y + 0.175 for y in y_pos])
        ax.set_yticklabels(ps_df.index, fontsize=6)
        ax.set_xlabel("AUC")
        ax.legend()
        ax.set_title("Per-Skill AUC: Text vs ID Model")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_per_skill_auc_comparison.pdf", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fig_per_skill_auc_comparison.pdf")

    with open(data_dir / "per_skill_auc_results.json", "w") as f:
        json.dump(per_skill, f, indent=2)

    # ── Experiment 5: Q-Row Approximation Quality ──
    print("\n" + "=" * 60 + "\nExperiment 5: Q-Row Approximation Quality\n" + "=" * 60)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(text_embeddings[train_items])

    jaccards = []
    for ti in test_items:
        dists, indices = nn.kneighbors(text_embeddings[int(ti):int(ti) + 1])
        nn_item = train_items[indices[0, 0]]
        true_q = q_matrix[int(ti)]
        approx_q = q_matrix[int(nn_item)]
        inter = np.logical_and(true_q, approx_q).sum()
        union = np.logical_or(true_q, approx_q).sum()
        jaccards.append(inter / union if union > 0 else 1.0)

    qrow_results = {
        "mean_jaccard": float(np.mean(jaccards)),
        "median_jaccard": float(np.median(jaccards)),
        "std_jaccard": float(np.std(jaccards)),
    }
    print(f"  Q-row Jaccard: mean={qrow_results['mean_jaccard']:.4f}, median={qrow_results['median_jaccard']:.4f}")

    with open(data_dir / "qrow_approximation_results.json", "w") as f:
        json.dump(qrow_results, f, indent=2)

    print("\n" + "=" * 60 + "\nAll experiments complete.\n" + "=" * 60)


if __name__ == "__main__":
    main()
