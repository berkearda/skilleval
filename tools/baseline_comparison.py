"""Compare routing baselines against CDM-based routing.

Usage:
    python tools/baseline_comparison.py device=mps
    python tools/baseline_comparison.py device=mps model.epochs=1
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_text_dataloader
    from cdmeval.data.response_matrix import build_triplets, load_q_matrix, load_response_matrix
    from cdmeval.evaluation.baselines import (
        cdm_router,
        evaluate_rankings,
        irt_2pl_router,
        majority_router,
        random_router,
        sbert_nearest_neighbor_router,
        text_similarity_router,
    )
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    K = cfg.skills.n_clusters
    print(f"Device: {device}")

    # ── Load data ──
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    q_matrix, skill_cols = load_q_matrix(data_dir / f"q_matrix_hac{K}.csv")
    n_skills = q_matrix.shape[1]
    triplets = build_triplets(response_df)
    response_vals = response_df.values

    text_embeddings = np.load(data_dir / "item_text_embeddings.npz")["embeddings"]
    text_dim = text_embeddings.shape[1]

    print(f"Data: {n_llms} LLMs, {n_items} items, {n_skills} skills")

    # ── Protocol B split (exercise-level, same as run_experiments.py) ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(
        all_items, test_size=0.2, random_state=42
    )
    print(f"Split: {len(train_items)} train items, {len(test_items)} test items")

    ks = [1, 3, 5, 10]
    results = {}

    # ── 1. Random baseline ──
    print("\n" + "=" * 60)
    print("Baseline 1: Random Router")
    print("=" * 60)
    rankings = random_router(test_items, n_llms)
    results["Random"] = evaluate_rankings(rankings, response_vals, ks)
    results["Random"]["type"] = "naive"
    print(f"  Acc@1={results['Random']['accuracy@1']:.4f}")

    # ── 2. Majority baseline ──
    print("\n" + "=" * 60)
    print("Baseline 2: Majority Router (best LLM)")
    print("=" * 60)
    rankings = majority_router(test_items, response_vals, train_items)
    results["Majority (best LLM)"] = evaluate_rankings(rankings, response_vals, ks)
    results["Majority (best LLM)"]["type"] = "naive"
    print(f"  Acc@1={results['Majority (best LLM)']['accuracy@1']:.4f}")

    # ── 3. SBERT k-NN ──
    print("\n" + "=" * 60)
    print("Baseline 3: SBERT k-NN Router (k=5)")
    print("=" * 60)
    rankings = sbert_nearest_neighbor_router(
        test_items, train_items, text_embeddings, response_vals, k=5
    )
    results["SBERT k-NN (k=5)"] = evaluate_rankings(rankings, response_vals, ks)
    results["SBERT k-NN (k=5)"]["type"] = "embedding"
    print(f"  Acc@1={results['SBERT k-NN (k=5)']['accuracy@1']:.4f}")

    # ── 4. Text similarity (weighted k-NN) ──
    print("\n" + "=" * 60)
    print("Baseline 4: Text Similarity Router (k=5)")
    print("=" * 60)
    rankings = text_similarity_router(
        test_items, train_items, text_embeddings, response_vals, k=5
    )
    results["Text similarity (k=5)"] = evaluate_rankings(rankings, response_vals, ks)
    results["Text similarity (k=5)"]["type"] = "embedding"
    print(f"  Acc@1={results['Text similarity (k=5)']['accuracy@1']:.4f}")

    # ── 5. IRT 2PL ──
    print("\n" + "=" * 60)
    print("Baseline 5: IRT 2PL Router")
    print("=" * 60)
    rankings = irt_2pl_router(
        response_vals, train_items, test_items, n_llms, n_items,
        epochs=cfg.model.epochs, lr=cfg.model.lr,
        batch_size=cfg.model.batch_size, device=device,
    )
    results["IRT 2PL"] = evaluate_rankings(rankings, response_vals, ks)
    results["IRT 2PL"]["type"] = "psychometric"
    print(f"  Acc@1={results['IRT 2PL']['accuracy@1']:.4f}")

    # ── 6. CDMEval (ours) ──
    print("\n" + "=" * 60)
    print("CDMEval: Text-Conditioned NCDM Router")
    print("=" * 60)

    # Train text-conditioned model (Protocol B)
    train_val_trip = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    tv_idx = np.arange(len(train_val_trip))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)

    bs = cfg.model.batch_size
    train_loader = make_text_dataloader(
        train_val_trip[tr_idx], text_embeddings, q_matrix, bs, shuffle=True
    )
    val_loader = make_text_dataloader(
        train_val_trip[va_idx], text_embeddings, q_matrix, bs, shuffle=False
    )

    net = TextConditionedNet(n_skills, n_llms, text_dim)
    net = train_text_model(
        net, train_loader, val_loader,
        epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
    )

    rankings = cdm_router(net, test_items, text_embeddings, q_matrix, n_llms, device)
    results["CDMEval (ours)"] = evaluate_rankings(rankings, response_vals, ks)
    results["CDMEval (ours)"]["type"] = "CDM"
    print(f"  Acc@1={results['CDMEval (ours)']['accuracy@1']:.4f}")

    # ── Summary table ──
    print("\n" + "=" * 80)
    print("ROUTING BASELINE COMPARISON")
    print("=" * 80)
    header = f"{'Method':<28} {'Type':<14} {'@1':>7} {'@3':>7} {'@5':>7} {'@10':>7}"
    print(header)
    print("-" * 80)

    order = [
        "Random",
        "Majority (best LLM)",
        "SBERT k-NN (k=5)",
        "Text similarity (k=5)",
        "IRT 2PL",
        "CDMEval (ours)",
    ]
    for name in order:
        r = results[name]
        print(
            f"{name:<28} {r['type']:<14} "
            f"{r['accuracy@1']:>7.4f} {r['accuracy@3']:>7.4f} "
            f"{r['accuracy@5']:>7.4f} {r['accuracy@10']:>7.4f}"
        )

    # ── Save results ──
    out_path = data_dir / "baseline_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── Bar chart ──
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    names = order
    acc1_vals = [results[n]["accuracy@1"] for n in names]
    types = [results[n]["type"] for n in names]

    type_colors = {
        "naive": "#AAAAAA",
        "embedding": "#DD8452",
        "psychometric": "#8172B2",
        "CDM": "#4C72B0",
    }
    colors = [type_colors[t] for t in types]

    bars = ax.bar(range(len(names)), acc1_vals, color=colors, edgecolor="white", linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, acc1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Routing Accuracy @ 1")
    ax.set_title("Routing Baseline Comparison (529 held-out items)")
    ax.set_ylim(0, max(acc1_vals) * 1.15)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=type_colors[t], label=t.capitalize())
        for t in ["naive", "embedding", "psychometric", "CDM"]
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

    plt.tight_layout()
    fig_path = fig_dir / "fig_baseline_comparison.pdf"
    fig.savefig(fig_path, **SAVE_KW)
    plt.close()
    print(f"Saved {fig_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
