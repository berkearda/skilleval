"""Cost-constrained routing analysis.

Sweeps budget ceilings (parameter count) and computes routing accuracy,
then generates a cost-accuracy scatter plot.

Usage:
    python tools/cost_analysis.py
    python tools/cost_analysis.py device=mps model.epochs=1
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


K = 50


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_text_dataloader
    from cdmeval.data.response_matrix import build_triplets, load_q_matrix, load_response_matrix
    from cdmeval.evaluation.cost_analysis import (
        cost_constrained_routing_curve,
        get_model_sizes,
        plot_routing_curve,
    )
    from cdmeval.evaluation.model_scatter import plot_cost_accuracy_scatter
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # ── Load data ──
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    q_matrix, skill_cols = load_q_matrix(data_dir / f"q_matrix_hac{K}.csv")
    n_skills = q_matrix.shape[1]
    triplets = build_triplets(response_df)
    response_vals = response_df.values

    text_emb_data = np.load(data_dir / "item_text_embeddings.npz")
    text_embeddings = text_emb_data["embeddings"]
    text_dim = text_embeddings.shape[1]

    print(f"Data: {n_llms} LLMs, {n_items} items, {n_skills} skills")

    # ── Exercise-level split (same as run_experiments.py) ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    train_val_triplets = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    test_triplets = triplets[np.isin(triplets[:, 1].astype(int), test_items)]
    tidx = np.arange(len(train_val_triplets))
    train_idx, val_idx = train_test_split(tidx, test_size=0.1, random_state=42)
    train_trip = train_val_triplets[train_idx]
    val_trip = train_val_triplets[val_idx]

    print(f"Exercise split: {len(train_items)} train items, {len(test_items)} test items")

    # ── Train text-conditioned model (Protocol B) ──
    print("\nTraining text-conditioned model on Protocol B train split...")
    bs = cfg.model.batch_size
    tr_txt = make_text_dataloader(train_trip, text_embeddings, q_matrix, bs, shuffle=True)
    va_txt = make_text_dataloader(val_trip, text_embeddings, q_matrix, bs, shuffle=False)

    net = TextConditionedNet(n_skills, n_llms, text_dim)
    net = train_text_model(net, tr_txt, va_txt,
                           epochs=cfg.model.epochs, lr=cfg.model.lr, device=device)

    # ── Extract model sizes ──
    print("\n" + "=" * 60)
    print("Extracting model sizes from names")
    print("=" * 60)
    sizes, valid_mask = get_model_sizes(llm_names)

    # ── Cost-constrained routing curve ──
    print("\n" + "=" * 60)
    print("Cost-Constrained Routing Curve")
    print("=" * 60)
    curve_df = cost_constrained_routing_curve(
        net, text_embeddings, q_matrix, response_vals,
        test_items, llm_names, sizes, valid_mask, device,
    )

    # Save results
    out_json = data_dir / "cost_routing_results.json"
    curve_df.to_json(out_json, orient="records", indent=2)
    print(f"\nSaved {out_json}")

    # ── Plot routing curve ──
    plot_routing_curve(curve_df, fig_dir)

    # ── Plot cost-accuracy scatter ──
    print("\n" + "=" * 60)
    print("Cost-Accuracy Scatter Plot")
    print("=" * 60)
    plot_cost_accuracy_scatter(
        llm_names, sizes, valid_mask, response_vals,
        net, text_embeddings, q_matrix, test_items, device, fig_dir,
    )

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Budget':>8} {'Pool':>6} {'Acc@1':>8} {'Acc@3':>8} {'Acc@5':>8} {'Majority':>10} {'Oracle':>8}")
    print("-" * 62)
    for _, row in curve_df.iterrows():
        print(
            f"{row['budget']:8.1f} {int(row['pool_size']):6d} "
            f"{row['acc1']:8.3f} {row['acc3']:8.3f} {row['acc5']:8.3f} "
            f"{row['majority']:10.3f} {row['oracle']:8.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
