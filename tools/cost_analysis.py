"""Cost-constrained routing analysis.

Sweeps budget ceilings (parameter count AND API price) and computes
routing accuracy, then generates cost-accuracy figures.

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
        cost_constrained_routing_by_price,
        cost_constrained_routing_curve,
        get_model_sizes,
        plot_cost_accuracy_scatter_price,
        plot_routing_curve,
        plot_routing_curve_price,
    )
    from cdmeval.evaluation.model_scatter import plot_cost_accuracy_scatter
    from cdmeval.evaluation.pricing import get_model_pricing
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

    # ══════════════════════════════════════════════════════════════
    #  Part 1: Parameter-count based (original)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: Parameter-Count Based Routing")
    print("=" * 70)

    sizes, valid_mask = get_model_sizes(llm_names)

    curve_df = cost_constrained_routing_curve(
        net, text_embeddings, q_matrix, response_vals,
        test_items, llm_names, sizes, valid_mask, device,
    )

    curve_df.to_json(data_dir / "cost_routing_results.json", orient="records", indent=2)
    plot_routing_curve(curve_df, fig_dir)
    plot_cost_accuracy_scatter(
        llm_names, sizes, valid_mask, response_vals,
        net, text_embeddings, q_matrix, test_items, device, fig_dir,
    )

    print(f"\n{'Budget(B)':>10} {'Pool':>6} {'Acc@1':>8} {'Acc@5':>8} {'Strongest':>10}")
    print("-" * 50)
    for _, row in curve_df.iterrows():
        print(
            f"{row['budget']:10.1f} {int(row['pool_size']):6d} "
            f"{row['acc1']:8.3f} {row['acc5']:8.3f} {row['strongest']:10.3f}"
        )

    # ══════════════════════════════════════════════════════════════
    #  Part 2: API Price based (new)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: API Price Based Routing ($/M output tokens)")
    print("=" * 70)

    pricing_df = get_model_pricing(llm_names)
    prices = pricing_df["price_per_m_tokens"].values

    # Save pricing
    pricing_df.to_csv(data_dir / "llm_pricing.csv", index=False)

    price_curve_df = cost_constrained_routing_by_price(
        net, text_embeddings, q_matrix, response_vals,
        test_items, prices, device,
    )

    price_curve_df.to_json(
        data_dir / "cost_routing_results_price.json", orient="records", indent=2
    )
    plot_routing_curve_price(price_curve_df, fig_dir)
    plot_cost_accuracy_scatter_price(
        llm_names, prices, response_vals,
        net, text_embeddings, q_matrix, test_items, device, fig_dir,
    )

    print(f"\n{'Budget($/M)':>12} {'Pool':>6} {'Acc@1':>8} {'Acc@5':>8} {'Strongest':>10}")
    print("-" * 52)
    for _, row in price_curve_df.iterrows():
        print(
            f"${row['budget']:<11.2f} {int(row['pool_size']):6d} "
            f"{row['acc1']:8.3f} {row['acc5']:8.3f} {row['strongest']:10.3f}"
        )

    # ── Key finding ──
    # Find cheapest budget where CDM acc@5 beats the most expensive strongest model
    strongest_acc = price_curve_df.iloc[-1]["strongest"]
    for _, row in price_curve_df.iterrows():
        if row["acc5"] >= strongest_acc:
            print(
                f"\nKey finding: CDM acc@5 at ${row['budget']:.2f}/M "
                f"({row['acc5']:.3f}) >= strongest model at "
                f"${price_curve_df.iloc[-1]['budget']:.2f}/M "
                f"({strongest_acc:.3f})"
            )
            break

    # ── Verify and log ──
    from cdmeval.utils.experiment import verify_splits, log_experiment
    verified = verify_splits(train_items, test_items, label="cost_analysis")
    log_experiment(
        name="cost_analysis",
        config={"K": K, "epochs": cfg.model.epochs, "lr": cfg.model.lr,
                "device": device, "seed": 42},
        results={"price_sweep": price_curve_df.to_dict(orient="records")},
        split_info={"n_train_items": len(train_items),
                    "n_test_items": len(test_items), "n_llms": n_llms},
        verified=verified,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
