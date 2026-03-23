"""Pareto frontier analysis: threshold-based routing with FLOPs cost.

Usage:
    python tools/pareto_routing.py device=mps
    python tools/pareto_routing.py device=mps model.epochs=1
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


K = 50


def _get_cdm_predictions(net, test_items, text_embeddings, q_matrix, n_llms, device):
    """Get P(correct | LLM_s, item_j) for all LLMs on all test items.

    Returns: dict {item_idx: np.ndarray of shape (n_llms,)}
    """
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    preds = {}
    with torch.no_grad():
        for item_idx in test_items:
            emb = torch.tensor(
                text_embeddings[int(item_idx)], dtype=torch.float32, device=device
            )
            emb_batch = emb.unsqueeze(0).expand(n_llms, -1)
            q_row = torch.tensor(
                q_matrix[int(item_idx)], dtype=torch.float32, device=device
            )
            q_batch = q_row.unsqueeze(0).expand(n_llms, -1)
            p = net(all_llm_ids, emb_batch, q_batch).cpu().numpy()
            preds[int(item_idx)] = p

    return preds


def _get_irt_predictions(response_vals, train_items, test_items, n_llms, n_items,
                         epochs, lr, batch_size, device):
    """Train IRT 2PL and return raw predictions for all test items."""
    from cdmeval.evaluation.baselines import IRT2PL, _train_irt

    student_ids = np.repeat(np.arange(n_llms), len(train_items))
    item_ids = np.tile(train_items.astype(int), n_llms)
    scores = response_vals[
        np.repeat(np.arange(n_llms), len(train_items)),
        np.tile(train_items.astype(int), n_llms),
    ].astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])

    idx = np.arange(len(triplets))
    tr_idx, va_idx = train_test_split(idx, test_size=0.1, random_state=42)

    print("  Training IRT 2PL...")
    model = IRT2PL(n_llms, n_items)
    model = _train_irt(
        model, triplets[tr_idx], triplets[va_idx],
        epochs=epochs, lr=lr, batch_size=batch_size, device=device,
    )

    model.eval()
    model = model.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    preds = {}
    with torch.no_grad():
        for item_idx in test_items:
            item_t = torch.full(
                (n_llms,), int(item_idx), dtype=torch.int64, device=device
            )
            p = model(all_llm_ids, item_t).cpu().numpy()
            preds[int(item_idx)] = p

    return preds


def _threshold_sweep(preds, test_items, response_vals, flops_array, thresholds,
                     fallback="best_pred"):
    """Run threshold-based routing for a set of predictions.

    For each threshold t:
      For each item, pick the cheapest LLM with P(correct) > t.
      If none, fall back according to *fallback* strategy.

    Args:
        fallback: ``"best_pred"`` = highest P(correct) model,
                  ``"strongest"`` = globally best LLM by train accuracy.

    Returns list of dicts with (threshold, accuracy, mean_cost).
    """
    n_total = len(test_items)

    results = []
    for t in thresholds:
        correct = 0
        total_cost = 0.0

        for item_idx in test_items:
            p = preds[int(item_idx)]
            gt = response_vals[:, int(item_idx)]

            # Models above threshold
            above = np.where(p > t)[0]
            if len(above) > 0:
                # Pick cheapest among confident models
                costs_above = flops_array[above]
                chosen = above[np.argmin(costs_above)]
            else:
                # Fallback: pick the model with highest predicted P(correct)
                chosen = int(np.argmax(p))

            total_cost += flops_array[chosen]
            if gt[chosen] > 0:
                correct += 1

        results.append({
            "threshold": float(t),
            "accuracy": correct / n_total,
            "mean_cost": total_cost / n_total,
        })

    return results


def _pareto_frontier(points):
    """Extract Pareto-optimal points (minimise cost, maximise accuracy).

    Returns indices of non-dominated points sorted by cost.
    """
    n = len(points)
    costs = np.array([p["mean_cost"] for p in points])
    accs = np.array([p["accuracy"] for p in points])

    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if costs[j] <= costs[i] and accs[j] >= accs[i]:
                if costs[j] < costs[i] or accs[j] > accs[i]:
                    dominated[i] = True
                    break

    pareto_idx = np.where(~dominated)[0]
    pareto_idx = pareto_idx[np.argsort(costs[pareto_idx])]
    return pareto_idx


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_text_dataloader
    from cdmeval.data.response_matrix import build_triplets, load_q_matrix, load_response_matrix
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.visualization import SAVE_KW, setup_style

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

    text_embeddings = np.load(data_dir / "item_text_embeddings.npz")["embeddings"]
    text_dim = text_embeddings.shape[1]

    # Load FLOPs
    with open(data_dir / "model_flops.json") as f:
        flops_dict = json.load(f)
    flops_array = np.array([flops_dict[name] for name in llm_names])

    print(f"Data: {n_llms} LLMs, {n_items} items, {n_skills} skills")

    # ── Protocol B split ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(
        all_items, test_size=0.2, random_state=42
    )
    train_val_triplets = triplets[np.isin(triplets[:, 1].astype(int), train_items)]
    tv_idx = np.arange(len(train_val_triplets))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)

    print(f"Split: {len(train_items)} train items, {len(test_items)} test items")

    # ── Train text-conditioned NCDM ──
    print("\n" + "=" * 60)
    print("Training TextConditionedNet (Protocol B)")
    print("=" * 60)
    bs = cfg.model.batch_size
    train_loader = make_text_dataloader(
        train_val_triplets[tr_idx], text_embeddings, q_matrix, bs, shuffle=True
    )
    val_loader = make_text_dataloader(
        train_val_triplets[va_idx], text_embeddings, q_matrix, bs, shuffle=False
    )
    net = TextConditionedNet(n_skills, n_llms, text_dim)
    net = train_text_model(
        net, train_loader, val_loader,
        epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
    )

    # ── CDM predictions ──
    print("\nGenerating CDM predictions on 529 test items...")
    cdm_preds = _get_cdm_predictions(
        net, test_items, text_embeddings, q_matrix, n_llms, device
    )

    # ── IRT predictions ──
    print("\n" + "=" * 60)
    print("IRT 2PL Predictions")
    print("=" * 60)
    irt_preds = _get_irt_predictions(
        response_vals, train_items, test_items, n_llms, n_items,
        epochs=cfg.model.epochs, lr=cfg.model.lr,
        batch_size=cfg.model.batch_size, device=device,
    )

    # ── Prediction diagnostics ──
    print("\n" + "=" * 60)
    print("Prediction Diagnostics")
    print("=" * 60)
    all_cdm_p = np.concatenate([cdm_preds[int(i)] for i in test_items])
    per_item_max = np.array([cdm_preds[int(i)].max() for i in test_items])
    print(f"  CDM P(correct) overall: mean={all_cdm_p.mean():.4f}, "
          f"median={np.median(all_cdm_p):.4f}")
    print(f"  Per-item max P: mean={per_item_max.mean():.4f}, "
          f"median={np.median(per_item_max):.4f}, "
          f"min={per_item_max.min():.4f}")
    for t in [0.3, 0.5, 0.7, 0.9]:
        frac = (per_item_max > t).mean()
        print(f"  Items with any LLM P > {t}: {frac*100:.1f}%")

    # ── Threshold sweep ──
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65,
                  0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    print("\n" + "=" * 60)
    print("CDM Threshold Sweep (fallback=best_pred)")
    print("=" * 60)
    cdm_sweep = _threshold_sweep(
        cdm_preds, test_items, response_vals, flops_array, thresholds,
        fallback="best_pred",
    )

    print("\n" + "=" * 60)
    print("IRT Threshold Sweep (fallback=best_pred)")
    print("=" * 60)
    irt_sweep = _threshold_sweep(
        irt_preds, test_items, response_vals, flops_array, thresholds,
        fallback="best_pred",
    )

    # ── Baselines (single points) ──
    n_total = len(test_items)

    # Always strongest (best LLM by train accuracy)
    train_acc = response_vals[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    strongest_correct = sum(
        response_vals[strongest_idx, int(i)] for i in test_items
    )
    strongest_point = {
        "accuracy": strongest_correct / n_total,
        "mean_cost": float(flops_array[strongest_idx]),
        "label": f"Strongest ({llm_names[strongest_idx].split('__')[-1]})",
    }

    # Always cheapest
    cheapest_idx = int(np.argmin(flops_array))
    cheapest_correct = sum(
        response_vals[cheapest_idx, int(i)] for i in test_items
    )
    cheapest_point = {
        "accuracy": cheapest_correct / n_total,
        "mean_cost": float(flops_array[cheapest_idx]),
        "label": f"Cheapest ({llm_names[cheapest_idx].split('__')[-1]})",
    }

    # Oracle cheapest: for each item, pick cheapest correct model
    oracle_cost = 0.0
    oracle_correct = 0
    for item_idx in test_items:
        gt = response_vals[:, int(item_idx)]
        correct_models = np.where(gt > 0)[0]
        if len(correct_models) > 0:
            oracle_correct += 1
            oracle_cost += flops_array[correct_models].min()
        else:
            oracle_cost += flops_array.min()
    oracle_point = {
        "accuracy": oracle_correct / n_total,
        "mean_cost": oracle_cost / n_total,
        "label": "Oracle cheapest",
    }

    # ── Pareto frontier ──
    cdm_pareto_idx = _pareto_frontier(cdm_sweep)

    # ── Summary table ──
    print("\n" + "=" * 80)
    print("THRESHOLD ROUTING RESULTS")
    print("=" * 80)
    print(f"{'Threshold':>10} {'CDM Acc':>10} {'CDM Cost':>12} {'IRT Acc':>10} {'IRT Cost':>12}")
    print("-" * 80)
    for c, i in zip(cdm_sweep, irt_sweep):
        pareto = "*" if cdm_sweep.index(c) in cdm_pareto_idx else " "
        print(
            f"  {c['threshold']:>7.2f}{pareto} "
            f"{c['accuracy']:>10.4f} {c['mean_cost']:>11.0f} "
            f"{i['accuracy']:>10.4f} {i['mean_cost']:>11.0f}"
        )
    print(f"\n  Strongest: acc={strongest_point['accuracy']:.4f}, "
          f"cost={strongest_point['mean_cost']:.0f} GFLOPs")
    print(f"  Cheapest:  acc={cheapest_point['accuracy']:.4f}, "
          f"cost={cheapest_point['mean_cost']:.0f} GFLOPs")
    print(f"  Oracle:    acc={oracle_point['accuracy']:.4f}, "
          f"cost={oracle_point['mean_cost']:.0f} GFLOPs")
    print("  * = Pareto-optimal")

    # ── Plot ──
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cdm_costs = [p["mean_cost"] for p in cdm_sweep]
    cdm_accs = [p["accuracy"] for p in cdm_sweep]
    irt_costs = [p["mean_cost"] for p in irt_sweep]
    irt_accs = [p["accuracy"] for p in irt_sweep]

    # CDM threshold curve
    ax.plot(cdm_costs, cdm_accs, "o-", color="#4C72B0", lw=2, markersize=6,
            label="CDMEval", zorder=4)

    # Pareto frontier (highlight)
    pareto_costs = [cdm_sweep[i]["mean_cost"] for i in cdm_pareto_idx]
    pareto_accs = [cdm_sweep[i]["accuracy"] for i in cdm_pareto_idx]
    ax.plot(pareto_costs, pareto_accs, "-", color="#4C72B0", lw=4, alpha=0.3, zorder=3)

    # IRT threshold curve
    ax.plot(irt_costs, irt_accs, "s--", color="#DD8452", lw=1.5, markersize=5,
            label="IRT 2PL", zorder=4)

    # Annotate CDM thresholds
    for p in cdm_sweep:
        ax.annotate(
            f"t={p['threshold']:.2f}",
            xy=(p["mean_cost"], p["accuracy"]),
            xytext=(5, -10), textcoords="offset points",
            fontsize=6.5, color="#4C72B0", alpha=0.8,
        )

    # Baseline points
    ax.plot(strongest_point["mean_cost"], strongest_point["accuracy"],
            "*", color="#C44E52", markersize=14, zorder=5, label="Strongest model")
    ax.plot(cheapest_point["mean_cost"], cheapest_point["accuracy"],
            "*", color="#55A868", markersize=14, zorder=5, label="Cheapest model")
    ax.plot(oracle_point["mean_cost"], oracle_point["accuracy"],
            "D", color="#999999", markersize=8, zorder=5, label="Oracle cheapest")

    ax.set_xscale("log")
    ax.set_xlabel("Mean FLOPs cost per query (GFLOPs, log scale)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Cost-Accuracy Pareto Frontier (Threshold-Based Routing)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, max(cdm_accs + irt_accs + [strongest_point["accuracy"],
                oracle_point["accuracy"]]) * 1.08)

    plt.tight_layout()
    out = fig_dir / "fig_pareto_frontier.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"\nSaved {out}")

    # ── Save results ──
    save_data = {
        "cdm_sweep": cdm_sweep,
        "irt_sweep": irt_sweep,
        "cdm_pareto_indices": cdm_pareto_idx.tolist(),
        "strongest_baseline": strongest_point,
        "cheapest_baseline": cheapest_point,
        "oracle_baseline": oracle_point,
        "thresholds": thresholds,
    }
    out_json = data_dir / "pareto_routing_results.json"
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved {out_json}")

    # ── Verify and log ──
    from cdmeval.utils.experiment import verify_splits, log_experiment
    verified = verify_splits(train_items, test_items, label="pareto_routing")
    log_experiment(
        name="pareto_routing",
        config={"K": K, "epochs": cfg.model.epochs, "lr": cfg.model.lr,
                "device": device, "seed": 42, "thresholds": thresholds},
        results={"cdm_sweep": cdm_sweep, "strongest": strongest_point,
                 "cheapest": cheapest_point, "oracle": oracle_point},
        split_info={"n_train_items": len(train_items),
                    "n_test_items": len(test_items), "n_llms": n_llms},
        verified=verified,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
