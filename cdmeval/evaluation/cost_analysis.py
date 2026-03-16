"""Cost-constrained routing analysis: budget sweeps and accuracy curves."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def extract_model_size(name: str) -> Optional[float]:
    """Parse parameter count (in billions) from a HuggingFace-style model name.

    Handles MoE notation (``8x7B`` -> 56.0), standard billions (``70B``),
    and millions (``220M`` -> 0.22).  Skips context-window tokens like
    ``32K`` and version strings like ``v0.1``.

    Returns ``None`` when the size cannot be determined.
    """
    # Use the portion after the last '__' (model name, not org)
    parts = name.split("__")
    suffix = parts[-1] if len(parts) > 1 else name

    # MoE: 8x7B, 4x7B, etc.
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))

    # Standard billions: 70B, 1.5B, 0.5B
    billions = re.findall(r"(\d+\.?\d*)[bB]", suffix)
    # Filter out things that look like context windows or versions
    for b in billions:
        val = float(b)
        # Skip if preceded by 'v' (version) — check original suffix
        idx = suffix.find(b + "B") if (b + "B") in suffix else suffix.find(b + "b")
        if idx > 0 and suffix[idx - 1].lower() == "v":
            continue
        return val

    # Millions: 220M, 125M
    millions = re.findall(r"(\d+\.?\d*)[mM]", suffix)
    for m in millions:
        val = float(m)
        # Skip context windows (32K -> skip), but M is fine
        # Skip if too large to be param count in millions (e.g., > 5000M = 5B, use B)
        if val > 0:
            return val / 1000.0

    return None


def get_model_sizes(llm_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract parameter counts for all models.

    Returns:
        sizes: float array of length n_llms (0.0 where unparseable).
        valid_mask: boolean array, True where size was successfully parsed.
    """
    n = len(llm_names)
    sizes = np.zeros(n)
    valid_mask = np.zeros(n, dtype=bool)

    for i, name in enumerate(llm_names):
        s = extract_model_size(name)
        if s is not None:
            sizes[i] = s
            valid_mask[i] = True

    n_valid = valid_mask.sum()
    print(f"Model size coverage: {n_valid}/{n} ({100 * n_valid / n:.1f}%)")
    missing = [llm_names[i] for i in range(n) if not valid_mask[i]]
    if missing:
        print(f"  Could not parse: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return sizes, valid_mask


DEFAULT_BUDGETS = [0.5, 1, 2, 3, 5, 7, 8, 10, 13, 15, 20, 34, 50, 70]


def cost_constrained_routing_curve(
    net,
    text_embeddings: np.ndarray,
    q_matrix: np.ndarray,
    response_vals: np.ndarray,
    test_items: np.ndarray,
    llm_names: list[str],
    sizes: np.ndarray,
    valid_mask: np.ndarray,
    device: str,
    budgets: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep budget ceilings and compute routing accuracy at each level.

    For every budget ceiling *b*, only models whose parameter count <= *b* are
    in the candidate pool.  The CDM model ranks them and we measure acc@k.

    Returns a DataFrame with columns
    ``[budget, acc1, acc3, acc5, majority, oracle, pool_size]``.
    """
    if budgets is None:
        budgets = DEFAULT_BUDGETS

    net.eval()
    net = net.to(device)
    n_llms = len(llm_names)

    rows = []
    for budget in budgets:
        # Models within budget (must also have a valid parsed size)
        pool_mask = valid_mask & (sizes <= budget)
        pool_indices = np.where(pool_mask)[0]
        pool_size = len(pool_indices)

        if pool_size == 0:
            rows.append(dict(budget=budget, acc1=0, acc3=0, acc5=0,
                             majority=0, oracle=0, pool_size=0))
            continue

        pool_ids = torch.tensor(pool_indices, dtype=torch.long, device=device)

        # Majority baseline: best single model in pool by overall accuracy
        pool_response = response_vals[pool_indices]
        best_in_pool = pool_indices[pool_response.mean(axis=1).argmax()]

        accs_at = {1: 0, 3: 0, 5: 0}
        majority_correct = 0
        oracle_correct = 0
        total = 0

        for item_idx in test_items:
            gt = response_vals[:, int(item_idx)]
            if gt.sum() == 0:
                continue

            gt_pool = gt[pool_indices]
            total += 1

            # Oracle: any model in pool gets it right
            if gt_pool.sum() > 0:
                oracle_correct += 1

            # Majority baseline
            if gt[best_in_pool] > 0:
                majority_correct += 1

            # CDM routing within pool
            emb = torch.tensor(
                text_embeddings[int(item_idx)], dtype=torch.float32, device=device
            )
            emb_batch = emb.unsqueeze(0).expand(pool_size, -1)
            q_row = torch.tensor(
                q_matrix[int(item_idx)], dtype=torch.float32, device=device
            )
            q_batch = q_row.unsqueeze(0).expand(pool_size, -1)

            with torch.no_grad():
                preds = net(pool_ids, emb_batch, q_batch).cpu().numpy()

            ranking = pool_indices[np.argsort(-preds)]
            for k in accs_at:
                if gt[ranking[:k]].sum() > 0:
                    accs_at[k] += 1

        if total == 0:
            rows.append(dict(budget=budget, acc1=0, acc3=0, acc5=0,
                             majority=0, oracle=0, pool_size=pool_size))
            continue

        rows.append(dict(
            budget=budget,
            acc1=accs_at[1] / total,
            acc3=accs_at[3] / total,
            acc5=accs_at[5] / total,
            majority=majority_correct / total,
            oracle=oracle_correct / total,
            pool_size=pool_size,
        ))
        print(
            f"  Budget {budget:5.1f}B  pool={pool_size:3d}  "
            f"acc@1={rows[-1]['acc1']:.3f}  acc@5={rows[-1]['acc5']:.3f}  "
            f"majority={rows[-1]['majority']:.3f}  oracle={rows[-1]['oracle']:.3f}"
        )

    return pd.DataFrame(rows)


def plot_routing_curve(curve_df: pd.DataFrame, fig_dir) -> None:
    """Plot cost-constrained routing curve and save PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cdmeval.utils.visualization import SAVE_KW, setup_style

    setup_style()
    fig_dir = type(fig_dir) is str and __import__("pathlib").Path(fig_dir) or fig_dir

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(curve_df["budget"], curve_df["acc1"], "o-", label="CDM acc@1",
            color="#4C72B0", markersize=5, linewidth=2)
    ax.plot(curve_df["budget"], curve_df["acc5"], "s-", label="CDM acc@5",
            color="#55A868", markersize=5, linewidth=2)
    ax.plot(curve_df["budget"], curve_df["majority"], "^--", label="Majority baseline",
            color="#DD8452", markersize=5, linewidth=1.5)
    ax.plot(curve_df["budget"], curve_df["oracle"], "d:", label="Oracle (any in pool)",
            color="#8172B3", markersize=5, linewidth=1.5)

    ax.set_xlabel("Budget ceiling (billion parameters)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost-Constrained Routing Accuracy")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    # Secondary axis: pool size
    ax2 = ax.twinx()
    ax2.fill_between(curve_df["budget"], curve_df["pool_size"],
                     alpha=0.08, color="grey")
    ax2.set_ylabel("Pool size (# models)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.spines["right"].set_visible(True)

    plt.tight_layout()
    out = fig_dir / "fig_cost_routing_curve.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Price-based routing ($/M tokens)
# ════════════════════════════════════════════════════════════════════

DEFAULT_PRICE_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 2.00, 5.00, 10.00]


def cost_constrained_routing_by_price(
    net,
    text_embeddings: np.ndarray,
    q_matrix: np.ndarray,
    response_vals: np.ndarray,
    test_items: np.ndarray,
    prices: np.ndarray,
    device: str,
    budgets: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep price-based budget ceilings and compute routing accuracy.

    Args:
        net: Trained TextConditionedNet.
        text_embeddings: ``(n_items, text_dim)`` SBERT embeddings.
        q_matrix: ``(n_items, K)`` binary Q-matrix.
        response_vals: ``(n_llms, n_items)`` binary response matrix.
        test_items: Held-out item indices.
        prices: ``(n_llms,)`` price per million output tokens.
        device: Torch device.
        budgets: Price ceilings to sweep ($/M tokens).

    Returns:
        DataFrame with columns
        ``[budget, acc1, acc3, acc5, majority, oracle, pool_size]``.
    """
    if budgets is None:
        budgets = DEFAULT_PRICE_BUDGETS

    net.eval()
    net = net.to(device)
    n_llms = len(prices)

    rows = []
    for budget in budgets:
        pool_indices = np.where(prices <= budget)[0]
        pool_size = len(pool_indices)

        if pool_size == 0:
            rows.append(dict(budget=budget, acc1=0, acc3=0, acc5=0,
                             majority=0, oracle=0, pool_size=0))
            continue

        pool_ids = torch.tensor(pool_indices, dtype=torch.long, device=device)

        pool_response = response_vals[pool_indices]
        best_in_pool = pool_indices[pool_response.mean(axis=1).argmax()]

        accs_at = {1: 0, 3: 0, 5: 0}
        majority_correct = 0
        oracle_correct = 0
        total = 0

        for item_idx in test_items:
            gt = response_vals[:, int(item_idx)]
            if gt.sum() == 0:
                continue

            gt_pool = gt[pool_indices]
            total += 1

            if gt_pool.sum() > 0:
                oracle_correct += 1
            if gt[best_in_pool] > 0:
                majority_correct += 1

            emb = torch.tensor(
                text_embeddings[int(item_idx)], dtype=torch.float32, device=device
            )
            emb_batch = emb.unsqueeze(0).expand(pool_size, -1)
            q_row = torch.tensor(
                q_matrix[int(item_idx)], dtype=torch.float32, device=device
            )
            q_batch = q_row.unsqueeze(0).expand(pool_size, -1)

            with torch.no_grad():
                preds = net(pool_ids, emb_batch, q_batch).cpu().numpy()

            ranking = pool_indices[np.argsort(-preds)]
            for k in accs_at:
                if gt[ranking[:k]].sum() > 0:
                    accs_at[k] += 1

        if total == 0:
            rows.append(dict(budget=budget, acc1=0, acc3=0, acc5=0,
                             majority=0, oracle=0, pool_size=pool_size))
            continue

        rows.append(dict(
            budget=budget,
            acc1=accs_at[1] / total,
            acc3=accs_at[3] / total,
            acc5=accs_at[5] / total,
            majority=majority_correct / total,
            oracle=oracle_correct / total,
            pool_size=pool_size,
        ))
        print(
            f"  Budget ${budget:<6.2f}/M  pool={pool_size:3d}  "
            f"acc@1={rows[-1]['acc1']:.3f}  acc@5={rows[-1]['acc5']:.3f}  "
            f"majority={rows[-1]['majority']:.3f}"
        )

    return pd.DataFrame(rows)


def plot_routing_curve_price(curve_df: pd.DataFrame, fig_dir) -> None:
    """Plot price-based cost-constrained routing curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cdmeval.utils.visualization import SAVE_KW, setup_style

    setup_style()
    fig_dir = type(fig_dir) is str and __import__("pathlib").Path(fig_dir) or fig_dir

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(curve_df["budget"], curve_df["acc1"], "o-", label="CDM acc@1",
            color="#4C72B0", markersize=5, linewidth=2)
    ax.plot(curve_df["budget"], curve_df["acc5"], "s-", label="CDM acc@5",
            color="#55A868", markersize=5, linewidth=2)
    ax.plot(curve_df["budget"], curve_df["majority"], "^--", label="Majority baseline",
            color="#DD8452", markersize=5, linewidth=1.5)
    ax.plot(curve_df["budget"], curve_df["oracle"], "d:", label="Oracle (any in pool)",
            color="#8172B3", markersize=5, linewidth=1.5)

    ax.set_xlabel("Max cost per query ($/M output tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost-Constrained Routing by API Price")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)

    ax2 = ax.twinx()
    ax2.fill_between(curve_df["budget"], curve_df["pool_size"],
                     alpha=0.08, color="grey")
    ax2.set_ylabel("Pool size (# models)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.spines["right"].set_visible(True)

    plt.tight_layout()
    out = fig_dir / "fig_cost_routing_curve_price.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


def plot_cost_accuracy_scatter_price(
    llm_names: list[str],
    prices: np.ndarray,
    response_vals: np.ndarray,
    net,
    text_embeddings: np.ndarray,
    q_matrix: np.ndarray,
    test_items: np.ndarray,
    device: str,
    fig_dir,
) -> None:
    """Scatter: per-model accuracy vs price, highlighting CDM picks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    from cdmeval.utils.visualization import SAVE_KW, setup_style

    setup_style()
    fig_dir = type(fig_dir) is str and __import__("pathlib").Path(fig_dir) or fig_dir

    n_llms = len(llm_names)
    per_model_acc = response_vals.mean(axis=1)

    # CDM top-1 picks
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)
    picks = []

    for item_idx in test_items:
        gt = response_vals[:, int(item_idx)]
        if gt.sum() == 0:
            continue
        emb = torch.tensor(
            text_embeddings[int(item_idx)], dtype=torch.float32, device=device
        )
        emb_batch = emb.unsqueeze(0).expand(n_llms, -1)
        q_row = torch.tensor(
            q_matrix[int(item_idx)], dtype=torch.float32, device=device
        )
        q_batch = q_row.unsqueeze(0).expand(n_llms, -1)
        with torch.no_grad():
            preds = net(all_llm_ids, emb_batch, q_batch).cpu().numpy()
        picks.append(int(np.argmax(preds)))

    freq = Counter(picks)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # All models (grey)
    ax.scatter(
        prices, per_model_acc,
        c="#cccccc", s=40, alpha=0.6, edgecolors="white", linewidths=0.3,
        label="All models", zorder=2,
    )

    # CDM picks (coloured)
    selected_indices = sorted(freq.keys())
    selected_prices = prices[selected_indices]
    selected_acc = per_model_acc[selected_indices]
    selected_freq = np.array([freq[i] for i in selected_indices])

    if len(selected_indices) > 0:
        sc = ax.scatter(
            selected_prices, selected_acc,
            c=selected_freq, cmap="YlOrRd", s=80,
            edgecolors="black", linewidths=0.6, zorder=3,
            label="CDM-routed picks",
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Selection frequency", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Price ($/M output tokens, log scale)")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Model Price vs. Accuracy with CDM Routing Selections")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = fig_dir / "fig_cost_accuracy_scatter_price.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")
