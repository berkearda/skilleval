"""Cost-accuracy scatter plot with CDM routing selections highlighted."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch


def plot_cost_accuracy_scatter(
    llm_names: list[str],
    sizes: np.ndarray,
    valid_mask: np.ndarray,
    response_vals: np.ndarray,
    net,
    text_embeddings: np.ndarray,
    q_matrix: np.ndarray,
    test_items: np.ndarray,
    device: str,
    fig_dir,
) -> None:
    """Scatter: per-model accuracy vs param count, highlighting CDM picks.

    Grey circles = all models; coloured markers = CDM-selected models
    (colour encodes selection frequency).  Log-scale x-axis with trend line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cdmeval.utils.visualization import SAVE_KW, setup_style

    setup_style()
    fig_dir = type(fig_dir) is str and __import__("pathlib").Path(fig_dir) or fig_dir

    n_llms = len(llm_names)
    per_model_acc = response_vals.mean(axis=1)  # (n_llms,)

    # --- CDM top-1 picks for each test item ---
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)
    picks: list[int] = []

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

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # All models with valid sizes (grey)
    vm = valid_mask
    ax.scatter(
        sizes[vm], per_model_acc[vm],
        c="#cccccc", s=40, alpha=0.6, edgecolors="white", linewidths=0.3,
        label="All models", zorder=2,
    )

    # CDM-selected models (coloured by frequency)
    selected_indices = sorted(freq.keys())
    selected_sizes = sizes[selected_indices]
    selected_acc = per_model_acc[selected_indices]
    selected_freq = np.array([freq[i] for i in selected_indices])
    # Only plot those with valid sizes
    sel_valid = valid_mask[selected_indices]

    if sel_valid.sum() > 0:
        sc = ax.scatter(
            selected_sizes[sel_valid], selected_acc[sel_valid],
            c=selected_freq[sel_valid], cmap="YlOrRd", s=80,
            edgecolors="black", linewidths=0.6, zorder=3,
            label="CDM-routed picks",
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Selection frequency", fontsize=9)

    # Log-scale trend line
    valid_sizes = sizes[vm]
    valid_acc = per_model_acc[vm]
    if len(valid_sizes) > 2:
        log_s = np.log10(valid_sizes)
        coeffs = np.polyfit(log_s, valid_acc, 1)
        xs = np.linspace(valid_sizes.min(), valid_sizes.max(), 100)
        ys = np.clip(np.polyval(coeffs, np.log10(xs)), 0, 1)
        ax.plot(xs, ys, "--", color="#666666", linewidth=1.2, alpha=0.7,
                label=f"Log trend (slope={coeffs[0]:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (billions, log scale)")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Model Cost vs. Accuracy with CDM Routing Selections")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = fig_dir / "fig_cost_accuracy_scatter.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")
