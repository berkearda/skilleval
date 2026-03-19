"""Visualisation helpers for cluster sweep analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualization import SAVE_KW, setup_style


# ════════════════════════════════════════════════════════════════════
#  Figure 1 – K vs downstream metrics (multi-panel)
# ════════════════════════════════════════════════════════════════════


def plot_k_vs_metrics(sweep_df: pd.DataFrame, fig_dir: Path) -> None:
    """Four-panel figure: AUC, routing, clustering quality, Q-matrix density."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ks = sweep_df["K"].values

    SELECTED_K = 50
    TITLE_SIZE = 12
    LABEL_SIZE = 11
    TICK_SIZE = 9
    GRID_ALPHA = 0.3

    def _style_ax(ax):
        ax.grid(True, alpha=GRID_ALPHA, linewidth=0.5)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.axvline(SELECTED_K, ls="--", color="#C44E52", alpha=0.45, lw=1, zorder=0)

    # ── Panel 1: K vs downstream AUC (Protocol A) ──
    ax = axes[0, 0]
    ax.plot(ks, sweep_df["proto_a_auc"], "o-", color="#4C72B0", lw=2, markersize=5)
    _style_ax(ax)
    # Label the K=50 line at the top of the subplot
    ax.text(
        SELECTED_K + 3, ax.get_ylim()[1] - 0.0003,
        "K=50 (selected)", fontsize=8, color="#C44E52", va="top",
    )
    ax.set_xlabel("K (number of skill clusters)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Test AUC", fontsize=LABEL_SIZE)
    ax.set_title("(a) Protocol A: ID-based NCDM", fontsize=TITLE_SIZE, fontweight="bold", loc="left")

    # ── Panel 2: K vs routing acc@1 and acc@5 ──
    ax = axes[0, 1]
    ax.plot(ks, sweep_df["routing_acc1"], "s-", color="#55A868", lw=2, markersize=5, label="Acc@1")
    ax.plot(ks, sweep_df["routing_acc5"], "D-", color="#DD8452", lw=2, markersize=5, label="Acc@5")
    _style_ax(ax)
    ax.set_xlabel("K (number of skill clusters)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Routing Accuracy", fontsize=LABEL_SIZE)
    ax.set_title("(b) Protocol B: Routing accuracy", fontsize=TITLE_SIZE, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # ── Panel 3: K vs silhouette and intra-cluster similarity (dual y-axis) ──
    ax = axes[1, 0]
    color1, color2 = "#8172B2", "#CCB974"
    ln1 = ax.plot(ks, sweep_df["silhouette"], "o-", color=color1, lw=2, markersize=5, label="Silhouette")
    ax.set_xlabel("K (number of skill clusters)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Silhouette score", color=color1, fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelcolor=color1, labelsize=TICK_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.5)
    ax.axvline(SELECTED_K, ls="--", color="#C44E52", alpha=0.45, lw=1, zorder=0)

    ax2 = ax.twinx()
    ln2 = ax2.plot(ks, sweep_df["mean_intra_cosine_sim"], "^-", color=color2, lw=2, markersize=5, label="Intra-cluster sim")
    ax2.set_ylabel("Mean intra-cluster cosine sim", color=color2, fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=TICK_SIZE)
    ax2.spines["top"].set_visible(False)

    # Separate legends to avoid overlap
    ax.legend(ln1, ["Silhouette"], frameon=False, fontsize=9, loc="upper left")
    ax2.legend(ln2, ["Intra-cluster sim"], frameon=False, fontsize=9, loc="lower right")
    ax.set_title("(c) Intrinsic clustering quality", fontsize=TITLE_SIZE, fontweight="bold", loc="left")

    # ── Panel 4: K vs mean skills per item (Q-matrix row sum) ──
    ax = axes[1, 1]
    ax.plot(ks, sweep_df["mean_skills_per_item"], "o-", color="#64B5CD", lw=2, markersize=5)
    _style_ax(ax)
    ax.set_xlabel("K (number of skill clusters)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean skills per item", fontsize=LABEL_SIZE)
    ax.set_title("(d) Q-matrix density", fontsize=TITLE_SIZE, fontweight="bold", loc="left")

    out = fig_dir / "fig_cluster_sweep.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Figure 2 – Cluster example comparison (K=20 vs 50 vs 100)
# ════════════════════════════════════════════════════════════════════


def plot_cluster_examples(
    k_values_to_show: list[int],
    skill_embeddings: np.ndarray,
    unique_skills: list[str],
    skill_counts,
    fig_dir: Path,
    n_example_clusters: int = 3,
    n_top_skills: int = 5,
) -> None:
    """Show example clusters at different K granularities.

    For each K, pick the 3 largest clusters and show their top-5 member skills.
    """
    from cdmeval.skills.clustering import cluster_hac, label_clusters

    setup_style()
    n_k = len(k_values_to_show)
    fig, axes = plt.subplots(1, n_k, figsize=(5.5 * n_k, 4.5))
    if n_k == 1:
        axes = [axes]

    titles = {}
    if len(k_values_to_show) == 3:
        titles = {
            k_values_to_show[0]: f"(a) K={k_values_to_show[0]} — too coarse",
            k_values_to_show[1]: f"(b) K={k_values_to_show[1]} — sweet spot",
            k_values_to_show[2]: f"(c) K={k_values_to_show[2]} — too fine",
        }

    for ax, K in zip(axes, k_values_to_show):
        labels = cluster_hac(skill_embeddings, K)
        cluster_labels = label_clusters(unique_skills, labels, skill_counts)

        # Pick the n largest clusters
        cluster_sizes = {}
        cluster_members = {}
        for cid in sorted(set(labels)):
            members = [unique_skills[i] for i in range(len(labels)) if labels[i] == cid]
            cluster_sizes[cid] = len(members)
            # Sort members by frequency
            member_freqs = [(s, skill_counts.get(s, 0)) for s in members]
            member_freqs.sort(key=lambda x: -x[1])
            cluster_members[cid] = member_freqs

        top_clusters = sorted(cluster_sizes, key=lambda c: -cluster_sizes[c])[:n_example_clusters]

        # Build text table
        y_pos = 0.95
        for ci, cid in enumerate(top_clusters):
            label = cluster_labels[cid]
            size = cluster_sizes[cid]
            ax.text(
                0.05, y_pos, f"{label}  ({size} skills)",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left",
            )
            y_pos -= 0.06

            top_members = cluster_members[cid][:n_top_skills]
            for skill_name, freq in top_members:
                display = skill_name if len(skill_name) <= 45 else skill_name[:42] + "..."
                ax.text(
                    0.08, y_pos, f"• {display}  ({freq})",
                    transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                    color="#555555",
                )
                y_pos -= 0.05

            y_pos -= 0.03  # gap between clusters

        title = titles.get(K, f"K={K}")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.axis("off")

    plt.tight_layout()
    out = fig_dir / "fig_cluster_examples.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")
