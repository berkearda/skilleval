"""Publication-quality figure generation for CDM reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

SAVE_KW = dict(dpi=300, bbox_inches="tight", transparent=False)


def setup_style() -> None:
    """Apply publication-quality plot style."""
    sns.set_theme(
        style="whitegrid",
        font_scale=1.1,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
        },
    )


# ════════════════════════════════════════════════════════════════════
#  Figure 1 – Skill distribution (two-panel)
# ════════════════════════════════════════════════════════════════════


def fig_skill_distribution(data_dir: Path, fig_dir: Path) -> None:
    setup_style()
    with open(data_dir / "skills_extracted.json") as f:
        items = json.load(f)
    with open(data_dir / "skill_taxonomy.json") as f:
        taxonomy = json.load(f)

    skills_per_item = [item["num_skills"] for item in items]
    cluster_sizes = {k: len(v) for k, v in taxonomy.items()}
    cluster_sizes = dict(sorted(cluster_sizes.items(), key=lambda x: -x[1])[:20])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    bins = np.arange(0.5, max(skills_per_item) + 1.5, 1)
    ax1.hist(skills_per_item, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax1.set_xlabel("Number of skills per problem")
    ax1.set_ylabel("Frequency")
    ax1.set_title("(a) Skills per problem", fontsize=11, fontweight="bold", loc="left")
    mean_val = np.mean(skills_per_item)
    ax1.axvline(mean_val, color="#C44E52", ls="--", lw=1.4, label=f"Mean = {mean_val:.1f}")
    ax1.legend(frameon=False)

    names = list(cluster_sizes.keys())[::-1]
    vals = [cluster_sizes[n] for n in names]
    palette = sns.color_palette("mako_r", len(names))
    ax2.barh(range(len(names)), vals, color=palette, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Number of raw skills in cluster")
    ax2.set_title("(b) Largest 20 skill clusters", fontsize=11, fontweight="bold", loc="left")

    plt.tight_layout(w_pad=3)
    out = fig_dir / "fig_skill_distribution.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Figure 2 – UMAP skill embedding space
# ════════════════════════════════════════════════════════════════════


def fig_umap(data_dir: Path, fig_dir: Path) -> None:
    setup_style()
    npz = np.load(data_dir / "skill_embeddings.npz", allow_pickle=True)
    coords = npz["coords_2d"]
    labels = npz["labels"]

    with open(data_dir / "skill_taxonomy.json") as f:
        taxonomy = json.load(f)

    cluster_names = list(taxonomy.keys())
    cluster_ids = sorted(set(labels))
    id_to_name = {}
    for cid in cluster_ids:
        if cid < len(cluster_names):
            id_to_name[cid] = cluster_names[cid]
        else:
            id_to_name[cid] = f"Cluster {cid}"

    sizes = Counter(labels)
    top_ids = [cid for cid, _ in sizes.most_common(15)]

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab20", 20)

    other_mask = ~np.isin(labels, top_ids)
    ax.scatter(
        coords[other_mask, 0], coords[other_mask, 1],
        c="#d0d0d0", s=6, alpha=0.35, rasterized=True,
    )

    for rank, cid in enumerate(top_ids):
        mask = labels == cid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(rank)], s=12, alpha=0.65,
            label=id_to_name.get(cid, f"Cluster {cid}"),
            rasterized=True,
        )

    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5,
        markerscale=2, frameon=False, title="Top-15 clusters", title_fontsize=8,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Skill embedding space (7,723 skills, 281 clusters)")
    ax.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    out = fig_dir / "fig_umap.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Figure 3 – Q-matrix sparsity heatmap
# ════════════════════════════════════════════════════════════════════


def fig_qmatrix(data_dir: Path, fig_dir: Path) -> None:
    setup_style()
    q = pd.read_csv(data_dir / "q_matrix.csv")
    source = q["source"].values
    skill_cols = [c for c in q.columns if c not in ("item_idx", "source")]
    mat = q[skill_cols].values

    order = np.argsort(source, kind="stable")[::-1]
    mat_sorted = mat[order]
    source_sorted = source[order]

    freq = mat.sum(axis=0)
    skill_order = np.argsort(-freq)
    mat_sorted = mat_sorted[:, skill_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(mat_sorted, aspect="auto", cmap="Blues", interpolation="nearest")

    n_gsm = (source_sorted == "GSM8K").sum()
    ax.axhline(n_gsm - 0.5, color="#C44E52", lw=1.2, ls="--")
    ax.text(
        mat_sorted.shape[1] + 2, n_gsm / 2, "GSM8K",
        va="center", fontsize=9, color="#C44E52",
    )
    ax.text(
        mat_sorted.shape[1] + 2, n_gsm + (len(source_sorted) - n_gsm) / 2,
        "MATH", va="center", fontsize=9, color="#C44E52",
    )

    ax.set_xlabel(f"Skill clusters (n = {len(skill_cols)})")
    ax.set_ylabel(f"Problems (n = {mat.shape[0]})")
    ax.set_title("Q-matrix sparsity pattern")
    ax.tick_params(labelbottom=False, labelleft=False)

    density = mat.sum() / mat.size
    ax.text(
        0.98, 0.02, f"Density = {density:.3f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
    )

    plt.tight_layout()
    out = fig_dir / "fig_qmatrix_heatmap.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Figure 4 – Skill mastery heatmap (top/bottom LLMs)
# ════════════════════════════════════════════════════════════════════


def fig_mastery_heatmap(data_dir: Path, fig_dir: Path) -> None:
    setup_style()
    mastery = pd.read_csv(data_dir / "skill_mastery_profiles.csv", index_col=0)

    overall = mastery.mean(axis=1).sort_values(ascending=False)
    top20 = overall.head(20).index.tolist()
    bottom10 = overall.tail(10).index.tolist()
    selected_llms = top20 + bottom10

    skill_var = mastery.var(axis=0).sort_values(ascending=False)
    selected_skills = skill_var.head(30).index.tolist()

    sub = mastery.loc[selected_llms, selected_skills]

    def short_name(name):
        parts = name.split("__")
        return parts[-1] if len(parts) > 1 else name

    sub.index = [short_name(n) for n in sub.index]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    sns.heatmap(
        sub, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.15, linecolor="white",
        cbar_kws={"label": "Mastery probability", "shrink": 0.6},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha="right", fontsize=6.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    ax.axhline(20, color="black", lw=1.5)
    ax.text(
        -0.5, 10, "Top 20", va="center", ha="right",
        fontsize=8, fontstyle="italic", rotation=90,
    )
    ax.text(
        -0.5, 25, "Bottom 10", va="center", ha="right",
        fontsize=8, fontstyle="italic", rotation=90,
    )

    ax.set_title("Skill mastery profiles: highest- and lowest-performing LLMs")
    plt.tight_layout()
    out = fig_dir / "fig_mastery_heatmap.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Figure 5 – LLM ranking strip / bar plot
# ════════════════════════════════════════════════════════════════════


def fig_llm_ranking(data_dir: Path, fig_dir: Path) -> None:
    setup_style()
    mastery = pd.read_csv(data_dir / "skill_mastery_profiles.csv", index_col=0)
    overall = mastery.mean(axis=1).sort_values(ascending=True)

    def family(name):
        org = name.split("__")[0] if "__" in name else "other"
        mapping = {
            "meta-llama": "Meta (Llama)",
            "mistralai": "Mistral",
            "Qwen": "Qwen",
            "google": "Google",
            "microsoft": "Microsoft",
            "01-ai": "Yi",
            "deepseek-ai": "DeepSeek",
            "openai": "OpenAI",
            "NousResearch": "Nous",
            "lmsys": "LMSYS",
        }
        for key, label in mapping.items():
            if key in org:
                return label
        return "Other"

    families = overall.index.map(family)
    unique_fam = sorted(families.unique(), key=lambda x: (x == "Other", x))
    palette = {}
    base_colors = sns.color_palette("husl", len(unique_fam) - 1)
    j = 0
    for f in unique_fam:
        if f == "Other":
            palette[f] = "#bbbbbb"
        else:
            palette[f] = base_colors[j]
            j += 1

    colors = [palette[f] for f in families]

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(range(len(overall)), overall.values, color=colors, edgecolor="none", height=0.8)

    ax.set_yticks([])
    ax.set_xlabel("Mean skill mastery (across 281 skills)")
    ax.set_title(f"Overall LLM ranking (n = {len(overall)})")

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=palette[f], label=f) for f in unique_fam if f != "Other"]
    handles.append(Patch(facecolor="#bbbbbb", label="Other"))
    ax.legend(
        handles=handles, loc="lower right", fontsize=7.5,
        frameon=True, title="Model family", title_fontsize=8,
    )

    def short(n):
        parts = n.split("__")
        return parts[-1] if len(parts) > 1 else n

    for i in range(5):
        idx = len(overall) - 1 - i
        ax.text(
            overall.iloc[idx] + 0.005, idx, short(overall.index[idx]),
            va="center", fontsize=5.5, color="#333333",
        )
    for i in range(5):
        ax.text(
            overall.iloc[i] + 0.005, i, short(overall.index[i]),
            va="center", fontsize=5.5, color="#333333",
        )

    plt.tight_layout()
    out = fig_dir / "fig_llm_ranking.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Figure 6 – NCDM fit metrics
# ════════════════════════════════════════════════════════════════════


def fig_model_fit(data_dir: Path, fig_dir: Path) -> None:
    setup_style()
    with open(data_dir / "ncdm_metrics.json") as f:
        m = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    vals = [m["test_auc"], m["test_accuracy"], m["test_rmse"]]
    names = ["AUC", "Accuracy", "RMSE"]
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    for ax, v, name, col in zip(axes, vals, names, colors):
        ax.bar(0, v, color=col, width=0.45, edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(name, fontweight="bold")
        ax.text(0, v + 0.035, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
        ax.spines["bottom"].set_visible(False)

    fig.suptitle("NCDM — held-out test performance", fontsize=11, y=1.0)
    plt.tight_layout()
    out = fig_dir / "fig_model_fit.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")
