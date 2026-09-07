"""Capability landscape: representative LLMs x high-variance skills.

Heatmap of per-LLM per-skill mastery for 15 representative models spanning
size and family vs the 15 highest-variance skills. Style anchored to
NEURIPS_FIGURE_CHECKLIST.md Part J (custom sequential palette built around
Okabe-Ito blue, vertical skill labels read in full, no header).

Design choices
  - Sequential, perceptually monotone colormap (custom light-grey -> Okabe-Ito
    blue #0072B2 -> deep navy). Colorblind-safe: monotone in lightness, no
    hue-only distinctions; passes a grayscale conversion.
  - Skill labels rendered vertically (rotation=90) so the FULL label fits in
    the column width without ellipses.
  - Rows sorted by mean mastery (low at top, high at bottom).
  - Columns sorted by descending mastery std (most informative skill first).
  - Cell value annotations only at the extremes (<=0.10 / >=0.92), printed
    in 6.5 pt with auto-contrast colour.
  - No suptitle (caption carries the finding per Part J6).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parent.parent

# Representative model picks: span size, family, capability tier
TARGET_MODELS = [
    # tiny / small (~1-3B)
    "meta-llama__Llama-3.2-1B-Instruct",
    "google__gemma-2-2b-it",
    "meta-llama__Llama-3.2-3B-Instruct",
    "microsoft__Phi-3-mini-4k-instruct",
    # mid (~7-9B)
    "mistralai__Mistral-7B-Instruct-v0.3",
    "meta-llama__Llama-3.1-8B-Instruct",
    "google__gemma-2-9b-it",
    # mid-large (~14-32B)
    "microsoft__Phi-3-medium-4k-instruct",
    "Qwen__Qwen2.5-14B-Instruct",
    "google__gemma-2-27b-it",
    "Qwen__Qwen2.5-32B-Instruct",
    "mistralai__Mixtral-8x7B-Instruct-v0.1",
    # large (70B+)
    "meta-llama__Llama-3.3-70B-Instruct",
    "Qwen__Qwen2.5-72B-Instruct",
    "dfurman__CalmeRys-78B-Orpo-v0.1",
]

DISPLAY = {
    "meta-llama__Llama-3.2-1B-Instruct":      "Llama-3.2-1B",
    "google__gemma-2-2b-it":                  "Gemma-2-2B",
    "meta-llama__Llama-3.2-3B-Instruct":      "Llama-3.2-3B",
    "microsoft__Phi-3-mini-4k-instruct":      "Phi-3-mini (3.8B)",
    "mistralai__Mistral-7B-Instruct-v0.3":    "Mistral-7B-v0.3",
    "meta-llama__Llama-3.1-8B-Instruct":      "Llama-3.1-8B",
    "google__gemma-2-9b-it":                  "Gemma-2-9B",
    "microsoft__Phi-3-medium-4k-instruct":    "Phi-3-medium (14B)",
    "Qwen__Qwen2.5-14B-Instruct":             "Qwen-2.5-14B",
    "google__gemma-2-27b-it":                 "Gemma-2-27B",
    "Qwen__Qwen2.5-32B-Instruct":             "Qwen-2.5-32B",
    "mistralai__Mixtral-8x7B-Instruct-v0.1":  "Mixtral-8x7B",
    "meta-llama__Llama-3.3-70B-Instruct":     "Llama-3.3-70B",
    "Qwen__Qwen2.5-72B-Instruct":             "Qwen-2.5-72B",
    "dfurman__CalmeRys-78B-Orpo-v0.1":        "CalmeRys-78B",
}

N_SKILLS = 15
N_MODELS = len(TARGET_MODELS)

# Tier assignment by parameter count (Option B: 3 horizontal bands)
TIER_OF = {
    "meta-llama__Llama-3.2-1B-Instruct":      "Small",
    "google__gemma-2-2b-it":                  "Small",
    "meta-llama__Llama-3.2-3B-Instruct":      "Small",
    "microsoft__Phi-3-mini-4k-instruct":      "Small",
    "mistralai__Mistral-7B-Instruct-v0.3":    "Mid",
    "meta-llama__Llama-3.1-8B-Instruct":      "Mid",
    "google__gemma-2-9b-it":                  "Mid",
    "microsoft__Phi-3-medium-4k-instruct":    "Mid",
    "Qwen__Qwen2.5-14B-Instruct":             "Mid",
    "google__gemma-2-27b-it":                 "Large",
    "Qwen__Qwen2.5-32B-Instruct":             "Large",
    "mistralai__Mixtral-8x7B-Instruct-v0.1":  "Large",
    "meta-llama__Llama-3.3-70B-Instruct":     "Large",
    "Qwen__Qwen2.5-72B-Instruct":             "Large",
    "dfurman__CalmeRys-78B-Orpo-v0.1":        "Large",
}
TIER_ORDER = ["Small", "Mid", "Large"]
# Use mathtext-friendly strings; entire label rendered inside $...$ to keep
# the inequality glyphs and "B" suffix in a single math run (avoids matplotlib
# rendering the LaTeX commands as literal text outside math mode).
TIER_LABEL = {
    "Small": r"Small  ($\leq\!3.8$B)",
    "Mid":   r"Mid  ($7\!-\!14$B)",
    "Large": r"Large  ($\geq\!27$B)",
}


def tidy_skill_label(label: str) -> str:
    """Normalize skill labels: strip trailing whitespace, drop redundant
    connector words. Preserves full semantic content (no ellipses)."""
    label = label.strip()
    # Collapse internal multi-spaces just in case
    label = re.sub(r"\s+", " ", label)
    # Light prefix trims to keep labels under ~50 chars without losing meaning
    label = re.sub(r"^Identifying And ", "Identifying ", label)
    return label


def make_palette() -> LinearSegmentedColormap:
    """Custom sequential palette anchored to Okabe-Ito blue (#0072B2).
    Light-grey -> mid blue -> deep navy. Monotone in luminance, so it
    survives grayscale conversion (Part B3 / G6)."""
    stops = [
        (0.00, "#F2F2F2"),  # near-white grey
        (0.20, "#CFE0EC"),  # pale blue
        (0.45, "#7FB1D3"),  # mid blue
        (0.70, "#0072B2"),  # Okabe-Ito blue (paper protagonist)
        (1.00, "#08306B"),  # deep navy
    ]
    return LinearSegmentedColormap.from_list(
        "skilleval_blue", [(p, c) for p, c in stops], N=256,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    device = resolve_device("cpu")

    print("Loading...", flush=True)
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    K = q_matrix.shape[1]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    n_llms = len(llm_names)

    cluster_labels = json.load(open(data_dir / "cluster_labels_v2_K100.json"))
    skill_names = [cluster_labels[str(k)] for k in range(K)]

    ckpt_path = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device).eval()
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).cpu().numpy()  # (n_llms, K)

    # Map representative model names to indices
    name_to_idx = {n: i for i, n in enumerate(llm_names)}
    model_idxs, model_labels = [], []
    for full in TARGET_MODELS:
        if full in name_to_idx:
            model_idxs.append(name_to_idx[full])
            model_labels.append(DISPLAY[full])
        else:
            print(f"  WARN: {full} not in pool; skipping", flush=True)
    print(f"  Found {len(model_idxs)}/{len(TARGET_MODELS)} representative models",
          flush=True)

    # High-variance skills measured across the 15 selected LLMs (not the full
    # 3,811 pool). This guarantees the columns separate THIS subset of rows
    # rather than columns that vary mostly between LLMs not on the plot.
    theta_subset = theta[np.array(model_idxs)]  # (n_models, K)
    skill_var = theta_subset.std(axis=0)
    top_var_idxs = np.argsort(-skill_var)[:N_SKILLS]
    print(f"  Top-{N_SKILLS} within-subset variance skills (std range "
          f"{skill_var[top_var_idxs[-1]]:.3f}-{skill_var[top_var_idxs[0]]:.3f})",
          flush=True)

    # Build the matrix
    M = theta_subset[:, top_var_idxs]  # (n_models, N_SKILLS)

    # Group rows by parameter-count tier (Option B). Within each tier, sort
    # by mean mastery (best at the bottom of the tier band).
    full_names_in_order = [TARGET_MODELS[i] for i in range(len(model_idxs))]
    full_names_present = [TARGET_MODELS[i] for i, idx in enumerate(model_idxs)]
    means = M.mean(axis=1)

    grouped_indices = []
    tier_band_sizes = []  # number of rows in each tier, in TIER_ORDER
    for tier in TIER_ORDER:
        in_tier = [i for i, full in enumerate(full_names_present)
                   if TIER_OF.get(full) == tier]
        in_tier_sorted = sorted(in_tier, key=lambda i: means[i])
        grouped_indices.extend(in_tier_sorted)
        tier_band_sizes.append(len(in_tier_sorted))

    M = M[grouped_indices]
    row_labels = [model_labels[i] for i in grouped_indices]
    # Cumulative band boundaries (row indices where a tier ENDS)
    band_boundaries = np.cumsum(tier_band_sizes).tolist()

    # Columns already sorted by descending variance
    col_labels = [tidy_skill_label(skill_names[i]) for i in top_var_idxs]

    # Render
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
    })

    cmap = make_palette()

    # Wider + taller figure to give vertical x-labels room without overlap.
    # Aspect handled by aspect="equal" so each cell is a square.
    fig, ax = plt.subplots(figsize=(11.0, 8.2))

    im = ax.imshow(M, aspect="equal", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    # Axes
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, ha="center", va="top")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Move x-tick labels to BOTTOM (default), keep tick marks subtle
    ax.tick_params(axis="x", which="major", length=2, pad=2)
    ax.tick_params(axis="y", which="major", length=2, pad=2)

    # No cell annotations: with within-subset variance selection most cells
    # span the full 0-1 range and the colour already encodes the value.
    # Numbers added clutter without signal in the saturated regions.

    # Faint cell separator grid (white lines on top of cells)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    # Clean horizontal separators between parameter-count tiers (Option B).
    # A single 1.6pt white line cleanly divides bands without the heavy
    # white+dark sandwich that fights with the cell colours.
    n_cols = len(col_labels)
    for b in band_boundaries[:-1]:
        ax.axhline(y=b - 0.5, color="white", linewidth=1.6, zorder=5)

    # Tier labels in the left margin (one label per band, vertically centred).
    # Use Axes-fraction y-coords mixed with data via blended transform so
    # placement is robust regardless of figure scaling. Label sits to the left
    # of the y-tick labels; the brace-like grouping is conveyed by a short
    # vertical accent line drawn just inside the heatmap edge.
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    cum = 0
    LABEL_X = -0.42   # axes-fraction x of the tier text (moved out to clear "Phi-3-medium (14B)")
    BAR_X   = -0.36   # axes-fraction x of the vertical accent bar
    for tier, n_rows in zip(TIER_ORDER, tier_band_sizes):
        if n_rows == 0:
            continue
        midpoint = cum + (n_rows - 1) / 2.0
        ax.text(LABEL_X, midpoint, TIER_LABEL[tier],
                transform=trans,
                rotation=90, ha="center", va="center",
                fontsize=10, color="#1f2937", weight="medium",
                clip_on=False)
        # Short vertical accent bar grouping the band's rows
        top = cum - 0.45
        bot = cum + n_rows - 1 + 0.45
        ax.plot([BAR_X, BAR_X], [top, bot],
                transform=trans,
                color="#9ca3af", linewidth=1.4,
                solid_capstyle="round", clip_on=False, zorder=4)
        cum += n_rows

    # Hide top/right spines, thin remaining spines
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.6)
        ax.spines[sp].set_color("#666666")

    # Colorbar — short, snug to the heatmap
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012, aspect=22,
                        ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("Skill mastery $\\theta$", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#666666")

    ax.set_xlabel(
        f"Skills (top {N_SKILLS} by mastery variance across the {N_MODELS} LLMs shown)",
        fontsize=9.5, labelpad=8,
    )

    # Reserve enough left margin for the tier labels; tight_layout alone is not
    # reliable when labels live in negative axes-fraction space.
    plt.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.24)

    out_pdf = fig_dir / "main_ready" / "fig_capability_landscape.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.10)
    print(f"Saved: {out_pdf}")
    print(f"PNG:   {str(out_pdf).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
