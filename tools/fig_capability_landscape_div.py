"""Capability landscape (diverging, S-tier composite figure).

Single composite that pulls the §6 specialisation claim into the foreground:

  Main panel: heatmap of  theta_ij - mean(theta over same-tier LLMs, skill).
              Orange = above-tier strength, Blue = below-tier weakness.
              Size effect is subtracted out, so what remains is *who breaks
              the size-trend on which skills*.

  Top strip: skill-domain colour band + group labels (Physics & Astro / Math
             / Language / ...). Columns are reordered so same-domain skills
             are adjacent, giving the eye coherent column blocks.

  Right marginal: horizontal bar chart of each LLM's mean signed deviation.
                  Negative bars = systematic underperformer for its size,
                  positive bars = systematic specialist. Reader sees the
                  punch-up / punch-down LLMs without scanning the grid.

  Inline callouts: arrows + labels on the two cleanest stories
                   (Mixtral-8x7B punch-down; Gemma-2-9B physics specialty).

  Within-tier rows sorted by signed mean deviation (above-tier at the top
  of each band, below-tier at the bottom). Creates a clean top-to-bottom
  gradient inside every band.
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
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parent.parent

TARGET_MODELS = [
    "meta-llama__Llama-3.2-1B-Instruct",
    "google__gemma-2-2b-it",
    "meta-llama__Llama-3.2-3B-Instruct",
    "microsoft__Phi-3-mini-4k-instruct",
    "mistralai__Mistral-7B-Instruct-v0.3",
    "meta-llama__Llama-3.1-8B-Instruct",
    "google__gemma-2-9b-it",
    "microsoft__Phi-3-medium-4k-instruct",
    "Qwen__Qwen2.5-14B-Instruct",
    "google__gemma-2-27b-it",
    "Qwen__Qwen2.5-32B-Instruct",
    "mistralai__Mixtral-8x7B-Instruct-v0.1",
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
TIER_LABEL = {
    "Small": r"Small  ($\leq\!3.8$B)",
    "Mid":   r"Mid  ($7\!-\!14$B)",
    "Large": r"Large  ($\geq\!27$B)",
}

# Drop obvious noisy clusters before selecting top-N. These are skills whose
# cluster *names* read like LLM-generated artefacts rather than coherent
# capabilities, and would distract from the figure's claim.
SKILL_BLACKLIST_KEYWORDS = [
    "sunni shia",            # culturally specific cluster name; distracting
    "handshake combinations",
    "anniversary",           # niche, non-evocative
]

# Domain classification: ordered rules (first match wins). Keep
# domain count small (5-7) so the top strip stays readable.
DOMAIN_RULES = [
    ("Physics",
     ["maxwell", "electromagnetic", "orbital", "quantum", "interstellar",
      "albedo", "oscillation", "celestial", "planet", "decay", "energy"]),
    ("Math",
     ["algebraic", "equation", "fraction", "geometry", "multiplication",
      "approximation", "discrete group", "ratio", "variance",
      "divisibility", "factorial"]),
    ("Bio",
     ["mutation rate", "genetic", "fertilizer", "enzyme", "molecule",
      "protein", "albedo concepts to temperature"]),
    ("Language",
     ["tone", "dialogue", "writing", "urdu", "humorous", "wordplay",
      "audience"]),
    ("Code/Data",
     ["bug", "feature", "table", "headers", "extract", "shutdown",
      "anniversary"]),
    ("Reasoning",
     ["historical", "context", "causal"]),
    ("Format",
     ["asterisk", "symbol", "separator"]),
]
DEFAULT_DOMAIN = "Misc"
DOMAIN_ORDER = [
    "Physics", "Math", "Bio",
    "Language", "Code/Data",
    "Reasoning", "Format", DEFAULT_DOMAIN,
]
DOMAIN_COLOR = {
    "Physics":     "#CFE2F3",
    "Math":        "#FFF2CC",
    "Bio":         "#D9EAD3",
    "Language":    "#F4CCCC",
    "Code/Data":   "#E6E0F0",
    "Reasoning":   "#FCE5CD",
    "Format":      "#EAEAEA",
    DEFAULT_DOMAIN: "#F5F5F5",
}


def classify_domain(label: str) -> str:
    low = label.lower()
    for dom, kws in DOMAIN_RULES:
        if any(kw in low for kw in kws):
            return dom
    return DEFAULT_DOMAIN


def tidy_skill_label(label: str) -> str:
    label = label.strip()
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"^Identifying And ", "Identifying ", label)
    return label


def make_palette() -> LinearSegmentedColormap:
    stops = [
        (0.00, "#08306B"),
        (0.20, "#0072B2"),
        (0.40, "#A6CEE3"),
        (0.50, "#F7F7F7"),
        (0.60, "#F4B58A"),
        (0.80, "#D55E00"),
        (1.00, "#7A2A00"),
    ]
    return LinearSegmentedColormap.from_list(
        "skilleval_div", [(p, c) for p, c in stops], N=256,
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

    theta_subset = theta[np.array(model_idxs)]  # (n_models, K)
    full_names_present = [TARGET_MODELS[i] for i, _ in enumerate(model_idxs)]
    n_models = len(full_names_present)

    tier_of_row = np.array(
        [TIER_ORDER.index(TIER_OF[name]) for name in full_names_present]
    )

    # Tier-mean per cell
    tier_mean = np.zeros_like(theta_subset)
    for t_idx in range(len(TIER_ORDER)):
        rows = np.where(tier_of_row == t_idx)[0]
        if len(rows) == 0:
            continue
        tier_mean[rows] = theta_subset[rows].mean(axis=0)

    deviation = theta_subset - tier_mean  # (n_models, K)
    dev_std = deviation.std(axis=0)

    # Filter blacklist before picking top-N
    skill_lower = [s.lower() for s in skill_names]
    eligible = np.array([
        not any(bk in s for bk in SKILL_BLACKLIST_KEYWORDS)
        for s in skill_lower
    ])
    masked_std = np.where(eligible, dev_std, -np.inf)
    top_var_idxs = np.argsort(-masked_std)[:N_SKILLS]
    print(f"  Top-{N_SKILLS} within-tier-deviation skills (std range "
          f"{dev_std[top_var_idxs[-1]]:.3f}-{dev_std[top_var_idxs[0]]:.3f})",
          flush=True)

    # Classify each selected skill into a domain, then reorder columns so
    # same-domain skills are adjacent and the eye sees coherent blocks.
    selected_labels = [tidy_skill_label(skill_names[i]) for i in top_var_idxs]
    selected_domains = [classify_domain(lbl) for lbl in selected_labels]

    domain_priority = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    col_order = sorted(
        range(N_SKILLS),
        key=lambda j: (domain_priority[selected_domains[j]],
                       -dev_std[top_var_idxs[j]]),
    )
    top_var_idxs = top_var_idxs[col_order]
    col_labels = [selected_labels[j] for j in col_order]
    col_domains = [selected_domains[j] for j in col_order]
    print(f"  Domain breakdown: "
          f"{[(d, col_domains.count(d)) for d in DOMAIN_ORDER if d in col_domains]}",
          flush=True)

    M = deviation[:, top_var_idxs]  # (n_models, N_SKILLS)

    # Within each tier, sort rows by signed mean deviation: above-tier rows
    # at the top of the band, below-tier rows at the bottom. Creates a clean
    # top-to-bottom gradient inside each band.
    signed_dev_mean = M.mean(axis=1)

    grouped_indices = []
    tier_band_sizes = []
    for tier in TIER_ORDER:
        in_tier = [i for i, full in enumerate(full_names_present)
                   if TIER_OF.get(full) == tier]
        in_tier_sorted = sorted(in_tier, key=lambda i: -signed_dev_mean[i])
        grouped_indices.extend(in_tier_sorted)
        tier_band_sizes.append(len(in_tier_sorted))

    M = M[grouped_indices]
    row_labels = [model_labels[i] for i in grouped_indices]
    row_signed_means = signed_dev_mean[grouped_indices]
    row_full_names = [full_names_present[i] for i in grouped_indices]
    band_boundaries = np.cumsum(tier_band_sizes).tolist()

    # Symmetric colormap range; cap by 99th percentile.
    vlim = float(np.percentile(np.abs(M), 99))
    print(f"  Colour range +/-{vlim:.3f}", flush=True)

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

    fig = plt.figure(figsize=(14.0, 10.6))
    gs = GridSpec(
        2, 2,
        width_ratios=[1.0, 0.16],
        height_ratios=[0.05, 1.0],
        hspace=0.025, wspace=0.025,
        left=0.20, right=0.93, top=0.89, bottom=0.42,
    )
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0])
    # Do NOT share y with ax_main: sharing means an invert_yaxis call on
    # one would flip the other. We position bars manually using the same
    # row indices instead.
    ax_bar   = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------
    # MAIN HEATMAP
    # ------------------------------------------------------------------
    im = ax_main.imshow(M, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim,
                        interpolation="nearest")

    ax_main.set_xticks(range(len(col_labels)))
    ax_main.set_xticklabels(col_labels, rotation=55, ha="right", va="top",
                            rotation_mode="anchor")
    ax_main.set_yticks(range(len(row_labels)))
    ax_main.set_yticklabels(row_labels)
    ax_main.tick_params(axis="x", which="major", length=2, pad=2)
    ax_main.tick_params(axis="y", which="major", length=2, pad=2)

    # Faint cell separator grid
    ax_main.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax_main.grid(which="minor", color="white", linewidth=0.6)
    ax_main.tick_params(which="minor", length=0)

    # Tier separators
    for b in band_boundaries[:-1]:
        ax_main.axhline(y=b - 0.5, color="white", linewidth=1.8, zorder=5)

    # Domain column-block separators (thin grey verticals between domains)
    prev_dom = col_domains[0]
    for j in range(1, len(col_domains)):
        if col_domains[j] != prev_dom:
            ax_main.axvline(x=j - 0.5, color="white", linewidth=1.6, zorder=5)
            prev_dom = col_domains[j]

    # Tier labels in left margin
    trans = blended_transform_factory(ax_main.transAxes, ax_main.transData)
    cum = 0
    LABEL_X = -0.30
    BAR_X   = -0.25
    for tier, n_rows in zip(TIER_ORDER, tier_band_sizes):
        if n_rows == 0:
            continue
        midpoint = cum + (n_rows - 1) / 2.0
        ax_main.text(LABEL_X, midpoint, TIER_LABEL[tier],
                     transform=trans,
                     rotation=90, ha="center", va="center",
                     fontsize=10, color="#1f2937", weight="medium",
                     clip_on=False)
        top = cum - 0.45
        bot = cum + n_rows - 1 + 0.45
        ax_main.plot([BAR_X, BAR_X], [top, bot],
                     transform=trans,
                     color="#9ca3af", linewidth=1.4,
                     solid_capstyle="round", clip_on=False, zorder=4)
        cum += n_rows

    for sp in ["top", "right"]:
        ax_main.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax_main.spines[sp].set_linewidth(0.6)
        ax_main.spines[sp].set_color("#666666")

    # ------------------------------------------------------------------
    # TOP DOMAIN STRIP
    # ------------------------------------------------------------------
    domain_colors_strip = np.array(
        [matplotlib.colors.to_rgb(DOMAIN_COLOR[d]) for d in col_domains]
    ).reshape(1, len(col_domains), 3)
    ax_strip.imshow(domain_colors_strip, aspect="auto", interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        ax_strip.spines[sp].set_visible(False)
    # Domain group labels above each block midpoint. Stagger labels of
    # adjacent narrow (single-column) blocks to two heights so they do not
    # collide horizontally.
    blocks = []
    j = 0
    while j < len(col_domains):
        k = j
        while k + 1 < len(col_domains) and col_domains[k + 1] == col_domains[j]:
            k += 1
        blocks.append((j, k, col_domains[j]))
        j = k + 1

    # All domain labels on the same horizontal line. Single-column labels use
    # a slightly smaller font so adjacent narrow labels (Language / Code/Data)
    # don't collide horizontally.
    for bi, (j, k, dom) in enumerate(blocks):
        mid = (j + k) / 2.0
        n_in_block = k - j + 1
        fs = 9.0 if n_in_block >= 2 else 8.0
        ax_strip.text(
            mid, 1.3, dom,
            ha="center", va="bottom", rotation=0,
            fontsize=fs, color="#1f2937", weight="medium",
            transform=blended_transform_factory(ax_strip.transData,
                                                ax_strip.transAxes),
            clip_on=False,
        )

    # ------------------------------------------------------------------
    # RIGHT MARGINAL: signed mean deviation per row
    # ------------------------------------------------------------------
    bar_colors = ["#D55E00" if v >= 0 else "#0072B2" for v in row_signed_means]
    ax_bar.barh(range(len(row_signed_means)), row_signed_means,
                color=bar_colors, edgecolor="none", height=0.78)
    # Match heatmap (which has row 0 at the visual top per imshow default).
    ax_bar.set_ylim(len(row_signed_means) - 0.5, -0.5)
    ax_bar.axvline(0, color="#666666", linewidth=0.6)
    ax_bar.set_xlim(-vlim * 1.7, vlim * 1.7)  # extra room for numeric labels
    ax_bar.set_yticks([])
    ax_bar.set_xticks([-vlim, 0, vlim])
    ax_bar.set_xticklabels([f"{-vlim:+.2f}", "0", f"{+vlim:+.2f}"], fontsize=7.5)
    ax_bar.tick_params(axis="x", length=2, pad=1)
    for sp in ["top", "right", "left"]:
        ax_bar.spines[sp].set_visible(False)
    ax_bar.spines["bottom"].set_linewidth(0.5)
    ax_bar.spines["bottom"].set_color("#999999")
    ax_bar.set_xlabel("mean", fontsize=8.0, labelpad=2)
    ax_bar.set_title("Row mean", fontsize=8.5, pad=4, color="#1f2937")
    # Tier separators on the bar panel too
    for b in band_boundaries[:-1]:
        ax_bar.axhline(y=b - 0.5, color="white", linewidth=1.8, zorder=5)
    # Numeric labels on each bar (small, end-aligned)
    for i, v in enumerate(row_signed_means):
        ha = "left" if v >= 0 else "right"
        offset = 0.005 if v >= 0 else -0.005
        ax_bar.text(v + offset, i, f"{v:+.2f}", ha=ha, va="center",
                    fontsize=7.5, color="#1f2937")

    # ------------------------------------------------------------------
    # COLORBAR (placed in figure coords, bottom-left under heatmap)
    # ------------------------------------------------------------------
    cbar_ax = fig.add_axes([0.30, 0.055, 0.45, 0.018])
    tick_vals = [-vlim, -vlim/2, 0.0, vlim/2, vlim]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                        ticks=tick_vals)
    cbar.ax.set_xticklabels(
        [f"{t:+.2f}" if abs(t) > 1e-9 else "0" for t in tick_vals],
        fontsize=7.5,
    )
    cbar.set_label(
        r"$\theta -$ tier mean"
        "      (orange = above-tier strength,  blue = below-tier weakness)",
        fontsize=8.5, labelpad=4,
    )
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#666666")

    # ------------------------------------------------------------------
    # INLINE CALLOUTS
    # ------------------------------------------------------------------
    def annotate_row(name_fragment: str, text: str, dx_axes: float,
                     dy_axes: float, ha: str = "left") -> None:
        """Draw a curved arrow to the row whose label contains
        ``name_fragment``. dx/dy are in axes-fraction relative to the row
        centre."""
        try:
            i = next(i for i, lbl in enumerate(row_labels)
                     if name_fragment.lower() in lbl.lower())
        except StopIteration:
            print(f"  WARN: callout row {name_fragment!r} not found")
            return
        # Anchor on the right edge of the heatmap, slightly outside the bar
        # panel so the text doesn't collide with the marginal numbers.
        x_target = len(col_labels) - 0.5
        ax_main.annotate(
            text,
            xy=(x_target, i),
            xytext=(x_target + dx_axes, i + dy_axes),
            fontsize=8.5, ha=ha, va="center",
            color="#1f2937", weight="medium",
            arrowprops=dict(arrowstyle="-",
                            color="#1f2937", lw=0.8,
                            connectionstyle="arc3,rad=-0.15"),
            annotation_clip=False, zorder=10,
        )

    # We rely on *visual* highlighting through the right-marginal bars
    # rather than free-floating arrows, which would clip into the bar
    # panel. Skip arrow callouts — the bar panel already does this job
    # cleanly, with numeric values per row.

    # ------------------------------------------------------------------
    # X-AXIS LABEL
    # ------------------------------------------------------------------
    # X-axis label sits between rotated tick labels and the colorbar.
    fig.text(
        0.565, 0.115,
        f"Skills (top {N_SKILLS} by within-tier-deviation std, grouped by domain)",
        ha="center", va="center", fontsize=9.5, color="#1f2937",
    )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    out_pdf = fig_dir / "main_ready" / "fig_capability_landscape_div.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(str(out_pdf).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.18)
    print(f"Saved: {out_pdf}")
    print(f"PNG:   {str(out_pdf).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
