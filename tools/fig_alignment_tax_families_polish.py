"""S-tier polish of fig_alignment_tax_families.

Forks the family-heatmap section of tools/run_alignment_tax.py.
Adds (1) benchmark-grouped column ordering with a top color strip,
(2) a top marginal bar of cross-family per-skill mean delta,
(3) symmetric diverging palette clipped at the 95th percentile of |delta|,
(4) a right marginal bar of per-family overall mean delta,
(5) cleaner row labels with n-pairs.

Output: figure_review_morning/01_iter_fig_alignment_tax_families.{png,pdf}
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import torch

# Make sibling cdmeval package importable
sys.path.insert(0, ".")

from cdmeval.modeling.text_conditioned import TextConditionedNet  # noqa: E402

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(".")
DATA = ROOT / "cdm_exploration/data/cdm_ready"
CKPT = ROOT / "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"
OUT_DIR = ROOT / "figure_review_morning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

BENCH_ORDER = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
# Okabe-Ito palette
BENCH_COLORS = {
    "MATH":   "#0072B2",  # blue
    "BBH":    "#E69F00",  # orange
    "GPQA":   "#009E73",  # bluish green
    "MuSR":   "#CC79A7",  # reddish purple
    "IFEval": "#56B4E9",  # sky blue
}

# Diverging palette: vermillion (negative) ↔ near-white ↔ blue (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_div",
    [
        (0.00, "#7B1A0A"),  # deep vermillion
        (0.20, "#D55E00"),  # vermillion (Okabe-Ito)
        (0.45, "#F4D9C8"),
        (0.50, "#FFFFFF"),
        (0.55, "#CDE2F0"),
        (0.80, "#0072B2"),  # blue (Okabe-Ito)
        (1.00, "#0A3F66"),
    ],
    N=256,
)


def detect_family(name: str) -> str:
    nl = name.lower()
    for fam in ["llama", "qwen", "mistral", "gemma", "phi", "falcon",
                "yi", "solar", "olmo", "stablelm", "granite"]:
        if fam in nl:
            return fam
    return "other"


def find_pairs(names):
    aligned_markers = ["-sft", "_sft", "-dpo", "_dpo", "-mdpo", "-orpo",
                       "-ipo", "-simpo", "-kto", "-rlaif", "-rlhf", "_rlhf",
                       "-tuned", "-aligned"]

    def norm(n):
        return n.replace("__", "/").lower()

    def is_aligned(nl):
        return any(m in nl for m in aligned_markers)

    suffixes = ["-instruct", "_instruct", "-chat", "-it", ".instruct", "-rlhf"]
    pairs, seen = [], set()
    for idx, name in enumerate(names):
        nl = norm(name)
        for suf in suffixes:
            if suf in nl:
                pos = nl.rfind(suf)
                base_nl = nl[:pos] + nl[pos + len(suf):]
                for idx2, name2 in enumerate(names):
                    if idx2 == idx:
                        continue
                    nl2 = norm(name2)
                    if nl2 == base_nl or nl2.replace("-", "") == base_nl.replace("-", ""):
                        if is_aligned(nl2):
                            continue
                        key = tuple(sorted([idx, idx2]))
                        if key not in seen:
                            seen.add(key)
                            pairs.append({"base_idx": idx2, "inst_idx": idx,
                                          "base_name": name2, "inst_name": name})
    return pairs


def main():
    print("Loading inputs ...", flush=True)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")
    with open(DATA / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(DATA / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(DATA / "cluster_labels_v2_K100.json") as f:
        skill_labels = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    skill_names = [skill_labels[str(i)] for i in range(K)]

    print("Loading theta from checkpoint ...", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    state = torch.load(CKPT, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        net.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        net.load_state_dict(state["state_dict"])
    else:
        net.load_state_dict(state)
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).numpy()

    print("Building pairs and deltas ...", flush=True)
    pairs = find_pairs(llm_names)
    for p in pairs:
        p["family"] = detect_family(p["base_name"])
    deltas = np.zeros((len(pairs), K))
    for i, p in enumerate(pairs):
        deltas[i] = theta[p["inst_idx"]] - theta[p["base_idx"]]
    print(f"  {len(pairs)} pairs, K={K}", flush=True)

    family_deltas = defaultdict(list)
    for i, p in enumerate(pairs):
        family_deltas[p["family"]].append(deltas[i])

    # Alphabetical, but pin "other" to the bottom since it is a residual
    # bucket of unmatched-family pairs and shouldn't sit between named
    # families.
    _named = sorted(f for f in family_deltas
                    if f != "other" and len(family_deltas[f]) >= 3)
    fam_order = _named + (["other"]
                          if "other" in family_deltas
                             and len(family_deltas["other"]) >= 3
                          else [])
    print(f"  families kept (n>=3 pairs): {fam_order}", flush=True)

    heat = np.zeros((len(fam_order), K))
    fam_n = {}
    for fi, fam in enumerate(fam_order):
        fd = np.array(family_deltas[fam])
        heat[fi] = fd.mean(axis=0)
        fam_n[fam] = len(family_deltas[fam])

    # Primary benchmark per skill (argmax over per-benchmark item Q-column counts)
    skill_bench_counts = np.zeros((K, len(BENCH_ORDER)), dtype=int)
    for i, it in enumerate(items_data):
        b = it.get("benchmark")
        if b in BENCH_ORDER:
            bi = BENCH_ORDER.index(b)
            skill_bench_counts[:, bi] += q_matrix[i].astype(int)
    primary = skill_bench_counts.argmax(axis=1)
    primary[skill_bench_counts.sum(axis=1) == 0] = -1

    cross_fam_mean = heat.mean(axis=0)

    # Group columns by primary benchmark, sorted within each group by cross_fam_mean
    col_order = []
    group_spans = []  # (bench_idx_in_BENCH_ORDER, start, end, label)
    cur = 0
    for bi, b in enumerate(BENCH_ORDER):
        idxs = np.where(primary == bi)[0]
        if len(idxs) == 0:
            continue
        idxs_sorted = idxs[np.argsort(cross_fam_mean[idxs])]
        col_order.extend(idxs_sorted.tolist())
        group_spans.append((bi, cur, cur + len(idxs_sorted), b))
        cur += len(idxs_sorted)
    # Append unassigned (if any) at the end
    other = np.where(primary == -1)[0]
    if len(other):
        col_order.extend(other.tolist())
        group_spans.append((-1, cur, cur + len(other), "other"))
        cur += len(other)
    col_order = np.array(col_order)
    n_cols = len(col_order)

    heat_o = heat[:, col_order]
    cross_o = cross_fam_mean[col_order]
    primary_o = primary[col_order]

    # Per-family mean (right marginal)
    fam_mean = heat.mean(axis=1)

    # Symmetric vmax = 95th percentile of |delta|
    vmax = float(np.percentile(np.abs(heat_o), 95))
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # ─── Figure layout ───────────────────────────────────────────────────────
    n_rows = len(fam_order)
    # Make the figure tall enough that each row gets ~0.6" vertical space
    row_h = 0.55
    main_h = max(2.4, n_rows * row_h)
    fig_h = 1.2 + 1.0 + main_h + 1.0  # title + top marginal + main + bottom
    fig = plt.figure(figsize=(16.5, fig_h))

    # GridSpec: top strip (bench color), top marginal (cross-fam mean bar),
    # main heatmap, with right marginal column for per-family mean bar.
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        nrows=3, ncols=5,
        height_ratios=[0.22, 0.85, main_h],
        # cols: main heatmap | spacer | colorbar | spacer-for-cb-labels | right marginal
        width_ratios=[1.0, 0.025, 0.018, 0.045, 0.16],
        hspace=0.06, wspace=0.0,
        left=0.115, right=0.965, top=0.90, bottom=0.08,
    )
    ax_strip = fig.add_subplot(gs[0, 0])
    ax_top   = fig.add_subplot(gs[1, 0], sharex=ax_strip)
    ax_main  = fig.add_subplot(gs[2, 0], sharex=ax_strip)
    cax      = fig.add_subplot(gs[2, 2])
    # NOTE: do NOT use sharey=ax_main here. matplotlib's sharey machinery hides
    # the tick labels on one of the linked axes, and in our layout it ends up
    # hiding ax_main's row labels. Manually align the y-limits below instead.
    ax_right = fig.add_subplot(gs[2, 4])

    # -- Bench color strip (groups columns) --
    strip = np.zeros((1, n_cols, 3))
    for bi, s, e, b in group_spans:
        col = BENCH_COLORS.get(b, "#888888")
        rgb = matplotlib.colors.to_rgb(col)
        strip[0, s:e, :] = rgb
    ax_strip.imshow(strip, aspect="auto", interpolation="nearest",
                    extent=(-0.5, n_cols - 0.5, 0, 1))
    ax_strip.set_xlim(-0.5, n_cols - 0.5)
    ax_strip.set_yticks([])
    ax_strip.set_xticks([])
    for sp in ax_strip.spines.values():
        sp.set_visible(False)
    # Group labels centered on each span (n = number of skills primary to that benchmark)
    for bi, s, e, b in group_spans:
        if b == "other":
            continue
        n_sk = e - s
        # Show benchmark name; suppress count if span too narrow to fit text
        label = f"{b}  ({n_sk} skills)" if n_sk >= 5 else b
        ax_strip.text((s + e - 1) / 2.0, 0.5, label,
                      ha="center", va="center",
                      fontsize=10, fontweight="bold",
                      color="white" if b in ("MATH", "GPQA", "MuSR")
                      else "#1a1a1a")

    # -- Top marginal: cross-family per-skill mean delta as colored bars --
    bar_colors = [DIVERGING(norm(v)) for v in cross_o]
    ax_top.bar(np.arange(n_cols), cross_o, width=1.0,
               color=bar_colors, edgecolor="none")
    ax_top.axhline(0, color="#444", lw=0.6)
    # Add light vertical separators between benchmark groups
    for bi, s, e, b in group_spans[:-1]:
        ax_top.axvline(e - 0.5, color="#cccccc", lw=0.6, ls=":")
    ax_top.set_xlim(-0.5, n_cols - 0.5)
    y_extreme = max(abs(cross_o.min()), abs(cross_o.max())) * 1.15
    ax_top.set_ylim(-y_extreme, y_extreme)
    ax_top.set_ylabel("Cross-family\n" + r"mean $\Delta$",
                      fontsize=8.5, labelpad=4)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=7.5)
    for sp in ["top", "right"]:
        ax_top.spines[sp].set_visible(False)
    ax_top.grid(axis="y", ls=":", lw=0.4, alpha=0.4)

    # -- Main heatmap --
    im = ax_main.imshow(heat_o, aspect="auto", cmap=DIVERGING,
                        norm=norm, interpolation="nearest")
    # vertical separators between benchmark groups
    for bi, s, e, b in group_spans[:-1]:
        ax_main.axvline(e - 0.5, color="#666666", lw=0.7)
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels(
        [f"{f}  (n={fam_n[f]})" for f in fam_order],
        fontsize=11, fontweight="medium",
    )
    ax_main.tick_params(axis="y", which="both", left=True, labelleft=True,
                        length=3.0)
    ax_main.set_xticks([])
    ax_main.set_xlabel(
        "Skills, grouped by primary benchmark, sorted within group by cross-family mean $\\Delta$",
        fontsize=10, labelpad=8,
    )
    for sp in ["top", "right"]:
        ax_main.spines[sp].set_visible(False)

    # -- Colorbar -- (shrink the cax so end-tick labels don't clip the figure)
    cax_pos = cax.get_position()
    pad_v = cax_pos.height * 0.10
    cax.set_position([cax_pos.x0, cax_pos.y0 + pad_v,
                      cax_pos.width, cax_pos.height - 2 * pad_v])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Mean $\Delta$ (instruct $-$ base)", fontsize=9.5, labelpad=4)
    # Symmetric ticks at 0 and ±vmax/2 and ±vmax
    ticks = [-vmax, -vmax / 2, 0, vmax / 2, vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:+.2f}" if t != 0 else "0.00" for t in ticks])
    cb.ax.tick_params(labelsize=8.5)
    cb.outline.set_linewidth(0.5)

    # -- Right marginal: per-family overall mean --
    bar_c = [DIVERGING(norm(v)) for v in fam_mean]
    ax_right.barh(np.arange(n_rows), fam_mean, height=0.8,
                  color=bar_c, edgecolor="#222", linewidth=0.4)
    ax_right.axvline(0, color="#444", lw=0.6)
    # Manually align y-limits with ax_main (we deliberately avoid sharey)
    ax_right.set_ylim(n_rows - 0.5, -0.5)
    x_ext = max(abs(fam_mean.min()), abs(fam_mean.max())) * 1.25
    ax_right.set_xlim(-x_ext, x_ext)
    ax_right.set_yticks([])
    ax_right.set_xlabel("Row mean", fontsize=8.5, labelpad=4)
    ax_right.tick_params(axis="x", labelsize=7.5)
    for sp in ["top", "right"]:
        ax_right.spines[sp].set_visible(False)
    ax_right.grid(axis="x", ls=":", lw=0.4, alpha=0.4)
    # Numeric annotations
    for i, v in enumerate(fam_mean):
        ax_right.text(v + (0.012 * np.sign(v) if v != 0 else 0.012), i,
                      f"{v:+.3f}", ha="left" if v >= 0 else "right",
                      va="center", fontsize=7.5, color="#222")

    # No in-figure title or subtitle: that detail belongs in the LaTeX
    # \caption{}. Only the figure axes, panels, and colorbar remain.

    out_png = OUT_DIR / "01_iter_fig_alignment_tax_families.png"
    out_pdf = OUT_DIR / "01_iter_fig_alignment_tax_families.pdf"
    # Note: do NOT use bbox_inches="tight" because it confuses our manual GridSpec
    # left margin and ends up clipping the y tick labels.
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)
    print(f"Saved {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
