"""Polished UpSet plot for benchmark skill redundancy (fig:upset).

Reads cached redundancy JSON (no recomputation) and renders a true
UpSet plot:
  - top: bar chart of intersection sizes (skills in exactly that
    benchmark combination)
  - bottom: dot matrix indicating which benchmarks belong to each
    intersection
  - right: marginal bar of total skill coverage per benchmark
  - inline annotations carry the load-bearing numbers

All numeric content is read from
``cdm_exploration/experiments/v2_benchmark_redundancy.json``.

Output:
  figure_review_morning/01_iter_fig_benchmark_redundancy_upset.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ── Paths (absolute) ──
REPO = Path(".")
DATA = REPO / "cdm_exploration/experiments/v2_benchmark_redundancy.json"
OUT_DIR = REPO / "figure_review_morning"
OUT_PNG = OUT_DIR / "01_iter_fig_benchmark_redundancy_upset.png"
OUT_PDF = OUT_DIR / "01_iter_fig_benchmark_redundancy_upset.pdf"

# ── Locked palette (Part J1, categorical-5 per benchmark) ──
BENCH_COLOR = {
    "MATH":   "#1D4ED8",  # blue-700
    "BBH":    "#059669",  # emerald-700
    "GPQA":   "#BE185D",  # pink-700
    "MuSR":   "#7C3AED",  # violet-600
    "IFEval": "#0891B2",  # cyan-600
}
INK = "#1E293B"        # slate-800 main text
MUTE = "#94A3B8"       # slate-400 dot-off
GRID = "#E2E8F0"       # slate-200 faint grid
ACCENT = "#0F172A"     # slate-900 emphasis bars (intersection bars)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError(f"Missing data file: {DATA}")
    with open(DATA) as f:
        d = json.load(f)

    benchmarks = list(d["benchmarks"])  # ['MATH','BBH','GPQA','MuSR','IFEval']
    coverage_total = d["binary_coverage_count"]
    unique = {b: d["unique_skills"][b]["count"] for b in benchmarks}
    total_unique = sum(unique.values())
    K = int(d["K"])
    optimal = d["optimal_subsets"]
    pair_2 = optimal["2"]
    pair_2_label = " + ".join(pair_2["benchmarks"])
    pair_2_cov = int(pair_2["coverage"])

    intersections = d["multi_way_intersections"]  # already sorted desc
    # take top N for readability
    TOP_N = 12
    inter = intersections[:TOP_N]

    # ── Order benchmarks (rows in dot matrix) by total coverage desc ──
    bench_order = sorted(benchmarks, key=lambda b: -coverage_total[b])
    row_of = {b: i for i, b in enumerate(bench_order)}

    # ── Style ──
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── Figure layout ──
    # rows: [intersection bars] / [dot matrix] / [footer space]
    # cols: [dot+bar area]      / [right marginal totals]
    fig = plt.figure(figsize=(9.4, 5.6))
    gs = GridSpec(
        2, 2,
        height_ratios=[2.6, 1.55],
        width_ratios=[4.2, 1.0],
        hspace=0.10, wspace=0.06,
        left=0.085, right=0.98, top=0.90, bottom=0.18,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_dots = fig.add_subplot(gs[1, 0], sharex=ax_bars)
    ax_marg = fig.add_subplot(gs[1, 1], sharey=ax_dots)

    # ── Top: intersection size bars ──
    n = len(inter)
    x = np.arange(n)
    counts = np.array([it["n_skills"] for it in inter])

    # Color bars: solid dark for "unique-to-one-benchmark" intersections,
    # mid-grey for shared. Carries the unique-vs-shared story directly.
    bar_colors = []
    for it in inter:
        if len(it["benchmarks"]) == 1:
            b = it["benchmarks"][0]
            bar_colors.append(BENCH_COLOR[b])
        else:
            bar_colors.append("#334155")  # slate-700 for shared

    ax_bars.bar(x, counts, color=bar_colors, edgecolor="white",
                linewidth=0.6, width=0.78, zorder=3)
    for xi, c in zip(x, counts):
        ax_bars.text(xi, c + 0.35, str(int(c)), ha="center", va="bottom",
                     fontsize=8, color=INK)

    ax_bars.set_ylabel("Skills in intersection", fontsize=9.5)
    ax_bars.set_ylim(0, counts.max() * 1.55)
    ax_bars.tick_params(axis="y", labelsize=8)
    ax_bars.tick_params(axis="x", which="both", bottom=False,
                        labelbottom=False)
    ax_bars.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax_bars.set_axisbelow(True)
    ax_bars.set_xlim(-0.55, n - 0.45)

    # ── Bottom: dot matrix ──
    n_rows = len(bench_order)
    # Faint horizontal row stripes for readability
    for r in range(n_rows):
        if r % 2 == 0:
            ax_dots.axhspan(r - 0.45, r + 0.45,
                            color="#F1F5F9", zorder=0)

    for xi, it in enumerate(inter):
        active_rows = sorted(row_of[b] for b in it["benchmarks"])
        # off-dots
        for b in bench_order:
            r = row_of[b]
            if b in it["benchmarks"]:
                continue
            ax_dots.scatter(xi, r, s=46, color=MUTE, zorder=2,
                            edgecolors="none")
        # connecting line
        if len(active_rows) > 1:
            ax_dots.plot([xi, xi],
                         [min(active_rows), max(active_rows)],
                         color="#334155", lw=2.0, zorder=3,
                         solid_capstyle="round")
        # on-dots — colored by membership but solid dark to keep matrix
        # legible; we use one color (slate-900) so the row labels carry
        # the per-benchmark color identity.
        for r in active_rows:
            ax_dots.scatter(xi, r, s=70, color=ACCENT, zorder=4,
                            edgecolors="white", linewidths=0.7)

    ax_dots.set_xlim(-0.55, n - 0.45)
    ax_dots.set_ylim(n_rows - 0.5, -0.5)  # invert
    ax_dots.set_yticks(range(n_rows))
    ax_dots.set_yticklabels(bench_order, fontsize=9.5)
    # Color row labels with per-benchmark hue
    for tick, b in zip(ax_dots.get_yticklabels(), bench_order):
        tick.set_color(BENCH_COLOR[b])
        tick.set_fontweight("bold")
    ax_dots.tick_params(axis="x", which="both", bottom=False,
                        labelbottom=False)
    for sp in ["top", "right", "bottom"]:
        ax_dots.spines[sp].set_visible(False)
    ax_dots.spines["left"].set_visible(False)
    ax_dots.tick_params(axis="y", left=False)

    # ── Right marginal: total skill coverage per benchmark ──
    cov_vals = np.array([coverage_total[b] for b in bench_order])
    uniq_vals = np.array([unique[b] for b in bench_order])
    shared_vals = cov_vals - uniq_vals
    yrows = np.arange(n_rows)
    bar_h = 0.62

    # shared (lighter) + unique (saturated) stacked
    for r, b in enumerate(bench_order):
        col = BENCH_COLOR[b]
        # shared
        ax_marg.barh(r, shared_vals[r], height=bar_h,
                     color=col, alpha=0.32, edgecolor="white", linewidth=0.5,
                     zorder=3)
        # unique on top
        ax_marg.barh(r, uniq_vals[r], left=shared_vals[r], height=bar_h,
                     color=col, alpha=1.0, edgecolor="white", linewidth=0.5,
                     zorder=4)
        # numeric label: "13u / 62"
        ax_marg.text(cov_vals[r] + 2.0, r,
                     f"{uniq_vals[r]}u / {cov_vals[r]}",
                     va="center", ha="left", fontsize=8, color=INK)

    ax_marg.set_xlim(0, K)
    ax_marg.set_ylim(n_rows - 0.5, -0.5)
    ax_marg.set_xticks([0, 50, 100])
    ax_marg.set_xticklabels(["0", "50", "100"], fontsize=7.5)
    ax_marg.set_xlabel("Skills covered\n(of 100)", fontsize=8.5)
    ax_marg.tick_params(axis="y", left=False, labelleft=False)
    for sp in ["top", "right"]:
        ax_marg.spines[sp].set_visible(False)
    ax_marg.spines["left"].set_visible(False)
    ax_marg.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax_marg.set_axisbelow(True)

    # ── Headline annotations on the bars ──
    # IFEval-only is the first bar (largest); MATH+BBH+GPQA+IFEval is the
    # second (also 13). Locate them by content.
    def find_idx(target_set):
        ts = frozenset(target_set)
        for i, it in enumerate(inter):
            if frozenset(it["benchmarks"]) == ts:
                return i
        return None

    idx_ifeval = find_idx({"IFEval"})
    idx_core4 = find_idx({"MATH", "BBH", "GPQA", "IFEval"})
    idx_universal = find_idx({"MATH", "BBH", "GPQA", "MuSR", "IFEval"})

    # Annotation: IFEval-only -> max unique. Place text in upper area
    # of plot, leader line down to the bar tip.
    y_top = counts.max() * 1.55
    if idx_ifeval is not None:
        ax_bars.annotate(
            "IFEval: most unique\n(13 skills only it tests)",
            xy=(idx_ifeval + 0.35, counts[idx_ifeval] + 0.6),
            xytext=(idx_ifeval + 1.6, y_top * 0.94),
            fontsize=8.8, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                            shrinkA=2, shrinkB=2),
            ha="left", va="top",
        )

    if idx_universal is not None:
        ax_bars.annotate(
            "universal core\n(7 skills shared by all 5)",
            xy=(idx_universal, counts[idx_universal] + 0.6),
            xytext=(idx_universal + 0.6, y_top * 0.74),
            fontsize=8.8, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                            shrinkA=2, shrinkB=2),
            ha="left", va="top",
        )

    # ── Headline title strip (figure-level finding) ──
    headline = (
        f"{total_unique} of {K} skills are unique to a single benchmark; "
        f"{pair_2_label} together cover {pair_2_cov} of {K}"
    )
    fig.text(0.085, 0.955, headline, fontsize=11.5, color=INK,
             ha="left", va="bottom", fontweight="bold")
    fig.text(0.085, 0.928,
             "Top 12 of 22 non-empty intersections shown. "
             "Dark bars = skills unique to one benchmark; "
             "grey bars = shared across multiple.",
             fontsize=8.5, color="#475569", ha="left", va="bottom")

    # ── Footer summary ──
    foot_bits = []
    foot_bits.append(f"Unique-to-one-benchmark counts: " +
                     ", ".join(f"{b} {unique[b]}" for b in bench_order))
    foot_bits.append(
        f"Greedy 1->5 coverage: "
        f"MATH {optimal['1']['coverage']}, "
        f"+IFEval {optimal['2']['coverage']}, "
        f"+GPQA {optimal['3']['coverage']}, "
        f"+BBH {optimal['4']['coverage']}, "
        f"+MuSR {optimal['5']['coverage']} of {K}"
    )
    fig.text(0.085, 0.04, "\n".join(foot_bits),
             fontsize=8.0, color="#475569", ha="left", va="bottom")

    # Save
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
