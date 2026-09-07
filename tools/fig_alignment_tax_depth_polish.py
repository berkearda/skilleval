"""Polished alignment-tax-by-depth figure for fig:aligndepth.

Forks the depth panel from tools/run_alignment_tax.py (Fig 3 there) and
upgrades it to S-tier:

  * blue scatter (skill cloud) at low alpha as background context
  * bold mean trajectory line with circular markers, white edges
  * 95% CI band from per-depth bootstrap over the skill-level mean-delta
    distribution (per design_system N4: full sweep, no cherry-picked points)
  * zero baseline drawn as faint dashed grey line
  * sample-size labels (n=K) above each depth tick
  * annotation calling out the depth-4 transition to negative
  * Okabe-Ito palette, 6.4 x 4.2 inches, dpi=300

Outputs both PNG and PDF to figure_review_morning/.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(".")
ALIGN_JSON = REPO / "cdm_exploration/experiments/v2_alignment_tax.json"
PREREQ_JSON = REPO / "cdm_exploration/experiments/v2_skill_prerequisites.json"
OUT_DIR = REPO / "figure_review_morning"
OUT_PNG = OUT_DIR / "01_iter_fig_alignment_tax_depth.png"
OUT_PDF = OUT_DIR / "01_iter_fig_alignment_tax_depth.pdf"

# Okabe-Ito palette
COL_SCATTER = "#56B4E9"   # sky blue, low alpha
COL_LINE = "#0072B2"      # dark blue, mean trajectory
COL_MARK = "#0072B2"
COL_BAND = "#0072B2"      # CI band
COL_ZERO = "#999999"
COL_NEG_HIGHLIGHT = "#D55E00"  # vermilion for the "turns negative" call-out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000,
                      alpha: float = 0.05, rng: np.random.Generator | None = None,
                      ) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile-bootstrap CI on the mean of `values`.

    For depth bins with n=1 the CI collapses to the point estimate; we widen
    those visually downstream.
    """
    rng = rng or np.random.default_rng(42)
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) <= 1:
        return m, m, m
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return m, lo, hi


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    align = json.loads(ALIGN_JSON.read_text())
    prereq = json.loads(PREREQ_JSON.read_text())

    K = int(align["K"])
    depth_map: dict[str, int] = prereq["depth_per_skill"]
    per_skill = align["per_skill"]

    # Build (depth, mean_delta) per skill
    depths_per_skill = np.array(
        [int(depth_map[str(k)]) for k in range(K)], dtype=int
    )
    delta_per_skill = np.array(
        [per_skill[str(k)]["mean_delta"] for k in range(K)], dtype=float
    )

    # Group by depth
    depth_to_deltas: dict[int, np.ndarray] = {}
    for d in sorted(set(depths_per_skill.tolist())):
        depth_to_deltas[d] = delta_per_skill[depths_per_skill == d]

    # Bootstrap CIs per depth on the skill-level mean
    rng = np.random.default_rng(42)
    depth_levels = sorted(depth_to_deltas.keys())
    means, los, his, ns = [], [], [], []
    for d in depth_levels:
        v = depth_to_deltas[d]
        m, lo, hi = bootstrap_mean_ci(v, rng=rng)
        means.append(m)
        los.append(lo)
        his.append(hi)
        ns.append(len(v))
    means = np.array(means)
    los = np.array(los)
    his = np.array(his)
    ns = np.array(ns)

    # ── Figure ──
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Background: per-skill scatter with horizontal jitter
    jitter_rng = np.random.default_rng(7)
    jitter = jitter_rng.uniform(-0.18, 0.18, size=K)
    ax.scatter(
        depths_per_skill + jitter, delta_per_skill,
        s=22, alpha=0.28, color=COL_SCATTER,
        edgecolor="none", zorder=2, label="Individual skills",
    )

    # Zero line — faint dashed grey
    ax.axhline(0, color=COL_ZERO, lw=0.9, ls=(0, (4, 3)), zorder=1.5)

    # 95% CI band (shaded)
    ax.fill_between(
        depth_levels, los, his,
        color=COL_BAND, alpha=0.18, linewidth=0,
        zorder=3, label="95% CI (bootstrap)",
    )

    # Mean trajectory line + markers
    ax.plot(
        depth_levels, means,
        color=COL_LINE, lw=2.2, zorder=4, solid_capstyle="round",
    )
    ax.scatter(
        depth_levels, means,
        s=85, color=COL_MARK, edgecolor="white", linewidths=1.6,
        zorder=5, label="Mean per depth",
    )

    # Highlight the transition: depth where mean goes negative
    neg_depths = [d for d, m in zip(depth_levels, means) if m < 0]
    if neg_depths:
        first_neg = min(neg_depths)
        # Draw a translucent vertical band from first_neg-0.5 onward
        ax.axvspan(first_neg - 0.5, max(depth_levels) + 0.5,
                   color=COL_NEG_HIGHLIGHT, alpha=0.06, zorder=0.5)
        # Annotation
        m_first_neg = means[depth_levels.index(first_neg)]
        ax.annotate(
            "alignment turns\nnegative",
            xy=(first_neg, m_first_neg),
            xytext=(first_neg - 2.05, m_first_neg + 0.13),
            fontsize=9.2, color=COL_NEG_HIGHLIGHT, ha="left", va="bottom",
            arrowprops=dict(
                arrowstyle="->", color=COL_NEG_HIGHLIGHT, lw=1.0,
                shrinkA=2, shrinkB=5,
                connectionstyle="arc3,rad=0.25",
            ),
            zorder=6,
        )

    # Sample-size labels: pinned just under the top spine on the LEFT side
    # so they don't collide with the legend (upper-right).
    ymin = min(delta_per_skill.min(), los.min()) - 0.04
    ymax = max(delta_per_skill.max(), his.max()) + 0.14
    ax.set_ylim(ymin, ymax)
    n_y = ymax - 0.025
    for d, n in zip(depth_levels, ns):
        ax.text(
            d, n_y, f"n={n}",
            ha="center", va="top", fontsize=8.5, color="#555555",
        )

    # Axes
    ax.set_xticks(depth_levels)
    ax.set_xlabel("Skill depth in prerequisite DAG")
    ax.set_ylabel(r"Alignment $\Delta$  (instruct $-$ base $\theta$)")
    ax.set_xlim(-0.6, max(depth_levels) + 0.6)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Legend (compact, upper right but lowered slightly so it sits below
    # the n=K row of sample-size labels along the top edge).
    leg = ax.legend(
        loc="upper right", frameon=False, fontsize=8.8,
        handletextpad=0.5, borderaxespad=0.4,
        bbox_to_anchor=(1.0, 0.92),
    )

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Console summary for traceability
    print(f"n_pairs (from align json): {align['n_pairs']}")
    print("Depth | n  | mean    | 95% CI")
    for d, n, m, lo, hi in zip(depth_levels, ns, means, los, his):
        print(f"  {d:>3} | {n:>2} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}]")
    print(f"\nWrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()
