"""S-tier polish of fig_alignment_tax_skills.

Forks the per-skill alignment-tax figure from tools/run_alignment_tax.py
(Fig 1 there) and upgrades it for the paper. Reads ALL
numbers (per-benchmark Wilcoxon p, per-skill mean Delta + SEM,
Bonferroni-significance flags) from the existing
cdm_exploration/experiments/v2_alignment_tax.json so the figure stays in
lockstep with the paper's verified results. Primary-benchmark assignment
per skill is recomputed from the Q-matrix and items metadata.

Improvements vs the production figure:
  * 38/62 panel split (was ~25/75) so panel (a) is no longer cramped.
  * Panel (a): horizontal bars enlarged, asterisk significance code
    (***, **, *, n.s.) printed in bold next to each bar; raw p-value kept
    as small subscript context. The IFEval bar is annotated with a
    headline call-out so the load-bearing finding lands instantly.
  * Panel (b): replaces hatched fills with white-fill + colored-edge for
    non-significant bars, solid-fill + dark edge for Bonferroni-significant
    bars (alpha = 5e-4). Strong contrast at print size.
  * Skill labels shortened with display-only prefix trimming when the
    prefix is non-load-bearing (e.g. "Applying " / "Calculating ").
  * Bonferroni threshold annotated below panel (b) legend.
  * No in-figure title or subtitle: that detail belongs in the LaTeX
    \\caption{} so figure does not duplicate it.

Outputs (NEW path, never overwrites production):
  figure_review_morning/01_iter_fig_alignment_tax_skills.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────
REPO = Path(".")
DATA = REPO / "cdm_exploration/data/cdm_ready"
ALIGN_JSON = REPO / "cdm_exploration/experiments/v2_alignment_tax.json"
OUT_DIR = REPO / "figure_review_morning"
OUT_PNG = OUT_DIR / "01_iter_fig_alignment_tax_skills.png"
OUT_PDF = OUT_DIR / "01_iter_fig_alignment_tax_skills.pdf"

# ─── Style ─────────────────────────────────────────────────────────────────
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
BENCH_COLORS = {
    "MATH":   "#0072B2",  # blue
    "BBH":    "#E69F00",  # orange
    "GPQA":   "#009E73",  # bluish green
    "MuSR":   "#CC79A7",  # reddish purple
    "IFEval": "#56B4E9",  # sky blue
}

# Display-only label fixes: original cluster labels truncated to 50 chars upstream.
LABEL_FIXES = {
    "Applying Inequality Theorems To Minimize Expressio": "Applying Inequality Theorems to Minimize Expressions",
    "Applying Linear Combination Of Operators In Quantu": "Applying Linear Combinations of Operators in Quantum Mechanics",
    "Evaluating Sporting Event Plausibility Based On Co": "Evaluating Sporting Event Plausibility Based on Context",
    "Applying Trigonometric Identities To Solve Equatio": "Applying Trigonometric Identities to Solve Equations",
    "Analyzing Geometric Interpretations Of Matrix Oper": "Analyzing Geometric Interpretations of Matrix Operations",
    "Applying Properties Of Rectangles To Find Dimensio": "Applying Properties of Rectangles to Find Dimensions",
    "Identifying Structural Isomers In Organic Compound": "Identifying Structural Isomers in Organic Compounds",
    "Evaluating Climate Control System Activation Condi": "Evaluating Climate Control System Activation Conditions",
    "Crafting Concise Descriptions Of Professional Acti": "Crafting Concise Descriptions of Professional Activities",
    "Embedding Comments With Security Keywords In Code ": "Embedding Comments with Security Keywords in Code",
    "Articulating Customer Rights Under Consumer Protec": "Articulating Customer Rights Under Consumer Protection",
    "Calculating Compound Interest With Quarterly Compo": "Calculating Compound Interest with Quarterly Compounding",
    "Applying Maxwell Equations To Electromagnetic Fiel": "Applying Maxwell's Equations to Electromagnetic Fields",
    "Applying Synonyms To Replace Restricted Vocabulary": "Applying Synonyms to Replace Restricted Vocabulary",
    "Analyzing Historical Significance Of Chess Program": "Analyzing Historical Significance of Chess Programs",
    "Applying Pattern Recognition To Algorithmic Output": "Applying Pattern Recognition to Algorithmic Outputs",
    "Calculating Future Dates From Anniversary Informat": "Calculating Future Dates from Anniversary Information",
    "Applying Random Variable Concepts To Sleep Pattern": "Applying Random Variable Concepts to Sleep Patterns",
    "Applying Discrete Group Approximations To Gauge Th": "Applying Discrete Group Approximations to Gauge Theory",
    "Calculating Oscillation Frequency Of Quantum Parti": "Calculating Oscillation Frequency of Quantum Particles",
    "Evaluating Possible Storage Options Based On Story": "Evaluating Possible Storage Options Based on Story",
    "Applying Mott Gurney Equation To Device Characteri": "Applying Mott-Gurney Equation to Device Characterization",
    "Applying Properties Of Complex Numbers In Equation": "Applying Properties of Complex Numbers in Equations",
    "Calculating Handshake Combinations In Group Intera": "Calculating Handshake Combinations in Group Interactions",
    "Ensuring Consistent Section Headers In Document St": "Ensuring Consistent Section Headers in Document Structure",
    "Interpreting Luminescence Behavior In Zinc Silicat": "Interpreting Luminescence Behavior in Zinc Silicates",
    "Balancing Instrumentation For Optimal Sound Qualit": "Balancing Instrumentation for Optimal Sound Quality",
    "Identifying Transformations In Quantum Field Theor": "Identifying Transformations in Quantum Field Theory",
}


def _clean(name: str) -> str:
    return LABEL_FIXES.get(name, name)


def _sig_code(p: float) -> str:
    """Return asterisk significance code in mathmode-safe form."""
    if p < 1e-3:
        return r"$\bf{***}$"
    if p < 1e-2:
        return r"$\bf{**}$"
    if p < 5e-2:
        return r"$\bf{*}$"
    return r"$\rm{n.s.}$"


def _p_text(p: float) -> str:
    """Compact, formatted p-value in math mode."""
    if p < 1e-3:
        # floor exponent (e.g. p=2.57e-11 -> 10^-11)
        exp = int(np.floor(np.log10(max(p, 1e-300))))
        return rf"$p\!<\!10^{{{exp}}}$"
    if p < 1e-2:
        return rf"$p\!=\!{p:.3f}$"
    return rf"$p\!=\!{p:.2f}$"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading verified JSON ...", flush=True)
    align = json.loads(ALIGN_JSON.read_text())
    K = int(align["K"])
    n_pairs = int(align["n_pairs"])

    per_bench_sem = align["statistical_tests"]["per_benchmark_sem"]
    per_skill = align["statistical_tests"]["per_skill"]
    n_bonf = int(align["statistical_tests"]["n_significant_bonferroni"])
    bonf_alpha = 0.05 / K  # 5e-4

    # Per-skill arrays in skill-index order
    mean_delta = np.array([per_skill[k]["mean_delta"] for k in range(K)])
    sem_per_skill = np.array([per_skill[k]["sem"] for k in range(K)])
    sig_bonf = np.array([per_skill[k]["sig_bonferroni"] for k in range(K)])
    skill_names = [per_skill[k]["name"] for k in range(K)]

    # Recompute primary benchmark per skill from Q-matrix + items metadata
    print("Loading Q-matrix to compute primary benchmark per skill ...",
          flush=True)
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")
    with open(DATA / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    skill_bench_counts = np.zeros((K, len(BENCH_ORDER)), dtype=int)
    for i, it in enumerate(items_data):
        b = it.get("benchmark")
        if b in BENCH_ORDER:
            bi = BENCH_ORDER.index(b)
            skill_bench_counts[:, bi] += q_matrix[i].astype(int)
    primary_bench_idx = skill_bench_counts.argmax(axis=1)
    primary_bench_idx[skill_bench_counts.sum(axis=1) == 0] = -1

    # Sort skills for top-improved / top-degraded selection
    sorted_idx = np.argsort(-mean_delta)
    n_show = 10
    top_up = sorted_idx[:n_show]
    top_down = sorted_idx[-n_show:][::-1]

    # ─── Figure ────────────────────────────────────────────────────────────
    # Width ratios 38/62. Shorter canvas so panel (a)'s 5 bars dominate
    # vertically rather than feel sparse.
    fig, axes = plt.subplots(
        1, 2, figsize=(16.5, 6.2),
        gridspec_kw={"width_ratios": [0.42, 0.58], "wspace": 1.15},
    )

    # ════════════════════════════════════════════════════════════════════════
    # Panel (a): per-benchmark mean Δ ± SEM with asterisk significance code
    # ════════════════════════════════════════════════════════════════════════
    ax_a = axes[0]
    means_a = np.array([per_bench_sem[b]["mean"] for b in BENCH_ORDER])
    sems_a = np.array([per_bench_sem[b]["sem"] for b in BENCH_ORDER])
    ps_a = np.array([per_bench_sem[b]["p"] for b in BENCH_ORDER])
    y_pos = np.arange(len(BENCH_ORDER))
    colors_a = [BENCH_COLORS[b] for b in BENCH_ORDER]

    # IFEval row gets a thicker border to draw the eye to the load-bearing finding
    ifeval_i = BENCH_ORDER.index("IFEval")
    edges = ["#1a1a1a"] * len(BENCH_ORDER)
    lws = [1.0] * len(BENCH_ORDER)
    edges[ifeval_i] = "#0a0a0a"
    lws[ifeval_i] = 1.8
    for i in range(len(BENCH_ORDER)):
        ax_a.barh(
            y_pos[i], means_a[i], xerr=sems_a[i],
            color=colors_a[i], alpha=0.92,
            edgecolor=edges[i], linewidth=lws[i],
            capsize=5, height=0.82,
            error_kw={"elinewidth": 1.1, "capthick": 1.1, "ecolor": "#222"},
        )
    ax_a.axvline(0, color="black", lw=0.8)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(BENCH_ORDER, fontsize=12, fontweight="medium")
    ax_a.set_xlabel(r"Mean $\Delta$ (instruct $-$ base)",
                    fontsize=11, labelpad=6)
    # Panel marker only — descriptive text lives in the LaTeX caption
    ax_a.text(0.0, 1.02, "(a)", transform=ax_a.transAxes,
              ha="left", va="bottom", fontsize=13, fontweight="bold")

    # Asterisk-only annotation column (drop p-values; the convention is
    # standard and the caption + footer already record the Bonferroni
    # threshold, so dual-encoding only added visual collision).
    x_min_data = float((means_a - sems_a).min())
    x_max_data = float((means_a + sems_a).max())
    span = x_max_data - x_min_data
    ax_a.set_xlim(x_min_data - span * 0.10, x_max_data + span * 0.30)
    code_x = x_max_data + span * 0.10
    for i, (m, s, p) in enumerate(zip(means_a, sems_a, ps_a)):
        code = _sig_code(p)
        code_size = 18 if i == ifeval_i else 13
        text_color = "#0a0a0a" if i == ifeval_i else "#222222"
        ax_a.text(code_x, i, code,
                  ha="left", va="center", fontsize=code_size,
                  color=text_color, fontweight="bold")

    ax_a.invert_yaxis()
    for sp in ["top", "right"]:
        ax_a.spines[sp].set_visible(False)
    ax_a.grid(axis="x", ls=":", lw=0.5, alpha=0.35)

    # ════════════════════════════════════════════════════════════════════════
    # Panel (b): top-10 improved + top-10 degraded skills.
    # Solid fill+dark edge = Bonferroni-significant; white fill+colored edge
    # = not significant.
    # ════════════════════════════════════════════════════════════════════════
    ax_b = axes[1]
    show_idx = np.concatenate([top_down, top_up])  # degraded on top
    show_delta = mean_delta[show_idx]
    show_sem = sem_per_skill[show_idx]
    show_names = [_clean(skill_names[k]) for k in show_idx]
    show_sig = sig_bonf[show_idx]
    show_pb = primary_bench_idx[show_idx]
    bar_colors_b = [
        BENCH_COLORS[BENCH_ORDER[pb]] if pb >= 0 else "#888888"
        for pb in show_pb
    ]

    y_b = np.arange(len(show_idx))
    for i, (d, s, c, sig) in enumerate(zip(show_delta, show_sem,
                                             bar_colors_b, show_sig)):
        if sig:
            ax_b.barh(
                i, d, xerr=s, color=c, alpha=0.95,
                edgecolor="#0a0a0a", linewidth=1.3, capsize=3, height=0.78,
                error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#222"},
            )
        else:
            # White fill + colored edge — clearly distinct from solid sig bars
            ax_b.barh(
                i, d, xerr=s, facecolor="white",
                edgecolor=c, linewidth=1.6, capsize=3, height=0.78,
                error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#888"},
            )

    # Divider between degraded (rows 0..n_show-1) and improved (rows n_show..)
    ax_b.axhline(n_show - 0.5, color="#999999", lw=0.9, ls="-", zorder=1)
    ax_b.axvline(0, color="black", lw=0.8)

    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(show_names, fontsize=9.5)
    ax_b.set_xlabel(r"Mean $\Delta$ (instruct $-$ base)",
                    fontsize=11, labelpad=6)
    # Panel marker only — descriptive text lives in the LaTeX caption
    ax_b.text(0.0, 1.02, "(b)", transform=ax_b.transAxes,
              ha="left", va="bottom", fontsize=13, fontweight="bold")
    ax_b.invert_yaxis()
    for sp in ["top", "right"]:
        ax_b.spines[sp].set_visible(False)
    ax_b.grid(axis="x", ls=":", lw=0.5, alpha=0.35)

    # Tighten ymargin so bars feel thicker, after invert_yaxis is in effect.
    ax_b.set_ylim(len(show_idx) - 0.45, -0.55)

    # Region labels anchored INSIDE the axes at the right edge. Top-half rows
    # (0..n_show-1) are the degraded skills; bottom-half rows are the improved
    # skills (y-axis is inverted, so row 0 sits at visual top). Use axes
    # fraction so labels are robust to data-range changes.
    ax_b.text(0.995, 0.985,
              "DEGRADED  $\\downarrow$", ha="right", va="top",
              fontsize=10, color="#7a3a1a", fontweight="bold", alpha=0.95,
              transform=ax_b.transAxes,
              bbox=dict(facecolor="white", edgecolor="#7a3a1a",
                        alpha=0.85, pad=2.5, linewidth=0.6))
    ax_b.text(0.995, 0.015,
              "IMPROVED  $\\uparrow$", ha="right", va="bottom",
              fontsize=10, color="#1a4a7a", fontweight="bold", alpha=0.95,
              transform=ax_b.transAxes,
              bbox=dict(facecolor="white", edgecolor="#1a4a7a",
                        alpha=0.85, pad=2.5, linewidth=0.6))

    # ─── Single-row legend below panel (b) ────────────────────────────────
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BENCH_COLORS[b],
                      edgecolor="#222222", linewidth=0.8, label=b)
        for b in BENCH_ORDER
    ] + [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4C72B0",
                      edgecolor="#0a0a0a", linewidth=1.3,
                      label=r"Bonferroni-sig."),
        plt.Rectangle((0, 0), 1, 1, facecolor="white",
                      edgecolor="#4C72B0", linewidth=1.6,
                      label="Not sig."),
    ]
    ax_b.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.085),
        fontsize=9, frameon=False, ncol=len(legend_handles),
        handletextpad=0.4, columnspacing=1.2,
    )

    # Footer note: combined context line spanning the full width so it reads
    # as one strip and pairs with the single-row legend.
    fig.text(
        0.5, 0.012,
        rf"Primary benchmark color $|$ "
        rf"$n_{{\mathrm{{base-instruct\ pairs}}}}\!=\!{n_pairs}$ $|$ "
        rf"Wilcoxon signed-rank, two-sided $|$ "
        rf"$\alpha\!=\!5\!\times\!10^{{-4}}$ (Bonferroni) $|$ "
        rf"$n_{{\mathrm{{Bonf\!-\!sig}}}}\!=\!{n_bonf}/100$",
        ha="center", va="bottom", fontsize=8.5, color="#666666",
    )

    fig.subplots_adjust(left=0.055, right=0.965, top=0.91, bottom=0.16)
    # Avoid bbox_inches="tight" — it clobbers our manual margins and shifts
    # the footer text out of frame.
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Saved {OUT_PNG}", flush=True)
    print(f"Saved {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
