"""Batch-2 render of fig:aligntax (alignment tax 2-panel).

Reads cached `cdm_exploration/experiments/v2_alignment_tax.json` plus the
small q-matrix + items file to recover each skill's primary benchmark.

Differences from batch-1 (`tools/fig_alignment_tax_revised.py`):
  - LOCKED Tailwind-700 palette (Part J1). No orange anywhere.
  - No suptitles / no figure-level header. Panel letters `(a)` `(b)` only;
    finding goes in the LaTeX caption (Part J6 rule).
  - Less bold: panel letters are the only bold element. p-values, axis
    labels, value annotations stay regular weight.
  - Tighter layout (smaller figsize, smaller pad).
  - Negative tax bars use `#86198F` (purple-800) per Part J1 negative role.
  - Non-significant bars use redundant encoding (faded fill + hatching).

Run:
    python tools/fig_alignment_tax_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration" / "experiments"
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
OUT_DIR = REPO / "cdm_exploration" / "figures" / "review_batch2"

# Part J1 locked palette (Tailwind-700, NO ORANGE)
BENCH_COLORS = {
    "MATH":   "#1D4ED8",  # blue-700
    "BBH":    "#059669",  # emerald-700
    "GPQA":   "#BE185D",  # pink-700
    "MuSR":   "#7C3AED",  # violet-600
    "IFEval": "#0891B2",  # cyan-600
}
TAX_COLOR = "#86198F"      # purple-800 (negative / tax role)
SPINE_COLOR = "#475569"    # slate-600
NEUTRAL_GRAY = "#64748B"   # slate-500 for non-sig hatched edges

BENCH_ORDER = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]

# Display-only label fixes (truncated cluster labels).
LABEL_FIXES = {
    "Applying Inequality Theorems To Minimize Expressio": "Applying Inequality Theorems to Minimize Expressions",
    "Applying Linear Combination Of Operators In Quantu": "Applying Linear Combinations of Operators in QM",
    "Evaluating Sporting Event Plausibility Based On Co": "Evaluating Sporting Event Plausibility from Context",
    "Applying Trigonometric Identities To Solve Equatio": "Applying Trig Identities to Solve Equations",
    "Analyzing Geometric Interpretations Of Matrix Oper": "Geometric Interpretation of Matrix Operations",
    "Applying Properties Of Rectangles To Find Dimensio": "Properties of Rectangles to Find Dimensions",
    "Identifying Structural Isomers In Organic Compound": "Identifying Structural Isomers in Organic Compounds",
    "Evaluating Climate Control System Activation Condi": "Evaluating Climate Control Activation Conditions",
    "Crafting Concise Descriptions Of Professional Acti": "Concise Descriptions of Professional Activities",
    "Embedding Comments With Security Keywords In Code ": "Embedding Security-Keyword Comments in Code",
    "Articulating Customer Rights Under Consumer Protec": "Articulating Customer Rights (Consumer Protection)",
    "Calculating Compound Interest With Quarterly Compo": "Compound Interest with Quarterly Compounding",
    "Applying Maxwell Equations To Electromagnetic Fiel": "Applying Maxwell's Equations to EM Fields",
    "Applying Synonyms To Replace Restricted Vocabulary": "Synonyms to Replace Restricted Vocabulary",
    "Analyzing Historical Significance Of Chess Program": "Historical Significance of Chess Programs",
    "Applying Pattern Recognition To Algorithmic Output": "Pattern Recognition on Algorithmic Outputs",
    "Calculating Future Dates From Anniversary Informat": "Future Dates from Anniversary Information",
    "Applying Random Variable Concepts To Sleep Pattern": "Random Variable Concepts on Sleep Patterns",
    "Applying Discrete Group Approximations To Gauge Th": "Discrete Group Approximations to Gauge Theory",
    "Calculating Oscillation Frequency Of Quantum Parti": "Oscillation Frequency of Quantum Particles",
    "Evaluating Possible Storage Options Based On Story": "Possible Storage Options from Story Context",
    "Applying Mott Gurney Equation To Device Characteri": "Mott-Gurney Equation for Device Characterization",
    "Applying Properties Of Complex Numbers In Equation": "Properties of Complex Numbers in Equations",
    "Calculating Handshake Combinations In Group Intera": "Handshake Combinations in Group Interactions",
    "Ensuring Consistent Section Headers In Document St": "Consistent Section Headers in Document Structure",
    "Interpreting Luminescence Behavior In Zinc Silicat": "Luminescence Behavior in Zinc Silicates",
    "Balancing Instrumentation For Optimal Sound Qualit": "Balancing Instrumentation for Sound Quality",
    "Identifying Transformations In Quantum Field Theor": "Identifying Transformations in QFT",
    "Applying Set Theory To Count Unique Items": "Applying Set Theory to Count Unique Items",
}


def _clean(name: str) -> str:
    return LABEL_FIXES.get(name, name)


def _fmt_p(p: float) -> str:
    if p < 1e-3:
        exp = int(np.floor(np.log10(max(p, 1e-300))))
        return rf"$p{{<}}10^{{{exp}}}$"
    if p < 1e-2:
        return rf"$p{{=}}{p:.3f}$"
    return rf"$p{{=}}{p:.2f}$"


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SPINE_COLOR,
        "axes.labelcolor": SPINE_COLOR,
        "xtick.color": SPINE_COLOR,
        "ytick.color": SPINE_COLOR,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _primary_bench_per_skill(K: int) -> np.ndarray:
    q = np.load(DATA / "qmatrix_v2_K100.npy")
    with open(DATA / "response_matrix_v2_full_items.json") as f:
        items = json.load(f)
    counts = np.zeros((K, len(BENCH_ORDER)), dtype=int)
    for i, it in enumerate(items):
        b = it.get("benchmark")
        if b in BENCH_ORDER:
            bi = BENCH_ORDER.index(b)
            counts[:, bi] += q[i].astype(int)
    pb = counts.argmax(axis=1)
    pb[counts.sum(axis=1) == 0] = -1
    return pb


def main() -> None:
    src = EXP / "v2_alignment_tax.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing")
    d = json.load(open(src))

    K = int(d["K"])
    n_pairs = int(d["n_pairs"])
    per_skill_tests = d["statistical_tests"]["per_skill"]
    per_bench_sem = d["statistical_tests"]["per_benchmark_sem"]
    n_bonf = int(d["statistical_tests"]["n_significant_bonferroni"])

    mean_delta = np.array([t["mean_delta"] for t in per_skill_tests])
    sem = np.array([t["sem"] for t in per_skill_tests])
    sig_bonf = np.array([t["sig_bonferroni"] for t in per_skill_tests], dtype=bool)
    names = [t["name"] for t in per_skill_tests]

    primary_bench = _primary_bench_per_skill(K)

    # ── Layout (tighter than batch-1: 12.5 x 5.6 vs 13.5 x 6.4) ───────
    _set_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.6),
        gridspec_kw={"width_ratios": [0.85, 2.15]},
    )

    # ── Panel (a): per-benchmark pair-level mean ±SEM, with p-values
    ax_a = axes[0]
    means_a = [per_bench_sem[b]["mean"] for b in BENCH_ORDER]
    sems_a = [per_bench_sem[b]["sem"] for b in BENCH_ORDER]
    ps_a = [per_bench_sem[b]["p"] for b in BENCH_ORDER]

    bonf_alpha = 0.05 / K
    y_pos = np.arange(len(BENCH_ORDER))
    for i, (b, m, s, p) in enumerate(zip(BENCH_ORDER, means_a, sems_a, ps_a)):
        c = BENCH_COLORS[b]
        if p < bonf_alpha:
            ax_a.barh(i, m, xerr=s, color=c, alpha=0.92,
                      edgecolor=SPINE_COLOR, linewidth=0.8, capsize=3.5,
                      error_kw={"elinewidth": 0.9, "capthick": 0.9,
                                 "ecolor": SPINE_COLOR})
        else:
            ax_a.barh(i, m, xerr=s, facecolor=c, alpha=0.22,
                      edgecolor=c, linewidth=1.0, hatch="////",
                      capsize=3.5,
                      error_kw={"elinewidth": 0.8, "capthick": 0.8,
                                 "ecolor": NEUTRAL_GRAY})
    ax_a.axvline(0, color=SPINE_COLOR, lw=0.8, zorder=2)

    # P-value column (regular weight, no bold)
    x_left = min(means_a) - max(sems_a) - 0.025
    x_right_data = max(means_a) + max(sems_a)
    x_lim_right = x_right_data + 0.085
    ax_a.set_xlim(x_left, x_lim_right)
    p_x = x_right_data + 0.012
    for i, p in enumerate(ps_a):
        color = "#111827" if p < bonf_alpha else "#6B7280"
        ax_a.text(p_x, i, _fmt_p(p), ha="left", va="center",
                  fontsize=8.2, color=color)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(BENCH_ORDER, fontsize=9.5)
    ax_a.set_xlabel(r"Mean $\Delta\theta$ (instruct $-$ base)", fontsize=9.5)
    # Panel letter only — no finding-first suptitle (Part J6 rule)
    ax_a.text(-0.05, 1.02, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", color="#111827",
              ha="left", va="bottom")
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", ls=":", lw=0.5, alpha=0.30, color=NEUTRAL_GRAY)
    ax_a.set_axisbelow(True)

    # ── Panel (b): top-N improved + top-N degraded skills
    ax_b = axes[1]
    n_show = 10
    sorted_idx = np.argsort(-mean_delta)
    top_up = sorted_idx[:n_show]
    top_down = sorted_idx[-n_show:][::-1]
    show_idx = np.concatenate([top_down, top_up])
    show_delta = mean_delta[show_idx]
    show_sem = sem[show_idx]
    show_sig = sig_bonf[show_idx]
    show_names = [_clean(names[k]) for k in show_idx]

    bar_colors = []
    for k, d_val in zip(show_idx, show_delta):
        if d_val < 0:
            bar_colors.append(TAX_COLOR)
        else:
            pb = primary_bench[k]
            bar_colors.append(BENCH_COLORS[BENCH_ORDER[pb]] if pb >= 0 else NEUTRAL_GRAY)

    y_b = np.arange(len(show_idx))
    for i, (val, s, c, sig) in enumerate(zip(show_delta, show_sem, bar_colors, show_sig)):
        if sig:
            ax_b.barh(i, val, xerr=s, color=c, alpha=0.95,
                      edgecolor=SPINE_COLOR, linewidth=0.8, capsize=2.8,
                      error_kw={"elinewidth": 0.8, "capthick": 0.8,
                                 "ecolor": SPINE_COLOR})
        else:
            ax_b.barh(i, val, xerr=s, facecolor=c, alpha=0.22,
                      edgecolor=c, linewidth=1.1, hatch="////",
                      capsize=2.8,
                      error_kw={"elinewidth": 0.7, "capthick": 0.7,
                                 "ecolor": NEUTRAL_GRAY})

    ax_b.axhline(n_show - 0.5, color="#CBD5E1", lw=0.6, ls="-", zorder=1)
    ax_b.axvline(0, color=SPINE_COLOR, lw=0.8, zorder=2)
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(show_names, fontsize=8.3)
    ax_b.set_xlabel(r"Mean $\Delta\theta$ (instruct $-$ base)", fontsize=9.5)
    # Panel letter only — caption carries the finding
    ax_b.text(-0.005, 1.02, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", color="#111827",
              ha="left", va="bottom")
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", ls=":", lw=0.5, alpha=0.30, color=NEUTRAL_GRAY)
    ax_b.set_axisbelow(True)

    xb_left = float((show_delta - show_sem).min()) - 0.03
    xb_right = float((show_delta + show_sem).max()) + 0.05
    ax_b.set_xlim(xb_left, xb_right)

    # Legends below panel (b): bench swatches + tax + sig encoding
    bench_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BENCH_COLORS[b],
                      edgecolor=SPINE_COLOR, linewidth=0.6, label=b)
        for b in BENCH_ORDER
    ]
    tax_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor=TAX_COLOR, edgecolor=SPINE_COLOR,
        linewidth=0.6, label="Negative (tax)")
    sig_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL_GRAY,
                      edgecolor=SPINE_COLOR, linewidth=0.8,
                      label="Bonferroni-significant"),
        plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL_GRAY, alpha=0.22,
                      edgecolor=NEUTRAL_GRAY, linewidth=1.1, hatch="////",
                      label="Not significant"),
    ]
    leg1 = ax_b.legend(
        handles=bench_handles + [tax_handle],
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        fontsize=8.3, frameon=False, ncol=6,
        handletextpad=0.4, columnspacing=1.0,
        title="Bar colour: primary benchmark (positive) / tax (negative)",
        title_fontsize=8.3,
    )
    ax_b.add_artist(leg1)
    ax_b.legend(handles=sig_handles,
                loc="upper center", bbox_to_anchor=(0.5, -0.18),
                fontsize=8.3, frameon=False, ncol=2,
                handletextpad=0.4, columnspacing=1.0)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_alignment_tax_skills_REVISED.pdf"
    out_png = OUT_DIR / "fig_alignment_tax_skills_REVISED.png"
    plt.tight_layout(pad=0.4)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close()

    n_sig_pos = int(((mean_delta > 0) & sig_bonf).sum())
    n_sig_neg = int(((mean_delta < 0) & sig_bonf).sum())
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"  n_pairs={n_pairs}, n_bonf_significant={n_bonf}/{K}")
    print(f"  positive sig: {n_sig_pos}, negative sig: {n_sig_neg}")


if __name__ == "__main__":
    main()
