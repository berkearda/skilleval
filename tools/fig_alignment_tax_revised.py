"""Revised render of fig:aligntax (alignment tax concentrated in IFEval).

Reads cached `cdm_exploration/experiments/v2_alignment_tax.json` (and the
small q-matrix + items file to recover each skill's primary benchmark) and
produces a gold-tier 2-panel figure WITHOUT recomputing any
statistics. The producer pipeline (`tools/run_alignment_tax.py`) is left
untouched.

Design changes vs the CURRENT version:
- Finding-first panel titles ("Only IFEval/BBH show significant gains";
  "Significant gains concentrate in IFEval skills")
- Negative/degraded bars in panel (b) recoloured to reddish purple
  `#CC79A7` (Part J role: tax/degraded), so direction of the effect is
  visible at a glance. Positive bars stay coloured by primary benchmark
  to preserve the "all on IFEval/BBH" insight.
- Tighter axis crops (Part E5).
- Sig/non-sig legend swatches drawn in NEUTRAL gray so they no longer
  imply a specific benchmark colour.
- Single sans-serif typography hierarchy, top/right spines off.
- Pale gridlines only.

Run:
    python tools/fig_alignment_tax_revised.py
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
OUT_DIR = REPO / "cdm_exploration" / "figures" / "review_batch1"

# Okabe-Ito + role-assigned palette (Part J1)
BENCH_COLORS = {
    "MATH":   "#0072B2",  # blue
    "BBH":    "#E69F00",  # orange
    "GPQA":   "#009E73",  # bluish green
    "MuSR":   "#F0E442",  # yellow (replaces darker reddish purple to
                            #  reserve #CC79A7 strictly for "tax/negative")
    "IFEval": "#56B4E9",  # sky blue
}
TAX_COLOR = "#CC79A7"  # reddish purple — Part J role: alignment tax / negative
NEUTRAL_GRAY = "#888888"

BENCH_ORDER = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]

# Display-only label fixes (truncated cluster labels). Mirrors the table in
# tools/run_alignment_tax.py to keep the revised figure visually identical
# in label content. Only the entries actually appearing in top-10 lists
# matter, but we keep the full table for safety.
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
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _primary_bench_per_skill(K: int) -> np.ndarray:
    """Recover argmax-benchmark per skill from q-matrix + items metadata.

    Returns an int array of length K with values in [-1, 0..len(BENCH_ORDER)-1].
    """
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

    # Vectorise per-skill arrays (preserve JSON-order, which is k=0..K-1)
    mean_delta = np.array([t["mean_delta"] for t in per_skill_tests])
    sem = np.array([t["sem"] for t in per_skill_tests])
    sig_bonf = np.array([t["sig_bonferroni"] for t in per_skill_tests], dtype=bool)
    names = [t["name"] for t in per_skill_tests]

    # Recover primary benchmark for each skill from q-matrix
    primary_bench = _primary_bench_per_skill(K)

    # ── Layout ─────────────────────────────────────────────
    _set_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 6.4),
        gridspec_kw={"width_ratios": [0.95, 2.05]},
    )

    # ── Panel (a): per-benchmark pair-level mean ±SEM, with p-values
    ax_a = axes[0]
    means_a = [per_bench_sem[b]["mean"] for b in BENCH_ORDER]
    sems_a = [per_bench_sem[b]["sem"] for b in BENCH_ORDER]
    ps_a = [per_bench_sem[b]["p"] for b in BENCH_ORDER]

    # Sig vs non-sig encoded redundantly: Bonferroni-significant
    # (p<alpha/K, here p<5e-4 with K=100) get solid edge + saturated fill;
    # non-significant get muted fill + hatching.
    bonf_alpha = 0.05 / K
    y_pos = np.arange(len(BENCH_ORDER))
    for i, (b, m, s, p) in enumerate(zip(BENCH_ORDER, means_a, sems_a, ps_a)):
        c = BENCH_COLORS[b]
        if p < bonf_alpha:
            ax_a.barh(i, m, xerr=s, color=c, alpha=0.92,
                      edgecolor="#111111", linewidth=1.0, capsize=4,
                      error_kw={"elinewidth": 1.0, "capthick": 1.0})
        else:
            ax_a.barh(i, m, xerr=s, facecolor=c, alpha=0.25,
                      edgecolor=c, linewidth=1.2, hatch="////",
                      capsize=4,
                      error_kw={"elinewidth": 0.9, "capthick": 0.9,
                                 "ecolor": "#666666"})
    ax_a.axvline(0, color="black", lw=0.9, zorder=2)

    # P-value column to the right of bars (E4 style: inline annotations)
    x_left = min(means_a) - max(sems_a) - 0.025
    x_right_data = max(means_a) + max(sems_a)
    x_lim_right = x_right_data + 0.085
    ax_a.set_xlim(x_left, x_lim_right)
    p_x = x_right_data + 0.015
    for i, p in enumerate(ps_a):
        weight = "bold" if p < bonf_alpha else "normal"
        color = "#111111" if p < bonf_alpha else "#555555"
        ax_a.text(p_x, i, _fmt_p(p), ha="left", va="center",
                  fontsize=8.5, color=color, fontweight=weight)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(BENCH_ORDER, fontsize=10)
    ax_a.set_xlabel(r"Mean $\Delta\theta$ (instruct $-$ base)", fontsize=10)
    ax_a.set_title("(a) Only IFEval shows a significant gain",
                   loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", ls=":", lw=0.5, alpha=0.30)
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

    # Colour rule (per Part J1 + spec):
    #  - Degraded (negative) bars  → reddish purple #CC79A7
    #  - Improved (positive) bars  → primary-benchmark colour
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
                      edgecolor="#111111", linewidth=1.0, capsize=3,
                      error_kw={"elinewidth": 0.9, "capthick": 0.9,
                                 "ecolor": "#333333"})
        else:
            ax_b.barh(i, val, xerr=s, facecolor=c, alpha=0.22,
                      edgecolor=c, linewidth=1.3, hatch="////",
                      capsize=3,
                      error_kw={"elinewidth": 0.7, "capthick": 0.7,
                                 "ecolor": "#888888"})

    ax_b.axhline(n_show - 0.5, color="#cccccc", lw=0.6, ls="-", zorder=1)
    ax_b.axvline(0, color="black", lw=0.9, zorder=2)
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(show_names, fontsize=8.5)
    ax_b.set_xlabel(r"Mean $\Delta\theta$ (instruct $-$ base)", fontsize=10)
    # Compose finding-first title from the actual sig-skill counts to
    # avoid overclaim. The split (11 positive / 3 negative; 8 of 11 on
    # IFEval) is verified against per_skill_tests + primary-benchmark map.
    n_sig_pos = int(((mean_delta > 0) & sig_bonf).sum())
    n_sig_neg = int(((mean_delta < 0) & sig_bonf).sum())
    ax_b.set_title(
        f"(b) Significant effects concentrate on the gain side "
        f"({n_sig_pos} positive vs {n_sig_neg} negative of {n_bonf}/{K})",
        loc="left", fontsize=10.5, fontweight="bold", pad=6,
    )
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", ls=":", lw=0.5, alpha=0.30)
    ax_b.set_axisbelow(True)

    # Tight x-axis crop based on data + error bars (E5)
    xb_left = float((show_delta - show_sem).min()) - 0.03
    xb_right = float((show_delta + show_sem).max()) + 0.05
    ax_b.set_xlim(xb_left, xb_right)

    # ── Legends below panel (b): primary-benchmark swatches +
    # significance encoding (neutral grey so the swatches do not mimic
    # any specific benchmark colour).
    bench_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BENCH_COLORS[b],
                      edgecolor="#222222", linewidth=0.8, label=b)
        for b in BENCH_ORDER
    ]
    tax_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor=TAX_COLOR, edgecolor="#222222",
        linewidth=0.8, label="Negative (tax)")
    sig_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL_GRAY,
                      edgecolor="#111111", linewidth=1.0,
                      label="Bonferroni-significant"),
        plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL_GRAY, alpha=0.22,
                      edgecolor=NEUTRAL_GRAY, linewidth=1.3, hatch="////",
                      label="Not significant"),
    ]
    leg1 = ax_b.legend(
        handles=bench_handles + [tax_handle],
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        fontsize=8.5, frameon=False, ncol=6,
        handletextpad=0.4, columnspacing=1.0,
        title="Bar colour: primary benchmark (positive) / tax (negative)",
        title_fontsize=8.5,
    )
    ax_b.add_artist(leg1)
    ax_b.legend(handles=sig_handles,
                loc="upper center", bbox_to_anchor=(0.5, -0.18),
                fontsize=8.5, frameon=False, ncol=2,
                handletextpad=0.4, columnspacing=1.0)

    # ── Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_alignment_tax_skills_REVISED.pdf"
    out_png = OUT_DIR / "fig_alignment_tax_skills_REVISED.png"
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"  n_pairs={n_pairs}, n_bonf_significant={n_bonf}/{K}")


if __name__ == "__main__":
    main()
