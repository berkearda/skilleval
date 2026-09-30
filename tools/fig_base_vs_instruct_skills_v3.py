#!/usr/bin/env python3
"""Main-text figure, compact version 3 (24 Sep 2026): estimated mastery, instruction-tuned minus base.

Same data, colours, significance rules and labels as tools/fig_base_vs_instruct_skills_v2.py (kept); drawn directly at
the paper's text width (5.5 in) with Times fonts so it takes less height. Version 2 notes follow.


Drawn from STORED results only (v2_alignment_tax.json); no analysis is re-run. The primary benchmark of a
skill (for colours) is recomputed from the Q-matrix and the item list, as in run_alignment_tax.py.

Changes against version 1 (figure review of 2026-09-20):
  - panel (b) shows ALL skills that pass Bonferroni correction across 100 skills (14), sorted by mean
    difference, instead of the ten largest positive and ten largest negative differences;
  - no footer under the figure: the two significance rules are explained in the caption
    (panel a: stars for unadjusted benchmark-level p-values; panel b: Bonferroni across skills);
  - "Higher / Lower estimated mastery" instead of IMPROVED / DEGRADED; x label names the quantity;
  - display-only short skill labels so they are readable at column width; panel (b) gets more room.

Output: cdm_exploration/figures/report/fig_base_vs_instruct_skills_v2.{pdf,png}

    python tools/fig_base_vs_instruct_skills_v2.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration/data/cdm_ready"
SRC = REPO / "cdm_exploration/experiments/v2_alignment_tax.json"
OUT = REPO / "cdm_exploration/figures/report/fig_base_vs_instruct_skills_v3"

BENCH = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
COLOR = {"MATH": "#0072B2", "BBH": "#E69F00", "GPQA": "#009E73", "MuSR": "#CC79A7", "IFEval": "#56B4E9"}  # Okabe-Ito
XLABEL = "Mean mastery difference (instruction-tuned − base)"

# Display-only short labels (the exact skill names are in the released skill list).
SHORT = {
    71: "Random variable concepts for sleep patterns", 10: "Set theory to count unique items",
    81: "Possible storage options based on story", 5: "Sporting event plausibility based on context",
    59: "Combinatorial principles for pet ownership", 65: "Historical significance of chess programs",
    22: "Work done in variable force scenarios", 61: "Asterisk symbols for visual separation",
    41: "Organizing information based on given clues", 8: "Consistent tone in creative writing",
    23: "Word count constraints in responses", 28: "Feature vs. bug distinction",
    21: "Language for age-appropriate audience", 33: "Tone for dialogue in Urdu",
}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def primary_benchmark():
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    counts = np.zeros((Q.shape[1], len(BENCH)), int)
    for i, it in enumerate(items):
        if it.get("benchmark") in BENCH:
            counts[:, BENCH.index(it["benchmark"])] += Q[i].astype(int)
    return [BENCH[j] for j in counts.argmax(axis=1)]


def main():
    d = json.load(open(SRC))
    st = d["statistical_tests"]
    sig = sorted([s for s in st["per_skill"] if s["sig_bonferroni"]], key=lambda s: s["mean_delta"])
    assert len(sig) == st["n_significant_bonferroni"] == 14 and all(s["skill"] in SHORT for s in sig)
    prim = primary_benchmark()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
    fig = plt.figure(figsize=(5.5, 1.95))
    ax_a = fig.add_axes([0.075, 0.27, 0.19, 0.63])     # explicit boxes: the long skill labels need a fixed gutter
    ax_b = fig.add_axes([0.605, 0.27, 0.39, 0.63])

    # (a) per benchmark
    y = np.arange(len(BENCH))[::-1]
    m = [st["per_benchmark_sem"][b]["mean"] for b in BENCH]
    e = [st["per_benchmark_sem"][b]["sem"] for b in BENCH]
    ax_a.barh(y, m, xerr=e, color=[COLOR[b] for b in BENCH], edgecolor="#222222", linewidth=0.5, height=0.62,
              capsize=1.5, error_kw={"elinewidth": 0.6})
    for yi, b, mi, ei in zip(y, BENCH, m, e):
        s_ = stars(st["per_benchmark_sem"][b]["p"])
        ax_a.text(max(mi + ei, 0) + 0.008, yi, s_, va="center", ha="left", fontsize=6.5 if s_ != "n.s." else 6,
                  fontweight="bold" if s_ != "n.s." else "normal", color="#222222")
    ax_a.axvline(0, color="#222222", lw=0.6)
    ax_a.set_yticks(y); ax_a.set_yticklabels(BENCH, fontsize=7)
    ax_a.set_xlim(-0.05, 0.20); ax_a.set_xticks([0, 0.1, 0.2])
    ax_a.set_title("(a) By benchmark", fontsize=7.5, loc="left", pad=3)

    # (b) all Bonferroni-significant skills
    yb = np.arange(len(sig))
    mb = [s_["mean_delta"] for s_ in sig]; eb = [s_["sem"] for s_ in sig]
    ax_b.barh(yb, mb, xerr=eb, color=[COLOR[prim[s_["skill"]]] for s_ in sig], edgecolor="#222222", linewidth=0.5,
              height=0.7, capsize=1.3, error_kw={"elinewidth": 0.6})
    ax_b.axvline(0, color="#222222", lw=0.6)
    n_neg = sum(v < 0 for v in mb)
    ax_b.axhline(n_neg - 0.5, color="#999999", lw=0.6, ls=":")
    ax_b.set_yticks(yb); ax_b.set_yticklabels([SHORT[s_["skill"]] for s_ in sig], fontsize=6.5)
    ax_b.set_xlim(-0.12, 0.54)
    ax_b.set_title("(b) Skills significant after Bonferroni correction (14 of 100)", fontsize=7.5, loc="left", pad=3)
    ax_b.text(0.53, n_neg - 1.25, "Lower estimated mastery", ha="right", va="center", fontsize=6.3, color="#444444", style="italic")
    ax_b.text(0.53, n_neg + 0.2, "Higher estimated mastery", ha="right", va="center", fontsize=6.3, color="#444444", style="italic")
    for ax in (ax_a, ax_b):
        ax.set_axisbelow(True); ax.grid(axis="x", ls=":", lw=0.4, alpha=0.45)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.5, 0.135, XLABEL, ha="center", va="center", fontsize=7)
    used = [b for b in BENCH if any(prim[s_["skill"]] == b for s_ in sig)]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none", label="Primary benchmark:")] + \
              [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[b], edgecolor="#222222", linewidth=0.5, label=b) for b in BENCH]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=6, frameon=False, fontsize=6.8,
               handlelength=1.2, handleheight=0.8, columnspacing=1.2, handletextpad=0.4)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight")
    print("wrote", f"{OUT}.pdf and .png")
    print("benchmarks of the 11 increases:", [prim[s["skill"]] for s in sig if s["mean_delta"] > 0])
    print("benchmarks of the 3 decreases :", [prim[s["skill"]] for s in sig if s["mean_delta"] < 0], "| in legend but unused:", [b for b in BENCH if b not in used])


if __name__ == "__main__":
    main()
