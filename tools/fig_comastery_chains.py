#!/usr/bin/env python3
"""Main-text figure for Section 5: the longest co-mastery chain among MATH skills whose links appear at every mastery
cut-off (0.4, 0.5, 0.6).

Reads v2_comastery_robust.json (tools/diag_comastery_robust_links.py), which fixes the chain by rule. Each box is a skill
with the share of LLMs that master it (predicted accuracy on the skill's items above 0.5). Each arrow A -> B is a
co-mastery link; the number under it is the share of the LLMs mastering B that also master A (the reverse share is in
the JSON and in the text); the caption explains both numbers. Drawn at the paper's text width (5.5 in), so font sizes
are the printed sizes, and every label is checked to sit inside its box. Display names shorten the cluster labels of cluster_labels_v2_K100.json, the label set the paper prints.

    python tools/fig_comastery_chains.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration/experiments"
OUT = REPO / "cdm_exploration/figures/report/main_ready/fig_comastery_chain"
MATH_BLUE = "#0072B2"   # Okabe-Ito, MATH in every benchmark figure
INK, MUTED = "#222222", "#555555"
SHORT = {"Calculating Handshake Combinations In Group Intera": ("Handshake", "combinations"),
         "Manipulating Algebraic Expressions For Variance": ("Algebraic expressions", "for variance"),
         "Applying Divisibility Rules To Factorials": ("Divisibility rules", "for factorials"),
         "Applying Combinatorial Principles To Dice Rolls": ("Combinatorics", "of dice rolls")}


def main():
    res = json.load(open(EXP / "v2_comastery_robust.json"))
    assert res["verified"]
    chain = res["chains"]["MATH"]
    skills, links = chain["skills"], chain["links"]
    assert all(s["primary_benchmark"] == "MATH" for s in skills) and len(links) == len(skills) - 1
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "font.size": 8.5, "pdf.fonttype": 42})
    n, w, h, gap = len(skills), 1.14, 0.62, 0.32
    width = n * w + (n - 1) * gap
    fig, ax = plt.subplots(figsize=(width, h + 0.02))
    ax.set_xlim(-0.01, width + 0.01); ax.set_ylim(-0.01, h + 0.01); ax.axis("off")
    boxes, gap_labels = [], []
    for i, s in enumerate(skills):
        x0 = i * (w + gap)
        patch = ax.add_patch(FancyBboxPatch((x0, 0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                            facecolor="white", edgecolor=MATH_BLUE, linewidth=1.3))
        top, bottom = SHORT[s["name"]]
        texts = [ax.text(x0 + w / 2, 0.47, top, ha="center", va="center", fontsize=8, color=INK),
                 ax.text(x0 + w / 2, 0.32, bottom, ha="center", va="center", fontsize=8, color=INK),
                 ax.text(x0 + w / 2, 0.13, f"{100 * s['prevalence']:.1f}% of LLMs", ha="center", va="center",
                         fontweight="bold", color=INK)]
        boxes.append((patch, texts))
        if i + 1 < n:
            xa, xb = x0 + w + 0.05, x0 + w + gap - 0.05
            ax.add_patch(FancyArrowPatch((xa, h / 2), (xb, h / 2), arrowstyle="-|>", mutation_scale=8,
                                         linewidth=1.0, color=MUTED))
            gap_labels.append((x0 + w, x0 + w + gap,
                               ax.text((xa + xb) / 2, h / 2 - 0.07, f"{100 * links[i]['p_from_given_to']:.1f}%",
                                       ha="center", va="top", fontsize=7, color=MUTED)))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    renderer = fig.canvas.get_renderer()
    margin = 0.06 * fig.dpi   # at least 0.06 in between a label and its box edge
    for patch, texts in boxes:
        b = patch.get_window_extent(renderer)
        for t in texts:
            e = t.get_window_extent(renderer)
            assert e.x0 >= b.x0 + margin and e.x1 <= b.x1 - margin, f"label does not fit: {t.get_text()}"
    for left, right, t in gap_labels:   # arrow labels stay clear of both boxes
        e = t.get_window_extent(renderer)
        lo, hi = ax.transData.transform([(left, 0), (right, 0)])[:, 0]
        assert e.x0 >= lo + 0.03 * fig.dpi and e.x1 <= hi - 0.03 * fig.dpi, f"arrow label does not fit: {t.get_text()}"
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("saved", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
