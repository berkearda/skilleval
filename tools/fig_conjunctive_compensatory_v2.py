#!/usr/bin/env python3
"""Appendix figure: exploratory classification of accuracy patterns in multi-skill items (version 2).

Redraws fig_conjunctive_compensatory from the STORED results only; no analysis is re-run.
Changes against version 1 (review of 2026-09-20):
  - legend says Conjunctive-like / Compensatory-like / Intermediate (no "need ALL", "ANY helps");
  - both panels say "Eligible items (%)" and show the number of eligible items under each bar;
  - panel (b) is titled "Classification across methods and eligible subsets";
  - both panels share one y range, so their zero baselines align;
  - Okabe-Ito colours (project rule).

Source: cdm_exploration/experiments/v2_conjunctive_compensatory.json
Output: cdm_exploration/figures/report/fig_conjunctive_compensatory_v2.{pdf,png}

    python tools/fig_conjunctive_compensatory_v2.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "cdm_exploration/experiments/v2_conjunctive_compensatory.json"
OUT = REPO / "cdm_exploration/figures/report/fig_conjunctive_compensatory_v2"

TYPES = ["conjunctive", "compensatory", "additive"]
LABEL = {"conjunctive": "Conjunctive-like", "compensatory": "Compensatory-like", "additive": "Intermediate"}
COLOR = {"conjunctive": "#0072B2", "compensatory": "#E69F00", "additive": "#009E73"}   # Okabe-Ito


def stacked(ax, xlabels, pct_rows, title):
    x = np.arange(len(xlabels))
    bottom = np.zeros(len(xlabels))
    for t in TYPES:
        vals = np.array([row[t] for row in pct_rows])
        ax.bar(x, vals, 0.58, bottom=bottom, color=COLOR[t], edgecolor="white", linewidth=0.6)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 5:
                ax.text(x[i], b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8.5 if v >= 10 else 7, fontweight="bold", color="white")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9.5, linespacing=1.35)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Eligible items (%)", fontsize=10.5)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    d = json.load(open(SRC))
    ter = d["tertile"]
    assert sum(v["total"] for v in ter["per_benchmark"].values()) == ter["n_analyzed"] == 2725

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))   # smaller canvas = larger text at column width

    benches = sorted(ter["per_benchmark"])
    rows_a = [{t: ter["per_benchmark"][b][f"{t}_pct"] for t in TYPES} for b in benches]
    labels_a = [f"{b}\nn={ter['per_benchmark'][b]['total']:,}" for b in benches]
    stacked(axes[0], labels_a, rows_a, "(a) Per-benchmark classification (tertile method)")

    methods = [("Tertile\n(primary)", d["tertile"]), ("Binary\nmastery", d["binary_sensitivity"]),
               ("Pattern\n(2-skill items)", d["pattern_robustness_2skill"])]
    rows_b = [{t: m["overall_pct"][t] for t in TYPES} for _, m in methods]
    labels_b = [f"{name}\nn={m['n_analyzed']:,}" for name, m in methods]
    stacked(axes[1], labels_b, rows_b, "(b) Classification across methods and eligible subsets")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[t], edgecolor="white", label=LABEL[t]) for t in TYPES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight")
    print("wrote", f"{OUT}.pdf", "and .png")
    print("panel (a):", {b: {t: round(r[t], 1) for t in TYPES} for b, r in zip(benches, rows_a)})
    print("panel (b):", [{t: round(r[t], 1) for t in TYPES} for r in rows_b])


if __name__ == "__main__":
    main()
