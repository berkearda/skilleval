#!/usr/bin/env python3
"""Figure 2 (cost against accuracy) with corrected LLM prices. Drawn from v2_pareto_costfix.json only.

Same design as tools/fig_pareto_paper.py (the figure in the submitted paper): single panel, Okabe-Ito colours, inline
labels, one star. Differences:
  - prices corrected for four LLMs (tools/run_pareto_costfix.py);
  - the star marks the cheapest threshold at which the main model reaches the strongest single LLM (tau = 0.90,
    33% of its cost), not tau = 0.80;
  - the five-seed mean (batch size 64) is added as a dashed line with a +-1 sample-SD band, which is the multi-seed
    cost curve planned for the paper;
  - the y axis starts lower, so the whole sweep stays visible (after the correction the cheap end is much lower).

Output: cdm_exploration/figures/report/main_ready/fig_pareto_costfix.{pdf,png}

    python tools/fig_pareto_costfix.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "cdm_exploration/experiments/v2_pareto_costfix.json"
OUT = REPO / "cdm_exploration/figures/report/main_ready/fig_pareto_costfix"
COLORS = {"skilleval_main": "#0072B2", "irtnet_d232": "#D55E00", "knn_k10": "#009E73"}     # Okabe-Ito
LABELS = {"skilleval_main": "SkillEval (K=100)", "irtnet_d232": "IrtNet (d=232)", "knn_k10": "KNN (K=10)"}
MARKERS = {"skilleval_main": "o", "irtnet_d232": "^", "knn_k10": "D"}
REF, STAR = "#555555", "#F0E442"


def main():
    d = json.load(open(SRC))
    s_acc = d["strongest"]["accuracy"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    five = d["five_seed"]["corrected_prices"]["points"]
    fx = [p["cost_pct_mean"] for p in five]; fy = [100 * p["acc_mean"] / s_acc for p in five]
    fs = [100 * p["acc_sd_sample"] / s_acc for p in five]
    ax.fill_between(fx, [a - b for a, b in zip(fy, fs)], [a + b for a, b in zip(fy, fs)], color=COLORS["skilleval_main"], alpha=0.12, lw=0, zorder=2)
    ax.plot(fx, fy, "--", color=COLORS["skilleval_main"], lw=1.3, alpha=0.9, zorder=3)
    ax.annotate("SkillEval, five-seed mean\n(batch size 64)", xy=(fx[-1], fy[-1]), xytext=(fx[-1] + 1.5, fy[-1] - 4.5), fontsize=8.5,
                color=COLORS["skilleval_main"], ha="left", va="top")

    ends = {}
    for key in ("knn_k10", "irtnet_d232", "skilleval_main"):
        pts = sorted(d["methods"][key]["corrected_prices"]["points"], key=lambda p: (p["cost_pct"], -p["accuracy"]))
        x = [p["cost_pct"] for p in pts]; y = [p["acc_pct_of_strongest"] for p in pts]
        main_line = key == "skilleval_main"
        ax.plot(x, y, MARKERS[key] + "-", color=COLORS[key], lw=2.0 if main_line else 1.5, markersize=6.0 if main_line else 5.0,
                alpha=1.0 if main_line else 0.85, zorder=6 if main_line else 4)
        ends[key] = (x[-1], y[-1])

    ax.axhline(100, ls=":", color=REF, lw=0.9, zorder=0)
    ax.text(0.6, 100, "strongest single LLM", color=REF, fontsize=9, va="center", ha="left", style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.95), zorder=2)

    par = d["methods"]["skilleval_main"]["corrected_prices"]["cheapest_point_at_or_above_strongest"]
    cx, cy = par["cost_pct"], par["acc_pct_of_strongest"]
    ax.scatter([cx], [cy], marker="*", s=300, color=STAR, edgecolor=COLORS["skilleval_main"], linewidth=1.2, zorder=8)
    ax.annotate(f"matches strongest LLM\nat {cx:.0f}% of its cost\n($\\tau$={par['threshold']:.2f})", xy=(cx, cy), xytext=(cx - 19, cy + 3.5),
                fontsize=9, color="#1A3A5F", ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="#1A3A5F", pad=3, alpha=0.95, linewidth=0.5),
                arrowprops=dict(arrowstyle="-", color="#1A3A5F", lw=0.6, connectionstyle="arc3,rad=0.15"))

    off = {"skilleval_main": (1.2, 1.5, "left", "bottom"), "irtnet_d232": (1.0, 0.0, "left", "center"), "knn_k10": (-18.4, -5.5, "left", "center")}
    for key, (xe, ye) in ends.items():
        dx, dy, ha, va = off[key]      # the KNN label sits left of its curve, where the panel is empty
        ax.annotate(LABELS[key], xy=(xe, ye), xytext=(xe + dx, ye + dy), fontsize=9.5, color=COLORS[key], ha=ha, va=va,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9))

    ax.set_xlim(0, 68); ax.set_ylim(40, 112)
    ax.set_xlabel("Cost (% of strongest LLM, GFLOPs per query)"); ax.set_ylabel("Accuracy (% of strongest LLM)")
    ax.grid(True, alpha=0.18, lw=0.5, zorder=0); ax.set_axisbelow(True)
    plt.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.05); fig.savefig(f"{OUT}.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    print("wrote", f"{OUT}.pdf and .png | star at", round(cx, 1), round(cy, 1), "tau", par["threshold"])


if __name__ == "__main__":
    main()
