"""Render the paper version of fig:pareto (gold-tier per NEURIPS_FIGURE_CHECKLIST.md).

Apples-to-apples cheapest-above-threshold Pareto curves for three methods:
  - CDMEval (K=100, paper checkpoint)
  - IrtNet d=232 (seed 42, comparator paper's published default)
  - KNN K=10 (cheap floor)

Gold-tier patterns applied (from NeurIPS 2024-2025 award winners,
RouterBench, FrugalGPT — see NEURIPS_FIGURE_CHECKLIST.md Part E + H):
- E1 single sans-serif family, 8/10/11 type hierarchy
- E2 inline curve labels, no legend box
- E3 NOT applicable here (single-method curves, no envelope of dots)
- E4 inline punchline value at CDMEval top point ("+6 pt over strongest")
- E5 tight axis cropping
- E7 one color per routing method (CDMEval blue, IrtNet vermillion, KNN green)
- E8 finding-first caption (in the document, not the figure)
- H6 star marker at the recommended operating point with inline annotation
- H10 pale dotted reference line at y=100 with corner italic label

EmbedLLM is excluded from the submission paper per 2026-05-01 editorial
decision (the project log).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration" / "experiments"
FIG_DIR = REPO / "cdm_exploration" / "figures" / "report" / "main_ready"

# Okabe-Ito 8-color, colorblind-safe (colour rule)
COLORS = {
    "cdmeval":     "#0072B2",  # blue (headline)
    "irtnet_d232": "#D55E00",  # vermillion (comparator paper's published default)
    "knn_k10":     "#009E73",  # bluish green (floor)
}
LABELS = {
    "cdmeval":     "SkillEval (K=100)",
    "irtnet_d232": "IrtNet (d=232)",
    "knn_k10":     "KNN (K=10)",
}
MARKERS = {
    "cdmeval":     "o",
    "irtnet_d232": "^",
    "knn_k10":     "D",
}


def main() -> None:
    src = EXP / "v2_pareto_multibaseline.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing — run tools/run_pareto_multibaseline.py first")
    d = json.load(open(src))

    # Gold-tier typography hierarchy (E1): tick 8 pt, axis label 10 pt,
    # title 11 pt. Single sans-serif family throughout.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Plot CDMEval LAST so it sits on top
    plot_order = ["knn_k10", "irtnet_d232", "cdmeval"]
    last_points = {}
    cdm_match_strongest = None  # the τ=0.80 point: 25% cost, 99.9% acc
    for key in plot_order:
        if key not in d["methods"]:
            continue
        sweep = d["methods"][key]["normalized"]
        x = [p["cost_pct"] for p in sweep]
        y = [p["acc_pct"] for p in sweep]
        order = sorted(range(len(x)), key=lambda i: (x[i], -y[i]))
        x = [x[i] for i in order]
        y = [y[i] for i in order]
        if key == "cdmeval":
            ax.plot(x, y, MARKERS[key] + "-", color=COLORS[key], lw=2.0,
                     markersize=6.5, alpha=1.0, zorder=6)
            # Find the τ=0.80 operating point ("matches strongest at 25% cost")
            for p in sweep:
                if abs(p["threshold"] - 0.80) < 1e-6:
                    cdm_match_strongest = (p["cost_pct"], p["acc_pct"])
                    break
        else:
            ax.plot(x, y, MARKERS[key] + "-", color=COLORS[key], lw=1.5,
                     markersize=5.5, alpha=0.85, zorder=4)
        last_points[key] = (x[-1], y[-1])

    # Faint horizontal reference line at 100% acc (= strongest single LLM).
    # Label sits ON the line at the LEFT edge — empty region (KNN stops at
    # x=19/y=98, CDMEval crosses 100 at x=25, IrtNet at x=42).
    ax.axhline(100, ls=":", color="#bbbbbb", lw=0.8, zorder=0)
    ax.text(0.5, 100, "strongest single LLM",
             color="#888888", fontsize=9, va="center", ha="left",
             style="italic", bbox=dict(facecolor="white", edgecolor="none",
                                         pad=2.0, alpha=0.95), zorder=2)

    # H6 — gold-tier: star marker at the τ=0.80 point ("matches strongest at
    # 4× lower cost") with inline punchline value (E4)
    if cdm_match_strongest is not None:
        cx, cy = cdm_match_strongest
        ax.scatter([cx], [cy], marker="*", s=280, color="#FFCB7A",
                    edgecolor=COLORS["cdmeval"], linewidth=1.2, zorder=8)
        # Inline punchline value annotation with thin leader line
        ax.annotate("matches strongest LLM\nat 4× lower cost\n(τ=0.80)",
                     xy=(cx, cy), xytext=(cx - 11, cy + 5),
                     fontsize=9, color="#1A3A5F",
                     ha="left", va="bottom",
                     bbox=dict(facecolor="white", edgecolor="#1A3A5F",
                                 pad=3, alpha=0.95, linewidth=0.5),
                     arrowprops=dict(arrowstyle="-", color="#1A3A5F",
                                       lw=0.6,
                                       connectionstyle="arc3,rad=0.15"))

    # Direct line labels at the right end of each curve (E2)
    label_offsets = {
        "cdmeval":     (1.5, -2.0),    # below-right of (37, 106), avoid star annotation
        "irtnet_d232": (1.0, 0.0),     # right of (54, 101)
        "knn_k10":     (-1.0, -5.0),   # below-left of (19, 98)
    }
    label_ha = {
        "cdmeval":     "left",
        "irtnet_d232": "left",
        "knn_k10":     "right",
    }
    label_va = {
        "cdmeval":     "top",
        "irtnet_d232": "center",
        "knn_k10":     "top",
    }
    for key, (x_end, y_end) in last_points.items():
        dx, dy = label_offsets[key]
        ax.annotate(LABELS[key], xy=(x_end, y_end),
                     xytext=(x_end + dx, y_end + dy),
                     fontsize=9.5, color=COLORS[key],
                     ha=label_ha[key], va=label_va[key],
                     bbox=dict(facecolor="white", edgecolor="none",
                                 pad=1.2, alpha=0.9))

    # Tight axis cropping (E5)
    ax.set_xlim(0, 60)
    ax.set_ylim(74, 112)
    # Specific axis labels with units (E7)
    ax.set_xlabel("Cost (% of strongest LLM, GFLOPs per query)", fontsize=10)
    ax.set_ylabel("Accuracy (% of strongest LLM)", fontsize=10)
    # No suptitle: finding lives in the LaTeX caption per Part J6
    # Pale gridlines (anti-pattern F5 avoided)
    ax.grid(True, alpha=0.18, lw=0.5, zorder=0)

    plt.tight_layout(pad=0.4)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "fig_pareto_paper.pdf"
    out_png = FIG_DIR / "fig_pareto_paper.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", dpi=200, pad_inches=0.05)
    plt.close()
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
