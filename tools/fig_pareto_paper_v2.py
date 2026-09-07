"""Render the v2 paper version of fig:pareto.

Identical to tools/fig_pareto_paper.py (gold-tier exemplar) except:
  - Palette migrated to LOCKED Tailwind-700 spec (NEURIPS_FIGURE_CHECKLIST Part J).
    NO orange / vermillion anywhere:
      protagonist (SkillEval / CDMEval) #1D4ED8 (blue-700)
      comparator  (IrtNet d=232)        #BE185D (pink-700)
      tertiary    (KNN K=10)             #059669 (emerald-700)
      reference   (strongest LLM line)   #475569 (slate-600)
      highlight   (headline star)         #FACC15 (yellow-500)
  - Bold-text discipline (Part J6.5): no bold inline labels, no bold title,
    no bold annotations. Only sans-serif regular weight.
  - No suptitle / no figure-level header — finding lives in the LaTeX caption.

Apples-to-apples cheapest-above-threshold Pareto curves for three methods:
  - SkillEval / CDMEval (K=100, paper checkpoint)
  - IrtNet d=232 (seed 42, comparator paper's published default)
  - KNN K=10 (cheap floor)

Structural identity preserved: single-panel, inline endpoint labels,
headline star at tau=0.80, "matches strongest LLM at 4x lower cost"
annotation, KNN ceiling visible via curve termination.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration" / "experiments"
OUT_DIR = REPO / "cdm_exploration" / "figures" / "review_batch2"

# Part J palette (LOCKED Tailwind-700). NO ORANGE.
COLORS = {
    "cdmeval":     "#1D4ED8",  # blue-700, protagonist
    "irtnet_d232": "#BE185D",  # pink-700, comparator (replaces vermillion)
    "knn_k10":     "#059669",  # emerald-700, tertiary (replaces bluish-green)
}
REFERENCE_COLOR = "#475569"   # slate-600, strongest-single-LLM dotted line
STAR_COLOR = "#FACC15"        # yellow-500, headline star (replaces orange star)

LABELS = {
    "cdmeval":     "CDMEval (K=100)",
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
            f"{src} missing -- run tools/run_pareto_multibaseline.py first")
    d = json.load(open(src))

    # Typography hierarchy (Part J2). Single sans-serif, regular weight default.
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
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    # Plot CDMEval LAST so it sits on top
    plot_order = ["knn_k10", "irtnet_d232", "cdmeval"]
    last_points = {}
    cdm_match_strongest = None  # the tau=0.80 point: ~25% cost, ~100% acc
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
            for p in sweep:
                if abs(p["threshold"] - 0.80) < 1e-6:
                    cdm_match_strongest = (p["cost_pct"], p["acc_pct"])
                    break
        else:
            ax.plot(x, y, MARKERS[key] + "-", color=COLORS[key], lw=1.5,
                    markersize=5.5, alpha=0.85, zorder=4)
        last_points[key] = (x[-1], y[-1])

    # Faint dotted reference at 100% acc = strongest single LLM (Part L7).
    ax.axhline(100, ls=":", color=REFERENCE_COLOR, lw=0.9, zorder=0, alpha=0.7)
    ax.text(0.5, 100, "strongest single LLM",
            color=REFERENCE_COLOR, fontsize=9, va="center", ha="left",
            style="italic", bbox=dict(facecolor="white", edgecolor="none",
                                      pad=2.0, alpha=0.95), zorder=2)

    # Headline star at the tau=0.80 operating point (Part J7: exactly one star).
    if cdm_match_strongest is not None:
        cx, cy = cdm_match_strongest
        ax.scatter([cx], [cy], marker="*", s=300, color=STAR_COLOR,
                   edgecolor=COLORS["cdmeval"], linewidth=1.2, zorder=8)
        # Inline punchline annotation (Part E4). Regular weight per J6.5.
        ax.annotate("matches strongest LLM\nat 4x lower cost\n($\\tau$=0.80)",
                    xy=(cx, cy), xytext=(cx - 11, cy + 5),
                    fontsize=9, color="#1A3A5F",
                    ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="#1A3A5F",
                              pad=3, alpha=0.95, linewidth=0.5),
                    arrowprops=dict(arrowstyle="-", color="#1A3A5F",
                                    lw=0.6,
                                    connectionstyle="arc3,rad=0.15"))

    # Direct line labels at the right end of each curve (Part J4).
    # Regular weight per J6.5 (no fontweight="bold").
    label_offsets = {
        "cdmeval":     (1.5, -2.0),
        "irtnet_d232": (1.0, 0.0),
        "knn_k10":     (-1.0, -5.0),
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

    # Tight axis cropping (Part J5).
    ax.set_xlim(0, 60)
    ax.set_ylim(74, 112)
    ax.set_xlabel("Cost (% of strongest LLM, GFLOPs per query)", fontsize=10)
    ax.set_ylabel("Accuracy (% of strongest LLM)", fontsize=10)
    # No suptitle / no figure-level header (Part J6: finding lives in caption).
    ax.grid(True, alpha=0.18, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_pareto_paper_REVISED.pdf"
    out_png = OUT_DIR / "fig_pareto_paper_REVISED.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
