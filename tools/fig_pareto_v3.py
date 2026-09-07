"""Render fig_pareto_v3 from v2_pareto_multibaseline.json.

Four Pareto curves on the same axes (cost%, acc% relative to strongest):
  - CDMEval (paper checkpoint)
  - IrtNet d=100 (seed 42)
  - IrtNet d=232 (seed 42)
  - KNN K=10

No single-LLM anchors. Okabe-Ito palette. Re-runs from cached JSON in <5s.
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

# Okabe-Ito 8-color palette, colorblind-safe (colour rule)
COLORS = {
    "cdmeval":      "#0072B2",  # blue
    "irtnet_d100":  "#E69F00",  # orange
    "irtnet_d232":  "#D55E00",  # vermillion
    "embedllm_d232": "#CC79A7",  # reddish purple
    "knn_k10":      "#009E73",  # bluish green
}
LABELS = {
    "cdmeval":      "CDMEval (K=100)",
    "irtnet_d100":  "IrtNet d=100",
    "irtnet_d232":  "IrtNet d=232",
    "embedllm_d232": "EmbedLLM d=232",
    "knn_k10":      "KNN K=10",
}
MARKERS = {
    "cdmeval":      "o",
    "irtnet_d100":  "s",
    "irtnet_d232":  "^",
    "embedllm_d232": "v",
    "knn_k10":      "D",
}


def main() -> None:
    src = EXP / "v2_pareto_multibaseline.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing — run tools/run_pareto_multibaseline.py first")
    d = json.load(open(src))

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 140,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for key in ["knn_k10", "irtnet_d100", "irtnet_d232", "embedllm_d232", "cdmeval"]:
        sweep = d["methods"][key]["normalized"]
        x = [p["cost_pct"] for p in sweep]
        y = [p["acc_pct"] for p in sweep]
        # Sort by cost so the curve doesn't cross back on itself; ties on
        # cost broken by accuracy descending so the upper-Pareto edge is
        # visible if the threshold sweep is not strictly monotone in cost.
        order = sorted(range(len(x)), key=lambda i: (x[i], -y[i]))
        x = [x[i] for i in order]
        y = [y[i] for i in order]
        ax.plot(x, y, MARKERS[key] + "-", color=COLORS[key], lw=1.8,
                 markersize=6, label=LABELS[key], alpha=0.9)

    # Reference horizontal at 100% (strongest acc)
    ax.axhline(100, ls="--", color="#888888", lw=0.7, zorder=0)
    ax.text(0.5, 100.5, "strongest single LLM (100% acc / 100% cost)",
             color="#888888", fontsize=8.5, va="bottom")

    ax.set_xlim(0, 105)
    ax.set_ylim(60, 120)
    ax.set_xlabel("Cost relative to strongest model (%)")
    ax.set_ylabel("Accuracy relative to strongest model (%)")
    ax.set_title("Cost–accuracy Pareto: matched cheapest-above-threshold routing")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "fig_pareto_v3.pdf"
    out_png = FIG_DIR / "fig_pareto_v3.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
