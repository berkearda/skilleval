"""Render fig_pareto_v4 with calibrated CDMEval + calibrated EmbedLLM curves.

Reads from v2_pareto_multibaseline_v4.json which extends v3 with:
  - cdmeval_calibrated  (CDMEval K=100 trained on 60% items + isotonic on val)
  - embedllm_calibrated (EmbedLLM d=232 trained on 60% items + isotonic on val)
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

# Okabe-Ito 8-color palette, colorblind-safe
COLORS = {
    "cdmeval":             "#0072B2",  # blue
    "cdmeval_calibrated":  "#56B4E9",  # sky blue (lighter cdmeval variant)
    "irtnet_d100":         "#E69F00",  # orange
    "irtnet_d232":         "#D55E00",  # vermillion
    "embedllm_d232":       "#CC79A7",  # reddish purple
    "embedllm_calibrated": "#882255",  # darker purple variant
    "knn_k10":             "#009E73",  # bluish green
}
LABELS = {
    "cdmeval":             "CDMEval (K=100, paper)",
    "cdmeval_calibrated":  "CDMEval (K=100, 60% train + iso-cal)",
    "irtnet_d100":         "IrtNet d=100",
    "irtnet_d232":         "IrtNet d=232",
    "embedllm_d232":       "EmbedLLM d=232 (paper-style 80% train)",
    "embedllm_calibrated": "EmbedLLM d=232 (60% train + iso-cal)",
    "knn_k10":             "KNN K=10",
}
MARKERS = {
    "cdmeval":             "o",
    "cdmeval_calibrated":  "o",
    "irtnet_d100":         "s",
    "irtnet_d232":         "^",
    "embedllm_d232":       "v",
    "embedllm_calibrated": "v",
    "knn_k10":             "D",
}
LINESTYLES = {
    "cdmeval":             "-",
    "cdmeval_calibrated":  "--",
    "irtnet_d100":         "-",
    "irtnet_d232":         "-",
    "embedllm_d232":       "-",
    "embedllm_calibrated": "--",
    "knn_k10":             "-",
}


def main() -> None:
    src = EXP / "v2_pareto_multibaseline_v4.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing — run tools/run_calibration_pareto.py first")
    d = json.load(open(src))

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.dpi": 140,
    })

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    plot_order = ["knn_k10", "irtnet_d100", "irtnet_d232",
                  "embedllm_d232", "embedllm_calibrated",
                  "cdmeval", "cdmeval_calibrated"]
    for key in plot_order:
        if key not in d["methods"]:
            continue
        sweep = d["methods"][key]["normalized"]
        x = [p["cost_pct"] for p in sweep]
        y = [p["acc_pct"] for p in sweep]
        order = sorted(range(len(x)), key=lambda i: (x[i], -y[i]))
        x = [x[i] for i in order]
        y = [y[i] for i in order]
        ax.plot(x, y, MARKERS[key] + LINESTYLES[key],
                 color=COLORS[key], lw=1.8, markersize=6,
                 label=LABELS[key], alpha=0.9)

    ax.axhline(100, ls=":", color="#888888", lw=0.7, zorder=0)
    ax.text(0.5, 100.5, "strongest single LLM (100% acc / 100% cost)",
             color="#888888", fontsize=8.5, va="bottom")

    ax.set_xlim(0, 105)
    ax.set_ylim(60, 120)
    ax.set_xlabel("Cost relative to strongest model (%)")
    ax.set_ylabel("Accuracy relative to strongest model (%)")
    ax.set_title("Cost-accuracy Pareto: matched cheapest-above-threshold + post-hoc isotonic calibration")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "fig_pareto_v4.pdf"
    out_png = FIG_DIR / "fig_pareto_v4.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
