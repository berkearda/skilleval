"""Render fig_pareto_v5 with K-sweep + beta calibration + ensemble (when ready).

Reads from v2_pareto_multibaseline_v5.json which extends v4 with:
  - cdmeval_K{50,150,200,300,400,500} canonical-train CDMEval at varying K
  - cdmeval_beta_calibrated (60% train + Beta calibration on val)
  - ensemble_cdm_embedllm (60% train + LR stack — needs corrected EmbedLLM)
  - ensemble_uniform_5050 (constrained 50/50 average)
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

COLORS = {
    "cdmeval":              "#0072B2",  # blue
    "cdmeval_calibrated":   "#56B4E9",  # sky blue (iso-cal CDMEval 60%)
    "cdmeval_beta_calibrated": "#003366",  # navy (beta-cal CDMEval 60%)
    "cdmeval_K50":          "#88AABB",
    "cdmeval_K150":         "#88AABB",
    "cdmeval_K200":         "#88AABB",
    "cdmeval_K300":         "#88AABB",
    "cdmeval_K400":         "#88AABB",
    "cdmeval_K500":         "#88AABB",
    "irtnet_d100":          "#E69F00",
    "irtnet_d232":          "#D55E00",
    "embedllm_d232":        "#CC79A7",
    "embedllm_calibrated":  "#882255",
    "ensemble_cdm_embedllm": "#000000",
    "ensemble_uniform_5050": "#666666",
    "knn_k10":              "#009E73",
}
LABELS = {
    "cdmeval":              "CDMEval (K=100, paper)",
    "cdmeval_calibrated":   "CDMEval (60% + iso-cal)",
    "cdmeval_beta_calibrated": "CDMEval (60% + beta-cal)",
    "cdmeval_K50":          "CDMEval K=50",
    "cdmeval_K150":         "CDMEval K=150",
    "cdmeval_K200":         "CDMEval K=200",
    "cdmeval_K300":         "CDMEval K=300",
    "cdmeval_K400":         "CDMEval K=400",
    "cdmeval_K500":         "CDMEval K=500",
    "irtnet_d100":          "IrtNet d=100",
    "irtnet_d232":          "IrtNet d=232",
    "embedllm_d232":        "EmbedLLM d=232",
    "embedllm_calibrated":  "EmbedLLM (60% + iso-cal)",
    "ensemble_cdm_embedllm": "Ensemble (LR stack)",
    "ensemble_uniform_5050": "Ensemble (50/50)",
    "knn_k10":              "KNN K=10",
}
MARKERS = {
    "cdmeval":              "o",
    "cdmeval_calibrated":   "o",
    "cdmeval_beta_calibrated": "o",
    "cdmeval_K50":          ".",
    "cdmeval_K150":         ".",
    "cdmeval_K200":         ".",
    "cdmeval_K300":         ".",
    "cdmeval_K400":         ".",
    "cdmeval_K500":         ".",
    "irtnet_d100":          "s",
    "irtnet_d232":          "^",
    "embedllm_d232":        "v",
    "embedllm_calibrated":  "v",
    "ensemble_cdm_embedllm": "*",
    "ensemble_uniform_5050": "*",
    "knn_k10":              "D",
}


def main(layout: str = "main") -> None:
    """layout='main' shows main story; layout='kswep' shows the K-sweep family."""
    src = EXP / "v2_pareto_multibaseline_v5.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run run_pareto_extensions.py")
    d = json.load(open(src))

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 140,
    })

    if layout == "main":
        # Main figure: KNN floor, IrtNet d=100/232, EmbedLLM, CDMEval variants,
        # ensemble (when available)
        plot_order = [
            "knn_k10",
            "irtnet_d100", "irtnet_d232",
            "embedllm_d232", "embedllm_calibrated",
            "cdmeval", "cdmeval_calibrated", "cdmeval_beta_calibrated",
            "ensemble_uniform_5050", "ensemble_cdm_embedllm",
        ]
        title = "Cost-accuracy Pareto: matched cheapest-above-threshold (audit-corrected)"
        out_name = "fig_pareto_v5"
    elif layout == "kswep":
        plot_order = [
            "cdmeval_K50", "cdmeval", "cdmeval_K150", "cdmeval_K200",
            "cdmeval_K300", "cdmeval_K400", "cdmeval_K500",
            "embedllm_d232",
        ]
        title = "CDMEval Pareto across K ∈ {50, 100, 150, 200, 300, 400, 500} vs EmbedLLM"
        out_name = "fig_pareto_v5_kswep"
    else:
        raise ValueError(f"Unknown layout: {layout}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key in plot_order:
        if key not in d["methods"]:
            print(f"  SKIP {key}: not in JSON")
            continue
        sweep = d["methods"][key]["normalized"]
        x = [p["cost_pct"] for p in sweep]
        y = [p["acc_pct"] for p in sweep]
        order = sorted(range(len(x)), key=lambda i: (x[i], -y[i]))
        x = [x[i] for i in order]
        y = [y[i] for i in order]
        # K-sweep curves de-emphasized in main; emphasized in kswep
        if "K" in key and key not in {"cdmeval", "cdmeval_calibrated",
                                      "cdmeval_beta_calibrated"} and layout == "main":
            ax.plot(x, y, MARKERS[key] + "-", color=COLORS[key], lw=0.8,
                    markersize=3, alpha=0.45)
        else:
            ax.plot(x, y, MARKERS[key] + "-", color=COLORS[key], lw=1.7,
                    markersize=6, label=LABELS[key], alpha=0.9)

    ax.axhline(100, ls=":", color="#888888", lw=0.7, zorder=0)
    ax.text(0.5, 100.7, "strongest single LLM (100% acc / 100% cost)",
             color="#888888", fontsize=8.5, va="bottom")

    ax.set_xlim(0, 105)
    ax.set_ylim(60, 120)
    ax.set_xlabel("Cost relative to strongest model (%)")
    ax.set_ylabel("Accuracy relative to strongest model (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower right", ncol=1)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / f"{out_name}.pdf"
    out_png = FIG_DIR / f"{out_name}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main("main")
    main("kswep")
