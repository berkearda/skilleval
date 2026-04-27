"""Polish-only re-render of fig_pareto_v2 from cached JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

CDM_COLOR = "#1E40AF"   # deep blue
BEST_CHEAP_COLOR = "#059669"  # emerald green (single-model baseline)
STAR_COLOR = "#DC2626"  # strong red


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    setup_style()

    fig_dir = Path(cfg.paths.figures)
    d = json.load(open("cdm_exploration/experiments/v2_pareto_routing.json"))
    diag = json.load(open("cdm_exploration/experiments/v2_pareto_diagnostic.json"))

    cdm = d["cdm_normalized"]
    cdm_x = [p["cost_pct"] for p in cdm]
    cdm_y = [p["acc_pct"] for p in cdm]

    # Ceiling point: highest CDMEval accuracy (beats strongest single model)
    ceiling = max(cdm, key=lambda p: p["acc_pct"])

    # Best single cheap LLM from diagnostic (verified on full test set)
    dominators = diag["results"]["single_model_dominators_at_irt_pareto_point"]
    best_cheap = max(dominators, key=lambda p: p["acc_pct"])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.8, 4.4),
        gridspec_kw={"width_ratios": [6, 1], "wspace": 0.06},
    )

    # ---- Left panel: CDMEval Pareto curve over 0-40% cost ----
    cdm_h, = ax.plot(cdm_x, cdm_y, "o-", color=CDM_COLOR, lw=2.0,
                     markersize=6, label="CDMEval router (K=100)",
                     zorder=4)

    # Best single cheap LLM (baseline): one diamond marker
    best_h, = ax.plot(
        best_cheap["cost_pct"], best_cheap["acc_pct"],
        "D", color=BEST_CHEAP_COLOR, markersize=10,
        markeredgecolor="white", markeredgewidth=0.8,
        label="Best single cheap LLM", zorder=5,
    )
    ax.annotate(
        f"{best_cheap['acc_pct']:.0f}% @ {best_cheap['cost_pct']:.1f}%\n(single model, no router)",
        xy=(best_cheap["cost_pct"], best_cheap["acc_pct"]),
        xytext=(best_cheap["cost_pct"] + 3, best_cheap["acc_pct"] + 4),
        fontsize=9, color=BEST_CHEAP_COLOR, fontweight="bold",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=BEST_CHEAP_COLOR, lw=0.7),
        zorder=8,
    )

    # threshold labels on CDM curve
    for p in cdm:
        if p["t"] in {0.30, 0.50, 0.70, 0.90}:
            ax.annotate(fr"$\tau$={p['t']:.1f}",
                        xy=(p["cost_pct"], p["acc_pct"]),
                        xytext=(6, -10), textcoords="offset points",
                        fontsize=8, color=CDM_COLOR, alpha=0.85)

    # Ceiling marker: CDM's unique win - exceeds strongest single model
    cx, cy = ceiling["cost_pct"], ceiling["acc_pct"]
    ax.axhline(100, ls=":", color="#888888", lw=0.8, zorder=1)
    ax.plot(cx, cy, "o", markersize=11, markerfacecolor="none",
            markeredgecolor="#111111", markeredgewidth=1.4, zorder=6)
    ax.annotate(
        f"{cy:.0f}% of strongest\nat {cx:.0f}% cost\n"
        fr"($\tau$={ceiling['t']:.2f}, beats strongest)",
        xy=(cx, cy), xytext=(18, 72),
        fontsize=9.5, color="#111111", fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#888888", lw=0.7, alpha=0.97),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8,
                        connectionstyle="arc3,rad=-0.15"),
        zorder=7,
    )

    ax.set_xlim(-1, 42)
    ax.set_ylim(60, 108)
    ax.set_ylabel("Accuracy relative to strongest model (%)", fontsize=11)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Right panel: strongest-model reference at 100% ----
    star_h, = ax2.plot(100, 100, "*", color=STAR_COLOR, markersize=20,
                       markeredgecolor="white", markeredgewidth=0.6,
                       label="Strongest single model", zorder=5)
    ax2.set_xlim(96, 104)
    ax2.set_ylim(60, 108)
    ax2.set_xticks([100])
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.15, lw=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Broken-axis diagonal marks
    d_len = 0.015
    kw = dict(transform=ax.transAxes, color="#666666",
              clip_on=False, lw=1.0)
    ax.plot([1 - d_len, 1 + d_len], [-d_len, +d_len], **kw)
    kw = dict(transform=ax2.transAxes, color="#666666",
              clip_on=False, lw=1.0)
    ax2.plot([-d_len * 6, +d_len * 6], [-d_len, +d_len], **kw)

    # Explicit subplot adjustment to leave room below for label+legend
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.28,
                        wspace=0.04)

    # Shared x-label (spans both panels) - placed below x-ticks
    fig.text(0.54, 0.15, "Cost relative to strongest model (%)",
             ha="center", va="center", fontsize=11)

    # Shared legend below x-label
    fig.legend(handles=[cdm_h, best_h, star_h],
               loc="lower center", bbox_to_anchor=(0.54, 0.02),
               ncol=3, frameon=False, fontsize=9.5, handlelength=2.0)
    out = fig_dir / "main_ready" / "fig_pareto_v2.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **SAVE_KW)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180,
                bbox_inches="tight")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
