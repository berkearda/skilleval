"""Polish-only re-render of fig_pareto_v2 from cached JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

CDM_COLOR = "#1E40AF"   # deep blue (matches benchmark prediction figure)
IRT_COLOR = "#FCA5A5"   # soft coral
STAR_COLOR = "#DC2626"  # strong red


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    setup_style()

    fig_dir = Path(cfg.paths.figures)
    d = json.load(open("cdm_exploration/experiments/v2_pareto_routing.json"))

    cdm = d["cdm_normalized"]
    irt = d["irt_normalized"]
    sweet = d["sweet_spot"]

    cdm_x = [p["cost_pct"] for p in cdm]
    cdm_y = [p["acc_pct"] for p in cdm]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    # 100% reference line (subtle)
    ax.axhline(100, ls=":", color="#bbbbbb", lw=0.8, zorder=0)

    cdm_h, = ax.plot(cdm_x, cdm_y, "o-", color=CDM_COLOR, lw=2.0,
                     markersize=6, label="CDMEval (K=100)", zorder=4)
    star_h, = ax.plot(100, 100, "*", color=STAR_COLOR, markersize=18,
                      zorder=5, markeredgecolor="white", markeredgewidth=0.6,
                      label="Strongest single model")

    # Note: CDM can exceed 100% by routing per-query to non-strongest models
    ax.text(55, 102, "router beats strongest\nvia per-query routing",
            fontsize=9, color="#222222", style="italic", ha="left",
            va="center", zorder=10)
    ax.plot([cdm_x[-1] + 1, 54], [cdm_y[-1], 102.5], "-",
            color="#888888", lw=0.6, zorder=10)

    # Annotate selected thresholds on CDM
    for p in cdm:
        if p["t"] in {0.30, 0.50, 0.70, 0.90}:
            ax.annotate(f"t={p['t']:.1f}",
                        xy=(p["cost_pct"], p["acc_pct"]),
                        xytext=(7, -11), textcoords="offset points",
                        fontsize=8, color=CDM_COLOR, alpha=0.85)

    # Sweet spot: subtle inline marker + small label
    sx, sy = sweet["cost_pct"], sweet["acc_pct"]
    ax.plot([sx, sx], [0, sy], ":", color="#666666", lw=0.9, zorder=1)
    ax.plot([0, sx], [sy, sy], ":", color="#666666", lw=0.9, zorder=1)
    ax.plot(sx, sy, "o", markersize=10, markerfacecolor="none",
            markeredgecolor="#111111", markeredgewidth=1.4, zorder=6)
    ax.annotate(
        f"{sy:.0f}% accuracy\nat {sx:.0f}% cost",
        xy=(sx, sy), xytext=(45, 78),
        fontsize=10, color="#111111", fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#888888", lw=0.7, alpha=0.97),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8))

    ax.set_xlim(-2, 105)
    ax.set_ylim(60, 105)
    ax.set_xlabel("Cost relative to strongest model (%)", fontsize=11)
    ax.set_ylabel("Accuracy relative to strongest model (%)", fontsize=11)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.grid(True, alpha=0.15, lw=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.legend(handles=[cdm_h, star_h],
              frameon=False, fontsize=9.5, loc="lower right",
              handlelength=2.0)

    plt.tight_layout(pad=1.0)
    out = fig_dir / "main_ready" / "fig_pareto_v2.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **SAVE_KW)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
