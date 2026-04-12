"""Cold-start figure: AUC vs N calibration items for new LLMs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig

CDM_COLOR = "#1E40AF"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    setup_style()

    fig_dir = Path(cfg.paths.figures)
    data = json.load(open("cdm_exploration/experiments/v2_cold_start.json"))

    Ns = [r["N"] for r in data["summary"]]
    aucs = np.array([r["auc_mean"] for r in data["summary"]])
    stds = np.array([r["auc_std"] for r in data["summary"]])
    full = data["full_training_auc"]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    # Random baseline
    ax.axhline(0.5, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.text(0.45, 0.502, "random AUC = 0.500", fontsize=8, color="#666666",
            ha="left", va="bottom")

    # Full training reference
    ax.axhline(full, color="#22C55E", lw=1.0, ls="--", zorder=1)
    ax.text(0.45, full + 0.004, f"full training AUC = {full:.3f}",
            fontsize=8, color="#15803D", ha="left", va="bottom",
            fontweight="bold")

    # Use symlog so N=0 sits at left edge
    x = np.array([max(n, 0.5) for n in Ns])

    ax.fill_between(x, aucs - stds, aucs + stds,
                    color=CDM_COLOR, alpha=0.18, zorder=2)
    ax.plot(x, aucs, "o-", color=CDM_COLOR, lw=1.8, markersize=6,
            markerfacecolor="white", markeredgewidth=1.6, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 5, 10, 50, 100, 500])
    ax.set_xticklabels(["0", "1", "5", "10", "50", "100", "500"])
    ax.set_xlim(0.4, 900)
    ax.set_ylim(0.48, max(0.74, full + 0.02))
    ax.set_xlabel("Calibration items per new LLM", fontsize=11)
    ax.set_ylabel("AUC on held-out items", fontsize=11)
    ax.grid(True, axis="y", alpha=0.15, lw=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    pct0 = data["summary"][0]["pct_of_full"]
    pct500 = data["summary"][-1]["pct_of_full"]

    # Annotate endpoints above the curve
    ax.annotate(f"N=0: {aucs[0]:.3f}\n({pct0:.0f}% of full)",
                xy=(0.5, aucs[0]), xytext=(1.0, aucs[0] + 0.03),
                fontsize=8.5, color=CDM_COLOR, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=CDM_COLOR, lw=0.6))
    ax.annotate(f"N=500: {aucs[-1]:.3f}\n({pct500:.0f}% of full)",
                xy=(500, aucs[-1]), xytext=(60, aucs[-1] + 0.025),
                fontsize=8.5, color=CDM_COLOR, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=CDM_COLOR, lw=0.6))

    # Bracket showing the gap to full training at N=500
    gap = full - aucs[-1]
    ax.annotate("", xy=(500, full), xytext=(500, aucs[-1]),
                arrowprops=dict(arrowstyle="<->", color="#666666", lw=0.9))
    ax.text(560, (full + aucs[-1]) / 2, f"Δ = {gap:.3f}\n(remaining\ngap)",
            fontsize=8, color="#444444", va="center", ha="left")

    plt.tight_layout(pad=1.0)
    out = fig_dir / "main_ready" / "fig_cold_start_v2.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **SAVE_KW)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
