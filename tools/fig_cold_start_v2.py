"""Cold-start figure (T-036 fixed trajectory).

Reads cdm_exploration/experiments/v2_cold_start_fixed.json and renders
AUC vs N calibration items for 763 held-out LLMs (mean +- std over 3 seeds).

Style anchored to NEURIPS_FIGURE_CHECKLIST.md Part J:
  - protagonist color: Okabe-Ito blue #0072B2
  - reference lines: dotted gray #888888
  - sans-serif typography (DejaVu Sans), tick 8pt / axis 10pt
  - tight axis cropping; spines top+right hidden; faint y-grid only
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig

# Part J palette (LOCKED: Tailwind-700 / Option K)
PROTAGONIST = "#1D4ED8"  # blue-700
REFERENCE = "#475569"    # slate-600
HIGHLIGHT = "#FACC15"    # yellow-500 (headline star only)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    fig_dir = Path(cfg.paths.figures)
    # +in_tag=<suffix> draws from another cold-start result file and writes a figure with the same suffix (T-123).
    in_tag = str(cfg.in_tag) if hasattr(cfg, "in_tag") else ""
    data_path = Path(f"cdm_exploration/experiments/v2_cold_start_fixed{in_tag}.json")
    data = json.load(open(data_path))

    Ns = [r["N"] for r in data["summary"]]
    aucs = np.array([r["auc_mean"] for r in data["summary"]])
    stds = np.array([r["auc_std"] for r in data["summary"]])
    # +ref=full_calibration uses the same-LLM ceiling (each held-out LLM calibrated on all training items), which is the
    # reference the paper text uses; the default keeps the submitted reference (50 training LLMs with trained vectors).
    same_llm = hasattr(cfg, "ref") and str(cfg.ref) == "full_calibration"
    pcts = np.array([r["pct_of_full_calibration" if same_llm else "pct_of_full"] for r in data["summary"]])
    full = data["full_calibration_auc_same_llms"] if same_llm else data["full_training_auc"]
    ref_name = "full-calibration" if same_llm else "full-training"

    # Per Part J: sans-serif, tight typography hierarchy
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    # Symlog-like trick: place N=0 at x=0.5 so it appears on the log axis
    x = np.array([max(n, 0.5) for n in Ns])

    # Full-training reference (Part J: reference dotted, neutral gray)
    ax.axhline(full, color=REFERENCE, lw=1.2, ls=":", zorder=1)
    ax.text(0.45, full + 0.002, f"{ref_name} AUC = {full:.3f}", fontsize=8,      # above the line, left, clear of the data
            color=REFERENCE, ha="left", va="bottom", style="italic")

    # Data: error band + line + open markers
    ax.fill_between(x, aucs - stds, aucs + stds,
                    color=PROTAGONIST, alpha=0.15, zorder=2, linewidth=0)
    ax.plot(x, aucs, "-", color=PROTAGONIST, lw=2.0, zorder=3)
    ax.plot(x, aucs, "o", color=PROTAGONIST, markersize=6,
            markerfacecolor="white", markeredgewidth=1.6, zorder=4)

    # Inline % labels at each operating point (Part J: inline labels)
    for xi, ai, pi, n in zip(x, aucs, pcts, Ns):
        if n == 0:
            ax.annotate(f"{pi:.0f}%", xy=(xi, ai),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=8.5, color=PROTAGONIST,
                        ha="left", va="bottom")
        elif n == 500:                                   # left of the star, below the reference line
            ax.annotate(f"{pi:.0f}%", xy=(xi, ai),
                        xytext=(-11, 2), textcoords="offset points",
                        fontsize=8.5, color=PROTAGONIST,
                        ha="right", va="bottom")
        elif n in (50, 100):
            ax.annotate(f"{pi:.0f}%", xy=(xi, ai),
                        xytext=(0, 9), textcoords="offset points",
                        fontsize=8.5, color=PROTAGONIST,
                        ha="center", va="bottom")

    # Star at headline operating point: N=500 reaches 97% of full
    ax.plot(500, aucs[-1], "*", color=HIGHLIGHT, markersize=14,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    # Inline punchline (Part J E4: headline number ON the figure)
    ax.annotate(f"$N{{=}}500$ recovers {pcts[-1]:.0f}% of\n{ref_name} AUC",
                xy=(500, aucs[-1]),
                xytext=(60, 0.585),
                fontsize=9, color="#222222", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#888888",
                                lw=0.8, alpha=0.7,
                                connectionstyle="arc3,rad=-0.15"))

    # Axes
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 5, 10, 50, 100, 500])
    ax.set_xticklabels(["0", "1", "5", "10", "50", "100", "500"])
    ax.set_xlim(0.4, 1500)
    ax.set_ylim(0.50, 0.72)
    ax.set_xlabel("Calibration items per new LLM ($N$, log scale)", fontsize=10)
    ax.set_ylabel("AUC on held-out items", fontsize=10)

    # Spines + grid (Part J: only left+bottom, faint y-grid)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.12, lw=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    out = fig_dir / "main_ready" / f"fig_cold_start_v2{in_tag}{'_fullcal' if same_llm else ''}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out}")
    print(f"Saved: {str(out).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
