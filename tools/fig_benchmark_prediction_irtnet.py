#!/usr/bin/env python3
"""Figure 6 (revised, 23 Sep 2026): benchmark-level prediction on held-out items, SkillEval against IrtNet d=232.

Replaces the IRT 2PL series of fig_benchmark_prediction.pdf: on the item-wise split the 2PL has no trained parameters for
held-out items, so its series was not a 2PL prediction. IrtNet derives item parameters from text like SkillEval and is a
genuine OOD comparator. Reads v2_benchpred_skilleval_vs_irtnet.{json,npz} (tools/diag_benchpred_skilleval_vs_irtnet.py,
registered as benchpred.skilleval_* and benchpred.irtnet232_*). One dot per LLM; x = observed accuracy on the benchmark's
held-out items, y = mean predicted probability of a correct answer over the same items. Okabe-Ito colours plus marker
shape as a second encoding (palette checked with the dataviz validator).

    python tools/fig_benchmark_prediction_irtnet.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration/experiments"
OUT = REPO / "cdm_exploration/figures/report/main_ready/fig_benchmark_prediction_irtnet"
SKILL, IRTNET = "#0072B2", "#E69F00"
PANELS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval", "all"]
TITLES = {"all": "All benchmarks"}


def main():
    res = json.load(open(EXP / "v2_benchpred_skilleval_vs_irtnet.json"))
    assert res["verified"], "diagnostic not verified"
    z = np.load(EXP / "v2_benchpred_skilleval_vs_irtnet.npz")
    true, bench = z["true"].astype(float), z["bench_te"]
    pred = {"SkillEval": z["pred_skilleval"], "IrtNet": z["pred_irtnet"]}

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), PANELS):
        m = np.ones(len(bench), bool) if name == "all" else bench == name
        t = true[:, m].mean(1)
        for label, colour, marker, alpha, z_order in (("IrtNet", IRTNET, "s", 0.30, 1), ("SkillEval", SKILL, "o", 0.55, 2)):
            ax.scatter(t, pred[label][:, m].mean(1), s=4, c=colour, marker=marker, alpha=alpha, linewidths=0, zorder=z_order,
                       rasterized=True)
        ax.plot([0, 1], [0, 1], color="#333333", lw=0.9, zorder=3)
        r_s, r_i = res["skilleval"]["r"][name], res["irtnet_d232_s42"]["r"][name]
        ax.text(0.97, 0.05, f"SkillEval  r = {r_s:.3f}\nIRTNet     r = {r_i:.3f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", linewidth=0.6))
        n_items = int(m.sum())
        ax.set_title(f"{TITLES.get(name, name)}  ({n_items:,} held-out items)", fontsize=10, loc="left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(True, alpha=0.15, linewidth=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Observed accuracy")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted accuracy")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SKILL, markersize=6, label="SkillEval"),
               plt.Line2D([], [], marker="s", ls="", color=IRTNET, markersize=6, label="IRTNet ($d{=}232$)"),
               plt.Line2D([], [], color="#333333", lw=0.9, label="perfect prediction")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("saved", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
