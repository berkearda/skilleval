#!/usr/bin/env python3
"""Figure 2, compact one-row version (24 Sep 2026): benchmark-level prediction on held-out items, SkillEval against
IrtNet d=232, drawn at the paper's text width so it takes about half the height of the 2 x 3 version.

Same data, colours and markers as tools/fig_benchmark_prediction_irtnet.py (which keeps the 2 x 3 layout): reads
v2_benchpred_skilleval_vs_irtnet.{json,npz} (tools/diag_benchpred_skilleval_vs_irtnet.py, registered as
benchpred.skilleval_* and benchpred.irtnet232_*). One dot per LLM; x = observed accuracy on the benchmark's held-out
items, y = mean predicted probability of a correct answer over the same items. Font sizes are the printed sizes; the
script checks that every r label stays inside its panel.

    python tools/fig_benchmark_prediction_irtnet_row.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration/experiments"
OUT = REPO / "cdm_exploration/figures/report/main_ready/fig_benchmark_prediction_irtnet_row"
SKILL, IRTNET = "#0072B2", "#E69F00"
PANELS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval", "all"]
TITLES = {"all": "All"}


def main():
    res = json.load(open(EXP / "v2_benchpred_skilleval_vs_irtnet.json"))
    assert res["verified"], "diagnostic not verified"
    z = np.load(EXP / "v2_benchpred_skilleval_vs_irtnet.npz")
    true, bench = z["true"].astype(float), z["bench_te"]
    pred = {"SkillEval": z["pred_skilleval"], "IrtNet": z["pred_irtnet"]}

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "font.size": 7, "pdf.fonttype": 42, "mathtext.fontset": "stix", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    fig, axes = plt.subplots(1, 6, figsize=(5.5, 1.42), sharex=True, sharey=True)
    labels = []
    for ax, name in zip(axes, PANELS):
        m = np.ones(len(bench), bool) if name == "all" else bench == name
        t = true[:, m].mean(1)
        for label, colour, marker, alpha, z_order in (("IrtNet", IRTNET, "s", 0.30, 1), ("SkillEval", SKILL, "o", 0.55, 2)):
            ax.scatter(t, pred[label][:, m].mean(1), s=1.2, c=colour, marker=marker, alpha=alpha, linewidths=0,
                       zorder=z_order, rasterized=True)
        ax.plot([0, 1], [0, 1], color="#333333", lw=0.6, zorder=3)
        r_s, r_i = res["skilleval"]["r"][name], res["irtnet_d232_s42"]["r"][name]
        # r for each model, each line led by that model's marker (text stays in ink; the marker carries identity)
        box = ax.text(0.97, 0.035, f"\u2003$r$ = {r_s:.3f}\n\u2003$r$ = {r_i:.3f}", transform=ax.transAxes, ha="right",
                      va="bottom", fontsize=6, linespacing=1.3, color="#222222",
                      bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", facecolor="white", edgecolor="#cccccc",
                                linewidth=0.4))
        labels.append((ax, box))
        ax.set_title(f"{TITLES.get(name, name)} ({int(m.sum()):,} items)", fontsize=6.5, pad=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, linewidth=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Predicted accuracy", fontsize=7)
    fig.supxlabel("Observed accuracy", fontsize=7, y=0.06)
    handles = [plt.Line2D([], [], marker="o", ls="", color=SKILL, markersize=3.5, label="SkillEval"),
               plt.Line2D([], [], marker="s", ls="", color=IRTNET, markersize=3.5, label="IRTNet ($d{=}232$)"),
               plt.Line2D([], [], color="#333333", lw=0.6, label="perfect prediction")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=6.5,
               handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.2, wspace=0.2)
    renderer = fig.canvas.get_renderer()
    for ax, txt in labels:   # model markers inside the r box, in the space left by the leading em space
        e = txt.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        x = e.x0 + 0.075
        ax.plot([x], [e.y0 + 0.75 * e.height], transform=ax.transAxes, marker="o", ms=2.6, color=SKILL, ls="", zorder=5)
        ax.plot([x], [e.y0 + 0.27 * e.height], transform=ax.transAxes, marker="s", ms=2.4, color=IRTNET, ls="", zorder=5)
    for ax, txt in labels:   # every r label inside its panel
        a, e = ax.get_window_extent(renderer), txt.get_window_extent(renderer)
        assert e.x0 >= a.x0 and e.x1 <= a.x1 and e.y0 >= a.y0 and e.y1 <= a.y1, txt.get_text()
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("saved", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
