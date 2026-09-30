#!/usr/bin/env python3
"""Figure for Section 4.2 (T-137): profiling a new LLM from N answers, drawn from v2_profiling_bayes.json (the summary
written by tools/run_profiling_bayes.py; no model is run here).

(a) Benchmark-score error: mean absolute difference between predicted and real accuracy on the held-out items,
    averaged over the new LLMs and the five benchmarks (lower is better).
(b) Profile agreement: per skill, Pearson r across the new LLMs between the profile from N answers and the profile
    from all 7,618 answers estimated WITHOUT the prior (strict version, v2_profiling_bayes_robustness.json), averaged
    over skills (higher is better).
Lines: SkillEval from the average profile with random or adaptive questions and, in (a), the simple count. (The old
recipe that starts every skill at 0.5 is kept in the result files but not shown: Berke, 24 Sep 2026.) Drawn at the paper's text width, Okabe-Ito colours, markers as a second cue.

    python tools/fig_profiling_bayes.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration/experiments"
OUT = REPO / "cdm_exploration/figures/report/main_ready/fig_profiling_bayes"
C = {"random": "#0072B2", "count": "#009E73", "adaptive": "#D55E00"}   # first three of the order validated with the dataviz script


def main():
    res = json.load(open(EXP / "v2_profiling_bayes.json"))
    assert res["verified"]
    rows = res["summary"]
    strict = {r["N"]: r for r in json.load(open(EXP / "v2_profiling_bayes_robustness.json"))["rows"]}
    for r in rows:   # panel (b) uses the strict reference (no prior)
        st = strict[r["N"]]
        r["random_profile_agreement"], r["old_profile_agreement"] = st["random_noprior_ref_all"], st["old_noprior_ref_all"]
        if "adaptive_noprior_ref_all" in st:
            r["adaptive_profile_agreement"] = st["adaptive_noprior_ref_all"]
    ext = {r["N"]: r for r in json.load(open(EXP / "v2_profiling_bayes_adaptive500.json"))["rows"]}
    for r in rows:   # adaptive selection up to 500 answers (budgets 5-100 reproduce the first run exactly)
        if r["N"] in ext:
            r["adaptive_bench_err"] = ext[r["N"]]["adaptive_bench_err"]
            r["adaptive_profile_agreement"] = ext[r["N"]]["adaptive_agree_strict"]
    N = [r["N"] for r in rows]
    x = list(range(len(N)))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "font.size": 7, "pdf.fonttype": 42, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 1.75))

    def line(ax, key, series, label, marker, ls="-"):
        pts = [(xi, r[key]) for xi, r in zip(x, rows) if key in r]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[series], marker=marker, ms=3, lw=1.1, ls=ls,
                label=label)

    line(ax_a, "random_bench_err", "random", "SkillEval, random questions", "o")
    line(ax_a, "count_bench_err", "count", "Simple count", "^")
    line(ax_a, "adaptive_bench_err", "adaptive", "SkillEval, adaptive questions", "s")
    ax_a.axhline(res["full_answers"]["bench_err"], color="#222222", lw=0.6, ls=":")
    ax_a.text(0.1, res["full_answers"]["bench_err"] - 0.003, "all 7,618 answers", ha="left", va="top", fontsize=6,
              color="#444444")
    ax_a.set_ylim(0.015, 0.23)
    ax_a.set_ylabel("Mean absolute error")
    ax_a.set_title("(a) Predicting benchmark scores (lower is better)", fontsize=7.5, loc="left", pad=3)

    line(ax_b, "random_profile_agreement", "random", "SkillEval, random questions", "o")
    line(ax_b, "adaptive_profile_agreement", "adaptive", "SkillEval, adaptive questions", "s")
    ax_b.set_ylim(0, 0.72)
    ax_b.set_ylabel("Agreement with full profile")
    ax_b.set_title("(b) Recovering the skill profile (higher is better)", fontsize=7.5, loc="left", pad=3)

    for ax in (ax_a, ax_b):
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n in N])
        ax.set_xlabel("Number of answers from the new LLM")
        ax.grid(True, alpha=0.2, linewidth=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = ax_a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=6.5,
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    print("saved", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
