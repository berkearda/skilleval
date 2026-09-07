"""Weak-beats-strong v2 (locked Tailwind-700 palette, no header, less bold).

Renders the top-12 (out of 41) skill clusters with the highest fraction of
held-out items where some smaller (<=13B) LLM beats the global strongest
single LLM. The headline finding lives in the LaTeX caption only — no
matplotlib suptitle, no figure-level header, per Part J6 / J6.5.

Style anchored to NEURIPS_FIGURE_CHECKLIST.md Part J (locked):
  - J1 Tailwind-700 categorical palette (NO ORANGE)
  - J2 single sans-serif family, 8/9/10 hierarchy
  - J3 spines top+right hidden, faint x-grid only
  - J4 small in-frame legend (frameless)
  - J5 tight axis cropping, explicit "%" tick formatting
  - J6 no suptitle / no figure-level header
  - J6.5 bold only for axis labels / panel letters; never bars / annotations
  - J7 single highlight star at the lead bar

Numbers traced to:
  - response_matrix_v2_full.npy (R, ground-truth correctness)
  - qmatrix_v2_K100.npy (skill -> item incidence)
  - cluster_labels_v2_K100.json (skill names)
  - response_matrix_v2_full_items.json (per-item benchmark tag)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# Part J1 LOCKED palette (Tailwind-700 / Option K). NO ORANGE.
BENCH_COLORS = {
    "MATH":   "#1D4ED8",  # blue-700
    "BBH":    "#059669",  # emerald-700
    "GPQA":   "#BE185D",  # pink-700
    "MuSR":   "#7C3AED",  # violet-600
    "IFEval": "#0891B2",  # cyan-600
}
BENCH_MARKERS = {
    "MATH":   "o",
    "BBH":    "s",
    "GPQA":   "D",
    "MuSR":   "^",
    "IFEval": "v",
}
HIGHLIGHT = "#FACC15"   # yellow-500 (Part J1: headline star only)
REFERENCE = "#475569"   # slate-600 (annotation text / spines)

PHI_SIZES = [("phi-3.5-mini", 3.8), ("phi-3-mini", 3.8), ("phi-3-small", 7.0),
             ("phi-3-medium", 14.0), ("phi-2", 2.7), ("phi-1_5", 1.3)]


def parse_size(name: str) -> float:
    low = name.lower()
    parts = name.split("__")
    suffix = parts[-1] if len(parts) > 1 else name
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
        idx = suffix.find(m.group(0))
        if idx > 0 and suffix[idx - 1].lower() == "v":
            continue
        return float(m.group(1))
    for kw, sz in PHI_SIZES:
        if kw in low:
            return sz
    return np.nan


SHORT = {
    "Applying Molarity Concepts To Solution Mixing": "Molarity / solutions",
    "Applying Snell'S Law To Light Refraction In Media": "Snell's law (optics)",
    "Calculating Decay Length From Energy And Mass": "Decay length",
    "Applying Trigonometric Identities To Solve Equatio": "Trig. identities",
    "Interpreting Numerical Values In Context": "Numerical context",
    "Recognizing Symmetry In Two Dimensional Figures": "2D symmetry",
    "Applying Properties Of Complex Numbers In Equation": "Complex numbers",
    "Identifying Structural Isomers In Organic Compound": "Organic isomers",
    "Applying Combinatorial Principles To Dice Rolls": "Combinatorics (dice)",
    "Applying Quadratic Formula To Find Roots": "Quadratic formula",
    "Applying Divisibility Rules To Integer Sets": "Divisibility rules",
    "Calculating Orbital Period Ratio Between Planets": "Orbital periods",
}


def short(name: str, max_len: int = 28) -> str:
    if name in SHORT:
        return SHORT[name]
    s = name
    for prefix in ("Applying ", "Identifying ", "Calculating ", "Solving ",
                   "Evaluating ", "Working with ", "Understanding ",
                   "Interpreting ", "Analyzing "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.paths.cdm_ready)

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        cluster_labels = json.load(f)
    K = q_matrix.shape[1]
    n_items = R.shape[1]
    skill_names = [cluster_labels[str(i)] for i in range(K)]

    sizes = np.array([parse_size(n) for n in llm_names])
    train_items, test_items = train_test_split(
        np.arange(n_items), test_size=0.2, random_state=42)

    train_acc = R[:, train_items].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    small_mask = (sizes <= 13.0) & (~np.isnan(sizes))
    small_indices = np.where(small_mask)[0]

    # Item-level WBS
    wbs_items = []
    for it in test_items:
        gt = R[:, int(it)]
        if gt[strongest_idx] == 0 and gt[small_indices].sum() > 0:
            wbs_items.append(int(it))
    n_wbs = len(wbs_items)
    pct_wbs = n_wbs / len(test_items) * 100

    # Per-skill WBS
    wbs_per_skill = []
    for si in range(K):
        skill_items = test_items[q_matrix[test_items, si] > 0]
        if len(skill_items) < 10:
            continue
        gt_s = R[strongest_idx, skill_items.astype(int)]
        gt_sm = R[small_indices][:, skill_items.astype(int)].max(axis=0)
        wbs_n = int(((gt_s == 0) & (gt_sm > 0)).sum())
        bm_counts = Counter()
        for it in skill_items[:50]:
            bm_counts[items_data[int(it)].get("benchmark", "?")] += 1
        dom_bm = bm_counts.most_common(1)[0][0]
        wbs_per_skill.append({
            "skill": skill_names[si],
            "wbs_pct": wbs_n / len(skill_items) * 100,
            "n": len(skill_items), "benchmark": dom_bm,
        })
    wbs_per_skill.sort(key=lambda x: -x["wbs_pct"])
    top12 = wbs_per_skill[:12]
    n_skills_total = sum(1 for s in wbs_per_skill if s["wbs_pct"] > 0)

    print(f"WBS items: {n_wbs} ({pct_wbs:.1f}%); "
          f"skills with any WBS: {n_skills_total}", flush=True)

    # ---------- Part J2 typography (sans-serif, no bold-by-default) ----------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "font.weight": "regular",
        "axes.titleweight": "regular",
        "axes.labelweight": "regular",
    })

    # Tighter compact print width (single-column ~5.5")
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    y = np.arange(len(top12))

    # ---------- Bars (lollipop): redundant encoding via marker shape ----------
    for i, s in enumerate(top12):
        c = BENCH_COLORS.get(s["benchmark"], "#888888")
        m = BENCH_MARKERS.get(s["benchmark"], "o")
        ax.plot([0, s["wbs_pct"]], [i, i], color=c, lw=1.8,
                solid_capstyle="round", alpha=0.85, zorder=3)
        ax.scatter(s["wbs_pct"], i, marker=m, s=46, color=c,
                   edgecolors="white", linewidths=0.9, zorder=5)
        # Value label: regular weight (J6.5)
        ax.text(s["wbs_pct"] + 1.6, i, f"{s['wbs_pct']:.0f}%",
                fontsize=7.5, va="center", ha="left",
                color=REFERENCE)

    # ---------- Headline highlight star (single, J7) ----------
    lead = top12[0]
    ax.scatter(lead["wbs_pct"], 0, marker="*", s=170, color=HIGHLIGHT,
               edgecolors="#1F2937", linewidths=0.6, zorder=8)

    # ---------- Y axis: skill labels ----------
    ax.set_yticks(y)
    ax.set_yticklabels([short(s["skill"]) for s in top12])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)

    # ---------- X axis: 0..100 with explicit % formatting ----------
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([f"{v}%" for v in [0, 20, 40, 60, 80, 100]])
    # Axis label: bold allowed by J6.5 for readability
    ax.set_xlabel(
        r"Held-out items where a smaller LLM ($\leq$13B) beats the strongest model",
        fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", length=3, color="#bbbbbb")

    # ---------- Grid, spines (J3) ----------
    for sp in ["left", "top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(REFERENCE)
    ax.grid(axis="x", alpha=0.13, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ---------- Legend: small, frameless, in-frame (J4) ----------
    from matplotlib.lines import Line2D
    shown_bm = sorted(
        set(s["benchmark"] for s in top12),
        key=lambda b: ["MATH", "BBH", "GPQA", "MuSR", "IFEval"].index(b),
    )
    handles = [
        Line2D([0], [0], color=BENCH_COLORS[b], marker=BENCH_MARKERS[b],
               markersize=6, lw=1.8, label=b,
               markeredgecolor="white", markeredgewidth=0.8)
        for b in shown_bm
    ]
    # Place legend in lower-right, frameless. Bars 9-11 end at ~60-65%, so
    # x>=70 / y>=8 is clear of all bars.
    ax.legend(handles=handles, fontsize=8, frameon=False,
              loc="lower right", bbox_to_anchor=(1.0, 0.02),
              ncol=1, columnspacing=1.2,
              handletextpad=0.5, handlelength=1.3,
              labelcolor=REFERENCE)

    # NO suptitle / no figure-level header (J6).
    # Finding lives in the LaTeX caption only.

    plt.tight_layout(pad=0.4)

    review_dir = Path("cdm_exploration/"
                      "figures/review_batch2")
    review_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = review_dir / "fig_weak_beats_strong_REVISED.pdf"
    out_png = review_dir / "fig_weak_beats_strong_REVISED.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
