"""Polished weak-beats-strong figure (single panel)."""
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

PHI_SIZES = [("phi-3.5-mini", 3.8), ("phi-3-mini", 3.8), ("phi-3-small", 7.0),
             ("phi-3-medium", 14.0), ("phi-2", 2.7), ("phi-1_5", 1.3)]


def parse_size(name):
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
    "Applying Combinatorial Principles To Dice Rolls": "Dice combinatorics",
    "Applying Quadratic Formula To Find Roots": "Quadratic formula",
    "Applying Divisibility Rules To Integer Sets": "Divisibility rules",
    "Calculating Orbital Period Ratio Between Planets": "Orbital periods",
}


def short(name, max_len=28):
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
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    setup_style()

    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)

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

    wbs_items = []
    wbs_by_bm = Counter()
    for it in test_items:
        gt = R[:, int(it)]
        if gt[strongest_idx] == 0 and gt[small_indices].sum() > 0:
            wbs_items.append(int(it))
            wbs_by_bm[items_data[int(it)].get("benchmark", "?")] += 1
    n_wbs = len(wbs_items)
    pct_wbs = n_wbs / len(test_items) * 100

    # Per-skill WBS (using only WBS items so each skill tells the right story)
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

    print(f"WBS items: {n_wbs} ({pct_wbs:.1f}%); skills with any WBS: {n_skills_total}", flush=True)

    bm_colors = {"MATH": "#3B82F6", "BBH": "#22C55E", "GPQA": "#F97316",
                 "MuSR": "#A855F7", "IFEval": "#EF4444"}

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    y = np.arange(len(top12))

    for i, s in enumerate(top12):
        c = bm_colors.get(s["benchmark"], "#999999")
        ax.plot([0, s["wbs_pct"]], [i, i], color=c, lw=2.0,
                solid_capstyle="round", zorder=3)
        ax.scatter(s["wbs_pct"], i, s=72, color=c, edgecolors="white",
                   linewidths=0.9, zorder=5)
        ax.text(s["wbs_pct"] + 1.5, i, f"{s['wbs_pct']:.0f}%",
                fontsize=8.5, va="center", color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([short(s["skill"]) for s in top12], fontsize=10)
    ax.set_xlabel("Items where some small model (≤13B) beats the strongest single model (%)",
                  fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="#bbbbbb")
    ax.grid(axis="x", alpha=0.15, lw=0.5, zorder=0)

    # Benchmark legend (above plot, centered, single row)
    from matplotlib.patches import Patch
    shown_bm = sorted(set(s["benchmark"] for s in top12),
                      key=["MATH", "BBH", "GPQA", "MuSR", "IFEval"].index)
    handles = [Patch(facecolor=bm_colors[b], label=b, edgecolor="none")
               for b in shown_bm]
    leg = ax.legend(handles=handles, fontsize=9.5, frameon=False,
                    loc="upper center", ncol=len(shown_bm),
                    bbox_to_anchor=(0.5, 1.10), columnspacing=1.6,
                    handlelength=1.0, handleheight=1.0)

    # Headline inset placed in the empty bottom-right region
    ax.text(0.99, 0.06,
            f"{n_wbs} / {len(test_items)} test items ({pct_wbs:.1f}%)\n"
            f"{n_skills_total} skills with WBS · top 12 shown",
            transform=ax.transAxes, fontsize=9, color="#222222",
            ha="right", va="bottom", style="italic",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.35", lw=0.6, alpha=0.95))

    plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.94])
    out = fig_dir / "main_ready" / "fig_weak_beats_strong.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **SAVE_KW)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
