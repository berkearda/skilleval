"""F06 v2 model-family profile figure (batch 2 audit, 2026-05-04).

Three-panel design (preserved from batch 1 revised) with the locked
Tailwind-700 palette and tightened layout:

  (a) Llama mean mastery on the 8 skills with the LARGEST signed
      Qwen-minus-Llama gap.
  (b) Qwen mean mastery on the same 8 skills, sharing y-axis with (a).
  (c) Signed gap (Qwen - Llama) as a horizontal bar with the dominant
      family colored, plus per-skill benchmark tag at the right margin.

Changes vs. batch 1 revised script:
  - Palette swapped to LOCKED Tailwind-700 (no orange):
      Llama #1D4ED8, Qwen #BE185D, axis #475569,
      benchmark tags use Categorical-5 from Part J.
  - Suptitle / figure-level finding header REMOVED (lives in caption).
  - Bold font weight removed everywhere except panel letters (a) (b) (c).
  - Tighter layout (smaller figure height, reduced wspace, smaller pad).
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
import torch
import hydra
from omegaconf import DictConfig


PHI_SIZES = [
    ("phi-3.5-mini", 3.8), ("phi-3-mini", 3.8), ("phi-3-small", 7.0),
    ("phi-3-medium", 14.0), ("phi-2", 2.7), ("phi-1_5", 1.3),
]

# LOCKED Tailwind-700 palette (Part J1, batch-2 update)
LLAMA_C = "#1D4ED8"   # blue-700  (protagonist family in F06 prose)
QWEN_C = "#BE185D"    # pink-700  (comparator family)
AXIS_C = "#475569"    # slate-600 (reference / axis / neutral text)

# Categorical-5 for benchmark tags (Part J1)
BENCH_COLOR = {
    "MATH": "#1D4ED8",     # blue-700
    "BBH": "#059669",      # emerald-700
    "GPQA": "#BE185D",     # pink-700
    "MuSR": "#7C3AED",     # violet-600
    "IFEval": "#0891B2",   # cyan-600
}


def parse_family_and_size(name):
    low = name.lower()
    fam = "Other"
    for kw, f in [("llama", "Llama"), ("qwen", "Qwen"), ("mistral", "Mistral"),
                  ("gemma", "Gemma"), ("phi-", "Phi"), ("phi2", "Phi"),
                  ("phi3", "Phi")]:
        if kw in low:
            fam = f
            break
    suffix = name.split("__")[-1] if "__" in name else name
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe:
        return fam, float(moe.group(1)) * float(moe.group(2))
    for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
        idx = suffix.find(m.group(0))
        if idx > 0 and suffix[idx - 1].lower() == "v":
            continue
        return fam, float(m.group(1))
    if fam == "Phi":
        for kw, sz in PHI_SIZES:
            if kw in low:
                return fam, sz
    return fam, None


SHORT = {
    "Applying Quadratic Formula To Find Roots": "Quadratic formula",
    "Applying Trigonometric Identities To Solve Equatio": "Trig. identities",
    "Calculating Decay Length From Energy And Mass": "Decay length",
    "Identifying Structural Isomers In Organic Compound": "Organic isomers",
    "Recognizing Symmetry In Two Dimensional Figures": "2D symmetry",
    "Applying Boolean Logic To Combinations": "Boolean logic",
    "Solving Logical Word Puzzles": "Logical puzzles",
    "Following Multi-Step Instructions": "Multi-step instr.",
    "Reading Tables And Extracting Data": "Table reading",
    "Reasoning About Object Locations": "Object location",
    "Applying Inequality Theorems To Minimize Expressio": "Inequality bounds",
    "Applying Combinatorial Principles To Dice Rolls": "Dice combinatorics",
    "Applying Divisibility Rules To Integer Sets": "Divisibility rules",
    "Applying Set Theory To Count Unique Items": "Set-theory counts",
    "Calculating Ratios Of Segments In Triangles": "Triangle ratios",
    "Analyzing Geometric Interpretations Of Matrix Oper": "Matrix geometry",
    "Analyzing Coordinates To Determine Shape Type": "Coordinate shapes",
    "Maintaining Consistent Tone In Creative Writing": "Tone consistency",
    "Adapting Language For Age Appropriate Audience": "Age-appropriate lang.",
    "Maintaining Word Count Constraints In Responses": "Word-count limits",
    "Articulating Feature Vs Bug Distinction": "Feature vs. bug",
    "Adapting Tone For Dialogue In Urdu Language": "Urdu dialogue tone",
    "Crafting Concise Descriptions Of Professional Acti": "Concise descriptions",
    "Inferring Time Intervals From Sequential Events": "Time-interval reason.",
    "Evaluating Sporting Event Plausibility Based On Co": "Sport plausibility",
    "Identifying Themes In Science Fiction Movies": "Sci-fi themes",
    "Evaluating Conditions For Automated Shutdown": "Shutdown conditions",
    "Identifying Penguin Attributes From Tabular Data": "Penguin tables",
    "Inferring Object Locations From Contextual Clues": "Object locations",
    "Applying Linear Combination Of Operators In Quantu": "Quantum operators",
    "Applying Wetting Theory To Liquid Droplets": "Wetting theory",
    "Interpreting Numerical Values In Context": "Numerical context",
    "Calculating Orbital Period Ratio Between Planets": "Orbital periods",
    "Applying Relativity To Calculate Velocity Vectors": "Relativistic velocity",
    "Applying Nernst Equation To Redox Potential": "Nernst equation",
    "Identifying Interstellar Medium Types In Astronomy": "Interstellar medium",
    "Identifying Transformations In Quantum Field Theor": "QFT transformations",
    "Applying Maxwell Equations To Electromagnetic Fiel": "Maxwell equations",
    "Applying Mutation Rate To Genetic Equilibrium": "Mutation rates",
    "Calculating Eigenvalues Of Quantum Operators": "Quantum eigenvalues",
    "Analyzing User Access Patterns Over Time": "User-access patterns",
    "Identifying Humorous Wordplay In Artist Names": "Humor in artist names",
    "Identifying Penguin Attributes From Tabular Data": "Penguin attributes",
    "Evaluating Possible Storage Options Based On Story": "Storage options",
    "Coordinating Workflow Between Service Providers": "Service workflow",
    "Balancing Instrumentation For Optimal Sound Qualit": "Instrument balance",
    "Calculating Compound Interest With Quarterly Compo": "Compound interest",
    "Applying Interpolation To Estimate Values": "Interpolation",
    "Avoiding Sensitive Terms In Condolence Messages": "Sensitive-term avoidance",
}


def short(name, n=24):
    if name in SHORT:
        return SHORT[name]
    s = name
    for p in ("Applying ", "Identifying ", "Calculating ", "Solving ",
              "Working with ", "Recognizing ", "Reasoning About ",
              "Following ", "Reading ", "Inferring ", "Evaluating ",
              "Maintaining ", "Adapting ", "Articulating ", "Crafting ",
              "Analyzing "):
        if s.startswith(p):
            s = s[len(p):]
            break
    return s if len(s) <= n else s[:n - 1] + "..."


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint
    from cdmeval.utils.visualization import setup_style

    seed_everything(42)
    setup_style()

    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        cluster_labels = json.load(f)
    n_llms, _ = R.shape
    K = q_matrix.shape[1]
    skill_names = [cluster_labels[str(i)] for i in range(K)]

    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"),
        net, device,
    )
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        raw = net.student_emb(torch.arange(n_llms, device=device)).cpu().numpy()
    mastery = 1.0 / (1.0 + np.exp(-raw))

    families = []
    for n in llm_names:
        f, _ = parse_family_and_size(n)
        families.append(f)
    families = np.array(families)

    llama_mask = families == "Llama"
    qwen_mask = families == "Qwen"
    n_llama = int(llama_mask.sum())
    n_qwen = int(qwen_mask.sum())

    llama_mean = mastery[llama_mask].mean(axis=0)
    qwen_mean = mastery[qwen_mask].mean(axis=0)
    gap = qwen_mean - llama_mean  # +ve = Qwen leads

    # Skill -> primary benchmark
    skill_bm = {}
    for si in range(K):
        idx = np.where(q_matrix[:, si] > 0)[0]
        if len(idx) == 0:
            skill_bm[si] = "?"
            continue
        bc = Counter()
        for it in idx[:200]:
            bc[items_data[int(it)].get("benchmark", "?")] += 1
        skill_bm[si] = bc.most_common(1)[0][0]

    def topk_in(pool, k, sign):
        return sorted(pool, key=lambda s: sign * gap[s])[:k]

    math_pool = [si for si in range(K) if skill_bm[si] == "MATH"]
    gpqa_pool = [si for si in range(K) if skill_bm[si] == "GPQA"]
    ifeval_pool = [si for si in range(K) if skill_bm[si] == "IFEval"]
    bbh_pool = [si for si in range(K) if skill_bm[si] == "BBH"]

    qwen_top = topk_in(math_pool, 2, -1) + topk_in(gpqa_pool, 2, -1)
    llama_top = topk_in(ifeval_pool, 2, +1) + topk_in(bbh_pool, 2, +1)

    selected = qwen_top + llama_top
    selected_sorted = sorted(selected, key=lambda s: -gap[s])
    skill_short = [short(skill_names[s]) for s in selected_sorted]

    # Per Part J typography (regular weight default; only panel letters bold)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.weight": "regular",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "axes.labelweight": "regular",
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 9.0,
        "pdf.fonttype": 42,
    })

    # Tighter layout: smaller height, reduced wspace
    fig = plt.figure(figsize=(10.6, 4.0))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.22
    )
    ax_l = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1], sharey=ax_l)
    ax_g = fig.add_subplot(gs[0, 2], sharey=ax_l)

    y_pos = np.arange(len(selected_sorted))

    # ---- panels (a) and (b): mean mastery bars per family ----
    for ax, fam_color, fam_mask, fam_label, panel_letter, n_panel in [
        (ax_l, LLAMA_C, llama_mask, "Llama", "a", n_llama),
        (ax_q, QWEN_C, qwen_mask, "Qwen", "b", n_qwen),
    ]:
        fam_m = mastery[fam_mask].mean(axis=0)[selected_sorted]
        ax.barh(
            y_pos, fam_m, color=fam_color, edgecolor="white",
            linewidth=0.5, height=0.7, alpha=0.92,
        )
        for yi, val in zip(y_pos, fam_m):
            ax.text(val + 0.012, yi, f"{val:.2f}", fontsize=8.0,
                    color=AXIS_C, va="center", ha="left")
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Mean mastery", fontsize=9.5)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.15, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # Panel letter (bold) + family label (regular)
        ax.text(
            -0.02, 1.08, f"({panel_letter})",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=AXIS_C, va="bottom", ha="left",
        )
        ax.text(
            0.13, 1.08, f"{fam_label}  ({n_panel} models)",
            transform=ax.transAxes, fontsize=10.0, fontweight="regular",
            color=fam_color, va="bottom", ha="left",
        )

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(skill_short, fontsize=9.0)
    plt.setp(ax_q.get_yticklabels(), visible=False)
    ax_q.tick_params(axis="y", length=0)

    # ---- panel (c): signed gap, color by dominant family ----
    gap_vals = gap[selected_sorted]
    bar_colors = [QWEN_C if v > 0 else LLAMA_C for v in gap_vals]
    ax_g.barh(
        y_pos, gap_vals, color=bar_colors, edgecolor="white",
        linewidth=0.5, height=0.7, alpha=0.92,
    )
    for yi, si, val in zip(y_pos, selected_sorted, gap_vals):
        bm = skill_bm[si]
        if val >= 0:
            ax_g.text(val + 0.005, yi, f"+{val:.2f}", fontsize=8.0,
                      color=AXIS_C, va="center", ha="left")
        else:
            ax_g.text(val - 0.005, yi, f"{val:.2f}", fontsize=8.0,
                      color=AXIS_C, va="center", ha="right")
        # Benchmark tag at far right with categorical-5 color
        ax_g.text(
            1.02, yi, bm,
            transform=ax_g.get_yaxis_transform(),
            fontsize=8.0, color=BENCH_COLOR.get(bm, AXIS_C),
            va="center", ha="left", family="monospace",
        )

    ax_g.axvline(0, color=AXIS_C, lw=0.7, zorder=3)
    abs_max = float(np.abs(gap_vals).max())
    ax_g.set_xlim(-abs_max - 0.06, abs_max + 0.06)
    ax_g.set_xlabel("Mastery gap  (Qwen $-$ Llama)", fontsize=9.5)
    ax_g.invert_yaxis()
    ax_g.grid(True, axis="x", alpha=0.15, lw=0.5)
    ax_g.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_g.spines[sp].set_visible(False)
    ax_g.tick_params(axis="y", length=0)
    plt.setp(ax_g.get_yticklabels(), visible=False)
    # Panel letter (bold) + sub-title (regular)
    ax_g.text(
        -0.02, 1.08, "(c)",
        transform=ax_g.transAxes, fontsize=10.5, fontweight="bold",
        color=AXIS_C, va="bottom", ha="left",
    )
    ax_g.text(
        0.13, 1.08, "Cross-family gap",
        transform=ax_g.transAxes, fontsize=10.0, fontweight="regular",
        color=AXIS_C, va="bottom", ha="left",
    )
    # Direction-of-advantage labels (regular weight, family-tinted)
    ax_g.text(
        0.02, 1.005, "Llama leads",
        transform=ax_g.transAxes, fontsize=8.5, color=LLAMA_C,
        fontweight="regular", va="bottom", ha="left",
    )
    ax_g.text(
        0.98, 1.005, "Qwen leads",
        transform=ax_g.transAxes, fontsize=8.5, color=QWEN_C,
        fontweight="regular", va="bottom", ha="right",
    )

    # NO suptitle - finding lives in LaTeX caption
    plt.tight_layout(pad=0.4)

    out_pdf = Path("cdm_exploration/figures/review_batch2/fig_model_family_profiles_REVISED.pdf")
    out_png = Path("cdm_exploration/figures/review_batch2/fig_model_family_profiles_REVISED.png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, bbox_inches="tight", dpi=200, pad_inches=0.04)
    print(f"Saved: {out_pdf}", flush=True)
    print(f"Saved: {out_png}", flush=True)

    print("\n--- selection trace ---")
    for si in selected_sorted:
        print(
            f"  {skill_bm[si]:<6} gap={gap[si]:+.3f}  "
            f"L={llama_mean[si]:.3f} Q={qwen_mean[si]:.3f}  "
            f"{skill_names[si]}"
        )


if __name__ == "__main__":
    main()
