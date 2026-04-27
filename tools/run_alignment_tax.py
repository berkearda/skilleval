"""Per-skill alignment tax: base vs instruct/chat model comparison.

Compares theta profiles of base models vs their instruction-tuned
versions to discover which skills alignment improves and which it hurts.

Usage:
    python tools/run_alignment_tax.py device=cpu
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from scipy.stats import wilcoxon

ALPHA = 0.05
MIN_RELIABLE_N = 10  # min pairs per family to call results reliable


def _wilcoxon_p(x):
    """Two-sided Wilcoxon signed-rank p-value, with guards for edge cases."""
    x = np.asarray(x, dtype=float)
    if np.all(x == 0) or len(x) < 2:
        return 1.0
    try:
        _, p = wilcoxon(x, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def find_pairs(names):
    """Find base-instruct pairs by stripping alignment suffixes.

    Excludes candidate bases whose names contain alignment markers (SFT,
    DPO, ORPO, etc.) since those are already-aligned checkpoints rather
    than true base models.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}

    def norm(n):
        return n.replace("__", "/").lower()

    # Markers in a candidate 'base' name that indicate it is already aligned.
    # See the project log 2026-04-19 R8 audit for rationale.
    aligned_markers = [
        "-sft", "_sft", "-dpo", "_dpo", "-mdpo", "-orpo",
        "-ipo", "-simpo", "-kto", "-rlaif", "-rlhf", "_rlhf",
        "-tuned", "-aligned",
    ]

    def is_already_aligned(nl):
        return any(m in nl for m in aligned_markers)

    suffixes = ["-instruct", "_instruct", "-chat", "-it", ".instruct", "-rlhf"]
    pairs = []
    seen = set()

    for idx, name in enumerate(names):
        nl = norm(name)
        for suf in suffixes:
            if suf in nl:
                pos = nl.rfind(suf)
                base_nl = nl[:pos] + nl[pos + len(suf):]
                for idx2, name2 in enumerate(names):
                    if idx2 == idx:
                        continue
                    nl2 = norm(name2)
                    if nl2 == base_nl or nl2.replace("-", "") == base_nl.replace("-", ""):
                        if is_already_aligned(nl2):
                            continue
                        key = tuple(sorted([idx, idx2]))
                        if key not in seen:
                            seen.add(key)
                            pairs.append({"base_idx": idx2, "inst_idx": idx,
                                          "base_name": name2, "inst_name": name})
    return pairs


def detect_family(name):
    """Heuristic family detection from model name."""
    nl = name.lower()
    for fam in ["llama", "qwen", "mistral", "gemma", "phi", "falcon",
                 "yi", "solar", "olmo", "stablelm", "granite"]:
        if fam in nl:
            return fam
    return "other"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment
    from cdmeval.utils.visualization import setup_style
    from cdmeval.validation import validate_data

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load data ──
    print("\nLoading data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        skill_labels = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)
    skill_names = [skill_labels[str(i)] for i in range(K)]

    # ── Load theta ──
    print("\nLoading trained theta...", flush=True)
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, "cpu",
    )
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight).numpy()  # (n_llms, K)

    # ════════════════════════════════════════════════════════════════
    # 1. FIND BASE-INSTRUCT PAIRS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("1. FINDING BASE-INSTRUCT PAIRS", flush=True)
    print(f"{'='*60}", flush=True)
    pairs = find_pairs(llm_names)
    print(f"  Found {len(pairs)} pairs", flush=True)

    if len(pairs) < 10:
        print("  WARNING: fewer than 10 pairs — results may be unreliable", flush=True)

    # Annotate families
    for p in pairs:
        p["family"] = detect_family(p["base_name"])

    family_counts = defaultdict(int)
    for p in pairs:
        family_counts[p["family"]] += 1
    print(f"  Pairs per family:", flush=True)
    for fam, c in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"    {fam:<10}: {c}", flush=True)

    # Sanity: instruct models generally have higher overall accuracy
    acc_base = np.array([R[p["base_idx"]].mean() for p in pairs])
    acc_inst = np.array([R[p["inst_idx"]].mean() for p in pairs])
    n_inst_better = (acc_inst > acc_base).sum()
    print(f"\n  Instruct accuracy > base: {n_inst_better}/{len(pairs)} pairs "
          f"({100*n_inst_better/len(pairs):.0f}%)", flush=True)
    print(f"  Mean accuracy: base={acc_base.mean():.3f}, instruct={acc_inst.mean():.3f}",
          flush=True)

    # ════════════════════════════════════════════════════════════════
    # 2. PER-SKILL ALIGNMENT DELTA
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("2. PER-SKILL ALIGNMENT TAX", flush=True)
    print(f"{'='*60}", flush=True)

    deltas = np.zeros((len(pairs), K))
    for i, p in enumerate(pairs):
        deltas[i] = theta[p["inst_idx"]] - theta[p["base_idx"]]

    mean_delta = deltas.mean(axis=0)  # (K,)
    std_delta = deltas.std(axis=0)
    n_improved = (deltas > 0).sum(axis=0)   # per skill, how many pairs improved
    n_degraded = (deltas < 0).sum(axis=0)

    # Overall stats
    n_skills_improved = (mean_delta > 0.01).sum()
    n_skills_degraded = (mean_delta < -0.01).sum()
    n_skills_neutral = K - n_skills_improved - n_skills_degraded
    print(f"  Skills improved (mean delta > 0.01): {n_skills_improved}", flush=True)
    print(f"  Skills degraded (mean delta < -0.01): {n_skills_degraded}", flush=True)
    print(f"  Skills neutral: {n_skills_neutral}", flush=True)
    print(f"  Overall mean delta: {mean_delta.mean():.4f}", flush=True)

    # Top improved
    sorted_idx = np.argsort(-mean_delta)
    print(f"\n  Top 10 IMPROVED skills (alignment helps):", flush=True)
    for rank, k in enumerate(sorted_idx[:10]):
        print(f"    +{mean_delta[k]:.4f} ({n_improved[k]}/{len(pairs)} pairs): "
              f"{skill_names[k][:55]}", flush=True)

    # Top degraded
    print(f"\n  Top 10 DEGRADED skills (alignment tax):", flush=True)
    for rank, k in enumerate(sorted_idx[-10:][::-1]):
        print(f"    {mean_delta[k]:+.4f} ({n_degraded[k]}/{len(pairs)} pairs): "
              f"{skill_names[k][:55]}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 3. PER-BENCHMARK ALIGNMENT TAX
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("3. PER-BENCHMARK ALIGNMENT TAX", flush=True)
    print(f"{'='*60}", flush=True)

    benchmarks = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
    bench_skills = {}
    for b in benchmarks:
        b_items = [i for i, it in enumerate(items_data) if it.get("benchmark") == b]
        Q_b = q_matrix[b_items]
        skills = set(np.where(Q_b.sum(axis=0) > 0)[0])
        bench_skills[b] = skills

    bench_delta = {}
    for b in benchmarks:
        skills = list(bench_skills[b])
        d = mean_delta[skills].mean()
        bench_delta[b] = float(d)
        n_up = (mean_delta[skills] > 0.01).sum()
        n_down = (mean_delta[skills] < -0.01).sum()
        print(f"  {b:<8}: mean delta={d:+.4f}, skills improved={n_up}, degraded={n_down}",
              flush=True)

    # ════════════════════════════════════════════════════════════════
    # 4. PER-FAMILY ANALYSIS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("4. PER-FAMILY ALIGNMENT TAX", flush=True)
    print(f"{'='*60}", flush=True)

    family_deltas = defaultdict(list)
    for i, p in enumerate(pairs):
        family_deltas[p["family"]].append(deltas[i])

    per_family = {}
    for fam in sorted(family_deltas.keys()):
        fd = np.array(family_deltas[fam])
        fm = fd.mean(axis=0)
        n_up = (fm > 0.01).sum()
        n_down = (fm < -0.01).sum()
        per_family[fam] = {
            "n_pairs": len(family_deltas[fam]),
            "mean_delta": float(fm.mean()),
            "skills_improved": int(n_up),
            "skills_degraded": int(n_down),
            "mean_delta_per_skill": fm.tolist(),
        }
        print(f"  {fam:<10} ({len(family_deltas[fam]):>2} pairs): "
              f"mean={fm.mean():+.4f}, up={n_up}, down={n_down}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 5. CONNECT TO PREREQUISITE DAG
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("5. ALIGNMENT TAX BY PREREQUISITE DEPTH", flush=True)
    print(f"{'='*60}", flush=True)

    prereq_path = Path("cdm_exploration/experiments/v2_skill_prerequisites.json")
    depth_delta = {}
    if prereq_path.exists():
        prereq = json.load(open(prereq_path))
        depth_map = prereq.get("depth_per_skill", {})
        depth_groups = defaultdict(list)
        for k in range(K):
            d = depth_map.get(str(k), 0)
            depth_groups[d].append(k)

        for d in sorted(depth_groups.keys()):
            skills = depth_groups[d]
            dm = mean_delta[skills].mean()
            depth_delta[d] = float(dm)
            print(f"  Depth {d} ({len(skills):>2} skills): mean delta = {dm:+.4f}",
                  flush=True)
    else:
        print("  Prerequisite data not found, skipping", flush=True)

    # ════════════════════════════════════════════════════════════════
    # FIGURES
    # ════════════════════════════════════════════════════════════════
    print("\nGenerating figures...", flush=True)
    setup_style()

    # ── Fig 1: Alignment-tax concentration (2-panel) ──
    # Panel (a): per-benchmark mean delta across pairs, error bars = SEM.
    # Panel (b): top improved + top degraded skills, colored by primary
    #            benchmark, Bonferroni-significant bars at full saturation.

    # Reverse-engineered full names for the 33 cluster labels truncated to 50
    # chars by the upstream clustering pipeline. Display-only fix.
    LABEL_FIXES = {
        "Applying Inequality Theorems To Minimize Expressio": "Applying Inequality Theorems to Minimize Expressions",
        "Applying Linear Combination Of Operators In Quantu": "Applying Linear Combinations of Operators in Quantum Mechanics",
        "Evaluating Sporting Event Plausibility Based On Co": "Evaluating Sporting Event Plausibility Based on Context",
        "Applying Trigonometric Identities To Solve Equatio": "Applying Trigonometric Identities to Solve Equations",
        "Analyzing Geometric Interpretations Of Matrix Oper": "Analyzing Geometric Interpretations of Matrix Operations",
        "Applying Properties Of Rectangles To Find Dimensio": "Applying Properties of Rectangles to Find Dimensions",
        "Identifying Structural Isomers In Organic Compound": "Identifying Structural Isomers in Organic Compounds",
        "Evaluating Climate Control System Activation Condi": "Evaluating Climate Control System Activation Conditions",
        "Crafting Concise Descriptions Of Professional Acti": "Crafting Concise Descriptions of Professional Activities",
        "Embedding Comments With Security Keywords In Code ": "Embedding Comments with Security Keywords in Code",
        "Articulating Customer Rights Under Consumer Protec": "Articulating Customer Rights Under Consumer Protection",
        "Calculating Compound Interest With Quarterly Compo": "Calculating Compound Interest with Quarterly Compounding",
        "Applying Maxwell Equations To Electromagnetic Fiel": "Applying Maxwell's Equations to Electromagnetic Fields",
        "Applying Synonyms To Replace Restricted Vocabulary": "Applying Synonyms to Replace Restricted Vocabulary",
        "Analyzing Historical Significance Of Chess Program": "Analyzing Historical Significance of Chess Programs",
        "Applying Pattern Recognition To Algorithmic Output": "Applying Pattern Recognition to Algorithmic Outputs",
        "Calculating Future Dates From Anniversary Informat": "Calculating Future Dates from Anniversary Information",
        "Applying Random Variable Concepts To Sleep Pattern": "Applying Random Variable Concepts to Sleep Patterns",
        "Applying Discrete Group Approximations To Gauge Th": "Applying Discrete Group Approximations to Gauge Theory",
        "Calculating Oscillation Frequency Of Quantum Parti": "Calculating Oscillation Frequency of Quantum Particles",
        "Evaluating Possible Storage Options Based On Story": "Evaluating Possible Storage Options Based on Story",
        "Applying Mott Gurney Equation To Device Characteri": "Applying Mott-Gurney Equation to Device Characterization",
        "Applying Properties Of Complex Numbers In Equation": "Applying Properties of Complex Numbers in Equations",
        "Calculating Handshake Combinations In Group Intera": "Calculating Handshake Combinations in Group Interactions",
        "Ensuring Consistent Section Headers In Document St": "Ensuring Consistent Section Headers in Document Structure",
        "Interpreting Luminescence Behavior In Zinc Silicat": "Interpreting Luminescence Behavior in Zinc Silicates",
        "Balancing Instrumentation For Optimal Sound Qualit": "Balancing Instrumentation for Optimal Sound Quality",
        "Identifying Transformations In Quantum Field Theor": "Identifying Transformations in Quantum Field Theory",
    }

    def _clean(n):
        return LABEL_FIXES.get(n, n)

    bench_colors = {
        "MATH":   "#0072B2",
        "BBH":    "#E69F00",
        "GPQA":   "#009E73",
        "MuSR":   "#CC79A7",
        "IFEval": "#56B4E9",
    }
    # Primary benchmark per skill (argmax over benchmark item counts)
    skill_bench_counts = np.zeros((K, len(benchmarks)), dtype=int)
    for i, it in enumerate(items_data):
        b = it.get("benchmark")
        if b in benchmarks:
            bi = benchmarks.index(b)
            skill_bench_counts[:, bi] += q_matrix[i].astype(int)
    primary_bench_idx = skill_bench_counts.argmax(axis=1)
    primary_bench_idx[skill_bench_counts.sum(axis=1) == 0] = -1

    # Per-skill SEM and significance (inline so the figure block is
    # self-contained; cheap because deltas[:, k] is small).
    sem_per_skill = deltas.std(axis=0) / np.sqrt(deltas.shape[0])
    bonf_alpha_fig = ALPHA / K
    sig_bonf_mask = np.zeros(K, dtype=bool)
    for k in range(K):
        sig_bonf_mask[k] = _wilcoxon_p(deltas[:, k]) < bonf_alpha_fig
    n_bonf = int(sig_bonf_mask.sum())

    # Per-benchmark pair-level mean/SEM/p using primary-benchmark skills.
    _per_benchmark_stats = {}
    for bi, b in enumerate(bench_order := ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]):
        sk = np.where(primary_bench_idx == bi)[0]
        if len(sk) == 0:
            _per_benchmark_stats[b] = {"mean": 0.0, "sem": 0.0, "p": 1.0}
            continue
        per_pair = deltas[:, sk].mean(axis=1)
        _per_benchmark_stats[b] = {
            "mean": float(per_pair.mean()),
            "sem": float(per_pair.std() / np.sqrt(len(per_pair))),
            "p": _wilcoxon_p(per_pair),
        }

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.6),
                             gridspec_kw={"width_ratios": [0.95, 2.00]})

    # Panel (a): per-benchmark mean pair-level delta with ±SEM
    ax_a = axes[0]
    means_a = [_per_benchmark_stats[b]["mean"] for b in bench_order]
    sems_a = [_per_benchmark_stats[b]["sem"] for b in bench_order]
    ps_a = [_per_benchmark_stats[b]["p"] for b in bench_order]
    y_pos = np.arange(len(bench_order))
    colors_a = [bench_colors[b] for b in bench_order]
    ax_a.barh(y_pos, means_a, xerr=sems_a, color=colors_a, alpha=0.85,
              edgecolor="#333333", linewidth=0.7, capsize=4,
              error_kw={"elinewidth": 1.0, "capthick": 1.0})
    ax_a.axvline(0, color="black", lw=0.8)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(bench_order, fontsize=10)
    ax_a.set_xlabel(r"Mean $\Delta$ (instruct $-$ base)", fontsize=10.5)
    ax_a.set_title("(a) Per-benchmark delta", fontsize=10.5, loc="left")
    # P-values in a dedicated right column (outside the bars)
    x_right = max(means_a) + max(sems_a) + 0.02
    ax_a.set_xlim(min(means_a) - max(sems_a) - 0.025,
                  x_right + 0.055)
    for i, p in enumerate(ps_a):
        if p < 0.001:
            p_str = f"$p{{<}}10^{{{int(np.floor(np.log10(max(p,1e-300))))}}}$"
        elif p < 0.01:
            p_str = f"$p{{=}}{p:.3f}$"
        else:
            p_str = f"$p{{=}}{p:.2f}$"
        ax_a.text(x_right, i, p_str, ha="left", va="center",
                  fontsize=8.5, color="#444444")
    ax_a.invert_yaxis()
    for sp in ["top", "right"]:
        ax_a.spines[sp].set_visible(False)
    ax_a.grid(axis="x", ls=":", lw=0.5, alpha=0.35)

    # Panel (b): top N improved + top N degraded skills
    ax_b = axes[1]
    n_show = 10
    top_up = sorted_idx[:n_show]
    top_down = sorted_idx[-n_show:][::-1]
    # Add a visual gap between degraded (top) and improved (bottom) rows
    show_idx = np.concatenate([top_down, top_up])
    show_delta = mean_delta[show_idx]
    show_sem = sem_per_skill[show_idx]
    show_names = [_clean(skill_names[k]) for k in show_idx]
    show_sig = sig_bonf_mask[show_idx]

    bar_colors = []
    for k in show_idx:
        pb = primary_bench_idx[k]
        bar_colors.append(bench_colors[benchmarks[pb]] if pb >= 0 else "#888888")

    y_b = np.arange(len(show_idx))
    for i, (d, s, c, sig) in enumerate(zip(show_delta, show_sem,
                                             bar_colors, show_sig)):
        if sig:
            ax_b.barh(i, d, xerr=s, color=c, alpha=0.95,
                      edgecolor="#111111", linewidth=1.1, capsize=3,
                      error_kw={"elinewidth": 0.9, "capthick": 0.9,
                                "ecolor": "#333333"})
        else:
            # Muted fill + hatching for non-significant bars so they remain
            # visible at small magnitudes while clearly distinct from sig bars.
            ax_b.barh(i, d, xerr=s, facecolor=c, alpha=0.20,
                      edgecolor=c, linewidth=1.4, hatch="////",
                      capsize=3,
                      error_kw={"elinewidth": 0.7, "capthick": 0.7,
                                "ecolor": "#888888"})
    # Subtle divider between degraded (rows 0..n_show-1) and improved rows
    ax_b.axhline(n_show - 0.5, color="#cccccc", lw=0.6, ls="-", zorder=1)
    ax_b.axvline(0, color="black", lw=0.8)
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(show_names, fontsize=8.5)
    ax_b.set_xlabel(r"Mean $\Delta$ (instruct $-$ base)", fontsize=10.5)
    ax_b.set_title(
        f"(b) Top-{n_show} improved and degraded skills "
        f"($n_{{\\mathrm{{Bonf-sig}}}}{{=}}{n_bonf}/100$ across all skills)",
        fontsize=10.5, loc="left")
    ax_b.invert_yaxis()
    for sp in ["top", "right"]:
        ax_b.spines[sp].set_visible(False)
    ax_b.grid(axis="x", ls=":", lw=0.5, alpha=0.35)

    # Shared benchmark-color legend BELOW panel b, outside data area
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=bench_colors[b],
                      edgecolor="#222222", linewidth=0.8, label=b)
        for b in bench_order
    ]
    # A small style legend for sig/non-sig
    sig_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4C72B0",
                      edgecolor="#111111", linewidth=1.1,
                      label="Bonferroni-significant"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#4C72B0", alpha=0.20,
                      edgecolor="#4C72B0", linewidth=1.4, hatch="////",
                      label="Not significant"),
    ]
    first_leg = ax_b.legend(handles=legend_handles,
                            loc="upper center", bbox_to_anchor=(0.5, -0.11),
                            fontsize=8.5, frameon=False, ncol=5,
                            handletextpad=0.4, columnspacing=1.0,
                            title="Primary benchmark", title_fontsize=8.5)
    ax_b.add_artist(first_leg)
    ax_b.legend(handles=sig_handles,
                loc="upper center", bbox_to_anchor=(0.5, -0.18),
                fontsize=8.5, frameon=False, ncol=2,
                handletextpad=0.4, columnspacing=1.0)

    plt.tight_layout()
    out1 = fig_dir / "fig_alignment_tax_skills.pdf"
    out1_png = fig_dir / "fig_alignment_tax_skills.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    fig.savefig(out1_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1} and {out1_png}", flush=True)

    # ── Fig 2: Family heatmap ──
    fam_order = [f for f in sorted(per_family.keys()) if per_family[f]["n_pairs"] >= 3]
    if fam_order:
        heat = np.zeros((len(fam_order), K))
        for fi, fam in enumerate(fam_order):
            heat[fi] = np.array(per_family[fam]["mean_delta_per_skill"])

        # Sort skills by overall delta for visual clarity
        skill_order = np.argsort(mean_delta)
        heat = heat[:, skill_order]

        fig, ax = plt.subplots(figsize=(14, max(3, len(fam_order) * 0.8)))
        vmax = max(abs(heat.min()), abs(heat.max()), 0.05)
        im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(len(fam_order)))
        ax.set_yticklabels([f"{f} ({per_family[f]['n_pairs']})" for f in fam_order],
                           fontsize=9)
        ax.set_xlabel("Skills (sorted by mean delta)", fontsize=11)
        plt.colorbar(im, ax=ax, label="Mastery delta (instruct - base)", shrink=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough families", ha="center", va="center")
    out2 = fig_dir / "fig_alignment_tax_families.pdf"
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}", flush=True)

    # ── Fig 3: Depth vs delta scatter ──
    fig, ax = plt.subplots(figsize=(7, 5))
    if depth_delta:
        depths = [depth_map.get(str(k), 0) for k in range(K)]
        ax.scatter(depths, mean_delta, s=30, alpha=0.5, color="#4C72B0")
        # Mean per depth
        for d in sorted(depth_delta.keys()):
            ax.scatter(d, depth_delta[d], s=150, color="#C44E52", zorder=5,
                       edgecolors="white", linewidths=1.5,
                       label=f"Depth {d} mean" if d == 0 else "")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Skill depth in prerequisite DAG", fontsize=12)
        ax.set_ylabel("Mean alignment delta", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No prerequisite data", ha="center", va="center")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out3 = fig_dir / "fig_alignment_tax_depth.pdf"
    fig.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # 6. STATISTICAL TESTS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("6. WILCOXON SIGNED-RANK TESTS", flush=True)
    print(f"{'='*60}", flush=True)

    bonf_alpha = ALPHA / K
    per_skill_tests = []
    n_sig_raw = 0
    n_sig_bonf = 0
    for k in range(K):
        d = deltas[:, k]
        p_val = _wilcoxon_p(d)
        sem_k = float(d.std() / np.sqrt(len(d))) if len(d) > 1 else 0.0
        sig_raw = bool(p_val < ALPHA)
        sig_bonf = bool(p_val < bonf_alpha)
        n_sig_raw += int(sig_raw)
        n_sig_bonf += int(sig_bonf)
        per_skill_tests.append({
            "skill": int(k),
            "name": skill_names[k],
            "mean_delta": float(mean_delta[k]),
            "sem": sem_k,
            "p_value": p_val,
            "sig_raw": sig_raw,
            "sig_bonferroni": sig_bonf,
        })
    print(f"  Per-skill (n={len(pairs)}): raw-significant (p<{ALPHA})={n_sig_raw}/{K}, "
          f"Bonferroni (p<{bonf_alpha:.4g})={n_sig_bonf}/{K}", flush=True)

    # Primary-benchmark skill assignment: each skill is attributed to the
    # benchmark whose items use it most (argmax of per-benchmark Q-column sum).
    # This avoids double-counting shared skills across benchmarks.
    skill_bench_counts = np.zeros((K, len(benchmarks)), dtype=int)
    for i, it in enumerate(items_data):
        b = it.get("benchmark")
        if b in benchmarks:
            bi = benchmarks.index(b)
            skill_bench_counts[:, bi] += q_matrix[i].astype(int)
    primary_bench = skill_bench_counts.argmax(axis=1)

    per_benchmark_sem = {}
    for bi, b in enumerate(benchmarks):
        skill_ids = np.where(primary_bench == bi)[0]
        if len(skill_ids) == 0:
            continue
        per_pair = deltas[:, skill_ids].mean(axis=1)
        p_val = _wilcoxon_p(per_pair)
        sem_b = float(per_pair.std() / np.sqrt(len(per_pair)))
        per_benchmark_sem[b] = {
            "mean": float(per_pair.mean()),
            "sem": sem_b,
            "p": p_val,
        }
        print(f"  {b:<8} (n_skills_primary={len(skill_ids)}): "
              f"mean={per_pair.mean():+.4f}  sem={sem_b:.4f}  p={p_val:.4g}",
              flush=True)

    family_reliability = {
        fam: {"n": int(n_fam), "reliable": bool(n_fam >= MIN_RELIABLE_N)}
        for fam, n_fam in family_counts.items()
    }

    # ── Save JSON ──
    save_data = {
        "experiment": "alignment_tax",
        "n_llms": int(n_llms), "K": int(K),
        "n_pairs": len(pairs),
        "pairs": [{"base": p["base_name"], "instruct": p["inst_name"],
                    "family": p["family"]} for p in pairs],
        "instruct_better_pct": float(100 * n_inst_better / len(pairs)),
        "mean_acc_base": float(acc_base.mean()),
        "mean_acc_instruct": float(acc_inst.mean()),
        "overall_mean_delta": float(mean_delta.mean()),
        "skills_improved": int(n_skills_improved),
        "skills_degraded": int(n_skills_degraded),
        "skills_neutral": int(n_skills_neutral),
        "per_skill": {str(k): {"name": skill_names[k],
                                 "mean_delta": float(mean_delta[k]),
                                 "std_delta": float(std_delta[k]),
                                 "n_improved": int(n_improved[k]),
                                 "n_degraded": int(n_degraded[k])}
                       for k in range(K)},
        "top_improved": [{"skill": skill_names[int(k)],
                           "delta": float(mean_delta[k])}
                          for k in sorted_idx[:15]],
        "top_degraded": [{"skill": skill_names[int(k)],
                           "delta": float(mean_delta[k])}
                          for k in sorted_idx[-15:][::-1]],
        "per_benchmark": bench_delta,
        "per_family": {f: {"n_pairs": v["n_pairs"],
                            "mean_delta": v["mean_delta"],
                            "skills_improved": v["skills_improved"],
                            "skills_degraded": v["skills_degraded"]}
                        for f, v in per_family.items()},
        "per_depth": {str(k): v for k, v in depth_delta.items()},
        "statistical_tests": {
            "n_pairs": len(pairs),
            "n_significant_raw_005": n_sig_raw,
            "n_significant_bonferroni": n_sig_bonf,
            "per_skill": per_skill_tests,
            "per_benchmark_sem": per_benchmark_sem,
            "family_reliability": family_reliability,
        },
    }

    out_json = Path("cdm_exploration/experiments/v2_alignment_tax.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}", flush=True)

    log_experiment(
        name="alignment_tax",
        config={"K": K, "n_pairs": len(pairs), "device": device},
        results={
            "n_pairs": len(pairs),
            "mean_delta": float(mean_delta.mean()),
            "skills_improved": int(n_skills_improved),
            "skills_degraded": int(n_skills_degraded),
            "per_benchmark": bench_delta,
            "per_benchmark_sem": per_benchmark_sem,
            "n_significant_bonferroni": save_data["statistical_tests"].get(
                "n_significant_bonferroni"),
            "family_reliability": save_data["statistical_tests"].get(
                "family_reliability"),
        },
        split_info={"n_llms": n_llms, "K": K},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
