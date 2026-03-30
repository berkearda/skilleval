"""Interpretability analyses for v2 dataset — publication-quality figures.

1. Model Family Profiling (Llama vs Qwen by scale)
2. Radar: small vs medium vs large Qwen
3. Weak-Beats-Strong bar chart
4. UMAP of skill profiles

Usage:
    python tools/interpretability_v2.py device=cpu
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# ── Short readable skill labels ──────────────────────────────────
SHORT_SKILL_MAP = {
    "Inferring Time Intervals From Sequential Events": "Time intervals",
    "Applying Inequality Theorems To Minimize Expressio": "Inequality theorems",
    "Analyzing Coordinates To Determine Shape Type": "Coordinate geometry",
    "Applying Linear Combination Of Operators In Quantu": "Quantum operators",
    "Applying Wetting Theory To Liquid Droplets": "Wetting theory",
    "Evaluating Sporting Event Plausibility Based On Co": "Sports plausibility",
    "Applying Quadratic Formula To Find Roots": "Quadratic formula",
    "Identifying Themes In Science Fiction Movies": "Sci-fi themes",
    "Maintaining Consistent Tone In Creative Writing": "Tone consistency",
    "Applying Combinatorial Principles To Dice Rolls": "Combinatorics (dice)",
    "Applying Set Theory To Count Unique Items": "Set theory counting",
    "Applying Trigonometric Identities To Solve Equatio": "Trigonometric identities",
    "Evaluating Consistency In League Punishments": "Rule consistency",
    "Evaluating Conditions For Automated Shutdown": "Conditional evaluation",
    "Calculating Decay Length From Energy And Mass": "Decay length calc.",
    "Interpreting Numerical Values In Context": "Number interpretation",
    "Analyzing Geometric Interpretations Of Matrix Oper": "Matrix geometry",
    "Calculating Ratios Of Segments In Triangles": "Triangle ratios",
    "Identifying Penguin Attributes From Tabular Data": "Table reading",
    "Applying Divisibility Rules To Integer Sets": "Divisibility rules",
    "Inferring Object Locations From Contextual Clues": "Object location",
    "Adapting Language For Age Appropriate Audience": "Age-adapted language",
    "Calculating Work Done In Variable Force Scenarios": "Work-energy calc.",
    "Maintaining Word Count Constraints In Responses": "Word count control",
    "Analyzing Effects Of Vertical Stretch On Graphs": "Graph transformations",
    "Calculating Orbital Period Ratio Between Planets": "Orbital mechanics",
    "Applying Properties Of Rectangles To Find Dimensio": "Rectangle properties",
    "Manipulating Algebraic Expressions For Variance": "Algebraic manipulation",
    "Identifying Sarcasm In Verbal Expressions": "Sarcasm detection",
    "Applying Relativity To Calculate Velocity Vectors": "Relativistic velocity",
    "Applying Nernst Equation To Redox Potential": "Nernst equation",
    "Identifying Humorous Wordplay In Artist Names": "Humor/wordplay",
    "Adapting Tone For Dialogue In Urdu Language": "Dialogue tone",
    "Identifying Interstellar Medium Types In Astronomy": "Interstellar medium",
    "Evaluating Function Values At Specific Points": "Function evaluation",
    "Interpreting Directional Commands For Navigation": "Spatial navigation",
    "Assessing Causal Relationships In Fertilizer Use": "Causal reasoning",
    "Applying Recursion To Define Number Sequences": "Recursive sequences",
    "Summing Coefficients Of Polynomial Expression": "Polynomial coefficients",
    "Organizing Information Based On Given Clues": "Clue-based reasoning",
    "Identifying Structural Isomers In Organic Compound": "Organic isomers",
    "Applying Molarity Concepts To Solution Mixing": "Molarity/solutions",
    "Identifying Mutations In Nucleotide Sequences": "Gene mutations",
    "Tracking Object Trades Among Multiple Participants": "Object tracking",
    "Applying De Morgan Laws To Boolean Logic": "Boolean logic (DeMorgan)",
    "Evaluating Truthfulness Of Nested Statements": "Nested truth eval.",
    "Distinguishing Fruits From Non Fruits": "Fruit classification",
    "Calculating Future Dates From Anniversary Informat": "Date calculation",
    "Applying Percentage Rates To Income Segments": "Percentage rates",
    "Applying Properties Of Complex Numbers In Equation": "Complex numbers",
    "Calculating Handshake Combinations In Group Intera": "Handshake combos",
    "Applying Fraction Multiplication To Unit Prices": "Fraction arithmetic",
    "Calculating Eigenvalues Of Quantum Operators": "Eigenvalue problems",
    "Calculating Compound Interest With Quarterly Compo": "Compound interest",
    "Recognizing Symmetry In Two Dimensional Figures": "2D symmetry",
    "Applying Pattern Recognition To Algorithmic Output": "Pattern recognition",
    "Calculating Oscillation Frequency Of Quantum Parti": "Quantum oscillations",
    "Articulating Feature Vs Bug Distinction": "Feature articulation",
    "Identifying Alphabetical Order Of Names In Table": "Alphabetical ordering",
    "Extracting Data From Table Headers": "Data extraction",
    "Applying Snell'S Law To Light Refraction In Media": "Snell's law (optics)",
    "Embedding Asterisk Symbols For Visual Separation": "Asterisk formatting",
    "Applying Synonyms To Replace Restricted Vocabulary": "Synonym replacement",
    "Separating Versions With Specific Delimiters": "Version delimiters",
    "Ensuring Consistent Section Headers In Document St": "Section headers",
    "Embedding Comments With Security Keywords In Code ": "Security keywords",
    "Constructing Bash Scripts For File Downloads": "Bash scripting",
    "Avoiding Sensitive Terms In Condolence Messages": "Sensitive terms",
    "Incorporating Friends Name In Personalized Message": "Name personalization",
    "Crafting Concise Descriptions Of Professional Acti": "Concise descriptions",
    "Analyzing Door Count Distribution In Car Types": "Door count analysis",
    "Analyzing Historical Significance Of Chess Program": "Chess history",
    "Analyzing Surface Properties Of Materials": "Surface properties",
    "Coordinating Workflow Between Service Providers": "Workflow coordination",
    "Optimizing User Experience Based On Feedback": "UX optimization",
    "Analyzing Sentiment In Financial Statements": "Financial sentiment",
    "Balancing Instrumentation For Optimal Sound Qualit": "Audio mixing",
    "Analyzing Fragmentation Patterns In Ei Ms": "Mass spec. fragments",
    "Identifying Transformations In Quantum Field Theor": "QFT transformations",
    "Iterating Weight Loss Over Multiple Weeks": "Weight loss iteration",
    "Interpreting Luminescence Behavior In Zinc Silicat": "Luminescence behavior",
    "Analyzing Historical Context Of Sunni Shia Split": "Sunni-Shia history",
    "Evaluating Climate Control System Activation Condi": "Climate control eval.",
    "Evaluating Possible Storage Options Based On Story": "Storage reasoning",
    "Applying Edington Limit To Black Hole Mass Ratio": "Eddington limit",
    "Analyzing User Access Patterns Over Time": "Access pattern analysis",
    "Applying Mutation Rate To Genetic Equilibrium": "Genetic equilibrium",
    "Applying Albedo Concepts To Temperature Difference": "Albedo/temperature",
    "Applying Discrete Group Approximations To Gauge Th": "Group theory",
    "Applying Mott Gurney Equation To Device Characteri": "Mott-Gurney equation",
    "Applying Random Variable Concepts To Sleep Pattern": "Random variables",
}

def short_skill(name, fallback_len=22):
    """Get short readable skill label."""
    for long, short in SHORT_SKILL_MAP.items():
        if name.startswith(long[:30]):
            return short
    # Fallback: clean up
    name = (name.replace("Applying ", "").replace("Identifying ", "")
            .replace("Calculating ", "").replace("Evaluating ", "")
            .replace("Interpreting ", "").replace("Analyzing ", ""))
    return name[:fallback_len]


def parse_family_and_size(name):
    low = name.lower()
    family_rules = [
        ("llama", "Llama"), ("qwen", "Qwen"), ("mistral", "Mistral"),
        ("gemma", "Gemma"), ("phi-", "Phi"), ("phi2", "Phi"), ("phi3", "Phi"),
        ("yi-", "Yi"), ("deepseek", "DeepSeek"), ("solar", "SOLAR"),
        ("starcoder", "StarCoder"), ("falcon", "Falcon"), ("mpt-", "MPT"),
        ("bloom", "BLOOM"), ("pythia", "Pythia"), ("opt-", "OPT"),
        ("internlm", "InternLM"), ("baichuan", "Baichuan"),
        ("vicuna", "Vicuna"), ("zephyr", "Zephyr"), ("openchat", "OpenChat"),
        ("command", "Cohere"),
    ]
    family = "Other"
    for kw, fam in family_rules:
        if kw in low:
            family = fam
            break
    parts = name.split("__")
    suffix = parts[-1] if len(parts) > 1 else name
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe:
        return family, float(moe.group(1)) * float(moe.group(2))
    for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
        val = float(m.group(1))
        idx = suffix.find(m.group(0))
        if idx > 0 and suffix[idx - 1].lower() == "v":
            continue
        return family, val
    for m in re.finditer(r"(\d+\.?\d*)[mM]", suffix):
        return family, float(m.group(1)) / 1000.0
    return family, None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment
    from cdmeval.utils.visualization import SAVE_KW, setup_style
    import seaborn as sns

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)
    with open(data_dir / "cluster_labels_v2_K100.json") as f:
        cluster_labels = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    skill_names = [cluster_labels[str(i)] for i in range(K)]

    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device); net.eval()

    with torch.no_grad():
        raw = net.student_emb(torch.arange(n_llms, device=device)).cpu().numpy()
    mastery = 1.0 / (1.0 + np.exp(-raw))

    families, sizes = [], []
    for name in llm_names:
        f, s = parse_family_and_size(name)
        families.append(f); sizes.append(s)
    sizes = np.array([s if s is not None else np.nan for s in sizes])
    fam_counts = Counter(families)
    major_fams = [f for f, c in fam_counts.most_common() if c >= 10 and f != "Other"]

    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  {n_llms} LLMs, K={K}, {len(test_items)} test items", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Fig 1: Family profiles (Llama vs Qwen)
    # ══════════════════════════════════════════════════════════════
    print("\nFig 1: Family profiles...", flush=True)
    setup_style()

    # Curate 8 diverse skills: 2 math, 2 science, 2 reasoning, 2 language
    # Pick from high-variance skills but enforce diversity
    skill_var = mastery.var(axis=0)
    # Categorize skills by dominant benchmark
    skill_benchmarks = {}
    for si in range(K):
        skill_items_idx = np.where(q_matrix[:, si] > 0)[0]
        if len(skill_items_idx) == 0:
            skill_benchmarks[si] = "?"
            continue
        bm_counts = Counter()
        for it in skill_items_idx[:100]:
            bm_counts[items_data[int(it)].get("benchmark", "?")] += 1
        skill_benchmarks[si] = bm_counts.most_common(1)[0][0]

    # Select top-variance skills per category, no near-duplicates
    selected = []
    seen_short = set()
    for target_bm, n_pick in [("MATH", 2), ("GPQA", 2), ("BBH", 2), ("IFEval", 1), ("MuSR", 1)]:
        candidates = [(si, skill_var[si]) for si in range(K)
                       if skill_benchmarks.get(si) == target_bm]
        candidates.sort(key=lambda x: -x[1])
        picked = 0
        for si, _ in candidates:
            sname = short_skill(skill_names[si])
            if sname not in seen_short and picked < n_pick:
                selected.append(si)
                seen_short.add(sname)
                picked += 1
    # Fill to 8 if needed
    if len(selected) < 8:
        for si in np.argsort(-skill_var):
            if si not in selected:
                selected.append(si)
            if len(selected) == 8:
                break
    top8 = np.array(selected[:8])

    size_buckets = [(0, 3, "<3B"), (3, 10, "3-10B"), (10, 35, "10-35B"), (35, 200, "35B+")]
    bucket_colors = ["#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A5F"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fam in zip(axes, ["Llama", "Qwen"]):
        fam_mask = np.array([f == fam for f in families])
        fam_sizes = sizes[fam_mask]
        fam_mastery = mastery[fam_mask]

        y_pos = np.arange(len(top8))
        bar_h = 0.18
        for bi, (lo, hi, label) in enumerate(size_buckets):
            bucket_mask = (fam_sizes >= lo) & (fam_sizes < hi)
            if bucket_mask.sum() < 1:
                continue
            mean_m = fam_mastery[bucket_mask].mean(axis=0)
            ax.barh(y_pos + bi * bar_h, mean_m[top8], height=bar_h,
                    color=bucket_colors[bi],
                    label=f"{label} (n={bucket_mask.sum()})",
                    edgecolor="white", linewidth=0.3)

        ax.set_yticks(y_pos + 0.27)
        ax.set_yticklabels([short_skill(skill_names[i]) for i in top8], fontsize=11)
        ax.set_xlabel("Mean mastery", fontsize=13)
        ax.set_title(f"{fam} ({fam_mask.sum()} models)", fontsize=15, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.15, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Per-panel legend — place above the bars to avoid overlap
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.88)
    fig.savefig(fig_dir / "fig_model_family_profiles.pdf", **SAVE_KW)
    plt.close()
    print(f"  Saved fig_model_family_profiles.pdf", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Fig 2: Radar (Qwen small / medium / large)
    # ══════════════════════════════════════════════════════════════
    print("Fig 2: Radar chart...", flush=True)

    qwen_mask = np.array([f == "Qwen" for f in families])
    qwen_sizes_v = sizes[qwen_mask]
    qwen_mastery = mastery[qwen_mask]
    qwen_names_l = [llm_names[i] for i in range(n_llms) if families[i] == "Qwen"]
    valid = ~np.isnan(qwen_sizes_v)

    # Pick 3 sizes: smallest, ~7B, largest
    v_sizes = qwen_sizes_v[valid]
    v_mastery = qwen_mastery[valid]
    v_names = [n for n, vv in zip(qwen_names_l, valid) if vv]
    small_i = np.argmin(v_sizes)
    large_i = np.argmax(v_sizes)
    mid_i = np.argmin(np.abs(v_sizes - 7.0))

    picks = [(small_i, "#DD8452"), (mid_i, "#55A868"), (large_i, "#4C72B0")]
    pick_labels = [
        f"Qwen {v_sizes[small_i]:.1f}B",
        f"Qwen {v_sizes[mid_i]:.0f}B",
        f"Qwen {v_sizes[large_i]:.0f}B",
    ]
    # 8 skills: must be monotonic (small < mid < large), good spread,
    # mid not at floor, and recognizable skill names
    # Skip trivial IFEval formatting skills
    skip_keywords = {"asterisk", "formatting", "delimiter", "header", "keyword",
                     "word count", "placeholder", "wording", "section header"}
    candidates = []
    for si in range(K):
        sname = skill_names[si].lower()
        if any(kw in sname for kw in skip_keywords):
            continue
        s_val = v_mastery[small_i, si]
        m_val = v_mastery[mid_i, si]
        l_val = v_mastery[large_i, si]
        # Require monotonic: small <= mid <= large
        if not (s_val <= m_val + 0.02 and m_val <= l_val + 0.02):
            continue
        # Require mid not at floor
        if m_val < 0.15:
            continue
        # Require good spread
        spread = l_val - s_val
        if spread < 0.15:
            continue
        candidates.append((si, spread, m_val))

    # Sort by spread, pick top 8
    candidates.sort(key=lambda x: -x[1])
    radar_skills = np.array([c[0] for c in candidates[:8]])
    radar_labels = [short_skill(skill_names[i], 22) for i in radar_skills]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False).tolist()
    angles += angles[:1]

    for (idx, color), lbl in zip(picks, pick_labels):
        vals = v_mastery[idx, radar_skills].tolist() + [v_mastery[idx, radar_skills[0]]]
        ax.plot(angles, vals, "o-", color=color, lw=2, markersize=5, label=lbl)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_yticklabels([])
    # Make grid circles very faint so label overlap is not visible
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.4)
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.4)
    ax.spines["polar"].set_color("#e0e0e0")
    ax.spines["polar"].set_linewidth(0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 0.85), frameon=False, fontsize=11)

    plt.tight_layout(pad=3.0)  # extra padding for bottom label
    fig.savefig(fig_dir / "fig_model_radar.pdf", **SAVE_KW)
    plt.close()
    print(f"  Saved fig_model_radar.pdf", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Fig 3: Weak-Beats-Strong
    # ══════════════════════════════════════════════════════════════
    print("Fig 3: Weak-beats-strong...", flush=True)

    train_acc = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    small_mask = sizes <= 13.0
    small_mask[np.isnan(sizes)] = False
    small_indices = np.where(small_mask)[0]

    wbs_items = []
    wbs_by_benchmark = Counter()
    total_by_benchmark = Counter()
    for item_idx in test_items:
        gt = R[:, int(item_idx)]
        bm = items_data[int(item_idx)].get("benchmark", "?")
        total_by_benchmark[bm] += 1
        if gt[strongest_idx] == 0 and gt[small_indices].sum() > 0:
            wbs_items.append(int(item_idx))
            wbs_by_benchmark[bm] += 1
    wbs_pct = len(wbs_items) / len(test_items) * 100

    # Per-skill WBS %
    wbs_per_skill = []
    for si in range(K):
        skill_items = test_items[q_matrix[test_items, si] > 0]
        if len(skill_items) < 10:
            continue
        gt_s = R[strongest_idx, skill_items.astype(int)]
        gt_sm = R[small_indices][:, skill_items.astype(int)].max(axis=0)
        wbs_n = ((gt_s == 0) & (gt_sm > 0)).sum()
        # Determine benchmark for this skill (most common among its items)
        bm_counts = Counter()
        for it in skill_items[:50]:
            bm_counts[items_data[int(it)].get("benchmark", "?")] += 1
        dom_bm = bm_counts.most_common(1)[0][0]
        wbs_per_skill.append({
            "skill": skill_names[si], "wbs_pct": wbs_n / len(skill_items) * 100,
            "n": len(skill_items), "benchmark": dom_bm,
        })
    wbs_per_skill.sort(key=lambda x: -x["wbs_pct"])
    top15 = wbs_per_skill[:15]

    bm_colors = {"MATH": "#3B82F6", "BBH": "#22C55E", "GPQA": "#F97316",
                 "MuSR": "#A855F7", "IFEval": "#EF4444"}

    top12 = wbs_per_skill[:12]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(top12))

    for i, s in enumerate(top12):
        color = bm_colors.get(s["benchmark"], "#999999")
        # Thin line from 0 to dot
        ax.plot([0, s["wbs_pct"]], [i, i], color=color, lw=1.5, solid_capstyle="round")
        # Dot
        ax.scatter(s["wbs_pct"], i, s=60, color=color, edgecolors="white",
                   linewidths=0.8, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([short_skill(s["skill"]) for s in top12], fontsize=11)
    ax.set_xlabel("Items where small model (≤13B) beats strongest (%)", fontsize=13)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, None)
    ax.grid(False)

    # Benchmark legend
    from matplotlib.patches import Patch
    shown_bm = set(s["benchmark"] for s in top12)
    handles = [Patch(facecolor=bm_colors[b], label=b, edgecolor="none")
               for b in ["MATH", "BBH", "GPQA", "MuSR", "IFEval"] if b in shown_bm]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="lower right")

    plt.tight_layout(pad=2.0)
    fig.savefig(fig_dir / "fig_weak_beats_strong.pdf", **SAVE_KW)
    plt.close()
    print(f"  Saved fig_weak_beats_strong.pdf", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Fig 4: UMAP
    # ══════════════════════════════════════════════════════════════
    print("Fig 4: UMAP...", flush=True)
    try:
        import umap
        coords = umap.UMAP(n_components=2, metric="cosine", random_state=42,
                           n_neighbors=30).fit_transform(mastery)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(mastery)
        method = "t-SNE"
    print(f"  {method} done", flush=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    top5 = ["Llama", "Qwen", "Gemma", "Mistral", "Phi"]
    pal = sns.color_palette("tab10", 5)
    fam_col = {f: pal[i] for i, f in enumerate(top5)}

    # Marker size from param count
    ms = np.clip(sizes, 0.5, 100)
    ms = np.where(np.isnan(ms), 7.0, ms)
    ms = np.clip(np.log2(ms + 1) * 8, 5, 80)

    # Skip "Other" entirely (invisible and bloats file size)

    for fam in top5:
        mask = np.array([f == fam for f in families])
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[fam_col[fam]],
                  s=ms[mask], alpha=0.6, edgecolors="white", linewidths=0.2,
                  label=f"{fam} ({mask.sum()})", rasterized=True)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10,
             frameon=False, title="Model Family", title_fontsize=11, markerscale=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    plt.tight_layout(pad=2.0)
    fig.savefig(fig_dir / "fig_umap_profiles.pdf", **SAVE_KW)
    plt.close()
    print(f"  Saved fig_umap_profiles.pdf", flush=True)

    # ── Save results ──
    save_data = {
        "dataset": "v2_full", "n_llms": n_llms, "K": K,
        "major_families": {f: fam_counts[f] for f in major_fams},
        "weak_beats_strong": {
            "n_items": len(wbs_items), "pct": wbs_pct,
            "by_benchmark": dict(wbs_by_benchmark),
        },
        "skills_with_wbs": sum(1 for s in wbs_per_skill if s["wbs_pct"] > 0),
    }
    out = Path("cdm_exploration/experiments/v2_interpretability.json")
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {out}", flush=True)

    from cdmeval.utils.experiment import log_experiment
    log_experiment(
        name="interpretability_v2",
        config={"K": K, "n_llms": n_llms},
        results={"wbs_pct": wbs_pct, "wbs_items": len(wbs_items)},
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
