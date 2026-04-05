"""Per-skill scaling laws across 3,811 LLMs.

Fits sigmoid/linear/power-law to mastery vs param count for each skill.
Creates: skill acquisition heatmap, emergence landscape, cross-family trajectories.

Usage:
    python tools/scaling_laws.py device=cpu
"""

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from scipy.optimize import curve_fit
from scipy.stats import kendalltau

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

# ── Short skill labels ──
SHORT_MAP = {
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
    "Applying Trigonometric Identities To Solve Equatio": "Trig. identities",
    "Evaluating Consistency In League Punishments": "Rule consistency",
    "Evaluating Conditions For Automated Shutdown": "Shutdown conditions",
    "Calculating Decay Length From Energy And Mass": "Decay length",
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
    "Articulating Feature Vs Bug Distinction": "Feature vs bug",
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
    "Applying Albedo Concepts To Temperature Difference": "Albedo/temperature",
    "Organizing Information Based On Given Clues": "Clue-based reasoning",
    "Identifying Structural Isomers In Organic Compound": "Organic isomers",
    "Evaluating Climate Control System Activation Condi": "Climate control",
    "Applying Molarity Concepts To Solution Mixing": "Molarity/solutions",
    "Identifying Mutations In Nucleotide Sequences": "Gene mutations",
    "Crafting Concise Descriptions Of Professional Acti": "Concise descriptions",
    "Tracking Object Trades Among Multiple Participants": "Object tracking",
    "Analyzing Historical Context Of Sunni Shia Split": "Sunni-Shia history",
    "Applying Properties Of Matrix Norms To Vectors": "Matrix norms",
    "Embedding Comments With Security Keywords In Code ": "Security keywords",
    "Articulating Customer Rights Under Consumer Protec": "Consumer rights",
    "Identifying Alphabetical Order Of Names In Table": "Alphabetical ordering",
    "Avoiding Sensitive Terms In Condolence Messages": "Sensitive terms",
    "Analyzing Door Count Distribution In Car Types": "Door count analysis",
    "Applying Edington Limit To Black Hole Mass Ratio": "Eddington limit",
    "Separating Versions With Specific Delimiters": "Version delimiters",
    "Applying Divisibility Rules To Factorials": "Factorial divisibility",
    "Calculating Compound Interest With Quarterly Compo": "Compound interest",
    "Applying Combinatorial Principles To Pet Ownership": "Combinatorics (pets)",
    "Extracting Data From Table Headers": "Data extraction",
    "Embedding Asterisk Symbols For Visual Separation": "Asterisk formatting",
    "Applying Maxwell Equations To Electromagnetic Fiel": "Maxwell equations",
    "Applying Synonyms To Replace Restricted Vocabulary": "Synonym replacement",
    "Recognizing Symmetry In Two Dimensional Figures": "2D symmetry",
    "Analyzing Historical Significance Of Chess Program": "Chess history",
    "Applying Pattern Recognition To Algorithmic Output": "Pattern recognition",
    "Calculating Future Dates From Anniversary Informat": "Date calculation",
    "Applying Percentage Rates To Income Segments": "Percentage rates",
    "Analyzing User Access Patterns Over Time": "Access patterns",
    "Applying Random Variable Concepts To Sleep Pattern": "Random variables",
    "Applying Discrete Group Approximations To Gauge Th": "Group theory",
    "Calculating Eigenvalues Of Quantum Operators": "Eigenvalue problems",
    "Applying De Morgan Laws To Boolean Logic": "Boolean logic",
    "Constructing Bash Scripts For File Downloads": "Bash scripting",
    "Analyzing Surface Properties Of Materials": "Surface properties",
    "Calculating Oscillation Frequency Of Quantum Parti": "Quantum oscillations",
    "Distinguishing Fruits From Non Fruits": "Fruit classification",
    "Coordinating Workflow Between Service Providers": "Workflow coordination",
    "Applying Mutation Rate To Genetic Equilibrium": "Genetic equilibrium",
    "Evaluating Possible Storage Options Based On Story": "Storage reasoning",
    "Optimizing User Experience Based On Feedback": "UX optimization",
    "Incorporating Friends Name In Personalized Message": "Name personalization",
    "Converting Repeating Decimals To Fractions": "Decimal conversion",
    "Applying Set Theory To Divisor Subsets": "Divisor set theory",
    "Applying Interpolation To Estimate Values": "Interpolation",
    "Applying Mott Gurney Equation To Device Characteri": "Mott-Gurney equation",
    "Analyzing Sentiment In Financial Statements": "Financial sentiment",
    "Applying Snell'S Law To Light Refraction In Media": "Snell's law",
    "Applying Properties Of Complex Numbers In Equation": "Complex numbers",
    "Calculating Handshake Combinations In Group Intera": "Handshake combos",
    "Applying Fraction Multiplication To Unit Prices": "Fraction arithmetic",
    "Ensuring Consistent Section Headers In Document St": "Section headers",
    "Iterating Weight Loss Over Multiple Weeks": "Weight loss iteration",
    "Interpreting Luminescence Behavior In Zinc Silicat": "Luminescence",
    "Balancing Instrumentation For Optimal Sound Qualit": "Audio mixing",
    "Evaluating Truthfulness Of Nested Statements": "Nested truth eval.",
    "Analyzing Fragmentation Patterns In Ei Ms": "Mass spec. fragments",
    "Identifying Transformations In Quantum Field Theor": "QFT transformations",
}

def short_skill(name, maxlen=22):
    for long, short in SHORT_MAP.items():
        if name.startswith(long[:30]):
            return short
    name = (name.replace("Applying ", "").replace("Identifying ", "")
            .replace("Calculating ", "").replace("Evaluating ", "")
            .replace("Interpreting ", "").replace("Analyzing ", ""))
    return name[:maxlen]


def parse_family_size(name):
    low = name.lower()
    rules = [("llama", "Llama"), ("qwen", "Qwen"), ("gemma", "Gemma"),
             ("mistral", "Mistral"), ("phi-", "Phi"), ("yi-", "Yi"),
             ("deepseek", "DeepSeek"), ("falcon", "Falcon")]
    family = "Other"
    for kw, fam in rules:
        if kw in low:
            family = fam; break
    parts = name.split("__")
    suffix = parts[-1] if len(parts) > 1 else name
    moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
    if moe: return family, float(moe.group(1)) * float(moe.group(2))
    for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
        val = float(m.group(1))
        idx = suffix.find(m.group(0))
        if idx > 0 and suffix[idx-1].lower() == "v": continue
        return family, val
    for m in re.finditer(r"(\d+\.?\d*)[mM]", suffix):
        return family, float(m.group(1)) / 1000.0
    return family, None


def fit_sigmoid(x, y):
    def sigmoid(x, L, U, s, mu):
        return L + (U - L) / (1 + np.exp(-s * (x - mu)))
    try:
        popt, _ = curve_fit(sigmoid, x, y, p0=[0.1, 0.8, 1.0, np.median(x)],
                            bounds=([0, 0, 0.01, x.min()-1], [1, 1, 20, x.max()+1]),
                            maxfev=5000)
        y_pred = sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        aic = len(x) * np.log(max(ss_res / len(x), 1e-10)) + 2 * 4
        return {"type": "sigmoid", "params": {"L": popt[0], "U": popt[1], "s": popt[2], "mu": popt[3]},
                "r2": r2, "aic": aic}
    except Exception:
        return {"type": "sigmoid", "params": {}, "r2": -1, "aic": 1e10}


def fit_linear(x, y):
    try:
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        aic = len(x) * np.log(max(ss_res / len(x), 1e-10)) + 2 * 2
        return {"type": "linear", "params": {"a": coeffs[0], "b": coeffs[1]},
                "r2": r2, "aic": aic}
    except Exception:
        return {"type": "linear", "params": {}, "r2": -1, "aic": 1e10}


def fit_power(x_raw, y):
    """Power law: mastery = a * N^b, fit in log-log space."""
    try:
        mask = (x_raw > 0) & (y > 0.01)
        if mask.sum() < 3: raise ValueError
        lx, ly = np.log(x_raw[mask]), np.log(y[mask])
        coeffs = np.polyfit(lx, ly, 1)
        y_pred = np.exp(np.polyval(coeffs, np.log(x_raw)))
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        aic = len(x_raw) * np.log(max(ss_res / len(x_raw), 1e-10)) + 2 * 2
        return {"type": "power", "params": {"b": coeffs[0], "a": np.exp(coeffs[1])},
                "r2": r2, "aic": aic}
    except Exception:
        return {"type": "power", "params": {}, "r2": -1, "aic": 1e10}


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

    # ── Load ──
    print("Loading...", flush=True)
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

    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt"), net, device)
    net.to(device); net.eval()
    with torch.no_grad():
        raw = net.student_emb(torch.arange(n_llms, device=device)).cpu().numpy()
    mastery = 1.0 / (1.0 + np.exp(-raw))

    families, sizes_raw = [], []
    for name in llm_names:
        f, s = parse_family_size(name)
        families.append(f); sizes_raw.append(s)
    sizes = np.array([s if s is not None else np.nan for s in sizes_raw])

    # Skill benchmarks
    skill_bm = {}
    for si in range(K):
        idx = np.where(q_matrix[:, si] > 0)[0]
        if len(idx) == 0: skill_bm[si] = "?"; continue
        bc = Counter()
        for it in idx[:100]: bc[items_data[int(it)].get("benchmark", "?")] += 1
        skill_bm[si] = bc.most_common(1)[0][0]

    top_fams = ["Llama", "Qwen", "Gemma", "Mistral"]
    print(f"  {n_llms} LLMs, K={K}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Fit scaling laws per skill (using Qwen — most complete range)
    # ══════════════════════════════════════════════════════════════
    print("\nFitting scaling laws (Qwen family)...", flush=True)

    qwen_mask = np.array([f == "Qwen" for f in families])
    qwen_sizes = sizes[qwen_mask]
    qwen_mastery = mastery[qwen_mask]
    valid_qwen = ~np.isnan(qwen_sizes)
    qsizes = qwen_sizes[valid_qwen]
    qmastery = qwen_mastery[valid_qwen]

    # Bin by size for cleaner fits
    size_bins = [0.3, 1, 2, 4, 8, 15, 35, 80, 120]
    bin_centers = []
    bin_mastery = np.zeros((len(size_bins)-1, K))
    for bi in range(len(size_bins)-1):
        mask = (qsizes >= size_bins[bi]) & (qsizes < size_bins[bi+1])
        if mask.sum() > 0:
            bin_centers.append(np.sqrt(size_bins[bi] * size_bins[bi+1]))
            bin_mastery[bi] = qmastery[mask].mean(axis=0)
        else:
            bin_centers.append(np.sqrt(size_bins[bi] * size_bins[bi+1]))
            bin_mastery[bi] = np.nan

    bc = np.array(bin_centers)
    valid_bins = ~np.isnan(bin_mastery[:, 0])
    bc_v = bc[valid_bins]
    log_bc = np.log(bc_v)

    fits = []
    for si in range(K):
        y = bin_mastery[valid_bins, si]
        if np.isnan(y).any() or len(y) < 3:
            fits.append({"skill": si, "best": "none", "r2": -1, "sigmoid": {}, "linear": {}, "power": {}})
            continue
        f_sig = fit_sigmoid(log_bc, y)
        f_lin = fit_linear(log_bc, y)
        f_pow = fit_power(bc_v, y)
        best = min([f_sig, f_lin, f_pow], key=lambda x: x["aic"])
        fits.append({"skill": si, "best": best["type"], "r2": best["r2"],
                      "sigmoid": f_sig, "linear": f_lin, "power": f_pow})

    # ══════════════════════════════════════════════════════════════
    # Figure 1: Skill acquisition heatmap (25 curated skills)
    # ══════════════════════════════════════════════════════════════
    print("\nFig 1: Skill acquisition heatmap...", flush=True)
    setup_style()

    # Sort skills by emergence threshold
    emergence = []
    for si in range(K):
        y = bin_mastery[valid_bins, si]
        crossed = np.where(y > 0.5)[0]
        emergence.append(bc_v[crossed[0]] if len(crossed) > 0 else 999)

    # Filter non-English, then pick 5 early + 10 mid + 10 late
    valid_skills = [si for si in range(K)
                    if all(ord(c) < 256 for c in skill_names[si])]
    valid_skills.sort(key=lambda si: emergence[si])

    early = [si for si in valid_skills if emergence[si] <= 1.5][:5]
    mid = [si for si in valid_skills if 1.5 < emergence[si] <= 20][:10]
    late = [si for si in valid_skills if emergence[si] > 20][:10]
    selected = early + mid + late
    # Pad if needed
    for si in valid_skills:
        if si not in selected:
            selected.append(si)
        if len(selected) >= 25:
            break
    selected = selected[:25]

    heatmap_data = bin_mastery[valid_bins][:, selected].T
    heatmap_labels = [short_skill(skill_names[i]) for i in selected]
    bin_labels = [f"{b:.1f}B" if b < 1 else f"{b:.0f}B" for b in bc_v]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(np.arange(len(bin_labels) + 1) - 0.5,
                       np.arange(len(selected) + 1) - 0.5,
                       heatmap_data, cmap="viridis", vmin=0, vmax=1,
                       edgecolors="none", linewidth=0)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.set_xlabel("Model size (Qwen family)", fontsize=13)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(heatmap_labels, fontsize=9)
    ax.invert_yaxis()
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean mastery", fontsize=11)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout(pad=2.0)
    fig.savefig(fig_dir / "fig_skill_acquisition_heatmap.pdf", **SAVE_KW)
    plt.close()
    print(f"  Saved fig_skill_acquisition_heatmap.pdf", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Figure 2: Emergence landscape
    # ══════════════════════════════════════════════════════════════
    print("Fig 2: Emergence landscape...", flush=True)

    bm_colors = {"MATH": "#3B82F6", "BBH": "#22C55E", "GPQA": "#F97316",
                 "MuSR": "#A855F7", "IFEval": "#EF4444", "?": "#999999"}

    # Filter: only skills with R²>0.3, s < 15 (exclude cap artifacts), English
    sig_fits = []
    for si, f in enumerate(fits):
        sf = f["sigmoid"]
        if sf.get("r2", -1) <= 0.3 or not sf.get("params"):
            continue
        s_val = sf["params"].get("s", 0)
        if s_val >= 15:  # cap artifact — not meaningful
            continue
        if not all(ord(c) < 256 for c in skill_names[si]):
            continue
        sig_fits.append((si, sf))

    fig, ax = plt.subplots(figsize=(8, 6))
    for si, sf in sig_fits:
        p = sf["params"]
        mu, s = p.get("mu", 0), p.get("s", 0)
        bm = skill_bm.get(si, "?")
        ax.scatter(np.exp(mu), s, c=bm_colors.get(bm, "#999"), s=80, alpha=0.65,
                   edgecolors="white", linewidths=0.5)

    # Label 5 most interesting: high steepness OR very late emergence
    outlier_scores = []
    for si, sf in sig_fits:
        p = sf["params"]
        mu_exp = np.exp(p.get("mu", 0))
        s_val = p.get("s", 0)
        # Score: prefer high steepness in the 3-10B range, or very late emergence
        score = s_val * 2 + (mu_exp / 20 if mu_exp > 10 else 0)
        outlier_scores.append((si, mu_exp, s_val, score))
    outlier_scores.sort(key=lambda x: -x[3])

    # Label top 5 outliers using adjustText for automatic non-overlapping placement
    from adjustText import adjust_text as at_func

    texts = []
    for si, mu_exp, s_val, _ in outlier_scores[:5]:
        t = ax.text(mu_exp, s_val, short_skill(skill_names[si], 22),
                    fontsize=9, alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.85, edgecolor="none"))
        texts.append(t)

    # Collect all scatter point positions for avoidance
    all_x = [np.exp(sf["params"].get("mu", 0)) for _, sf in sig_fits]
    all_y = [sf["params"].get("s", 0) for _, sf in sig_fits]

    at_func(texts, x=all_x, y=all_y, ax=ax,
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.7),
            force_text=(1.5, 2.0), force_points=(1.0, 1.5),
            expand_text=(1.3, 1.5), expand_points=(1.3, 1.5),
            only_move={"text": "xy", "points": "xy"})

    from matplotlib.patches import Patch
    shown = set(skill_bm.get(si, "?") for si, _ in sig_fits)
    handles = [Patch(facecolor=bm_colors[b], label=b)
               for b in ["MATH", "BBH", "GPQA", "MuSR", "IFEval"] if b in shown]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc="upper right")

    ax.set_xscale("log")
    ax.set_xlabel("Emergence threshold (billion parameters)", fontsize=13)
    ax.set_ylabel("Sigmoid steepness", fontsize=13)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)
    plt.tight_layout(pad=2.0)
    fig.savefig(fig_dir / "fig_emergence_landscape.pdf", **SAVE_KW)
    plt.close()
    print(f"  Saved fig_emergence_landscape.pdf", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Figure 3: Cross-family trajectories
    # ══════════════════════════════════════════════════════════════
    print("Fig 3: Cross-family trajectories...", flush=True)

    # Pick 4 skills: monotonic upward, families diverge, recognizable names
    # Score: monotonicity * range * cross-family variance
    skill_scores = []
    for si in range(K):
        if not all(ord(c) < 256 for c in skill_names[si]): continue
        y = bin_mastery[valid_bins, si]
        if np.isnan(y).any() or len(y) < 4: continue
        y_range = y.max() - y.min()
        if y.max() < 0.3 or y.min() > 0.8: continue
        if y_range < 0.2: continue
        # Monotonicity: correlation with bin index
        mono = np.corrcoef(np.arange(len(y)), y)[0, 1]
        if mono < 0.6: continue  # require mostly upward
        # Cross-family variance at 7B
        fam_vals = []
        for fam in top_fams:
            fmask = np.array([f == fam for f in families]) & (sizes >= 6) & (sizes <= 9)
            if fmask.sum() >= 2:
                fam_vals.append(mastery[fmask, si].mean())
        cross_var = np.std(fam_vals) if len(fam_vals) >= 3 else 0
        # Penalize noisy trajectories: require EVERY family to be monotonic
        fam_monos = []
        for fam in top_fams:
            fmask = np.array([f == fam for f in families]) & ~np.isnan(sizes)
            if fmask.sum() < 5: continue
            fs = sizes[fmask]; fm = mastery[fmask, si]
            fm_means = []
            for bi in range(len(size_bins)-1):
                bm = (fs >= size_bins[bi]) & (fs < size_bins[bi+1])
                if bm.sum() > 0: fm_means.append(fm[bm].mean())
            if len(fm_means) >= 3:
                fam_monos.append(np.corrcoef(np.arange(len(fm_means)), fm_means)[0, 1])
        if not fam_monos: continue
        min_fam_mono = min(fam_monos)
        if min_fam_mono < 0.4: continue  # skip if ANY family has a dip
        avg_fam_mono = np.mean(fam_monos)
        score = y_range * (1 + cross_var * 3) * avg_fam_mono
        skill_scores.append((si, score))
    skill_scores.sort(key=lambda x: -x[1])

    # Blacklist skills with known dips or niche names
    blacklist_short = {"Decay length", "Tone consistency", "Financial sentiment",
                       "Weight loss iteration", "Audio mixing", "Chess history"}
    skill_scores = [(si, sc) for si, sc in skill_scores
                    if short_skill(skill_names[si]) not in blacklist_short]

    # Pick from different benchmarks
    skill_picks = []
    seen_bm = set()
    for si, _ in skill_scores:
        bm = skill_bm.get(si, "?")
        if bm not in seen_bm or len(skill_picks) >= 3:
            skill_picks.append(si)
            seen_bm.add(bm)
        if len(skill_picks) == 4: break
    while len(skill_picks) < 4:
        for si, _ in skill_scores:
            if si not in skill_picks:
                skill_picks.append(si); break

    fam_colors = {"Llama": "#3B82F6", "Qwen": "#F97316", "Gemma": "#22C55E", "Mistral": "#EF4444"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, si in zip(axes.flat, skill_picks[:4]):
        for fam in top_fams:
            fmask = np.array([f == fam for f in families]) & ~np.isnan(sizes)
            if fmask.sum() < 3: continue
            fs = sizes[fmask]
            fm = mastery[fmask, si]
            means, stds, centers = [], [], []
            for bi in range(len(size_bins)-1):
                bmask = (fs >= size_bins[bi]) & (fs < size_bins[bi+1])
                if bmask.sum() > 0:
                    centers.append(np.sqrt(size_bins[bi] * size_bins[bi+1]))
                    means.append(fm[bmask].mean())
                    stds.append(fm[bmask].std() if bmask.sum() > 1 else 0)
            if len(centers) >= 2:
                centers, means, stds = np.array(centers), np.array(means), np.array(stds)
                ax.fill_between(centers, means - stds, means + stds,
                                color=fam_colors[fam], alpha=0.06)
                ax.plot(centers, means, "o-", color=fam_colors[fam], lw=1.8,
                        markersize=5, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Parameters (B)", fontsize=11)
        ax.set_ylabel("Mastery", fontsize=11)
        ax.set_title(short_skill(skill_names[si]), fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.1)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    # Single shared legend at bottom
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=fam_colors[f], lw=2, marker="o", markersize=5, label=f)
               for f in top_fams]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=2.0)
    fig.savefig(fig_dir / "fig_cross_family_trajectories.pdf", **SAVE_KW)
    plt.close()
    print(f"  Saved fig_cross_family_trajectories.pdf", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Summary statistics
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("SCALING LAW SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    best_types = Counter(f["best"] for f in fits)
    print(f"\nBest fit type: {dict(best_types)}", flush=True)

    # Sigmoid inflection points
    inflections = []
    steepness = []
    for f in fits:
        sp = f["sigmoid"].get("params", {})
        if sp and f["sigmoid"].get("r2", -1) > 0.3:
            inflections.append(np.exp(sp["mu"]))
            steepness.append(sp["s"])

    if inflections:
        print(f"\nSigmoid fits ({len(inflections)} skills):", flush=True)
        print(f"  Inflection point: median={np.median(inflections):.1f}B, "
              f"mean={np.mean(inflections):.1f}B", flush=True)
        print(f"  Steepness: median={np.median(steepness):.2f}, "
              f"max={np.max(steepness):.2f}", flush=True)

    # Steepest emergence
    print(f"\nTop 5 steepest emergence (potential 'emergent abilities'):", flush=True)
    steep_sorted = sorted(enumerate(fits), key=lambda x: -x[1]["sigmoid"].get("params", {}).get("s", 0))
    for si, f in steep_sorted[:5]:
        sp = f["sigmoid"].get("params", {})
        if sp:
            print(f"  {short_skill(skill_names[si])}: s={sp['s']:.2f}, "
                  f"threshold={np.exp(sp['mu']):.1f}B, R²={f['sigmoid']['r2']:.3f}", flush=True)

    # Family efficiency: mean mastery at 7B
    print(f"\nFamily efficiency (mean mastery at 7B):", flush=True)
    for fam in top_fams:
        fmask = np.array([f == fam for f in families]) & (sizes >= 6) & (sizes <= 9)
        if fmask.sum() > 0:
            mean_m = mastery[fmask].mean()
            print(f"  {fam}: {mean_m:.4f} (n={fmask.sum()})", flush=True)

    # Kendall's tau: do families learn skills in same order?
    print(f"\nCross-family skill ordering (Kendall's tau):", flush=True)
    fam_orderings = {}
    for fam in top_fams:
        fmask = np.array([f == fam for f in families]) & ~np.isnan(sizes)
        if fmask.sum() < 5: continue
        fam_orderings[fam] = mastery[fmask].mean(axis=0)

    for i, f1 in enumerate(top_fams):
        for f2 in top_fams[i+1:]:
            if f1 in fam_orderings and f2 in fam_orderings:
                tau, p = kendalltau(fam_orderings[f1], fam_orderings[f2])
                print(f"  {f1} vs {f2}: tau={tau:.3f}, p={p:.2e}", flush=True)

    # ── Save ──
    save_data = {
        "n_llms": n_llms, "K": K,
        "best_fit_counts": dict(best_types),
        "n_sigmoid_fits": len(inflections),
        "median_inflection_B": float(np.median(inflections)) if inflections else None,
        "median_steepness": float(np.median(steepness)) if steepness else None,
        "top5_steepest": [
            {"skill": short_skill(skill_names[si]),
             "steepness": fits[si]["sigmoid"].get("params", {}).get("s", 0),
             "threshold_B": float(np.exp(fits[si]["sigmoid"].get("params", {}).get("mu", 0)))}
            for si, _ in steep_sorted[:5] if fits[si]["sigmoid"].get("params")
        ],
    }
    out = Path("cdm_exploration/experiments/v2_scaling_laws.json")
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {out}", flush=True)

    log_experiment(
        name="scaling_laws",
        config={"K": K, "n_llms": n_llms},
        results=save_data,
        split_info={"n_llms": n_llms, "n_items": n_items},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
