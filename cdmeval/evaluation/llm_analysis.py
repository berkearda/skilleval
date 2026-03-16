"""LLM taxonomy classification and skill mastery analysis."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ════════════════════════════════════════════════════════════════════
#  LLM classification
# ════════════════════════════════════════════════════════════════════

# Model families: (keyword in org or model name) -> family label
_FAMILY_RULES = [
    ("meta-llama", "Llama"),
    ("llama", "Llama"),
    ("mistralai", "Mistral"),
    ("mistral", "Mistral"),
    ("Qwen", "Qwen"),
    ("qwen", "Qwen"),
    ("google", "Gemma"),
    ("gemma", "Gemma"),
    ("microsoft", "Phi"),
    ("phi-", "Phi"),
    ("01-ai", "Yi"),
    ("Yi-", "Yi"),
    ("deepseek", "DeepSeek"),
    ("EleutherAI", "EleutherAI"),
    ("pythia", "EleutherAI"),
    ("gpt-neo", "EleutherAI"),
    ("CohereForAI", "Cohere"),
    ("bigcode", "StarCoder"),
    ("starcoder", "StarCoder"),
    ("SOLAR", "SOLAR"),
    ("upstage", "SOLAR"),
    ("Intel__neural-chat", "Intel"),
    ("NousResearch", "Nous"),
    ("hermes", "Nous"),
    ("Deci", "Deci"),
    ("togethercomputer", "Together"),
    ("RedPajama", "Together"),
    ("pankajmathur", "Orca"),
    ("orca", "Orca"),
    ("vicgalle", "Vicgalle"),
    ("BEE-spoke", "BEE-spoke"),
    ("Azure99", "Blossom"),
    ("TIGER-Lab", "TIGER-Lab"),
]

# Math-specialized keywords
_MATH_KW = ["math", "metamath", "mammoth", "abel", "numina"]

# Code-specialized keywords
_CODE_KW = [
    "code", "coder", "starcoder", "codellama", "codebooga",
    "codeninja", "nxcode", "speechless-coder",
]

# Instruct/chat keywords
_INSTRUCT_KW = [
    "instruct", "chat", "dpo", "rlhf", "orpo", "sft",
    "gguf", "gptq", "ultra", "plus",
]


def _extract_size(name: str) -> float | None:
    """Extract parameter count in billions from model name."""
    low = name.lower()
    # Match patterns like 7B, 70B, 1.5B, 7b, 220M, 32M
    m = re.search(r"(\d+(?:\.\d+)?)\s*b(?:ase|[-_\s]|$)", low)
    if m:
        return float(m.group(1))
    # MoE: 8x7B
    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", low)
    if m:
        return float(m.group(1)) * float(m.group(2))
    # Millions: 220M, 160m
    m = re.search(r"(\d+)\s*m(?:[-_\s]|$)", low)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def classify_llms(llm_names: list[str]) -> pd.DataFrame:
    """Parse LLM names into family, size, and model type.

    Returns:
        DataFrame with columns ``[llm_name, family, size_b, model_type]``.
    """
    rows = []
    for name in llm_names:
        low = name.lower()

        # Family
        family = "Other"
        for kw, fam in _FAMILY_RULES:
            if kw.lower() in low:
                family = fam
                break

        # Size
        size = _extract_size(name)

        # Model type
        if any(k in low for k in _MATH_KW):
            model_type = "math"
        elif any(k in low for k in _CODE_KW):
            model_type = "code"
        elif any(k in low for k in _INSTRUCT_KW):
            model_type = "instruct"
        else:
            model_type = "base"

        rows.append({
            "llm_name": name,
            "family": family,
            "size_b": size,
            "model_type": model_type,
        })

    df = pd.DataFrame(rows)

    # Print stats
    print(f"\nLLM Classification ({len(df)} models):")
    print(f"\n  By type:")
    for t, cnt in df["model_type"].value_counts().items():
        print(f"    {t}: {cnt}")
    print(f"\n  By family (top 10):")
    for f, cnt in df["family"].value_counts().head(10).items():
        print(f"    {f}: {cnt}")
    print(f"    (+ {(df['family'] == 'Other').sum()} Other)")

    sized = df["size_b"].notna().sum()
    print(f"\n  Size parsed: {sized}/{len(df)} ({sized/len(df)*100:.0f}%)")

    return df


# ════════════════════════════════════════════════════════════════════
#  Mastery by model type
# ════════════════════════════════════════════════════════════════════


def analyze_mastery_by_type(
    mastery_df: pd.DataFrame,
    llm_info: pd.DataFrame,
    fig_dir: Path,
) -> pd.DataFrame:
    """Compute and plot mean mastery profile per model type."""
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    setup_style()

    merged = mastery_df.copy()
    merged["model_type"] = llm_info.set_index("llm_name").loc[
        mastery_df.index, "model_type"
    ].values

    skill_cols = [c for c in mastery_df.columns]
    type_means = merged.groupby("model_type")[skill_cols].mean()

    # Sort skills by overall variance across types
    skill_var = type_means.var(axis=0).sort_values(ascending=False)
    top_skills = skill_var.head(30).index.tolist()

    # Reorder types
    type_order = ["base", "instruct", "math", "code"]
    type_order = [t for t in type_order if t in type_means.index]
    plot_data = type_means.loc[type_order, top_skills]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    sns.heatmap(
        plot_data, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Mean mastery", "shrink": 0.8},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_title("Mean skill mastery by model type (top-30 differentiating skills)")

    plt.tight_layout()
    out = fig_dir / "fig_mastery_by_type.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")

    return type_means


# ════════════════════════════════════════════════════════════════════
#  Mastery embedding (t-SNE) by family
# ════════════════════════════════════════════════════════════════════


def analyze_mastery_by_family(
    mastery_df: pd.DataFrame,
    llm_info: pd.DataFrame,
    fig_dir: Path,
    top_n_families: int = 8,
) -> None:
    """t-SNE of mastery profiles colored by family, shaped by type."""
    from sklearn.manifold import TSNE

    from cdmeval.utils.visualization import SAVE_KW, setup_style

    setup_style()

    info = llm_info.set_index("llm_name").loc[mastery_df.index]
    families = info["family"].values
    types = info["model_type"].values

    # Top families
    fam_counts = pd.Series(families).value_counts()
    top_fams = fam_counts.head(top_n_families).index.tolist()
    fam_labels = np.array([f if f in top_fams else "Other" for f in families])

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(mastery_df.values)

    # Type -> marker
    type_markers = {"base": "o", "instruct": "^", "math": "*", "code": "s"}

    # Family -> color
    all_fams = top_fams + ["Other"]
    palette = sns.color_palette("husl", len(top_fams))
    fam_colors = {f: palette[i] for i, f in enumerate(top_fams)}
    fam_colors["Other"] = "#cccccc"

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot "Other" first (background)
    for mtype, marker in type_markers.items():
        mask = (fam_labels == "Other") & (types == mtype)
        if mask.sum() > 0:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c="#cccccc", marker=marker, s=25, alpha=0.3,
                edgecolors="none", rasterized=True,
            )

    # Plot top families
    for fam in top_fams:
        for mtype, marker in type_markers.items():
            mask = (fam_labels == fam) & (types == mtype)
            if mask.sum() > 0:
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[fam_colors[fam]], marker=marker, s=50, alpha=0.75,
                    edgecolors="white", linewidths=0.3,
                    label=f"{fam} ({mtype})" if mask.sum() >= 2 else None,
                )

    # Legends
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fam_handles = [Patch(facecolor=fam_colors[f], label=f) for f in top_fams]
    fam_handles.append(Patch(facecolor="#cccccc", label="Other"))

    type_handles = [
        Line2D([0], [0], marker=m, color="grey", linestyle="", markersize=7, label=t)
        for t, m in type_markers.items()
    ]

    leg1 = ax.legend(
        handles=fam_handles, title="Family", loc="upper left",
        fontsize=7, title_fontsize=8, frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=type_handles, title="Type", loc="lower left",
        fontsize=7, title_fontsize=8, frameon=False,
    )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("LLM skill mastery profiles (t-SNE, 235 models × 50 skills)")
    ax.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    out = fig_dir / "fig_llm_embedding_tsne.pdf"
    fig.savefig(out, **SAVE_KW)
    plt.close()
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════
#  Differentiating skills (ANOVA)
# ════════════════════════════════════════════════════════════════════


def find_differentiating_skills(
    mastery_df: pd.DataFrame,
    llm_info: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """One-way ANOVA per skill across model types. Return top differentiators."""
    info = llm_info.set_index("llm_name").loc[mastery_df.index]
    types = info["model_type"].values
    unique_types = sorted(set(types))
    skill_cols = list(mastery_df.columns)

    rows = []
    for skill in skill_cols:
        groups = [mastery_df.loc[types == t, skill].values for t in unique_types]
        # Need at least 2 groups with data
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        means = {
            t: float(mastery_df.loc[types == t, skill].mean())
            for t in unique_types
        }
        rows.append({"skill": skill, "F": f_stat, "p": p_val, **means})

    result = pd.DataFrame(rows).sort_values("F", ascending=False).head(top_n)

    print(f"\nTop {top_n} differentiating skills (ANOVA across model types):")
    print("-" * 90)
    header = f"{'Skill':<35} {'F':>8} {'p':>10}"
    for t in unique_types:
        header += f" {t:>10}"
    print(header)
    print("-" * 90)
    for _, row in result.iterrows():
        line = f"{row['skill']:<35} {row['F']:>8.2f} {row['p']:>10.2e}"
        for t in unique_types:
            line += f" {row.get(t, 0):>10.3f}"
        print(line)

    return result
