"""LLM taxonomy and skill mastery analysis.

Usage:
    python tools/llm_analysis.py device=mps
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.response_matrix import load_response_matrix
    from cdmeval.evaluation.llm_analysis import (
        analyze_mastery_by_family,
        analyze_mastery_by_type,
        classify_llms,
        find_differentiating_skills,
    )

    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    K = cfg.skills.n_clusters

    # ── Load data ──
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    print(f"Data: {n_llms} LLMs, {n_items} items")

    # Load mastery profiles (from existing HAC-K NCDM)
    mastery_path = data_dir / f"skill_mastery_profiles_hac{K}.csv"
    mastery_df = pd.read_csv(mastery_path, index_col=0)
    print(f"Mastery profiles: {mastery_df.shape}")

    # ── 1. Classify LLMs ──
    print("\n" + "=" * 60)
    print("LLM Classification")
    print("=" * 60)
    llm_info = classify_llms(llm_names)

    # Save classification
    llm_info.to_csv(data_dir / "llm_classification.csv", index=False)
    print(f"\nSaved llm_classification.csv")

    # ── 2. Mastery by model type ──
    print("\n" + "=" * 60)
    print("Mastery Analysis by Model Type")
    print("=" * 60)
    type_means = analyze_mastery_by_type(mastery_df, llm_info, fig_dir)

    # Print summary
    print("\nMean overall mastery by type:")
    for t in type_means.index:
        print(f"  {t}: {type_means.loc[t].mean():.4f}")

    # ── 3. t-SNE by family ──
    print("\n" + "=" * 60)
    print("t-SNE Embedding by Family")
    print("=" * 60)
    analyze_mastery_by_family(mastery_df, llm_info, fig_dir)

    # ── 4. Differentiating skills ──
    print("\n" + "=" * 60)
    print("Differentiating Skills (ANOVA)")
    print("=" * 60)
    diff_skills = find_differentiating_skills(mastery_df, llm_info)

    # Save results
    diff_skills.to_csv(data_dir / "differentiating_skills.csv", index=False)
    print(f"\nSaved differentiating_skills.csv")

    # ── Save all results ──
    results = {
        "n_llms": n_llms,
        "type_counts": llm_info["model_type"].value_counts().to_dict(),
        "family_counts": llm_info["family"].value_counts().to_dict(),
        "mean_mastery_by_type": {
            t: float(type_means.loc[t].mean()) for t in type_means.index
        },
        "top_differentiating_skills": diff_skills["skill"].tolist(),
    }
    with open(data_dir / "llm_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved llm_analysis_results.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
