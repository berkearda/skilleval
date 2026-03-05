"""Generate publication-quality figures for the CDM report.

Usage:
    python tools/generate_figures.py
    python tools/generate_figures.py paths.figures=output/figs
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.utils.visualization import (
        fig_llm_ranking,
        fig_mastery_heatmap,
        fig_model_fit,
        fig_qmatrix,
        fig_skill_distribution,
        fig_umap,
    )

    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures for progress report...\n")
    fig_skill_distribution(data_dir, fig_dir)
    fig_umap(data_dir, fig_dir)
    fig_qmatrix(data_dir, fig_dir)
    fig_mastery_heatmap(data_dir, fig_dir)
    fig_llm_ranking(data_dir, fig_dir)
    fig_model_fit(data_dir, fig_dir)
    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
