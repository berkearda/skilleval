"""Extract skills from math problems via LLM (or mock extractor).

Usage:
    python tools/extract_skills.py
    python tools/extract_skills.py skills.api=openai skills.llm_model=gpt-4o-mini
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.skills.extraction import analyze_extracted_skills, run_extraction

    data_dir = Path(cfg.paths.combined)
    cdm_dir = Path(cfg.paths.cdm_ready)
    cdm_dir.mkdir(parents=True, exist_ok=True)

    input_file = str(data_dir / "sample_small.csv")
    output_file = str(cdm_dir / "skills_extracted.csv")

    print(f"API: {cfg.skills.api}, Model: {cfg.skills.llm_model or 'default'}")

    results_df = run_extraction(
        input_file=input_file,
        output_file=output_file,
        api=cfg.skills.api,
        model=cfg.skills.llm_model,
    )

    analyze_extracted_skills(results_df)


if __name__ == "__main__":
    main()
