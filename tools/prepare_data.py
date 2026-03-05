"""Prepare combined MATH + GSM8K dataset from HuggingFace.

Usage:
    python tools/prepare_data.py
    python tools/prepare_data.py paths.combined=other/dir
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.prepare import (
        create_stratified_sample,
        load_gsm8k_dataset,
        load_math_dataset,
    )

    import pandas as pd

    out_dir = Path(cfg.paths.combined)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MATH dataset...")
    math_df = load_math_dataset()
    print(f"  MATH: {len(math_df):,} problems")

    print("Loading GSM8K dataset...")
    gsm8k_df = load_gsm8k_dataset()
    print(f"  GSM8K: {len(gsm8k_df):,} problems")

    combined = pd.concat([math_df, gsm8k_df], ignore_index=True)
    print(f"  Combined: {len(combined):,} problems")

    sample_small = create_stratified_sample(combined, n_per_group=10)
    sample_medium = create_stratified_sample(combined, n_per_group=50)
    print(f"  Samples: small={len(sample_small)}, medium={len(sample_medium)}")

    combined.to_csv(out_dir / "full_dataset.csv", index=False)
    sample_small.to_csv(out_dir / "sample_small.csv", index=False)
    sample_medium.to_csv(out_dir / "sample_medium.csv", index=False)

    with open(out_dir / "sample_small.json", "w") as f:
        json.dump(sample_small.to_dict(orient="records"), f, indent=2)

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
