"""Dataset preparation: load MATH and GSM8K from HuggingFace."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_math_dataset() -> pd.DataFrame:
    """Load MATH dataset (all 7 subjects, test split) from HuggingFace."""
    from datasets import load_dataset

    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    rows = []
    for subject in subjects:
        ds = load_dataset(
            "EleutherAI/hendrycks_math", subject, split="test", trust_remote_code=True
        )
        for idx, item in enumerate(ds):
            rows.append({
                "id": f"MATH_{subject}_{idx}",
                "source": "MATH",
                "subject": item["type"],
                "level": item["level"],
                "problem": item["problem"],
                "solution": item["solution"],
            })

    return pd.DataFrame(rows)


def load_gsm8k_dataset() -> pd.DataFrame:
    """Load GSM8K dataset (test split) from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)

    rows = []
    for idx, item in enumerate(ds):
        rows.append({
            "id": f"GSM8K_{idx}",
            "source": "GSM8K",
            "subject": "Word Problems",
            "level": "Grade School",
            "problem": item["question"],
            "solution": item["answer"],
        })

    return pd.DataFrame(rows)


def create_stratified_sample(df: pd.DataFrame, n_per_group: int = 50) -> pd.DataFrame:
    """Create a stratified sample by source and subject."""
    samples = []
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        for subject in source_df["subject"].unique():
            subject_df = source_df[source_df["subject"] == subject]
            n_sample = min(n_per_group, len(subject_df))
            sample = subject_df.sample(n=n_sample, random_state=42)
            samples.append(sample)
    return pd.concat(samples, ignore_index=True)
