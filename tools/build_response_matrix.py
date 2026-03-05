"""Build the LLM response matrix from RouterEval leaderboard data.

Usage:
    python tools/build_response_matrix.py
    python tools/build_response_matrix.py paths.routereval=other/dir
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.routereval import (
        build_gsm8k_block,
        build_math_block,
        find_shared_llms,
        load_leaderboards,
        load_prompts,
    )

    routereval_dir = Path(cfg.paths.routereval)
    out_dir = Path(cfg.paths.cdm_ready)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RouterEval leaderboard data...")
    old, new = load_leaderboards(routereval_dir)
    old_prompts, new_prompts = load_prompts(routereval_dir)

    shared_names, old_idx, new_idx = find_shared_llms(old["model"], new["model"])
    print(f"Shared LLMs: {len(shared_names)}")

    gsm_matrix, gsm_questions = build_gsm8k_block(old, old_idx, old_prompts)
    math_matrix, math_questions, math_subjects = build_math_block(new, new_idx, new_prompts)
    print(f"GSM8K block: {gsm_matrix.shape}, MATH block: {math_matrix.shape}")

    response_matrix = np.vstack([gsm_matrix, math_matrix])
    print(f"Combined response matrix: {response_matrix.shape}")
    print(f"Overall accuracy: {response_matrix.mean():.3f}")

    items = []
    for i, q in enumerate(gsm_questions):
        items.append({
            "item_idx": i, "source": "GSM8K",
            "subject": "Word Problems", "question_preview": q[:150],
        })
    offset = len(gsm_questions)
    for i, (q, s) in enumerate(zip(math_questions, math_subjects)):
        items.append({
            "item_idx": offset + i, "source": "MATH",
            "subject": s, "question_preview": q[:150],
        })

    items_df = pd.DataFrame(items)

    response_df = pd.DataFrame(
        response_matrix.T,
        index=shared_names,
        columns=[f"item_{i}" for i in range(response_matrix.shape[0])],
    )
    response_df.index.name = "llm"

    response_df.to_csv(out_dir / "response_matrix.csv")
    items_df.to_csv(out_dir / "items.csv", index=False)
    pd.Series(shared_names, name="llm").to_csv(out_dir / "llm_list.csv", index=False)

    all_questions = gsm_questions + math_questions
    all_sources = ["GSM8K"] * len(gsm_questions) + ["MATH"] * len(math_questions)
    all_subjects = ["Word Problems"] * len(gsm_questions) + math_subjects

    questions_df = pd.DataFrame({
        "item_idx": range(len(all_questions)),
        "source": all_sources, "subject": all_subjects,
        "question": all_questions,
    })
    questions_df.to_csv(out_dir / "questions_for_extraction.csv", index=False)

    print(f"\nSaved to {out_dir}/:")
    print(f"  response_matrix.csv  ({response_df.shape[0]} LLMs x {response_df.shape[1]} items)")
    print(f"  items.csv            ({len(items_df)} items)")
    print(f"  llm_list.csv         ({len(shared_names)} LLMs)")
    print(f"  questions_for_extraction.csv ({len(questions_df)} questions)")


if __name__ == "__main__":
    main()
