"""Load and process RouterEval leaderboard data."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_leaderboards(routereval_dir: str | Path) -> tuple:
    """Load old (GSM8K) and new (MATH) leaderboard pickle files."""
    base = Path(routereval_dir) / "leaderboard_score" / "leaderboard_score"
    with open(base / "leaderboard_old.pkl", "rb") as f:
        old = pickle.load(f)
    with open(base / "leaderboard_new.pkl", "rb") as f:
        new = pickle.load(f)
    return old, new


def load_prompts(routereval_dir: str | Path) -> tuple:
    """Load question text for alignment verification."""
    base = Path(routereval_dir) / "leaderboard_prompt" / "leaderboard_prompt"
    with open(base / "leaderboard_old_prompt.pkl", "rb") as f:
        old_prompts = pickle.load(f)
    with open(base / "leaderboard_new_prompt.pkl", "rb") as f:
        new_prompts = pickle.load(f)
    return old_prompts, new_prompts


def normalize_model_name(name: str) -> str:
    """Strip prefix/suffix differences between old and new leaderboard naming."""
    name = name.replace("open-llm-leaderboard-old/", "")
    if name.startswith("details_"):
        name = name[len("details_"):]
    if name.endswith("-details"):
        name = name[: -len("-details")]
    return name


def find_shared_llms(
    old_models: list, new_models: list
) -> tuple[list, list, list]:
    """Find LLMs present in both leaderboards, return aligned index maps."""
    old_norm = {normalize_model_name(m): i for i, m in enumerate(old_models)}
    new_norm = {normalize_model_name(m): i for i, m in enumerate(new_models)}
    shared = sorted(set(old_norm) & set(new_norm))
    old_indices = [old_norm[m] for m in shared]
    new_indices = [new_norm[m] for m in shared]
    return shared, old_indices, new_indices


def build_gsm8k_block(
    old_data: dict, old_indices: list, old_prompts: dict
) -> tuple[np.ndarray, list[str]]:
    """Extract GSM8K response sub-matrix for shared LLMs."""
    correctness = old_data["data"]["harness_gsm8k_5"]["correctness"]
    sub = correctness[:, old_indices]

    prompts = old_prompts["harness_gsm8k_5"]
    questions = []
    for p in prompts:
        text = str(p)
        marker = text.rfind("Question: ")
        if marker != -1:
            q = text[marker + len("Question: "):]
            q = q.split("\nAnswer:")[0].strip()
        else:
            q = text[-300:]
        questions.append(q)

    return sub, questions


def build_math_block(
    new_data: dict, new_indices: list, new_prompts: dict
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract MATH Lvl 5 response sub-matrix for shared LLMs."""
    math_keys = sorted(k for k in new_data["data"] if k.startswith("math_"))
    subject_map = {
        "math_algebra_hard": "Algebra",
        "math_counting_and_prob_hard": "Counting & Probability",
        "math_geometry_hard": "Geometry",
        "math_intermediate_algebra_hard": "Intermediate Algebra",
        "math_num_theory_hard": "Number Theory",
        "math_prealgebra_hard": "Prealgebra",
        "math_precalculus_hard": "Precalculus",
    }

    all_rows, all_questions, all_subjects = [], [], []

    for key in math_keys:
        correctness = new_data["data"][key]["correctness"]
        sub = correctness[:, new_indices]
        all_rows.append(sub)

        prompts = new_prompts.get(key, [])
        subject = subject_map.get(key, key)

        for p in prompts:
            text = str(p)
            marker = text.rfind("Problem:")
            if marker != -1:
                q = text[marker + len("Problem:"):].strip()
                q = q.split("\nSolution:")[0].split("\nAnswer:")[0].strip()
            else:
                q = text[-300:]
            all_questions.append(q)
            all_subjects.append(subject)

    combined = np.vstack(all_rows)
    return combined, all_questions, all_subjects
