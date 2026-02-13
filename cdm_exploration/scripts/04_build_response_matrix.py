"""
Build the LLM response matrix from RouterEval leaderboard data.

Loads the RouterEval pickle files (MATH Lvl 5 + GSM8K), finds LLMs
present in both leaderboards, aligns question ordering with our
dataset, and outputs a binary response matrix (LLMs x items).

Note: requires numpy >= 2.0 to unpickle the RouterEval data.
Run with: python3.12 04_build_response_matrix.py
"""

import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ROUTEREVAL_DIR = DATA_DIR / "routereval"
OUTPUT_DIR = DATA_DIR / "cdm_ready"


def load_leaderboards():
    """Load old (GSM8K) and new (MATH) leaderboard pickle files."""
    with open(ROUTEREVAL_DIR / "leaderboard_score/leaderboard_score/leaderboard_old.pkl", "rb") as f:
        old = pickle.load(f)
    with open(ROUTEREVAL_DIR / "leaderboard_score/leaderboard_score/leaderboard_new.pkl", "rb") as f:
        new = pickle.load(f)
    return old, new


def load_prompts():
    """Load question text for alignment verification."""
    with open(ROUTEREVAL_DIR / "leaderboard_prompt/leaderboard_prompt/leaderboard_old_prompt.pkl", "rb") as f:
        old_prompts = pickle.load(f)
    with open(ROUTEREVAL_DIR / "leaderboard_prompt/leaderboard_prompt/leaderboard_new_prompt.pkl", "rb") as f:
        new_prompts = pickle.load(f)
    return old_prompts, new_prompts


def normalize_model_name(name):
    """Strip prefix/suffix differences between old and new leaderboard naming."""
    name = name.replace("open-llm-leaderboard-old/", "")
    if name.startswith("details_"):
        name = name[len("details_"):]
    if name.endswith("-details"):
        name = name[:-len("-details")]
    return name


def find_shared_llms(old_models, new_models):
    """Find LLMs present in both leaderboards, return aligned index maps."""
    old_norm = {normalize_model_name(m): i for i, m in enumerate(old_models)}
    new_norm = {normalize_model_name(m): i for i, m in enumerate(new_models)}
    shared = sorted(set(old_norm) & set(new_norm))

    old_indices = [old_norm[m] for m in shared]
    new_indices = [new_norm[m] for m in shared]
    return shared, old_indices, new_indices


def build_gsm8k_block(old_data, old_indices, old_prompts):
    """Extract GSM8K response sub-matrix for shared LLMs."""
    correctness = old_data['data']['harness_gsm8k_5']['correctness']
    sub = correctness[:, old_indices]  # (1319, n_shared)

    # Extract question text (strip the few-shot prompt prefix to get core question)
    prompts = old_prompts['harness_gsm8k_5']
    questions = []
    for p in prompts:
        text = str(p)
        # GSM8K prompts end with the actual question after "Question: "
        marker = text.rfind("Question: ")
        if marker != -1:
            q = text[marker + len("Question: "):]
            q = q.split("\nAnswer:")[0].strip()
        else:
            q = text[-300:]
        questions.append(q)

    return sub, questions


def build_math_block(new_data, new_indices, new_prompts):
    """Extract MATH Lvl 5 response sub-matrix for shared LLMs."""
    math_keys = sorted(k for k in new_data['data'] if k.startswith("math_"))
    subject_map = {
        "math_algebra_hard": "Algebra",
        "math_counting_and_prob_hard": "Counting & Probability",
        "math_geometry_hard": "Geometry",
        "math_intermediate_algebra_hard": "Intermediate Algebra",
        "math_num_theory_hard": "Number Theory",
        "math_prealgebra_hard": "Prealgebra",
        "math_precalculus_hard": "Precalculus",
    }

    all_rows = []
    all_questions = []
    all_subjects = []

    for key in math_keys:
        correctness = new_data['data'][key]['correctness']
        sub = correctness[:, new_indices]
        all_rows.append(sub)

        prompts = new_prompts.get(key, [])
        subject = subject_map.get(key, key)

        for i, p in enumerate(prompts):
            text = str(p)
            marker = text.rfind("Problem:")
            if marker != -1:
                q = text[marker + len("Problem:"):].strip()
                q = q.split("\nSolution:")[0].split("\nAnswer:")[0].strip()
            else:
                q = text[-300:]
            all_questions.append(q)
            all_subjects.append(subject)

    combined = np.vstack(all_rows)  # (1324, n_shared)
    return combined, all_questions, all_subjects


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading RouterEval leaderboard data...")
    old, new = load_leaderboards()
    old_prompts, new_prompts = load_prompts()

    # Find shared LLMs
    shared_names, old_idx, new_idx = find_shared_llms(old['model'], new['model'])
    print(f"Shared LLMs: {len(shared_names)}")

    # Build response blocks
    gsm_matrix, gsm_questions = build_gsm8k_block(old, old_idx, old_prompts)
    math_matrix, math_questions, math_subjects = build_math_block(new, new_idx, new_prompts)
    print(f"GSM8K block: {gsm_matrix.shape}, MATH block: {math_matrix.shape}")

    # Combine into single response matrix (items x LLMs)
    response_matrix = np.vstack([gsm_matrix, math_matrix])  # (2643, 235)
    print(f"Combined response matrix: {response_matrix.shape}")
    print(f"Overall accuracy: {response_matrix.mean():.3f}")

    # Build item metadata
    items = []
    for i, q in enumerate(gsm_questions):
        items.append({
            "item_idx": i,
            "source": "GSM8K",
            "subject": "Word Problems",
            "question_preview": q[:150],
        })
    offset = len(gsm_questions)
    for i, (q, s) in enumerate(zip(math_questions, math_subjects)):
        items.append({
            "item_idx": offset + i,
            "source": "MATH",
            "subject": s,
            "question_preview": q[:150],
        })

    items_df = pd.DataFrame(items)

    # Save response matrix (transposed: LLMs as rows for CDM convention)
    response_df = pd.DataFrame(
        response_matrix.T,  # (235, 2643)
        index=shared_names,
        columns=[f"item_{i}" for i in range(response_matrix.shape[0])],
    )
    response_df.index.name = "llm"

    response_df.to_csv(OUTPUT_DIR / "response_matrix.csv")
    items_df.to_csv(OUTPUT_DIR / "items.csv", index=False)
    pd.Series(shared_names, name="llm").to_csv(OUTPUT_DIR / "llm_list.csv", index=False)

    # Save full question texts for skill extraction
    all_questions = gsm_questions + math_questions
    all_sources = ["GSM8K"] * len(gsm_questions) + ["MATH"] * len(math_questions)
    all_subjects = ["Word Problems"] * len(gsm_questions) + math_subjects

    questions_df = pd.DataFrame({
        "item_idx": range(len(all_questions)),
        "source": all_sources,
        "subject": all_subjects,
        "question": all_questions,
    })
    questions_df.to_csv(OUTPUT_DIR / "questions_for_extraction.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/:")
    print(f"  response_matrix.csv  ({response_df.shape[0]} LLMs x {response_df.shape[1]} items)")
    print(f"  items.csv            ({len(items_df)} items with metadata)")
    print(f"  llm_list.csv         ({len(shared_names)} LLM names)")
    print(f"  questions_for_extraction.csv ({len(questions_df)} questions for skill extraction)")

    # Summary stats
    acc_by_source = {}
    for src, mat in [("GSM8K", gsm_matrix), ("MATH", math_matrix)]:
        acc_by_source[src] = mat.mean()
    print(f"\nAccuracy by source: {', '.join(f'{k}={v:.3f}' for k, v in acc_by_source.items())}")

    # Per-LLM accuracy distribution
    llm_accs = response_matrix.mean(axis=0)
    print(f"Per-LLM accuracy: min={llm_accs.min():.3f}, max={llm_accs.max():.3f}, "
          f"median={np.median(llm_accs):.3f}, std={llm_accs.std():.3f}")


if __name__ == "__main__":
    main()
