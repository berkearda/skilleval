"""Build the earlier two-benchmark response matrix (MATH + BBH), superseded by tools/build_response_matrix_v2.py.

Uses only the new leaderboard (3,811 models), no OLD leaderboard dependency.
Filters out low-accuracy LLMs and unsolvable items.

Usage:
    python tools/build_expanded_matrix.py
"""

import json
import pickle
from pathlib import Path

import numpy as np


def normalize_name(name):
    name = name.replace("open-llm-leaderboard-old/", "")
    if name.startswith("details_"):
        name = name[len("details_"):]
    if name.endswith("-details"):
        name = name[: -len("-details")]
    return name


def extract_prompts(prompts_data, key, benchmark):
    """Extract question text from prompt data."""
    prompts = prompts_data.get(key, [])
    questions = []
    for p in prompts:
        text = str(p)
        if benchmark == "MATH":
            marker = text.rfind("Problem:")
            if marker != -1:
                q = text[marker + len("Problem:"):].strip()
                q = q.split("\nSolution:")[0].split("\nAnswer:")[0].strip()
            else:
                q = text[-300:]
        else:
            # BBH: extract the question after the last newline pair
            q = text[-500:]
        questions.append(q)
    return questions


def main():
    base = Path("cdm_exploration/data/routereval/leaderboard_score/leaderboard_score")
    prompt_base = Path("cdm_exploration/data/routereval/leaderboard_prompt/leaderboard_prompt")
    out_dir = Path("cdm_exploration/data/cdm_ready")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load new leaderboard ──
    print("Loading new leaderboard...")
    with open(base / "leaderboard_new.pkl", "rb") as f:
        new = pickle.load(f)

    with open(prompt_base / "leaderboard_new_prompt.pkl", "rb") as f:
        new_prompts = pickle.load(f)

    raw_model_names = list(new["model"])
    n_raw_models = len(raw_model_names)
    model_names = [normalize_name(m) for m in raw_model_names]
    print(f"Raw models: {n_raw_models}")

    # ── Select benchmarks ──
    math_keys = sorted(k for k in new["data"] if k.startswith("math_"))
    bbh_keys = sorted(k for k in new["data"] if k.startswith("bbh_"))

    math_subject_map = {
        "math_algebra_hard": "Algebra",
        "math_counting_and_prob_hard": "Counting & Probability",
        "math_geometry_hard": "Geometry",
        "math_intermediate_algebra_hard": "Intermediate Algebra",
        "math_num_theory_hard": "Number Theory",
        "math_prealgebra_hard": "Prealgebra",
        "math_precalculus_hard": "Precalculus",
    }

    selected_keys = math_keys + bbh_keys
    print(f"\nSelected benchmarks: {len(math_keys)} MATH + {len(bbh_keys)} BBH = {len(selected_keys)}")

    # ── Build combined matrix (items × models) ──
    blocks = []
    items_meta = []
    item_offset = 0

    for key in selected_keys:
        correctness = new["data"][key]["correctness"]  # (n_items_in_key, n_models)
        n_items_key = correctness.shape[0]
        blocks.append(correctness)

        if key.startswith("math_"):
            benchmark = "MATH"
            subject = math_subject_map.get(key, key)
        else:
            benchmark = "BBH"
            subject = key.replace("bbh_", "")

        prompts = extract_prompts(new_prompts, key, benchmark)

        for i in range(n_items_key):
            items_meta.append({
                "item_idx": item_offset + i,
                "benchmark": benchmark,
                "subtask": subject,
                "key": key,
                "question_preview": prompts[i][:200] if i < len(prompts) else "",
            })
        item_offset += n_items_key

    # Stack: (total_items, n_models)
    raw_matrix = np.vstack(blocks)
    print(f"\nRaw combined matrix: {raw_matrix.shape[0]} items x {raw_matrix.shape[1]} models")

    # Transpose to (models, items) for consistency with existing code
    R = raw_matrix.T  # (n_models, n_items)

    # ── Filter LLMs: remove those with <5% overall accuracy ──
    llm_acc = R.mean(axis=1)
    llm_mask = llm_acc >= 0.05
    R = R[llm_mask]
    kept_model_indices = np.where(llm_mask)[0]
    kept_model_names = [model_names[i] for i in kept_model_indices]
    print(f"\nLLM filter (acc >= 5%): {llm_mask.sum()}/{n_raw_models} kept, "
          f"{n_raw_models - llm_mask.sum()} removed")

    # ── Filter items: remove those solved by 0 LLMs ──
    item_solvers = R.sum(axis=0)
    item_mask = item_solvers > 0
    R = R[:, item_mask]
    kept_item_indices = np.where(item_mask)[0]
    items_meta_filtered = [items_meta[i] for i in kept_item_indices]
    # Reindex
    for new_idx, meta in enumerate(items_meta_filtered):
        meta["item_idx"] = new_idx

    n_items_removed = len(item_mask) - item_mask.sum()
    print(f"Item filter (>0 solvers): {item_mask.sum()}/{len(item_mask)} kept, "
          f"{n_items_removed} removed")

    n_llms, n_items = R.shape
    print(f"\nFinal matrix: {n_llms} LLMs x {n_items} items")

    # ── Per-benchmark breakdown ──
    print(f"\n{'Benchmark':<15} {'Subtask':<40} {'Items':>6} {'Mean Acc':>10}")
    print("-" * 75)

    benchmarks = {}
    for meta in items_meta_filtered:
        bm = meta["benchmark"]
        st = meta["subtask"]
        key = (bm, st)
        if key not in benchmarks:
            benchmarks[key] = []
        benchmarks[key].append(meta["item_idx"])

    total_math = 0
    total_bbh = 0
    for (bm, st), indices in sorted(benchmarks.items()):
        acc = R[:, indices].mean()
        print(f"{bm:<15} {st:<40} {len(indices):>6} {acc:>10.4f}")
        if bm == "MATH":
            total_math += len(indices)
        else:
            total_bbh += len(indices)

    print(f"\n  MATH total: {total_math} items")
    print(f"  BBH total:  {total_bbh} items")
    print(f"  Grand total: {n_items} items")
    print(f"  Grand mean accuracy: {R.mean():.4f}")

    # ── Sparsity ──
    # RouterBench uses binary correctness, no missing values
    nan_count = np.isnan(R.astype(float)).sum()
    print(f"  Missing values: {nan_count} ({nan_count / R.size * 100:.2f}%)")

    # ── Save ──
    np.save(out_dir / "response_matrix_expanded.npy", R)
    print(f"\nSaved: response_matrix_expanded.npy ({R.shape})")

    with open(out_dir / "response_matrix_expanded_items.json", "w") as f:
        json.dump(items_meta_filtered, f, indent=2)
    print(f"Saved: response_matrix_expanded_items.json ({len(items_meta_filtered)} items)")

    with open(out_dir / "response_matrix_expanded_llms.json", "w") as f:
        json.dump(kept_model_names, f, indent=2)
    print(f"Saved: response_matrix_expanded_llms.json ({len(kept_model_names)} LLMs)")

    # ── Verification ──
    assert R.shape[0] == len(kept_model_names)
    assert R.shape[1] == len(items_meta_filtered)
    assert nan_count == 0
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
