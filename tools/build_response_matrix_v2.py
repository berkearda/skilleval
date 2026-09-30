#!/usr/bin/env python3
"""Build the response matrix used in the paper: 3,811 LLMs x 9,523 items from five benchmarks.

Reads the Open LLM Leaderboard v2 results packaged by RouterEval (Huang et al., 2025) and writes to
cdm_exploration/data/cdm_ready/:

    response_matrix_v2_full.npy         (3811, 9523) float64, 1 = correct
    response_matrix_v2_full_llms.json   LLM names, in row order
    response_matrix_v2_full_items.json  per item: item_idx, benchmark, subtask, key, question_preview
    item_text_embeddings_v2_full.npz    with --embeddings: (9523, 768) float32 all-mpnet-base-v2 embeddings of the
                                        item texts, normalised; the text is the full question for MATH and
                                        question_preview for the other benchmarks, as for the paper's model

Items are ordered MATH, BBH, then the remaining benchmarks (GPQA, IFEval, MuSR), subtasks alphabetically within each.
LLMs below 5% overall accuracy and items that no LLM answers correctly are removed (no LLM is removed on this data;
51 items are). question_preview is a 200-character excerpt of the prompt: for MATH the start of the text after the
last "Problem:", for BBH the start of the last 500 characters, for GPQA, IFEval and MuSR the last 200 characters.

    python tools/download_data.py            # fetches RouterEval and runs this script
    python tools/build_response_matrix_v2.py # if the RouterEval files are already in place
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ROUTEREVAL = Path("cdm_exploration/data/routereval")
OUT = Path("cdm_exploration/data/cdm_ready")

MATH_SUBJECTS = {
    "math_algebra_hard": "Algebra",
    "math_counting_and_prob_hard": "Counting & Probability",
    "math_geometry_hard": "Geometry",
    "math_intermediate_algebra_hard": "Intermediate Algebra",
    "math_num_theory_hard": "Number Theory",
    "math_prealgebra_hard": "Prealgebra",
    "math_precalculus_hard": "Precalculus",
}


def normalize_name(name):
    name = name.replace("open-llm-leaderboard-old/", "")
    if name.startswith("details_"):
        name = name[len("details_"):]
    if name.endswith("-details"):
        name = name[: -len("-details")]
    return name


def benchmark_and_subtask(key):
    if key.startswith("math_"):
        return "MATH", MATH_SUBJECTS.get(key, key)
    if key.startswith("bbh_"):
        return "BBH", key[len("bbh_"):]
    if key.startswith("gpqa_"):
        return "GPQA", key[len("gpqa_"):].title()
    if key.startswith("musr_"):
        return "MuSR", key[len("musr_"):].replace("_", " ").title()
    if key == "ifeval":
        return "IFEval", "IFEval"
    raise ValueError(f"unknown RouterEval key: {key}")


def math_question(text):
    # the MATH prompts are few-shot; the question is the text after the last "Problem:"
    marker = text.rfind("Problem:")
    if marker == -1:
        return None
    q = text[marker + len("Problem:"):].strip()
    return q.split("\nSolution:")[0].split("\nAnswer:")[0].strip()


def question_preview(prompt, benchmark):
    text = str(prompt)
    if benchmark == "MATH":
        q = math_question(text)
        return q[:200] if q is not None else text[-300:][:200]
    if benchmark == "BBH":
        return text[-500:][:200]
    return text[-200:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--routereval", type=Path, default=ROUTEREVAL, help="folder with the unzipped RouterEval files")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--embeddings", action="store_true", help="also write item_text_embeddings_v2_full.npz")
    a = ap.parse_args()

    score = a.routereval / "leaderboard_score" / "leaderboard_score" / "leaderboard_new.pkl"
    prompt = a.routereval / "leaderboard_prompt" / "leaderboard_prompt" / "leaderboard_new_prompt.pkl"
    for p in (score, prompt):
        if not p.exists():
            raise SystemExit(f"missing {p}; run tools/download_data.py first")
    with open(score, "rb") as f:
        lb = pickle.load(f)
    with open(prompt, "rb") as f:
        prompts = pickle.load(f)

    keys = sorted(k for k in lb["data"] if k.startswith("math_"))
    keys += sorted(k for k in lb["data"] if k.startswith("bbh_"))
    keys += sorted(k for k in lb["data"] if k not in keys)

    blocks, items, embed_texts = [], [], []
    for key in keys:
        correctness = lb["data"][key]["correctness"]  # (items in this subtask, LLMs)
        benchmark, subtask = benchmark_and_subtask(key)
        raw = [str(p) for p in prompts.get(key, [])]
        texts = [question_preview(p, benchmark) for p in raw]
        for i in range(correctness.shape[0]):
            preview = texts[i] if i < len(texts) else ""
            full = math_question(raw[i]) if benchmark == "MATH" and i < len(raw) else None
            items.append({"item_idx": len(items), "benchmark": benchmark, "subtask": subtask, "key": key,
                          "question_preview": preview})
            embed_texts.append(full if full is not None else preview)
        blocks.append(correctness)
    R = np.vstack(blocks).T  # (LLMs, items)
    llms = [normalize_name(str(m)) for m in lb["model"]]

    keep_llm = R.mean(axis=1) >= 0.05
    R, llms = R[keep_llm], [n for n, k in zip(llms, keep_llm) if k]
    keep_item = R.sum(axis=0) > 0
    R = R[:, keep_item]
    items = [it for it, k in zip(items, keep_item) if k]
    embed_texts = [t for t, k in zip(embed_texts, keep_item) if k]
    # The paper's embeddings looked each text up by its preview. Two MATH geometry items share their first 200
    # characters, so both got the question of the later one; kept so that the embeddings match the paper's.
    by_preview = {it["question_preview"]: t for it, t in zip(items, embed_texts)}
    embed_texts = [by_preview[it["question_preview"]] for it in items]
    for i, it in enumerate(items):
        it["item_idx"] = i

    assert not np.isnan(R).any() and set(np.unique(R)) <= {0.0, 1.0}
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "response_matrix_v2_full.npy", R)
    with open(a.out / "response_matrix_v2_full_llms.json", "w") as f:
        json.dump(llms, f)
    with open(a.out / "response_matrix_v2_full_items.json", "w") as f:
        json.dump(items, f, indent=2)
    print(f"response matrix: {R.shape[0]} LLMs x {R.shape[1]} items, mean accuracy {R.mean():.4f} -> {a.out}", flush=True)

    if a.embeddings:
        from sentence_transformers import SentenceTransformer

        sbert = SentenceTransformer("all-mpnet-base-v2")
        emb = sbert.encode(embed_texts, show_progress_bar=True, normalize_embeddings=True, batch_size=256)
        np.savez(a.out / "item_text_embeddings_v2_full.npz", embeddings=emb)
        print(f"item text embeddings: {emb.shape} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
