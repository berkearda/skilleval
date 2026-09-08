#!/usr/bin/env python3
"""Draw the gold set and write a hand-labelling sheet.

The doc ranks this first: "Build the gold set (200 hand-labeled questions). This
single step converts 'we keep changing things and it's a mess' into 'we're
changing things in a direction.' Best return on effort by a wide margin." And in
Step 6: "Hand-label 100-200 questions and leave it fixed. Any pipeline change has
to improve on this set to be kept."

Two properties this sampler has to get right.

FROZEN. The sample is drawn once under a fixed seed and written to
gold/gold_set.json. It must never be redrawn, or every score before and after
is measured on a different set and the comparison is worthless.

INDEPENDENT OF THE PIPELINE. Selection uses only benchmark, subtask and question
length. It does not look at what the pipeline assigned, how many skills it
assigned, or how confident it was. A gold set stratified by the thing it is
meant to judge would agree with the pipeline by construction.

Sampling is proportional to the corpus with a floor per benchmark, because
strict proportion gives 60 BBH and 6 IFEval, and 6 questions cannot say anything
about the benchmark where coverage is known to be weakest. Both the sample
weight and the corpus weight are recorded so a score can be reported either
per-question or re-weighted to the corpus.

    python3 tools/gold_sample.py --n 100
"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
GOLD = REPO / "gold"
SEED = 20260908
FLOOR = 12          # minimum questions per benchmark
DECOYS = 2          # unrelated skills mixed into part B as a null control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="redraw an existing frozen set")
    a = ap.parse_args()
    GOLD.mkdir(exist_ok=True)
    frozen = GOLD / "gold_set.json"
    if frozen.exists() and not a.force:
        raise SystemExit(f"{frozen} already exists. The set is frozen by design; "
                         f"pass --force only if you truly mean to invalidate every "
                         f"score measured against it.")

    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {p["item_idx"]: p for p in
            json.load(open(REPO / "cdm_exploration/experiments/oldtax_repaired_FINAL.json"))["per_item"]}
    items = [i for i in sorted(txt) if i in meta]
    by_bench = defaultdict(list)
    for i in items:
        by_bench[meta[i]["benchmark"]].append(i)
    corpus = {b: len(v) for b, v in by_bench.items()}
    total = sum(corpus.values())

    # floor first, then the remainder proportional to the corpus
    quota = {b: min(FLOOR, len(v)) for b, v in by_bench.items()}
    left = a.n - sum(quota.values())
    order = sorted(corpus, key=lambda b: -corpus[b])
    for b in order:
        quota[b] += int(left * corpus[b] / total)
    while sum(quota.values()) < a.n:
        quota[order[0]] += 1

    rng = random.Random(SEED)
    chosen = []
    for b in order:
        # spread across subtasks, then across question length, so a benchmark is
        # not represented by one subtask or only by its short questions
        pool = by_bench[b]
        subs = defaultdict(list)
        for i in pool:
            subs[meta[i].get("subtask") or "-"].append(i)
        for k in subs:
            rng.shuffle(subs[k])
            subs[k].sort(key=lambda i: len(txt[i]))          # short..long
        keys, picked, r = sorted(subs), [], 0
        while len(picked) < quota[b]:
            k = keys[r % len(keys)]
            band = subs[k]
            if band:
                # walk the length bands round-robin as well
                picked.append(band.pop(len(band) * (r // len(keys) * 37 % 100) // 100))
            r += 1
            if r > 100000:
                break
        chosen += picked[:quota[b]]

    chosen = sorted(set(chosen))
    blob = {"seed": SEED, "n": len(chosen), "floor_per_benchmark": FLOOR,
            "drawn": "2026-09-08",
            "selection": "benchmark + subtask + question length only; nothing from the pipeline",
            "quota": quota, "corpus_counts": corpus,
            "corpus_weight": {b: corpus[b] / total for b in corpus},
            "sample_weight": {b: quota[b] / len(chosen) for b in quota},
            "items": [{"item_idx": i, "benchmark": meta[i]["benchmark"],
                       "subtask": meta[i].get("subtask")} for i in chosen]}
    frozen.write_text(json.dumps(blob, indent=1))
    print(f"froze {len(chosen)} questions -> {frozen.relative_to(REPO)}")
    for b in order:
        print(f"  {b:8s} {quota[b]:3d}  (corpus {corpus[b]/total:5.1%}, sample {quota[b]/len(chosen):5.1%})")
    print("\nnow run: python3 tools/gold_sheet.py")


if __name__ == "__main__":
    main()
