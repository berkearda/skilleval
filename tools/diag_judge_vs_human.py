#!/usr/bin/env python3
"""T-114: does the coherence judge disagree with the human on the SAME decisions?

The 29-point gap (judge 65.7% item-weighted against human 95%) compares two
different sets of questions. The judge scored up to 10 items per skill across 234
skills; the human scored 42 specific assignments. Part of that gap could be which
questions each one saw rather than how strict each one is, and no aggregate
difference can separate the two.

This asks the judge about exactly the 42 assignments the human judged, with the
same prompt, the same model, and the same batch sizes, then compares them verdict
by verdict. A paired comparison can attribute the gap; two aggregates cannot.

WHAT IS REPRODUCED, AND WHAT IS NOT. Batch SIZE is reproduced: coherence judges up
to 10 questions per skill in a single call, and judging a lone question is a
different task, so each gold item is placed inside a batch of min(10, skill size)
questions drawn from its own skill. Batch COMPOSITION cannot be reproduced,
because the original sample was seeded with `hash(code)` while PYTHONHASHSEED was
unset, so it differed on every run and is unrecoverable (T-115). Batches here use
the repaired deterministic seed.

`verify_splits` does not apply: there is no train/test split here, only a fixed
set of 42 human-labelled assignments, so the split_info logged below records that
rather than inventing one.

    python3 tools/diag_judge_vs_human.py --smoke
    python3 tools/diag_judge_vs_human.py
"""
import argparse, hashlib, json, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.gemini import Gemini, GeminiError, JUDGE
from tools.textclip import clip
from tools.metrics import row_codes
from tools.gold_score_partb import parse_sheet
from tools.step6_validate import COH_SYS, COH_U     # the identical prompt, not a copy
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment

P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
WORKERS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="gold/gold_sheet_2_labelled.md")
    ap.add_argument("--key", default="gold/gold_partB_key_2.json")
    ap.add_argument("--codebook", default="codebook_v2_amended_b150.json")
    ap.add_argument("--labels", default="item_labels_b150.jsonl")
    ap.add_argument("--stored", default="validation_coherence_v2_amended_b150.json",
                    help="the original coherence run, for a replication sanity check")
    ap.add_argument("--sample", type=int, default=10, help="batch size cap, as Step 6 uses")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="cdm_exploration/experiments/diag_judge_vs_human.json")
    a = ap.parse_args()
    seed_everything(42)

    cb = json.loads((P / a.codebook).read_text())
    alias = cb.get("alias", {})
    codes = cb["codes"]
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    by_code = defaultdict(list)
    for line in (P / a.labels).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        for c in row_codes(r, alias):
            if c in codes:
                by_code[c].append(r["item_idx"])

    key = json.loads((REPO / a.key).read_text())["by_item"]
    ticks, _, anchored = parse_sheet(REPO / a.sheet)
    gold = [(int(i), c, c in ticks.get(i, ()), i in anchored)
            for i, v in key.items() for c in v["assigned"]]
    by_skill = defaultdict(list)
    for it, c, human, anch in gold:
        by_skill[c].append((it, human, anch))
    skills = sorted(by_skill)
    if a.smoke:
        skills = skills[:4]
    print(f"paired judge-vs-human on {sum(len(by_skill[c]) for c in skills)} assignments "
          f"over {len(skills)} skills, judge {JUDGE}")

    def one(c):
        gold_items = [it for it, _, _ in by_skill[c]]
        pool = [i for i in by_code.get(c, []) if i not in gold_items]
        seed = 42 + int(hashlib.sha256(c.encode()).hexdigest()[:8], 16)
        random.Random(seed).shuffle(pool)
        batch = gold_items + pool[:max(0, min(a.sample, len(by_code.get(c, []))) - len(gold_items))]
        random.Random(seed + 1).shuffle(batch)      # so the gold item is not always first
        qs = "\n".join(f"{k+1}. {clip(txt[j], 4000)}" for k, j in enumerate(batch))
        g = Gemini()
        try:
            obj = g.json_obj(COH_SYS, COH_U.format(definition=codes[c]["definition"],
                                                   questions=qs),
                             model=JUDGE, max_out=8000)
        except GeminiError as e:
            return {"code": c, "error": str(e)[:120], "batch": len(batch)}
        seen = {}
        for v in (obj.get("verdicts") or []):
            try:
                k = int(v.get("i")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(batch):
                seen[batch[k]] = bool(v.get("match"))
        return {"code": c, "batch": len(batch), "verdicts": seen,
                "batch_precision": (sum(seen.values()) / len(seen)) if seen else None}

    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(one, skills), 1):
            res.append(r)
            if k % 10 == 0 or k == len(skills):
                print(f"  {k}/{len(skills)} skills | {time.time()-t0:.0f}s", flush=True)

    verdict = {}
    for r in res:
        for it, m in (r.get("verdicts") or {}).items():
            verdict[(int(it), r["code"])] = m
    paired = [(it, c, human, anch, verdict[(it, c)])
              for it, c, human, anch in gold if (it, c) in verdict]
    n_err = sum(1 for r in res if "error" in r)
    missing = sum(1 for it, c, _, _ in gold
                  if c in set(skills) and (it, c) not in verdict)

    cell = Counter((h, j) for _, _, h, _, j in paired)
    hj = cell[(True, True)]; hn = cell[(True, False)]
    nj = cell[(False, True)]; nn = cell[(False, False)]
    n = len(paired)
    print(f"\npaired on {n} assignments ({n_err} skill calls errored, {missing} verdicts "
          f"not returned)")
    print(f"\n                     judge accepts   judge rejects")
    print(f"  human accepts  {hj:>12}  {hn:>15}")
    print(f"  human rejects  {nj:>12}  {nn:>15}")
    if n:
        print(f"\n  human acceptance on these: {(hj+hn)}/{n} = {(hj+hn)/n:.0%}")
        print(f"  judge acceptance on these: {(hj+nj)}/{n} = {(hj+nj)/n:.0%}")
        print(f"  agreement: {(hj+nn)}/{n} = {(hj+nn)/n:.0%}")
        print(f"  discordant: judge stricter {hn}, human stricter {nj}")
        print(f"\n  The stored run's item-weighted rate was 65.7%. If the judge's rate")
        print(f"  here is close to that, the gap is strictness on identical decisions.")
        print(f"  If it is close to the human's, the gap was composition.")

    st = {}
    f = P / a.stored
    if f.exists():
        st = {r["code"]: r["precision"] for r in json.loads(f.read_text())["results"]
              if r.get("precision") is not None}
    rep = [(r["code"], r["batch_precision"], st[r["code"]])
           for r in res if r.get("batch_precision") is not None and r["code"] in st]
    if rep:
        d = [abs(b - s) for _, b, s in rep]
        print(f"\nreplication check against the stored run, {len(rep)} skills scored both ways:")
        print(f"  mean |this run - stored| = {sum(d)/len(d):.3f}")
        print(f"  (a large value is expected in part: T-115 means the stored run's")
        print(f"   question sample cannot be reproduced, only its batch size)")

    strict = [(it, c) for it, c, h, _, j in paired if h and not j]
    if strict:
        print(f"\nassignments the human accepted and the judge rejected ({len(strict)}):")
        for it, c in strict[:12]:
            print(f"  item {it}  {c}  {codes[c]['name']}  "
                  f"({len(by_code.get(c, []))} questions)")

    out = {"judge": JUDGE, "sample": a.sample, "sheet": a.sheet,
           "codebook": a.codebook, "labels": a.labels,
           "batch_composition_reproducible": False, "see": "T-115",
           "n_paired": n, "human_accept": hj + hn, "judge_accept": hj + nj,
           "agreement": hj + nn, "judge_stricter": hn, "human_stricter": nj,
           "cells": {"both_accept": hj, "human_only": hn, "judge_only": nj,
                     "both_reject": nn},
           "anchored_items": sorted(anchored),
           "per_assignment": [{"item_idx": it, "code": c, "human": h,
                               "anchored": anch, "judge": j}
                              for it, c, h, anch, j in paired],
           "errors": [r for r in res if "error" in r]}
    if not a.smoke:
        (REPO / a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
        log_experiment(
            name="diag_judge_vs_human",
            config={"judge": JUDGE, "batch_cap": a.sample, "seed": 42,
                    "codebook": a.codebook, "labels": a.labels, "sheet": a.sheet},
            results={"n_paired": n, "human_acceptance": (hj + hn) / n if n else None,
                     "judge_acceptance": (hj + nj) / n if n else None,
                     "agreement": (hj + nn) / n if n else None,
                     "judge_stricter": hn, "human_stricter": nj},
            split_info={"note": "no train/test split applies; the unit is the fixed "
                                "set of human-labelled assignments",
                        "n_assignments": n, "n_skills": len(skills)},
            verified=(n_err == 0 and missing == 0),
        )


if __name__ == "__main__":
    main()
