#!/usr/bin/env python3
"""Build a multiple-choice labelling sheet. No free text anywhere.

Open-ended labelling was rejected as too slow and too hard, which is fair: it is
also harder to score. Everything here is one keystroke per item.

Two tasks, both pure choice:

TASK 1 - which definition is right (settles the Step 8 decision).
  Step 8 rewrote 265 definitions. A judge said 95 codes improved and 70 got
  worse, and nothing arbitrates that. For each question you see the SAME code's
  old and new definition, unlabelled and in random order, and pick which better
  describes what the question requires. This is the doc's rule made operational:
  "any pipeline change has to improve on this set to be kept".

TASK 2 - is this skill actually required (calibrates the judge).
  A yes/no on question-and-skill pairs. Comparing your answers with the judge's
  gives an agreement rate, which is what lets the judge's verdicts be used at
  scale with an honest number attached rather than on trust. Includes decoys as
  a null control.

Sampling for Task 1 is deliberately targeted at codes where the two definitions
most disagree. Questions both versions agree on carry no information about which
version is better, so labelling them is wasted effort.

    python3 tools/gold_choice.py --pairs 24 --checks 20
"""
import argparse, json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.textclip import clip
P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
GOLD = REPO / "gold"
SEED = 20260908
QMAX = 1200
SKIP = {"GPQA"}      # graduate-level science; excluded and the exclusion recorded

HEAD = """# Skill labelling — multiple choice only

Two tasks. Every answer is one letter. No writing.

Budget: about 20 minutes total.

## What counts as a skill

The smallest named mental operation a solver must perform. Judge by the
**operation**, never by the subject matter.

| | |
|---|---|
| right level | apply the triangle inequality |
| too coarse | geometry |
| too fine | apply the triangle inequality to one specific stick problem |
| wrong split | "track dance-partner swaps" and "track gift swaps" as two skills |

The last row is a real failure from an earlier run: same task, different cover
story, so it needs one skill and not two.

**GPQA questions are excluded** from this sheet. They are graduate-level
physics, chemistry and biology, and labelling them needs domain expertise rather
than care. That exclusion is recorded rather than silently applied.

---

## TASK 1 — which definition fits better?

{n1} questions. Each shows one question and two candidate definitions of the
same skill. Write `A`, `B`, `both` or `neither` after `answer:`.

- `A` or `B` — that one describes what the question actually requires
- `both` — they are equally good, or the difference does not matter here
- `neither` — neither describes what this question requires

The two are the old and the rewritten definition, in random order. You are not
told which is which, and neither am I when scoring.

---
"""

MID = """
---

## TASK 2 — is this skill required?

{n2} pairs. Write `y` or `n` after `answer:`.

`y` only if a solver genuinely must perform that operation to answer. Not if it
is merely related, on the same topic, or plausible. Some entries are wrong on
purpose.

---
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--checks", type=int, default=20)
    a = ap.parse_args()

    fz = json.loads((P / "codebook_v4_definitions.json").read_text())
    codes = fz["codes"]
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {p["item_idx"]: p for p in
            json.load(open(REPO / "cdm_exploration/experiments/oldtax_repaired_FINAL.json"))["per_item"]}
    before = {r["code"]: r for r in json.loads((P / "validation_coherence_holdout_before.json").read_text())["results"] if r.get("precision") is not None}
    after = {r["code"]: r for r in json.loads((P / "validation_coherence_holdout_after.json").read_text())["results"] if r.get("precision") is not None}

    rng = random.Random(SEED)
    # Task 1: codes whose two definitions most disagree, alternating the
    # direction so the sheet is not all cases the judge liked
    cand = [c for c in set(before) & set(after)
            if c in codes and "definition_before" in codes[c]
            and abs(after[c]["precision"] - before[c]["precision"]) >= 0.4]
    gains = sorted([c for c in cand if after[c]["precision"] > before[c]["precision"]],
                   key=lambda c: -(after[c]["precision"] - before[c]["precision"]))
    losses = sorted([c for c in cand if after[c]["precision"] < before[c]["precision"]],
                    key=lambda c: -(before[c]["precision"] - after[c]["precision"]))
    picked, i = [], 0
    while len(picked) < a.pairs and (i < len(gains) or i < len(losses)):
        for src in (gains, losses):
            if i < len(src) and len(picked) < a.pairs:
                picked.append(src[i])
        i += 1

    out, key1 = [HEAD.format(n1=len(picked))], {}
    n = 0
    for c in picked:
        hold = [j for j in codes[c].get("holdout_items", [])
                if j in txt and meta.get(j, {}).get("benchmark") not in SKIP]
        if not hold:
            continue
        j = rng.choice(hold)
        n += 1
        old, new = codes[c]["definition_before"], codes[c]["definition"]
        flip = rng.random() < 0.5
        A, B = (new, old) if flip else (old, new)
        key1[f"T1-{n:02d}"] = {"code": c, "item_idx": j, "A": "new" if flip else "old",
                               "B": "old" if flip else "new",
                               "judge_preferred": "new" if after[c]["precision"] > before[c]["precision"] else "old"}
        out.append(f"### T1-{n:02d} · {meta[j]['benchmark']}\n")
        out.append(f"> {clip(txt[j], QMAX)}\n")
        out.append(f"**A)** {A}\n")
        out.append(f"**B)** {B}\n")
        out.append("```\nanswer: \n```\n")
        out.append("---\n")

    # Task 2: real assignments plus decoys, shuffled
    rows = [json.loads(l) for l in (P / "item_labels.jsonl").open()]
    pool = [r for r in rows if "error" not in r and r.get("assigned")
            and meta.get(r["item_idx"], {}).get("benchmark") not in SKIP]
    rng.shuffle(pool)
    checks, key2, m = [], {}, 0
    all_codes = sorted(codes)
    for r in pool:
        if m >= a.checks:
            break
        j = r["item_idx"]
        real = rng.random() < 0.6
        c = rng.choice([x["code"] for x in r["assigned"]]) if real else rng.choice(all_codes)
        if not real and c in {x["code"] for x in r["assigned"]}:
            continue
        if c not in codes:
            continue
        m += 1
        key2[f"T2-{m:02d}"] = {"item_idx": j, "code": c, "is_pipeline_assignment": real}
        checks.append((m, j, c))
    out.append(MID.format(n2=len(checks)))
    for m_, j, c in checks:
        out.append(f"### T2-{m_:02d} · {meta[j]['benchmark']}\n")
        out.append(f"> {clip(txt[j], QMAX)}\n")
        out.append(f"**Skill:** {codes[c]['name']} — {codes[c]['definition']}\n")
        out.append("```\nanswer: \n```\n")
        out.append("---\n")

    (GOLD / "choice_sheet.md").write_text("\n".join(out))
    (GOLD / "choice_key.json").write_text(json.dumps(
        {"note": "answer key. Do not open before labelling.",
         "seed": SEED, "excluded_benchmarks": sorted(SKIP),
         "task1": key1, "task2": key2}, indent=1))
    dec = sum(1 for v in key2.values() if not v["is_pipeline_assignment"])
    print(f"wrote gold/choice_sheet.md")
    print(f"  Task 1: {n} definition pairs (drawn from the {len(cand)} codes whose "
          f"two definitions most disagree)")
    print(f"  Task 2: {len(checks)} yes/no checks, {dec} of them decoys")
    print(f"  excluded: {sorted(SKIP)}")


if __name__ == "__main__":
    main()
