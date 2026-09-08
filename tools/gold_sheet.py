#!/usr/bin/env python3
"""Render the frozen gold set as a hand-labelling sheet.

Two parts per question, deliberately in this order.

PART A is free text, written before you see any candidate skill. It is the
version-independent record: it can be scored against any future codebook,
including ones that do not exist yet. If you saw the pipeline's answer first you
would anchor on it, and the gold set would agree with the pipeline by
construction.

PART B is a checklist of candidate skills, to be filled only after Part A. It
gives a precision number that needs no model at scoring time. Two of the
candidates in each list are DECOYS drawn from unrelated codes. They are a null
control: if decoys get ticked at a similar rate to real candidates, the exercise
is measuring agreeableness rather than skill.

    python3 tools/gold_sheet.py
"""
import json, random, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.textclip import clip
P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
GOLD = REPO / "gold"
DECOYS = 2
QMAX = 1400

HEAD = """# Gold set — 100 hand-labelled questions

**Do not regenerate this file.** The 100 questions are frozen in
`gold/gold_set.json` under seed 20260908. Every score measured against this set,
before and after any pipeline change, has to be measured on the same questions
or the comparison means nothing.

## Why you are doing this

The design doc ranks it first, above everything else:

> Build the gold set (200 hand-labeled questions). This single step converts
> "we keep changing things and it's a mess" into "we're changing things in a
> direction." Best return on effort by a wide margin.

and gives the rule it exists to enforce:

> Any pipeline change has to improve on this set to be kept.

Right now there is no such reference. The last change I made rewrote 265 skill
definitions: 95 codes got better, 70 got worse, and I have no principled way to
decide whether to keep it, revert it, or apply it selectively. That decision is
what this set unblocks.

## What a "skill" means here

The smallest named mental operation a solver must perform to answer correctly.
Judge by the **operation**, never by the subject matter.

| | |
|---|---|
| right level | apply the triangle inequality |
| too coarse | geometry |
| too fine | apply the triangle inequality to one specific stick problem |
| wrong split | "track dance-partner swaps" and "track gift swaps" as two skills |

That last row is a real failure from an earlier run. Both are the same BBH task
with a different cover story, and the solver does the same thing in each: follow
a list of swaps and work out where everything ended up. If two questions differ
only in what they are *about*, they need the same skill.

## How to fill it in

For each of the 100 blocks:

**Part A — write this first, before scrolling to Part B.**

- `n_operations:` how many distinct operations answering this genuinely
  requires. Usually 1 or 2. Write 0 if none of it is a nameable skill.
- `op_1:` / `op_2:` / `op_3:` name each one, lowercase, 3-8 words, verb-first.
  For example `track object positions through swaps`, not `object tracking` and
  not `tracking five objects through three swaps in a dance`.
- `notes:` anything ambiguous. Worth writing when you hesitate.

Do not aim for a particular number of skills per question. Single-skill
questions are valuable: they are the only ones that say cleanly which skill a
model is missing.

**Part B — only after Part A is written for that question.**

Tick `[x]` for each listed skill that the solver genuinely must perform.
Leave unticked anything that is merely related, on the same topic, or plausible
but not actually required. Some entries are deliberately wrong.

## Pacing

About 60-90 seconds per question, so roughly two hours. It does not have to be
one sitting: the file is plain markdown and partial progress is fine. When you
are done, run `python3 tools/gold_score.py` and it will parse this file.

---
"""


def main():
    gs = json.loads((GOLD / "gold_set.json").read_text())
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    fz = json.loads((P / "codebook_v4_definitions.json").read_text())
    codes = fz["codes"]
    rows = {json.loads(l)["item_idx"]: json.loads(l)
            for l in (P / "item_labels.jsonl").open()}
    rng = random.Random(gs["seed"])
    all_ids = sorted(codes)

    out = [HEAD]
    key = {}
    for n, it in enumerate(gs["items"], 1):
        i = it["item_idx"]
        q = txt[i]
        trunc = len(q) > QMAX
        r = rows.get(i, {})
        assigned = [x["code"] for x in r.get("assigned", []) if x["code"] in codes]
        decoys = []
        while len(decoys) < DECOYS:
            c = rng.choice(all_ids)
            if c not in assigned and c not in decoys:
                decoys.append(c)
        shown = assigned + decoys
        rng.shuffle(shown)
        key[str(i)] = {"assigned": assigned, "decoys": decoys, "shown": shown}

        out.append(f"### Q{n:03d} · item {i} · {it['benchmark']} / {it.get('subtask') or '-'}\n")
        out.append(f"> {clip(q, QMAX)}\n")
        out.append("**Part A** — write before reading Part B\n")
        out.append("```\nn_operations: \nop_1: \nop_2: \nop_3: \nnotes: \n```\n")
        out.append("**Part B** — only after Part A\n")
        for c in shown:
            out.append(f"- [ ] `{c}` **{codes[c]['name']}** — {codes[c]['definition']}")
        out.append("\n---\n")

    (GOLD / "gold_sheet.md").write_text("\n".join(out))
    (GOLD / "gold_partB_key.json").write_text(json.dumps(
        {"note": "which Part B entries were the pipeline's and which were decoys. "
                 "Do not read before labelling; it is the answer key.",
         "by_item": key}, indent=1))
    n_dec = sum(len(v["decoys"]) for v in key.values())
    n_asg = sum(len(v["assigned"]) for v in key.values())
    print(f"wrote gold/gold_sheet.md  ({gs['n']} questions)")
    print(f"  Part B entries: {n_asg} from the pipeline + {n_dec} decoys")
    print(f"  answer key (do not open): gold/gold_partB_key.json")


if __name__ == "__main__":
    main()
