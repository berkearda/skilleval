#!/usr/bin/env python3
"""Render a frozen gold set as a hand-labelling sheet.

Two parts per question, deliberately in this order.

PART A is free text, written before you see any candidate skill. It is the
version-independent record: it can be scored against any future codebook,
including ones that do not exist yet. If you saw the pipeline's answer first you
would anchor on it, and the gold set would agree with the pipeline by
construction.

PART B is a checklist of candidate skills, to be filled only after Part A. It
gives a precision number that needs no model at scoring time. Some candidates in
each list are DECOYS drawn from unrelated codes. They are a null control: if
decoys get ticked at a similar rate to real candidates, the exercise is measuring
agreeableness rather than skill.

Every output path is an argument. They were constants, so rendering a second
sheet overwrote the first sheet and its answer key, which are the only record of
what a human was actually shown. A sheet whose options no longer match its key
cannot be scored at all.

    python3 tools/gold_sheet.py                        # the reference sheet
    python3 tools/gold_sheet.py --set gold/gold_set_2.json \
        --out gold/gold_sheet_2.md --key gold/gold_partB_key_2.json \
        --codebook codebook_v2_amended_b150.json --labels item_labels_b150.jsonl
"""
import argparse, json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.textclip import clip

P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
GOLD = REPO / "gold"
OPTIONS = 6              # CONSTANT. Was `assigned + 2 decoys`, so the length of the
                         # list told the labeller exactly how many to tick: 3 options
                         # meant 1 real, 5 meant 3. That is the same leak Berke caught
                         # in the first Task 2 sheet, and it voids the control.
QMAX = 1400

RULES = """## What a "skill" means here

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

Do not name a code id in Part A. On the first sheet 22 of 100 Part A answers
cited one, which proves Part B had already been read for those questions and
costs Part A its independence.

**Part B — only after Part A is written for that question.**

Tick `[x]` for each listed skill that the solver genuinely must perform.
Leave unticked anything that is merely related, on the same topic, or plausible
but not actually required. Some entries are deliberately wrong.
"""


def head(gs, setname, out, why, minutes):
    return f"""# {gs['n']} hand-labelled questions

**Do not regenerate this file.** The {gs['n']} questions are frozen in
`{setname}` under seed {gs['seed']}. Every score measured against this set,
before and after any pipeline change, has to be measured on the same questions
or the comparison means nothing. Regenerating also rewrites the answer key, and
then nothing can be scored.

{why}

{RULES}
## Pacing

About 60-90 seconds per question, so roughly {minutes}. It does not have to be
one sitting: the file is plain markdown and partial progress is fine. Save your
answers in this file. When you are done, run

```
python3 tools/gold_score_partb.py --sheet {out} --key {{key}} \\
    --codebook {{cb}} --labels {{lb}}
```

---
"""


WHY_REF = """## Why you are doing this

The design doc ranks it first, above everything else:

> Build the gold set (200 hand-labeled questions). This single step converts
> "we keep changing things and it's a mess" into "we're changing things in a
> direction." Best return on effort by a wide margin.

and gives the rule it exists to enforce:

> Any pipeline change has to improve on this set to be kept.

Right now there is no such reference. The last change I made rewrote 265 skill
definitions: 95 codes got better, 70 got worse, and I have no principled way to
decide whether to keep it, revert it, or apply it selectively. That decision is
what this set unblocks."""

WHY_FLOOR = """## Why you are doing this

One question decides the shape of the taxonomy: should skills holding fewer than
20 questions be dropped or merged into bigger ones? Right now that call rests on
an automated judge, which says small skills are much worse (53% against 75%).
Your 100 labels say the gap is 5 points, not 22, but they cannot settle it: only
10 of their 145 assignments came from small skills, because that set was drawn to
cover benchmarks evenly and small skills appear in few questions by construction.

This set is drawn to fix exactly that. Half the questions are held by small
skills and half by large ones, so the comparison has enough data to come out one
way or the other. The number that matters is the difference between those two
halves, not the overall percentage.

It is also a different taxonomy from the one you labelled before. The skill list
was rebuilt (274 skills, down from 1,332) and the ids are not comparable, so
treat every entry as new rather than trying to remember your earlier answer."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="setfile", default="gold/gold_set.json",
                    help="the frozen sample to render")
    ap.add_argument("--out", default="gold/gold_sheet.md")
    ap.add_argument("--key", default="gold/gold_partB_key.json",
                    help="answer key; never overwrite one that has been labelled against")
    ap.add_argument("--codebook", default="codebook_v9_double_judged.json",
                    help="the taxonomy being tested; must match --labels")
    ap.add_argument("--labels", default="item_labels_codebook_v9_double_judged.jsonl")
    ap.add_argument("--why", choices=["reference", "floor"], default="reference",
                    help="which rationale to print in the header")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing sheet or key")
    a = ap.parse_args()

    out_f, key_f = REPO / a.out, REPO / a.key
    for f in (out_f, key_f):
        if f.exists() and not a.force:
            raise SystemExit(
                f"{f.relative_to(REPO)} already exists. Overwriting a sheet or key that "
                f"has been labelled against destroys the only record of what was shown. "
                f"Pass --force if you are certain, or choose another --out/--key.")

    gs = json.loads((REPO / a.setfile).read_text())
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    fz = json.loads((P / a.codebook).read_text())
    codes = fz["codes"]
    rows = {json.loads(l)["item_idx"]: json.loads(l)
            for l in (P / a.labels).open() if l.strip()}
    rng = random.Random(gs["seed"])
    all_ids = sorted(c for c in codes if c not in fz.get("alias", {}))

    minutes = f"{max(1, round(gs['n'] * 75 / 3600))} hour{'s' if gs['n'] * 75 > 5400 else ''}" \
        if gs["n"] * 75 >= 3000 else f"{round(gs['n'] * 75 / 60)} minutes"
    why = WHY_REF if a.why == "reference" else WHY_FLOOR
    out = [head(gs, a.setfile, a.out, why, minutes)
           .replace("{key}", a.key).replace("{cb}", a.codebook).replace("{lb}", a.labels)]
    key, trimmed, leaky = {}, 0, 0
    for n, it in enumerate(gs["items"], 1):
        i = it["item_idx"]
        q = txt[i]
        r = rows.get(i, {})
        # dedupe: the labeller names the same code twice in 2,139 rows, which put
        # the identical option in the list twice on 18 of the 100 questions
        assigned = sorted({x["code"] for x in r.get("assigned", []) if x["code"] in codes})
        if len(assigned) >= OPTIONS:
            # every list must keep at least one decoy or it has no null control
            assigned, trimmed = assigned[:OPTIONS - 1], trimmed + 1
        # A decoy has to be a skill the question does NOT need. Drawing from every
        # live id lets a genuine near-miss be served as a decoy: on sheet 2, item
        # 528 offered c_0148 as a decoy and it was the second half of the actual
        # solution path, so ticking it was correct and was scored as a false
        # positive. gold_task2.py already bans the item's retrieved candidates
        # for this reason, and this did not.
        banned = set(assigned) | set(r.get("candidates") or [])
        pool = [c for c in all_ids if c not in banned]
        if len(pool) < OPTIONS - len(assigned):
            pool = [c for c in all_ids if c not in assigned]
            leaky += 1
        decoys = []
        while len(assigned) + len(decoys) < OPTIONS:
            c = rng.choice(pool)
            if c not in decoys:
                decoys.append(c)
        shown = assigned + decoys
        rng.shuffle(shown)   # so position carries no information
        key[str(i)] = {"assigned": assigned, "decoys": decoys, "shown": shown}

        out.append(f"### Q{n:03d} · item {i} · {it['benchmark']} / {it.get('subtask') or '-'}\n")
        out.append(f"> {clip(q, QMAX)}\n")
        out.append("**Part A** — write before reading Part B\n")
        out.append("```\nn_operations: \nop_1: \nop_2: \nop_3: \nnotes: \n```\n")
        out.append("**Part B** — only after Part A\n")
        for c in shown:
            out.append(f"- [ ] `{c}` **{codes[c]['name']}** — {codes[c]['definition']}")
        out.append("\n---\n")

    out_f.write_text("\n".join(out))
    key_f.write_text(json.dumps(
        {"note": "which Part B entries were the pipeline's and which were decoys. "
                 "Do not read before labelling; it is the answer key.",
         "sheet": a.out, "set": a.setfile,
         "codebook": a.codebook, "labels": a.labels, "by_item": key}, indent=1))
    n_dec = sum(len(v["decoys"]) for v in key.values())
    n_asg = sum(len(v["assigned"]) for v in key.values())
    print(f"wrote {a.out}  ({gs['n']} questions, {a.codebook})")
    print(f"  Part B entries: {n_asg} from the pipeline + {n_dec} decoys")
    if trimmed:
        print(f"  {trimmed} questions had >={OPTIONS} assignments, trimmed to keep a decoy")
    if leaky:
        print(f"  {leaky} questions had too few non-candidate skills to draw clean "
              f"decoys; their decoys may include a skill the question needs")
    print(f"  answer key (do not open): {a.key}")


if __name__ == "__main__":
    main()
