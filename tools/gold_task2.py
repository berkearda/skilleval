#!/usr/bin/env python3
"""Rebuild Task 2 as a proper null control.

The first version sampled its composition with rng.random() < 0.6 instead of
fixing it, and never constrained ordering. It came out 12 decoys of 20 with 7 of
the last 8 consecutive, which the labeller noticed. A null control the subject
can detect stops being a control: expectations on the later items differ from
the earlier ones, and the acceptance rate is measured under changed conditions.

Three properties this version guarantees rather than samples:

  BALANCED     exactly half real, half decoy
  INTERLEAVED  no run of the same kind longer than 2
  FRESH        no item reused from the previous sheet, since remembering an
               earlier answer is its own contamination

Decoys are drawn from codes that are NOT in the item's own retrieval candidates,
so a "decoy" cannot accidentally be a skill the question genuinely needs. That
makes them clearly unrelated, which means a high rejection rate is a floor check
(are the answers discriminating at all) rather than a hard test.

    python3 tools/gold_task2.py --n 20
"""
import argparse, json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.textclip import clip

P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
GOLD = REPO / "gold"
SEED = 20260908 + 2
QMAX = 1200
SKIP = {"GPQA"}

HEAD = """# Task 2 (clean rerun) — is this skill required?

{n} items. Write `y` or `n` after `answer:`. About 8 minutes.

`y` only if a solver genuinely must perform that operation to answer the
question. Not if it is merely related, on the same topic, or plausible-sounding.

## Why you are doing this again

The first version had a flaw of mine. It sampled how many entries were wrong
instead of fixing it, and did not constrain their order, so it came out 12 wrong
of 20 with 7 of the last 8 in a row. You noticed, which means your expectations
on the later items were not the ones you started with, and the number I wanted
from it is not measurable under those conditions.

This version is exactly half real and half wrong, no more than two of a kind in
a row, and uses questions that did not appear in the previous sheet. You still
should not be able to tell which is which, and this time that is guaranteed by
construction rather than left to chance.

What it measures: how often you and the LLM judge agree. Every Step 6 number in
this project comes from that judge, and right now they are reported on trust.
With this, they can be reported with an agreement rate attached.

---
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    assert a.n % 2 == 0, "n must be even for an exact half-and-half split"

    fz = json.loads((P / "codebook_v4_definitions.json").read_text())
    codes = fz["codes"]
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {p["item_idx"]: p for p in
            json.load(open(REPO / "cdm_exploration/experiments/oldtax_repaired_FINAL.json"))["per_item"]}
    rows = [json.loads(l) for l in (P / "item_labels.jsonl").open()]

    used = set()
    old_key = GOLD / "choice_key.json"
    if old_key.exists():
        k = json.loads(old_key.read_text())
        used = {v["item_idx"] for v in k.get("task2", {}).values()} | \
               {v["item_idx"] for v in k.get("task1", {}).values()}

    pool = [r for r in rows if "error" not in r and r.get("assigned")
            and r["item_idx"] not in used
            and meta.get(r["item_idx"], {}).get("benchmark") not in SKIP
            and r["item_idx"] in txt]
    rng = random.Random(SEED)
    rng.shuffle(pool)

    half = a.n // 2
    all_codes = sorted(codes)
    reals, decoys = [], []
    for r in pool:
        j = r["item_idx"]
        assigned = [x["code"] for x in r["assigned"] if x["code"] in codes]
        if not assigned:
            continue
        if len(reals) < half:
            reals.append((j, rng.choice(assigned), True))
            continue
        if len(decoys) < half:
            # never draw a decoy from this item's own candidates: a near-miss
            # could genuinely apply, which would make it a false decoy
            banned = set(r.get("candidates", [])) | set(assigned)
            c = rng.choice([x for x in all_codes if x not in banned])
            decoys.append((j, c, False))
        if len(reals) == half and len(decoys) == half:
            break

    # Shuffle under a run-length constraint, by rejection. Strict alternation
    # would satisfy "no long runs" and be far more detectable than the bug it
    # replaces: after three items the pattern is obvious. What is wanted is a
    # random order that merely never runs long.
    def longest_run(order):
        best = cur = 1
        for x, y in zip(order, order[1:]):
            cur = cur + 1 if x[2] == y[2] else 1
            best = max(best, cur)
        return best

    seq = reals + decoys
    for _ in range(10000):
        rng.shuffle(seq)
        if longest_run(seq) <= 2:
            break
    else:
        raise SystemExit("could not find an ordering with no run longer than 2")

    out, key = [HEAD.format(n=len(seq))], {}
    for i, (j, c, real) in enumerate(seq, 1):
        tid = f"T2b-{i:02d}"
        key[tid] = {"item_idx": j, "code": c, "is_pipeline_assignment": real}
        out.append(f"### {tid} · {meta[j]['benchmark']}\n")
        out.append(f"> {clip(txt[j], QMAX)}\n")
        out.append(f"**Skill:** {codes[c]['name']} — {codes[c]['definition']}\n")
        out.append("```\nanswer: \n```\n")
        out.append("---\n")

    (GOLD / "task2_sheet.md").write_text("\n".join(out))
    (GOLD / "task2_key.json").write_text(json.dumps(
        {"note": "answer key; do not open before labelling", "seed": SEED,
         "excluded_benchmarks": sorted(SKIP), "items": key}, indent=1))
    kinds = "".join("R" if v["is_pipeline_assignment"] else "d" for v in key.values())
    longest = max(len(x) for x in kinds.replace("Rd", "R d").replace("dR", "d R").split())
    print(f"wrote gold/task2_sheet.md  ({len(seq)} items)")
    print(f"  composition : {kinds.count('R')} real / {kinds.count('d')} decoy")
    print(f"  order       : {kinds}")
    print(f"  longest run : {longest}")
    print(f"  fresh items : none reused from the previous sheet")


if __name__ == "__main__":
    main()
