#!/usr/bin/env python3
"""Draw a SECOND gold sample, stratified so small skills are actually testable.

Why this is a separate file from gold_sample.py, and not a flag on it.

gold_sample.py draws the reference set, and its defining property is that
selection never looks at pipeline output: a set stratified by the thing it judges
agrees with the pipeline by construction. A test parses that file with AST to
enforce it. This sampler deliberately breaks that property, so it must not share
the file, and the reference set must stay drawn the way it was drawn.

What breaking it buys, and what it costs.

The floor argument (drop or merge skills holding fewer than 20 questions) is the
biggest open question in the taxonomy. The LLM judge reports a 22-point
acceptance gap by skill size; the 100 human labels report 5 points and cannot
refute the judge, because only 10 of their 145 assignments come from skills below
the floor. The reference set was stratified by benchmark, and small skills appear
in few questions by construction, so it has no power there. That is what this
draw fixes: it oversamples questions whose assigned skills are small.

The cost is that this set is NOT a corpus estimate. Its overall acceptance rate
is biased downward by design if small skills are worse, and upward if they are
better, so the number to read off it is the DIFFERENCE between its two strata,
never its headline. It carries a control stratum drawn from skills at or above
the floor for exactly that comparison, labelled in the same sitting by the same
person, so the difference is not confounded with when or how it was labelled.

What is NOT compromised: the labeller still writes Part A before seeing Part B,
and still never sees which Part B entries are the pipeline's and which are
decoys. Stratification changes which questions are asked, not what the human is
shown about the answer.

    python3 tools/gold_sample2.py --n 40
"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.metrics import row_codes

P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
GOLD = REPO / "gold"
SEED = 20260912
FLOOR = 20          # Step 0's minimum questions per skill
BENCH_FLOOR = 2     # minimum per benchmark per stratum, where the pool allows


def already_labelled():
    """Every item a human has already answered, across both earlier sheets.

    Remembering an earlier answer is its own contamination, and the second sheet
    is meant to add power, not to re-ask.
    """
    seen = set()
    f = GOLD / "gold_set.json"
    if f.exists():
        seen |= {i["item_idx"] for i in json.loads(f.read_text())["items"]}
    f = GOLD / "task2_key.json"
    if f.exists():
        seen |= {v["item_idx"] for v in json.loads(f.read_text())["items"].values()}
    f = GOLD / "choice_key.json"
    if f.exists():
        o = json.loads(f.read_text())
        for part in ("task1", "task2"):
            seen |= {v["item_idx"] for v in o.get(part, {}).values()}
    return seen


def spread(pool, quota, meta, txt, rng):
    """Pick `quota` items from `pool`, spread across subtask and question length.

    Without this a benchmark gets represented by one subtask, or only by its
    short questions, and the stratum stops being about skill size.
    """
    if not pool or quota <= 0:
        return []
    subs = defaultdict(list)
    for i in pool:
        subs[meta[i].get("subtask") or "-"].append(i)
    for k in subs:
        rng.shuffle(subs[k])
        subs[k].sort(key=lambda i: len(txt[i]))
    keys, picked, r = sorted(subs), [], 0
    while len(picked) < quota and any(subs[k] for k in keys):
        band = subs[keys[r % len(keys)]]
        if band:
            # walk the length bands too, so picks are not all short or all long
            picked.append(band.pop(len(band) * (r // len(keys) * 37 % 100) // 100))
        r += 1
        if r > 100000:
            break
    return picked[:quota]


def draw(pool, n, meta, txt, rng, floor=BENCH_FLOOR):
    """Allocate exactly n across benchmarks, then spread within each.

    `floor` guarantees a minimum per benchmark. That is right when this is called
    once for a whole stratum and wrong when it is called once per matching band,
    because the floor then applies inside every band and the stratum overshoots
    n. The first draw asked for 20 control questions and got 29 that way, which
    pushed the below-floor share of assignments down to 35% when reaching half is
    the entire purpose of the set.
    """
    if not pool or n <= 0:
        return []
    by_b = defaultdict(list)
    for i in pool:
        by_b[meta[i]["benchmark"]].append(i)
    total = sum(len(v) for v in by_b.values())
    order = sorted(by_b, key=lambda b: -len(by_b[b]))
    quota = {b: 0 for b in by_b}
    for b in order:                                   # the floor, but never past n
        room = n - sum(quota.values())
        if room <= 0:
            break
        quota[b] = min(floor, len(by_b[b]), room)
    left = n - sum(quota.values())                    # remainder, by pool size
    for b in order:
        add = min(int(left * len(by_b[b]) / total), len(by_b[b]) - quota[b])
        quota[b] += max(0, add)
    while sum(quota.values()) < n and any(quota[b] < len(by_b[b]) for b in order):
        for b in order:
            if sum(quota.values()) >= n:
                break
            if quota[b] < len(by_b[b]):
                quota[b] += 1
    out = []
    for b in order:
        out += spread(by_b[b], quota[b], meta, txt, rng)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--codebook", default="codebook_v2_amended_b150.json")
    ap.add_argument("--labels", default="item_labels_b150.jsonl")
    ap.add_argument("--out", default="gold/gold_set_2.json")
    ap.add_argument("--force", action="store_true", help="redraw an existing frozen set")
    a = ap.parse_args()
    GOLD.mkdir(exist_ok=True)
    frozen = REPO / a.out
    if frozen.exists() and not a.force:
        raise SystemExit(f"{a.out} already exists. The set is frozen by design; pass "
                         f"--force only if you truly mean to invalidate every score "
                         f"measured against it.")

    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {p["item_idx"]: p for p in json.load(
        open(REPO / "cdm_exploration/experiments/oldtax_repaired_FINAL.json"))["per_item"]}

    cb = json.loads((P / a.codebook).read_text())
    alias = cb.get("alias", {})
    live = [c for c in cb["codes"] if c not in alias]
    per_item, cnt = {}, Counter()
    for line in (P / a.labels).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        cs = [c for c in row_codes(r, alias) if c in cb["codes"]]
        if cs:
            per_item[r["item_idx"]] = cs
            for c in cs:
                cnt[c] += 1
    under = {c for c in live if 0 < cnt.get(c, 0) < FLOOR}

    seen = already_labelled()
    avail = [i for i in sorted(per_item) if i not in seen and i in meta and i in txt]
    small = [i for i in avail if any(c in under for c in per_item[i])]
    # EVERY assignment below the floor, not merely one: otherwise a small skill
    # rides along with two large ones and a disagreement cannot be attributed.
    pure = [i for i in small if all(c in under for c in per_item[i])]
    pool_small = pure if len(pure) >= a.n else small
    pool_ctrl = [i for i in avail if not any(c in under for c in per_item[i])]

    rng = random.Random(SEED)
    pick_s = draw(pool_small, a.n // 2, meta, txt, rng)
    # Match the control on assignments per question. Unmatched, the control
    # carries about 1.8 skills per question against the treatment's 1.1, so the
    # two strata would differ in how much there is to disagree about as well as
    # in skill size, and the gap could not be read as a size effect. Matching
    # also brings the share of below-floor assignments near the half T-110 asks
    # for, because the two strata then contribute equal numbers of assignments.
    want = Counter(len(per_item[i]) for i in pick_s)
    pick_c = []
    for k, n_k in sorted(want.items()):
        band = [i for i in pool_ctrl if len(per_item[i]) == k]
        # floor=0: a per-benchmark minimum inside every band overshoots the stratum
        pick_c += draw(band, min(n_k, len(band)), meta, txt, rng, floor=0)
    short = a.n - len(pick_s) - len(pick_c)
    if short > 0:
        taken = set(pick_c)
        pick_c += draw([i for i in pool_ctrl if i not in taken], short,
                       meta, txt, rng, floor=0)
    chosen = pick_s + pick_c
    if len(chosen) != a.n:
        raise SystemExit(
            f"asked for {a.n} questions and drew {len(chosen)} ({len(pick_s)} "
            f"below-floor + {len(pick_c)} control). Unequal strata are what this "
            f"set exists to avoid, so the allocation is wrong rather than the ask.")
    stratum = {**{i: "below_floor" for i in pick_s}, **{i: "at_or_above" for i in pick_c}}

    asg = [(i, c) for i in chosen for c in per_item[i]]
    n_under = sum(1 for _, c in asg if c in under)
    blob = {
        "seed": SEED, "n": len(chosen), "drawn": "2026-09-12",
        "purpose": "power to test the floor argument (the task list T-110); NOT a corpus "
                   "estimate. Read the difference between the two strata, never the "
                   "headline acceptance rate.",
        "pipeline_dependent": True,
        "pipeline_dependence": f"selection used the pipeline's own assignments to find "
                               f"questions held by skills with fewer than {FLOOR} items. "
                               f"gold_set.json deliberately does not do this; that set "
                               f"stays the unbiased reference.",
        "stratified_on": {"codebook": a.codebook, "labels": a.labels, "floor": FLOOR},
        "excluded_already_labelled": len(seen),
        "corpus_estimate": False,
        "strata": {"below_floor": len(pick_s), "at_or_above": len(pick_c)},
        "control_matched_on": "assignments per question, so the two strata differ in "
                              "skill size and not in how many skills are on offer",
        "assignments_per_item": {
            "below_floor": round(sum(len(per_item[i]) for i in pick_s) / max(1, len(pick_s)), 2),
            "at_or_above": round(sum(len(per_item[i]) for i in pick_c) / max(1, len(pick_c)), 2)},
        "assignments": {"total": len(asg), "from_below_floor": n_under,
                        "share_below_floor": round(n_under / max(1, len(asg)), 3)},
        "pool_sizes": {"available_unseen": len(avail), "touching_a_small_skill": len(small),
                       "all_assignments_below_floor": len(pure), "control": len(pool_ctrl)},
        "items": [{"item_idx": i, "benchmark": meta[i]["benchmark"],
                   "subtask": meta[i].get("subtask"), "stratum": stratum[i],
                   "n_assigned": len(per_item[i]),
                   "n_assigned_below_floor": sum(c in under for c in per_item[i])}
                  for i in chosen],
    }
    frozen.write_text(json.dumps(blob, indent=1))
    print(f"froze {len(chosen)} questions -> {a.out}")
    print(f"  skills below the floor of {FLOOR}: {len(under)} of {len(live)} live")
    print(f"  excluded {len(seen)} already-labelled questions")
    print(f"  strata: {len(pick_s)} below-floor + {len(pick_c)} control")
    print(f"  assignments from below-floor skills: {n_under}/{len(asg)} "
          f"= {n_under/max(1,len(asg)):.0%}  (reference set: 10/145 = 7%)")
    print(f"  assignments per question: {blob['assignments_per_item']['below_floor']} "
          f"below-floor vs {blob['assignments_per_item']['at_or_above']} control "
          f"(matched, so size is the only systematic difference)")
    for s, picks in (("below_floor", pick_s), ("at_or_above", pick_c)):
        bb = Counter(meta[i]["benchmark"] for i in picks)
        print(f"  {s:12s} {dict(bb)}")
    print(f"\nnow run: python3 tools/gold_sheet.py --set {a.out} "
          f"--out gold/gold_sheet_2.md --key gold/gold_partB_key_2.json \\\n"
          f"    --codebook {a.codebook} --labels {a.labels}")


if __name__ == "__main__":
    main()
