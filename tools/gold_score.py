#!/usr/bin/env python3
"""Score the hand-labelled choice sheet against the answer key.

Task 1 settles the doc's rule: "any pipeline change has to improve on this set
to be kept." Each item shows one code's old and rewritten definition, unlabelled
and in random order. A human preference for the rewrite is evidence to keep
Step 8; a preference for the original is evidence to revert it.

Task 2 measures whether the LLM judge used throughout Step 6 can be trusted, by
comparing a human yes/no against the pipeline's own assignment, with decoys as a
null control.

    python3 tools/gold_score.py
"""
import json, re, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "gold"


def parse():
    t = (GOLD / "choice_sheet.md").read_text()
    out = {}
    for m in re.finditer(r"### (T[12]-\d+).*?```\s*answer:\s*([^\n`]*)", t, re.S):
        v = m.group(2).strip().lower()
        if v:
            out[m.group(1)] = v
    return out


def main():
    key = json.loads((GOLD / "choice_key.json").read_text())
    ans = parse()

    print("=" * 62)
    print("TASK 1 — which definition fits the question better?")
    print("=" * 62)
    t1 = key["task1"]
    tally, agree, dis = Counter(), 0, 0
    for tid, k in sorted(t1.items()):
        a = ans.get(tid)
        if not a:
            tally["unanswered"] += 1
            continue
        if a in ("both", "neither"):
            tally[a] += 1
            continue
        if a not in ("a", "b"):
            tally["unparsed"] += 1
            continue
        chose = k["A"] if a == "a" else k["B"]      # "old" or "new"
        tally[chose] += 1
        if chose == k["judge_preferred"]:
            agree += 1
        else:
            dis += 1
    n_dec = tally["old"] + tally["new"]
    print(f"  answered {sum(tally.values()) - tally['unanswered']} of {len(t1)}")
    print(f"  preferred the REWRITTEN definition : {tally['new']}")
    print(f"  preferred the ORIGINAL definition  : {tally['old']}")
    print(f"  'both'                             : {tally['both']}")
    print(f"  'neither'                          : {tally['neither']}")
    if n_dec:
        print(f"\n  of the {n_dec} where a side was chosen, the rewrite won {tally['new']}/{n_dec} "
              f"({tally['new']/n_dec:.0%})")
        print(f"  agreement with the LLM judge's preference: {agree}/{agree+dis} "
              f"({agree/max(1,agree+dis):.0%})")

    print()
    print("=" * 62)
    print("TASK 2 — is the skill actually required?")
    print("=" * 62)
    t2 = key["task2"]
    tp = fp = tn = fn = 0
    for tid, k in sorted(t2.items()):
        a = ans.get(tid)
        if a not in ("y", "n"):
            continue
        yes, real = a == "y", k["is_pipeline_assignment"]
        if real and yes:
            tp += 1
        elif real and not yes:
            fn += 1
        elif not real and yes:
            fp += 1
        else:
            tn += 1
    print(f"  pipeline assignments accepted : {tp}/{tp+fn}"
          + (f"  ({tp/(tp+fn):.0%})" if tp + fn else ""))
    print(f"  decoys correctly rejected     : {tn}/{tn+fp}"
          + (f"  ({tn/(tn+fp):.0%})" if tn + fp else ""))
    if (tp + fn) and (tn + fp):
        sep = tp / (tp + fn) - fp / (fp + tn)
        print(f"\n  separation (accept rate on real minus on decoys): {sep:+.2f}")
        print("  near 0 would mean the answers do not distinguish real from decoy;")
        print(f"  {sep:+.2f} means they clearly do.")
    print()
    (GOLD / "score.json").write_text(json.dumps(
        {"task1": dict(tally), "task1_judge_agreement": [agree, agree + dis],
         "task2": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}, indent=1))
    print("wrote gold/score.json")


if __name__ == "__main__":
    main()
