"""Score the human evaluation of skill-question assignments.

Reports the two averages we promised to make comparable to the LLM judge:
  micro / question-weighted  -> every skill-question pair counts once (LLM: 0.834)
  macro / per-skill mean     -> every skill counts once           (LLM: 0.815)

Also runs consistency checks:
  - duplicate question_text judged differently (the option-shuffled GPQA items)
  - blanks and non yes/no values
"""
from __future__ import annotations
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else
            "human_eval_skills.csv")


def norm(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"y", "yes"}:
        return "yes"
    if v in {"n", "no", "mo", "noo"}:      # 'mo' = observed typo for no
        return "no"
    return ""


def main() -> None:
    rows = list(csv.DictReader(open(PATH, encoding="cp1252"), delimiter=";"))
    print(f"rows: {len(rows)}\n")

    raw = Counter((r.get("skill_needed_yes_no") or "").strip().lower() for r in rows)
    print("raw values:", dict(raw))
    bad = [r for r in rows if norm(r["skill_needed_yes_no"]) == ""]
    for r in bad:
        print(f"  BLANK/UNREADABLE: skill {r['skill_id']} item {r['item_idx']} "
              f"[{r['skill_name']}] -> {r['skill_needed_yes_no']!r}")
    typo = [r for r in rows if (r["skill_needed_yes_no"] or "").strip().lower() == "mo"]
    for r in typo:
        print(f"  TYPO 'mo' treated as no: skill {r['skill_id']} item {r['item_idx']} "
              f"[{r['skill_name']}]")

    scored = [r for r in rows if norm(r["skill_needed_yes_no"])]
    yes = sum(1 for r in scored if norm(r["skill_needed_yes_no"]) == "yes")
    micro = yes / len(scored)
    print(f"\n=== MICRO (question-weighted) ===")
    print(f"  {yes}/{len(scored)} = {micro:.3f}   (LLM judge gpt-4o-mini: 0.834)")

    per_skill = defaultdict(list)
    names = {}
    for r in scored:
        per_skill[int(r["skill_id"])].append(norm(r["skill_needed_yes_no"]) == "yes")
        names[int(r["skill_id"])] = r["skill_name"]
    rates = {k: float(np.mean(v)) for k, v in per_skill.items()}
    macro = float(np.mean(list(rates.values())))
    print(f"\n=== MACRO (per-skill mean, n={len(rates)} skills) ===")
    print(f"  {macro:.3f}   (LLM judge gpt-4o-mini: 0.815)")

    print(f"\n=== WORST 12 SKILLS ===")
    for k in sorted(rates, key=lambda k: rates[k])[:12]:
        n = len(per_skill[k])
        print(f"  {rates[k]:5.0%}  ({sum(per_skill[k])}/{n})  skill {k:3d}  {names[k]}")

    print(f"\n=== SKILLS AT 100% ===")
    perfect = [k for k in rates if rates[k] == 1.0]
    print(f"  {len(perfect)} of {len(rates)} skills")

    # consistency: identical question text judged differently
    print(f"\n=== CONSISTENCY: same question, different verdict ===")
    by_q = defaultdict(list)
    for r in scored:
        key = (int(r["skill_id"]), re.sub(r"\s+", " ", r["question_text"]).strip()[:300])
        by_q[key].append(r)
    clashes = 0
    for (sid, q), rs in by_q.items():
        verdicts = {norm(r["skill_needed_yes_no"]) for r in rs}
        if len(verdicts) > 1:
            clashes += 1
            print(f"  skill {sid} [{names[sid]}]: {[r['item_idx'] for r in rs]} -> "
                  f"{[norm(r['skill_needed_yes_no']) for r in rs]}")
            print(f"     {q[:150]}")
    if not clashes:
        print("  none found")

    print(f"\n=== DISTRIBUTION ===")
    b = Counter()
    for v in rates.values():
        b["100%" if v == 1 else "80-99%" if v >= .8 else "50-79%" if v >= .5 else "below 50%"] += 1
    for k in ["100%", "80-99%", "50-79%", "below 50%"]:
        print(f"  {k:10s} {b[k]:3d} skills")


if __name__ == "__main__":
    main()
