"""Diagnostic: is tagging validity worse on hard benchmarks (bad reasoning) or
uniform (judge issue)? Samples item-skill pairs per benchmark and judges whether
the item genuinely requires its tagged skill.

    .venv312/bin/python tools/diag_validity_by_bench.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
LAB = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "cdm_exploration/experiments/pilot_step4_noesc.json"
KEYFILE = Path.home() / ".cdmeval_openai_key"
PER_BENCH = 40
random.seed(42)

VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."


def load_key():
    t = KEYFILE.read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(LAB))
    defn = {b["name"]: b["definition"] for b in data["bank"]}
    qtext = {r["item_idx"]: r["question_full_text"]
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    by_bench = defaultdict(list)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            by_bench[pi["benchmark"]].append((pi["item_idx"], pi["subtask"], s))

    pairs = []
    for b, lst in by_bench.items():
        pairs += [(b,) + p for p in random.sample(lst, min(PER_BENCH, len(lst)))]

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(t):
        b, i, sub, sk = t
        q = " ".join(qtext.get(i, "").split())[:380]
        u = (f"Item ({b}/{sub}):\n{q}\n\nSkill: {sk}\nDefinition: {defn.get(sk,'')}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|partial|no"}.')
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": VSYS},
                              {"role": "user", "content": u}])
                return (b, json.loads(r.choices[0].message.content).get("verdict", "no"))
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return (b, "no")

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, pairs))

    agg = defaultdict(Counter)
    for b, v in res:
        agg[b][v] += 1
    print(f"tagging validity by benchmark ({PER_BENCH} pairs each):\n")
    print(f"{'benchmark':10s} {'yes':>6s} {'partial':>8s} {'no':>6s}")
    for b in sorted(agg):
        c = agg[b]
        n = sum(c.values())
        print(f"{b:10s} {c['yes']/n:6.0%} {c['partial']/n:8.0%} {c['no']/n:6.0%}")
    allc = Counter(v for _, v in res)
    n = sum(allc.values())
    print(f"\n{'OVERALL':10s} {allc['yes']/n:6.0%} {allc['partial']/n:8.0%} {allc['no']/n:6.0%}")


if __name__ == "__main__":
    main()
