"""A/B test of the "Be strict." phrase in the validity judge (Berke's challenge
2026-07-22): does removing it change own-item validity, stranger acceptance,
and the discrimination gap the pipeline depends on?

12 certified v5 skills x (10 own + 10 stranger) x 2 prompt variants, gpt-4o-mini.

    python3 tools/diag_strict_ab.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
E = REPO / "cdm_exploration/experiments"
random.seed(42)

VSYS_STRICT = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."
VSYS_PLAIN = "You judge whether solving a test item genuinely requires a given cognitive skill."


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    snap = json.load(open(E / "pipeline_fresh/round7_names.json"))
    cert = json.load(open(E / "pipeline_fresh/certified.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    ok = [r["skill"] for r in cert["rows"] if r["certified"]
          and r["skill"] in snap["skills"] and r["n_items"] >= 10]
    skills = random.sample(ok, 12)
    all_items = sorted({i for d in snap["skills"].values() for i in d["items"]})

    tasks = []
    for s in skills:
        d = snap["skills"][s]
        own = random.sample(sorted(d["items"]), 10)
        strangers = random.sample([i for i in all_items if i not in set(d["items"])], 10)
        for variant, sys_p in (("strict", VSYS_STRICT), ("plain", VSYS_PLAIN)):
            for side, ids in (("own", own), ("stranger", strangers)):
                for i in ids:
                    tasks.append((s, variant, side, sys_p, d["definition"], i))
    print(f"skills={len(skills)}  judgments={len(tasks)}")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(t):
        s, variant, side, sys_p, defn, i = t
        dline = f"\nDefinition: {defn}" if defn else ""
        u = (f"Item:\n{qtext.get(i, '')}\n\nSkill: {s}{dline}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": sys_p},
                              {"role": "user", "content": u}])
                return (s, variant, side,
                        json.loads(r.choices[0].message.content).get("verdict") == "yes")
            except Exception:  # noqa: BLE001
                if a == 2:
                    return (s, variant, side, False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, tasks))

    agg = defaultdict(lambda: [0, 0])
    for s, variant, side, yes in res:
        agg[(variant, side)][0] += yes
        agg[(variant, side)][1] += 1
    out = {}
    for (variant, side), (y, n) in sorted(agg.items()):
        out[f"{variant}_{side}"] = y / n
        print(f"  {variant:7s} {side:9s}: {y}/{n} = {y/n:.0%}")
    gs = out["strict_own"] - out["strict_stranger"]
    gp = out["plain_own"] - out["plain_stranger"]
    print(f"\n  gap WITH 'Be strict':    {gs:+.0%}")
    print(f"  gap WITHOUT 'Be strict': {gp:+.0%}")
    json.dump({"skills": skills, "rates": out, "gap_strict": gs, "gap_plain": gp},
              open(E / "preflight_strict_ab.json", "w"), indent=1)
    print(f"wrote {E/'preflight_strict_ab.json'}")


if __name__ == "__main__":
    main()
