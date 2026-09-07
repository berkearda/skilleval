"""Repair pass: regenerate every skill/cluster name from its member items.

Generic over any pilot-format labels file. Writes a new labels file with
renamed bank + remapped per_item, plus the evidence items used per skill
(so downstream validity sampling can exclude them; no self-grading).

    .venv312/bin/python tools/repair_names.py \
        cdm_exploration/experiments/oldtax_full_format.json \
        cdm_exploration/experiments/oldtax_full_renamed.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = Path(args[0])
OUT = Path(args[1])
SPECIFIC = "--specific" in sys.argv
EVID = 8
random.seed(42)

GENERIC = ("Write a concise name in English (at most 8 words) for the skill this group "
           "measures, regardless of the items' language.")
SPECIFIC_INSTR = ("Name the specific cognitive OPERATION this group requires, as a short English "
                  "verb phrase (for example 'solve algebraic equations', 'track object positions "
                  "after swaps', 'follow explicit output-format constraints'). Do NOT name a subject "
                  "area (like 'chemistry problems', 'mathematics', 'physics'). Do NOT use vague words "
                  "like 'problem solving', 'analysis', 'skills', 'reasoning', 'advanced'. If the items "
                  "do not share one operation, name the single most common one.")


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(SRC))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    sk_items = defaultdict(list)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])

    evidence = {sk: random.sample(items, min(EVID, len(items)))
                for sk, items in sk_items.items()}

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    instr = SPECIFIC_INSTR if SPECIFIC else GENERIC

    def rename(sk):
        ev = "\n".join(f"- {qtext.get(i,'')[:300]}" for i in evidence[sk])
        u = (f"These test items form one group in a skill taxonomy:\n{ev}\n\n"
             f'{instr} Return JSON {{"name": "<concise name>"}}')
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content":
                               "You name skill groups from their member items. You are not shown any existing name."},
                              {"role": "user", "content": u}])
                return sk, str(json.loads(r.choices[0].message.content).get("name", "")).strip()
            except Exception:  # noqa: BLE001
                if a == 2:
                    return sk, ""

    skills = list(sk_items)
    with ThreadPoolExecutor(max_workers=8) as ex:
        raw = dict(ex.map(rename, skills))

    # collisions or empty names: keep distinct clusters distinct
    mapping, used = {}, set()
    collisions = empties = 0
    for sk in skills:
        nm = raw[sk] or sk
        if not raw[sk]:
            empties += 1
        base, k = nm, 2
        while nm in used:
            nm = f"{base} ({k})"
            k += 1
            collisions += 1
        used.add(nm)
        mapping[sk] = nm

    out = {"bank": [{"name": mapping[sk], "definition": "", "rank": k}
                    for k, sk in enumerate(skills)],
           "per_item": [{**{k: pi[k] for k in ("item_idx", "benchmark", "subtask")},
                         "skills": [mapping[s] for s in pi["skills"]]}
                        for pi in data["per_item"]],
           "rename_map": mapping,
           "evidence": {mapping[sk]: evidence[sk] for sk in skills}}
    json.dump(out, open(OUT, "w"))
    print(f"renamed {len(skills)} skills  (collisions suffixed: {collisions}, "
          f"empty->kept-original: {empties})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
