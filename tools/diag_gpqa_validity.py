"""Judge tagging validity of a step-wise extraction output (extract_stepwise shape)
for GPQA, to compare a stronger solver against the gpt-4o-mini baseline (52% yes
/ 48% no). Samples item-skill pairs and asks whether the item requires the skill.

    .venv312/bin/python tools/diag_gpqa_validity.py \
        cdm_exploration/experiments/gpqa_gpt4o.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "cdm_exploration/experiments/gpqa_gpt4o.json"
KEYFILE = Path.home() / ".cdmeval_openai_key"
N = 40
random.seed(42)
VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."


def load_key():
    t = KEYFILE.read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    rows = json.load(open(SRC))
    qtext = {r["item_idx"]: r["question_full_text"]
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    pairs = []
    for r in rows:
        for sk in set(r.get("atomic_skills", [])):
            pairs.append((r["item_idx"], r.get("subtask", "Diamond"), sk))
    sample = random.sample(pairs, min(N, len(pairs)))

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(t):
        i, sub, sk = t
        q = " ".join(qtext.get(i, "").split())[:380]
        u = (f"Item (GPQA/{sub}):\n{q}\n\nSkill: {sk}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|partial|no"}.')
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": VSYS},
                              {"role": "user", "content": u}])
                return json.loads(r.choices[0].message.content).get("verdict", "no")
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return "no"

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, sample))
    c = Counter(res)
    n = sum(c.values())
    errored = sum(1 for r in rows if r.get("error"))
    nsk = len({s for r in rows for s in r.get("atomic_skills", [])})
    print(f"GPQA re-solved with stronger model: {len(rows)} items, {errored} errored, "
          f"{nsk} distinct skills")
    print(f"tagging validity (n={n}):  yes={c['yes']/n:.0%}  partial={c['partial']/n:.0%}  no={c['no']/n:.0%}")
    print("baseline (gpt-4o-mini): yes=52%  partial=0%  no=48%")


if __name__ == "__main__":
    main()
