"""Per-cluster NAME-FREE coherence: several intruder trials per cluster, no skill
name shown (pure "do these questions belong together"). The dedicated
mixedness detector for the metric-profile design.

    .venv312/bin/python tools/diag_coherence_per_cluster.py \
        cdm_exploration/experiments/oldtax_full_format.json \
        cdm_exploration/experiments/oldtax_coherence_percluster.json
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
LABF = Path(sys.argv[1])
OUTF = Path(sys.argv[2])
TRIALS = 6
random.seed(42)

ISYS = "You find the one item that does NOT belong with the others."


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(LABF))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    sk_items = defaultdict(list)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
    items_all = [pi["item_idx"] for pi in data["per_item"]]
    item_skills = defaultdict(set)
    for pi in data["per_item"]:
        item_skills[pi["item_idx"]].update(pi["skills"])

    tasks = []
    for sk, items in sk_items.items():
        uniq = list(set(items))
        if len(uniq) < 4:
            continue
        for t in range(TRIALS):
            own = random.sample(uniq, 4)
            pool = [i for i in items_all if sk not in item_skills[i]]
            intr = random.choice(pool)
            five = own + [intr]
            random.shuffle(five)
            tasks.append((sk, five, five.index(intr) + 1))
    print(f"clusters={len(set(t[0] for t in tasks))}  trials={len(tasks)}")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(t):
        sk, five, truth = t
        body = "\n".join(f"{k+1}. {qtext.get(i,'')[:400]}" for k, i in enumerate(five))
        u = ("These five test questions should all belong to one group that tests the same "
             f"skill. Exactly one does NOT belong. Which number is the odd one out?\n\n{body}\n\n"
             'Return JSON {"intruder": <number 1-5>}.')
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": ISYS},
                              {"role": "user", "content": u}])
                return (sk, int(json.loads(r.choices[0].message.content).get("intruder", -1)) == truth)
            except Exception:  # noqa: BLE001
                if a == 2:
                    return (sk, False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, tasks))

    hit = defaultdict(int); n = defaultdict(int)
    for sk, ok in res:
        n[sk] += 1; hit[sk] += ok
    prof = sorted(((hit[s]/n[s], n[s], s) for s in n))
    for v, k, s in prof[:10]:
        print(f"{v:6.0%} {k:2d}  {s[:60]}")
    print("...")
    import numpy as np
    vals = np.array([p[0] for p in prof])
    print(f"clusters tested={len(prof)}  mean detection={vals.mean():.0%}  <=33%: {(vals<=1/3).sum()}")
    json.dump({"per_cluster": [{"skill": s, "coherence": v, "trials": k} for v, k, s in prof],
               "trials_per_cluster": TRIALS},
              open(OUTF, "w"))
    print(f"wrote {OUTF}")


if __name__ == "__main__":
    main()
