"""Phase 1: per-cluster tagging-validity profile (judge v3: binary, no benchmark
header). Settles whether invalid tags are CONCENTRATED in a few clusters or
SPREAD broadly, and whether validity falls with cluster size.

    .venv312/bin/python tools/diag_validity_per_cluster.py \
        cdm_exploration/experiments/oldtax_full_format.json \
        cdm_exploration/experiments/oldtax_percluster_v3.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
LABF = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "cdm_exploration/experiments/oldtax_full_format.json"
OUTF = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    REPO / "cdm_exploration/experiments/oldtax_percluster_v3.json"
PAIRS_PER = 10
random.seed(42)

VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(LABF))
    defn = {b["name"]: b["definition"] for b in data["bank"]}
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}

    sk_items = defaultdict(list)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
    evidence = data.get("evidence", {})   # items used to generate names: never judge them

    tasks = []
    for sk, items in sk_items.items():
        pool = [i for i in items if i not in set(evidence.get(sk, []))] or items
        for i in random.sample(pool, min(PAIRS_PER, len(pool))):
            tasks.append((sk, i))
    print(f"clusters={len(sk_items)}  judged pairs={len(tasks)}  (up to {PAIRS_PER} each)")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(t):
        sk, i = t
        d = defn.get(sk, "")
        dline = f"\nDefinition: {d}" if d else ""
        u = (f"Item:\n{qtext.get(i,'')}\n\nSkill: {sk}{dline}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": VSYS},
                              {"role": "user", "content": u}])
                return (sk, json.loads(r.choices[0].message.content).get("verdict") == "yes")
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return (sk, False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, tasks))

    yes = defaultdict(int)
    n = defaultdict(int)
    for sk, ok in res:
        n[sk] += 1
        yes[sk] += ok
    prof = sorted(((yes[s] / n[s], n[s], len(sk_items[s]), s) for s in n), key=lambda x: x[0])

    print(f"\n{'validity':>8s} {'judged':>7s} {'size':>6s}  cluster")
    for v, k, sz, s in prof:
        print(f"{v:8.0%} {k:7d} {sz:6d}  {s[:60]}")

    vals = np.array([p[0] for p in prof])
    sizes = np.array([p[2] for p in prof])
    w = np.array([p[1] for p in prof])
    overall = (vals * w).sum() / w.sum()
    print(f"\noverall (pair-weighted) = {overall:.0%}")
    print(f"clusters <=20% valid: {(vals <= .2).sum()}  |  20-60%: {((vals > .2) & (vals < .6)).sum()}"
          f"  |  >=60%: {(vals >= .6).sum()}")
    lowshare = sum(p[1] - yes[p[3]] for p in prof if p[0] <= .2)
    totno = sum(n[s] - yes[s] for s in n)
    print(f"share of ALL invalid tags coming from the <=20% clusters: "
          f"{lowshare}/{totno} = {lowshare/max(totno,1):.0%}")
    r = np.corrcoef(np.log(sizes), vals)[0, 1]
    print(f"corr( log cluster size , validity ) = {r:.2f}")

    json.dump({"per_cluster": [{"skill": s, "validity": v, "judged": k, "size": sz}
                               for v, k, sz, s in prof],
               "overall": overall, "pairs_per": PAIRS_PER},
              open(OUTF, "w"))
    print(f"\nwrote {OUTF}")


if __name__ == "__main__":
    main()
