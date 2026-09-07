"""Top-up the borderline clusters with more judge samples, then rebuild the
repair worklist with the tightened scores.

  - validity in (0.3, 0.7)  -> re-check with up to 30 fresh pairs (was 10)
  - group test in [1/3, 2/3] -> add 12 trials (6 -> 18 total)

    .venv312/bin/python tools/topup_borderline.py
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
random.seed(77)

VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."
ISYS = "You find the one item that does NOT belong with the others."


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    lab = json.load(open(E / "oldtax_full_format.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    sk_items = defaultdict(list)
    item_skills = defaultdict(set)
    for pi in lab["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
            item_skills[pi["item_idx"]].add(s)
    items_all = list(item_skills)

    val0 = {c["skill"]: (c["validity"], c["size"]) for c in json.load(open(E / "oldtax_percluster_v3.json"))["per_cluster"]}
    coh0 = {c["skill"]: (c["coherence"], c["trials"]) for c in json.load(open(E / "oldtax_coherence_percluster.json"))["per_cluster"]}

    vb = [s for s, (v, sz) in val0.items() if 0.3 < v < 0.7]
    cb = [s for s, (c, t) in coh0.items() if 1/3 - 1e-6 <= c <= 2/3 + 1e-6]
    print(f"borderline: validity {len(vb)} clusters (30 pairs each), "
          f"group test {len(cb)} clusters (+12 trials each)")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def call(system, user, field, cast):
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return cast(json.loads(r.choices[0].message.content).get(field))
            except Exception:  # noqa: BLE001
                if a == 2:
                    return None

    # ---- validity top-up ----
    vtasks = []
    for sk in vb:
        pool = list(set(sk_items[sk]))
        for i in random.sample(pool, min(30, len(pool))):
            vtasks.append((sk, i))
    def vjudge(t):
        sk, i = t
        u = (f"Item:\n{qtext.get(i,'')}\n\nSkill: {sk}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return (sk, call(VSYS, u, "verdict", lambda x: x == "yes"))
    with ThreadPoolExecutor(max_workers=8) as ex:
        vres = list(ex.map(vjudge, vtasks))
    vy = defaultdict(int); vn = defaultdict(int)
    for sk, ok in vres:
        if ok is not None:
            vn[sk] += 1; vy[sk] += ok

    # ---- group-test top-up ----
    ctasks = []
    for sk in cb:
        uniq = list(set(sk_items[sk]))
        for t in range(12):
            own = random.sample(uniq, 4)
            intr = random.choice([i for i in items_all if sk not in item_skills[i]])
            five = own + [intr]; random.shuffle(five)
            ctasks.append((sk, five, five.index(intr) + 1))
    def cjudge(t):
        sk, five, truth = t
        body = "\n".join(f"{k+1}. {qtext.get(i,'')[:400]}" for k, i in enumerate(five))
        u = ("These five test questions should all belong to one group that tests the same "
             f"skill. Exactly one does NOT belong. Which number is the odd one out?\n\n{body}\n\n"
             'Return JSON {"intruder": <number 1-5>}.')
        got = call(ISYS, u, "intruder", int)
        return (sk, got == truth if got else False)
    with ThreadPoolExecutor(max_workers=8) as ex:
        cres = list(ex.map(cjudge, ctasks))
    ch = defaultdict(int); cn = defaultdict(int)
    for sk, ok in cres:
        cn[sk] += 1; ch[sk] += ok

    # ---- final scores ----
    val_f = {}
    for sk, (v, sz) in val0.items():
        val_f[sk] = (vy[sk] / vn[sk]) if vn.get(sk) else v
    coh_f = {}
    for sk, (c, t) in coh0.items():
        if cn.get(sk):
            coh_f[sk] = (c * t + ch[sk]) / (t + cn[sk])   # combine 6 old + 12 new
        else:
            coh_f[sk] = c

    # ---- rebuild worklist ----
    def pile(sk):
        if sk not in coh_f:
            return "CHECK-TINY"
        if coh_f[sk] <= 0.5:
            return "SPLIT"
        if val_f[sk] <= 0.5:
            return "RENAME"
        return "OK"
    old_pile = {}
    for sk in val0:
        c = coh0.get(sk, (None,))[0]
        old_pile[sk] = ("CHECK-TINY" if c is None else
                        "SPLIT" if c <= 0.5 else
                        "RENAME" if val0[sk][0] <= 0.5 else "OK")
    new_pile = {sk: pile(sk) for sk in val0}

    from collections import Counter
    print("\npiles      before  after")
    for k in ("SPLIT", "RENAME", "CHECK-TINY", "OK"):
        print(f"{k:11s} {Counter(old_pile.values())[k]:4d} {Counter(new_pile.values())[k]:6d}")
    moved = [(sk, old_pile[sk], new_pile[sk]) for sk in val0 if old_pile[sk] != new_pile[sk]]
    print(f"\nclusters that changed pile: {len(moved)}")
    for sk, a, b in moved:
        print(f"  {a:7s} -> {b:7s}  group {coh0.get(sk,(None,))[0]} -> {coh_f.get(sk):.0%} | "
              f"valid {val0[sk][0]:.0%} -> {val_f[sk]:.0%}  {sk[:45]}")

    json.dump({"validity_final": val_f, "coherence_final": coh_f,
               "piles": {k: sorted(s for s in new_pile if new_pile[s] == k)
                         for k in ("SPLIT", "RENAME", "CHECK-TINY", "OK")}},
              open(E / "oldtax_repair_worklist_v2.json", "w"), indent=1)
    print(f"\nsaved -> {E/'oldtax_repair_worklist_v2.json'}")


if __name__ == "__main__":
    main()
