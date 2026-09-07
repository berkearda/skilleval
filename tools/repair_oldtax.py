"""Metric-driven repair of the submitted K=100 taxonomy, using worklist v2.

Stages:
  1 SPLIT  : recursively split flagged clusters; the group test decides when done
  2 RENAME : specific names for the RENAME pile and all split parts (evidence held out)
  3 MERGE  : judge nearest name pairs, merge confirmed duplicates
  4 MEASURE: validity + group test + granularity on the repaired taxonomy

    .venv312/bin/python tools/repair_oldtax.py --smoke   (2 splits, 3 renames)
    .venv312/bin/python tools/repair_oldtax.py
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
E = REPO / "cdm_exploration/experiments"
SMOKE = "--smoke" in sys.argv
random.seed(42)

ISYS = "You find the one item that does NOT belong with the others."
VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."
NSYS = "You name skill groups from their member items. You are not shown any existing name."
DSYS = "You judge whether two cognitive-skill labels denote essentially the SAME operation."
GSYS = "You rate the granularity of a cognitive-skill label for a skill taxonomy."
SPEC = ("Name the specific cognitive OPERATION this group requires, as a short English verb "
        "phrase. Do NOT name a subject area. Do NOT use vague words like 'problem solving', "
        "'analysis', 'skills', 'reasoning', 'advanced'. If the items do not share one "
        "operation, name the single most common one.")


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    lab = json.load(open(E / "oldtax_full_format.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {pi["item_idx"]: (pi["benchmark"], pi["subtask"]) for pi in lab["per_item"]}
    sk_items = defaultdict(list)
    for pi in lab["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
    wl = json.load(open(E / "oldtax_repair_worklist_v2.json"))["piles"]
    if SMOKE:
        wl = {"SPLIT": wl["SPLIT"][:2], "RENAME": wl["RENAME"][:3],
              "CHECK-TINY": wl["CHECK-TINY"], "OK": wl["OK"]}

    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    client = OpenAI(api_key=load_key())
    emb = SentenceTransformer("all-MiniLM-L6-v2")

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

    items_all = list(meta)
    item_skills = defaultdict(set)
    for pi in lab["per_item"]:
        item_skills[pi["item_idx"]].update(pi["skills"])

    def group_test(items, origin_skill, trials=6):
        """name-free intruder trials; returns pass-rate"""
        uniq = list(set(items))
        if len(uniq) < 4:
            return None
        hits = tasks = 0
        for _ in range(trials):
            own = random.sample(uniq, 4)
            intr = random.choice([i for i in items_all if origin_skill not in item_skills[i]])
            five = own + [intr]
            random.shuffle(five)
            truth = five.index(intr) + 1
            body = "\n".join(f"{k+1}. {qtext.get(i,'')[:400]}" for k, i in enumerate(five))
            u = ("These five test questions should all belong to one group that tests the same "
                 f"skill. Exactly one does NOT belong. Which number is the odd one out?\n\n{body}\n\n"
                 'Return JSON {"intruder": <number 1-5>}.')
            got = call(ISYS, u, "intruder", int)
            tasks += 1
            hits += (got == truth)
        return hits / tasks

    # ---------- stage 1: recursive split ----------
    print("=== stage 1: SPLIT ===", flush=True)
    final_groups = {}          # group_key -> [items]
    for sk in wl["OK"] + wl["RENAME"] + wl["CHECK-TINY"]:
        final_groups[sk] = sk_items[sk]
    n_parts = 0
    for sk in wl["SPLIT"]:
        queue = [(sk_items[sk], 0)]
        leaves = []
        while queue:
            items, depth = queue.pop()
            uniq = list(set(items))
            if len(uniq) < 4 or depth >= 3:
                leaves.append(items)
                continue
            score = group_test(items, sk) if depth > 0 else 0.0   # root already failed
            if score is not None and score > 0.5:
                leaves.append(items)
                continue
            V = emb.encode([qtext.get(i, "")[:400] for i in uniq], normalize_embeddings=True)
            from sklearn.cluster import AgglomerativeClustering
            labs = AgglomerativeClustering(n_clusters=2, metric="cosine",
                                           linkage="average").fit_predict(V)
            for c in (0, 1):
                part = [uniq[j] for j in range(len(uniq)) if labs[j] == c]
                if part:
                    queue.append((part, depth + 1))
        for k, part in enumerate(leaves):
            final_groups[f"{sk}##part{k}"] = part
        n_parts += len(leaves)
        print(f"  {sk[:50]}: -> {len(leaves)} parts", flush=True)
    print(f"split {len(wl['SPLIT'])} clusters into {n_parts} parts; "
          f"taxonomy now {len(final_groups)} groups", flush=True)

    # ---------- stage 2: rename ----------
    print("=== stage 2: RENAME ===", flush=True)
    to_name = [g for g in final_groups
               if "##part" in g or g in set(wl["RENAME"])]
    evidence = {}
    def gen_name(g):
        items = final_groups[g]
        ev = random.sample(items, min(8, len(items)))
        evidence[g] = ev
        block = "\n".join(f"- {qtext.get(i,'')[:300]}" for i in ev)
        u = (f"These test items form one group in a skill taxonomy:\n{block}\n\n"
             f'{SPEC} Return JSON {{"name": "<concise name>"}}')
        nm = call(NSYS, u, "name", str)
        return g, (nm or g).strip()
    with ThreadPoolExecutor(max_workers=8) as ex:
        raw = dict(ex.map(gen_name, to_name))
    mapping, used = {}, set()
    for g in final_groups:
        nm = raw.get(g, g)
        base, k = nm, 2
        while nm in used:
            nm = f"{base} ({k})"; k += 1
        used.add(nm)
        mapping[g] = nm
    print(f"renamed {len(to_name)} groups", flush=True)

    # ---------- stage 3: merge duplicates ----------
    print("=== stage 3: MERGE ===", flush=True)
    names = [mapping[g] for g in final_groups]
    V = emb.encode(names, normalize_embeddings=True)
    S = V @ V.T
    np.fill_diagonal(S, -1)
    pairs = set()
    for i in range(len(names)):
        j = int(np.argmax(S[i]))
        if S[i, j] >= 0.55:
            pairs.add((min(i, j), max(i, j)))
    def djudge(p):
        a, b = names[p[0]], names[p[1]]
        u = (f"Skill A: {a}\nSkill B: {b}\n\n"
             'Near-duplicates (essentially the same cognitive operation)? Return JSON {"duplicate": true|false}.')
        return p, bool(call(DSYS, u, "duplicate", bool))
    with ThreadPoolExecutor(max_workers=8) as ex:
        dres = list(ex.map(djudge, sorted(pairs)))
    parent = list(range(len(names)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    n_merged = 0
    for (i, j), same in dres:
        if same:
            if find(i) != find(j):
                parent[find(i)] = find(j)
                n_merged += 1
    glist = list(final_groups)
    canon_items = defaultdict(list)
    canon_evid = defaultdict(list)
    root_name = {}
    for idx, g in enumerate(glist):
        r = find(idx)
        root_name.setdefault(r, mapping[glist[r]])
        canon_items[root_name[r]] += final_groups[g]
        canon_evid[root_name[r]] += evidence.get(g, [])
    print(f"judged {len(dres)} nearest pairs; merged {n_merged}; "
          f"final skills: {len(canon_items)}", flush=True)

    # save repaired taxonomy
    item2 = defaultdict(list)
    for nm, items in canon_items.items():
        for i in set(items):
            item2[i].append(nm)
    out = {"bank": [{"name": n, "definition": ""} for n in sorted(canon_items)],
           "per_item": [{"item_idx": i, "benchmark": meta[i][0], "subtask": meta[i][1],
                         "skills": sorted(item2[i])} for i in sorted(item2)],
           "evidence": {n: sorted(set(v)) for n, v in canon_evid.items()}}
    json.dump(out, open(E / "oldtax_repaired.json", "w"))
    print(f"wrote oldtax_repaired.json ({len(canon_items)} skills, {len(item2)} items)", flush=True)

    if SMOKE:
        print("SMOKE done (skipping stage 4)")
        return

    # ---------- stage 4: measure ----------
    print("=== stage 4: MEASURE ===", flush=True)
    skills = sorted(canon_items)
    # validity (10 pairs each, exclude evidence)
    vtasks = []
    for sk in skills:
        pool = [i for i in set(canon_items[sk]) if i not in set(canon_evid.get(sk, []))] \
               or list(set(canon_items[sk]))
        for i in random.sample(pool, min(10, len(pool))):
            vtasks.append((sk, i))
    def vj(t):
        sk, i = t
        u = (f"Item:\n{qtext.get(i,'')}\n\nSkill: {sk}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return sk, call(VSYS, u, "verdict", lambda x: x == "yes")
    with ThreadPoolExecutor(max_workers=8) as ex:
        vres = list(ex.map(vj, vtasks))
    vy = defaultdict(int); vn = defaultdict(int)
    for sk, ok in vres:
        if ok is not None:
            vn[sk] += 1; vy[sk] += ok
    # group test (6 trials each) on repaired membership
    item_sk2 = defaultdict(set)
    for i, ns in item2.items():
        item_sk2[i].update(ns)
    def gt(sk):
        uniq = list(set(canon_items[sk]))
        if len(uniq) < 4:
            return sk, None
        hits = 0
        for _ in range(6):
            own = random.sample(uniq, 4)
            intr = random.choice([i for i in items_all if sk not in item_sk2[i]])
            five = own + [intr]; random.shuffle(five)
            truth = five.index(intr) + 1
            body = "\n".join(f"{k+1}. {qtext.get(i,'')[:400]}" for k, i in enumerate(five))
            u = ("These five test questions should all belong to one group that tests the same "
                 f"skill. Exactly one does NOT belong. Which number is the odd one out?\n\n{body}\n\n"
                 'Return JSON {"intruder": <number 1-5>}.')
            hits += (call(ISYS, u, "intruder", int) == truth)
        return sk, hits / 6
    with ThreadPoolExecutor(max_workers=8) as ex:
        gres = dict(ex.map(gt, skills))
    # granularity (with 2 examples, as before)
    def gr(sk):
        ex2 = "\n".join(f"- {qtext.get(i,'')[:140]}" for i in list(set(canon_items[sk]))[:2])
        u = (f"Skill: {sk}\nExample items:\n{ex2}\n\n"
             "Rate granularity: 'too_broad' (vague catch-all like logical_reasoning), "
             "'appropriate' (a specific reusable cognitive operation), or 'too_narrow' "
             '(tied to one specific item). Return JSON {"grain":"too_broad|appropriate|too_narrow"}')
        return sk, call(GSYS, u, "grain", str)
    with ThreadPoolExecutor(max_workers=8) as ex:
        grres = dict(ex.map(gr, skills))

    val = {s: vy[s] / vn[s] for s in skills if vn.get(s)}
    coh = {s: v for s, v in gres.items() if v is not None}
    import statistics as st
    print(f"\nREPAIRED taxonomy: {len(skills)} skills")
    print(f"validity:  mean {st.mean(val.values()):.0%}   <=50%: {sum(1 for v in val.values() if v <= .5)}")
    print(f"group test: mean {st.mean(coh.values()):.0%}  <=50%: {sum(1 for v in coh.values() if v <= .5)}")
    gc = defaultdict(int)
    for s, g in grres.items():
        gc[g or "?"] += 1
    print(f"granularity: appropriate {gc['appropriate']}/{len(skills)}")
    print(f"merge stage residual duplicates: {n_merged} merged during repair")
    json.dump({"validity": val, "group_test": coh,
               "granularity": {s: grres[s] for s in skills}},
              open(E / "oldtax_repaired_metrics.json", "w"), indent=1)
    print("wrote oldtax_repaired_metrics.json", flush=True)


if __name__ == "__main__":
    main()
