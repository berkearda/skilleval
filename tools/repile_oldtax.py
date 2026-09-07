"""Final repair design: rename the 40, re-pile the 25 (one full-text call each,
gpt-4.1-mini), whole-list merge, group-test certification, 4-metric measurement.

    .venv312/bin/python tools/repile_oldtax.py --smoke   (1 small pile + 2 renames)
    .venv312/bin/python tools/repile_oldtax.py
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
E = REPO / "cdm_exploration/experiments"
SMOKE = "--smoke" in sys.argv
BIG = "gpt-4.1-mini"
SMALL = "gpt-4o-mini"
random.seed(42)

NSYS = "You name skill groups from their member items. You are not shown any existing name."
PSYS = ("You sort test questions into groups by the cognitive skill each requires. You judge "
        "by the mental operation needed to solve a question, never by surface topic words.")
ISYS = "You find the one item that does NOT belong with the others."
VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."
GSYS = "You rate the granularity of a cognitive-skill label for a skill taxonomy."
DSYS = "You judge whether two cognitive-skill labels denote essentially the SAME operation."

RENAME_U = ("These test items form one group in a skill taxonomy:\n{ev}\n\n"
            "Name the specific cognitive OPERATION this group requires, as a short English verb "
            "phrase. Do NOT name a subject area. Do NOT use the words \"problem solving\", "
            "\"reasoning\", \"analysis\", \"skills\", \"advanced\". If the items do not share one "
            'operation, name the single most common one. Return JSON {{"name": "<concise name>"}}')

PILE_U = """Below are {n} test questions, numbered. Sort EVERY question into piles so that all questions in one pile require the SAME specific cognitive skill.

Requirements:
1. Judge by the operation needed to solve. Questions about the same topic (both mention fruit, both mention cars) do NOT belong together unless solving them needs the same operation. Questions worded very differently DO belong together if the operation is the same.
2. The piles must be a flat list of mutually exclusive groups: every question in exactly one pile.
3. Each pile must be describable as ONE specific operation. If describing a pile needs the word "and", split it into two piles.
4. Do not create vague piles such as "Other", "General", "Miscellaneous" or "Mixed". A question that truly fits nowhere may get its own single-question pile, but ONLY as a last resort: prefer the smallest number of piles such that each pile is still one operation.
5. Name each pile as a short verb phrase naming the operation (like "track object positions after swaps"). Do not name a subject area. Do not use the words "problem solving", "reasoning", "analysis", "skills", "advanced".
6. Give each pile a description of at most 12 words that differentiates it from the other piles.

Questions:
{questions}

Return JSON only, no other text:
{{"piles": [{{"name": "...", "description": "...", "questions": [1, 4, 7]}}, ...]}}
Every number from 1 to {n} must appear in exactly one pile."""

MERGE_U = """You will receive a list of skill piles, each with a name and a one-line description. Merge piles that name essentially the SAME cognitive operation. Similar topic words are not enough; different operations stay separate even if worded similarly. Return "None" if no modification is needed.
When merging, output the updated name and description, followed by the original pile names it absorbed.

[Pile list]
{plist}

Return JSON only: {{"merges": [{{"name": "...", "description": "...", "absorbed": ["...", "..."]}}], "none_needed": false}}"""


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    lab = json.load(open(E / "oldtax_full_format.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {pi["item_idx"]: (pi["benchmark"], pi["subtask"]) for pi in lab["per_item"]}
    sk_items = defaultdict(list)
    item_skills = defaultdict(set)
    for pi in lab["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
            item_skills[pi["item_idx"]].add(s)
    items_all = list(meta)
    wl = json.load(open(E / "oldtax_repair_worklist_v2.json"))["piles"]
    if SMOKE:
        small = sorted(wl["SPLIT"], key=lambda s: len(sk_items[s]))[:1]
        wl = {"SPLIT": small, "RENAME": wl["RENAME"][:2], "CHECK-TINY": [], "OK": []}

    from openai import OpenAI
    client = OpenAI(api_key=load_key(), timeout=900)

    def call(model, system, user, max_out=4000):
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=max_out,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return json.loads(r.choices[0].message.content)
            except Exception as ex:  # noqa: BLE001
                if a == 2:
                    print(f"  ! call failed: {str(ex)[:120]}", flush=True)
                    return {}
        return {}

    # ---------- Track A: rename the 40 ----------
    print(f"=== Track A: rename {len(wl['RENAME'])} clusters ===", flush=True)
    evidence = {}
    def rn(sk):
        ev_items = random.sample(list(set(sk_items[sk])), min(8, len(set(sk_items[sk]))))
        evidence[sk] = ev_items
        ev = "\n".join(f"- {qtext.get(i,'')[:300]}" for i in ev_items)
        nm = call(SMALL, NSYS, RENAME_U.format(ev=ev))
        return sk, str(nm.get("name", sk)).strip() or sk
    with ThreadPoolExecutor(max_workers=8) as ex:
        renames = dict(ex.map(rn, wl["RENAME"]))
    print(f"renamed {len(renames)}", flush=True)

    # ---------- Track B1: pile discovery, one call per cluster ----------
    print(f"=== Track B1: discover piles in {len(wl['SPLIT'])} clusters ===", flush=True)
    piles = {}          # (cluster, pile_name) -> {"desc":..., "items":[...]}
    for sk in wl["SPLIT"]:
        order = sorted(set(sk_items[sk]))
        qs = "\n".join(f"{k+1}. {qtext.get(i,'')}" for k, i in enumerate(order))
        u = PILE_U.format(n=len(order), questions=qs)
        res = call(BIG, PSYS, u, max_out=30000)
        def partition_ok(res):
            got = sorted(n for p in res.get("piles", []) for n in p.get("questions", []))
            return got == list(range(1, len(order) + 1))
        if not partition_ok(res):
            u2 = u + "\n\nYour previous answer did not place every number exactly once. Try again carefully."
            res = call(BIG, PSYS, u2, max_out=30000)
        if not partition_ok(res):
            print(f"  ! {sk[:40]}: partition failed twice, keeping cluster whole", flush=True)
            piles[(sk, sk)] = {"desc": "", "items": order}
            continue
        for p in res["piles"]:
            nm = str(p.get("name", "")).strip() or "unnamed"
            piles[(sk, nm)] = {"desc": str(p.get("description", ""))[:200],
                               "items": [order[n-1] for n in p["questions"]]}
        print(f"  {sk[:46]}: {len(order)} q -> {len(res['piles'])} piles", flush=True)
    print(f"total new piles: {len(piles)}", flush=True)

    # ---------- assemble pre-merge skill table ----------
    table = {}          # name -> {"desc":..., "items":[...]}
    for sk in wl["OK"] + wl["CHECK-TINY"]:
        table[sk] = {"desc": "", "items": sorted(set(sk_items[sk]))}
    for sk, nm in renames.items():
        base, k = nm, 2
        while nm in table:
            nm = f"{base} ({k})"; k += 1
        table[nm] = {"desc": "", "items": sorted(set(sk_items[sk])), "renamed_from": sk}
    for (sk, nm), v in piles.items():
        base, k = nm, 2
        while nm in table:
            nm = f"{base} ({k})"; k += 1
        table[nm] = {"desc": v["desc"], "items": v["items"], "from_cluster": sk}

    # ---------- Track B2: whole-list merge ----------
    if not SMOKE:
        print("=== Track B2: whole-list merge ===", flush=True)
        plist = "\n".join(f"{n}: {v['desc'] or '(no description)'}" for n, v in sorted(table.items()))
        mres = call(BIG, DSYS, MERGE_U.format(plist=plist), max_out=8000)
        canon = {}
        for m in mres.get("merges", []) or []:
            tgt = str(m.get("name", "")).strip()
            for ab in m.get("absorbed", []) or []:
                if ab in table and tgt:
                    canon[ab] = tgt
        merged = defaultdict(lambda: {"desc": "", "items": []})
        for n, v in table.items():
            tgt = canon.get(n, n)
            merged[tgt]["items"] += v["items"]
            merged[tgt]["desc"] = merged[tgt]["desc"] or v["desc"]
        table = {n: {"desc": v["desc"], "items": sorted(set(v["items"]))} for n, v in merged.items()}
        print(f"merge decisions: {len(canon)} absorbed; final skills: {len(table)}", flush=True)

    # ---------- Track B3: certify piles with the group test ----------
    print("=== Track B3: certify with the group test ===", flush=True)
    def gtest(name, items, trials=6):
        uniq = list(set(items))
        if len(uniq) < 4:
            return None
        others = [i for i in items_all if i not in set(uniq)]
        hits = 0
        for _ in range(trials):
            own = random.sample(uniq, 4)
            intr = random.choice(others)
            five = own + [intr]; random.shuffle(five)
            truth = five.index(intr) + 1
            body = "\n".join(f"{k+1}. {qtext.get(i,'')[:400]}" for k, i in enumerate(five))
            u = ("These five test questions should all belong to one group that tests the same "
                 f"skill. Exactly one does NOT belong. Which number is the odd one out?\n\n{body}\n\n"
                 'Return JSON {"intruder": <number 1-5>}.')
            got = call(SMALL, ISYS, u, max_out=100).get("intruder")
            hits += (got == truth)
        return hits / trials
    with ThreadPoolExecutor(max_workers=8) as ex:
        certs = dict(ex.map(lambda n: (n, gtest(n, table[n]["items"])), list(table)))
    fails = [n for n, c in certs.items() if c is not None and c <= 0.5]
    print(f"certified: {sum(1 for c in certs.values() if c is not None and c > 0.5)} pass, "
          f"{len(fails)} fail, {sum(1 for c in certs.values() if c is None)} too small", flush=True)

    out = {"bank": [{"name": n, "definition": table[n]["desc"]} for n in sorted(table)],
           "per_item": None, "certification": certs,
           "uncertified": fails,
           "evidence": {renames.get(k, k): v for k, v in evidence.items()}}
    item2 = defaultdict(list)
    for n, v in table.items():
        for i in v["items"]:
            item2[i].append(n)
    out["per_item"] = [{"item_idx": i, "benchmark": meta[i][0], "subtask": meta[i][1],
                        "skills": sorted(set(item2[i]))} for i in sorted(item2)]
    json.dump(out, open(E / ("repiled_smoke.json" if SMOKE else "oldtax_repiled.json"), "w"))
    print(f"wrote {'repiled_smoke' if SMOKE else 'oldtax_repiled'}.json "
          f"({len(table)} skills, {len(item2)} items)", flush=True)
    if SMOKE:
        return

    # ---------- final measurement ----------
    print("=== MEASURE: 4 metrics on the repaired taxonomy ===", flush=True)
    skills = sorted(table)
    ev_by_name = out["evidence"]
    def vj(t):
        sk, i = t
        u = (f"Item:\n{qtext.get(i,'')}\n\nSkill: {sk}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return sk, call(SMALL, VSYS, u, max_out=50).get("verdict") == "yes"
    vtasks = []
    for sk in skills:
        pool = [i for i in table[sk]["items"] if i not in set(ev_by_name.get(sk, []))] or table[sk]["items"]
        for i in random.sample(pool, min(10, len(pool))):
            vtasks.append((sk, i))
    with ThreadPoolExecutor(max_workers=8) as ex:
        vres = list(ex.map(vj, vtasks))
    vy = defaultdict(int); vn = defaultdict(int)
    for sk, ok in vres:
        vn[sk] += 1; vy[sk] += ok
    def gr(sk):
        ex2 = "\n".join(f"- {qtext.get(i,'')[:140]}" for i in table[sk]["items"][:2])
        u = (f"Skill: {sk}\nExample items:\n{ex2}\n\n"
             "Rate granularity: 'too_broad' (vague catch-all like logical_reasoning), 'appropriate' "
             "(a specific reusable cognitive operation), or 'too_narrow' (tied to one specific item). "
             'Return JSON {"grain":"too_broad|appropriate|too_narrow"}')
        return sk, call(SMALL, GSYS, u, max_out=50).get("grain")
    with ThreadPoolExecutor(max_workers=8) as ex:
        grres = dict(ex.map(gr, skills))

    import statistics as st
    val = {s: vy[s]/vn[s] for s in skills if vn.get(s)}
    coh = {s: c for s, c in certs.items() if c is not None}
    gc = defaultdict(int)
    for s, g in grres.items():
        gc[g or "?"] += 1
    print(f"\nREPAIRED: {len(skills)} skills")
    print(f"validity   mean {st.mean(val.values()):.0%}  <=50%: {sum(1 for v in val.values() if v<=.5)}")
    print(f"group test mean {st.mean(coh.values()):.0%}  <=50%: {len(fails)}")
    print(f"granularity appropriate {gc['appropriate']}/{len(skills)}")
    json.dump({"validity": val, "group_test": coh,
               "granularity": grres, "uncertified": fails},
              open(E / "oldtax_repiled_metrics.json", "w"), indent=1)
    print("wrote oldtax_repiled_metrics.json", flush=True)


if __name__ == "__main__":
    main()
