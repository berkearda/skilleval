"""Rescue the clusters whose one-call piling failed twice in pass 2.

Attempt A: one call, COMPACT plain-text output (no JSON) + tolerant parsing
           (a few missing numbers become singleton piles instead of failing).
Attempt B: discover piles on a 150-question sample, then assign the rest in
           batches of 40 (TnT-LLM style). Every question still judged by reading.

Then: rebuild the repaired taxonomy, whole-list merge, certify, measure.

    .venv312/bin/python tools/rescue_piles.py --smoke   (smallest failed cluster, attempt A only)
    .venv312/bin/python tools/rescue_piles.py
"""
from __future__ import annotations

import json
import random
import re
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

PSYS = ("You sort test questions into groups by the cognitive skill each requires. You judge "
        "by the mental operation needed to solve a question, never by surface topic words.")
ISYS = "You find the one item that does NOT belong with the others."
VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."
GSYS = "You rate the granularity of a cognitive-skill label for a skill taxonomy."
DSYS = "You judge whether two cognitive-skill labels denote essentially the SAME operation."

RULES = """Requirements:
1. Judge by the operation needed to solve. Questions about the same topic do NOT belong together unless solving them needs the same operation. Questions worded very differently DO belong together if the operation is the same.
2. The piles must be mutually exclusive: every question in exactly one pile.
3. Each pile must be describable as ONE specific operation. If describing a pile needs the word "and", split it.
4. No vague piles ("Other", "General", "Miscellaneous", "Mixed"). Single-question piles only as a last resort: prefer the smallest number of piles such that each pile is still one operation.
5. Name each pile as a short verb phrase naming the operation. No subject-area names. Never use the words "problem solving", "reasoning", "analysis", "skills", "advanced".
6. Description: at most 12 words, differentiating the pile from the others."""

COMPACT_U = """Below are {n} test questions, numbered. Sort EVERY question into piles so that all questions in one pile require the SAME specific cognitive skill.

{rules}

Questions:
{questions}

Output format: PLAIN TEXT, one line per pile, exactly like this:
1) name of operation | short description | 3 7 12 15
2) another operation | short description | 1 2 4
No JSON, no other text. Every question number from 1 to {n} must appear on exactly one line."""

DISC_U = """Below are {n} test questions, numbered, sampled from a larger group. Sort EVERY question into piles so that all questions in one pile require the SAME specific cognitive skill.

{rules}

Questions:
{questions}

Return JSON only: {{"piles": [{{"name": "...", "description": "...", "questions": [1,4,7]}}, ...]}}
Every number from 1 to {n} must appear in exactly one pile."""

ASSIGN_U = """Here is a table of skill piles:
{table}

Below are test questions, each with an id. For EACH question, choose the pile whose operation it requires. If none fits, use 0.

Questions:
{questions}

Output format: PLAIN TEXT, one line per question, exactly:
<question id> -> <pile number>
No other text."""

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
    for pi in lab["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
    items_all = list(meta)
    wl = json.load(open(E / "oldtax_repair_worklist_v2.json"))["piles"]
    rep = json.load(open(E / "oldtax_repiled.json"))

    # failed clusters = SPLIT names that survived as skills in the pass-2 bank
    bank_names = {b["name"] for b in rep["bank"]}
    failed = [s for s in wl["SPLIT"] if s in bank_names]
    if SMOKE:
        failed = sorted(failed, key=lambda s: len(set(sk_items[s])))[:1]
    print(f"clusters to rescue: {len(failed)}", flush=True)

    from openai import OpenAI
    client = OpenAI(api_key=load_key(), timeout=900)

    def call_text(model, system, user, max_out=30000):
        for a in range(2):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=max_out,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return r.choices[0].message.content or ""
            except Exception as ex:  # noqa: BLE001
                if a == 1:
                    print(f"  ! text call failed: {str(ex)[:100]}", flush=True)
                    return ""
        return ""

    def call_json(model, system, user, max_out=8000):
        for a in range(2):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=max_out,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return json.loads(r.choices[0].message.content)
            except Exception as ex:  # noqa: BLE001
                if a == 1:
                    print(f"  ! json call failed: {str(ex)[:100]}", flush=True)
                    return {}
        return {}

    LINE = re.compile(r"^\s*\d+\)\s*(.+?)\s*\|\s*(.*?)\s*\|\s*([\d\s]+)$")

    def attempt_A(sk, order):
        qs = "\n".join(f"{k+1}. {qtext.get(i,'')}" for k, i in enumerate(order))
        txt = call_text(BIG, PSYS, COMPACT_U.format(n=len(order), rules=RULES, questions=qs))
        piles, seen = [], set()
        for line in txt.splitlines():
            m = LINE.match(line)
            if not m:
                continue
            nums = [int(x) for x in m.group(3).split() if x.isdigit()
                    and 1 <= int(x) <= len(order) and int(x) not in seen]
            seen.update(nums)
            if nums:
                piles.append({"name": m.group(1).strip(), "desc": m.group(2).strip()[:200],
                              "nums": nums})
        missing = [n for n in range(1, len(order) + 1) if n not in seen]
        if len(missing) > max(3, 0.05 * len(order)):
            return None
        for n in missing:                       # tolerate a few: singleton piles
            piles.append({"name": f"unplaced question {n}", "desc": "", "nums": [n]})
        return piles

    def attempt_B(sk, order):
        sample_n = min(150, len(order))
        sample = random.sample(range(1, len(order) + 1), sample_n)
        qs = "\n".join(f"{j+1}. {qtext.get(order[n-1],'')[:600]}" for j, n in enumerate(sample))
        res = call_json(BIG, PSYS, DISC_U.format(n=sample_n, rules=RULES, questions=qs),
                        max_out=15000)
        scheme = [{"name": str(p.get("name", "")).strip() or "unnamed",
                   "desc": str(p.get("description", ""))[:200]}
                  for p in res.get("piles", []) if p.get("questions")]
        if not scheme:
            return None
        assigned = defaultdict(list)            # pile_idx -> local nums
        for p, spec in zip(res["piles"], scheme):
            for j in p.get("questions", []):
                if 1 <= j <= sample_n:
                    assigned[scheme.index(spec)].append(sample[j - 1])
        rest = [n for n in range(1, len(order) + 1) if n not in {x for v in assigned.values() for x in v}]
        table = "\n".join(f"{k+1}) {s['name']} | {s['desc']}" for k, s in enumerate(scheme))
        def batch_assign(chunk):
            qs = "\n".join(f"{n}. {qtext.get(order[n-1],'')[:600]}" for n in chunk)
            txt = call_text(SMALL, PSYS, ASSIGN_U.format(table=table, questions=qs), max_out=2000)
            out = {}
            for line in txt.splitlines():
                m = re.match(r"^\s*(\d+)\s*->\s*(\d+)", line)
                if m:
                    out[int(m.group(1))] = int(m.group(2))
            return out
        chunks = [rest[i:i+40] for i in range(0, len(rest), 40)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res_map in ex.map(batch_assign, chunks):
                for n, p in res_map.items():
                    if 1 <= p <= len(scheme):
                        assigned[p - 1].append(n)
                    else:
                        assigned.setdefault(f"solo{n}", []).append(n)
        piles = []
        for k, s in enumerate(scheme):
            if assigned.get(k):
                piles.append({"name": s["name"], "desc": s["desc"], "nums": sorted(set(assigned[k]))})
        for key, v in assigned.items():
            if isinstance(key, str):
                piles.append({"name": f"unplaced question {v[0]}", "desc": "", "nums": v})
        got = sorted({n for p in piles for n in p["nums"]})
        for n in range(1, len(order) + 1):      # absolute coverage guarantee
            if n not in got:
                piles.append({"name": f"unplaced question {n}", "desc": "", "nums": [n]})
        return piles

    rescued = {}
    for sk in failed:
        order = sorted(set(sk_items[sk]))
        piles = attempt_A(sk, order)
        how = "A"
        if piles is None:
            piles = attempt_B(sk, order)
            how = "B"
        if piles is None:
            print(f"  !! {sk[:45]}: BOTH attempts failed, stays whole", flush=True)
            continue
        rescued[sk] = [{"name": p["name"], "desc": p["desc"],
                        "items": [order[n-1] for n in p["nums"]]} for p in piles]
        print(f"  {sk[:45]}: {len(order)} q -> {len(piles)} piles  (attempt {how})", flush=True)
    if SMOKE:
        print("SMOKE done")
        return

    # ---------- rebuild table: pass-2 table minus failed clusters, plus rescues ----------
    table = {}
    old_items = {b["name"]: None for b in rep["bank"]}
    per_item_old = rep["per_item"]
    name_items = defaultdict(list)
    for pi in per_item_old:
        for s in pi["skills"]:
            name_items[s].append(pi["item_idx"])
    for b in rep["bank"]:
        n = b["name"]
        if n in rescued:
            continue
        table[n] = {"desc": b.get("definition", ""), "items": sorted(set(name_items[n]))}
    for sk, plist in rescued.items():
        for p in plist:
            nm, k = p["name"], 2
            while nm in table:
                nm = f"{p['name']} ({k})"; k += 1
            table[nm] = {"desc": p["desc"], "items": p["items"]}
    print(f"table before merge: {len(table)} skills", flush=True)

    # ---------- whole-list merge ----------
    plist = "\n".join(f"{n}: {v['desc'] or '(no description)'}" for n, v in sorted(table.items()))
    mres = call_json(BIG, DSYS, MERGE_U.format(plist=plist), max_out=10000)
    canon = {}
    for m in mres.get("merges", []) or []:
        tgt = str(m.get("name", "")).strip()
        for ab in m.get("absorbed", []) or []:
            if ab in table and tgt and ab != tgt:
                canon[ab] = tgt
    merged = defaultdict(lambda: {"desc": "", "items": []})
    for n, v in table.items():
        tgt = canon.get(n, n)
        merged[tgt]["items"] += v["items"]
        merged[tgt]["desc"] = merged[tgt]["desc"] or v["desc"]
    table = {n: {"desc": v["desc"], "items": sorted(set(v["items"]))} for n, v in merged.items()}
    print(f"after merge: {len(table)} skills ({len(canon)} absorbed)", flush=True)

    # ---------- certify + measure ----------
    def gtest(args):
        name, items = args
        uniq = list(set(items))
        if len(uniq) < 4:
            return name, None
        others = [i for i in items_all if i not in set(uniq)]
        hits = 0
        for _ in range(6):
            own = random.sample(uniq, 4)
            intr = random.choice(others)
            five = own + [intr]; random.shuffle(five)
            truth = five.index(intr) + 1
            body = "\n".join(f"{k+1}. {qtext.get(i,'')[:400]}" for k, i in enumerate(five))
            u = ("These five test questions should all belong to one group that tests the same "
                 f"skill. Exactly one does NOT belong. Which number is the odd one out?\n\n{body}\n\n"
                 'Return JSON {"intruder": <number 1-5>}.')
            got = call_json(SMALL, ISYS, u, max_out=100).get("intruder")
            hits += (got == truth)
        return name, hits / 6
    with ThreadPoolExecutor(max_workers=8) as ex:
        certs = dict(ex.map(gtest, [(n, v["items"]) for n, v in table.items()]))

    def vj(t):
        sk, i = t
        u = (f"Item:\n{qtext.get(i,'')}\n\nSkill: {sk}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return sk, call_json(SMALL, VSYS, u, max_out=60).get("verdict") == "yes"
    vtasks = []
    for sk, v in table.items():
        for i in random.sample(v["items"], min(10, len(v["items"]))):
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
        return sk, call_json(SMALL, GSYS, u, max_out=60).get("grain")
    with ThreadPoolExecutor(max_workers=8) as ex:
        grres = dict(ex.map(gr, list(table)))

    import statistics as st
    val = {s: vy[s]/vn[s] for s in table if vn.get(s)}
    coh = {s: c for s, c in certs.items() if c is not None}
    fails = [s for s, c in coh.items() if c <= 0.5]
    gc = defaultdict(int)
    for s, g in grres.items():
        gc[g or "?"] += 1
    print(f"\nFINAL repaired taxonomy: {len(table)} skills")
    print(f"validity   mean {st.mean(val.values()):.0%}   <=50%: {sum(1 for v in val.values() if v<=.5)}")
    print(f"group test mean {st.mean(coh.values()):.0%}  testable {len(coh)}, failing {len(fails)}")
    print(f"granularity appropriate {gc['appropriate']}/{len(table)}")

    item2 = defaultdict(list)
    for n, v in table.items():
        for i in v["items"]:
            item2[i].append(n)
    json.dump({"bank": [{"name": n, "definition": table[n]["desc"]} for n in sorted(table)],
               "per_item": [{"item_idx": i, "benchmark": meta[i][0], "subtask": meta[i][1],
                             "skills": sorted(set(item2[i]))} for i in sorted(item2)],
               "certification": certs, "uncertified": fails},
              open(E / "oldtax_repiled_final.json", "w"))
    json.dump({"validity": val, "group_test": coh, "granularity": grres,
               "uncertified": fails},
              open(E / "oldtax_repiled_final_metrics.json", "w"), indent=1)
    print("wrote oldtax_repiled_final.json + oldtax_repiled_final_metrics.json", flush=True)


if __name__ == "__main__":
    main()
