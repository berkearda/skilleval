"""The missing last stage: whole-list merge over the v2 taxonomy, then re-measure
merged skills only. Closes the duplicate ('(2)') problem.

    .venv312/bin/python tools/final_merge.py
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
BIG = "gpt-4.1-mini"
SMALL = "gpt-4o-mini"
random.seed(46)

DSYS = "You judge whether two cognitive-skill labels denote essentially the SAME operation."
ISYS = "You find the one item that does NOT belong with the others."
VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."

MERGE_U = """You will receive a list of skill piles, each with a name and (if present) a short description. Merge piles that name essentially the SAME cognitive operation. Similar topic words are not enough; different operations stay separate even if worded similarly. Names ending in "(2)", "(3)" etc. are collision suffixes: treat them as candidates to merge with their base name. Return "None" if no modification is needed.

Rules for the merged name: lowercase verb phrase, one verb plus one specific object, 3-8 words, no "and"/"or"/slashes, never "problem solving", "reasoning", "analysis", "skills", "advanced". Do NOT merge piles that merely share a verb: "calculate interest growth" and "calculate polygon area" are different operations.

[Pile list]
{plist}

Return JSON only: {{"merges": [{{"name": "...", "absorbed": ["...", "..."]}}], "none_needed": false}}"""


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    fin = json.load(open(E / "oldtax_repaired_FINAL.json"))
    met = json.load(open(E / "oldtax_repaired_FINAL_metrics.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {pi["item_idx"]: (pi["benchmark"], pi["subtask"]) for pi in fin["per_item"]}
    items_all = list(meta)
    cur = defaultdict(set)
    for pi in fin["per_item"]:
        for s in pi["skills"]:
            cur[s].add(pi["item_idx"])
    desc = {b["name"]: b.get("definition", "") for b in fin["bank"]}
    table = {n: {"desc": desc.get(n, ""), "items": sorted(v)} for n, v in cur.items()}
    gt = dict(met["group_test"]); va = dict(met["validity"])

    from openai import OpenAI
    client = OpenAI(api_key=load_key(), timeout=600)

    def call(model, system, user, field, max_out=12000):
        for a in range(2):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=max_out,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return json.loads(r.choices[0].message.content).get(field)
            except Exception as ex:  # noqa: BLE001
                if a == 1:
                    print(f"  ! call failed: {str(ex)[:100]}", flush=True)
        return None

    # merge over substantive skills only (dust stays as residue)
    subs = sorted(n for n, v in table.items() if len(v["items"]) >= 4)
    plist = "\n".join(f"{n}: {table[n]['desc'] or '(no description)'}" for n in subs)
    merges = call(BIG, DSYS, MERGE_U.format(plist=plist), "merges") or []
    canon = {}
    for m in merges:
        tgt = str(m.get("name", "")).strip().lower()
        for ab in m.get("absorbed", []) or []:
            if ab in table and tgt and ab.lower() != tgt:
                canon[ab] = tgt
    print(f"merge decisions: {len(canon)} names absorbed into {len(set(canon.values()))} targets")
    for ab, tgt in sorted(canon.items()):
        print(f"  {ab[:50]:50s} -> {tgt[:45]}")

    merged_targets = set()
    newtab = defaultdict(lambda: {"desc": "", "items": []})
    for n, v in table.items():
        tgt = canon.get(n, n)
        if n in canon:
            merged_targets.add(tgt)
        newtab[tgt]["items"] += v["items"]
        newtab[tgt]["desc"] = newtab[tgt]["desc"] or v["desc"]
    table = {n: {"desc": v["desc"], "items": sorted(set(v["items"]))} for n, v in newtab.items()}
    for n in list(gt):
        if n in canon:
            gt.pop(n)
    for n in list(va):
        if n in canon:
            va.pop(n)

    # re-measure merged targets
    def gmeas(n):
        uniq = table[n]["items"]
        if len(uniq) < 4:
            return n, None
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
            hits += (call(SMALL, ISYS, u, "intruder", 100) == truth)
        return n, hits / 6
    def vmeas(n):
        y = t = 0
        for i in random.sample(table[n]["items"], min(10, len(table[n]["items"]))):
            u = (f"Item:\n{qtext.get(i,'')[:2500]}\n\nSkill: {n}\n\n"
                 'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
            t += 1
            y += (call(SMALL, VSYS, u, "verdict", 60) == "yes")
        return n, (y / t if t else None)
    targets = sorted(n for n in merged_targets if n in table)
    print(f"re-measuring {len(targets)} merged skills")
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, g in ex.map(gmeas, targets):
            if g is not None:
                gt[n] = g
        for n, v in ex.map(vmeas, targets):
            if v is not None:
                va[n] = v

    import statistics as st
    subs2 = {n for n, v in table.items() if len(v["items"]) >= 4}
    vv = [va[n] for n in subs2 if n in va]
    gg = [gt[n] for n in subs2 if n in gt]
    fails = [n for n in subs2 if gt.get(n) is not None and gt[n] <= 0.5]
    print(f"\nFINAL v3: {len(table)} skills ({len(subs2)} substantive)")
    print(f"validity   mean {st.mean(vv):.0%}")
    print(f"group test mean {st.mean(gg):.0%}, failing {len(fails)}")

    item2 = defaultdict(list)
    for n, v in table.items():
        for i in v["items"]:
            item2[i].append(n)
    json.dump({"bank": [{"name": n, "definition": table[n]["desc"]} for n in sorted(table)],
               "per_item": [{"item_idx": i, "benchmark": meta[i][0], "subtask": meta[i][1],
                             "skills": sorted(set(item2[i]))} for i in sorted(item2)],
               "merged": canon},
              open(E / "oldtax_repaired_FINAL.json", "w"))
    json.dump({"validity": va, "group_test": gt, "uncertified": fails},
              open(E / "oldtax_repaired_FINAL_metrics.json", "w"), indent=1)
    print("updated FINAL files", flush=True)


if __name__ == "__main__":
    main()
