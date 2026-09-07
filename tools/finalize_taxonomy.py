"""Finalize the repaired taxonomy:
  1 lowercase all names (free), rewrite remaining contract violators (LLM)
  2 re-pile the 3 catch-alls fresh (compact one-call, batched fallback, sanity checks)
  3 re-measure only what changed; write FINAL v2 files

    .venv312/bin/python tools/finalize_taxonomy.py
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
BIG = "gpt-4.1-mini"
SMALL = "gpt-4o-mini"
random.seed(45)
CATCHALLS = ["Calculate", "Evaluate", "judge deductive validity"]
BANNED = ("problem solving", "reasoning", "analysis", "skills", "advanced")

NSYS = ("You rename one skill in a test-skill taxonomy so that the name follows a strict "
        "format. You are given the skill's questions; the name must describe what they require.")
PSYS = ("You sort test questions into groups by the cognitive skill each requires. You judge "
        "by the mental operation needed to solve a question, never by surface topic words.")
ISYS = "You find the one item that does NOT belong with the others."
VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."

RULES = """Requirements:
1. Judge by the operation needed to solve. Same topic does not mean same skill; different wording can be the same skill.
2. Piles are mutually exclusive: every question in exactly one pile.
3. Each pile is ONE specific operation; if a pile needs "and" or "or" to describe, split it.
4. Use FEW piles (between 2 and 25 for a group like this). Single-question piles only as a truly last resort.
5. Pile names: lowercase verb phrase, one verb plus one specific object, 3-8 words, no "and"/"or"/slashes, never the words "problem solving", "reasoning", "analysis", "skills", "advanced".
6. Description: at most 12 words."""

COMPACT_U = """Below are {n} test questions, numbered. Sort EVERY question into piles so that all questions in one pile require the SAME specific cognitive skill.

{rules}

Questions:
{questions}

Output format: PLAIN TEXT, one line per pile, exactly like this:
1) name of operation | short description | 3 7 12 15
No JSON, no other text. Every question number from 1 to {n} must appear on exactly one line."""

DISC_U = """Below are {n} test questions, numbered, sampled from a larger group. Sort EVERY question into piles so that all questions in one pile require the SAME specific cognitive skill.

{rules}

Questions:
{questions}

Return JSON only: {{"piles": [{{"name": "...", "description": "...", "questions": [1,4,7]}}, ...]}}"""

ASSIGN_U = """Here is a table of skill piles:
{table}

Below are test questions, each with an id. For EACH question, choose the pile whose operation it requires.{fallback}

Questions:
{questions}

Output format: PLAIN TEXT, one line per question, exactly:
<question id> -> <pile number>
No other text."""

REWRITE_U = """This skill currently has a name that violates our format. Write a new name for it based on its questions.

Questions tagged with this skill:
{ev}

Current name (for context only, do not copy its style): {old}

Name format, all rules mandatory:
1. A lowercase verb phrase: one verb plus one specific object, for example "track object positions after swaps".
2. Between 3 and 8 words. A bare verb alone is forbidden.
3. No "and", no "or", no slashes. If the questions span two operations, name the one most of them require.
4. Never use these words: "problem solving", "reasoning", "analysis", "skills", "advanced".
5. Do not name a subject area - name the operation.
{extra}
Return JSON only: {{"name": "<new name>"}}"""


def violations(name):
    v = []
    w = name.replace("_", " ").split()
    if name != name.lower():
        v.append("not lowercase")
    if len(w) < 3:
        v.append("fewer than 3 words")
    if len(w) > 8:
        v.append("more than 8 words")
    low = " " + name.lower() + " "
    if " and " in low or " or " in low or "/" in name:
        v.append("contains and/or/slash")
    for b in BANNED:
        if b in low:
            v.append(f"banned word '{b}'")
    return v


def degenerate(piles, n):
    if not piles:
        return True
    singles = sum(1 for p in piles if len(p["nums"]) == 1)
    return len(piles) > 0.5 * n or (len(piles) >= 5 and singles / len(piles) > 0.6)


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
    table = {b["name"]: {"desc": b.get("definition", ""), "items": sorted(cur[b["name"]])}
             for b in fin["bank"] if cur[b["name"]]}
    gt = dict(met["group_test"])
    va = dict(met["validity"])

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

    def call_json(model, system, user, field, max_out=8000):
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
                    print(f"  ! json call failed: {str(ex)[:100]}", flush=True)
        return None

    LINE = re.compile(r"^\s*\d+\)\s*(.+?)\s*\|\s*(.*?)\s*\|\s*([\d\s]+)$")

    def attempt_A(order):
        qs = "\n".join(f"{k+1}. {qtext.get(i,'')[:900]}" for k, i in enumerate(order))
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
                piles.append({"name": m.group(1).strip().lower(), "desc": m.group(2).strip()[:200],
                              "nums": nums})
        missing = [n for n in range(1, len(order) + 1) if n not in seen]
        if len(missing) > max(3, 0.05 * len(order)):
            return None
        for n in missing:
            piles.append({"name": f"unplaced question {n}", "desc": "", "nums": [n]})
        return None if degenerate(piles, len(order)) else piles

    def attempt_B(order):
        sample_n = min(150, len(order))
        sample = random.sample(range(1, len(order) + 1), sample_n)
        qs = "\n".join(f"{j+1}. {qtext.get(order[n-1],'')[:600]}" for j, n in enumerate(sample))
        txt = call_text(BIG, PSYS, DISC_U.format(n=sample_n, rules=RULES, questions=qs), max_out=15000)
        try:
            res = json.loads(txt[txt.index("{"):txt.rindex("}")+1])
        except Exception:  # noqa: BLE001
            return None
        scheme = [{"name": str(p.get("name", "")).strip().lower() or "unnamed",
                   "desc": str(p.get("description", ""))[:200],
                   "nums": [sample[j-1] for j in p.get("questions", []) if 1 <= j <= sample_n]}
                  for p in res.get("piles", [])]
        scheme = [s for s in scheme if s["nums"]]
        if not scheme or degenerate(scheme, sample_n):
            return None
        tbl = "\n".join(f"{k+1}) {s['name']} | {s['desc']}" for k, s in enumerate(scheme))
        assigned = {k: list(s["nums"]) for k, s in enumerate(scheme)}
        rest = [n for n in range(1, len(order) + 1)
                if n not in {x for v in assigned.values() for x in v}]
        def batch(chunk, force):
            fb = (" You MUST choose the closest pile; 0 is not allowed." if force
                  else " If none fits, use 0.")
            qs = "\n".join(f"{n}. {qtext.get(order[n-1],'')[:600]}" for n in chunk)
            txt = call_text(SMALL, PSYS, ASSIGN_U.format(table=tbl, questions=qs, fallback=fb),
                            max_out=2000)
            out = {}
            for line in txt.splitlines():
                m = re.match(r"^\s*(\d+)\s*->\s*(\d+)", line)
                if m:
                    out[int(m.group(1))] = int(m.group(2))
            return out
        unplaced = []
        chunks = [rest[i:i+40] for i in range(0, len(rest), 40)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res_map, chunk in zip(ex.map(lambda c: batch(c, False), chunks), chunks):
                for n in chunk:
                    p = res_map.get(n, 0)
                    (assigned[p-1].append(n) if 1 <= p <= len(scheme) else unplaced.append(n))
        if unplaced:
            still = []
            chunks = [unplaced[i:i+40] for i in range(0, len(unplaced), 40)]
            with ThreadPoolExecutor(max_workers=8) as ex:
                for res_map, chunk in zip(ex.map(lambda c: batch(c, True), chunks), chunks):
                    for n in chunk:
                        p = res_map.get(n, 0)
                        (assigned[p-1].append(n) if 1 <= p <= len(scheme) else still.append(n))
            unplaced = still
        piles = [{"name": scheme[k]["name"], "desc": scheme[k]["desc"], "nums": sorted(set(v))}
                 for k, v in assigned.items() if v]
        for n in unplaced:
            piles.append({"name": f"unplaced question {n}", "desc": "", "nums": [n]})
        return None if degenerate(piles, len(order)) else piles

    # ---------- 1. re-pile the catch-alls ----------
    print("=== re-pile the 3 catch-alls ===", flush=True)
    new_piles = {}
    for T in CATCHALLS:
        if T not in table:
            continue
        order = table[T]["items"]
        piles = attempt_A(order)
        how = "A"
        if piles is None:
            piles = attempt_B(order)
            how = "B"
        if piles is None:
            print(f"  !! {T}: both attempts degenerate, left as is", flush=True)
            continue
        del table[T]
        gt.pop(T, None); va.pop(T, None)
        for p in piles:
            nm, k = p["name"], 2
            while nm in table or nm in new_piles:
                nm = f"{p['name']} ({k})"; k += 1
            new_piles[nm] = {"desc": p["desc"], "items": [order[n-1] for n in p["nums"]]}
        print(f"  {T}: {len(order)} q -> {len(piles)} piles (attempt {how})", flush=True)
    table.update(new_piles)

    # ---------- 2. names: lowercase free fix, then rewrite violators ----------
    print("=== name contract ===", flush=True)
    lowered = {}
    for n in list(table):
        if n != n.lower() and not violations(n.lower()):
            nm, k = n.lower(), 2
            while nm in table:
                nm = f"{n.lower()} ({k})"; k += 1
            table[nm] = table.pop(n)
            lowered[n] = nm
    gt = {lowered.get(k, k): v for k, v in gt.items()}
    va = {lowered.get(k, k): v for k, v in va.items()}
    print(f"lowercased (free): {len(lowered)}", flush=True)

    subs = {n for n, v in table.items() if len(v["items"]) >= 4}
    bad = [n for n in subs if violations(n)]
    print(f"needing rewrite: {len(bad)}", flush=True)

    def rewrite(n):
        ev = "\n".join(f"- {qtext.get(i,'')[:280]}" for i in table[n]["items"][:8])
        nm = call_json(SMALL, NSYS, REWRITE_U.format(ev=ev, old=n, extra=""), "name", 300)
        nm = (nm or "").strip().lower()
        if not nm or violations(nm):
            why = "; ".join(violations(nm)) if nm else "empty"
            nm = call_json(SMALL, NSYS, REWRITE_U.format(
                ev=ev, old=n, extra=f"\nYour previous attempt {nm!r} violated: {why}. Fix exactly that.\n"),
                "name", 300)
            nm = (nm or "").strip().lower()
            if not nm or violations(nm):
                return n, None
        return n, nm
    with ThreadPoolExecutor(max_workers=8) as ex:
        rws = dict(ex.map(rewrite, bad))
    renamed, unfixed = {}, []
    for old, new in rws.items():
        if new:
            nm, k = new, 2
            while nm in table or nm in renamed.values():
                nm = f"{new} ({k})"; k += 1
            renamed[old] = nm
        else:
            unfixed.append(old)
    for old, new in renamed.items():
        table[new] = table.pop(old)
        if old in gt:
            gt[new] = gt.pop(old)     # group test is name-free: carries over
        va.pop(old, None)             # validity depends on the name: re-measure
    print(f"rewritten: {len(renamed)}; unfixable flagged: {len(unfixed)}", flush=True)

    # ---------- 3. re-measure what changed ----------
    need_v = sorted(set(renamed.values()) | {n for n in new_piles if n in table})
    need_g = sorted(n for n in new_piles if n in table and len(table[n]["items"]) >= 4)
    print(f"re-measuring: validity {len(need_v)}, group test {len(need_g)}", flush=True)

    def vmeas(n):
        y = t = 0
        for i in random.sample(table[n]["items"], min(10, len(table[n]["items"]))):
            u = (f"Item:\n{qtext.get(i,'')[:2500]}\n\nSkill: {n}\n\n"
                 'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
            t += 1
            y += (call_json(SMALL, VSYS, u, "verdict", 60) == "yes")
        return n, (y / t if t else None)
    def gmeas(n):
        uniq = table[n]["items"]
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
            hits += (call_json(SMALL, ISYS, u, "intruder", 100) == truth)
        return n, hits / 6
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, v in ex.map(vmeas, need_v):
            if v is not None:
                va[n] = v
        for n, g in ex.map(gmeas, need_g):
            gt[n] = g

    import statistics as st
    subs2 = {n for n, v in table.items() if len(v["items"]) >= 4}
    vv = [va[n] for n in subs2 if n in va]
    gg = [gt[n] for n in subs2 if n in gt]
    fails = [n for n in subs2 if gt.get(n) is not None and gt[n] <= 0.5]
    remaining_bad_names = [n for n in subs2 if violations(n)]
    print(f"\nFINAL v2: {len(table)} skills ({len(subs2)} substantive, "
          f"{len(remaining_bad_names)} names still non-compliant)")
    print(f"validity   mean {st.mean(vv):.0%}")
    print(f"group test mean {st.mean(gg):.0%}, failing {len(fails)}")

    item2 = defaultdict(list)
    for n, v in table.items():
        for i in v["items"]:
            item2[i].append(n)
    json.dump({"bank": [{"name": n, "definition": table[n]["desc"]} for n in sorted(table)],
               "per_item": [{"item_idx": i, "benchmark": meta[i][0], "subtask": meta[i][1],
                             "skills": sorted(set(item2[i]))} for i in sorted(item2)],
               "renamed": renamed, "lowered": lowered, "unfixed_names": unfixed},
              open(E / "oldtax_repaired_FINAL.json", "w"))
    json.dump({"validity": va, "group_test": gt, "uncertified": fails},
              open(E / "oldtax_repaired_FINAL_metrics.json", "w"), indent=1)
    print("updated oldtax_repaired_FINAL.json + metrics", flush=True)


if __name__ == "__main__":
    main()
