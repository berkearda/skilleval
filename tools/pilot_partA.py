"""Part A scorecard: is this a GOOD skill taxonomy? (judged by reading, not by
model scores). Structural metrics are automatic; semantic metrics use an LLM
judge on bounded samples, plus a small sample emitted for human calibration.

  tagging validity   - does an item genuinely require the skill it is tagged with
  granularity        - is each skill a specific operation (not vague, not item-specific)
  intruder/coherence - can a judge spot an item that does not belong to a skill
  distinctiveness    - are the nearest skill pairs still near-duplicates by meaning

    .venv312/bin/python tools/pilot_partA.py \
        cdm_exploration/experiments/pilot_merged_drop.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
BANKF = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "cdm_exploration/experiments/pilot_merged_drop.json"
OUTF = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    REPO / "cdm_exploration/experiments/pilot_partA.json"
KEYFILE = Path.home() / ".cdmeval_openai_key"
RAW_BEFORE = 2565   # raw independent skill names on this subset (from analyzer)
MODEL = "gpt-4o-mini"
N_VALIDITY = 500    # (item, skill) pairs judged for tagging validity
N_INTRUDER = None   # skills given the intruder test; None = all with >=4 items
N_NEAREST = 100     # closest skill pairs judged for distinctiveness
random.seed(42)


def load_key():
    t = KEYFILE.read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def q(qtext, idx, n=380):
    return " ".join(qtext.get(idx, "").split())

.
def main():
    data = json.load(open(BANKF))
    defn = {b["name"]: b["definition"] for b in data["bank"]}
    qtext = {r["item_idx"]: r["question_full_text"]
             for r in json.load(open(D / "item_full_text_recovered.json"))}

    sk_items = defaultdict(list)
    item_skills = {}
    item_meta = {}
    for pi in data["per_item"]:
        item_skills[pi["item_idx"]] = pi["skills"]
        item_meta[pi["item_idx"]] = (pi["benchmark"], pi["subtask"])
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(system, user):
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return json.loads(r.choices[0].message.content)
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return {}
        return {}

    def run(tasks, fn):
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(fn, tasks))

    # ---------- structural (automatic) ----------
    n_items = len(item_skills)
    covered = sum(1 for s in item_skills.values() if s)
    sizes = [len(v) for v in sk_items.values()]
    largest = max(sizes) if sizes else 0

    # ---------- 1. tagging validity ----------
    pairs = []
    items = list(item_skills)
    while len(pairs) < N_VALIDITY:
        i = random.choice(items)
        if item_skills[i]:
            pairs.append((i, random.choice(item_skills[i])))
    VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."
    def vfn(p):
        i, sk = p
        b, sub = item_meta[i]
        d = defn.get(sk, "")
        dline = f"\nDefinition: {d}" if d else ""
        u = (f"Item ({b}/{sub}):\n{q(qtext,i)}\n\nSkill: {sk}{dline}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|partial|no"}.')
        return (p, judge(VSYS, u).get("verdict", "no"))
    vres = run(pairs, vfn)
    vc = Counter(v for _, v in vres)

    # ---------- 2. granularity (all skills) ----------
    GSYS = "You rate the granularity of a cognitive-skill label for a skill taxonomy."
    def gfn(sk):
        ex = "\n".join(f"- {q(qtext,i,140)}" for i in sk_items[sk][:2])
        dline = f"\nDefinition: {defn[sk]}" if defn.get(sk) else ""
        u = (f"Skill: {sk}{dline}\nExample items:\n{ex}\n\n"
             "Rate granularity: 'too_broad' (vague catch-all like logical_reasoning), "
             "'appropriate' (a specific reusable cognitive operation), or 'too_narrow' "
             '(tied to one specific item). Return JSON {"grain":"too_broad|appropriate|too_narrow"}.')
        return (sk, judge(GSYS, u).get("grain", "?"))
    gres = run(list(defn), gfn)
    gc = Counter(g for _, g in gres)

    # ---------- 3. intruder / coherence ----------
    big = [s for s in sk_items if len(set(sk_items[s])) >= 4]
    random.shuffle(big)
    if N_INTRUDER is not None:
        big = big[:N_INTRUDER]
    ISYS = "You find the one item that does NOT belong with the others."
    def ifn(sk):
        own = random.sample(list(set(sk_items[sk])), 4)
        pool = [i for i in items if sk not in item_skills[i]]
        intr = random.choice(pool)
        five = own + [intr]
        random.shuffle(five)
        truth = five.index(intr) + 1
        body = "\n".join(f"{k+1}. {q(qtext,i,160)}" for k, i in enumerate(five))
        dpar = f" ({defn[sk]})" if defn.get(sk) else ""
        u = (f"These items should all require the skill '{sk}'{dpar}. "
             f"Exactly one does NOT. Which number is the intruder?\n\n{body}\n\n"
             'Return JSON {"intruder": <number 1-5>}.')
        got = judge(ISYS, u).get("intruder", -1)
        return (sk, got == truth)
    ires = run(big, ifn)
    intr_acc = np.mean([ok for _, ok in ires]) if ires else float("nan")

    # ---------- 4. distinctiveness (nearest pairs) ----------
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    names = list(defn)
    V = emb.encode([f"{n}: {defn[n]}" if defn.get(n) else n for n in names],
                   normalize_embeddings=True)
    S = V @ V.T
    np.fill_diagonal(S, -1)
    cand = []
    for i in range(len(names)):
        j = int(np.argmax(S[i]))
        if i < j:
            cand.append((S[i, j], names[i], names[j]))
    cand.sort(reverse=True)
    near = cand[:N_NEAREST]
    DSYS = "You judge whether two cognitive-skill labels denote essentially the SAME operation."
    def dfn(t):
        _, a, b = t
        da = f" - {defn[a]}" if defn.get(a) else ""
        db = f" - {defn[b]}" if defn.get(b) else ""
        u = (f"Skill A: {a}{da}\nSkill B: {b}{db}\n\n"
             'Near-duplicates (essentially the same cognitive operation)? Return JSON {"duplicate": true|false}.')
        return (a, b, bool(judge(DSYS, u).get("duplicate", False)))
    dres = run(near, dfn)
    dup = sum(1 for _, _, d in dres if d)

    # ---------- report ----------
    def pct(x, n):
        return f"{100*x/max(n,1):.0f}%"
    print("=" * 60)
    print(f"PART A SCORECARD  ({len(names)} skills, full pipeline drop-rare)")
    print("=" * 60)
    print("\nStructural (automatic):")
    print(f"  skills .................. {len(names)}  (raw before merge {RAW_BEFORE}, "
          f"{RAW_BEFORE/len(names):.1f}x reduction)")
    print(f"  coverage ................ {pct(covered,n_items)} of items have >=1 skill")
    print(f"  catch-all check ......... largest skill covers {pct(largest,n_items)} of items")
    print(f"  reusability ............. median items/skill = {int(np.median(sizes))}, rare(<3)=0")
    print("\nSemantic (LLM judge; human anchor pending):")
    print(f"  tagging validity (n={N_VALIDITY})  yes={pct(vc['yes'],N_VALIDITY)}  "
          f"partial={pct(vc['partial'],N_VALIDITY)}  no={pct(vc['no'],N_VALIDITY)}")
    print(f"  granularity (n={len(names)})    appropriate={pct(gc['appropriate'],len(names))}  "
          f"too_broad={pct(gc['too_broad'],len(names))}  too_narrow={pct(gc['too_narrow'],len(names))}")
    print(f"  intruder/coherence (n={len(big)})  detection={intr_acc:.0%}  (chance=20%)")
    print(f"  distinctiveness (n={len(near)})    nearest pairs still duplicate = {pct(dup,len(near))}")

    broad = [s for s, g in gres if g == "too_broad"]
    print(f"\nFlagged too_broad skills ({len(broad)}): " + ", ".join(broad[:18]))
    print("Residual duplicate pairs (nearest):")
    for a, b, d in dres:
        if d:
            print(f"   {a}  ==  {b}")

    json.dump({"scorecard": {"skills": len(names), "raw_before": RAW_BEFORE,
               "reduction": RAW_BEFORE / len(names), "coverage": covered / n_items,
               "largest_frac": largest / n_items, "median_size": int(np.median(sizes)),
               "validity": dict(vc), "validity_n": N_VALIDITY, "granularity": dict(gc),
               "intruder_detection": float(intr_acc), "intruder_n": len(big),
               "near_pairs": len(near), "near_duplicate": dup},
               "per_skill_grain": {s: g for s, g in gres},
               "duplicate_pairs": [[a, b] for a, b, d in dres if d],
               "human_verify_sample": [{"item": i, "bench": item_meta[i][0],
                   "skill": sk, "question": q(qtext, i, 200), "llm_verdict": v}
                   for (i, sk), v in vres[:20]]},
              open(OUTF, "w"))
    print(f"\nwrote {OUTF} (incl. 20-pair human-verify sample)")


if __name__ == "__main__":
    main()
