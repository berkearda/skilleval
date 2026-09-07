"""Diagnostic: would parent/domain context have improved merge nomination?

Recomputes the merge nomination on the pre-merge bank (as pilot_merge.py did),
re-runs the judge per pair, SAVES every pair decision, and classifies each pair
as same-domain vs cross-domain using the skill's dominant benchmark as the
domain proxy. Decision rule (written before running):
  - if most REJECTED pairs are cross-domain  -> parent prefix/gate would help
  - if most REJECTED pairs are same-domain   -> definitions already separate; skip
  - if any CONFIRMED merges are cross-domain -> a gate would risk true merges

    .venv312/bin/python tools/diag_merge_pairs.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BANKF = REPO / "cdm_exploration/experiments/pilot_skillbank_int.json"
STEPF = REPO / "cdm_exploration/experiments/stepwise_integrated.json"
OUTF = REPO / "cdm_exploration/experiments/merge_pair_decisions.json"
KEYFILE = Path.home() / ".cdmeval_openai_key"
THRESH, TOPN = 0.55, 8   # identical to pilot_merge.py

SYSTEM = """You decide whether two cognitive-skill labels denote the SAME cognitive operation. Judge by MEANING, not wording. They are the same only if a reasoning step requiring one would require the other. Skills from different domains, or different operations, are NOT the same even if worded similarly (e.g. order_of_operations vs order_adjectives are different). Return strict JSON."""

USER = """Skill A: {a}
  definition: {da}
  example steps: {ea}
Skill B: {b}
  definition: {db}
  example steps: {eb}

Same cognitive operation? Return JSON {{"same": true}} or {{"same": false}}."""


def load_key():
    t = KEYFILE.read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    new = json.load(open(BANKF))
    names = [b["name"] for b in new["bank"]]
    defn = {b["name"]: b["definition"] for b in new["bank"]}

    # example steps + benchmark counts per skill (same construction as pilot_merge)
    steps_of = {r["item_idx"]: [s.get("step", "") for s in r.get("reasoning_steps", [])]
                for r in json.load(open(STEPF))}
    sk_ex = defaultdict(list)
    sk_bench = defaultdict(Counter)
    for pi in new["per_item"]:
        sts = steps_of.get(pi["item_idx"], [])
        for j, sk in enumerate(pi["skills"]):
            sk_bench[sk][pi["benchmark"]] += 1
            if j < len(sts) and len(sk_ex[sk]) < 2 and sts[j]:
                sk_ex[sk].append(sts[j][:90])
    dom = {n: (sk_bench[n].most_common(1)[0][0] if sk_bench[n] else "?") for n in names}

    # nomination, exactly as pilot_merge.py
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    V = emb.encode([f"{n}: {defn[n]}" for n in names], normalize_embeddings=True)
    S = V @ V.T
    np.fill_diagonal(S, -1)
    pairs = set()
    for i in range(len(names)):
        for j in np.argsort(-S[i])[:TOPN]:
            if S[i, j] >= THRESH:
                pairs.add((min(i, int(j)), max(i, int(j))))
    pairs = sorted(pairs)
    print(f"bank={len(names)}  nominated pairs={len(pairs)}  (pilot_merge had 272)")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def decide(p):
        i, j = p
        a, b = names[i], names[j]
        u = USER.format(a=a, da=defn[a], ea=" | ".join(sk_ex[a]) or "(none)",
                        b=b, db=defn[b], eb=" | ".join(sk_ex[b]) or "(none)")
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": u}])
                return (a, b, bool(json.loads(r.choices[0].message.content).get("same")))
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return (a, b, False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(decide, pairs))

    # classify: 2x2 of (judge verdict) x (same-domain vs cross-domain)
    table = Counter()
    rows = []
    for a, b, same in res:
        cross = dom[a] != dom[b]
        table[("SAME" if same else "reject", "cross" if cross else "within")] += 1
        rows.append({"a": a, "b": b, "dom_a": dom[a], "dom_b": dom[b],
                     "judge_same": same, "cross_domain": cross})

    nrej = table[("reject", "within")] + table[("reject", "cross")]
    nsame = table[("SAME", "within")] + table[("SAME", "cross")]
    print(f"\njudge: {nsame} SAME / {nrej} rejected")
    print(f"\n{'':10s} {'within-domain':>14s} {'cross-domain':>13s}")
    print(f"{'rejected':10s} {table[('reject','within')]:14d} {table[('reject','cross')]:13d}")
    print(f"{'SAME':10s} {table[('SAME','within')]:14d} {table[('SAME','cross')]:13d}")

    print("\nconfirmed CROSS-domain merges (a gate would have endangered these):")
    for r in rows:
        if r["judge_same"] and r["cross_domain"]:
            print(f"   [{r['dom_a']} x {r['dom_b']}]  {r['a']}  ==  {r['b']}")
    print("\nsample of cross-domain junk nominations (a gate would have prevented):")
    shown = 0
    for r in rows:
        if not r["judge_same"] and r["cross_domain"]:
            print(f"   [{r['dom_a']} x {r['dom_b']}]  {r['a']}  vs  {r['b']}")
            shown += 1
            if shown >= 6:
                break

    json.dump({"pairs": rows, "table": {f"{k[0]}|{k[1]}": v for k, v in table.items()},
               "dominant_benchmark": dom}, open(OUTF, "w"))
    print(f"\nwrote {OUTF}")


if __name__ == "__main__":
    main()
