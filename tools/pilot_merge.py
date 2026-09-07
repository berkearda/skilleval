"""Step 3 of the pipeline: merge near-duplicate skills BY MEANING, then fold the
rare tail, on the pilot bank. Embeddings only NOMINATE candidate pairs; an LLM
decides same-or-not by reading definitions + example steps. Meaning wins.

    # dry: just count candidate pairs (no LLM, cost gate)
    .venv312/bin/python tools/pilot_merge.py --dry
    # run: decide merges, fold rare skills, write merged bank
    .venv312/bin/python tools/pilot_merge.py \
        --out cdm_exploration/experiments/pilot_skillbank_merged.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
NEWF = REPO / "cdm_exploration/experiments/pilot_skillbank_new.json"
STEPF = D / "skills_stepwise_full.json"
KEYFILE = Path.home() / ".cdmeval_openai_key"

THRESH = 0.55     # cosine floor to NOMINATE a candidate pair (recall only)
TOPN = 8          # max neighbours considered per skill
RARE = 3          # skills with fewer items than this get folded into nearest

SYSTEM = """You decide whether two cognitive-skill labels denote the SAME cognitive operation. Judge by MEANING, not wording. They are the same only if a reasoning step requiring one would require the other. Skills from different domains, or different operations, are NOT the same even if worded similarly (e.g. order_of_operations vs order_adjectives are different). Return strict JSON."""

USER = """Skill A: {a}
  definition: {da}
  example steps: {ea}
Skill B: {b}
  definition: {db}
  example steps: {eb}

Same cognitive operation? Return JSON {{"same": true}} or {{"same": false}}."""


def load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    t = KEYFILE.read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rare", choices=["fold", "drop", "keep"], default="fold",
                    help="how to handle skills with <RARE items after the LLM merge")
    ap.add_argument("--new", default=str(NEWF))
    ap.add_argument("--steps", default=str(STEPF))
    ap.add_argument("--out")
    args = ap.parse_args()

    new = json.load(open(args.new))
    names = [b["name"] for b in new["bank"]]
    defn = {b["name"]: b["definition"] for b in new["bank"]}

    # item counts + example step texts per skill (zip per-item skills with step texts)
    steps_of = {r["item_idx"]: [s.get("step", "") for s in r.get("reasoning_steps", [])]
                for r in json.load(open(args.steps))}
    sk_items = defaultdict(set)
    sk_ex = defaultdict(list)
    for pi in new["per_item"]:
        sts = steps_of.get(pi["item_idx"], [])
        for j, sk in enumerate(pi["skills"]):
            sk_items[sk].add(pi["item_idx"])
            if j < len(sts) and len(sk_ex[sk]) < 2 and sts[j]:
                sk_ex[sk].append(sts[j][:90])
    count = {n: len(sk_items[n]) for n in names}

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.embedder)
    V = emb.encode([f"{n}: {defn[n]}" for n in names], normalize_embeddings=True)
    S = V @ V.T
    np.fill_diagonal(S, -1)

    # nominate candidate pairs (recall only)
    pairs = set()
    for i in range(len(names)):
        nbr = np.argsort(-S[i])[:TOPN]
        for j in nbr:
            if S[i, j] >= THRESH:
                pairs.add((min(i, j), max(i, j)))
    pairs = sorted(pairs)
    print(f"bank={len(names)} skills | candidate pairs (cos>={THRESH}, top{TOPN}) = {len(pairs)}")
    print(f"rare skills (<{RARE} items) = {sum(1 for n in names if count[n] < RARE)}")
    if args.dry:
        print("dry run: no LLM calls made.")
        return

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def decide(pair):
        i, j = pair
        a, b = names[i], names[j]
        u = USER.format(a=a, da=defn[a], ea=" | ".join(sk_ex[a]) or "(none)",
                        b=b, db=defn[b], eb=" | ".join(sk_ex[b]) or "(none)")
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model, temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": u}])
                return pair, bool(json.loads(r.choices[0].message.content).get("same"))
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return pair, False
        return pair, False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(decide, pairs))
    same = [(i, j) for (i, j), yes in results if yes]
    print(f"LLM merge decisions: {sum(1 for _,y in results if y)}/{len(results)} pairs are SAME")

    # union-find on the SAME pairs
    parent = list(range(len(names)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in same:
        parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i in range(len(names)):
        groups[find(i)].append(i)
    # canonical = highest item-count member of the group
    canon = {}
    for g, members in groups.items():
        best = max(members, key=lambda k: count[names[k]])
        for k in members:
            canon[names[k]] = names[best]
    merged_by_llm = len(names) - len(set(canon.values()))

    # item counts per canonical (after the LLM meaning-merge)
    canon_items = defaultdict(set)
    for n in names:
        canon_items[canon[n]] |= sk_items[n]
    rare_canon = {c for c in set(canon.values()) if len(canon_items[c]) < RARE}
    folded = dropped = 0
    drop_set: set = set()
    if args.rare == "fold":
        big = [c for c in set(canon.values()) if c not in rare_canon]
        big_idx = [names.index(c) for c in big]
        BV = V[big_idx]
        for c in list(rare_canon):
            nearest = big[int(np.argmax(V[names.index(c)] @ BV.T))]
            for n in names:
                if canon[n] == c:
                    canon[n] = nearest
            folded += 1
    elif args.rare == "drop":
        drop_set = rare_canon
        dropped = len(rare_canon)

    final = sorted(set(canon.values()) - drop_set)
    per_item = []
    for pi in new["per_item"]:
        sk = []
        for s in pi["skills"]:
            c = canon.get(s, s)
            if c in drop_set or c in sk:
                continue
            sk.append(c)
        per_item.append({**{k: pi[k] for k in ("item_idx", "benchmark", "subtask")},
                         "skills": sk})

    print(f"\nbank: {len(names)} -> {len(final)} skills "
          f"(LLM-merged groups removed {merged_by_llm}, rare-{args.rare}: "
          f"folded {folded}, dropped {dropped})")

    if args.out:
        out_bank = [{"name": c, "definition": defn.get(c, ""), "rank": k}
                    for k, c in enumerate(final)]
        json.dump({"subset": new["subset"], "model": new["model"],
                   "embedder": args.embedder, "n_items": new["n_items"],
                   "bank": out_bank, "growth_curve": new["growth_curve"],
                   "per_item": per_item,
                   "counts": {**new["counts"], "merged_by_llm": merged_by_llm,
                              "rare_folded": folded, "final_skills": len(final)}},
                  open(args.out, "w"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
