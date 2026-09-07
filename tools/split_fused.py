"""Experiment B: split the fused clusters, then name every group specifically.

Coherent clusters are kept whole; clusters the blind review flagged two_groups /
multi_group_junk are split into k sub-clusters (k = reviewer subgroup estimate) by
HAC on the member items' question embeddings. Every resulting group is renamed with
the specificity-constrained prompt. Output is a new pilot-format taxonomy.

    .venv312/bin/python tools/split_fused.py \
        cdm_exploration/experiments/oldtax_split_specific.json
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
REV = Path("/private/tmp/claude-501/-Users-berkearda-Desktop-cdmeval/"
           "1de56163-6f9a-40d8-9c62-dc65a71bce0c/scratchpad/reviews")
LABF = REPO / "cdm_exploration/experiments/oldtax_full_format.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "cdm_exploration/experiments/oldtax_split_specific.json"
EVID = 8
random.seed(42)

SPECIFIC = ("Name the specific cognitive OPERATION this group requires, as a short English "
            "verb phrase (for example 'solve algebraic equations', 'track object positions after "
            "swaps', 'follow explicit output-format constraints'). Do NOT name a subject area. "
            "Do NOT use vague words like 'problem solving', 'analysis', 'skills', 'reasoning', "
            "'advanced'. If the items do not share one operation, name the single most common one.")


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(LABF))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    sk_items = defaultdict(list)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])

    review = {}
    for g in "ABCD":
        for r in json.load(open(REV / f"group{g}.json")):
            review[r["original_name"]] = r
    fused = {"two_groups", "multi_group_junk"}

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    emb = SentenceTransformer("all-MiniLM-L6-v2")

    # build the new groups: {group_id: [item_idx]}
    groups = {}
    n_split = 0
    for sk, items in sk_items.items():
        r = review.get(sk, {})
        k = int(r.get("subgroups", 1))
        if r.get("cluster_verdict") in fused and k >= 2 and len(items) >= k:
            V = emb.encode([qtext.get(i, "")[:400] for i in items], normalize_embeddings=True)
            lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                          linkage="average").fit_predict(V)
            for c in range(k):
                sub = [items[j] for j in range(len(items)) if lab[j] == c]
                if sub:
                    groups[f"{sk}##sub{c}"] = sub
            n_split += 1
        else:
            groups[sk] = items
    print(f"clusters: 100 -> groups: {len(groups)}  (split {n_split} fused clusters)")

    client_key = load_key()
    from openai import OpenAI
    client = OpenAI(api_key=client_key)

    def name_group(gid):
        items = groups[gid]
        ev = "\n".join(f"- {qtext.get(i,'')[:300]}" for i in random.sample(items, min(EVID, len(items))))
        u = (f"These test items form one group in a skill taxonomy:\n{ev}\n\n"
             f'{SPECIFIC} Return JSON {{"name": "<concise name>"}}')
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content":
                               "You name skill groups from their member items. You are not shown any existing name."},
                              {"role": "user", "content": u}])
                return gid, str(json.loads(r.choices[0].message.content).get("name", "")).strip()
            except Exception:  # noqa: BLE001
                if a == 2:
                    return gid, ""

    gids = list(groups)
    with ThreadPoolExecutor(max_workers=8) as ex:
        raw = dict(ex.map(name_group, gids))

    used, mapping = set(), {}
    for gid in gids:
        nm = raw[gid] or gid
        base, k = nm, 2
        while nm in used:
            nm = f"{base} ({k})"; k += 1
        used.add(nm)
        mapping[gid] = nm

    item2groups = defaultdict(list)
    for gid, items in groups.items():
        for i in items:
            item2groups[i].append(mapping[gid])
    meta = {pi["item_idx"]: (pi["benchmark"], pi["subtask"]) for pi in data["per_item"]}
    per_item = [{"item_idx": i, "benchmark": meta[i][0], "subtask": meta[i][1],
                 "skills": sorted(set(item2groups[i]))}
                for i in sorted(item2groups)]
    bank = [{"name": mapping[gid], "definition": ""} for gid in gids]

    json.dump({"bank": bank, "per_item": per_item,
               "group_map": {mapping[gid]: gid for gid in gids}}, open(OUT, "w"))
    print(f"final skills: {len(bank)}   items covered: {len(per_item)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
