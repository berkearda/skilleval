"""Compare step-wise (atomic) vs full-text (holistic) extraction on the pilot.

Same items, same embedder + HAC. Reports, simply:
  - atomicity: unique skill phrases and repetition (fewer-unique-per-mention
    = more canonical = clusters better)
  - separation: does clustering recover the benchmark subtasks (ARI)
  - the targeted bad-merge check: do sports_understanding and
    tracking_shuffled_objects land in the SAME cluster (bad) or different?
  - readable cards for the step-wise clusters.

Run after the step-wise pilot finishes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

DATA = Path("cdm_exploration/data/cdm_ready")
STEP = DATA / "skills_stepwise_pilot.json"
FULL = DATA / "skills_extracted_v2_fulltext.json"


def main() -> int:
    from sentence_transformers import SentenceTransformer
    step = {r["item_idx"]: r for r in json.load(open(STEP))}
    full = {r["item_idx"]: r for r in json.load(open(FULL))}
    ids = [i for i in step if i in full]
    subt = {r["item_idx"]: r["subtask"] for r in json.load(open(STEP))}
    embedder = SentenceTransformer("all-mpnet-base-v2")
    n_sub = len({subt[i] for i in ids})
    print(f"pilot items: {len(ids)} across {n_sub} subtasks\n")

    def skills_step(i):
        return [s.lower() for s in (step[i].get("atomic_skills") or [])]

    def skills_full(i):
        return [s.lower() for s in (full[i].get("skills") or [])]

    def analyze(getter, label, K):
        per_item = {i: getter(i) for i in ids}
        uniq = sorted({s for ss in per_item.values() for s in ss})
        ment = sum(len(ss) for ss in per_item.values())
        emb = embedder.encode(uniq, normalize_embeddings=True, show_progress_bar=False,
                              batch_size=256)
        lab = AgglomerativeClustering(n_clusters=K, metric="cosine",
                                      linkage="average").fit_predict(emb)
        p2c = {uniq[j]: int(lab[j]) for j in range(len(uniq))}
        dom, subs = [], []
        item_clusters = {}
        for i in ids:
            cs = [p2c[s] for s in per_item[i] if s in p2c]
            if not cs:
                continue
            item_clusters[i] = set(cs)
            dom.append(Counter(cs).most_common(1)[0][0]); subs.append(subt[i])
        ari = adjusted_rand_score(subs, dom)
        print(f"[{label}] unique_skills={len(uniq)} mentions={ment} "
              f"(uniq/mention={len(uniq)/max(ment,1):.2f})  K={K}  subtask-ARI={ari:.3f}")
        return p2c, per_item, item_clusters, uniq, lab, dom

    K = n_sub
    print("Clustering both into K = number of subtasks:")
    analyze(skills_full, "FULL-TEXT (holistic)", K)
    p2c, per_item, item_clusters, uniq, lab, dom = analyze(skills_step, "STEP-WISE (atomic)", K)

    # targeted bad-merge check: sports vs tracking
    def dominant_cluster_of_subtask(sub):
        cs = []
        for i in ids:
            if subt[i] == sub and i in item_clusters:
                cs += list(item_clusters[i])
        return Counter(cs).most_common(3)
    print("\nBad-merge check (step-wise) — dominant clusters per subtask:")
    for sub in ("sports_understanding", "tracking_shuffled_objects_seven_objects"):
        print(f"  {sub}: {dominant_cluster_of_subtask(sub)}")

    # readable cards for step-wise clusters
    cnt = Counter(s for ss in per_item.values() for s in ss)
    print("\nSTEP-WISE cluster cards:")
    for c in range(K):
        members = [i for i in ids if i in item_clusters and c in item_clusters[i]]
        if not members:
            continue
        subs = Counter(subt[i] for i in members)
        phr = Counter(s for i in members for s in per_item[i] if p2c.get(s) == c)
        name = max((uniq[j] for j in range(len(uniq)) if lab[j] == c),
                   key=lambda s: cnt.get(s, 0), default="?")
        print(f"\n  cluster {c} (n_items={len(members)}) ~ {name}")
        print(f"    subtasks: " + ", ".join(f"{s}:{v}" for s, v in subs.most_common(4)))
        print(f"    top atomic skills: " + ", ".join(f"{s}({v})" for s, v in phr.most_common(5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
