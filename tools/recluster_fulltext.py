"""Global re-cluster on full-text skills; compare taxonomy quality vs old.

Rebuilds the K=100 taxonomy the same way the pipeline does (embed unique
skill phrases with all-mpnet-base-v2, HAC cosine/average), for BOTH the old
(truncated-text) and new (full-text) extractions, and reports taxonomy-health
metrics side by side:
  - singleton / tiny (<5 item) skill counts
  - mean cluster size, items-per-skill spread
  - subtask homogeneity (ARI of the Q-matrix's per-item dominant skill vs the
    benchmark's own subtask labels -- higher = skills track real structure)
  - cross-benchmark skills (skills spanning >1 benchmark, a CDMEval goal)

Writes the new K=100 Q-matrix + labels to NEW files (nothing overwritten).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

DATA = Path("cdm_exploration/data/cdm_ready")
K = 100


def load(path):
    recs = json.load(open(path))
    return {r["item_idx"]: r for r in recs}


def skills_of(rec):
    return [s.lower().strip() for s in (rec.get("skills") or []) if s and s.strip()]


def build_taxonomy(recs_by_idx, items, embedder, label):
    # unique phrases across all items
    per_item = {i: skills_of(recs_by_idx[i]) for i in recs_by_idx}
    uniq = sorted({s for ss in per_item.values() for s in ss})
    emb = embedder.encode(uniq, normalize_embeddings=True, show_progress_bar=False,
                          batch_size=256)
    lab = AgglomerativeClustering(n_clusters=K, metric="cosine",
                                  linkage="average").fit_predict(emb)
    phrase2cluster = {uniq[i]: int(lab[i]) for i in range(len(uniq))}

    # per-item dominant skill cluster (first skill's cluster), Q rows
    item_ids = [it["item_idx"] for it in items if it["item_idx"] in per_item]
    dom, subtasks, benches = [], [], []
    cluster_items = {c: [] for c in range(K)}
    cluster_bench = {c: set() for c in range(K)}
    for it in items:
        i = it["item_idx"]
        if i not in per_item:
            continue
        sk = per_item[i]
        cs = [phrase2cluster[s] for s in sk if s in phrase2cluster]
        if not cs:
            continue
        d = Counter(cs).most_common(1)[0][0]
        dom.append(d); subtasks.append(it["subtask"]); benches.append(it["benchmark"])
        for c in set(cs):
            cluster_items[c].append(i)
            cluster_bench[c].add(it["benchmark"])

    sizes = np.array([len(cluster_items[c]) for c in range(K)])
    singles = int((sizes == 1).sum()); tiny = int((sizes < 5).sum())
    empty = int((sizes == 0).sum())
    ari = adjusted_rand_score(subtasks, dom)
    xbench = sum(1 for c in range(K) if len(cluster_bench[c]) > 1)
    print(f"  [{label}] uniq_phrases={len(uniq):5d}  empty={empty} singletons={singles} "
          f"tiny<5={tiny}  median_size={int(np.median(sizes))}  max_size={int(sizes.max())}")
    print(f"           subtask-ARI={ari:.3f}  cross-benchmark_skills={xbench}/{K}")
    return phrase2cluster, per_item, cluster_items, uniq, lab


def main() -> int:
    from sentence_transformers import SentenceTransformer
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    old = load(DATA / "skills_extracted_v2_full.json")
    new = load(DATA / "skills_extracted_v2_fulltext.json")
    embedder = SentenceTransformer("all-mpnet-base-v2")

    print("Rebuilding K=100 taxonomy for OLD vs NEW (same embedder + HAC):\n")
    build_taxonomy(old, items, embedder, "OLD (truncated)")
    p2c, per_item, cl_items, uniq, lab = build_taxonomy(new, items, embedder, "NEW (full text)")

    # persist the NEW taxonomy (new files only)
    # build binary Q-matrix (items x K) from new clustering
    idx_order = [it["item_idx"] for it in items]
    Q = np.zeros((len(items), K), dtype=int)
    pos = {iid: p for p, iid in enumerate(idx_order)}
    for iid, sk in per_item.items():
        for s in sk:
            c = p2c.get(s)
            if c is not None:
                Q[pos[iid], c] = 1
    np.save(DATA / "qmatrix_v2_K100_fulltext.npy", Q)
    # cluster label = most frequent member phrase
    cnt = Counter(s for ss in per_item.values() for s in ss)
    labels = {}
    for c in range(K):
        members = [uniq[i] for i in range(len(uniq)) if lab[i] == c]
        labels[str(c)] = (max(members, key=lambda s: cnt.get(s, 0)).replace("_", " ").title()
                          if members else f"cluster_{c}")
    json.dump(labels, open(DATA / "cluster_labels_v2_K100_fulltext.json", "w"), indent=2)
    print(f"\nwrote qmatrix_v2_K100_fulltext.npy {Q.shape} + cluster_labels_v2_K100_fulltext.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
