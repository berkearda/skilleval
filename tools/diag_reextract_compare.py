"""Compare clustering on OLD (truncated-text) vs NEW (full-text) skill labels.

Focus: the six BBH subtasks that were jammed into the single broken cluster
#97. Question: with skills re-extracted from REAL questions, does clustering
now separate them back into coherent, subtask-aligned skills?

For OLD and NEW separately: take each item's primary skill phrase, embed with
the same SBERT (all-mpnet-base-v2), cluster into K groups (HAC, cosine,
average -- the pipeline's method), and score against the benchmark's own
subtask labels (the ground-truth answer key) via ARI + cluster purity.

Read-only. Run after reextract_skills_fulltext.py finishes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

DATA = Path("cdm_exploration/data/cdm_ready")
OLD = DATA / "skills_extracted_v2_full.json"
NEW = DATA / "skills_extracted_v2_fulltext.json"
# the six subtasks bundled in broken cluster #97
MIXED_KEYS = ["bbh_disambiguation_qa", "bbh_hyperbaton", "bbh_web_of_lies",
              "bbh_salient_translation_error_detection", "bbh_ruin_names",
              "bbh_formal_fallacies"]


def primary(rec: dict) -> str:
    p = rec.get("primary_skill") or ""
    if not p:
        sk = rec.get("skills") or []
        p = sk[0] if sk else ""
    return p.lower().strip()


def cluster_and_score(recs, embedder, k, label):
    phrases = [primary(r) for r in recs]
    subtasks = [r["subtask"] for r in recs]
    keep = [i for i, p in enumerate(phrases) if p]
    phrases = [phrases[i] for i in keep]
    subtasks = [subtasks[i] for i in keep]
    emb = embedder.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
    cl = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                 linkage="average").fit_predict(emb)
    ari = adjusted_rand_score(subtasks, cl)
    # purity: for each cluster, fraction from its dominant subtask
    purities, sizes = [], []
    for c in sorted(set(cl)):
        members = [subtasks[i] for i in range(len(cl)) if cl[i] == c]
        top = Counter(members).most_common(1)[0][1]
        purities.append(top / len(members))
        sizes.append(len(members))
    wpur = float(np.average(purities, weights=sizes))
    n_unique = len(set(phrases))
    print(f"  [{label}] items={len(phrases)} unique_phrases={n_unique} "
          f"K={k}  ARI={ari:.3f}  weighted_purity={wpur:.1%}")
    return cl, subtasks, phrases, ari, wpur


def show_clusters(cl, subtasks, phrases, k, label):
    print(f"\n  --- {label}: what each cluster contains ---")
    for c in sorted(set(cl)):
        idx = [i for i in range(len(cl)) if cl[i] == c]
        sub = Counter(subtasks[i] for i in idx).most_common(2)
        ph = Counter(phrases[i] for i in idx).most_common(1)[0][0]
        subs = ", ".join(f"{s}:{n}" for s, n in sub)
        print(f"    cluster {c} (n={len(idx)}): subtasks[{subs}] | label~ {ph[:46]!r}")


def main() -> int:
    from sentence_transformers import SentenceTransformer
    old_all = {r["item_idx"]: r for r in json.load(open(OLD))}
    new_all = {r["item_idx"]: r for r in json.load(open(NEW))}

    # items present in BOTH and in the six mixed subtasks
    new_mixed = [r for r in new_all.values() if r["key"] in MIXED_KEYS]
    ids = [r["item_idx"] for r in new_mixed if r["item_idx"] in old_all]
    old_recs = [old_all[i] for i in ids]
    new_recs = [new_all[i] for i in ids]
    print(f"comparing {len(ids)} items across {len(MIXED_KEYS)} subtasks "
          f"(ground-truth: each item's BBH subtask)\n")

    embedder = SentenceTransformer("all-mpnet-base-v2")
    K = len(MIXED_KEYS)  # 6 true skills
    print("clustering both into K=6 (the number of true subtasks):")
    co, so, po, ario, _ = cluster_and_score(old_recs, embedder, K, "OLD (truncated text)")
    cn, sn, pn, arin, _ = cluster_and_score(new_recs, embedder, K, "NEW (full text)")

    show_clusters(co, so, po, K, "OLD")
    show_clusters(cn, sn, pn, K, "NEW")

    print(f"\n=== VERDICT ===")
    print(f"ARI (1.0 = perfectly recovers the 6 subtasks): OLD {ario:.3f} -> NEW {arin:.3f}")
    print("Higher NEW ARI = re-extraction on full text lets clustering separate the")
    print("six skills that were collapsed into one cluster under truncated text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
