"""Compare prompt A (current) vs B (surgically tightened) on the pilot.

Both extractions are on the SAME 2,000 items. Cluster each (HAC K=30) and
report the two things we care about:
  - do the GENERIC BLOBS shrink in B?  (largest cluster size; # clusters
    spanning >=4 subtasks; presence of procedural skills)
  - do we AVOID over-fragmentation?    (# tiny clusters <5 items; singletons)
plus subtask-ARI and atomicity.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

DATA = Path("cdm_exploration/data/cdm_ready")
A = DATA / "skills_stepwise_pilot.json"
B = DATA / "skills_stepwise_pilot_B.json"
K = 30
PROCEDURAL = {"make_a_choice_based_on_criteria", "select_correct_option", "make_a_choice",
              "evaluate_statement", "determine_truth_value", "apply_deductive_reasoning",
              "identify_key_components", "extract_key_information"}


def analyze(path, embedder, label):
    recs = json.load(open(path))
    subt = {r["item_idx"]: r["subtask"] for r in recs}
    per_item = {r["item_idx"]: [s.lower() for s in (r.get("atomic_skills") or [])] for r in recs}
    uniq = sorted({s for ss in per_item.values() for s in ss})
    ment = sum(len(ss) for ss in per_item.values())
    emb = embedder.encode(uniq, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
    lab = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit_predict(emb)
    p2c = {uniq[j]: int(lab[j]) for j in range(len(uniq))}

    item_cl, dom, subs = {}, [], []
    for i, ss in per_item.items():
        cs = {p2c[s] for s in ss if s in p2c}
        if cs:
            item_cl[i] = cs
            dom.append(Counter(p2c[s] for s in ss if s in p2c).most_common(1)[0][0])
            subs.append(subt[i])
    csize = Counter(c for cs in item_cl.values() for c in cs)
    cl_subtasks = {c: {subt[i] for i in item_cl if c in item_cl[i]} for c in csize}

    procedural_present = sum(1 for s in uniq for p in PROCEDURAL if p in s)
    ari = adjusted_rand_score(subs, dom)
    largest = csize.most_common(1)[0][1]
    big_generic = sum(1 for c in csize if len(cl_subtasks[c]) >= 4)
    tiny = sum(1 for c, n in csize.items() if n < 5)

    print(f"\n[{label}]")
    print(f"  unique skills {len(uniq)} | mentions {ment} | uniq/mention {len(uniq)/max(ment,1):.2f}")
    print(f"  subtask-ARI {ari:.3f}")
    print(f"  LARGEST cluster (blob) = {largest} items")
    print(f"  clusters spanning >=4 subtasks (generic) = {big_generic}")
    print(f"  tiny clusters (<5 items, over-fragment) = {tiny}")
    print(f"  procedural/generic skill names still present = {procedural_present}")
    names = []
    for c, n in csize.most_common(5):
        nm = max((uniq[j] for j in range(len(uniq)) if lab[j] == c),
                 key=lambda s: sum(1 for ss in per_item.values() for x in ss if x == s), default="?")
        names.append(f"{nm}({n},{len(cl_subtasks[c])}sub)")
    print("  5 largest: " + ", ".join(names))


def main() -> int:
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-mpnet-base-v2")
    print("Comparing A (current prompt) vs B (tightened prompt), HAC K=30, same 2000 items:")
    analyze(A, emb, "A current")
    analyze(B, emb, "B tightened")
    print("\nVerdict cues:")
    print("  B better if: LARGEST + generic-spanning + procedural all DOWN,")
    print("  WITHOUT tiny clusters going UP (that would mean over-fragmentation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
