"""Compare three skill taxonomies on the pilot subset:
  NEW   = running bank (retrieve-then-decide)            [pilot_skillbank_new.json]
  OLD   = independent extraction + HAC to the same K     [skills_stepwise_full.json]
  FREE  = the benchmark's own subtask labels (trivial baseline)

Part A (read/structure, no model scores): #skills, duplication, reuse, coverage,
cross-domain spread, rare skills.
Part B (diagnostic signal from the 3811x9523 response matrix): per-skill split-half
reliability of skill-specific mastery (accuracy residualised against general ability).

    .venv312/bin/python tools/pilot_analyze.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
NEWF = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    REPO / "cdm_exploration/experiments/pilot_skillbank_new.json"
OLDF = D / "skills_stepwise_full.json"
MATF = D / "response_matrix_v2_full.npy"

MIN_ITEMS = 8      # a skill needs this many items to get a reliability estimate
N_SPLITS = 20
np.random.seed(42)


def reliability(skill_items: dict, M, a):
    """Median split-half reliability of residual mastery across skills."""
    np.random.seed(42)   # reset per method so splits do not depend on call order
    rels = []
    for sk, items in skill_items.items():
        cols = [i for i in items if i < M.shape[1]]
        if len(cols) < MIN_ITEMS:
            continue
        vals = []
        for _ in range(N_SPLITS):
            perm = np.random.permutation(cols)
            h = len(perm) // 2
            A, B = perm[:h], perm[h:]
            xa, xb = M[:, A].mean(1), M[:, B].mean(1)
            # residualise each against general ability a(m)
            ra = xa - np.polyval(np.polyfit(a, xa, 1), a)
            rb = xb - np.polyval(np.polyfit(a, xb, 1), a)
            if ra.std() > 1e-9 and rb.std() > 1e-9:
                vals.append(np.corrcoef(ra, rb)[0, 1])
        if vals:
            rels.append(np.mean(vals))
    return (len(rels), float(np.median(rels)) if rels else float("nan"))


def main():
    new = json.load(open(NEWF))
    subset_idx = [pi["item_idx"] for pi in new["per_item"]]
    sset = set(subset_idx)
    bench_of = {pi["item_idx"]: pi["benchmark"] for pi in new["per_item"]}

    # ---- NEW skill -> items ----
    new_si = defaultdict(list)
    for pi in new["per_item"]:
        for sk in set(pi["skills"]):
            new_si[sk].append(pi["item_idx"])
    new_density = np.mean([len(pi["skills"]) for pi in new["per_item"]])

    # ---- OLD (independent atomic skills on same items) ----
    old_rows = [r for r in json.load(open(OLDF)) if r["item_idx"] in sset]
    old_si_raw = defaultdict(list)
    for r in old_rows:
        for sk in set(r.get("atomic_skills", [])):
            old_si_raw[sk].append(r["item_idx"])
    old_density = np.mean([len(r.get("atomic_skills", [])) for r in old_rows])

    # ---- FREE (subtask = skill) ----
    free_si = defaultdict(list)
    for pi in new["per_item"]:
        free_si[f'{pi["benchmark"]}/{pi["subtask"]}'].append(pi["item_idx"])

    K_new = len(new["bank"])
    print("=" * 64)
    print(f"PILOT COMPARISON  ({len(subset_idx)} items, "
          + ", ".join(f"{b}={n}" for b, n in Counter(bench_of.values()).items()) + ")")
    print("=" * 64)
    print("\n--- Part A: structure (read-based) ---")
    print(f"{'method':16s} {'#skills':>8s} {'skills/item':>12s}")
    print(f"{'NEW (bank)':16s} {K_new:8d} {new_density:12.2f}")
    print(f"{'OLD (raw)':16s} {len(old_si_raw):8d} {old_density:12.2f}")
    print(f"{'FREE (subtask)':16s} {len(free_si):8d} {1.0:12.2f}")
    print(f"\nduplication: OLD produced {len(old_si_raw)} raw skill names for the same "
          f"items;\n  NEW running bank = {K_new} "
          f"({len(old_si_raw)/max(K_new,1):.1f}x fewer, built without a clustering step).")
    c = new["counts"]
    tot = c["reuse"] + c["new"] + c["leaked"]
    print(f"reuse rate (NEW) = {c['reuse']/max(tot,1):.0%}   new={c['new']}  leaked={c['leaked']}")
    g = new["growth_curve"]
    if len(g) > 100:
        print(f"bank growth: after 100 items={g[99]}, final={K_new}, "
              f"added in last 100={K_new-g[-100]}  (converging if small)")
    xdom = sum(1 for sk, its in new_si.items()
               if len({bench_of[i] for i in its}) > 1)
    rare = sum(1 for sk, its in new_si.items() if len(its) < 3)
    print(f"NEW cross-domain skills (used in >1 benchmark) = {xdom}/{K_new}")
    print(f"NEW rare skills (<3 items) = {rare}/{K_new}")

    # ---- OLD clustered to K_new for a fair Part B comparison ----
    print("\n--- Part B: diagnostic signal (response matrix) ---")
    M = np.load(MATF)
    a = M.mean(1)   # general ability per model, over all items
    print(f"(matrix {M.shape}, general-ability range {a.min():.2f}-{a.max():.2f}, "
          f"min_items={MIN_ITEMS}, {N_SPLITS} splits)")

    old_clustered = None
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
        names = list(old_si_raw)
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        V = emb.encode(names, normalize_embeddings=True)
        k = min(K_new, len(names))
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                      linkage="average").fit_predict(V)
        cl_items = defaultdict(set)
        for nm, cid in zip(names, lab):
            for i in old_si_raw[nm]:
                cl_items[cid].add(i)
        old_clustered = {c: list(v) for c, v in cl_items.items()}
    except Exception as e:  # noqa: BLE001
        print(f"(old-clustered skipped: {str(e)[:80]})")

    print(f"\n{'method':18s} {'#skills eval':>12s} {'median reliability':>20s}")
    for name, si in [("NEW (bank)", new_si),
                     ("OLD (clustered)", old_clustered or {}),
                     ("FREE (subtask)", free_si)]:
        n_eval, med = reliability(si, M, a)
        print(f"{name:18s} {n_eval:12d} {med:20.3f}")
    print("\nHigher reliability = the skill separates models stably beyond general "
          "ability.\nNEW should match or beat OLD-clustered at equal granularity, and "
          "all should beat\nFREE if fine skills carry signal the coarse labels miss.")


if __name__ == "__main__":
    main()
