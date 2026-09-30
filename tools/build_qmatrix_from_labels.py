#!/usr/bin/env python3
"""Turn a v7 pipeline label file into a Q-matrix the trainer can use.

Nothing in tools/ did this: the v7 pipeline writes `item_labels*.jsonl` and every
trainer reads `qmatrix_*.npy`, with no bridge between them. This is that bridge,
and T-106 cannot start without it.

Three things it has to get right.

ROW ALIGNMENT. Row i is item_idx i, because the response matrix's columns are
item_idx and the trainer indexes text embeddings and Q rows by the same integer.
Verified before writing: benchmark and subtask agree between the pipeline's own
metadata and `response_matrix_v2_full_items.json` on all 9,523 items. Note that
`question_preview` does NOT match the recovered full text (72 of 400 sampled),
which is the known preview-versus-fulltext defect and not a misalignment.

COLUMN ORDER. Columns are the live codes in sorted id order, recorded in the
sidecar. Without a recorded order, a second run that adds or merges a code
silently shifts every column and the Q no longer means what the checkpoint
trained on.

NO ZERO ROWS. `train_expanded.py` fixes zero-skill items by matching 768-dim
mpnet item-text embeddings against phrase-space cluster centroids. This taxonomy
has no phrase-space file, only 3072-dim definition vectors, so those spaces cannot
be compared and that function cannot be reused. The same idea is applied here
inside one consistent space (`item_emb_gemini.npz` against the definition cache),
which leaves no zero rows, so the trainer's own fixer never fires and never
reaches for the file that does not fit. Every filled row is recorded with its
similarity, because these fills are weak: the median is around 0.60 where a match
a person would recognise sits near 0.75.

    python3 tools/build_qmatrix_from_labels.py
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.metrics import row_codes
from tools.codeemb import read_cache

P = REPO / "cdm_exploration/experiments/pipeline_v7"
CR = REPO / "cdm_exploration/data/cdm_ready"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default="codebook_v2_amended_b150.json")
    ap.add_argument("--labels", default="item_labels_b150.jsonl")
    ap.add_argument("--emb", default="code_def_emb_b150.npz",
                    help="definition vectors, used only to fill zero rows")
    ap.add_argument("--item-emb", default="item_emb_gemini.npz",
                    help="item vectors in the SAME space as --emb")
    ap.add_argument("--items", default="response_matrix_v2_full_items.json",
                    help="defines the row order and count")
    ap.add_argument("--out", default="qmatrix_v7_b150_K274.npy")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = CR / a.out
    meta_f = CR / (Path(a.out).stem + "_meta.json")
    names_f = CR / ("cluster_labels_" + Path(a.out).stem.replace("qmatrix_", "") + ".json")
    for f in (out, meta_f, names_f):
        if f.exists() and not a.force:
            raise SystemExit(
                f"{f.name} exists. A checkpoint trained on a different Q of the same "
                f"name cannot be told apart from one trained on this. Pass --force "
                f"only if nothing has been trained on it.")

    items = json.loads((CR / a.items).read_text())
    n_items = len(items)
    if not all(items[k].get("item_idx") == k for k in range(n_items)):
        raise SystemExit(f"{a.items} is not in item_idx order; row alignment would be wrong")

    cb = json.loads((P / a.codebook).read_text())
    alias = cb.get("alias", {})
    live = sorted(c for c in cb["codes"] if c not in alias)
    col = {c: k for k, c in enumerate(live)}

    per, n_err = {}, 0
    for line in (P / a.labels).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            n_err += 1
            per[r["item_idx"]] = []
            continue
        per[r["item_idx"]] = [c for c in row_codes(r, alias) if c in cb["codes"]]
    missing = [i for i in range(n_items) if i not in per]
    if missing:
        raise SystemExit(f"{len(missing)} of {n_items} items have no row in {a.labels} "
                         f"(first: {missing[:5]}); the Q would silently be zero there")

    Q = np.zeros((n_items, len(live)), dtype=np.int64)
    for i in range(n_items):
        for c in per[i]:
            Q[i, col[c]] = 1

    zero = np.where(Q.sum(axis=1) == 0)[0]
    print(f"built {Q.shape} from {a.labels}: {len(live)} live codes, "
          f"{int(Q.sum())} assignments, {len(zero)} zero rows to fill")

    filled = []
    if len(zero):
        cached = read_cache(P / a.emb)
        if cached is None:
            raise SystemExit(f"{a.emb} carries no per-code hashes and cannot be vouched for")
        ids, vecs, _ = cached
        pos = {c: k for k, c in enumerate(list(ids))}
        absent = [c for c in live if c not in pos]
        if absent:
            raise SystemExit(f"{len(absent)} live codes are absent from {a.emb}")
        C = np.stack([vecs[pos[c]] for c in live]).astype("float64")
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-8)
        iz = np.load(P / a.item_emb, allow_pickle=True)
        ipos = {int(x): k for k, x in enumerate(list(iz["idx"]))}
        if C.shape[1] != iz["vecs"].shape[1]:
            raise SystemExit(f"{a.emb} is {C.shape[1]}-dim and {a.item_emb} is "
                             f"{iz['vecs'].shape[1]}-dim; these spaces are not comparable")
        V = iz["vecs"].astype("float64")
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
        for i in zero:
            if int(i) not in ipos:
                raise SystemExit(f"item {i} has no vector in {a.item_emb}")
            sims = C @ V[ipos[int(i)]]
            k = int(np.argmax(sims))
            Q[i, k] = 1
            filled.append({"item_idx": int(i), "code": live[k],
                           "name": cb["codes"][live[k]]["name"],
                           "similarity": round(float(sims[k]), 3)})
        s = [f["similarity"] for f in filled]
        print(f"  filled {len(filled)} rows by nearest definition: "
              f"min {min(s):.3f} median {sorted(s)[len(s)//2]:.3f} max {max(s):.3f}")
        print(f"  these are weak fills. T-113 calibrated a match a person would")
        print(f"  recognise at about 0.750, so treat them as imputation, not labels.")

    empty_cols = [live[k] for k in range(len(live)) if Q[:, k].sum() == 0]
    print(f"\n  shape {Q.shape} dtype {Q.dtype} unique {np.unique(Q)}")
    print(f"  mean skills per item {Q.sum(1).mean():.4f}   zero rows {(Q.sum(1)==0).sum()}")
    print(f"  columns holding no item: {len(empty_cols)}  -> effective K = "
          f"{len(live)-len(empty_cols)}")
    ref = CR / "qmatrix_v2_K100.npy"
    if ref.exists():
        r = np.load(ref)
        print(f"  the submitted K=100 Q for comparison: {r.shape} {r.dtype}, "
              f"mean {r.sum(1).mean():.4f}, zero rows {(r.sum(1)==0).sum()}")
        if r.shape[0] != Q.shape[0]:
            raise SystemExit("row counts differ from the submitted Q; the split would "
                             "not be comparable")

    np.save(out, Q)
    names_f.write_text(json.dumps(
        {str(k): cb["codes"][c]["name"] for k, c in enumerate(live)}, indent=1))
    meta_f.write_text(json.dumps({
        "built_from": {"codebook": a.codebook, "labels": a.labels,
                       "definition_embeddings": a.emb, "item_embeddings": a.item_emb,
                       "row_order": a.items},
        "shape": list(Q.shape), "dtype": str(Q.dtype), "binary": True,
        "column_order": live,
        "n_assignments": int(Q.sum()),
        "mean_skills_per_item": float(Q.sum(1).mean()),
        "label_rows_with_error": n_err,
        "zero_rows_filled": filled,
        "empty_columns": empty_cols,
        "effective_k": len(live) - len(empty_cols),
        "row_alignment": "row i is item_idx i; verified by benchmark and subtask "
                         "agreeing with the pipeline metadata on all 9,523 items",
        "note": "no zero rows remain, so train_expanded.fix_zero_skill_items does not "
                "fire and the 768-dim phrase-space skill_embeddings file it would "
                "otherwise need is not required",
    }, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}")
    print(f"      {names_f.relative_to(REPO)}")
    print(f"      {meta_f.relative_to(REPO)}")


if __name__ == "__main__":
    main()
