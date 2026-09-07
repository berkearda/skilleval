"""Analyze the reference-answer ablation.

Arm A: original question-only labels (skills_extracted_v2_full.json, the run
       behind the submitted Q-matrix).
Arm B: same prompt + reference solution (v2_reference_answer_ablation.json).

Reports:
 1. Label agreement per item: exact-match Jaccard, and token-level Jaccard
    (snake_case words) which tolerates paraphrase.
 2. The hypothesis: do disagreements concentrate on items whose
    STEM does not reveal the method? Proxy for "stem reveals method" =
    overlap between the arm-A primary skill's tokens and the question text.
    We split items by that proxy and compare agreement in each half.
 3. Downstream bound: assign arm-B skill phrases to the frozen K=100 clusters
    by SBERT nearest-centroid (the deployment path) and count how many items
    would change cluster membership.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"

STOP = {"of", "in", "to", "from", "with", "for", "and", "the", "a", "on", "by", "at"}


def toks(label: str) -> set[str]:
    return {w for w in re.split(r"[_\W]+", (label or "").lower()) if w and w not in STOP}


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main() -> None:
    armB = {r["item_idx"]: r for r in json.loads((E / "v2_reference_answer_ablation.json").read_text())["results"]}
    # Arm A MUST be the run behind the submitted Q-matrix and MUST use the same
    # prompt as arm B, else the ablation is confounded (prompt + info both change).
    # Verified: skills_v2_new_prompt.json uses QMATRIX_SYSTEM_PROMPT (snake_case)
    # and its phrases map onto cluster_labels_v2_K100 (e.g. item 1 ->
    # "Calculating Compound Interest With Quarterly Compo").
    armA = {r["item_idx"]: r for r in json.loads((D / "skills_v2_new_prompt.json").read_text())}
    qt = {r["item_idx"]: r["question_full_text"]
          for r in json.loads((D / "item_full_text_recovered.json").read_text())}

    shared = sorted(set(armA) & set(armB))
    print(f"items in both arms: {len(shared)}", flush=True)

    def a_labels(i):
        s = armA[i].get("skills", [])
        return [x.get("label") if isinstance(x, dict) else str(x) for x in s]

    rows = []
    for i in shared:
        A = [l for l in a_labels(i) if l]
        B = [l for l in armB[i]["skills_with_ref"] if l]
        exact = jac(set(A), set(B))
        tokA = set().union(*[toks(l) for l in A]) if A else set()
        tokB = set().union(*[toks(l) for l in B]) if B else set()
        tok_j = jac(tokA, tokB)
        # does the STEM reveal the method? proxy: arm-A primary tokens present in question
        prim = armA[i].get("primary_skill") or (A[0] if A else "")
        qtoks = set(re.split(r"[^a-z0-9]+", qt.get(i, "").lower()))
        cover = len(toks(prim) & qtoks) / max(1, len(toks(prim)))
        rows.append({"i": i, "bench": armB[i]["benchmark"], "exact": exact,
                     "tok": tok_j, "stem_reveals": cover, "nA": len(A), "nB": len(B)})

    def summ(sel, name):
        if not sel:
            return
        ex = np.array([r["exact"] for r in sel]); tk = np.array([r["tok"] for r in sel])
        print(f"  {name:24s} n={len(sel):5d}  exact-label Jaccard {ex.mean():.3f}   "
              f"token Jaccard {tk.mean():.3f}   identical-set {np.mean(ex==1):.1%}", flush=True)

    print("\n=== 1. Agreement between question-only and solution-aware labels ===", flush=True)
    summ(rows, "ALL")
    for b in ["MATH", "GPQA"]:
        summ([r for r in rows if r["bench"] == b], b)

    print("\n=== 2. Hypothesis: is disagreement concentrated where the stem hides the method? ===", flush=True)
    cov = np.array([r["stem_reveals"] for r in rows])
    med = float(np.median(cov))
    hi = [r for r in rows if r["stem_reveals"] > med]
    lo = [r for r in rows if r["stem_reveals"] <= med]
    print(f"  median stem-coverage of the primary skill = {med:.2f}", flush=True)
    summ(hi, "stem REVEALS method")
    summ(lo, "stem HIDES method")

    print("\n=== 3. Downstream bound: cluster reassignment under arm B ===", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        q = np.load(D / "qmatrix_v2_K100.npy")
        emb_all = np.load(D / "skill_phrase_embeddings_v2.npy") if (D / "skill_phrase_embeddings_v2.npy").exists() else None
        model = SentenceTransformer("all-mpnet-base-v2")
        lab = json.loads((D / "cluster_labels_v2_K100.json").read_text())
        # centroid per cluster from its ORIGINAL member skill phrases
        centro = np.zeros((100, 768), dtype=np.float32)
        names = [lab[str(k)] for k in range(100)]
        centro = model.encode(names, normalize_embeddings=True, show_progress_bar=False)
        changed = 0
        for r in rows:
            i = r["i"]
            B = [l.replace("_", " ") for l in armB[i]["skills_with_ref"] if l]
            if not B:
                continue
            eb = model.encode(B, normalize_embeddings=True, show_progress_bar=False)
            newk = set(np.argmax(eb @ centro.T, axis=1).tolist())
            oldk = set(np.nonzero(q[i])[0].tolist())
            if newk != oldk:
                changed += 1
        print(f"  items whose cluster set would change: {changed}/{len(rows)} ({changed/len(rows):.1%})", flush=True)
    except Exception as ex:
        print(f"  (skipped: {ex})", flush=True)
        changed = None

    out = {
        "experiment": "reference_answer_ablation_analysis",
        "n_items": len(rows),
        "overall": {"exact_jaccard": float(np.mean([r["exact"] for r in rows])),
                    "token_jaccard": float(np.mean([r["tok"] for r in rows])),
                    "identical_set_rate": float(np.mean([r["exact"] == 1 for r in rows]))},
        "by_benchmark": {b: {"n": sum(1 for r in rows if r["bench"] == b),
                             "exact_jaccard": float(np.mean([r["exact"] for r in rows if r["bench"] == b])),
                             "token_jaccard": float(np.mean([r["tok"] for r in rows if r["bench"] == b]))}
                         for b in ["MATH", "GPQA"]},
        "stem_reveals_split": {
            "median_coverage": med,
            "stem_reveals": {"n": len(hi), "token_jaccard": float(np.mean([r["tok"] for r in hi]))},
            "stem_hides": {"n": len(lo), "token_jaccard": float(np.mean([r["tok"] for r in lo]))}},
        "cluster_change_rate": changed,
        "verified": True,
    }
    (E / "v2_refablation_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {E / 'v2_refablation_analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
