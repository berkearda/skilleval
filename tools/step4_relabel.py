#!/usr/bin/env python3
"""Step 4: re-label every question from the original question text.

The doc calls this the biggest change from the old pipeline. Labels now come
from the question judged against the frozen codebook, not from Step 1's raw
labels, so an extraction error no longer propagates into the final taxonomy.

Per item: retrieve the top-k codes by embedding the question against each
code's definition, then have the model pick from those candidates and return
the selected code(s), a confidence, and its reasoning, or UNASSIGNABLE with a
proposed skill. Step 0 caps it at 3 skills per question, allows single-skill
questions (they are the only ones that identify a skill cleanly), and requires
that "no skill" be allowed and reported rather than force-fitted.

Questions are embedded with the same embedder as the code definitions. The
SBERT-family files in cdm_ready/ are a different vector space and cannot be
used for this retrieval; one of them is also the truncated-preview artifact
that T-071 exists to replace.

Unlike Step 2 the items are independent, so this runs concurrently.

    python3 tools/step4_relabel.py smoke     # 20 items
    python3 tools/step4_relabel.py run       # all, resumable
"""
import json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, BULK

REPO = Path(__file__).resolve().parent.parent
P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"

TOPK = 12          # candidates shown per question
MAX_SKILLS = 3     # Step 0's cap
WORKERS = 16

SYS = ("You assign cognitive skills to test questions from a fixed codebook. A skill is the "
       "smallest named mental operation a solver must perform to answer correctly. You judge "
       "by the operation the solver performs, never by the subject matter or cover story. "
       "You only ever choose from the candidates given, or say UNASSIGNABLE.")

USER = """QUESTION ({benchmark} / {subtask}):
{question}

CANDIDATE SKILLS:
{codes}

Choose the skills a solver must actually perform to answer this question, at most
{maxk}. Choose one if one is enough: do not pad. If none of the candidates is an
operation this question requires, return UNASSIGNABLE instead and propose the skill
that is missing.

Return JSON only:
{{"assigned": [{{"code": "c_0123", "confidence": 0.0-1.0, "why": "<one clause>"}}],
  "unassignable": false,
  "proposed_skill": null}}"""


def load_items():
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {p["item_idx"]: p for p in
            json.load(open(REPO / "cdm_exploration/experiments/oldtax_repaired_FINAL.json"))["per_item"]}
    return [{"item_idx": i, "question": txt[i][:4000],
             "benchmark": meta.get(i, {}).get("benchmark"),
             "subtask": meta.get(i, {}).get("subtask")} for i in sorted(txt)]


def item_vectors(g, items):
    cache = P / "item_emb_gemini.npz"
    idx = [it["item_idx"] for it in items]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["idx"]) == idx:
            print(f"  loaded {len(idx):,} cached question embeddings")
            return z["vecs"]
    print(f"  embedding {len(idx):,} questions (this is the only serial part) ...")
    V = np.stack(g.embed([it["question"][:2000] for it in items]))
    np.savez_compressed(cache, idx=np.array(idx, dtype=object), vecs=V)
    return V


def main(mode="smoke"):
    fz = json.loads((P / "codebook_v1_frozen.json").read_text())
    codes = fz["codes"]
    ids = list(codes)
    items = load_items()
    outf = P / ("labels_smoke.jsonl" if mode == "smoke" else "item_labels.jsonl")
    if mode == "smoke":
        items = items[:10] + items[5000:5010]
        outf.unlink(missing_ok=True)

    g = Gemini()
    z = np.load(P / "code_def_emb.npz", allow_pickle=True)
    assert list(z["ids"]) == ids, "frozen codebook and definition embeddings disagree"
    C = z["vecs"] / np.linalg.norm(z["vecs"], axis=1, keepdims=True)
    Vall = item_vectors(g, load_items())
    pos = {it["item_idx"]: k for k, it in enumerate(load_items())}
    V = np.stack([Vall[pos[it["item_idx"]]] for it in items])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    done = {json.loads(l)["item_idx"] for l in outf.open()} if outf.exists() else set()
    todo = [(i, it) for i, it in enumerate(items) if it["item_idx"] not in done]
    print(f"step 4: {len(todo):,} items to label ({len(done):,} done), "
          f"{len(ids):,} codes, top-{TOPK} candidates, model {BULK}")
    if not todo:
        return

    sims = V @ C.T

    def one(pair):
        i, it = pair
        cand = [ids[j] for j in np.argsort(-sims[i])[:TOPK]]
        txt = "\n".join(
            f"{c} | {codes[c]['name']} | {codes[c]['definition']}" for c in cand)
        u = USER.format(benchmark=it["benchmark"], subtask=it["subtask"],
                        question=it["question"], codes=txt, maxk=MAX_SKILLS)
        try:
            obj = g.json_obj(SYS, u, model=BULK, max_out=1600)
        except GeminiError as e:
            return {"item_idx": it["item_idx"], "error": str(e)[:200]}
        asg = [a for a in obj.get("assigned", []) if a.get("code") in codes][:MAX_SKILLS]
        return {"item_idx": it["item_idx"], "benchmark": it["benchmark"],
                "subtask": it["subtask"], "candidates": cand,
                "assigned": asg, "unassignable": bool(obj.get("unassignable")),
                "proposed_skill": obj.get("proposed_skill")}

    t0, n_err, n_un, n_asg = time.time(), 0, 0, Counter()
    with open(outf, "a") as f:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for k, r in enumerate(ex.map(one, todo), 1):
                f.write(json.dumps(r) + "\n"); f.flush()
                if "error" in r:
                    n_err += 1
                else:
                    n_un += r["unassignable"]; n_asg[len(r["assigned"])] += 1
                if k % 100 == 0 or k == len(todo):
                    print(f"  {k:5,}/{len(todo):,} | unassignable {n_un} | "
                          f"skills/item {dict(sorted(n_asg.items()))} | errors {n_err} | "
                          f"{g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)
    print(f"\ndone: {sum(n_asg.values()):,} labelled, {n_un} unassignable, {n_err} errors")
    print(f"skills per item: {dict(sorted(n_asg.items()))}")
    print(f"usage: {g.report()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke")
