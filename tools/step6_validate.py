#!/usr/bin/env python3
"""Step 6: validation. Four measurements, run with a different model.

The annotation requires a judge that did not build the codebook, otherwise we
check our own work with the judge we tuned against. CODEBOOK built it
(gemini-3.1-pro-preview) and BULK labelled with it (gemini-3.5-flash-lite);
JUDGE (gemini-3.6-flash) is neither.

  coherence    within-code: sample up to 10 items per code, ask blind whether
               each requires the operation the definition names -> per-skill
               precision. Low scorers are split candidates.
  distinctness between-code: for pairs that are near-identical by definition
               embedding OR frequently co-assigned, ask for a discriminating
               rule about the OPERATION. A rule about subject matter does not
               count. If none can be written, MERGE.
  stability    re-label a sample with the candidate order shuffled and a
               different seed, measure agreement. Disagreement means an
               ambiguous definition.
  gold         NOT IMPLEMENTABLE HERE. The doc's highest-priority item is 200
               hand-labelled questions, and hand-labelled means by a person.

    python3 tools/step6_validate.py coherence [--sample 10] [--smoke]
"""
import argparse, json, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, JUDGE

REPO = Path(__file__).resolve().parent.parent
P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
WORKERS = 16

COH_SYS = ("You check whether a test question requires a specific cognitive operation. "
           "You judge by the operation a solver must perform, never by the subject matter. "
           "You are strict: partial topical overlap is not a match.")
COH_U = """OPERATION: {definition}

For each question below, does answering it actually require that operation?

{questions}

Return JSON only: {{"verdicts": [{{"i": 1, "match": true}}, ...]}}"""

DIS_SYS = ("You decide whether two cognitive skills are genuinely different operations. "
           "A rule that separates them by subject matter, topic or cover story does NOT "
           "count: if a solver does the same thing in both, they are the same skill.")
DIS_U = """For each pair, write a rule that tells them apart BY THE OPERATION a solver
performs. If no such rule exists, say MERGE.

{pairs}

Return JSON only:
{{"verdicts": [{{"pair": 1, "merge": false, "rule": "<operation-level rule, or null>"}}]}}"""


def load_items():
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    return txt


def load_state():
    f = P / "codebook_v2_amended.json"
    if not f.exists():
        f = P / "codebook_v1_frozen.json"
    fz = json.loads(f.read_text())
    rows = [json.loads(l) for l in (P / "item_labels.jsonl").open()]
    ok = [r for r in rows if "error" not in r]
    by_code = defaultdict(list)
    for r in ok:
        for a in r["assigned"]:
            by_code[a["code"]].append(r["item_idx"])
    return fz, ok, by_code


def coherence(a):
    fz, ok, by_code = load_state(); codes = fz["codes"]; txt = load_items()
    g = Gemini(); rng = random.Random(42)
    targets = [c for c in by_code if len(by_code[c]) >= 2]
    if a.smoke:
        targets = targets[:5]
    print(f"coherence: {len(targets):,} codes with >=2 items, up to {a.sample} questions each, judge {JUDGE}")

    def one(c):
        items = by_code[c][:]
        rng.shuffle(items)
        items = items[:a.sample]
        qs = "\n".join(f"{i+1}. {txt[j][:600]}" for i, j in enumerate(items))
        try:
            obj = g.json_obj(COH_SYS, COH_U.format(definition=codes[c]["definition"], questions=qs),
                             model=JUDGE, max_out=4000)
        except GeminiError as e:
            return {"code": c, "error": str(e)[:120]}
        v = [bool(x.get("match")) for x in obj.get("verdicts", [])][:len(items)]
        return {"code": c, "n": len(v), "matched": sum(v),
                "precision": (sum(v) / len(v)) if v else None}

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(one, targets), 1):
            out.append(r)
            if k % 100 == 0 or k == len(targets):
                print(f"  {k:,}/{len(targets):,} | {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)
    good = [r for r in out if r.get("precision") is not None]
    ps = sorted(r["precision"] for r in good)
    print(f"\ncodes judged: {len(good):,} ({sum(1 for r in out if 'error' in r)} errors)")
    if ps:
        print(f"per-skill precision: mean {sum(ps)/len(ps):.3f}, median {ps[len(ps)//2]:.3f}, "
              f"p10 {ps[len(ps)//10]:.3f}")
        for thr in (0.5, 0.7, 0.9):
            print(f"  codes below {thr:.0%}: {sum(1 for p in ps if p < thr):,} ({sum(1 for p in ps if p < thr)/len(ps):.0%})")
    if not a.smoke:
        (P / "validation_coherence.json").write_text(json.dumps(
            {"judge": JUDGE, "sample": a.sample, "results": out}, indent=1))
        print("wrote validation_coherence.json")


def distinctness(a):
    fz, ok, by_code = load_state(); codes = fz["codes"]
    z = np.load(P / "code_def_emb.npz", allow_pickle=True)
    ids = list(z["ids"]); C = z["vecs"] / np.linalg.norm(z["vecs"], axis=1, keepdims=True)
    S = C @ C.T; np.fill_diagonal(S, -1)
    sim_pairs = {(ids[i], ids[j]) for i in range(len(ids)) for j in np.where(S[i] >= 0.85)[0] if i < j}
    co = Counter()
    for r in ok:
        cs = sorted({x["code"] for x in r["assigned"]})
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                co[(cs[i], cs[j])] += 1
    co_pairs = {p for p, n in co.items() if n >= 20}
    pairs = sorted(sim_pairs | co_pairs)
    if a.smoke:
        pairs = pairs[:10]
    print(f"distinctness: {len(sim_pairs)} near-identical + {len(co_pairs)} frequently co-assigned "
          f"= {len(pairs)} pairs, judge {JUDGE}")
    g = Gemini(); B = 10
    chunks = [pairs[i:i + B] for i in range(0, len(pairs), B)]

    def one(ch):
        txt = "\n".join(
            f"PAIR {k+1}:\n  A {x} | {codes[x]['name']} | {codes[x]['definition']}\n"
            f"  B {y} | {codes[y]['name']} | {codes[y]['definition']}"
            for k, (x, y) in enumerate(ch) if x in codes and y in codes)
        try:
            obj = g.json_obj(DIS_SYS, DIS_U.format(pairs=txt), model=JUDGE, max_out=6000)
        except GeminiError as e:
            return [{"pair": list(p), "error": str(e)[:100]} for p in ch]
        out = []
        for v in obj.get("verdicts", []):
            k = v.get("pair", 0) - 1
            if 0 <= k < len(ch):
                out.append({"pair": list(ch[k]), "merge": bool(v.get("merge")), "rule": v.get("rule")})
        return out

    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(one, chunks), 1):
            res.extend(r)
            if k % 10 == 0 or k == len(chunks):
                print(f"  {k}/{len(chunks)} chunks | {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)
    m = [r for r in res if r.get("merge")]
    print(f"\npairs judged: {len(res):,} | MERGE verdicts: {len(m):,} ({len(m)/max(1,len(res)):.0%})")
    invol = {c for r in m for c in r["pair"]}
    print(f"codes implicated in a merge: {len(invol):,}")
    if not a.smoke:
        (P / "validation_distinctness.json").write_text(json.dumps(
            {"judge": JUDGE, "results": res}, indent=1))
        print("wrote validation_distinctness.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["coherence", "distinctness", "gold"])
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.what == "gold":
        print("The gold set is 200 hand-labelled questions. Hand-labelled means by a person,\n"
              "so this cannot be generated here. The doc ranks it first for return on effort:\n"
              "without it, no pipeline change can be shown to be an improvement.")
    else:
        {"coherence": coherence, "distinctness": distinctness}[a.what](a)
