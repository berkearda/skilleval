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
from tools.gemini import Gemini, GeminiError, JUDGE, BULK

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
    g = Gemini()
    targets = [c for c in by_code if len(by_code[c]) >= 2]
    if a.smoke:
        targets = targets[:5]
    print(f"coherence: {len(targets):,} codes with >=2 items, up to {a.sample} questions each, judge {JUDGE}")

    def one(c):
        items = by_code[c][:]
        random.Random(42 + hash(c) % 10**6).shuffle(items)   # per-code, so it reproduces
        items = items[:a.sample]
        # same window the labeller saw: judging on 600 chars what was decided on
        # 4,000 depresses precision by truncation rather than by disagreement
        qs = "\n".join(f"{i+1}. {txt[j][:4000]}" for i, j in enumerate(items))
        try:
            obj = g.json_obj(COH_SYS, COH_U.format(definition=codes[c]["definition"], questions=qs),
                             model=JUDGE, max_out=4000)
        except GeminiError as e:
            return {"code": c, "error": str(e)[:120]}
        got = {}
        for x in obj.get("verdicts", []):
            if not isinstance(x, dict):
                continue
            try:
                idx = int(x.get("i", 0)) - 1        # index by i, never by position:
            except (TypeError, ValueError):         # a skipped question would otherwise
                continue                            # shift every later verdict
            if 0 <= idx < len(items):
                got[idx] = bool(x.get("match"))
        missing = len(items) - len(got)
        return {"code": c, "sent": len(items), "returned": len(got), "missing": missing,
                "matched": sum(got.values()),
                # denominator is what was SENT; an unreturned verdict is not a match
                "precision": (sum(got.values()) / len(items)) if items else None,
                "precision_on_returned": (sum(got.values()) / len(got)) if got else None}

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(one, targets), 1):
            out.append(r)
            if k % 100 == 0 or k == len(targets):
                print(f"  {k:,}/{len(targets):,} | {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)
    good = [r for r in out if r.get("precision") is not None]
    n_err = sum(1 for r in out if "error" in r)
    n_empty = len(out) - len(good) - n_err
    ps = sorted(r["precision"] for r in good)
    miss = sum(r.get("missing", 0) for r in good)
    print(f"\ncodes: {len(out):,} = judged {len(good):,} + errors {n_err} + empty {n_empty}")
    print(f"verdicts not returned by the judge: {miss:,} "
          f"({miss/max(1,sum(r['sent'] for r in good)):.1%} of questions sent)")
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
    absent = [c for c in codes if c not in set(ids)]
    if absent:
        print(f"  WARNING: {len(absent)} codes have no embedding and cannot be "
              f"nominated or retrieved: {absent[:5]}{' ...' if len(absent) > 5 else ''}")
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
        ch = [(x, y) for x, y in ch if x in codes and y in codes]   # filter BEFORE
        if not ch:                                                   # numbering, or the
            return []                                                # echoed indices gap
        txt = "\n".join(
            f"PAIR {k+1}:\n  A {x} | {codes[x]['name']} | {codes[x]['definition']}\n"
            f"  B {y} | {codes[y]['name']} | {codes[y]['definition']}"
            for k, (x, y) in enumerate(ch))
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
    judged = [r for r in res if "error" not in r]
    m = [r for r in judged if r.get("merge")]
    print(f"\npairs: {len(res):,} = judged {len(judged):,} + errored {len(res)-len(judged):,}")
    print(f"MERGE verdicts: {len(m):,} ({len(m)/max(1,len(judged)):.0%} of judged)")
    invol = {c for r in m for c in r["pair"]}
    print(f"codes implicated in a merge: {len(invol):,}")
    if not a.smoke:
        (P / "validation_distinctness.json").write_text(json.dumps(
            {"judge": JUDGE, "results": res}, indent=1))
        print("wrote validation_distinctness.json")


def stability(a):
    """The doc: re-run labelling with the candidate order shuffled and a different
    seed, then measure agreement. Disagreement means an ambiguous definition.

    Default is the SAME model that did the labelling. The annotation's
    different-model rule is about not checking our own work with the judge we
    tuned against, which applies to coherence and distinctness. Here a model
    swap would confound the thing being measured: disagreement would mix
    order-sensitivity with cross-model difference. --cross-model measures that
    separately, and it is a different quantity."""
    fz, ok, _ = load_state(); codes = fz["codes"]; txt = load_items()
    z = np.load(P / "code_def_emb.npz", allow_pickle=True)
    ids = list(z["ids"]); C = z["vecs"] / np.linalg.norm(z["vecs"], axis=1, keepdims=True)
    iz = np.load(P / "item_emb_gemini.npz", allow_pickle=True)
    pos = {int(i): k for k, i in enumerate(list(iz["idx"]))}
    V = iz["vecs"] / np.linalg.norm(iz["vecs"], axis=1, keepdims=True)

    rng = random.Random(4242)                      # a different seed, as the doc asks
    pool = [r for r in ok if r["assigned"]]
    sample = rng.sample(pool, 30 if a.smoke else min(a.sample_items, len(pool)))
    model = JUDGE if a.cross_model else BULK
    print(f"stability: re-labelling {len(sample):,} items, shuffled candidate order, "
          f"model {model} ({'cross-model' if a.cross_model else 'same model as Step 4'})")
    g = Gemini()

    def one(r):
        i = r["item_idx"]
        from tools.step4_relabel import TOPK as S4TOPK
        cand = [ids[j] for j in np.argsort(-(V[pos[i]] @ C.T))[:S4TOPK]]
        random.Random(4242 + i).shuffle(cand)      # the shuffle the doc specifies
        ctxt = "\n".join(f"{c} | {codes[c]['name']} | {codes[c]['definition']}"
                          for c in cand if c in codes)
        u = (f"QUESTION:\n{txt[i][:2500]}\n\nCANDIDATE SKILLS:\n{ctxt}\n\n"
             "Choose the skills a solver must actually perform, at most 3. Choose one if "
             "one is enough. Return JSON only: {\"assigned\": [{\"code\": \"c_0123\"}]}")
        try:
            obj = g.json_obj("You assign cognitive skills to test questions from a fixed "
                             "codebook. You judge by the operation the solver performs, "
                             "never by the subject matter.", u, model=model, max_out=1200)
        except GeminiError as e:
            return {"item_idx": i, "error": str(e)[:100]}
        # the judge returns assigned as objects or as bare id strings; accept both
        b = set()
        for x in obj.get("assigned", []):
            cid = x.get("code") if isinstance(x, dict) else x
            if isinstance(cid, str) and cid in codes:
                b.add(cid)
        a_ = {x["code"] for x in r["assigned"]}
        inter, union = len(a_ & b), len(a_ | b)
        return {"item_idx": i, "orig": sorted(a_), "rerun": sorted(b),
                "jaccard": inter / union if union else 1.0, "exact": a_ == b}

    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(one, sample), 1):
            res.append(r)
            if k % 100 == 0 or k == len(sample):
                print(f"  {k:,}/{len(sample):,} | {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)
    good = [r for r in res if "error" not in r]
    if good:
        js = sorted(r["jaccard"] for r in good)
        print(f"\nitems compared: {len(good):,} ({len(res)-len(good)} errors)")
        print(f"exact set match: {sum(r['exact'] for r in good)/len(good):.1%}")
        print(f"Jaccard: mean {sum(js)/len(js):.3f}, median {js[len(js)//2]:.3f}, "
              f"zero-overlap {sum(1 for j in js if j == 0)/len(js):.1%}")
    if not a.smoke:
        (P / "validation_stability.json").write_text(json.dumps(
            {"model": model, "cross_model": bool(a.cross_model), "seed": 4242,
             "results": res}, indent=1))
        print("wrote validation_stability.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["coherence", "distinctness", "stability", "gold"])
    ap.add_argument("--sample-items", type=int, default=1000)
    ap.add_argument("--cross-model", action="store_true",
                    help="re-label with JUDGE instead; measures cross-model agreement, "
                         "which is a different quantity from order stability")
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.what == "gold":
        print("The gold set is 200 hand-labelled questions. Hand-labelled means by a person,\n"
              "so this cannot be generated here. The doc ranks it first for return on effort:\n"
              "without it, no pipeline change can be shown to be an improvement.")
    else:
        {"coherence": coherence, "distinctness": distinctness,
         "stability": stability}[a.what](a)
