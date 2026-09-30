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
import argparse, hashlib, json, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, JUDGE, BULK
from tools.textclip import clip
from tools.metrics import coherence_precision, jaccard

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


def out_name(a, base):
    """Validation results are named for the codebook they describe.

    Every check wrote to a fixed filename, so scoring a second codebook silently
    destroyed the first one's per-skill results. That is how the v6 coherence
    detail was lost on 2026-09-09: the run log's by-size table and floor
    analysis were derived from a file the next run overwrote, and only the
    aggregates survived in prose. Aggregates are not enough to re-verify a claim.
    """
    cb = getattr(a, "codebook", None)
    if not cb:
        return base
    stem = Path(cb).stem.replace("codebook_", "")
    return f"{base.rsplit('.', 1)[0]}_{stem}.json"


def load_items():
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    return txt


def load_state(name=None, labels=None, allow_unused=0):
    """Codebook and labels, with the pairing checked rather than assumed.

    A later step migrates item_labels.jsonl onto a merged codebook, and the
    merged ids are a SUBSET of the earlier ones, so "every label names a known
    code" still passes and the mismatch is invisible. The invariant that catches
    it is the other direction: scoring codebook A against labels migrated to a
    later codebook B leaves A's merged-away codes holding nothing.
    """
    if name:
        f = P / name
    else:
        for cand in ("codebook_v5_deduped.json", "codebook_v4_definitions.json",
                     "codebook_v3_audited.json",
                     "codebook_v2_amended.json", "codebook_v1_frozen.json"):
            f = P / cand
            if f.exists():
                break
    fz = json.loads(f.read_text())
    lf = P / (labels or "item_labels.jsonl")
    print(f"  codebook: {f.name}  labels: {lf.name}")
    rows = [json.loads(l) for l in lf.open() if l.strip()]
    ok = [r for r in rows if "error" not in r]
    live = {c for c in fz["codes"] if c not in fz.get("alias", {})}
    used = {x["code"] for r in ok for x in r["assigned"]}
    empty = live - used
    stray = used - set(fz["codes"])
    print(f"  {len(live) - len(empty)} of {len(live)} live codes hold at least one item"
          + (f"; {len(empty)} unused" if empty else ""))
    if len(empty) > max(allow_unused, 0.05 * max(1, len(live))):
        raise SystemExit(
            f"{len(empty)} of {len(live)} live codes in {f.name} hold no item in "
            f"{lf.name}.\n"
            f"Two different situations look like this and only one is a bug.\n"
            f"  (a) MISMATCH: these files are from different stages, the labels having "
            f"been migrated onto a later codebook. Pass --labels with the snapshot "
            f"that matches, e.g. item_labels_before_<that codebook>.jsonl.\n"
            f"  (b) PROPERTY OF THE RUN: Step 2 created skills that Step 4 never "
            f"assigned. Nothing is wrong and the count is real.\n"
            f"Evidence here: {len(stray)} codes named in the labels are absent from "
            f"this codebook, and the codebook carries {len(fz.get('alias', {}))} "
            f"aliases. Both zero points to (b), since a migration leaves either "
            f"strays or aliases behind. If it is (b), say so explicitly with "
            f"--allow-unused {len(empty)} rather than loosening the 5% threshold, "
            f"which would disarm this check for every future run.")
    by_code = defaultdict(list)
    for r in ok:
        for a in r["assigned"]:
            by_code[a["code"]].append(r["item_idx"])
    return fz, ok, by_code


def coherence(a):
    fz, ok, by_code = load_state(getattr(a, "codebook", None), getattr(a, "labels", None),
                                   getattr(a, "allow_unused", 0))
    codes = fz["codes"]; txt = load_items()
    field = "definition_before" if getattr(a, "use_before", False) else "definition"
    rewritten = {c for c, v in codes.items() if "definition_before" in v}
    if getattr(a, "use_before", False) and not rewritten:
        raise SystemExit("--use-before needs a codebook that Step 8 has written")
    if rewritten and (getattr(a, "use_before", False) or getattr(a, "holdout", False)):
        # Compare like with like. Only codes Step 8 actually rewrote have a
        # before and an after; including the rest would dilute both sides with
        # identical scores and understate whatever the repair did.
        by_code = {c: v for c, v in by_code.items() if c in rewritten}
        print(f"  restricted to the {len(rewritten):,} codes Step 8 rewrote")
    if getattr(a, "holdout", False):
        # score only on items the definition writer never saw, or the comparison
        # is circular: a definition rewritten from items trivially matches them
        hold = {c: set(v.get("holdout_items", [])) for c, v in codes.items()}
        if not any(hold.values()):
            raise SystemExit("--holdout needs a codebook that Step 8 has written")
        by_code = {c: [i for i in v if i in hold.get(c, set())] for c, v in by_code.items()}
    g = Gemini()
    targets = [c for c in by_code if len(by_code[c]) >= 2]
    if a.smoke:
        targets = targets[:5]
    print(f"coherence: {len(targets):,} codes with >=2 items, up to {a.sample} questions each, judge {JUDGE}")

    def one(c):
        items = by_code[c][:]
        # sha256, not hash(): Python randomises string hashing per process unless
        # PYTHONHASHSEED is set, and it is set nowhere here, so this "reproducible"
        # per-code shuffle drew a different sample of questions on every run and no
        # stored coherence score can be re-derived from the artifacts. T-115.
        seed = 42 + int(hashlib.sha256(c.encode()).hexdigest()[:8], 16)
        random.Random(seed).shuffle(items)                   # per-code, and now truly reproducible
        items = items[:a.sample]
        # same window the labeller saw: judging on 600 chars what was decided on
        # 4,000 depresses precision by truncation rather than by disagreement
        qs = "\n".join(f"{i+1}. {clip(txt[j], 4000)}" for i, j in enumerate(items))
        try:
            obj = g.json_obj(COH_SYS, COH_U.format(definition=codes[c][field], questions=qs),
                             model=JUDGE, max_out=4000)
        except GeminiError as e:
            return {"code": c, "error": str(e)[:120]}
        prec, got, missing = coherence_precision(obj.get("verdicts"), len(items))
        # Record WHICH item each verdict was about, not only how many passed. The
        # artifact stored counts alone, so it could not answer "which assignments
        # did the judge reject", which is the first question asked of it the moment
        # the judge disagrees with a human (T-114). Counts cannot be re-examined.
        seen = {}
        for v in (obj.get("verdicts") or []):
            try:
                k = int(v.get("i")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= k < len(items):
                seen[items[k]] = bool(v.get("match"))
        return {"code": c, "sent": len(items), "returned": got, "missing": missing,
                "matched": round((prec or 0) * len(items)), "precision": prec,
                "verdicts": {str(i): m for i, m in seen.items()}}

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
        tag = ("_holdout_before" if (a.holdout and a.use_before)
               else "_holdout_after" if a.holdout else "")
        f = P / out_name(a, f"validation_coherence{tag}.json")
        f.write_text(json.dumps({"judge": JUDGE, "sample": a.sample, "field": field,
                                 "holdout_only": bool(a.holdout), "results": out}, indent=1))
        print(f"wrote {f.name}")


def distinctness(a):
    fz, ok, by_code = load_state(getattr(a, "codebook", None), getattr(a, "labels", None),
                                   getattr(a, "allow_unused", 0)); codes = fz["codes"]
    # Nominate on the definitions as they stand now. Reading the cache blind
    # meant this check selected its "near-identical" candidates from pre-Step-8
    # text, so the pairs it judged were themselves chosen on superseded wording
    # and the duplicate count it reports is a floor.
    from tools.codeemb import load as load_code_vecs, normed
    ids = [c for c in codes if c not in fz.get("alias", {})]
    C = normed(load_code_vecs(P / getattr(a, "emb", "code_def_emb.npz"), fz, ids, g=Gemini()))
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
        f = P / out_name(a, "validation_distinctness.json")
        f.write_text(json.dumps({"judge": JUDGE, "results": res}, indent=1))
        print(f"wrote {f.name}")


def stability(a):
    """The doc: re-run labelling with the candidate order shuffled and a different
    seed, then measure agreement. Disagreement means an ambiguous definition.

    Default is the SAME model that did the labelling. The annotation's
    different-model rule is about not checking our own work with the judge we
    tuned against, which applies to coherence and distinctness. Here a model
    swap would confound the thing being measured: disagreement would mix
    order-sensitivity with cross-model difference. --cross-model measures that
    separately, and it is a different quantity."""
    fz, ok, _ = load_state(getattr(a, "codebook", None), getattr(a, "labels", None),
                                   getattr(a, "allow_unused", 0)); codes = fz["codes"]; txt = load_items()
    # retrieval has to offer candidates described the way they are described now,
    # or the re-run is choosing between definitions that no longer exist
    from tools.codeemb import load as load_code_vecs, normed
    ids = [c for c in codes if c not in fz.get("alias", {})]
    C = normed(load_code_vecs(P / getattr(a, "emb", "code_def_emb.npz"), fz, ids, g=Gemini()))
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
        u = (f"QUESTION:\n{clip(txt[i], 4000)}\n\nCANDIDATE SKILLS:\n{ctxt}\n\n"
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
        return {"item_idx": i, "orig": sorted(a_), "rerun": sorted(b),
                "jaccard": jaccard(a_, b), "exact": a_ == b}

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
        f = P / out_name(a, "validation_stability.json")
        f.write_text(json.dumps(
            {"model": model, "cross_model": bool(a.cross_model), "seed": 4242,
             "results": res}, indent=1))
        print(f"wrote {f.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["coherence", "distinctness", "stability", "gold"])
    ap.add_argument("--sample-items", type=int, default=1000)
    ap.add_argument("--cross-model", action="store_true",
                    help="re-label with JUDGE instead; measures cross-model agreement, "
                         "which is a different quantity from order stability")
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--codebook", default=None, help="codebook filename to score")
    ap.add_argument("--labels", default=None,
                    help="label file matching that codebook; defaults to "
                         "item_labels.jsonl, which tracks the LATEST codebook")
    ap.add_argument("--allow-unused", type=int, default=0,
                    help="how many live codes are expected to hold no item. The guard "
                         "against scoring a codebook against another stage's labels "
                         "trips on this too, so a run whose Step 2 created skills that "
                         "Step 4 never assigned declares the count here. Declaring it "
                         "keeps the guard armed for every other run; raising the "
                         "threshold would not")
    ap.add_argument("--emb", default="code_def_emb.npz",
                    help="definition-embedding cache for THIS run. Distinctness nominates "
                         "its candidate pairs from it and stability retrieves candidates "
                         "from it, so a cache left over from another run judges another "
                         "taxonomy's definitions while reporting this one's ids. Code ids "
                         "restart at c_0001 every run, so the mismatch is silent. Pass the "
                         "run-tagged file, e.g. code_def_emb_b150.npz")
    ap.add_argument("--holdout", action="store_true",
                    help="score only items Step 8's definition writer never saw")
    ap.add_argument("--use-before", action="store_true",
                    help="score the pre-repair definitions, for a like-for-like baseline")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.what == "gold":
        print("The gold set is 200 hand-labelled questions. Hand-labelled means by a person,\n"
              "so this cannot be generated here. The doc ranks it first for return on effort:\n"
              "without it, no pipeline change can be shown to be an improvement.")
    else:
        {"coherence": coherence, "distinctness": distinctness,
         "stability": stability}[a.what](a)
