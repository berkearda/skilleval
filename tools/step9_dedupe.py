#!/usr/bin/env python3
"""Step 9: merge near-duplicate codes that every earlier criterion missed.

Diagnosis. Hand-labelling Task 1, Berke answered "neither definition fits" on 6
of 14 items. Four of those six were the same operation split four ways:

    c_0286 tracking object locations from narrative context   13 items
    c_0464 identifying object locations from narrative context 75 items
    c_0790 retrieving stated item locations from text          32 items
    c_0854 retrieving object locations from narratives         37 items

"Neither fits" was not a wording problem. It was being asked to choose between
two phrasings of a code that should not exist four times over.

Why every existing criterion missed them:

  size floor      three of the four are above 20 items
  similarity      pairwise cosine 0.67-0.79, all below the 0.85 threshold
  co-assignment   0-2, far below 20

The similarity threshold is the clearest error: on this codebook 0.85 sits above
the 99.9th percentile and nominates 28 pairs out of ~68,000. An auditor flagged
exactly this and observed that the annotation's own motivating case (three angle
skills, five area skills) also fell below it. That was recorded and not acted on.

The co-assignment criterion is worse than mis-tuned, it is backwards for this
purpose. Near-duplicates compete for the same item, so exactly one wins and they
co-occur almost never. Co-assignment finds COMPLEMENTARY skills, not
INTERCHANGEABLE ones. Using it to detect duplicates asks for the opposite of the
signature.

The signal that does work is competition: a pair that is offered together as
candidates for many items and chosen together for almost none is two names for
one operation. It catches 3 of the 4 above directly and the fourth transitively.

Verdicts come from JUDGE, calibrated on the 20-item human control at 90%
agreement, and conservative there (it under-accepts rather than over-accepts),
so a MERGE verdict from it is evidence rather than assertion.

    python3 tools/step9_dedupe.py --dry-run
"""
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.gemini import Gemini, GeminiError, JUDGE
from tools.metrics import apply_merges, resolve as m_resolve

P = REPO / "cdm_exploration/experiments/pipeline_v7"
SIM_PCT = 99.0        # percentile, not an absolute cosine: 0.85 was above p99.9 here
OFFERED_MIN = 10      # times a pair was offered together as candidates
CHOSEN_MAX = 2        # times it was chosen together; duplicates almost never are
WORKERS = 12

SYS = ("You decide whether two cognitive skills are genuinely different operations. A rule "
       "that separates them by subject matter, topic, phrasing or cover story does NOT count: "
       "if a solver does the same thing in both, they are the same skill and must be merged.")
USER = """For each pair, write a rule that tells them apart BY THE OPERATION a solver
performs. If no such rule exists, say MERGE.

These pairs were flagged because they compete: both were offered as candidates for
many of the same questions and almost never chosen together. That is the signature
of one operation with two names, so expect many genuine merges.

{pairs}

Return JSON only:
{{"verdicts": [{{"pair": 1, "merge": true|false, "rule": "<operation-level rule, or null>"}}]}}"""


def resolve(alias, cid):
    seen = set()
    while cid in alias and cid not in seen:
        seen.add(cid); cid = alias[cid]
    return cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ceiling", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--reuse-verdicts", action="store_true",
                    help="load validation_dedupe.json instead of re-judging, so a change "
                         "to the application is isolated from judge variance")
    a = ap.parse_args()
    if not a.no_gate:
        from tools.gate import require_tests_pass
        require_tests_pass()

    fz = json.loads((P / "codebook_v4_definitions.json").read_text())
    codes = fz["codes"]
    rows = [json.loads(l) for l in (P / "item_labels.jsonl").open()]
    ok = [r for r in rows if "error" not in r]
    # Nominate on the definitions as they stand NOW. Step 8 rewrote 271 of these
    # 379, and reading the cache blind meant similarity was computed on the text
    # they used to have: two codes Step 8 had renamed to the same string scored
    # 0.698 and were never nominated. `live` comes from the codebook, not the
    # cache, so a code absent from the cache is embedded rather than dropped.
    from tools.codeemb import load as load_code_vecs, normed
    live = [c for c in codes if c not in fz.get("alias", {})]
    g = Gemini()
    V = normed(load_code_vecs(P / "code_def_emb.npz", fz, live, g=g))
    M = V; S = M @ M.T; np.fill_diagonal(S, -1)
    thr = float(np.percentile(S[np.triu_indices(len(live), 1)], SIM_PCT))

    offered, chosen = defaultdict(set), defaultdict(set)
    for r in ok:
        for c in r.get("candidates", []):
            offered[c].add(r["item_idx"])
        for x in r["assigned"]:
            chosen[x["code"]].add(r["item_idx"])
    counts = Counter(x["code"] for r in ok for x in r["assigned"])
    ceil_n = a.ceiling * len(ok)

    pairs, why = [], {}
    for i, x in enumerate(live):
        for j in range(i + 1, len(live)):
            y = live[j]
            sim = float(S[i, j])
            comp = len(offered[x] & offered[y]) >= OFFERED_MIN and \
                   len(chosen[x] & chosen[y]) <= CHOSEN_MAX
            if comp or sim >= thr:
                pairs.append((x, y))
                why[(x, y)] = ("competition" if comp else "") + ("+similar" if sim >= thr else "")
    if a.smoke:
        pairs = pairs[:20]
    print(f"step 9: {len(live):,} live codes, similarity p{SIM_PCT} = {thr:.3f}")
    print(f"  nominated {len(pairs):,} pairs "
          f"({sum(1 for p in pairs if 'competition' in why[p]):,} by competition)")

    # 6 not 10: JUDGE spends thinking tokens out of the same max_out budget, and
    # the post-Step-8 definitions are long, so 10 pairs truncated the JSON mid-array
    # and every such chunk was recorded as an error.
    CHUNK = 6
    chunks = [pairs[i:i + CHUNK] for i in range(0, len(pairs), CHUNK)]

    def one(ch):
        txt = "\n".join(
            f"PAIR {k+1}:\n  A {x} | {codes[x]['name']} | {codes[x]['definition']}\n"
            f"  B {y} | {codes[y]['name']} | {codes[y]['definition']}"
            for k, (x, y) in enumerate(ch))
        try:
            obj = g.json_obj(SYS, USER.format(pairs=txt), model=JUDGE, max_out=16000)
        except GeminiError as e:
            return [{"pair": list(p), "error": str(e)[:90]} for p in ch]
        out = []
        for v in obj.get("verdicts", []):
            k = (v.get("pair") or 0) - 1
            if 0 <= k < len(ch):
                out.append({"pair": list(ch[k]), "merge": bool(v.get("merge")),
                            "rule": v.get("rule"), "why": why[ch[k]]})
        return out

    vf = P / "validation_dedupe.json"
    if a.reuse_verdicts and vf.exists():
        res = json.loads(vf.read_text())["results"]
        print(f"  reusing {len(res):,} stored verdicts (no new judging)")
    else:
        res, t0 = [], time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for k, r in enumerate(ex.map(one, chunks), 1):
                res.extend(r)
                if k % 20 == 0 or k == len(chunks):
                    print(f"  {k}/{len(chunks)} chunks | {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)

    judged = [r for r in res if "error" not in r]
    merges = [r for r in judged if r["merge"]]
    print(f"\njudged {len(judged):,} pairs ({len(res)-len(judged)} errored)")
    print(f"MERGE verdicts: {len(merges):,} ({len(merges)/max(1,len(judged)):.0%})")

    # Merging is NOT transitive. Applying accepted pairs greedily let chains of
    # up to 8 hops compose into merges the judge had explicitly REJECTED: it
    # ruled c_0464 and c_0274 distinct, and c_0464 -> c_1047 -> c_0274 put them
    # together anyway. Greedy application also built attractors that grew to the
    # ceiling and then refused 63 further merges, leaving a barbell distribution.
    #
    # Instead: union-find over accepted edges, refusing any union that would put
    # an explicitly rejected pair in the same component, or breach the ceiling.
    accepted = [tuple(r["pair"]) for r in merges]
    rejected = [tuple(r["pair"]) for r in judged if not r["merge"]]
    alias, st = apply_merges(accepted, rejected, dict(counts), ceil_n, order="smallest")
    applied, refused, blocked = st["applied"], st["refused_ceiling"], st["blocked_rejected"]
    print(f"  components formed: {st['components']}")
    print(f"  unions blocked by an explicit keep-apart verdict: {blocked}")
    kept = [c for c in codes if resolve(alias, c) == c]
    print(f"unions applied {applied}, refused for the ceiling {refused}, blocked {blocked}")
    print(f"codes: {len(codes):,} -> {len(kept):,}")

    TAG = "step9"
    # snapshot before migrating: the alias map is many-to-one and cannot be
    # inverted, so without this a bad merge pass costs a full Step 4 re-run
    import shutil
    src = P / "item_labels.jsonl"
    if src.exists():
        shutil.copy(src, P / f"item_labels_before_{TAG}.jsonl")
        print(f"  snapshot -> item_labels_before_{TAG}.jsonl")
    moved = 0
    for r in rows:
        for x in r.get("assigned", []):
            t = resolve(alias, x["code"])
            if t != x["code"]:
                x["code"] = t; moved += 1
    per = Counter(x["code"] for r in rows if "error" not in r for x in r["assigned"])
    sz = sorted(per.values(), reverse=True)
    print(f"labels migrated: {moved:,}")
    if sz:
        print(f"items per code: max {sz[0]}, median {sz[len(sz)//2]}, used {len(per):,}")
        print(f"  below Step 0's floor of 20: {sum(1 for s in sz if s < 20):,} "
              f"({sum(1 for s in sz if s < 20)/len(sz):.0%})")
    for grp in (["c_0286", "c_0464", "c_0790", "c_0854"],):
        fam = {resolve(alias, c) for c in grp if c in codes}
        print(f"\nthe four object-location codes now resolve to {len(fam)} code(s): {sorted(fam)}")

    if a.dry_run or a.smoke:
        print("\nDRY RUN: nothing written"); return
    fz["codes"] = {c: v for c, v in codes.items() if c in set(kept) and per.get(c, 0) > 0}
    fz["version"] = "v5"
    fz["step9"] = {"nominated": len(pairs), "judged": len(judged), "merges": len(merges),
                   "applied": applied, "refused_ceiling": refused, "alias": alias,
                   "sim_percentile": SIM_PCT, "sim_threshold": thr,
                   "offered_min": OFFERED_MIN, "chosen_max": CHOSEN_MAX}
    (P / "codebook_v5_deduped.json").write_text(json.dumps(fz, indent=1))
    if not a.reuse_verdicts:
        (P / "validation_dedupe.json").write_text(json.dumps({"judge": JUDGE, "results": res}, indent=1))
    tmp = P / "item_labels.jsonl.tmp"
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, P / "item_labels.jsonl")
    print(f"\nwrote codebook_v5_deduped.json ({len(fz['codes']):,} codes)")


if __name__ == "__main__":
    main()
