#!/usr/bin/env python3
"""Step 7: the evidence-driven audit the doc defers the real decisions to.

The doc, closing Step 5:

    "The audit in Step 2 operates purely at the text level - it can see labels
    and definitions but not the actual distribution of questions. So treat that
    round as provisional ... The merge/split decisions worth trusting happen
    *after* this step, when you have real member counts per skill, coherence
    scores, and a co-assignment matrix showing which code pairs actually get
    confused. An audit driven by that evidence is a full tier better."

Step 2's audit was capped at 15 operations and hit that cap in all 31 audits, so
its merging was supply-limited throughout and its output is a capped number, not
a converged one. This step is where that is repaired, using four evidence
streams that did not exist at Step 2:

  member counts      Step 0's floor (20 questions) and ceiling (5% of corpus)
  coherence          per-skill precision from a judge that built neither the
                     codebook nor the labels
  co-assignment      which pairs actually get applied to the same question
  distinctness       an explicit MERGE verdict, or a written discriminating rule

Merges are applied through the same alias machinery Step 2 uses, so every item
label follows the chain and nothing is orphaned. Splits are PROPOSED, never
applied: splitting needs new labels, which means re-running Step 4, and the doc
is explicit that a split decision without evidence is "looks off to me".

    python3 tools/step7_audit.py [--dry-run] [--floor 20] [--coherence 0.5]
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P = Path(__file__).resolve().parent.parent / "cdm_exploration/experiments/pipeline_v7"


def resolve(alias, cid):
    seen = set()
    while cid in alias and cid not in seen:
        seen.add(cid); cid = alias[cid]
    return cid


def load():
    src = P / "codebook_v2_amended.json"
    if not src.exists():
        src = P / "codebook_v1_frozen.json"
    fz = json.loads(src.read_text())
    rows = [json.loads(l) for l in (P / "item_labels.jsonl").open()]
    ok = [r for r in rows if "error" not in r]
    coh = {}
    f = P / "validation_coherence.json"
    if f.exists():
        for r in json.loads(f.read_text())["results"]:
            if r.get("precision") is not None:
                coh[r["code"]] = r["precision"]
    dis = {}
    f = P / "validation_distinctness.json"
    if f.exists():
        for r in json.loads(f.read_text())["results"]:
            if "error" in r or len(r.get("pair", [])) != 2:
                continue
            dis[tuple(sorted(r["pair"]))] = (bool(r.get("merge")), r.get("rule"))
    return fz, rows, ok, coh, dis, src.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=int, default=20)       # Step 0's size floor
    ap.add_argument("--ceiling", type=float, default=0.05)  # Step 0's size ceiling
    ap.add_argument("--coherence", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    if not a.no_gate:
        from tools.gate import require_tests_pass
        require_tests_pass()

    fz, rows, ok, coh, dis, srcname = load()
    codes = fz["codes"]
    alias = {}
    counts = Counter(x["code"] for r in ok for x in r["assigned"])
    co = Counter()
    for r in ok:
        cs = sorted({x["code"] for x in r["assigned"]})
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                co[(cs[i], cs[j])] += 1
    ceil_n = a.ceiling * len(ok)
    print(f"step 7: {len(codes):,} codes from {srcname}, {len(ok):,} labelled items")
    print(f"  evidence: counts yes | coherence {len(coh):,} codes | "
          f"distinctness {len(dis):,} pairs | co-assignment {len(co):,} pairs")

    # the co-assignment matrix was previously built inside a local and discarded
    (P / "co_assignment.json").write_text(json.dumps(
        {"pairs": [{"a": x, "b": y, "n": n} for (x, y), n in co.most_common()]}, indent=1))

    emb = P / "code_def_emb.npz"
    sim = {}
    if emb.exists():
        z = np.load(emb, allow_pickle=True)
        ids = list(z["ids"]); V = z["vecs"] / np.linalg.norm(z["vecs"], axis=1, keepdims=True)
        S = V @ V.T; np.fill_diagonal(S, -1)
        pos = {c: i for i, c in enumerate(ids)}
        for c in ids:
            j = int(np.argmax(S[pos[c]]))
            sim[c] = (ids[j], float(S[pos[c], j]))

    ops, reasons = [], Counter()

    # 1. an explicit MERGE verdict from a judge that saw both definitions
    for (x, y), (merge, rule) in dis.items():
        if merge and x in codes and y in codes:
            keep, drop = (x, y) if counts[x] >= counts[y] else (y, x)
            ops.append({"op": "MERGE", "from": drop, "into": keep,
                        "evidence": "distinctness: no operation-level discriminator"})
            reasons["distinctness_verdict"] += 1

    # 2. below Step 0's floor, and confusable with a specific neighbour
    for c, n in counts.items():
        if c not in codes or n >= a.floor:
            continue
        partner, s = sim.get(c, (None, 0.0))
        best_co = max(((n2, o) for (p, q), n2 in co.items() if c in (p, q)
                       for o in [q if p == c else p]), default=(0, None))
        target = best_co[1] if best_co[0] >= 3 else (partner if s >= 0.80 else None)
        if target and target in codes and target != c:
            ops.append({"op": "MERGE", "from": c, "into": target,
                        "evidence": f"below floor ({n} < {a.floor} items), "
                                    f"{'co-assigned ' + str(best_co[0]) + 'x' if best_co[0] >= 3 else f'cos {s:.2f}'}"})
            reasons["below_floor"] += 1

    # 3. splits are proposed only: applying one needs new labels
    props = []
    for c, n in counts.items():
        if c not in codes:
            continue
        if n > ceil_n:
            props.append({"op": "SPLIT", "code": c, "reason": f"above ceiling: {n} > {ceil_n:.0f}"})
        elif coh.get(c, 1.0) < a.coherence and n >= a.floor:
            props.append({"op": "SPLIT_OR_EDIT_DEF", "code": c,
                          "reason": f"coherence {coh[c]:.2f} < {a.coherence} on {n} items"})

    # apply merges through the alias chain, longest-first so chains settle
    applied = 0
    for o in ops:
        f_, t_ = resolve(alias, o["from"]), resolve(alias, o["into"])
        if f_ == t_ or f_ not in codes or t_ not in codes:
            continue
        alias[f_] = t_
        applied += 1
    live = [c for c in codes if resolve(alias, c) == c]
    print(f"\nmerge operations proposed {len(ops)}, applied {applied} "
          f"({dict(reasons)})")
    print(f"split proposals (not applied): {len(props)}")
    print(f"codes: {len(codes):,} -> {len(live):,} live")

    moved = 0
    for r in rows:
        for x in r.get("assigned", []):
            t = resolve(alias, x["code"])
            if t != x["code"]:
                x["code"] = t; moved += 1
    per = Counter(x["code"] for r in rows if "error" not in r for x in r["assigned"])
    sz = sorted(per.values(), reverse=True)
    print(f"item labels migrated: {moved:,}")
    if sz:
        print(f"items per code now: max {sz[0]}, median {sz[len(sz)//2]}, used {len(per):,}")
        print(f"  below floor: {sum(1 for s in sz if s < a.floor):,} "
              f"({sum(1 for s in sz if s < a.floor)/len(sz):.0%} of used)")

    if a.dry_run:
        print("\nDRY RUN: nothing written"); return
    fz["codes"] = {c: v for c, v in codes.items() if c in set(live)}
    fz["version"] = "v3"
    fz["step7"] = {"source": srcname, "merges_applied": applied,
                   "merge_reasons": dict(reasons), "split_proposals": props,
                   "alias": alias, "floor": a.floor, "ceiling_items": ceil_n,
                   "coherence_threshold": a.coherence}
    (P / "codebook_v3_audited.json").write_text(json.dumps(fz, indent=1))
    tmp = P / "item_labels.jsonl.tmp"
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, P / "item_labels.jsonl")
    print("\nwrote codebook_v3_audited.json, co_assignment.json; item_labels.jsonl migrated")


if __name__ == "__main__":
    main()
