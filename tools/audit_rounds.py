#!/usr/bin/env python3
"""Run extra consolidation rounds on the finished codebook, before it is frozen.

Berke's argument, 2026-09-09: merging belongs in codebook creation, not bolted on
after questions have been assigned. The evidence agrees. Step 2's audit is
designed to merge and was simply starved: it emitted exactly its cap in all 31
rounds, 465 operations against 1,727 codes created. Every bug this project hit
today came from the late merging instead: a stale embedding cache, a label file
overwritten in place, definitions left describing questions they no longer hold,
and a second pass that made every measure worse.

This tests whether the same work done at the right stage removes the duplicates
without any of that. It runs the Step 2 audit repeatedly on codebook_final,
where nothing is assigned yet, so there are no labels to migrate and no
definitions to go stale.

STOPPING RULE, fixed before the first run and not to be changed after seeing
output: at most MAX_ROUNDS rounds, stopping early when a round applies fewer
than STOP_MERGES merges. Declared here because the project's failure mode has
been me deciding when to stop after seeing the number.

    python3 tools/audit_rounds.py --smoke
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.gemini import Gemini
from tools.step2_codebook import Codebook, run_audit, P
from tools.codeemb import load as load_code_vecs, normed

MAX_ROUNDS = 6
STOP_MERGES = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final", help="codebook_<tag>.json to consolidate")
    ap.add_argument("--out", default="consolidated")
    ap.add_argument("--cap", type=int, default=150,
                    help="operations per round. The doc says 15; that bound was "
                         "hit in all 31 audits of the original run.")
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    if not a.no_gate:
        from tools.gate import require_tests_pass
        require_tests_pass()

    cb = Codebook()
    cb.load(a.tag)
    live0 = [c for c in cb.codes if c not in cb.alias]
    print(f"  codebook_{a.tag}.json: {len(live0):,} live codes, "
          f"{len(cb.assign):,} raw labels assigned")
    print(f"  cap {a.cap} ops/round, at most {a.rounds} rounds, "
          f"stop when a round applies < {STOP_MERGES} merges")

    g = Gemini()
    fz = {"codes": cb.codes}
    V = load_code_vecs(P / "code_def_emb.npz", fz, live0, g=g)
    code_vec = {c: v for c, v in zip(live0, normed(V))}

    t0, hist = time.time(), []
    for r in range(1, (1 if a.smoke else a.rounds) + 1):
        live = [c for c in cb.codes if c not in cb.alias]
        churn, applied = run_audit(g, cb, code_vec, bidx=1000 + r,
                                   created_since=0, cap=a.cap)
        n_merge = applied.get("MERGE", 0)
        after = len([c for c in cb.codes if c not in cb.alias])
        hist.append({"round": r, "live_before": len(live), "live_after": after,
                     "ops": dict(applied), "churn": churn})
        print(f"  round {r}: {len(live):,} -> {after:,} codes | "
              f"{dict(applied)} | churn {churn if churn is None else f'{churn:.3f}'} "
              f"| {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)
        for cid in list(code_vec):
            if cid in cb.alias:
                code_vec.pop(cid, None)
        if churn is None:
            print("  audit failed; stopping rather than counting it as no change")
            break
        if n_merge < STOP_MERGES:
            print(f"  stopping: {n_merge} merges is below the declared threshold of {STOP_MERGES}")
            break

    live = [c for c in cb.codes if c not in cb.alias]
    print(f"\n{len(live0):,} -> {len(live):,} live codes "
          f"({len(live0)-len(live):,} merged away in {len(hist)} rounds)")
    if a.smoke:
        print("SMOKE: nothing written")
        return
    cb.save(a.out)
    (P / f"audit_rounds_{a.out}.json").write_text(json.dumps(
        {"tag": a.tag, "cap": a.cap, "stop_merges": STOP_MERGES,
         "live_before": len(live0), "live_after": len(live), "rounds": hist}, indent=1))
    print(f"wrote codebook_{a.out}.json and audit_rounds_{a.out}.json")


if __name__ == "__main__":
    main()
