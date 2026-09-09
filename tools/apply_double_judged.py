#!/usr/bin/env python3
"""Apply only the merges two independent judges both called the same skill.

Dedupe pass 2 (the project log 2026-09-09) merged 149 pairs on one judge's word and
measured worse: the duplicate rate held at 27-28% and stability fell 9 points. So
this does the opposite. It takes the intersection of two verdict sets produced by
separate runs against separate nominations, which is 11 pairs, and applies
nothing else.

Five of the 11 were previously refused because merging would breach the Step 0
ceiling of 5%. Berke raised that to 10% on 2026-09-09. 10% is the smallest value
that admits all 11 (the largest union is 930 items, 9.8%); 12.5% admits no more.

    python3 tools/apply_double_judged.py --dry-run
"""
import argparse, json, shutil, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.metrics import apply_merges, resolve, row_codes

P = REPO / "cdm_exploration/experiments/pipeline_v7"


def double_judged(dedupe, distinct):
    """Pairs both verdict files independently call a merge."""
    a = {tuple(sorted(r["pair"])) for r in dedupe if r.get("merge")}
    b = {tuple(sorted(r["pair"])) for r in distinct if r.get("merge")}
    return sorted(a & b)


def rejected_by_either(dedupe, distinct):
    """Any pair either judge explicitly kept apart stays apart."""
    out = set()
    for src in (dedupe, distinct):
        for r in src:
            if "merge" in r and not r["merge"]:
                out.add(tuple(sorted(r["pair"])))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default="codebook_v6_definitions.json")
    ap.add_argument("--labels", default="item_labels_before_codebook_v7_deduped.jsonl")
    ap.add_argument("--out", default="codebook_v9_double_judged.json")
    ap.add_argument("--ceiling", type=float, default=0.10,
                    help="Step 0 ceiling as a fraction of labelled items. Raised from "
                         "0.05 to 0.10 by Berke on 2026-09-09; see the project log.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    if not a.no_gate:
        from tools.gate import require_tests_pass
        require_tests_pass()

    fz = json.loads((P / a.codebook).read_text())
    codes = fz["codes"]
    alias0 = fz.get("alias", {})
    rows = [json.loads(l) for l in (P / a.labels).open() if l.strip()]
    ok = [r for r in rows if "error" not in r]
    sizes = Counter(c for r in ok for c in row_codes(r, alias0) if c in codes)
    live = [c for c in codes if c not in alias0]

    dedupe = json.loads((P / "validation_dedupe.json").read_text())["results"]
    distinct = json.loads((P / "validation_distinctness.json").read_text())["results"]
    pairs = [p for p in double_judged(dedupe, distinct)
             if p[0] in codes and p[1] in codes]
    ceil_n = a.ceiling * len(ok)
    print(f"  codebook: {a.codebook}  labels: {a.labels}")
    print(f"  ceiling: {a.ceiling:.0%} of {len(ok):,} labelled items = {ceil_n:,.0f}")
    print(f"  pairs both judges independently called the same skill: {len(pairs)}")

    alias, st = apply_merges(pairs, rejected_by_either(dedupe, distinct),
                             {c: sizes.get(c, 0) for c in live}, ceil_n, order="smallest")
    kept = [c for c in live if resolve(alias, c) == c]
    print(f"  applied {st['applied']}, refused for the ceiling {st['refused_ceiling']}, "
          f"blocked {st['blocked_rejected']}")
    print(f"  skills: {len(live)} -> {len(kept)}")

    for r in rows:
        for x in r.get("assigned", []):
            x["code"] = resolve(alias, x["code"])
    sz2 = Counter(c for r in ok for c in row_codes(r) if c in set(kept))
    v = sorted(sz2.values())
    print(f"  largest skill: {v[-1]} items ({v[-1]/len(ok):.1%}), median {v[len(v)//2]}")
    print(f"  below the floor of 20: {sum(1 for x in v if x < 20)} of {len(v)}")

    if a.dry_run:
        print("\nDRY RUN: nothing written")
        return
    fz["codes"] = {c: dict(codes[c]) for c in kept}
    fz["alias"] = {}
    fz["ceiling_change"] = {"from": 0.05, "to": a.ceiling, "decided_by": "Berke",
                            "date": "2026-09-09", "pairs_applied": st["applied"]}
    (P / a.out).write_text(json.dumps(fz, indent=1))
    lf = P / f"item_labels_{Path(a.out).stem}.jsonl"
    with lf.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out} ({len(kept)} skills) and {lf.name}")


if __name__ == "__main__":
    main()
