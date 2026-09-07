#!/usr/bin/env python3
"""Step 8: repair definitions that have drifted from the questions they hold.

Why this step exists. A definition is written once, at the moment a code is
created, from a handful of raw label strings and no questions at all. Across all
31 Step 2 audits there were 408 MERGEs, 58 RENAMEs and exactly 1 EDIT_DEF, and
of the 65 codes that grew past 50 raw labels, one ever had its definition
revised. Merging makes it worse: the surviving code keeps its own definition
while absorbing another's items, which is how "counting geometric elements",
defined as counting vertices in a polygon, ended up holding "how many objects do
I have?".

The circularity this has to avoid. Coherence asks a judge whether a code's items
require the operation its definition names. Rewriting the definition FROM those
same items and then re-measuring would raise the score by construction and prove
nothing. So each code's items are split in two by a fixed seed:

    REPAIR half   shown to the writer, which rewrites the definition
    HOLDOUT half  never shown, and the only half coherence is scored on

The holdout ids are written into the codebook so Step 6 can score on them alone.
A gain measured on the holdout is a real gain; a gain on the repair half is
tautological.

Blinding: the writer is CODEBOOK, the coherence judge stays JUDGE. What matters
is that the validator is not the writer, which still holds.

    python3 tools/step8_definitions.py --smoke
    python3 tools/step8_definitions.py [--min-items 4] [--model bulk]
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, BULK, CODEBOOK

REPO = Path(__file__).resolve().parent.parent
P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
SEED = 20260907
WORKERS = 12

SYS = ("You write definitions for a codebook of cognitive skills. A skill is the smallest "
       "named mental operation a solver must perform to answer a question correctly. You "
       "describe the operation the solver performs, never the subject matter or cover story. "
       "You describe what the questions in front of you actually require, not what the "
       "existing name suggests they ought to require.")

USER = """A code in our codebook currently reads:

  name:       {name}
  definition: {definition}

Here are {n} questions that were actually assigned to it:

{questions}

Rewrite the definition so it describes what these questions really require. Watch the
verb in particular: a definition that says "creating" when the questions ask the solver
to recognise, or "unique" when they ask for a plain count, is the failure being fixed.

If the questions genuinely fall into two different operations, say so instead of
stretching one definition to cover both.

Return JSON only:
{{"name": "<lowercase verb phrase, 3-8 words>",
  "definition": "<one sentence: what the solver must do>",
  "include": ["<when this code applies>"],
  "exclude": ["<when it does not>"],
  "changed": true|false,
  "two_operations": true|false,
  "note": "<one clause on what drifted, or null>"}}"""


def load():
    fz = json.loads((P / "codebook_v3_audited.json").read_text())
    rows = [json.loads(l) for l in (P / "item_labels.jsonl").open()]
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    by = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        for x in r.get("assigned", []):
            by[x["code"]].append(r["item_idx"])
    return fz, by, txt


def split_items(cid, items):
    """Deterministic per-code split. The holdout is never shown to the writer."""
    items = sorted(items)
    random.Random(f"{SEED}:{cid}").shuffle(items)
    half = max(1, len(items) // 2)
    return items[:half], items[half:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-items", type=int, default=4,
                    help="a code with fewer items than this has too little evidence to rewrite from")
    ap.add_argument("--show", type=int, default=8, help="repair-half questions shown to the writer")
    ap.add_argument("--model", choices=["codebook", "bulk"], default="codebook")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    if not a.no_gate:
        from tools.gate import require_tests_pass
        require_tests_pass()

    model = CODEBOOK if a.model == "codebook" else BULK
    fz, by, txt = load()
    codes = fz["codes"]
    targets = [c for c in codes if len(by.get(c, [])) >= a.min_items]
    if a.smoke:
        targets = targets[:4]
    print(f"step 8: {len(targets):,} of {len(codes):,} codes have >= {a.min_items} items, model {model}")
    print(f"  codes with too few items to rewrite from: {len(codes)-len(targets):,} (left unchanged)")

    g = Gemini()
    splits = {c: split_items(c, by[c]) for c in targets}

    def one(c):
        repair, _hold = splits[c]
        shown = repair[:a.show]
        qs = "\n".join(f"{i+1}. {txt[j][:700]}" for i, j in enumerate(shown))
        try:
            obj = g.json_obj(SYS, USER.format(name=codes[c]["name"], definition=codes[c]["definition"],
                                              n=len(shown), questions=qs),
                             model=model, max_out=3000)
        except GeminiError as e:
            return {"code": c, "error": str(e)[:120]}
        if not obj.get("definition") or not obj.get("name"):
            return {"code": c, "error": "writer returned no definition"}
        return {"code": c, "name": str(obj["name"]).strip().lower(),
                "definition": str(obj["definition"]).strip(),
                "include": obj.get("include") or [], "exclude": obj.get("exclude") or [],
                "changed": bool(obj.get("changed")), "two_operations": bool(obj.get("two_operations")),
                "note": obj.get("note"), "shown": len(shown)}

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(one, targets), 1):
            out.append(r)
            if k % 50 == 0 or k == len(targets):
                print(f"  {k:,}/{len(targets):,} | {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)

    ok = [r for r in out if "error" not in r]
    changed = [r for r in ok if r["changed"]]
    two = [r for r in ok if r["two_operations"]]
    renamed = [r for r in ok if r["name"] != codes[r["code"]]["name"]]
    print(f"\nrewritten: {len(ok):,} ({len(out)-len(ok)} errors)")
    print(f"  writer reports the definition changed: {len(changed):,} ({len(changed)/max(1,len(ok)):.0%})")
    print(f"  name also changed: {len(renamed):,}")
    print(f"  flagged as two operations (split candidates): {len(two):,}")
    if a.smoke:
        for r in ok[:3]:
            print(f"\n  {r['code']}\n    was: {codes[r['code']]['definition']}\n    now: {r['definition']}")
        print("\nSMOKE: nothing written")
        return

    for r in ok:
        c = r["code"]
        codes[c]["definition_before"] = codes[c]["definition"]
        codes[c]["name_before"] = codes[c]["name"]
        codes[c]["name"] = r["name"]
        codes[c]["definition"] = r["definition"]
        codes[c]["include"] = r["include"] or codes[c].get("include", [])
        codes[c]["exclude"] = r["exclude"] or codes[c].get("exclude", [])
        codes[c]["two_operations"] = r["two_operations"]
    for c in codes:
        rep, hold = splits.get(c, ([], sorted(by.get(c, []))))
        codes[c]["repair_items"] = rep
        codes[c]["holdout_items"] = hold      # Step 6 scores coherence on these only
    fz["version"] = "v4"
    fz["step8"] = {"model": model, "seed": SEED, "rewritten": len(ok),
                   "errors": len(out) - len(ok), "changed": len(changed),
                   "renamed": len(renamed), "two_operations": [r["code"] for r in two],
                   "min_items": a.min_items, "shown_per_code": a.show}
    (P / "codebook_v4_definitions.json").write_text(json.dumps(fz, indent=1))
    print("\nwrote codebook_v4_definitions.json (definition_before kept on every code)")
    print("Next: score coherence on the HOLDOUT half only, which the writer never saw.")


if __name__ == "__main__":
    main()
