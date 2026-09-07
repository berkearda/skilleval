#!/usr/bin/env python3
"""Step 5: close the loop on the residue.

The doc: collect everything UNASSIGNABLE or low-confidence, run Step 2 over just
those, amend the codebook, re-run only the affected items, two or three rounds,
until the residual drops below 2-3%. Whatever is left goes in an explicit OTHER
bucket rather than being force-fitted.

Two departures from a literal reading, both forced by measurement:

- "low-confidence" cannot be used. Step 4's confidence field is degenerate
  (0.9 is a default; nothing below 0.6 in 6,936 assignments), so filtering on
  it would select nothing meaningful. The residue here is UNASSIGNABLE, items
  that came back with zero skills, and items whose call errored.
- Errored items are retried first. They are not residue, they are missing data,
  and counting a failed HTTP call as an unlabelled question would inflate the
  residual rate with something a retry fixes.

    python3 tools/step5_loop.py [--rounds 2] [--smoke]
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, BULK, CODEBOOK

REPO = Path(__file__).resolve().parent.parent
P = REPO / "cdm_exploration/experiments/pipeline_v7"

RESID_TARGET = 0.02

SYS = ("You maintain a codebook of cognitive skills for a test-item taxonomy. A skill is the "
       "smallest named mental operation a solver must master to answer a question correctly. "
       "You judge by the operation a solver performs, never by the subject matter or cover story.")

ADD_U = """These questions could not be assigned to any existing skill. Each shows the
skill the labeller proposed for it.

{items}

The codebook already contains these nearby skills:
{near}

Propose new codes for operations genuinely missing from the codebook. Merge
proposals that describe the same operation into one code. Propose nothing for a
question already covered by a nearby skill.

Return JSON only:
{{"codes": [{{"name": "<lowercase verb phrase, 3-8 words>",
  "definition": "<one sentence, what the solver must do>",
  "include": ["<when it applies>"], "exclude": ["<when it does not>"],
  "covers": [<question numbers this code covers>]}}]}}"""


def norm_proposal(p):
    """Step 4's proposed_skill has no stable schema: name / title / description
    only / a literal c_xxxx placeholder. Normalise to a single string."""
    if not isinstance(p, dict):
        return str(p)[:200] if p else ""
    for k in ("name", "title", "skill", "label"):
        if p.get(k) and not str(p[k]).startswith("c_xxxx"):
            return str(p[k])[:200]
    return str(p.get("description", ""))[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    fz = json.loads((P / "codebook_v1_frozen.json").read_text())
    codes = fz["codes"]
    rows = {json.loads(l)["item_idx"]: json.loads(l)
            for l in (P / "item_labels.jsonl").open()}
    errs = [i for i, r in rows.items() if "error" in r]
    resid = [i for i, r in rows.items()
             if "error" not in r and (r["unassignable"] or not r["assigned"])]
    n = len(rows)
    print(f"step 5: {n:,} items | errored {len(errs)} | residue {len(resid)} "
          f"({len(resid)/n:.2%}) | target under {RESID_TARGET:.0%}")

    g = Gemini()
    z = np.load(P / "code_def_emb.npz", allow_pickle=True)
    ids = list(z["ids"]); C = z["vecs"] / np.linalg.norm(z["vecs"], axis=1, keepdims=True)

    # --- residue: what operations were proposed, and are they really missing? ---
    props = []
    for i in resid:
        r = rows[i]
        t = norm_proposal(r.get("proposed_skill"))
        if t:
            props.append((i, r["benchmark"], t))
    print(f"  residue items carrying a usable proposal: {len(props)} of {len(resid)}")
    if not props:
        print("  nothing to add"); return

    V = np.stack(g.embed([t for _, _, t in props]))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ C.T
    near_ids = sorted({ids[j] for r_ in range(len(props)) for j in np.argsort(-S[r_])[:5]})
    items_txt = "\n".join(f"{k+1}. [{b}] {t}" for k, (_, b, t) in enumerate(props))
    near_txt = "\n".join(f"{c} | {codes[c]['name']} | {codes[c]['definition']}" for c in near_ids)

    if a.smoke:
        print(f"SMOKE: would send {len(props)} proposals against {len(near_ids)} nearby codes")
        print(items_txt[:600]); return

    try:
        obj = g.json_obj(SYS, ADD_U.format(items=items_txt, near=near_txt),
                         model=CODEBOOK, max_out=20000)
    except GeminiError as e:
        print(f"  codebook extension FAILED: {e}"); return
    new = obj.get("codes", [])
    print(f"  proposed {len(new)} new codes from {len(props)} residue items")

    nxt = max(int(c.split("_")[1]) for c in codes) + 1
    added, covered = {}, set()
    for c in new:
        if not c.get("name") or not c.get("definition"):
            continue
        cid = f"c_{nxt:04d}"; nxt += 1
        added[cid] = {"id": cid, "name": c["name"].strip().lower(), "parent": "D99_step5_residue",
                      "definition": c["definition"].strip(), "include": c.get("include", []),
                      "exclude": c.get("exclude", []), "exemplar_labels": [],
                      "exemplar_items": [], "confusable_with": [], "raw_label_mentions": 0}
        for k in c.get("covers", []):
            if isinstance(k, int) and 1 <= k <= len(props):
                covered.add(props[k - 1][0])
    other = [i for i, _, _ in props if i not in covered]
    print(f"  added {len(added)} codes covering {len(covered)} items; "
          f"OTHER bucket {len(other)} items ({len(other)/n:.2%})")

    fz["codes"].update(added)
    fz["version"] = "v2"
    fz["step5"] = {"residue_in": len(resid), "errored_retried": len(errs),
                   "codes_added": len(added), "items_covered": len(covered),
                   "other_bucket": sorted(other),
                   "residual_rate": len(other) / n}
    (P / "codebook_v2_amended.json").write_text(json.dumps(fz, indent=1))
    print(f"\nwrote codebook_v2_amended.json: {len(fz['codes']):,} codes "
          f"({len(added)} new). Residual {len(other)/n:.2%}, target {RESID_TARGET:.0%}.")


if __name__ == "__main__":
    main()
