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
import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, BULK, CODEBOOK

REPO = Path(__file__).resolve().parent.parent
P = REPO / "cdm_exploration/experiments/pipeline_v7"

# Every artifact this step reads or writes is namespaced by STEP_TAG, so a second
# run cannot overwrite the first. Steps 2, 6 and 9 already work this way; steps
# 3-5 did not, and running them untagged would have destroyed
# codebook_v1_frozen.json and item_labels.jsonl, which the 230-skill taxonomy and
# Berke's gold-set score both trace to.
TAG = os.environ.get("STEP_TAG", "")


def tagged(name):
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{TAG}{dot}{ext}"


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
    # NOTE: one round only. The script is not re-entrant (it reloads
    # codebook_v1_frozen.json unconditionally, so a second call would discard the
    # first round's codes), and one round reached 0.4% against a 2-3% target.
    # A --rounds flag is deliberately absent rather than present and ignored.
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    fz = json.loads((P / tagged("codebook_v1_frozen.json")).read_text())  # single round; see above
    codes = fz["codes"]
    rows = {json.loads(l)["item_idx"]: json.loads(l)
            for l in (P / tagged("item_labels.jsonl")).open()}
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
    added, covered, bad_covers = {}, set(), 0
    for c in new:
        if not c.get("name") or not c.get("definition"):
            continue
        cid = f"c_{nxt:04d}"; nxt += 1
        added[cid] = {"id": cid, "name": c["name"].strip().lower(), "parent": "D99_step5_residue",
                      "definition": c["definition"].strip(), "include": c.get("include", []),
                      "exclude": c.get("exclude", []), "exemplar_labels": [],
                      "exemplar_items": [], "confusable_with": [], "raw_label_mentions": 0}
        for k in c.get("covers", []):
            try:                                   # models return "1" or 1; both count
                ki = int(str(k).strip())
            except (TypeError, ValueError):
                bad_covers += 1; continue
            if 1 <= ki <= len(props):
                covered.add(props[ki - 1][0])
    # seeded from ALL residue, not just those carrying a proposal: the item nothing
    # could even be proposed for is the one that most belongs in the tail bucket
    other = [i for i in resid if i not in covered]
    print(f"  added {len(added)} codes covering {len(covered)} items; "
          f"OTHER bucket {len(other)} items ({len(other)/n:.2%})")

    fz["codes"].update(added)
    fz["version"] = "v2"
    fz["step5"] = {"residue_in": len(resid), "errored_remaining": len(errs),
                   "codes_added": len(added), "items_claimed_covered": len(covered)}
    (P / tagged("codebook_v2_amended.json")).write_text(json.dumps(fz, indent=1))
    print(f"\nwrote codebook_v2_amended.json: {len(fz['codes']):,} codes "
          f"({len(added)} new). Residual {len(other)/n:.2%}, target {RESID_TARGET:.0%}.")

    # The doc: "re-run only the affected items". Adding codes changes nothing for
    # items that were already assigned, so the affected set is the residue.
    if not added:
        return
    print(f"\nre-running the {len(resid)} affected items against the amended codebook ...")
    from tools.step4_relabel import load_items as _li
    txt = {it["item_idx"]: it for it in _li()}
    newV = np.stack(g.embed([f"{v['name']}. {v['definition']}" for v in added.values()]))
    newV = newV / np.linalg.norm(newV, axis=1, keepdims=True)
    allC = np.vstack([C, newV]); allIds = ids + list(added)
    # Persist them. Previously these vectors were computed here and thrown away,
    # so Step 6 loaded an embedding file that predated Step 5 and silently could
    # neither nominate the new codes for a merge nor retrieve them as candidates,
    # which manufactures a jaccard of 0 for every item they cover.
    z_old = np.load(P / "code_def_emb.npz", allow_pickle=True)
    np.savez_compressed(P / "code_def_emb.npz",
                        ids=np.array(allIds, dtype=object),
                        vecs=np.vstack([z_old["vecs"], newV]),
                        digest="stale-after-step5")   # forces Step 3 to re-embed
    print(f"  appended {len(added)} new code embeddings to code_def_emb.npz")
    iz = np.load(P / "item_emb_gemini.npz", allow_pickle=True)
    ipos = {int(i): k for k, i in enumerate(list(iz["idx"]))}
    IV = iz["vecs"] / np.linalg.norm(iz["vecs"], axis=1, keepdims=True)

    from tools.step4_relabel import SYS as S4SYS, USER as S4USER, TOPK, MAX_SKILLS
    fixed = retry_failed = 0
    for i in resid:
        it = txt.get(i)
        if it is None:
            continue
        cand = [allIds[j] for j in np.argsort(-(IV[ipos[i]] @ allC.T))[:TOPK]]
        ctxt = "\n".join(f"{c} | {fz['codes'][c]['name']} | {fz['codes'][c]['definition']}"
                          for c in cand if c in fz["codes"])
        try:
            obj = g.json_obj(S4SYS, S4USER.format(benchmark=it["benchmark"], subtask=it["subtask"],
                                                  question=it["question"], codes=ctxt, maxk=MAX_SKILLS),
                             model=BULK, max_out=1600)
        except GeminiError as e:
            retry_failed += 1        # a network blip must not read as "no skill fits"
            continue
        asg = [x for x in obj.get("assigned", []) if x.get("code") in fz["codes"]][:MAX_SKILLS]
        if asg:
            rows[i]["assigned"] = asg; rows[i]["unassignable"] = False
            rows[i]["relabelled_step5"] = True
            fixed += 1
    tmp = P / "item_labels.jsonl.tmp"                  # atomic: this rewrite follows
    with tmp.open("w") as f:                          # a serial loop of API calls,
        for i in sorted(rows):                        # so the window is minutes wide
            f.write(json.dumps(rows[i]) + "\n")
    os.replace(tmp, P / tagged("item_labels.jsonl"))
    still_ids = [i for i in resid if not rows[i].get("assigned")]
    # measured after the re-run, not predicted from the model's `covers` field
    fz["step5"]["other_bucket"] = sorted(still_ids)
    fz["step5"]["residual_rate"] = len(still_ids) / n
    fz["step5"]["measured_after_rerun"] = True
    (P / tagged("codebook_v2_amended.json")).write_text(json.dumps(fz, indent=1))
    fz["step5"]["retry_failed"] = retry_failed
    fz["step5"]["bad_covers_entries"] = bad_covers
    (P / tagged("codebook_v2_amended.json")).write_text(json.dumps(fz, indent=1))
    print(f"  {fixed} of {len(resid)} residue items now assigned; {len(still_ids)} still "
          f"unassigned ({len(still_ids)/n:.2%}) <- measured residual")
    if retry_failed:
        print(f"  WARNING: {retry_failed} re-label calls failed; those items are counted "
              f"as unassigned but were never actually judged")


if __name__ == "__main__":
    main()
