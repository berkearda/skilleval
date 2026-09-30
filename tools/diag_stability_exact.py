#!/usr/bin/env python3
"""Why the stability re-run reproduces the original skill set for only 47% of questions.

Step 6's stability check (validation_stability_v2_amended_b150.json) re-labelled 1,000
questions and matched Step 4's skill set exactly for 464 of the 994 it compared. Its
design varies one thing, the order of the candidate skills. The code varies three:
  order       Step 4 lists candidates by similarity; the re-run shuffles them
  prompt      the re-run's prompt drops Step 4's "do not pad", its definition of a
              skill as the smallest operation needed, the per-skill confidence and
              reason, and the UNASSIGNABLE option
  candidates  the re-run retrieves against the amended codebook, whose 12 added
              skills change the top-12 list for 122 of the 994 questions
and nothing measures how often Step 4 reproduces itself with nothing changed.

This separates the causes on a random sample of the same questions. Every condition
shows the candidate list Step 4 stored, so the candidate change is removed by
construction, and every condition uses the same model at temperature 0:
  A  Step 4 prompt, Step 4 order         repeatability: the noise floor
  B  Step 4 prompt, shuffled order       A to B is the effect of order
  C  re-run prompt, shuffled order       B to C is the effect of the prompt
The shuffle is Step 6's own, random.Random(4242 + item_idx). On questions whose
candidate list did not change, C repeats the stored re-run as it was run, so its
agreement with the stored re-run is a replication check on this script.

Vectors are read with codeemb.read_cache and checked against the definition text
here, not with codeemb.load, which rewrites the cache file on every call.

`verify_splits` does not apply: there is no train/test split, only a sample of
labelled questions, so split_info records that rather than inventing one.

    python3 tools/diag_stability_exact.py --smoke
    python3 tools/diag_stability_exact.py
"""
import argparse, ast, inspect, json, random, sys, textwrap, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.gemini import Gemini, GeminiError, BULK
from tools.textclip import clip
from tools.metrics import jaccard
from tools.codeemb import read_cache, code_text, text_hash
from tools import step4_relabel as s4          # Step 4's own prompt and caps, not copies
from tools import step6_validate as s6
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment

P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
WORKERS = 16
CONDS = {"A": ("step4", False), "B": ("step4", True), "C": ("rerun", True)}

# The re-run prompt is written inline in step6_validate.stability(), so it cannot be
# imported. check_rerun_prompt() fails if this copy stops matching the source.
S6_SYS = ("You assign cognitive skills to test questions from a fixed codebook. You judge "
          "by the operation the solver performs, never by the subject matter.")
S6_ASK = ("Choose the skills a solver must actually perform, at most 3. Choose one if one "
          "is enough. Return JSON only: {\"assigned\": [{\"code\": \"c_0123\"}]}")


def check_rerun_prompt():
    tree = ast.parse(textwrap.dedent(inspect.getsource(s6.stability)))
    consts = [n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert S6_SYS in consts, "re-run system prompt no longer matches step6_validate.stability"
    assert any(c.endswith(S6_ASK) for c in consts), \
        "re-run instruction no longer matches step6_validate.stability"


def lines(cand, codes):
    return "\n".join(f"{c} | {codes[c]['name']} | {codes[c]['definition']}"
                     for c in cand if c in codes)


def step4_filter(obj, cand):
    # Step 4 keeps dict entries naming a shown candidate, capped at MAX_SKILLS
    raw = [a for a in (obj.get("assigned") or []) if isinstance(a, dict)]
    return sorted({a.get("code") for a in [a for a in raw if a.get("code") in cand][:s4.MAX_SKILLS]})


def rerun_filter(obj, codes):
    # Step 6 keeps dicts or bare ids naming any code in the codebook, uncapped
    out = set()
    for x in (obj.get("assigned") or []):
        cid = x.get("code") if isinstance(x, dict) else x
        if isinstance(cid, str) and cid in codes:
            out.add(cid)
    return sorted(out)


def category(o, m):
    o, m = set(o), set(m)
    if o == m:
        return "exact"
    if not m:
        return "empty"
    if o < m:
        return "added"
    if m < o:
        return "dropped"
    return "swapped" if o & m else "no_overlap"


def summarize(rows, orig, key):
    rows = [r for r in rows if "error" not in r]
    n = len(rows)
    if not n:
        return {"n": 0}
    cats = Counter(category(orig[r["item_idx"]], r[key]) for r in rows)
    return {"n": n,
            "exact": cats["exact"] / n,
            "mean_jaccard": sum(jaccard(set(orig[r["item_idx"]]), set(r[key])) for r in rows) / n,
            "mean_size": sum(len(r[key]) for r in rows) / n,
            "orig_mean_size": sum(len(orig[r["item_idx"]]) for r in rows) / n,
            "categories": {k: cats[k] / n for k in
                           ("exact", "added", "dropped", "swapped", "no_overlap", "empty")}}


def unchanged_candidates(items, lab, fz6):
    """Which sampled questions the stored re-run saw with Step 4's exact candidate list."""
    live = [c for c in fz6["codes"] if c not in fz6.get("alias", {})]
    ids, vecs, hashes = read_cache(P / "code_def_emb_b150.npz")
    at = {c: k for k, c in enumerate(ids)}
    for c in live:
        assert c in at and hashes[at[c]] == text_hash(code_text(fz6, c)), f"stale vector for {c}"
    C = np.stack([vecs[at[c]] for c in live])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    iz = np.load(P / "item_emb_gemini.npz", allow_pickle=True)
    pos = {int(x): k for k, x in enumerate(list(iz["idx"]))}
    V = iz["vecs"] / np.linalg.norm(iz["vecs"], axis=1, keepdims=True)
    return {i: [live[j] for j in np.argsort(-(V[pos[i]] @ C.T))[:s4.TOPK]] == lab[i]["candidates"]
            for i in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="questions sampled from the stored re-run")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="cdm_exploration/experiments/diag_stability_exact.json")
    a = ap.parse_args()
    seed_everything(42)
    check_rerun_prompt()
    assert (s4.TOPK, s4.MAX_SKILLS, s4.QCHARS) == (12, 3, 4000), "Step 4 settings changed"

    codes4 = json.loads((P / "codebook_v1_frozen_b150.json").read_text())["codes"]   # Step 4's
    fz6 = json.loads((P / "codebook_v2_amended_b150.json").read_text())              # Step 6's
    codes6 = fz6["codes"]
    stored = json.loads((P / "validation_stability_v2_amended_b150.json").read_text())
    assert stored["model"] == BULK and stored["seed"] == 4242 and not stored["cross_model"]
    lab = {}
    for line in (P / "item_labels_b150.jsonl").open():
        if line.strip():
            r = json.loads(line)
            lab[r["item_idx"]] = r
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    pool = sorted(r["item_idx"] for r in stored["results"] if "error" not in r)
    items = random.Random(42).sample(pool, 5 if a.smoke else min(a.n, len(pool)))
    for i in items:
        for c in lab[i]["candidates"]:
            assert codes4[c]["definition"] == codes6[c]["definition"] and \
                   codes4[c]["name"] == codes6[c]["name"], f"definition of {c} differs"
    orig = {i: sorted({x["code"] for x in lab[i]["assigned"]}) for i in items}
    rerun = {r["item_idx"]: r["rerun"] for r in stored["results"] if "error" not in r}
    same_cand = unchanged_candidates(items, lab, fz6)
    print(f"{len(items)} questions, {sum(same_cand.values())} with an unchanged candidate list; "
          f"conditions A B C, model {BULK}, temperature 0")

    g = Gemini()

    def one(job):
        cond, i = job
        kind, shuffled = CONDS[cond]
        cand = list(lab[i]["candidates"])
        if shuffled:
            random.Random(4242 + i).shuffle(cand)          # Step 6's shuffle
        q = clip(txt[i], 4000)
        if kind == "step4":
            system, user, mo = s4.SYS, s4.USER.format(question=q, codes=lines(cand, codes4),
                                                      maxk=s4.MAX_SKILLS), 1600
        else:
            system, mo = S6_SYS, 1200
            user = f"QUESTION:\n{q}\n\nCANDIDATE SKILLS:\n{lines(cand, codes6)}\n\n" + S6_ASK
        try:
            obj = g.json_obj(system, user, model=BULK, max_out=mo)
        except GeminiError as e:
            return {"cond": cond, "item_idx": i, "error": str(e)[:160]}
        return {"cond": cond, "item_idx": i, "shown": cand,
                "step4_set": step4_filter(obj, set(lab[i]["candidates"])),
                "rerun_set": rerun_filter(obj, codes6),
                "raw": obj.get("assigned")}

    jobs = [(c, i) for i in items for c in CONDS]          # interleaved, so drift hits all three
    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(one, jobs), 1):
            res.append(r)
            if k % 150 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} | {g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)
    by = {c: [r for r in res if r["cond"] == c] for c in CONDS}
    n_err = sum(1 for r in res if "error" in r)

    summary = {"A_step4_prompt_step4_order": summarize(by["A"], orig, "step4_set"),
               "B_step4_prompt_shuffled": summarize(by["B"], orig, "step4_set"),
               "C_rerun_prompt_shuffled": summarize(by["C"], orig, "rerun_set"),
               "C_with_step4_filter": summarize(by["C"], orig, "step4_set"),
               "stored_rerun_same_questions": summarize(
                   [{"item_idx": i, "rerun_set": rerun[i]} for i in items], orig, "rerun_set")}
    okC = [r for r in by["C"] if "error" not in r and same_cand[r["item_idx"]]]
    summary["replication_C_vs_stored_rerun"] = {
        "n_unchanged_candidates": len(okC),
        "identical_sets": sum(r["rerun_set"] == rerun[r["item_idx"]] for r in okC) / max(1, len(okC))}
    okA = {r["item_idx"]: r for r in by["A"] if "error" not in r}
    for cond in ("B", "C"):
        key = "step4_set" if cond == "B" else "rerun_set"
        pair = [(okA[r["item_idx"]]["step4_set"], r[key]) for r in by[cond]
                if "error" not in r and r["item_idx"] in okA]
        summary[f"A_vs_{cond}_identical"] = sum(x == y for x, y in pair) / max(1, len(pair))
    for cond, key in (("A", "step4_set"), ("B", "step4_set"), ("C", "rerun_set")):
        for label, test in (("low_conf", lambda i: min(x.get("confidence") or 0 for x in lab[i]["assigned"]) < 0.9),
                            ("high_conf", lambda i: min(x.get("confidence") or 0 for x in lab[i]["assigned"]) >= 0.9)):
            summary[f"{cond}_{label}"] = summarize([r for r in by[cond] if test(r["item_idx"])], orig, key)

    print(json.dumps(summary, indent=1))
    print(f"errors {n_err} | usage: {g.report()}")
    if a.smoke:
        for r in res:
            if "error" not in r:
                print(r["cond"], r["item_idx"], "orig", orig[r["item_idx"]], "step4", r["step4_set"],
                      "rerun", r["rerun_set"])
        return
    out = REPO / a.out
    out.write_text(json.dumps({"model": BULK, "temperature": 0.0, "seed": 42, "n": len(items),
                               "summary": summary, "results": res}, indent=1))
    print(f"wrote {a.out}")
    log_experiment(
        name="diag_stability_exact",
        config={"model": BULK, "temperature": 0.0, "seed": 42, "n": len(items),
                "shuffle": "random.Random(4242 + item_idx)",
                "codebook_step4": "codebook_v1_frozen_b150.json",
                "codebook_rerun": "codebook_v2_amended_b150.json",
                "labels": "item_labels_b150.jsonl",
                "stored": "validation_stability_v2_amended_b150.json"},
        results={"A_exact": summary["A_step4_prompt_step4_order"]["exact"],
                 "B_exact": summary["B_step4_prompt_shuffled"]["exact"],
                 "C_exact": summary["C_rerun_prompt_shuffled"]["exact"],
                 "stored_exact_same_questions": summary["stored_rerun_same_questions"]["exact"],
                 "A_mean_size": summary["A_step4_prompt_step4_order"]["mean_size"],
                 "B_mean_size": summary["B_step4_prompt_shuffled"]["mean_size"],
                 "C_mean_size": summary["C_rerun_prompt_shuffled"]["mean_size"],
                 "replication_identical": summary["replication_C_vs_stored_rerun"]["identical_sets"],
                 "errors": n_err},
        split_info={"note": "no train/test split applies; the unit is a random sample of "
                            "questions from the stored stability re-run",
                    "n_items": len(items)},
        verified=(n_err == 0),
    )


if __name__ == "__main__":
    main()
