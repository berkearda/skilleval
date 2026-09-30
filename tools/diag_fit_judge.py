#!/usr/bin/env python3
"""Definition fit with a judge that is not told to be strict, answers in three levels,
and sees decoy questions.

Step 6's coherence judge (COH_SYS, COH_U) is told "You are strict: partial topical
overlap is not a match" and answers yes or no. On the gold sheet 2 hand check it
accepted 32 of 42 pipeline assignments where the human accepted 40. A yes/no answer
also scores a skill the solver uses as a secondary step the same as a skill that does
not apply. And it has no decoys, so a softer prompt could only be judged by whether
its score went up.

This judge:
  - opens with the definition of a skill the labeller was given (Step 4's SYS) and
    says nothing about strictness;
  - answers "main", "supporting" or "no" for each question; fit counts main or
    supporting;
  - sees 3 questions from other skills mixed, unmarked, into every batch of up to 10,
    so a judge that agrees with everything shows up as decoy acceptance.

  validate   old and new prompt on identical batches built around gold sheet 2:
             42 pipeline assignments (40 accepted by hand) and 198 decoy skills
             (1 accepted by hand). Pass rule, fixed in the project log before the run:
             the new prompt accepts at least 85% of the 42 and at most 10% of the
             198, with at most 5% of the 240 verdicts missing.
  score      the new prompt on one skill list and the labels that match it.

`verify_splits` does not apply: there is no train/test split, only questions and their
skill labels, so split_info records that rather than inventing one.

    python3 tools/diag_fit_judge.py validate --smoke
    python3 tools/diag_fit_judge.py validate
    python3 tools/diag_fit_judge.py score --codebook codebook_v2_amended_b150.json \
        --labels item_labels_b150.jsonl --tag b150
"""
import argparse, hashlib, json, random, shutil, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.gemini import Gemini, GeminiError, JUDGE
from tools.textclip import clip
from tools.metrics import row_codes, separation
from tools.gold_score_partb import parse_sheet
from tools.step6_validate import COH_SYS, COH_U          # the old prompt, not a copy
from tools import step4_relabel as s4
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment

P = REPO / "cdm_exploration/experiments/pipeline_v7"
D = REPO / "cdm_exploration/data/cdm_ready"
E = REPO / "cdm_exploration/experiments"
WORKERS = 12
N_OWN, N_DECOY = 10, 3

SKILL_DEF = ("A skill is the smallest named mental operation a solver must perform to "
             "answer correctly. You judge by the operation the solver performs, never by "
             "the subject matter or cover story.")
assert SKILL_DEF in s4.SYS, "the labeller's definition of a skill changed; update SKILL_DEF"
NEW_SYS = "You check how a cognitive skill relates to test questions. " + SKILL_DEF
NEW_U = """SKILL: {definition}

For each question below, how does this skill relate to answering it?
- "main": the question mainly tests this skill
- "supporting": the solver uses this skill, but it is not the main thing the question tests
- "no": the solver does not need this skill

{questions}

Return JSON only: {{"verdicts": [{{"i": 1, "answer": "main"}}, ...]}}"""
LEVELS = ("main", "supporting", "no")


def sd(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def load_text():
    return {r["item_idx"]: " ".join(r["question_full_text"].split())
            for r in json.load(open(D / "item_full_text_recovered.json"))}


def load_labels(codebook, labels):
    cb = json.loads((P / codebook).read_text())
    alias, codes = cb.get("alias", {}), cb["codes"]
    by_code, items, seen = defaultdict(list), set(), set()
    for line in (P / labels).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        items.add(r["item_idx"])
        for c in sorted(row_codes(r, alias)):            # distinct codes per question
            seen.add(c)
            if c in codes:
                by_code[c].append(r["item_idx"])
    load_labels.label_codes = seen                       # every code the labels name
    return cb, codes, by_code, sorted(items)


def pick(pool, k, seed, exclude=()):
    ex = set(exclude)
    pool = [x for x in pool if x not in ex]
    return random.Random(seed).sample(pool, min(max(k, 0), len(pool)))


def ask(g, kind, definition, batch, txt):
    qs = "\n".join(f"{k+1}. {clip(txt[i], 4000)}" for k, i in enumerate(batch))
    if kind == "old":
        system, user = COH_SYS, COH_U.format(definition=definition, questions=qs)
    else:
        system, user = NEW_SYS, NEW_U.format(definition=definition, questions=qs)
    obj, last = None, None
    for _ in range(2):                  # one retry: truncated thinking is the usual failure
        try:
            obj = g.json_obj(system, user, model=JUDGE, max_out=8000)
            break
        except GeminiError as e:
            last = e
    if obj is None:
        return None, str(last)[:160]
    out = {}
    for v in (obj.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        try:
            k = int(v.get("i")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= k < len(batch):
            continue
        if kind == "old":
            m = v.get("match")          # bool("false") is True, so never bool()
            out[batch[k]] = "yes" if (m is True or (isinstance(m, str) and m.strip().lower() == "true")) else "no"
        else:
            ans = str(v.get("answer", "")).strip().lower()
            out[batch[k]] = ans if ans in LEVELS else "invalid"
    return out, None


def accepts(kind, ans):
    return ans == "yes" if kind == "old" else ans in ("main", "supporting")


def run_jobs(jobs, fn):
    res, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, r in enumerate(ex.map(fn, jobs), 1):
            res.append(r)
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} calls | {time.time()-t0:.0f}s", flush=True)
    return res


def validate(a):
    cb, codes, by_code, all_items = load_labels(a.codebook, a.labels)
    key = json.loads((REPO / a.key).read_text())
    assert key["codebook"] == a.codebook and key["labels"] == a.labels, \
        "the answer key was drawn from another skill list"
    ticks, shown, anchored = parse_sheet(REPO / a.sheet)
    txt = load_text()
    gold = []                                          # (item, skill, role, human accepted)
    for i, v in key["by_item"].items():
        assert set(v["shown"]) == shown.get(i, set()), f"sheet and key disagree on item {i}"
        for c in v["assigned"]:
            gold.append((int(i), c, "real", c in ticks.get(i, ())))
        for c in v["decoys"]:
            gold.append((int(i), c, "decoy", c in ticks.get(i, ())))
    n_real = sum(1 for x in gold if x[2] == "real")
    assert (n_real, len(gold) - n_real) == (42, 198), (n_real, len(gold) - n_real)
    gold_items = {x[0] for x in gold}
    per = defaultdict(lambda: {"real": [], "decoy": []})
    for it, c, role, _ in gold:
        per[c][role].append(it)

    batches = []
    for c in sorted(per):
        reals, decs = per[c]["real"], per[c]["decoy"]
        own_all = by_code.get(c, [])
        chunks = [decs[j:j + N_DECOY] for j in range(0, len(decs), N_DECOY)] or [[]]
        for k, chunk in enumerate(chunks):
            base = reals if k == 0 else []
            own = base + pick(own_all, min(N_OWN, len(own_all)) - len(base), sd(c, "own", k),
                              exclude=set(base) | gold_items)
            fill = pick(all_items, N_DECOY - len(chunk), sd(c, "decoy", k),
                        exclude=set(own_all) | gold_items)
            batch = own + chunk + fill
            random.Random(sd(c, "order", k)).shuffle(batch)
            role = {x: "own" for x in own}
            role.update({x: "decoy" for x in fill})
            role.update({x: "gold_real" for x in base})
            role.update({x: "gold_decoy" for x in chunk})
            batches.append({"code": c, "chunk": k, "batch": batch, "role": role})
    if a.smoke:
        batches = batches[:3]
    print(f"validate: {len(gold)} gold pairs over {len(per)} skills in {len(batches)} batches, "
          f"old and new prompt on each, judge {JUDGE}")

    g = Gemini()

    def one(job):
        kind, b = job
        out, err = ask(g, kind, codes[b["code"]]["definition"], b["batch"], txt)
        return {"kind": kind, "code": b["code"], "chunk": b["chunk"], "batch": b["batch"],
                "role": {str(x): r for x, r in b["role"].items()},
                "verdicts": {str(x): v for x, v in (out or {}).items()}, "error": err}

    res = run_jobs([(kind, b) for b in batches for kind in ("old", "new")], one)
    n_err = sum(1 for r in res if r["error"])
    ans, fill = {}, {"old": Counter(), "new": Counter()}
    for r in res:
        if r["error"]:
            continue
        for x, role in r["role"].items():
            v = r["verdicts"].get(x, "missing")
            if role.startswith("gold"):
                ans[(r["kind"], int(x), r["code"])] = v
            else:
                fill[r["kind"]][(role, accepts(r["kind"], v))] += 1

    done_pairs = [x for x in gold if any((k, x[0], x[1]) in ans for k in ("old", "new"))] \
        if a.smoke else gold
    summary = {}
    for kind in ("old", "new"):
        rows = [(it, c, role, h, ans.get((kind, it, c), "missing")) for it, c, role, h in done_pairs]
        real = [x for x in rows if x[2] == "real"]
        dec = [x for x in rows if x[2] == "decoy"]
        acc = lambda rr: sum(accepts(kind, x[4]) for x in rr)
        na = [x for x in rows if str(x[0]) not in anchored]
        own_n = fill[kind][("own", True)] + fill[kind][("own", False)]
        dec_n = fill[kind][("decoy", True)] + fill[kind][("decoy", False)]
        s = {"real_accept": acc(real), "n_real": len(real),
             "decoy_accept": acc(dec), "n_decoy": len(dec),
             "agreement_with_human": sum(accepts(kind, x[4]) == x[3] for x in rows), "n": len(rows),
             "missing_or_invalid": sum(x[4] in ("missing", "invalid") for x in rows),
             "separation": separation(acc(real), len(real), acc(dec), len(dec)),
             "human_yes_judge_no": sum(1 for x in rows if x[3] and not accepts(kind, x[4])),
             "human_no_judge_yes": sum(1 for x in rows if not x[3] and accepts(kind, x[4])),
             "filler_own_accept_rate": fill[kind][("own", True)] / own_n if own_n else None,
             "filler_decoy_accept_rate": fill[kind][("decoy", True)] / dec_n if dec_n else None,
             "without_anchored_items": {
                 "real_accept": acc([x for x in na if x[2] == "real"]),
                 "n_real": sum(1 for x in na if x[2] == "real"),
                 "decoy_accept": acc([x for x in na if x[2] == "decoy"]),
                 "n_decoy": sum(1 for x in na if x[2] == "decoy")}}
        if kind == "new":
            s["levels_real"] = dict(Counter(x[4] for x in real))
            s["levels_decoy"] = dict(Counter(x[4] for x in dec))
        summary[kind] = s
    human = {"real_accept": sum(x[3] for x in done_pairs if x[2] == "real"),
             "decoy_accept": sum(x[3] for x in done_pairs if x[2] == "decoy")}
    s = summary["new"]
    passed = (s["n_real"] and s["n_decoy"]
              and s["real_accept"] / s["n_real"] >= 0.85
              and s["decoy_accept"] / s["n_decoy"] <= 0.10
              and s["missing_or_invalid"] <= 0.05 * s["n"])
    summary["human"] = human
    summary["pass_rule"] = "new: real >= 85%, decoy <= 10%, missing <= 5% of pairs"
    summary["passed"] = bool(passed)
    print(json.dumps(summary, indent=1))
    print(f"errors {n_err} calls | usage: {g.report()}")
    disagree = [(it, c, role, h, ans.get(("new", it, c)), ans.get(("old", it, c)))
                for it, c, role, h in done_pairs if accepts("new", ans.get(("new", it, c))) != h]
    print(f"\nnew-prompt disagreements with the human ({len(disagree)}):")
    for it, c, role, h, nv, ov in disagree:
        print(f"  item {it} {role:5s} {c} {codes[c]['name']!r}: human {'yes' if h else 'no'}, "
              f"new {nv}, old {ov}")
    if a.smoke:
        return
    out = E / "diag_fit_judge_validate.json"
    if out.exists():
        raise SystemExit(f"{out.name} exists; move it aside rather than overwrite a result")
    out.write_text(json.dumps({"judge": JUDGE, "new_system": NEW_SYS, "new_user": NEW_U,
                               "summary": summary, "results": res}, indent=1))
    print(f"wrote {out.relative_to(REPO)}")
    log_experiment(
        name="diag_fit_judge_validate",
        config={"judge": JUDGE, "seed": 42, "own_per_batch": N_OWN, "decoys_per_batch": N_DECOY,
                "codebook": a.codebook, "labels": a.labels, "sheet": a.sheet, "key": a.key},
        results={"new_real_accept": s["real_accept"], "new_decoy_accept": s["decoy_accept"],
                 "old_real_accept": summary["old"]["real_accept"],
                 "old_decoy_accept": summary["old"]["decoy_accept"],
                 "n_real": s["n_real"], "n_decoy": s["n_decoy"],
                 "new_agreement": s["agreement_with_human"],
                 "old_agreement": summary["old"]["agreement_with_human"],
                 "human_real_accept": human["real_accept"],
                 "human_decoy_accept": human["decoy_accept"],
                 "passed": bool(passed), "error_calls": n_err},
        split_info={"note": "no train/test split applies; the unit is the gold sheet 2 "
                            "hand check (42 real and 198 decoy assignments)",
                    "n_pairs": len(gold)},
        verified=(n_err == 0 and s["missing_or_invalid"] == 0
                  and summary["old"]["missing_or_invalid"] == 0),
    )


def count_result(role, verdicts):
    """Per-batch counts from its roles and verdicts; anything not a level counts as missing."""
    per = Counter((r, verdicts.get(x, "missing")) for x, r in role.items())
    return {"own_n": sum(1 for r in role.values() if r == "own"),
            "own_main": per[("own", "main")], "own_supporting": per[("own", "supporting")],
            "decoy_n": sum(1 for r in role.values() if r == "decoy"),
            "decoy_accept": per[("decoy", "main")] + per[("decoy", "supporting")],
            "missing": sum(n for (_, y), n in per.items() if y not in LEVELS)}


def score_summary(res, tag, codebook, labels):
    good = [r for r in res if not r["error"] and r["own_n"]]
    n_err = sum(1 for r in res if r["error"])
    fit = [(r["own_main"] + r["own_supporting"]) / r["own_n"] for r in good]
    main = [r["own_main"] / r["own_n"] for r in good]
    own_n = sum(r["own_n"] for r in good)
    own_acc = sum(r["own_main"] + r["own_supporting"] for r in good)
    dec_n = sum(r["decoy_n"] for r in good)
    dec_acc = sum(r["decoy_accept"] for r in good)
    return {"tag": tag, "codebook": codebook, "labels": labels,
            "skills_scored": len(good), "error_calls": n_err,
            "mean_fit": sum(fit) / len(fit) if fit else None,
            "mean_main_only": sum(main) / len(main) if main else None,
            "pooled_fit": own_acc / own_n if own_n else None,
            "decoy_accept": dec_acc / dec_n if dec_n else None,
            "separation": separation(own_acc, own_n, dec_acc, dec_n),
            "skills_fit_0.2_or_less": sum(1 for f in fit if f <= 0.2),
            "questions_sent": own_n, "decoys_sent": dec_n,
            "missing_or_invalid": sum(r["missing"] for r in good)}, n_err


def reask(a):
    """Ask again only for verdicts the first run did not return, in the same batch.

    Every other verdict is kept from the first run, so re-asking cannot move a score
    except through the verdicts that were missing. The first run's file is kept beside
    the updated one, and the stored counts must reproduce before anything is asked.
    """
    if not a.tag:
        raise SystemExit("reask needs --tag")
    out = E / f"diag_fit_judge_score_{a.tag}.json"
    back = out.with_name(f"{out.stem}.before_reask.json")
    if back.exists():
        raise SystemExit(f"{back.name} exists; a re-ask was already applied")
    d = json.loads(out.read_text())
    assert d["new_system"] == NEW_SYS and d["new_user"] == NEW_U, "the prompt changed since the run"
    s0 = d["summary"]
    for r in d["results"]:
        if r["error"]:
            raise SystemExit(f"{r['code']} failed as a whole batch; re-run it rather than patch it")
        c = count_result(r["role"], r["verdicts"])
        assert all(c[k] == r[k] for k in c), f"stored counts do not reproduce for {r['code']}"
    again, _ = score_summary(d["results"], s0["tag"], s0["codebook"], s0["labels"])
    assert again == s0, "the stored summary does not reproduce from the stored verdicts"
    cb, codes, _, _ = load_labels(s0["codebook"], s0["labels"])
    txt = load_text()
    g = Gemini()
    fixed = []
    for r in d["results"]:
        gaps = [x for x in r["role"] if r["verdicts"].get(x) not in LEVELS]
        if not gaps:
            continue
        filled, tries = {}, 0
        while len(filled) < len(gaps) and tries < 3:
            tries += 1
            v, err = ask(g, "new", codes[r["code"]]["definition"], r["batch"], txt)
            for x in gaps:
                y = (v or {}).get(int(x))
                if y in LEVELS and x not in filled:
                    filled[x] = y
        r["verdicts"].update(filled)
        r.update(count_result(r["role"], r["verdicts"]))
        fixed.append({"code": r["code"], "items": gaps, "filled": filled, "calls": tries})
        print(f"  {r['code']}: filled {len(filled)} of {len(gaps)} after {tries} call(s): {filled}")
    summary, n_err = score_summary(d["results"], s0["tag"], s0["codebook"], s0["labels"])
    print(json.dumps(summary, indent=1))
    print(f"usage: {g.report()}")
    shutil.copy2(out, back)
    d["summary"] = summary
    d["reasked"] = {"rule": "only verdicts missing in the first run were asked again, in the same "
                            "batch; every other verdict is the first run's",
                    "first_run_file": back.name, "fixed": fixed}
    out.write_text(json.dumps(d, indent=1))
    print(f"wrote {out.relative_to(REPO)} (first run kept as {back.name})")
    log_experiment(
        name=f"diag_fit_judge_score_{s0['tag']}_reasked",
        config={"judge": JUDGE, "seed": 42, "codebook": s0["codebook"], "labels": s0["labels"],
                "reasked": fixed, "first_run_file": back.name},
        results={k: v for k, v in summary.items() if k not in ("tag", "codebook", "labels")},
        split_info={"note": "no train/test split applies; the unit is a skill and a sample "
                            "of its questions", "n_skills": summary["skills_scored"]},
        verified=(n_err == 0 and summary["missing_or_invalid"] == 0),
    )


def score(a):
    if not a.tag:
        raise SystemExit("score needs --tag, so results for different skill lists do not collide")
    out = E / f"diag_fit_judge_score_{a.tag}.json"
    if out.exists() and not a.smoke:
        raise SystemExit(f"{out.name} exists; move it aside rather than overwrite a result")
    cb, codes, by_code, all_items = load_labels(a.codebook, a.labels)
    live = [c for c in codes if c not in cb.get("alias", {})]
    # by_code keeps only codebook codes, so strays are counted from everything the labels name
    stray = sum(1 for c in load_labels.label_codes if c not in live)
    skills = [c for c in live if len(by_code.get(c, [])) >= 2]
    if a.smoke:
        skills = skills[:3]
    txt = load_text()
    batches = []
    for c in skills:
        own = pick(by_code[c], N_OWN, sd(c, "own"))
        dec = pick(all_items, N_DECOY, sd(c, "decoy"), exclude=set(by_code[c]))
        batch = own + dec
        random.Random(sd(c, "order")).shuffle(batch)
        role = {x: "own" for x in own}
        role.update({x: "decoy" for x in dec})
        batches.append({"code": c, "batch": batch, "role": role})
    print(f"score [{a.tag}]: {len(live)} live skills, {len(skills)} with 2+ questions, "
          f"{stray} label codes not live, judge {JUDGE}")

    g = Gemini()

    def one(b):
        v, err = ask(g, "new", codes[b["code"]]["definition"], b["batch"], txt)
        per = Counter()
        for x, role in b["role"].items():
            ansr = (v or {}).get(x, "missing")
            per[(role, ansr)] += 1
        return {"code": b["code"], "batch": b["batch"],
                "role": {str(x): r for x, r in b["role"].items()},
                "verdicts": {str(x): y for x, y in (v or {}).items()}, "error": err,
                "own_n": sum(1 for r in b["role"].values() if r == "own"),
                "own_main": per[("own", "main")], "own_supporting": per[("own", "supporting")],
                "decoy_n": sum(1 for r in b["role"].values() if r == "decoy"),
                "decoy_accept": per[("decoy", "main")] + per[("decoy", "supporting")],
                "missing": sum(n for (r, y), n in per.items() if y in ("missing", "invalid"))}

    res = run_jobs(batches, one)
    summary, n_err = score_summary(res, a.tag, a.codebook, a.labels)
    print(json.dumps(summary, indent=1))
    print(f"usage: {g.report()}")
    if a.smoke:
        return
    out.write_text(json.dumps({"judge": JUDGE, "new_system": NEW_SYS, "new_user": NEW_U,
                               "summary": summary, "results": res}, indent=1))
    print(f"wrote {out.relative_to(REPO)}")
    log_experiment(
        name=f"diag_fit_judge_score_{a.tag}",
        config={"judge": JUDGE, "seed": 42, "own_per_batch": N_OWN, "decoys_per_batch": N_DECOY,
                "codebook": a.codebook, "labels": a.labels},
        results={k: v for k, v in summary.items() if k not in ("tag", "codebook", "labels")},
        split_info={"note": "no train/test split applies; the unit is a skill and a sample "
                            "of its questions", "n_skills": len(skills)},
        verified=(n_err == 0 and summary["missing_or_invalid"] == 0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "score", "reask"])
    ap.add_argument("--codebook", default="codebook_v2_amended_b150.json")
    ap.add_argument("--labels", default="item_labels_b150.jsonl")
    ap.add_argument("--sheet", default="gold/gold_sheet_2_labelled.md")
    ap.add_argument("--key", default="gold/gold_partB_key_2.json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seed_everything(42)
    {"validate": validate, "score": score, "reask": reask}[a.mode](a)


if __name__ == "__main__":
    main()
