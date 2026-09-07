"""Blind quality judge with DUAL-ORDER control.

The single-order run showed severe position bias: the judge picked the set
shown second in 70% of cases (551/789), so the apparent 47/52 split was an
artifact of alternating presentation order, not a content preference.

Fix: judge every item TWICE, once in each order. A verdict counts only if it
survives the swap (same arm wins both times). Items where the winner flips
with the order are position-driven and are reported separately as UNSTABLE
rather than folded into the totals.

Reported: win rate among CONSISTENT verdicts, plus the consistency rate
itself, which is the honest measure of how separable the two arms are.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"
JUDGE = "gpt-4.1-mini"

# Judge criterion = the project's operational definition of a skill (identical to
# tools/taxonomy_pipeline.py SKILL_DEF, used for naming, name-fit judging and the
# human evaluation), so this ablation measures the same construct as everything else.
# The earlier wording ("what a solver must actually do ... accuracy and specificity")
# used the judge's own notion of skill and rewarded narrow phrasing; superseded.
SKILL_DEF = ("A skill is the smallest named mental operation a solver must master to answer "
             "a question correctly; the same skill can appear across different topics and formats.")
SYS = (f"You evaluate skill labels for test questions. {SKILL_DEF} "
       "Given a question and two candidate sets of skill labels, decide which set is better "
       "aligned with the cognitive skills the question actually requires. Judge by the mental "
       "operation needed to solve the question, never by surface topic words or wording style.")


def load_key() -> str:
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def load_arms():
    ref = {r["item_idx"]: [s for s in r["skills_with_ref"] if s]
           for r in json.loads((E / "v2_reference_answer_ablation.json").read_text())["results"]}
    ctrl = {r["item_idx"]: [s for s in r["skills"] if s]
            for r in json.loads((E / "v2_refablation_ctrl.json").read_text())["results"]}
    qt = {r["item_idx"]: r["question_full_text"]
          for r in json.loads((D / "item_full_text_recovered.json").read_text())}
    meta = {it["item_idx"]: it for it in json.loads((D / "response_matrix_v2_full_items.json").read_text())}
    return ctrl, ref, qt, meta, sorted(set(ref) & set(ctrl))


async def run(ctrl, ref, qt, meta, sel):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=load_key())
    sem = asyncio.Semaphore(10)

    async def one(i, ref_first: bool):
        first, second = (ref[i], ctrl[i]) if ref_first else (ctrl[i], ref[i])
        u = (f"Question:\n{' '.join(qt[i].split())[:1200]}\n\n"
             "Set 1:\n" + "\n".join(f"- {s}" for s in first) +
             "\n\nSet 2:\n" + "\n".join(f"- {s}" for s in second) +
             '\n\nWhich set is better aligned with the cognitive skills this question requires? '
             'Return JSON {"winner":"1|2|tie"}.')
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=JUDGE,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                    response_format={"type": "json_object"}, temperature=0, max_tokens=40)
                w = json.loads(r.choices[0].message.content).get("winner", "tie")
                if w == "tie":
                    arm = "tie"
                else:
                    arm = "ref" if ((w == "1") == ref_first) else "ctrl"
                return {"i": i, "ref_first": ref_first, "arm": arm, "pos": w,
                        "usage": r.usage.total_tokens}
            except Exception as ex:
                return {"i": i, "ref_first": ref_first, "error": str(ex)[:120]}

    tasks = [one(i, True) for i in sel] + [one(i, False) for i in sel]
    return await asyncio.gather(*tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    ctrl, ref, qt, meta, idx = load_arms()
    sel = sorted(random.Random(42).sample(idx, min(a.n, len(idx))))
    print(f"dual-order judging {len(sel)} items ({2*len(sel)} calls)", flush=True)

    res = asyncio.run(run(ctrl, ref, qt, meta, sel))
    ok = [r for r in res if "error" not in r]
    tok = sum(r.get("usage", 0) for r in ok)
    print(f"calls ok {len(ok)}/{len(res)}, tokens {tok:,} (~${tok/1e6*1.0:.2f})\n", flush=True)

    byi = {}
    for r in ok:
        byi.setdefault(r["i"], {})[r["ref_first"]] = r["arm"]
    both = {i: v for i, v in byi.items() if len(v) == 2}

    pos = Counter(r["pos"] for r in ok)
    print(f"position bias check: set1 {pos['1']}, set2 {pos['2']}, tie {pos['tie']} "
          f"-> second position chosen {pos['2']/(pos['1']+pos['2']):.1%} of the time", flush=True)

    consistent, unstable = {}, 0
    for i, v in both.items():
        if v[True] == v[False] and v[True] != "tie":
            consistent[i] = v[True]
        else:
            unstable += 1

    n = len(both)
    c = Counter(consistent.values())
    nc = len(consistent)
    print(f"\nitems judged in both orders: {n}", flush=True)
    print(f"  verdict FLIPS with order (position-driven): {unstable} ({unstable/n:.1%})", flush=True)
    print(f"  verdict STABLE across order               : {nc} ({nc/n:.1%})", flush=True)
    if nc:
        print(f"\namong stable verdicts (n={nc}):", flush=True)
        print(f"  solution-aware better : {c['ref']:4d}  ({c['ref']/nc:.1%})", flush=True)
        print(f"  question-only better  : {c['ctrl']:4d}  ({c['ctrl']/nc:.1%})", flush=True)
        for b in ["MATH", "GPQA"]:
            cb = Counter(v for i, v in consistent.items() if meta[i]["benchmark"] == b)
            nb = sum(cb.values())
            if nb:
                print(f"    {b}: solution-aware {cb['ref']/nb:.1%}, question-only {cb['ctrl']/nb:.1%} (n={nb})", flush=True)

    out = {"experiment": "refablation_quality_dual_order", "judge": JUDGE,
           "n_items": n, "n_calls": len(ok),
           "position_second_chosen_rate": pos["2"] / (pos["1"] + pos["2"]) if (pos["1"] + pos["2"]) else None,
           "unstable_rate": unstable / n if n else None,
           "stable_n": nc,
           "solution_aware_win_among_stable": c["ref"] / nc if nc else None,
           "question_only_win_among_stable": c["ctrl"] / nc if nc else None,
           "by_benchmark": {b: {k: (Counter(v for i, v in consistent.items()
                                            if meta[i]["benchmark"] == b)[k] /
                                    max(1, sum(Counter(v for i, v in consistent.items()
                                                       if meta[i]["benchmark"] == b).values())))
                                for k in ["ref", "ctrl"]} for b in ["MATH", "GPQA"]},
           "supersedes": "v2_refablation_quality.json single-order judge (position bias 70%)",
           "verified": True}
    out["judge_criterion"] = "project SKILL_DEF (taxonomy_pipeline.py), same construct as name-fit and human eval"
    (E / "v2_refablation_quality_dual.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {E / 'v2_refablation_quality_dual.json'}", flush=True)


if __name__ == "__main__":
    main()
