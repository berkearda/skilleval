"""Step-wise skill extraction: label the reasoning chain, not the question.

For each item the LLM solves it, labels each reasoning step with an atomic
skill, and gives one overall skill. Returns BOTH (atomic steps for the
Q-matrix; overall skill for labeling) in a single call.

Dry read:
    OPENAI_API_KEY=... python tools/extract_stepwise.py --item-idxs 2715 ... --pretty
Pilot/full run:
    OPENAI_API_KEY=... python tools/extract_stepwise.py --keys bbh_hyperbaton --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
SRC = DATA / "item_full_text_recovered.json"

SYSTEM_PROMPT = """You are a psychometrician building a Q-matrix for cognitive diagnosis. For a given test item you identify the cognitive skills it MEASURES by working through how it is solved.

Do this:
1. Solve the item, broken into reasoning steps.
2. Label EACH step with one atomic skill: a short snake_case phrase naming the specific cognitive OPERATION (not the surface topic). Atomic skills must be reusable across many items.
   - Good: isolate_the_variable, apply_quadratic_formula, negate_a_boolean_chain, order_adjectives_by_english_rule, track_object_positions_after_swaps
   - Bad (too generic): logical_reasoning, problem_solving, critical_thinking
   - Bad (too item-specific): solve_for_x_in_this_particular_equation
3. Give ONE overall_skill summarizing what the item primarily measures.

Return 2-6 reasoning steps. Each step's skill is the cognitive operation that step requires."""

# Variant B: surgical tweaks to reduce mid-level generic leakage without
# over-crafting -- (1) reframe reusability away from "many items", (2) name
# the actual leaked generics as bad, (3) skip purely procedural answer steps.
SYSTEM_PROMPT_B = """You are a psychometrician building a Q-matrix for cognitive diagnosis. For a given test item you identify the cognitive skills it MEASURES by working through how it is solved.

Do this:
1. Solve the item, broken into reasoning steps.
2. Label EACH step with one atomic skill: a short snake_case phrase naming the specific cognitive OPERATION (not the surface topic). A good skill is reusable across OTHER items that require the SAME operation, but it must still name a concrete, content-bearing operation, not a vague catch-all.
   - Good: isolate_the_variable, apply_quadratic_formula, negate_a_boolean_chain, order_adjectives_by_english_rule, track_object_positions_after_swaps, trace_truth_value_through_statement_chain
   - Bad (too generic / vague catch-alls): logical_reasoning, problem_solving, critical_thinking, evaluate_statement, determine_truth_value, apply_deductive_reasoning, identify_key_components, extract_key_information, make_a_choice_based_on_criteria
   - Bad (too item-specific): solve_for_x_in_this_particular_equation
   Do NOT emit purely procedural answering steps such as "select the correct option" or "make a choice" - skip those entirely and label only the substantive cognitive operations.
3. Give ONE overall_skill summarizing what the item primarily measures.

Return 2-6 reasoning steps. Each step's skill is the cognitive operation that step requires."""

USER_TEMPLATE = """Benchmark: {benchmark} | Subtask: {subtask}

Item:
{question_text}
{answer_block}
Respond in JSON:
{{
  "reasoning_steps": [
    {{"step": "<what you do>", "skill": "<atomic_skill_snake_case>"}}
  ],
  "overall_skill": "<one snake_case skill>"
}}"""


def truncate(text: str, head: int = 2000, tail: int = 4000) -> str:
    t = " ".join(text.split())
    return t if len(t) <= head + tail else t[:head] + "\n...\n" + t[-tail:]


def build_user(r):
    ans = ""
    if r.get("solution"):
        ans = f"\nWorked solution (ground truth):\n{r['solution'][:1200]}\n"
    return USER_TEMPLATE.format(benchmark=r["benchmark"], subtask=r["subtask"],
                               question_text=truncate(r["question_full_text"]),
                               answer_block=ans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--item-idxs", nargs="*", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--variant", choices=["a", "b"], default="a",
                    help="a = current prompt, b = surgically-tightened")
    ap.add_argument("--pretty", action="store_true", help="print results readably")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = json.load(open(SRC))
    if args.item_idxs:
        keep = set(args.item_idxs)
        rows = [r for r in rows if r["item_idx"] in keep]
    if args.keys:
        rows = [r for r in rows if r["key"] in set(args.keys)]
    if args.limit:
        rows = rows[: args.limit]
    SYS = SYSTEM_PROMPT if args.variant == "a" else SYSTEM_PROMPT_B
    print(f"items: {len(rows)} | variant={args.variant}", flush=True)

    def parse(r, res):
        steps = res.get("reasoning_steps", [])
        atomic = [s.get("skill", "") for s in steps if isinstance(s, dict)]
        return {**{k: r[k] for k in ("item_idx", "benchmark", "subtask", "key")},
                "atomic_skills": [a for a in atomic if a],
                "overall_skill": res.get("overall_skill", ""),
                "reasoning_steps": steps}

    if args.pretty:
        from openai import OpenAI
        client = OpenAI()
        for r in rows:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": build_user(r)}],
                temperature=0.0, response_format={"type": "json_object"})
            rec = parse(r, json.loads(resp.choices[0].message.content))
            q = " ".join(r["question_full_text"].split())
            print("\n" + "=" * 78)
            print(f"[{r['benchmark']}/{r['subtask']}]  item {r['item_idx']}")
            print(f"Q: {q[:200]}")
            print("reasoning steps -> atomic skill:")
            for s in rec["reasoning_steps"]:
                print(f"   - {s.get('step','')[:70]:70s} => {s.get('skill','')}")
            print(f"OVERALL SKILL: {rec['overall_skill']}")
        return 0

    # concurrent path for the pilot/full run (run under Python 3.12)
    import asyncio

    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    out = [None] * len(rows)
    done = {"n": 0}

    async def one(i, r):
        for attempt in range(4):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "system", "content": SYS},
                                  {"role": "user", "content": build_user(r)}],
                        temperature=0.0, response_format={"type": "json_object"})
                out[i] = parse(r, json.loads(resp.choices[0].message.content))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    out[i] = {**{k: r[k] for k in ("item_idx", "benchmark", "subtask", "key")},
                              "atomic_skills": [], "overall_skill": "", "error": str(e)[:150]}
                else:
                    await asyncio.sleep(2 * (attempt + 1))
        done["n"] += 1
        if done["n"] % 200 == 0:
            print(f"  {done['n']}/{len(rows)}", flush=True)

    async def runner():
        await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))

    asyncio.run(runner())
    errs = sum(1 for r in out if r and r.get("error"))
    json.dump(out, open(args.out, "w"))
    print(f"\nwrote {args.out} ({len(out)} items, {errs} errored)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
