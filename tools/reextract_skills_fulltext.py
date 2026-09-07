"""Re-extract skills on the RECOVERED FULL question text.

Identical to the v2 extraction pipeline in every respect EXCEPT the input:
it reads `question_full_text` from item_full_text_recovered.json instead of
the 200-char `question_preview`. Same QMATRIX prompt, same model call, so any
difference in the resulting skills/clusters is attributable to the input fix.

Cost control: --keys to restrict to specific subtasks (e.g. the broken
clusters), --limit N, and --smoke (3 items). --dry-run builds and prints the
prompt WITHOUT calling the API (no key needed) to verify the plumbing.

Run for real:  OPENAI_API_KEY=... python tools/reextract_skills_fulltext.py --model gpt-4o-mini --keys bbh_hyperbaton bbh_web_of_lies
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
SRC = DATA / "item_full_text_recovered.json"


# Inlined verbatim from cdmeval/skills/extraction.py so this script is
# self-contained (runs under any Python with `openai`, no cdmeval import).
QMATRIX_SYSTEM_PROMPT = """You are a psychometrician designing a Q-matrix for cognitive diagnostic assessment of AI systems. Your task: identify the specific cognitive skills each test item measures.

RULES:
1. Each skill must be a compound phrase (5-10 words): [specific_cognitive_process] + [specific_content_domain]
2. Extract 2-4 skills per item. At least one must be domain-specific.
3. FORBIDDEN generic labels (will be rejected):
   - "reading comprehension", "logical reasoning", "critical thinking"
   - "mathematical reasoning", "problem solving", "analytical thinking"
   - "scientific knowledge", "attention to detail", "numerical reasoning"
   - Any label that would apply to >15% of items in the benchmark
4. Specificity test: would fewer than 100 items in a 10,000-item test share this exact skill? If not, decompose further.
5. Use lowercase snake_case labels.

GOOD examples by benchmark type:

MATH: "factoring_higher_degree_polynomials_over_integers", "applying_pigeonhole_principle_to_combinatorial_bounds", "computing_modular_arithmetic_in_residue_classes"

BBH: "tracing_boolean_operator_precedence_in_nested_expressions", "identifying_causal_direction_from_correlational_evidence", "tracking_object_positions_through_spatial_transformations"

GPQA: "applying_gauss_law_to_cylindrical_charge_distributions", "predicting_reaction_products_via_retrosynthetic_analysis", "interpreting_phylogenetic_trees_from_molecular_sequence_data"

MuSR: "integrating_alibis_and_motives_to_identify_suspects", "tracking_object_locations_across_sequential_room_transfers", "resolving_team_allocation_under_mutual_exclusion_constraints"

IFEval: "enforcing_exact_word_count_in_structured_output", "maintaining_consistent_formatting_across_nested_lists", "embedding_required_keywords_while_preserving_coherence"

BAD examples (too generic, rejected):
- "problem solving" -> decompose to "applying_substitution_to_solve_nonlinear_systems"
- "reading comprehension" -> decompose to "extracting_temporal_ordering_from_narrative_clues"
- "scientific knowledge" -> decompose to "applying_conservation_of_angular_momentum_to_rotating_bodies"
"""

QMATRIX_USER_TEMPLATE = """Benchmark: {benchmark} | Subtask: {subtask}

Item:
{question_text}

Respond in JSON:
{{
  "skills": [
    {{
      "label": "5-10 word compound skill in snake_case",
      "process": "the specific cognitive operation",
      "domain": "the specific content area"
    }}
  ],
  "primary_skill": "label of the single most important skill"
}}"""

FORBIDDEN_LABELS = {
    "reading comprehension", "logical reasoning", "critical thinking",
    "mathematical reasoning", "problem solving", "analytical thinking",
    "scientific knowledge", "attention to detail", "numerical reasoning",
    "reading_comprehension", "logical_reasoning", "critical_thinking",
    "mathematical_reasoning", "problem_solving", "analytical_thinking",
    "scientific_knowledge", "attention_to_detail", "numerical_reasoning",
}


def truncate(text: str, head: int = 2000, tail: int = 4000) -> str:
    """Bound prompt length while preserving the trailing question.

    Long items (MuSR narratives ~4.8k chars) put the actual question at the
    END, so a head-only cut would drop it. Keep head + tail.
    """
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n...\n" + text[-tail:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="restrict to these item keys (subtasks)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="3 items, dry-run prompt")
    ap.add_argument("--dry-run", action="store_true",
                    help="build+print the prompt, do not call the API")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="concurrent API requests")
    ap.add_argument("--out", default=str(DATA / "skills_extracted_v2_fulltext.json"))
    args = ap.parse_args()

    rows = json.load(open(SRC))
    if args.keys:
        rows = [r for r in rows if r["key"] in set(args.keys)]
    if args.smoke:
        rows = rows[:3]
        args.dry_run = args.dry_run or True
    if args.limit:
        rows = rows[: args.limit]
    print(f"items to process: {len(rows)} (model={args.model}, dry_run={args.dry_run})",
          flush=True)

    # B4 sanity: confirm we're feeding FULL text, not 200-char preview
    lens = [r["n_chars"] for r in rows]
    print(f"input length: min {min(lens)} median {sorted(lens)[len(lens)//2]} max {max(lens)}",
          flush=True)

    def build_user(r):
        return QMATRIX_USER_TEMPLATE.format(
            benchmark=r["benchmark"], subtask=r["subtask"],
            question_text=truncate(r["question_full_text"]))

    if args.dry_run:
        r = rows[0]
        print("\n=== DRY-RUN: prompt for item", r["item_idx"], "===")
        print("[system]", QMATRIX_SYSTEM_PROMPT[:120], "...")
        print("[user]\n", build_user(r))
        print("\n(no API call made; set OPENAI_API_KEY and drop --dry-run to run)")
        return 0

    import asyncio as _asyncio  # noqa: F811

    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    out: list = [None] * len(rows)
    done = {"n": 0}

    async def one(i, r):
        for attempt in range(4):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "system", "content": QMATRIX_SYSTEM_PROMPT},
                                  {"role": "user", "content": build_user(r)}],
                        temperature=0.0, response_format={"type": "json_object"})
                result = json.loads(resp.choices[0].message.content)
                labels = [s.get("label", "") if isinstance(s, dict) else str(s)
                          for s in result.get("skills", [])]
                labels = [x for x in labels
                          if x.lower().replace("_", " ") not in FORBIDDEN_LABELS]
                out[i] = {**r, "skills": labels,
                          "primary_skill": result.get("primary_skill",
                                                      labels[0] if labels else "")}
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    out[i] = {**r, "skills": [], "primary_skill": "",
                              "error": str(e)[:200]}
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
    print(f"wrote {args.out} ({len(out)} items, {errs} errored)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
