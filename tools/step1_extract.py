"""Step 1: extract raw skill labels, one pass per question.

Per the pipeline doc, the extraction prompt is kept but gains two fields:
  rationale      - where in the solution the skill is used. This is the only
                   evidence available downstream when merging and validating;
                   without it every later judgment is a guess.
  canonical_form - noun phrase, modifiers stripped, verb-object order
                   normalised. This is what Step 2 dedupes and counts on.

Prompt rules are the v3 set (cdmeval/skills/extraction.py): format only. The
banned-word list and the "fewer than 100 in 10,000" rule were removed on
2026-08-11 (the project log) because both ask the model for a corpus statistic it
cannot see while looking at one item.

Reads FULL question text. The published pipeline embedded a 200-char preview,
which accounted for ~93% of every improved number in the follow-up experiments
(T-092). Not repeating that.

    python3 tools/step1_extract.py smoke     # 20 items, scratch output
    python3 tools/step1_extract.py run       # all 9,523, resumable
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, BULK

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
OUT = REPO / "cdm_exploration/experiments/pipeline_v7"
OUT.mkdir(parents=True, exist_ok=True)

SYS = """You are a psychometrician designing a Q-matrix for cognitive diagnostic assessment of AI systems. Your task: identify the specific cognitive skills each test item measures.

RULES:
1. Each skill must be a compound phrase (5-10 words): [specific_cognitive_process] + [specific_content_domain]
2. Extract 2-4 skills per item. At least one must be domain-specific.
3. Name the operation a solver must perform, not the subject area the item belongs to.
4. Use lowercase snake_case labels.

Judge by the operation, never by the cover story. Two questions that require the
same operation share a skill even if one is about dancers and the other about
gifts. Two questions on the same topic do not share a skill unless the operation
is the same.

GOOD examples by benchmark type:

MATH: "factoring_higher_degree_polynomials_over_integers", "applying_pigeonhole_principle_to_combinatorial_bounds", "computing_modular_arithmetic_in_residue_classes"

BBH: "tracing_boolean_operator_precedence_in_nested_expressions", "identifying_causal_direction_from_correlational_evidence", "tracking_object_positions_through_spatial_transformations"

GPQA: "applying_gauss_law_to_cylindrical_charge_distributions", "predicting_reaction_products_via_retrosynthetic_analysis", "interpreting_phylogenetic_trees_from_molecular_sequence_data"

MuSR: "integrating_alibis_and_motives_to_identify_suspects", "tracking_object_locations_across_sequential_room_transfers", "resolving_team_allocation_under_mutual_exclusion_constraints"

IFEval: "enforcing_exact_word_count_in_structured_output", "maintaining_consistent_formatting_across_nested_lists", "embedding_required_keywords_while_preserving_coherence"

BAD examples (name the operation instead):
- "problem solving" -> "applying_substitution_to_solve_nonlinear_systems"
- "reading comprehension" -> "extracting_temporal_ordering_from_narrative_clues\""""

USER = """Benchmark: {benchmark} | Subtask: {subtask}

Item:
{question}

Respond in JSON only:
{{"skills": [{{"label": "<snake_case, 5-10 words>",
              "process": "<the cognitive operation>",
              "domain": "<the content area>",
              "rationale": "<one sentence: where in the solution this skill is used>",
              "canonical_form": "<noun phrase, modifiers stripped, verb-object order>"}}],
 "primary_skill": "<label of the most important one>"}}"""


def load_items():
    txt = {r["item_idx"]: " ".join(r["question_full_text"].split())
           for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {p["item_idx"]: p for p in
            json.load(open(REPO / "cdm_exploration/experiments/oldtax_repaired_FINAL.json"))["per_item"]}
    return [{"item_idx": i, "question": txt[i],
             "benchmark": meta.get(i, {}).get("benchmark"),
             "subtask": meta.get(i, {}).get("subtask")} for i in sorted(txt)]


def main(mode):
    items = load_items()
    outf = OUT / ("raw_labels_smoke.jsonl" if mode == "smoke" else "raw_labels.jsonl")
    if mode == "smoke":
        items = items[:10] + items[5000:5010]
        outf.unlink(missing_ok=True)

    # an errored row is missing data, not a result: re-running must retry it.
    # Step 4 already did this; Step 1 had the opposite behaviour, so a budget trip
    # mid-run would have been recorded as 9,523 "done" items.
    done = set()
    if outf.exists():
        allrows = [json.loads(l) for l in outf.open()]
        keep = [r for r in allrows if "error" not in r]
        done = {r["item_idx"] for r in keep}
        if len(keep) != len(allrows):
            print(f"  dropping {len(allrows)-len(keep)} errored rows to retry them")
            outf.write_text("".join(json.dumps(r) + "\n" for r in keep))
    todo = [it for it in items if it["item_idx"] not in done]
    print(f"step 1: {len(todo)} items to extract ({len(done)} already done), model {BULK}")
    if not todo:
        return

    g = Gemini()
    t0 = time.time()

    def one(it):
        u = USER.format(benchmark=it["benchmark"], subtask=it["subtask"], question=it["question"])
        try:
            obj = g.json_obj(SYS, u, model=BULK, max_out=1600)
            return {"item_idx": it["item_idx"], "benchmark": it["benchmark"],
                    "subtask": it["subtask"], "skills": obj.get("skills", []),
                    "primary_skill": obj.get("primary_skill")}
        except GeminiError as e:
            # recorded as an explicit failure, never as "no skills" (T-100)
            return {"item_idx": it["item_idx"], "error": str(e)[:200]}

    n_err = 0
    with open(outf, "a") as f:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for k, r in enumerate(ex.map(one, todo)):
                f.write(json.dumps(r) + "\n")
                if "error" in r:
                    n_err += 1
                if k % 200 == 0:
                    f.flush()
                    print(f"  {k+1}/{len(todo)}  errors {n_err}  {g.total_tokens:,} tok  "
                          f"{time.time()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in outf.open()]
    ok = [r for r in rows if "error" not in r]
    labels = [s.get("canonical_form") or s.get("label") for r in ok for s in r["skills"]]
    c = Counter(labels)
    print(f"\nitems: {len(rows)}  ok: {len(ok)}  errors: {len(rows)-len(ok)}")
    print(f"labels emitted: {len(labels):,}   unique: {len(c):,}")
    print(f"mean labels per item: {len(labels)/max(1,len(ok)):.2f}")
    top = sum(n for _, n in c.most_common(100))
    print(f"top-100 labels cover {top/max(1,len(labels)):.1%} of mentions")
    print(f"usage: {g.report()}")
    print(f"wrote {outf}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke")
