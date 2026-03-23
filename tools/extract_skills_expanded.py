"""Extract skills for BBH items and combine with existing MATH skills.

Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python tools/extract_skills_expanded.py

    # Or with OpenAI:
    export OPENAI_API_KEY=sk-...
    python tools/extract_skills_expanded.py --api openai
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


def check_api_key(api):
    """Check if required API key is set."""
    if api == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            print("  export ANTHROPIC_API_KEY=sk-ant-...")
            return False
    elif api == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("ERROR: OPENAI_API_KEY not set.")
            print("  export OPENAI_API_KEY=sk-...")
            return False
    return True


def match_existing_math_skills(existing_skills, expanded_math_items):
    """Match existing MATH skills to expanded items by question content."""
    # Build lookup by question preview (first 150 chars)
    existing_by_preview = {}
    for item in existing_skills:
        if item.get("source") == "MATH":
            preview = item["problem"][:150].strip()
            existing_by_preview[preview] = item

    matched = []
    unmatched = []
    for exp_item in expanded_math_items:
        preview = exp_item["question_preview"][:150].strip()
        if preview in existing_by_preview:
            old = existing_by_preview[preview]
            matched.append({
                "item_idx": exp_item["item_idx"],
                "benchmark": "MATH",
                "subtask": exp_item["subtask"],
                "problem": old["problem"],
                "skills": old["skills"],
                "primary_skill": old.get("primary_skill", ""),
                "num_skills": old.get("num_skills", len(old.get("skills", []))),
                "reasoning": old.get("reasoning", ""),
            })
        else:
            unmatched.append(exp_item)

    return matched, unmatched


def extract_single(problem, api, model=None):
    """Extract skills for a single problem."""
    from cdmeval.skills.extraction import EXTRACTORS
    extract_fn = EXTRACTORS[api]
    if model:
        return extract_fn(problem, model)
    return extract_fn(problem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="anthropic", choices=["anthropic", "openai", "mock"])
    parser.add_argument("--model", default=None, help="Model override (e.g. claude-3-haiku-20240307)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    data_dir = Path("cdm_exploration/data/cdm_ready")
    checkpoint_path = data_dir / "skills_bbh_checkpoint.json"

    # Check API key (skip for mock)
    if args.api != "mock" and not check_api_key(args.api):
        sys.exit(1)

    # ── Load data ──
    with open(data_dir / "skills_extracted.json") as f:
        existing_skills = json.load(f)
    print(f"Existing skills: {len(existing_skills)} items")

    with open(data_dir / "response_matrix_expanded_items.json") as f:
        expanded_items = json.load(f)

    math_items = [it for it in expanded_items if it["benchmark"] == "MATH"]
    bbh_items = [it for it in expanded_items if it["benchmark"] == "BBH"]
    print(f"Expanded: {len(math_items)} MATH + {len(bbh_items)} BBH = {len(expanded_items)}")

    # ── Match existing MATH skills ──
    print("\nMatching existing MATH skills...")
    matched_math, unmatched_math = match_existing_math_skills(existing_skills, math_items)
    print(f"  Matched: {len(matched_math)}, Unmatched: {len(unmatched_math)}")

    # ── Load checkpoint if resuming ──
    bbh_results = []
    start_idx = 0
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            bbh_results = json.load(f)
        start_idx = len(bbh_results)
        print(f"  Resuming from checkpoint: {start_idx} BBH items already done")

    # ── Extract skills for BBH items ──
    items_to_extract = bbh_items[start_idx:]
    if unmatched_math:
        items_to_extract = unmatched_math + items_to_extract
        print(f"  Also extracting {len(unmatched_math)} unmatched MATH items")

    print(f"\nExtracting skills for {len(items_to_extract)} items using {args.api}...")
    print(f"  Estimated cost: ~${len(items_to_extract) * 0.0002:.2f} (Haiku)")
    print(f"  Estimated time: ~{len(items_to_extract) * args.delay / 60:.0f} minutes")
    print()

    for i, item in enumerate(tqdm(items_to_extract, desc="Extracting")):
        try:
            problem = item.get("question_preview", "")
            if len(problem) < 10:
                problem = f"[{item['benchmark']} / {item['subtask']}] No question text available"

            extraction = extract_single(problem, args.api, args.model)
            bbh_results.append({
                "item_idx": item["item_idx"],
                "benchmark": item["benchmark"],
                "subtask": item["subtask"],
                "problem": problem,
                "skills": extraction.get("skills", []),
                "primary_skill": extraction.get("primary_skill", ""),
                "num_skills": len(extraction.get("skills", [])),
                "reasoning": extraction.get("reasoning", ""),
            })

            if args.api != "mock":
                time.sleep(args.delay)

        except Exception as e:
            print(f"\n  Error on item {item['item_idx']}: {e}")
            bbh_results.append({
                "item_idx": item["item_idx"],
                "benchmark": item["benchmark"],
                "subtask": item["subtask"],
                "problem": item.get("question_preview", ""),
                "skills": [],
                "primary_skill": "",
                "num_skills": 0,
                "reasoning": f"Error: {str(e)}",
            })

        # Save checkpoint every 500 items
        if (i + 1) % 500 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(bbh_results, f)
            print(f"\n  Checkpoint saved: {len(bbh_results)} items")

    # ── Combine all ──
    print("\nCombining results...")
    all_results = matched_math + bbh_results

    # Sort by item_idx
    all_results.sort(key=lambda x: x["item_idx"])

    # ── Summary ──
    total_skills = sum(len(r["skills"]) if isinstance(r["skills"], list) else 0 for r in all_results)
    unique_skills = set()
    for r in all_results:
        skills = r["skills"]
        if isinstance(skills, str):
            import ast
            skills = ast.literal_eval(skills)
        for s in skills:
            unique_skills.add(s.lower().strip())

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total items: {len(all_results)}")
    print(f"  Total skill mentions: {total_skills}")
    print(f"  Unique skills: {len(unique_skills)}")
    print(f"  Mean skills/item: {total_skills / max(len(all_results), 1):.1f}")

    # Examples from BBH
    print(f"\n  Example BBH skills:")
    bbh_with_skills = [r for r in bbh_results if r["num_skills"] > 0]
    for r in bbh_with_skills[:3]:
        print(f"    [{r['subtask']}] {r['skills'][:4]}")

    # ── Save ──
    out_path = data_dir / "skills_extracted_expanded.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path} ({len(all_results)} items)")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file")

    print("\nDone.")


if __name__ == "__main__":
    main()
