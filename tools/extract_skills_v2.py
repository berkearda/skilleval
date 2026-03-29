"""Extract skills for v2 full dataset using benchmark-aware prompt.

Uses OpenAI GPT-4o-mini with async batching for speed.

Usage:
    python tools/extract_skills_v2.py
    python tools/extract_skills_v2.py --resume        # resume from checkpoint
    python tools/extract_skills_v2.py --limit 100     # test on first 100 items
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm


FORBIDDEN_LABELS = {
    "reading comprehension", "logical reasoning", "critical thinking",
    "mathematical reasoning", "problem solving", "analytical thinking",
    "scientific knowledge", "attention to detail", "numerical reasoning",
}


def check_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY=sk-...")
        return False
    return True


async def extract_batch_async(items_batch, model="gpt-4o-mini"):
    """Extract skills for a batch of items concurrently."""
    from openai import AsyncOpenAI
    from cdmeval.skills.extraction import QMATRIX_SYSTEM_PROMPT, QMATRIX_USER_TEMPLATE

    client = AsyncOpenAI()

    async def extract_one(item):
        question = item.get("question_preview", item.get("problem", ""))
        if len(question) < 10:
            question = f"[{item['benchmark']} / {item['subtask']}]"

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": QMATRIX_SYSTEM_PROMPT},
                    {"role": "user", "content": QMATRIX_USER_TEMPLATE.format(
                        benchmark=item["benchmark"],
                        subtask=item["subtask"],
                        question_text=question[:1500],
                    )},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            skills = result.get("skills", [])
            labels = []
            for s in skills:
                if isinstance(s, dict):
                    labels.append(s.get("label", ""))
                elif isinstance(s, str):
                    labels.append(s)

            # Filter forbidden
            labels = [l for l in labels
                      if l.lower().replace("_", " ") not in FORBIDDEN_LABELS and len(l) > 3]

            return {
                "item_idx": item["item_idx"],
                "benchmark": item["benchmark"],
                "subtask": item["subtask"],
                "problem": question[:500],
                "skills": labels,
                "primary_skill": result.get("primary_skill", labels[0] if labels else ""),
                "num_skills": len(labels),
                "reasoning": "",
            }
        except Exception as e:
            return {
                "item_idx": item["item_idx"],
                "benchmark": item["benchmark"],
                "subtask": item["subtask"],
                "problem": question[:500],
                "skills": [],
                "primary_skill": "",
                "num_skills": 0,
                "reasoning": f"Error: {str(e)}",
            }

    tasks = [extract_one(item) for item in items_batch]
    return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if not check_api_key():
        sys.exit(1)

    data_dir = Path("cdm_exploration/data/cdm_ready")
    out_path = data_dir / "skills_v2_new_prompt.json"
    checkpoint_path = data_dir / "skills_v2_checkpoint.json"

    # Load items
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        all_items = json.load(f)

    if args.limit:
        all_items = all_items[:args.limit]

    print(f"Total items: {len(all_items)}")

    # Resume from checkpoint
    results = []
    start_idx = 0
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx} items done")

    remaining = all_items[start_idx:]
    print(f"Remaining: {len(remaining)} items")
    print(f"Batch size: {args.batch_size} concurrent requests")
    print(f"Estimated time: ~{len(remaining) * 0.15 / 60:.0f} minutes")
    print()

    # Process in batches
    for batch_start in tqdm(range(0, len(remaining), args.batch_size),
                            desc="Batches", total=len(remaining) // args.batch_size + 1):
        batch = remaining[batch_start:batch_start + args.batch_size]
        batch_results = asyncio.run(extract_batch_async(batch, args.model))
        results.extend(batch_results)

        # Checkpoint every 500 items
        if len(results) % 500 < args.batch_size:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)
            tqdm.write(f"  Checkpoint: {len(results)} items")

        # Rate limit
        time.sleep(0.05)

    # Sort by item_idx
    results.sort(key=lambda x: x["item_idx"])

    # Save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({len(results)} items)")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # ── Stats ──
    print(f"\n{'='*70}")
    print("EXTRACTION STATS (v2 new prompt)")
    print(f"{'='*70}")

    all_labels = []
    per_benchmark = {}
    for r in results:
        labels = r.get("skills", [])
        if isinstance(labels, list):
            all_labels.extend(labels)
        bm = r.get("benchmark", "unknown")
        if bm not in per_benchmark:
            per_benchmark[bm] = {"items": 0, "skills": [], "zero": 0}
        per_benchmark[bm]["items"] += 1
        per_benchmark[bm]["skills"].extend(labels if isinstance(labels, list) else [])
        if r.get("num_skills", 0) == 0:
            per_benchmark[bm]["zero"] += 1

    unique_labels = set(l.lower().strip() for l in all_labels)
    skills_per_item = [r.get("num_skills", 0) for r in results]
    import numpy as np
    spa = np.array(skills_per_item)

    print(f"  Total items: {len(results)}")
    print(f"  Total unique skills: {len(unique_labels)}")
    print(f"  Skills/item: mean={spa.mean():.2f}, median={np.median(spa):.0f}, std={spa.std():.2f}")
    print(f"  Items with 0 skills: {(spa == 0).sum()}")

    # Forbidden check
    forbidden_count = sum(1 for l in all_labels
                          if l.lower().replace("_", " ") in FORBIDDEN_LABELS)
    print(f"  Forbidden labels remaining: {forbidden_count}")

    # Per-benchmark
    print(f"\n  {'Benchmark':<12} {'Items':>6} {'Unique':>8} {'Sk/Item':>8} {'Zero':>6}")
    print(f"  {'-'*45}")
    for bm in sorted(per_benchmark):
        info = per_benchmark[bm]
        uniq = len(set(s.lower().strip() for s in info["skills"]))
        mean_sk = len(info["skills"]) / max(info["items"], 1)
        print(f"  {bm:<12} {info['items']:>6} {uniq:>8} {mean_sk:>8.2f} {info['zero']:>6}")

    # Top 20 skills
    counts = Counter(l.lower().strip() for l in all_labels)
    print(f"\n  Top 20 skills:")
    for label, count in counts.most_common(20):
        pct = count / len(results) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    # Compare with old
    print(f"\n  Comparison with old extraction:")
    print(f"    Old: 18,785 unique, 4.5 sk/item, silhouette 0.008 at K=50")
    print(f"    New: {len(unique_labels)} unique, {spa.mean():.2f} sk/item")


if __name__ == "__main__":
    main()
