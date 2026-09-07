"""LLM-based clustering of the pilot's step-wise atomic skills (vs HAC).

Two stages (no pairwise comparison):
  Stage 1 - build ~K cognitive-skill categories from the full list of unique
            atomic skills (one call; skills are short, they all fit).
  Stage 2 - assign each unique skill to a category, in batches.

Then build item->categories and print the same kind of cards as the HAC run
(tools shows stepwise_clusters_K30.txt) so the two can be compared by eye.

Run:  OPENAI_API_KEY=... python tools/llm_cluster.py --k 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DATA = Path("cdm_exploration/data/cdm_ready")
STEP = DATA / "skills_stepwise_pilot.json"

TAX_SYSTEM = """You are a psychometrician building a cognitive-skill taxonomy for a Q-matrix. You are given a list of atomic skill phrases (with how often each occurs) that were extracted from test items by labeling reasoning steps.

{target} Rules:
- Each category is ONE genuine cognitive skill (a mental operation / competency), specific enough to be diagnostic.
- Do NOT create generic umbrella categories like "logical_reasoning", "problem_solving", "critical_thinking".
- Categories should be meaningfully distinct from each other (a skill, not a surface topic).
- Group by the underlying operation, not by wording. "player position" and "object position" are DIFFERENT if the operation differs; "trace_truth_chain" and "evaluate nested truth statements" are the SAME.

Return compact JSON, names only (no definitions): {{"categories": [{{"id": 0, "name": "snake_case_skill"}}, ...]}}"""

ASSIGN_SYSTEM = """You assign atomic skill phrases to the single best-fitting category from a fixed taxonomy. Group by underlying cognitive operation, not surface wording. If a skill fits none well, pick the closest. Return JSON: {"assignments": {"<skill phrase>": <category_id>, ...}} with an entry for EVERY skill given."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30,
                    help="number of categories; 0 = let the LLM decide")
    ap.add_argument("--tax-model", default="gpt-4o-mini")
    ap.add_argument("--assign-model", default="gpt-4o-mini")
    ap.add_argument("--batch", type=int, default=120)
    ap.add_argument("--out", default=str(DATA / "stepwise_llm_clusters_K30.txt"))
    args = ap.parse_args()

    step = json.load(open(STEP))
    subt = {r["item_idx"]: r["subtask"] for r in step}
    per_item = {r["item_idx"]: [s.lower() for s in (r.get("atomic_skills") or [])]
                for r in step}
    counts = Counter(s for ss in per_item.values() for s in ss)
    uniq = [s for s, _ in counts.most_common()]  # frequency order
    print(f"unique atomic skills: {len(uniq)}", flush=True)

    from openai import OpenAI
    client = OpenAI()

    # ---- Stage 1: build categories ----
    skill_list = "\n".join(f"{counts[s]:4d}  {s}" for s in uniq)
    if args.k == 0:
        target = ("Group them into AS MANY distinct, coherent COGNITIVE SKILL categories as "
                  "you naturally need to cover them cleanly (typically 40-120; do NOT force a "
                  "number). Each category is one genuine specific skill; merge true synonyms.")
    else:
        target = f"Group them into about {args.k} distinct, coherent COGNITIVE SKILL categories."
    print("Stage 1: building categories...", flush=True)
    r1 = client.chat.completions.create(
        model=args.tax_model,
        messages=[{"role": "system", "content": TAX_SYSTEM.format(target=target)},
                  {"role": "user", "content": "Atomic skills (count, name):\n" + skill_list}],
        temperature=0.0, response_format={"type": "json_object"}, max_tokens=16000)
    cats = json.loads(r1.choices[0].message.content)["categories"]
    catmap = {c["id"]: c for c in cats}
    print(f"  got {len(cats)} categories", flush=True)
    taxonomy_str = "\n".join(f'{c["id"]}: {c["name"]} - {c.get("definition","")}' for c in cats)

    # ---- Stage 2: assign skills in batches ----
    print("Stage 2: assigning skills...", flush=True)
    skill2cat = {}
    for b in range(0, len(uniq), args.batch):
        batch = uniq[b:b + args.batch]
        r2 = client.chat.completions.create(
            model=args.assign_model,
            messages=[{"role": "system", "content": ASSIGN_SYSTEM},
                      {"role": "user", "content": "TAXONOMY:\n" + taxonomy_str
                       + "\n\nAssign each of these skills:\n" + "\n".join(batch)}],
            temperature=0.0, response_format={"type": "json_object"})
        asg = json.loads(r2.choices[0].message.content).get("assignments", {})
        for s in batch:
            c = asg.get(s, asg.get(s.lower()))
            skill2cat[s] = int(c) if c is not None and str(c).lstrip("-").isdigit() else -1
        print(f"  assigned {min(b+args.batch,len(uniq))}/{len(uniq)}", flush=True)

    unassigned = sum(1 for v in skill2cat.values() if v == -1 or v not in catmap)
    print(f"  unassigned/invalid: {unassigned}", flush=True)

    # ---- build item -> categories and cards ----
    item_cats = {}
    for i, ss in per_item.items():
        cs = {skill2cat[s] for s in ss if skill2cat.get(s, -1) in catmap}
        if cs:
            item_cats[i] = cs

    lines = [f"LLM-BASED CLUSTERING of step-wise atomic skills (K={len(cats)})",
             f"taxonomy model: {args.tax_model} | assign model: {args.assign_model}", ""]
    csize = Counter(c for cs in item_cats.values() for c in cs)
    for cid, n in csize.most_common():
        members = [i for i in item_cats if cid in item_cats[i]]
        subs = Counter(subt[i] for i in members)
        phr = Counter(s for i in members for s in per_item[i] if skill2cat.get(s) == cid)
        c = catmap[cid]
        lines.append(f"category {cid}  (n_items={n})  ~ {c['name']}")
        lines.append(f"   definition: {c.get('definition','')}")
        lines.append("   subtasks: " + ", ".join(f"{s}:{v}" for s, v in subs.most_common(4)))
        lines.append("   top atomic skills: " + ", ".join(f"{s}({v})" for s, v in phr.most_common(6)))
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    json.dump({"categories": cats, "skill2cat": skill2cat},
              open(DATA / "stepwise_llm_cluster_map.json", "w"))
    print(f"\nwrote {args.out}", flush=True)
    print("\n".join(lines[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
