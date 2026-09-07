"""Pilot: running skill bank with retrieve-then-decide, vs independent labeling.

Isolates the memory effect: reasoning steps are held fixed (from
skills_stepwise_full.json); only the LABELING discipline changes. For each item,
in sequence, every reasoning step is shown ONLY the top-k most similar existing
bank skills (retrieve), and the model reuses one or creates a NEW one (decide).
This is the new pipeline's Step 2 (running memory, retrieve-then-decide). The old
pipeline's independent labels are already in the input file.

    # smoke (3 items per benchmark, pretty-printed, no save)
    .venv312/bin/python tools/pilot_skillbank.py --per 3 --smoke
    # full pilot (cap 200 per benchmark)
    .venv312/bin/python tools/pilot_skillbank.py --per 200 \
        --out cdm_exploration/experiments/pilot_skillbank_new.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "cdm_exploration/data/cdm_ready/skills_stepwise_full.json"
KEYFILE = Path.home() / ".cdmeval_openai_key"

SUBSET = [("MATH", "Algebra"), ("BBH", "web_of_lies"), ("GPQA", "Diamond"),
          ("MuSR", "Murder Mysteries"), ("IFEval", "IFEval")]

SYSTEM = """You are a psychometrician maintaining a SHARED, GROWING BANK of cognitive skills while labeling test items one at a time. For each reasoning step you are shown a short list of CANDIDATE skills already in the bank (the ones most similar to that step).

Label each step with the one atomic cognitive skill it requires.

Reuse rule: reuse a CANDIDATE only when the step requires the SAME cognitive operation. If no candidate genuinely fits, you MUST create a new skill, written as "NEW:snake_case_name" with a one-line definition. A WRONG reuse is worse than a new skill. Never force a step onto an unrelated candidate.

A skill names a specific cognitive OPERATION (good: isolate_the_variable, trace_truth_value_chain). Not a vague catch-all (bad: logical_reasoning, determine_implications, count_valid_solutions). Not item-specific (bad: solve_this_equation)."""

USER_TMPL = """ITEM [{bench}/{sub}] - label each reasoning step.

{steps}

Return JSON exactly: {{"labels": [{{"skill": "<a candidate name OR NEW:new_name>", "definition": "<one line, ONLY for NEW>"}}]}} with one entry per step, in order."""


def load_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    t = KEYFILE.read_text().strip()
    if t.startswith("OPENAI_API_KEY="):
        t = t.split("=", 1)[1].strip().strip('"').strip("'")
    return t


def norm(name: str) -> str:
    name = re.sub(r"^new\s*:\s*", "", name.strip(), flags=re.I)
    name = re.sub(r"[^a-z0-9_]", "", name.lower().strip().replace(" ", "_"))
    return re.sub(r"_+", "_", name).strip("_")


def select(rows, per):
    out = []
    for bench, sub in SUBSET:
        m = [r for r in rows if r["benchmark"] == bench and r["subtask"] == sub
             and r.get("reasoning_steps")]
        m.sort(key=lambda r: r["item_idx"])
        out.append((bench, sub, m[:per]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--topk", type=int, default=12, help="candidates shown per step")
    ap.add_argument("--per", type=int, default=200)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    client = OpenAI(api_key=load_key())
    print(f"loading embedder {args.embedder} ...", flush=True)
    emb = SentenceTransformer(args.embedder)

    rows = json.load(open(args.src))
    groups = select(rows, args.per)
    order = [(b, s, r) for b, s, items in groups for r in items]
    print("subset: " + ", ".join(f"{b}/{s}={len(it)}" for b, s, it in groups)
          + f"  |  total={len(order)} items  |  model={args.model}  topk={args.topk}",
          flush=True)

    bank: dict[str, str] = {}        # name -> definition
    bank_order: list[str] = []
    bank_vecs: list[np.ndarray] = []  # aligned with bank_order
    per_item, growth = [], []
    reuse = new = leak = 0
    calls = 0

    def add_skill(name, definition):
        bank[name] = definition
        bank_order.append(name)
        bank_vecs.append(emb.encode(f"{name}: {definition}", normalize_embeddings=True))

    def candidates_for(vecs):
        """Return list-of-lists: top-k bank skill names per step vector."""
        if not bank_vecs:
            return [[] for _ in range(len(vecs))]
        M = np.stack(bank_vecs)                       # (B, d)
        sims = vecs @ M.T                             # (S, B)
        out = []
        for row in sims:
            idx = np.argsort(-row)[: args.topk]
            out.append([bank_order[j] for j in idx])
        return out

    for i, (bench, sub, r) in enumerate(order):
        steps = [s.get("step", "") for s in r["reasoning_steps"] if isinstance(s, dict)]
        if not steps:
            continue
        svecs = emb.encode(steps, normalize_embeddings=True)
        cands = candidates_for(np.atleast_2d(svecs))
        lines = []
        for j, st in enumerate(steps):
            cs = "  |  ".join(f"{c}: {bank[c][:55]}" for c in cands[j]) or "(none yet)"
            lines.append(f"{j+1}. {st}\n   candidates: {cs}")
        user = USER_TMPL.format(bench=bench, sub=sub, steps="\n".join(lines))

        res = None
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                    temperature=0.0, response_format={"type": "json_object"})
                calls += 1
                res = json.loads(resp.choices[0].message.content)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"  ! item {r['item_idx']} failed: {str(e)[:120]}", flush=True)
                else:
                    time.sleep(2 * (attempt + 1))

        item_skills = []
        for lab in (res or {}).get("labels", []) or []:
            if not isinstance(lab, dict):
                continue
            raw = str(lab.get("skill", ""))
            is_new = raw.strip().lower().startswith("new:")
            name = norm(raw)
            if not name:
                continue
            if name in bank:
                reuse += 1
            elif is_new:
                add_skill(name, str(lab.get("definition", "")).strip() or "(no definition)")
                new += 1
            else:  # named a non-existent skill without NEW: -> leaked/renamed
                add_skill(name, "(leaked, no definition)")
                leak += 1
            item_skills.append(name)
        per_item.append({"item_idx": r["item_idx"], "benchmark": bench,
                         "subtask": sub, "skills": item_skills})
        growth.append(len(bank))
        if args.smoke:
            print(f"\n[{i+1}] {bench}/{sub} item {r['item_idx']}  (bank={len(bank)})")
            for nm in item_skills:
                print(f"    {nm}")
        elif (i + 1) % 50 == 0:
            recent = growth[-50] if len(growth) > 50 else 0
            print(f"  {i+1}/{len(order)}  bank={len(bank)}  (+{len(bank)-recent}/50)",
                  flush=True)

    tot = reuse + new + leak
    print("\n=== running-bank (retrieve-then-decide) summary ===")
    print(f"items={len(per_item)}  api_calls={calls}  skill-labels={tot}")
    print(f"bank size (unique skills) = {len(bank)}")
    print(f"reuse={reuse} ({reuse/max(tot,1):.0%})  new={new}  leaked={leak}")
    if len(growth) > 100:
        print(f"growth: after 100 items={growth[99]}, final={len(bank)}, "
              f"added in last 100={len(bank)-growth[-100]}")

    if args.out and not args.smoke:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"subset": [[b, s, len(it)] for b, s, it in groups],
                   "model": args.model, "embedder": args.embedder, "topk": args.topk,
                   "n_items": len(per_item),
                   "bank": [{"name": n, "definition": bank[n], "rank": k}
                            for k, n in enumerate(bank_order)],
                   "growth_curve": growth, "per_item": per_item,
                   "counts": {"reuse": reuse, "new": new, "leaked": leak,
                              "api_calls": calls}},
                  open(args.out, "w"))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
