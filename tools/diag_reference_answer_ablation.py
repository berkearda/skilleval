"""Reference-answer extraction ablation.

Question: does giving the skill extractor the reference solution change the
extracted skills, and where?

Design: strict A/B. Arm A is the ORIGINAL question-only extraction already in
the paper (reused from the stored labels, no re-run). Arm B uses the IDENTICAL
system prompt and user template, with one block appended to the item text:
the gold solution (MATH) or the question-writer explanation (GPQA). Only the
added information differs, so any label change is attributable to it.

Sources (both verified 100% matched to our items):
  MATH  1,278 items  -> cdm_exploration/data/math_processed.csv 'solution'
  GPQA  1,192 items  -> HF Idavidrein/gpqa gpqa_extended 'Explanation'

Outputs per item: arm-B skill labels + primary skill, stored alongside arm A
for downstream agreement analysis (label overlap, cluster-assignment change).

Usage:
  python tools/diag_reference_answer_ablation.py --smoke        # 5 items, B1 gate
  python tools/diag_reference_answer_ablation.py --bench MATH   # full arm
  python tools/diag_reference_answer_ablation.py                # both
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cdmeval.skills.extraction import QMATRIX_SYSTEM_PROMPT, QMATRIX_USER_TEMPLATE

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"
MODEL = "gpt-4o-mini"
CONCURRENCY = 12
OUT = E / "v2_reference_answer_ablation.json"


def load_key() -> str:
    p = Path.home() / ".cdmeval_openai_key"
    t = p.read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def norm(s: str) -> str:
    s = re.sub(r"^.*?correct answer to this question:", "", s, flags=re.I | re.S)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_targets(bench_filter: str | None):
    items = json.load(open(D / "response_matrix_v2_full_items.json"))
    qt = {r["item_idx"]: r["question_full_text"]
          for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {it["item_idx"]: it for it in items}

    refs: dict[int, str] = {}

    # MATH gold solutions
    if bench_filter in (None, "MATH"):
        sol = {}
        for r in csv.DictReader(open(REPO / "cdm_exploration" / "data" / "math_processed.csv")):
            sol[re.sub(r"\s+", " ", r["problem"]).strip()[:200]] = r["solution"]
        for i, it in meta.items():
            if it["benchmark"] != "MATH":
                continue
            k = re.sub(r"\s+", " ", qt.get(i, "")).strip()[:200]
            if k in sol:
                refs[i] = sol[k]

    # GPQA writer explanations
    if bench_filter in (None, "GPQA"):
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_extended", split="train")
        pairs = [(norm(r["Question"]), (r.get("Explanation") or "").strip())
                 for r in ds if (r.get("Explanation") or "").strip()]
        for i, it in meta.items():
            if it["benchmark"] != "GPQA":
                continue
            n = norm(qt.get(i, ""))
            key = n[:150]
            for q, ex in pairs:
                if key and (key in q or q[:150] in n):
                    refs[i] = ex
                    break

    return meta, qt, refs


async def run(targets, meta, qt, refs, limit=None):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=load_key())
    sem = asyncio.Semaphore(CONCURRENCY)
    if limit:
        targets = targets[:limit]

    async def one(i):
        it = meta[i]
        # IDENTICAL template; the reference block is appended to the item text.
        qtext = qt[i][:1500] + "\n\nReference solution:\n" + refs[i][:1500]
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": QMATRIX_SYSTEM_PROMPT},
                        {"role": "user", "content": QMATRIX_USER_TEMPLATE.format(
                            benchmark=it["benchmark"], subtask=it["subtask"],
                            question_text=qtext)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=400,
                )
                d = json.loads(r.choices[0].message.content)
                return {"item_idx": i, "benchmark": it["benchmark"],
                        "subtask": it["subtask"],
                        "skills_with_ref": [s.get("label") for s in d.get("skills", [])],
                        "primary_with_ref": d.get("primary_skill"),
                        "usage": r.usage.total_tokens}
            except Exception as ex:
                return {"item_idx": i, "error": str(ex)[:200]}

    return await asyncio.gather(*[one(i) for i in targets])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", choices=["MATH", "GPQA"], default=None)
    a = ap.parse_args()

    meta, qt, refs = build_targets(a.bench)
    targets = sorted(refs)
    from collections import Counter
    print(f"items with reference text: {len(targets)} "
          f"{dict(Counter(meta[i]['benchmark'] for i in targets))}", flush=True)

    if a.smoke:
        # B1: 5 items spanning both benchmarks, full code path
        mt = [i for i in targets if meta[i]["benchmark"] == "MATH"][:3]
        gp = [i for i in targets if meta[i]["benchmark"] == "GPQA"][:2]
        targets = mt + gp
        print(f"SMOKE on {targets}", flush=True)

    res = asyncio.run(run(targets, meta, qt, refs))
    ok = [r for r in res if "error" not in r]
    err = [r for r in res if "error" in r]
    tok = sum(r.get("usage", 0) for r in ok)
    print(f"\nok {len(ok)}, errors {len(err)}, tokens {tok:,} "
          f"(~${tok/1e6*0.6:.2f} at 4o-mini blended)", flush=True)
    for r in err[:3]:
        print("ERR:", r, flush=True)

    if a.smoke:
        for r in ok:
            print(f"\n--- item {r['item_idx']} [{r['benchmark']}/{r['subtask']}]", flush=True)
            print("  question:", " ".join(qt[r["item_idx"]].split())[:160], flush=True)
            print("  reference:", " ".join(refs[r["item_idx"]].split())[:160], flush=True)
            print("  skills WITH ref:", r["skills_with_ref"], flush=True)
            print("  primary:", r["primary_with_ref"], flush=True)
        return

    prev = json.loads(OUT.read_text()) if OUT.exists() else {"results": []}
    seen = {x["item_idx"] for x in prev["results"]}
    prev["results"] += [r for r in ok if r["item_idx"] not in seen]
    prev["model"] = MODEL
    prev["design"] = "identical prompt, reference solution appended to item text"
    prev["verified"] = True
    OUT.write_text(json.dumps(prev, indent=2))
    print(f"wrote {OUT} ({len(prev['results'])} items)", flush=True)


if __name__ == "__main__":
    main()
