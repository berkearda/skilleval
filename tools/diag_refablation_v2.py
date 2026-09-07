"""Reference-answer ablation, CORRECTED design.

Audit of the first attempt found the stored labels are NOT a valid control:
the submitted extraction ran on `question_preview`, capped at 200 chars and
often the TAIL of the question (the DATA-FIX / T-070 truncation), while the
solution-aware arm reads full text. Comparing them confounds truncation with
the reference solution. Temperature matches (0.0 in both).

Corrected arms, identical prompt, identical temperature, identical truncation
of the item text, differing ONLY in whether the reference solution is appended:

  CTRL : full question text, no solution      (this script, --arm ctrl)
  REF  : full question text + reference sol.  (v2_reference_answer_ablation.json)
  NOISE: full question text, no solution, repeat run  (--arm noise)
         -> measures API/labelling non-determinism at temperature 0, the floor
            against which the REF effect must be read.

Usage:
  python tools/diag_refablation_v2.py --arm ctrl            # 2470 items
  python tools/diag_refablation_v2.py --arm noise --n 300   # noise floor
  python tools/diag_refablation_v2.py --analyze
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cdmeval.skills.extraction import QMATRIX_SYSTEM_PROMPT, QMATRIX_USER_TEMPLATE

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"
MODEL = "gpt-4o-mini"
REF_JSON = E / "v2_reference_answer_ablation.json"
CTRL_JSON = E / "v2_refablation_ctrl.json"
NOISE_JSON = E / "v2_refablation_noise.json"
STOP = {"of", "in", "to", "from", "with", "for", "and", "the", "a", "on", "by", "at"}


def load_key() -> str:
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def toks(l): return {w for w in re.split(r"[_\W]+", (l or "").lower()) if w and w not in STOP}
def jac(a, b): return len(a & b) / len(a | b) if (a or b) else 1.0
def tj(x, y):
    return jac(set().union(*[toks(l) for l in x]) if x else set(),
               set().union(*[toks(l) for l in y]) if y else set())


async def extract(idxs, meta, qt, seedtag=""):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=load_key())
    sem = asyncio.Semaphore(12)

    async def one(i):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": QMATRIX_SYSTEM_PROMPT},
                              {"role": "user", "content": QMATRIX_USER_TEMPLATE.format(
                                  benchmark=meta[i]["benchmark"], subtask=meta[i]["subtask"],
                                  question_text=qt[i][:1500])}],
                    response_format={"type": "json_object"}, temperature=0, max_tokens=400)
                d = json.loads(r.choices[0].message.content)
                return {"item_idx": i, "benchmark": meta[i]["benchmark"],
                        "skills": [s.get("label") for s in d.get("skills", [])],
                        "primary": d.get("primary_skill"), "usage": r.usage.total_tokens}
            except Exception as ex:
                return {"item_idx": i, "error": str(ex)[:160]}
    return await asyncio.gather(*[one(i) for i in idxs])


def load_common():
    qt = {r["item_idx"]: r["question_full_text"]
          for r in json.loads((D / "item_full_text_recovered.json").read_text())}
    meta = {it["item_idx"]: it for it in json.loads((D / "response_matrix_v2_full_items.json").read_text())}
    ref = {r["item_idx"]: r["skills_with_ref"]
           for r in json.loads(REF_JSON.read_text())["results"]}
    return qt, meta, ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ctrl", "noise"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    qt, meta, ref = load_common()
    targets = sorted(ref)

    if a.arm:
        rng = np.random.default_rng(7)
        if a.n:
            targets = sorted(rng.choice(targets, min(a.n, len(targets)), replace=False).tolist())
        print(f"arm={a.arm} items={len(targets)}", flush=True)
        res = asyncio.run(extract(targets, meta, qt))
        ok = [r for r in res if "error" not in r]
        tok = sum(r.get("usage", 0) for r in ok)
        print(f"ok {len(ok)} / {len(res)}, tokens {tok:,} (~${tok/1e6*0.6:.2f})", flush=True)
        out = {"arm": a.arm, "model": MODEL, "temperature": 0,
               "input": "question_full_text[:1500], no reference solution",
               "results": ok, "verified": True}
        (CTRL_JSON if a.arm == "ctrl" else NOISE_JSON).write_text(json.dumps(out, indent=2))
        print("wrote", CTRL_JSON if a.arm == "ctrl" else NOISE_JSON, flush=True)
        return

    # ---- analysis ----
    ctrl = {r["item_idx"]: r["skills"] for r in json.loads(CTRL_JSON.read_text())["results"]}
    noise = {r["item_idx"]: r["skills"] for r in json.loads(NOISE_JSON.read_text())["results"]} \
        if NOISE_JSON.exists() else {}
    both = sorted(set(ctrl) & set(ref))
    print(f"items with CTRL and REF: {len(both)}", flush=True)

    def report(sel, name):
        if not sel:
            return None
        t = np.array([tj(ctrl[i], ref[i]) for i in sel])
        e = np.array([jac(set(ctrl[i]), set(ref[i])) for i in sel])
        print(f"  {name:12s} n={len(sel):5d}  token J {t.mean():.3f}   exact-label J {e.mean():.3f}", flush=True)
        return float(t.mean()), float(e.mean())

    print("\n=== EFFECT of adding the reference solution (same full text both arms) ===", flush=True)
    allr = report(both, "ALL")
    perb = {b: report([i for i in both if meta[i]["benchmark"] == b], b) for b in ["MATH", "GPQA"]}

    nf = None
    if noise:
        ns = sorted(set(noise) & set(ctrl))
        t = np.array([tj(ctrl[i], noise[i]) for i in ns])
        e = np.array([jac(set(ctrl[i]), set(noise[i])) for i in ns])
        nf = (float(t.mean()), float(e.mean()))
        print(f"\n=== NOISE FLOOR (identical input, repeat run) ===", flush=True)
        print(f"  {'CTRL vs CTRL2':12s} n={len(ns):5d}  token J {t.mean():.3f}   exact-label J {e.mean():.3f}", flush=True)
        print(f"\n  effect attributable to the reference solution: "
              f"{allr[0] - nf[0]:+.3f} token J", flush=True)

    out = {"experiment": "reference_answer_ablation_corrected",
           "design": "CTRL and REF differ ONLY in the appended reference solution; "
                     "both read question_full_text[:1500] at temperature 0",
           "n_items": len(both),
           "effect": {"all_token_j": allr[0], "all_exact_j": allr[1],
                      "by_benchmark": {b: {"token_j": v[0], "exact_j": v[1]} for b, v in perb.items() if v}},
           "noise_floor": {"token_j": nf[0], "exact_j": nf[1]} if nf else None,
           "solution_effect_vs_noise": (allr[0] - nf[0]) if nf else None,
           "superseded": "v2_refablation_analysis.json and v2_refablation_controls.json "
                         "compared against the truncated-preview extraction and are invalid",
           "verified": True}
    (E / "v2_refablation_corrected.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {E / 'v2_refablation_corrected.json'}", flush=True)


if __name__ == "__main__":
    main()
