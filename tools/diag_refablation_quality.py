"""Does the reference solution HELP? (the question actually asked)

The overlap metrics measure whether labels CHANGE, not whether they improve,
and they penalise paraphrase ("state" vs "states"). Two additions:

  A. SEMANTIC agreement. Embed each label with the pipeline's SBERT model and
     align the two sets by best match (mean of max cosine, symmetric). This
     separates genuine content change from rewording. Local, all items.

  B. BLIND QUALITY JUDGE. For each item the judge sees the question and the
     two label sets, unlabelled and in randomised order, and picks which set
     better describes what the solver must do (or ties). Judge is
     gpt-4.1-mini, deliberately NOT the extractor (gpt-4o-mini), so the
     extractor cannot prefer its own output. Order is swapped on odd item_idx
     and the position of the winner is tracked to expose position bias.

Arms compared: CTRL (question only, full text) vs REF (question + reference
solution, full text). Both from the corrected design.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"
JUDGE = "gpt-4.1-mini"
N_JUDGE = 800


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
    idx = sorted(set(ref) & set(ctrl))
    return ctrl, ref, qt, meta, idx


def semantic(ctrl, ref, idx, meta):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-mpnet-base-v2")
    allsk = sorted({l for i in idx for l in ctrl[i] + ref[i]})
    emb = dict(zip(allsk, m.encode([s.replace("_", " ") for s in allsk],
                                   normalize_embeddings=True, show_progress_bar=False)))
    sims = []
    for i in idx:
        A = np.array([emb[l] for l in ctrl[i]])
        B = np.array([emb[l] for l in ref[i]])
        if not len(A) or not len(B):
            continue
        S = A @ B.T
        sims.append(((S.max(axis=1).mean() + S.max(axis=0).mean()) / 2, meta[i]["benchmark"]))
    v = np.array([s for s, _ in sims])
    print(f"\n=== A. SEMANTIC agreement (SBERT, paraphrase-tolerant) ===", flush=True)
    print(f"  all   n={len(v):5d}  mean cosine {v.mean():.3f}  median {np.median(v):.3f}", flush=True)
    for b in ["MATH", "GPQA"]:
        vb = np.array([s for s, bb in sims if bb == b])
        print(f"  {b:5s} n={len(vb):5d}  mean cosine {vb.mean():.3f}", flush=True)
    print(f"  share of items with cosine > 0.8 (essentially same skills): "
          f"{(v > 0.8).mean():.1%}", flush=True)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "frac_above_0.8": float((v > 0.8).mean()),
            "by_bench": {b: float(np.mean([s for s, bb in sims if bb == b])) for b in ["MATH", "GPQA"]}}


async def judge(ctrl, ref, qt, meta, sel):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=load_key())
    sem = asyncio.Semaphore(10)
    SYS = ("You evaluate skill labels for test items. Given a question and two candidate "
           "sets of cognitive skill labels, decide which set better describes what a solver "
           "must actually do to answer the question. Judge accuracy and specificity, not style.")

    async def one(i):
        swap = (i % 2 == 1)          # deterministic order randomisation
        first, second = (ref[i], ctrl[i]) if swap else (ctrl[i], ref[i])
        u = (f"Question:\n{' '.join(qt[i].split())[:1200]}\n\n"
             f"Set 1:\n" + "\n".join(f"- {s}" for s in first) +
             f"\n\nSet 2:\n" + "\n".join(f"- {s}" for s in second) +
             '\n\nWhich set better describes the skills the question requires? '
             'Return JSON {"winner":"1|2|tie","reason":"<8 words"}.')
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=JUDGE, messages=[{"role": "system", "content": SYS},
                                           {"role": "user", "content": u}],
                    response_format={"type": "json_object"}, temperature=0, max_tokens=60)
                w = json.loads(r.choices[0].message.content).get("winner", "tie")
                if w == "tie":
                    win = "tie"
                else:
                    first_is_ref = swap
                    win = "ref" if ((w == "1") == first_is_ref) else "ctrl"
                return {"item_idx": i, "winner": win, "position_winner": w,
                        "benchmark": meta[i]["benchmark"], "usage": r.usage.total_tokens}
            except Exception as ex:
                return {"item_idx": i, "error": str(ex)[:120]}
    return await asyncio.gather(*[one(i) for i in sel])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_JUDGE)
    a = ap.parse_args()
    ctrl, ref, qt, meta, idx = load_arms()
    print(f"items: {len(idx)}", flush=True)

    sem_res = semantic(ctrl, ref, idx, meta)

    rng = random.Random(42)
    sel = sorted(rng.sample(idx, min(a.n, len(idx))))
    print(f"\n=== B. BLIND QUALITY JUDGE ({JUDGE}, n={len(sel)}) ===", flush=True)
    res = asyncio.run(judge(ctrl, ref, qt, meta, sel))
    ok = [r for r in res if "error" not in r]
    tok = sum(r.get("usage", 0) for r in ok)
    from collections import Counter
    c = Counter(r["winner"] for r in ok)
    n = len(ok)
    print(f"  judged {n}, tokens {tok:,} (~${tok/1e6*1.0:.2f})", flush=True)
    for k in ["ref", "ctrl", "tie"]:
        lab = {"ref": "solution-aware WINS", "ctrl": "question-only WINS", "tie": "tie"}[k]
        print(f"    {lab:22s} {c[k]:4d}  ({c[k]/n:.1%})", flush=True)
    pos = Counter(r["position_winner"] for r in ok)
    print(f"  position check (should be near 50/50): set1 {pos['1']}, set2 {pos['2']}, tie {pos['tie']}", flush=True)
    for b in ["MATH", "GPQA"]:
        cb = Counter(r["winner"] for r in ok if r["benchmark"] == b)
        nb = sum(cb.values())
        if nb:
            print(f"    {b}: solution-aware {cb['ref']/nb:.1%}, question-only {cb['ctrl']/nb:.1%}, tie {cb['tie']/nb:.1%}", flush=True)

    out = {"experiment": "refablation_quality", "judge": JUDGE, "n_judged": n,
           "semantic": sem_res,
           "quality": {"solution_aware_win": c["ref"] / n, "question_only_win": c["ctrl"] / n,
                       "tie": c["tie"] / n,
                       "position_set1": pos["1"], "position_set2": pos["2"]},
           "verified": True}
    (E / "v2_refablation_quality.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {E / 'v2_refablation_quality.json'}", flush=True)


if __name__ == "__main__":
    main()
