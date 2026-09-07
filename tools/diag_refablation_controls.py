"""Controls for the reference-answer ablation. Run BEFORE reporting.

Control 1 (noise floor): re-run the ORIGINAL question-only extraction with the
identical prompt on a subset. Free-form snake_case labels vary between runs by
construction, so the arm-A vs arm-B agreement is only interpretable relative to
the arm-A vs arm-A' agreement.

Control 2 (metric sanity): apply the same nearest-centroid cluster assignment to
arm-A phrases. If arm A also "changes" clusters at a high rate, the reassignment
metric measures my approximation, not the reference solution's effect, and must
not be reported.
"""
from __future__ import annotations
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
N_SUBSET = 400
STOP = {"of", "in", "to", "from", "with", "for", "and", "the", "a", "on", "by", "at"}


def load_key() -> str:
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def toks(l): return {w for w in re.split(r"[_\W]+", (l or "").lower()) if w and w not in STOP}
def jac(a, b): return len(a & b) / len(a | b) if (a or b) else 1.0


async def reextract(idxs, meta, qt):
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
                return i, [s.get("label") for s in d.get("skills", [])]
            except Exception as ex:
                return i, None
    return dict(await asyncio.gather(*[one(i) for i in idxs]))


def main() -> None:
    rng = np.random.default_rng(42)
    armB = {r["item_idx"]: r["skills_with_ref"]
            for r in json.loads((E / "v2_reference_answer_ablation.json").read_text())["results"]}
    armA = {r["item_idx"]: r["skills"]
            for r in json.loads((D / "skills_v2_new_prompt.json").read_text())}
    qt = {r["item_idx"]: r["question_full_text"]
          for r in json.loads((D / "item_full_text_recovered.json").read_text())}
    meta = {it["item_idx"]: it for it in json.loads((D / "response_matrix_v2_full_items.json").read_text())}

    shared = sorted(set(armA) & set(armB))
    subset = sorted(rng.choice(shared, min(N_SUBSET, len(shared)), replace=False).tolist())
    print(f"control subset: {len(subset)} items", flush=True)

    print("\n=== Control 1: question-only re-run (noise floor) ===", flush=True)
    armA2 = asyncio.run(reextract(subset, meta, qt))
    ok = [i for i in subset if armA2.get(i)]
    def tj(x, y): return jac(set().union(*[toks(l) for l in x]) if x else set(),
                             set().union(*[toks(l) for l in y]) if y else set())
    noise = np.array([tj(armA[i], armA2[i]) for i in ok])
    effect = np.array([tj(armA[i], armB[i]) for i in ok])
    ex_noise = np.array([jac(set(armA[i]), set(armA2[i])) for i in ok])
    ex_eff = np.array([jac(set(armA[i]), set(armB[i])) for i in ok])
    print(f"  A vs A' (same prompt, no solution): token J {noise.mean():.3f}, exact J {ex_noise.mean():.3f}", flush=True)
    print(f"  A vs B  (solution added)          : token J {effect.mean():.3f}, exact J {ex_eff.mean():.3f}", flush=True)
    delta = noise.mean() - effect.mean()
    print(f"  DIFFERENCE attributable to the reference solution: {delta:+.3f} token J", flush=True)

    print("\n=== Control 2: cluster-reassignment metric sanity ===", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    q = np.load(D / "qmatrix_v2_K100.npy")
    lab = json.loads((D / "cluster_labels_v2_K100.json").read_text())
    centro = model.encode([lab[str(k)] for k in range(100)], normalize_embeddings=True, show_progress_bar=False)

    def change_rate(src):
        ch = 0
        for i in ok:
            L = [l.replace("_", " ") for l in (src.get(i) or []) if l]
            if not L:
                continue
            eb = model.encode(L, normalize_embeddings=True, show_progress_bar=False)
            if set(np.argmax(eb @ centro.T, axis=1).tolist()) != set(np.nonzero(q[i])[0].tolist()):
                ch += 1
        return ch / len(ok)

    ra, rb = change_rate(armA), change_rate(armB)
    print(f"  arm A (the phrases that BUILT the Q-matrix): {ra:.1%} 'changed'", flush=True)
    print(f"  arm B (solution-aware)                     : {rb:.1%} 'changed'", flush=True)
    valid = ra < 0.25
    print(f"  metric usable: {valid}  ({'ok' if valid else 'NO - reassignment approximation does not reproduce the original clustering, do not report cluster-change numbers'})", flush=True)

    out = {"experiment": "refablation_controls", "n_subset": len(ok),
           "noise_floor_token_jaccard": float(noise.mean()),
           "effect_token_jaccard": float(effect.mean()),
           "difference": float(delta),
           "noise_floor_exact": float(ex_noise.mean()), "effect_exact": float(ex_eff.mean()),
           "cluster_change_armA": float(ra), "cluster_change_armB": float(rb),
           "cluster_metric_valid": bool(valid), "verified": True}
    (E / "v2_refablation_controls.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {E / 'v2_refablation_controls.json'}", flush=True)


if __name__ == "__main__":
    main()
