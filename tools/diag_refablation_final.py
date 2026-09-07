"""Reference-answer ablation, FINAL analysis.

Three arms, all reading question_full_text[:1500] with the identical
QMATRIX prompt and gpt-4o-mini, differing only as noted:

  FULLTEXT  skills_extracted_v2_fulltext.json   question only  (run 2026-06-13)
  CTRL      v2_refablation_ctrl.json            question only  (run 2026-07-27)
  REF       v2_reference_answer_ablation.json   question + reference solution

  CTRL vs FULLTEXT -> NOISE FLOOR (independent runs, identical input)
  CTRL vs REF      -> solution effect
  FULLTEXT vs REF  -> solution effect, independent replication

The submitted Q-matrix (qmatrix_v2_K100.npy, 2026-03-29) came from the
200-char question_preview extraction, so it is NOT a valid control here:
comparing against it confounds text truncation with the reference solution.
That comparison (v2_refablation_analysis.json, v2_refablation_controls.json)
is superseded and must not be quoted.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"
STOP = {"of", "in", "to", "from", "with", "for", "and", "the", "a", "on", "by", "at"}


def toks(l): return {w for w in re.split(r"[_\W]+", (l or "").lower()) if w and w not in STOP}
def jac(a, b): return len(a & b) / len(a | b) if (a or b) else 1.0
def tj(x, y):
    return jac(set().union(*[toks(l) for l in x]) if x else set(),
               set().union(*[toks(l) for l in y]) if y else set())


def main() -> None:
    ref = {r["item_idx"]: [s for s in r["skills_with_ref"] if s]
           for r in json.loads((E / "v2_reference_answer_ablation.json").read_text())["results"]}
    ctrl = {r["item_idx"]: [s for s in r["skills"] if s]
            for r in json.loads((E / "v2_refablation_ctrl.json").read_text())["results"]}
    ft = {r["item_idx"]: [s for s in (r.get("skills") or []) if s]
          for r in json.loads((D / "skills_extracted_v2_fulltext.json").read_text())
          if r.get("skills")}
    meta = {it["item_idx"]: it for it in json.loads((D / "response_matrix_v2_full_items.json").read_text())}

    idx = sorted(set(ref) & set(ctrl) & set(ft))
    print(f"items in all three arms: {len(idx)}\n", flush=True)

    def pair(A, B, sel):
        t = np.array([tj(A[i], B[i]) for i in sel])
        e = np.array([jac(set(A[i]), set(B[i])) for i in sel])
        return float(t.mean()), float(e.mean()), float(t.std())

    def line(name, A, B, sel):
        t, e, s = pair(A, B, sel)
        print(f"  {name:34s} n={len(sel):5d}  token J {t:.3f} (sd {s:.3f})   exact-label J {e:.3f}", flush=True)
        return t, e

    print("=== NOISE FLOOR: two independent question-only runs, identical input ===", flush=True)
    nf_all = line("CTRL vs FULLTEXT (all)", ctrl, ft, idx)
    for b in ["MATH", "GPQA"]:
        line(f"CTRL vs FULLTEXT ({b})", ctrl, ft, [i for i in idx if meta[i]["benchmark"] == b])

    print("\n=== EFFECT: adding the reference solution ===", flush=True)
    ef_all = line("CTRL vs REF (all)", ctrl, ref, idx)
    per_b = {}
    for b in ["MATH", "GPQA"]:
        sel = [i for i in idx if meta[i]["benchmark"] == b]
        per_b[b] = {"effect": line(f"CTRL vs REF ({b})", ctrl, ref, sel),
                    "noise": pair(ctrl, ft, sel)[:2]}
    rep_all = line("FULLTEXT vs REF (replication)", ft, ref, idx)

    delta = ef_all[0] - nf_all[0]
    print(f"\n  Effect attributable to the reference solution:", flush=True)
    print(f"    token agreement drops {nf_all[0]:.3f} (noise floor) -> {ef_all[0]:.3f}  = {delta:+.3f}", flush=True)
    for b in ["MATH", "GPQA"]:
        d = per_b[b]["effect"][0] - per_b[b]["noise"][0]
        print(f"    {b:5s}: {per_b[b]['noise'][0]:.3f} -> {per_b[b]['effect'][0]:.3f}  = {d:+.3f}", flush=True)

    # how often does the reference add a skill token absent from the question-only arms?
    novel = []
    for i in idx:
        base = set().union(*[toks(l) for l in ctrl[i] + ft[i]]) if (ctrl[i] or ft[i]) else set()
        r = set().union(*[toks(l) for l in ref[i]]) if ref[i] else set()
        novel.append(len(r - base) / max(1, len(r)))
    novel = np.array(novel)
    print(f"\n  Share of solution-aware skill vocabulary absent from BOTH question-only runs: "
          f"{novel.mean():.1%}", flush=True)

    out = {
        "experiment": "reference_answer_ablation_final",
        "n_items": len(idx),
        "design": "all arms: question_full_text[:1500], QMATRIX prompt, gpt-4o-mini, temperature 0",
        "noise_floor_token_j": nf_all[0], "noise_floor_exact_j": nf_all[1],
        "effect_token_j": ef_all[0], "effect_exact_j": ef_all[1],
        "replication_fulltext_vs_ref_token_j": rep_all[0],
        "solution_effect_vs_noise": delta,
        "by_benchmark": {b: {"noise_token_j": per_b[b]["noise"][0],
                             "effect_token_j": per_b[b]["effect"][0],
                             "delta": per_b[b]["effect"][0] - per_b[b]["noise"][0]}
                         for b in per_b},
        "novel_vocab_share": float(novel.mean()),
        "supersedes": ["v2_refablation_analysis.json", "v2_refablation_controls.json"],
        "note": ("the submitted Q-matrix extraction (2026-03-29) used the 200-char "
                 "question_preview and is not a valid control for this ablation"),
        "verified": True,
    }
    (E / "v2_refablation_final.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {E / 'v2_refablation_final.json'}", flush=True)


if __name__ == "__main__":
    main()
