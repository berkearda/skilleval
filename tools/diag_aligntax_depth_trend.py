"""Interrogate the alignment-tax depth gradient before conceding it.

Critique: "the 'alignment hurts the deepest skills' claim rests on
bins of 3 and 7 near-floor skills, which is noise."

The submitted figure compares per-depth bin means, which is exactly as
fragile as the critique says. The stronger test is a continuous trend over
ALL 100 skills, which does not depend on binning:

  1. Spearman rank correlation between skill depth and per-skill alignment
     delta, with a 10,000-round permutation p-value.
  2. Floor control: deep skills may sit near floor for base models, which
     could distort deltas. Recompute as a partial rank correlation
     controlling for the skills' mean base-model accuracy.
  3. Bin-robustness: repeat the trend on depths 0-3 only (the well
     populated bins). If the decline shows there too, it does not rest on
     the tiny deep bins.

Decision rule (stated before running): if the permutation p for the full
trend is >= 0.05, or the sign flips under the floor control, the drafted
concession stands unchanged. Otherwise the claim is upgraded to
the trend statistic and still concedes the bin phrasing.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "cdm_exploration" / "experiments"
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"

N_PERM = 10_000


def rank(v: np.ndarray) -> np.ndarray:
    r = np.empty_like(v, dtype=float)
    r[np.argsort(v, kind="stable")] = np.arange(len(v), dtype=float)
    return r


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = rank(x), rank(y), rank(z)
    Zc = np.column_stack([np.ones(len(z)), rz])
    ex = rx - Zc @ np.linalg.lstsq(Zc, rx, rcond=None)[0]
    ey = ry - Zc @ np.linalg.lstsq(Zc, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def perm_p(x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    """One-sided p for a negative trend."""
    count = 0
    for _ in range(N_PERM):
        r, _ = spearmanr(x, rng.permutation(y))
        if r <= observed:
            count += 1
    return count / N_PERM


def main() -> None:
    # --tag _predicted050 etc.: depths from the co-mastery graph built with that mastery definition
    # (tools/run_skill_prerequisites.py +mastery=..., an external review, point 2); output gets the same suffix.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    tag = ap.parse_args().tag
    rng = np.random.default_rng(42)
    align = json.loads((EXP / "v2_alignment_tax.json").read_text())
    prereq = json.loads((EXP / f"v2_skill_prerequisites{tag}.json").read_text())

    K = int(align["K"])
    depth = np.array([int(prereq["depth_per_skill"][str(k)]) for k in range(K)])
    delta = np.array([align["per_skill"][str(k)]["mean_delta"] for k in range(K)])

    from collections import Counter
    hist = sorted(Counter(depth.tolist()).items())
    print("depth histogram (depth, n_skills):", hist, flush=True)

    # 1. Full continuous trend
    rho, p_asym = spearmanr(depth, delta)
    p_perm = perm_p(depth, delta, rho, rng)
    print(f"\n1. FULL TREND (all {K} skills): Spearman rho = {rho:+.3f}, "
          f"permutation p (one-sided, negative) = {p_perm:.4f}", flush=True)

    # 2. Floor control: mean base-model accuracy per skill
    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    llms = json.loads((DATA / "response_matrix_v2_full_llms.json").read_text())
    name_to_idx = {}
    for i, entry in enumerate(llms):
        nm = entry if isinstance(entry, str) else entry.get("name", entry.get("llm", str(entry)))
        name_to_idx[nm] = i
    base_rows = [name_to_idx[p["base"]] for p in align["pairs"] if p["base"] in name_to_idx]
    print(f"\nbase models mapped: {len(base_rows)}/{align['n_pairs']}", flush=True)
    base_acc = np.zeros(K)
    for k in range(K):
        items = np.where(Q[:, k] > 0)[0]
        base_acc[k] = R[np.ix_(base_rows, items)].mean()
    r_depth_floor, _ = spearmanr(depth, base_acc)
    print(f"   depth vs base accuracy: rho = {r_depth_floor:+.3f} "
          f"(negative = deeper skills are harder, the floor concern)", flush=True)
    rho_partial = partial_spearman(depth, delta, base_acc)
    print(f"2. FLOOR CONTROL: partial Spearman(depth, delta | base_acc) = {rho_partial:+.3f}", flush=True)

    # 3. Depths 0-3 only (well-populated bins)
    m = depth <= 3
    rho03, _ = spearmanr(depth[m], delta[m])
    p03 = perm_p(depth[m], delta[m], rho03, np.random.default_rng(43))
    print(f"3. DEPTHS 0-3 ONLY (n={m.sum()}): rho = {rho03:+.3f}, permutation p = {p03:.4f}", flush=True)

    verdict = "TREND HOLDS" if (p_perm < 0.05 and rho_partial < 0 and p03 < 0.10) else \
              "TREND WEAK OR BIN-DRIVEN - concede as drafted"
    print(f"\nVERDICT: {verdict}", flush=True)

    out = {
        "experiment": "aligntax_depth_trend_test",
        "depth_histogram": hist,
        "full_trend": {"spearman_rho": float(rho), "perm_p_one_sided": float(p_perm), "n_perm": N_PERM},
        "floor_control": {
            "depth_vs_base_acc_rho": float(r_depth_floor),
            "partial_rho_given_base_acc": float(rho_partial),
            "n_base_models": len(base_rows),
        },
        "depths_0_3": {"spearman_rho": float(rho03), "perm_p": float(p03), "n_skills": int(m.sum())},
        "verdict": verdict,
        "verified": True,
    }
    if tag:
        out["prerequisites_file"] = f"v2_skill_prerequisites{tag}.json"
    (EXP / f"v2_aligntax_depth_trend{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {EXP / f'v2_aligntax_depth_trend{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
