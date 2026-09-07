"""Construct validity for CDMEval θ.

The question: show that θ on skill k actually measures the LLM's accuracy on
items requiring skill k.

For each (LLM m, skill k) pair:
  - x = θ[m, k] (sigmoid of student_emb logit)
  - y = mean accuracy of LLM m on items where Q[i, k] = 1

Then:
  - Per-skill Pearson r across the 3,811 LLMs (one r per skill)
  - Aggregate distribution of these per-skill r values
  - Headline: median per-skill r, fraction of skills with r > 0.5 / 0.7 / 0.9
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"
CKPT = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"


def main() -> None:
    print("=== Construct validity: theta vs per-skill accuracy ===\n", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    q_matrix = np.load(DATA / "qmatrix_v2_K100.npy")
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]

    # Load NCDM checkpoint, extract sigmoid(student_emb) = theta in [0, 1]
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    student_logits = sd["student_emb.weight"]
    theta = torch.sigmoid(student_logits).numpy()  # (n_llms, K)
    print(f"theta shape: {theta.shape}, range [{theta.min():.3f}, {theta.max():.3f}]",
          flush=True)

    # Per-skill mean accuracy: for each (m, k), accuracy of m on items where Q[i, k] = 1
    print("Computing per-skill accuracy per LLM...", flush=True)
    per_skill_acc = np.full((n_llms, K), np.nan, dtype=np.float64)
    items_per_skill = []
    for k in range(K):
        skill_items = np.where(q_matrix[:, k] > 0)[0]
        items_per_skill.append(len(skill_items))
        if len(skill_items) == 0:
            continue
        per_skill_acc[:, k] = R[:, skill_items].mean(axis=1)

    print(f"Items per skill: min={min(items_per_skill)}, "
          f"median={int(np.median(items_per_skill))}, "
          f"max={max(items_per_skill)}", flush=True)

    # Per-skill Pearson r across 3,811 LLMs (one per skill)
    per_skill_r = np.full(K, np.nan)
    per_skill_p = np.full(K, np.nan)
    for k in range(K):
        if items_per_skill[k] == 0:
            continue
        x = theta[:, k]
        y = per_skill_acc[:, k]
        valid = ~np.isnan(y)
        if valid.sum() < 50:
            continue
        try:
            r, p = pearsonr(x[valid], y[valid])
            per_skill_r[k] = r
            per_skill_p[k] = p
        except Exception:
            pass

    valid_mask = ~np.isnan(per_skill_r)
    rs = per_skill_r[valid_mask]
    print(f"\nValid skills: {valid_mask.sum()} / {K}", flush=True)
    print(f"\n=== HEADLINE: per-skill Pearson r distribution ===", flush=True)
    print(f"  mean   = {rs.mean():.4f}", flush=True)
    print(f"  median = {np.median(rs):.4f}", flush=True)
    print(f"  std    = {rs.std():.4f}", flush=True)
    print(f"  min    = {rs.min():.4f}", flush=True)
    print(f"  max    = {rs.max():.4f}", flush=True)
    print(f"  fraction with r > 0.5: {(rs > 0.5).mean():.3f}", flush=True)
    print(f"  fraction with r > 0.7: {(rs > 0.7).mean():.3f}", flush=True)
    print(f"  fraction with r > 0.9: {(rs > 0.9).mean():.3f}", flush=True)
    print(f"  fraction with r > 0.95: {(rs > 0.95).mean():.3f}", flush=True)

    # Cross-skill diagonal-vs-off-diagonal:
    # Strong construct validity = θ_k correlates much better with skill_k accuracy
    # than with skill_j accuracy (j != k).
    # Compute mean off-diagonal r as control.
    print(f"\n=== CONTROL: off-diagonal correlation ===", flush=True)
    off_diag_rs = []
    np.random.seed(42)
    sample_pairs = [(i, j) for i in range(K) for j in range(K)
                     if i != j and items_per_skill[i] > 0 and items_per_skill[j] > 0]
    sample = [sample_pairs[i] for i in
                np.random.choice(len(sample_pairs), 1000, replace=False)]
    for i, j in sample:
        x = theta[:, i]
        y = per_skill_acc[:, j]
        valid = ~np.isnan(y)
        if valid.sum() < 50:
            continue
        try:
            r, _ = pearsonr(x[valid], y[valid])
            off_diag_rs.append(r)
        except Exception:
            pass
    off = np.array(off_diag_rs)
    print(f"  mean off-diagonal r (theta_i vs accuracy_j, i != j): {off.mean():.4f}",
          flush=True)
    print(f"  diagonal mean r:                                     {rs.mean():.4f}",
          flush=True)
    print(f"  difference (construct validity gap):                 {rs.mean() - off.mean():+.4f}",
          flush=True)

    # Optionally: top-3 and bottom-3 skills for inspection
    sorted_idx = np.argsort(per_skill_r)
    sorted_idx = sorted_idx[~np.isnan(per_skill_r[sorted_idx])]
    print(f"\nTop 3 skills by r:", flush=True)
    for k in sorted_idx[-3:][::-1]:
        print(f"  skill {k}: r = {per_skill_r[k]:.4f}, n_items = {items_per_skill[k]}",
              flush=True)
    print(f"\nBottom 3 skills by r:", flush=True)
    for k in sorted_idx[:3]:
        print(f"  skill {k}: r = {per_skill_r[k]:.4f}, n_items = {items_per_skill[k]}",
              flush=True)

    # Save summary JSON
    out = {
        "experiment": "construct_validity_theta_vs_skill_accuracy",
        "ckpt": str(CKPT.relative_to(REPO)),
        "K": int(K),
        "n_llms": int(n_llms),
        "n_valid_skills": int(valid_mask.sum()),
        "diagonal_r": {
            "mean": float(rs.mean()),
            "median": float(np.median(rs)),
            "std": float(rs.std()),
            "min": float(rs.min()),
            "max": float(rs.max()),
            "frac_above_0.5": float((rs > 0.5).mean()),
            "frac_above_0.7": float((rs > 0.7).mean()),
            "frac_above_0.9": float((rs > 0.9).mean()),
            "frac_above_0.95": float((rs > 0.95).mean()),
            "per_skill": rs.tolist(),
        },
        "off_diagonal_r": {
            "n_pairs_sampled": len(off),
            "mean": float(off.mean()),
            "std": float(off.std()),
        },
        "construct_validity_gap": float(rs.mean() - off.mean()),
        "verified": True,
    }
    out_path = EXP / "v2_construct_validity.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
