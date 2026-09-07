"""Final three checks for the circularity analysis (all five-seed, held-out).

A. PLACEBO / null control (F9 discipline): permute which held-out items
   belong to which skill, WITHIN each benchmark (fake skills keep the real
   skill's size and benchmark mix, and the real multi-membership pattern,
   but random content). Recompute mean partial r | 5-benchmark profile.
   20 permutations -> null distribution. The real value (0.135) must sit
   far outside it, else the incremental-signal claim is an artifact.

B. SUBTASK boundary: control = per-subtask TRAIN accuracy profile
   (~40-60 columns). If partial r survives, skills carry sub-subtask
   signal. If it shrinks, skills align with subtask structure - note that
   where skill == subtask this control removes the skill itself, so this
   maps a boundary rather than testing existence.

C. Per-LLM profile correlation on held-out items (the axis the paper's
   L377 sentence actually claims). Raw, and skill-centered (subtracting
   each skill's across-model mean first, removing difficulty structure).

Theta-dependent numbers averaged over the 5 multi-seed checkpoints.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "cdm_exploration" / "data" / "cdm_ready"
EXP = REPO / "cdm_exploration" / "experiments"
MS = REPO / "cdm_exploration" / "checkpoints" / "multi_seed"

MIN_HELDOUT = 10
SEEDS = [42, 43, 44, 45, 46]
N_PERM = 20


def residualize(v: np.ndarray, Zc: np.ndarray) -> np.ndarray:
    return v - Zc @ np.linalg.lstsq(Zc, v, rcond=None)[0]


def main() -> None:
    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    K = Q.shape[1]
    meta = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    bench_of = np.array([m["benchmark"] for m in meta])
    subtask_of = np.array([m["benchmark"] + "/" + m["subtask"] for m in meta])
    benches = sorted(set(bench_of.tolist()))

    ck0 = torch.load(MS / "text_conditioned_seed_42.pt", map_location="cpu", weights_only=False)
    train_items = np.array(sorted(ck0["train_items"]), dtype=int)
    test_items = np.array(sorted(ck0["test_items"]), dtype=int)

    bench_profile = np.column_stack([
        R[:, train_items[bench_of[train_items] == b]].mean(axis=1) for b in benches
    ])
    Zb = np.column_stack([np.ones(R.shape[0]), bench_profile])

    subtasks = sorted(set(subtask_of[train_items].tolist()))
    sub_profile = np.column_stack([
        R[:, train_items[subtask_of[train_items] == s]].mean(axis=1) for s in subtasks
    ])
    Zs = np.column_stack([np.ones(R.shape[0]), sub_profile])
    print(f"benchmarks: {len(benches)}, subtasks: {len(subtasks)}", flush=True)

    skill_ho = {}
    for k in range(K):
        ho = test_items[Q[test_items, k] > 0]
        if len(ho) >= MIN_HELDOUT:
            skill_ho[k] = ho
    skills = sorted(skill_ho)
    n_sk = len(skills)
    print(f"measurable skills: {n_sk}\n", flush=True)

    thetas = {}
    for s in SEEDS:
        ck = torch.load(MS / f"text_conditioned_seed_{s}.pt", map_location="cpu", weights_only=False)
        thetas[s] = torch.sigmoid(ck["model_state_dict"]["student_emb.weight"]).numpy()

    def mean_partial(th: np.ndarray, Zc: np.ndarray, ho_map: dict) -> tuple[float, int]:
        prs = []
        for k in skills:
            target = R[:, ho_map[k]].mean(axis=1)
            prs.append(pearsonr(residualize(th[:, k], Zc), residualize(target, Zc))[0])
        prs = np.array(prs)
        return float(prs.mean()), int((prs > 0).sum())

    # ---- A. Placebo ----------------------------------------------------
    print("=== A. Placebo: fake skills (within-benchmark permutation), partial r | bench profile ===", flush=True)
    real_vals = [mean_partial(thetas[s], Zb, skill_ho)[0] for s in SEEDS]
    real_mean = float(np.mean(real_vals))
    print(f"  REAL skills, 5-seed mean: {real_mean:.4f}", flush=True)

    rng = np.random.default_rng(42)
    null_vals = []
    th42 = thetas[42]
    for p in range(N_PERM):
        # permute Q rows of test items within each benchmark
        perm = np.arange(len(test_items))
        for b in benches:
            idx = np.where(bench_of[test_items] == b)[0]
            perm[idx] = idx[rng.permutation(len(idx))]
        test_perm = test_items[perm]
        fake_ho = {}
        for k in skills:
            fake_ho[k] = test_perm[Q[test_items, k] > 0]  # same rows of Q, permuted item content
        m, _ = mean_partial(th42, Zb, fake_ho)
        null_vals.append(m)
    null_vals = np.array(null_vals)
    z = (real_mean - null_vals.mean()) / null_vals.std()
    p_emp = float((null_vals >= real_mean).mean())
    print(f"  PLACEBO ({N_PERM} perms, seed-42 theta): mean {null_vals.mean():+.4f} +/- {null_vals.std():.4f}, "
          f"max {null_vals.max():+.4f}", flush=True)
    print(f"  real vs null: z = {z:.1f}, empirical p = {p_emp} "
          f"({(null_vals >= real_mean).sum()}/{N_PERM} perms reach the real value)", flush=True)

    # ---- B. Subtask control --------------------------------------------
    print(f"\n=== B. Partial r | {len(subtasks)}-subtask profile (5 seeds) ===", flush=True)
    sub_means, sub_pos = [], []
    for s in SEEDS:
        m, npos = mean_partial(thetas[s], Zs, skill_ho)
        sub_means.append(m)
        sub_pos.append(npos)
    print(f"  mean partial r | subtask profile: {np.mean(sub_means):.4f} +/- {np.std(sub_means):.4f}, "
          f"positive on {min(sub_pos)}-{max(sub_pos)}/{n_sk}", flush=True)
    # dominant-subtask concentration per skill, for interpretation
    dom_share = []
    for k in skills:
        st, counts = np.unique(subtask_of[skill_ho[k]], return_counts=True)
        dom_share.append(counts.max() / counts.sum())
    dom_share = np.array(dom_share)
    print(f"  skill/subtask alignment: median dominant-subtask share {np.median(dom_share):.0%}, "
          f"{(dom_share > 0.9).sum()}/{n_sk} skills >90% one subtask", flush=True)

    # ---- C. Per-LLM profile correlation ---------------------------------
    print(f"\n=== C. Per-LLM profile correlation over the {n_sk} skills (held-out, 5 seeds) ===", flush=True)
    ho_acc = np.column_stack([R[:, skill_ho[k]].mean(axis=1) for k in skills])  # (n_llms, n_sk)
    raw_means, cen_means = [], []
    for s in SEEDS:
        th = thetas[s][:, skills]
        raw = [pearsonr(th[i], ho_acc[i])[0] for i in range(R.shape[0])]
        thc = th - th.mean(axis=0)
        hoc = ho_acc - ho_acc.mean(axis=0)
        cen = [pearsonr(thc[i], hoc[i])[0] for i in range(R.shape[0])]
        raw_means.append(float(np.mean(raw)))
        cen_means.append(float(np.mean(cen)))
    print(f"  raw per-LLM mean r      : {np.mean(raw_means):.4f} +/- {np.std(raw_means):.4f}", flush=True)
    print(f"  skill-centered mean r   : {np.mean(cen_means):.4f} +/- {np.std(cen_means):.4f}", flush=True)
    print("  (raw includes difficulty structure; centered is pure profile shape)", flush=True)

    out = {
        "experiment": "final_circularity_checks",
        "n_skills": n_sk,
        "min_heldout_items": MIN_HELDOUT,
        "seeds": SEEDS,
        "A_placebo": {
            "real_mean_partial_r_benchprofile_5seed": real_mean,
            "n_permutations": N_PERM,
            "null_mean": float(null_vals.mean()),
            "null_std": float(null_vals.std()),
            "null_max": float(null_vals.max()),
            "z_score": float(z),
            "empirical_p": p_emp,
        },
        "B_subtask_control": {
            "n_subtasks": len(subtasks),
            "mean_partial_r": float(np.mean(sub_means)),
            "std_across_seeds": float(np.std(sub_means)),
            "positive_range": [min(sub_pos), max(sub_pos)],
            "median_dominant_subtask_share": float(np.median(dom_share)),
            "n_skills_over_90pct_one_subtask": int((dom_share > 0.9).sum()),
        },
        "C_per_llm_profile": {
            "raw_mean_r": float(np.mean(raw_means)),
            "raw_std": float(np.std(raw_means)),
            "centered_mean_r": float(np.mean(cen_means)),
            "centered_std": float(np.std(cen_means)),
        },
        "verified": True,
    }
    outp = EXP / "v2_final_circularity_checks.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nWrote {outp}", flush=True)


if __name__ == "__main__":
    main()
