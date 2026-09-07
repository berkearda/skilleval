"""Check 1b for the circularity analysis: skill signal beyond BENCHMARK profile.

Check 1 showed theta_k adds signal beyond one overall-accuracy scalar
(partial r 0.43, positive on 38/41 skills). Skeptical reading: the extra
signal could be benchmark identity, not skill identity (skill k's items may
sit inside one benchmark, so "beyond overall accuracy" could just mean
"good at MATH-style benchmarks").

This closes that hole. Control = the model's TRAIN accuracy on each of the
five benchmarks separately (5 numbers per model, a full benchmark profile).
Question: after removing everything the benchmark profile explains, does
theta_k still track skill-k accuracy on HELD-OUT items?

If yes, the skill scores carry information finer than benchmark level.
A "benchmark-mixture in disguise" model cannot do this.
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
CKPT = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"

MIN_HELDOUT = 10  # same filter as checks so numbers are comparable


def partial_corr_multi(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """corr(x, y) after regressing the columns of Z (plus intercept) out of both."""
    Zc = np.column_stack([np.ones(len(x)), Z])
    rx = x - Zc @ np.linalg.lstsq(Zc, x, rcond=None)[0]
    ry = y - Zc @ np.linalg.lstsq(Zc, y, rcond=None)[0]
    return pearsonr(rx, ry)[0]


def main() -> None:
    np.random.seed(42)
    print("=== Check 1b: theta_k beyond the 5-benchmark train-accuracy profile ===\n", flush=True)

    R = np.load(DATA / "response_matrix_v2_full.npy")
    Q = np.load(DATA / "qmatrix_v2_K100.npy")
    K = Q.shape[1]
    items_meta = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    bench_of = np.array([m["benchmark"] for m in items_meta])
    benches = sorted(set(bench_of.tolist()))
    print(f"benchmarks: {benches}", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    theta = torch.sigmoid(ck["model_state_dict"]["student_emb.weight"]).numpy()
    train_items = np.array(sorted(ck["train_items"]), dtype=int)
    test_items = np.array(sorted(ck["test_items"]), dtype=int)
    assert set(train_items.tolist()).isdisjoint(set(test_items.tolist()))

    # Control: per-benchmark TRAIN accuracy profile (n_llms x 5).
    bench_profile = np.zeros((R.shape[0], len(benches)))
    for b_i, b in enumerate(benches):
        b_train = train_items[bench_of[train_items] == b]
        bench_profile[:, b_i] = R[:, b_train].mean(axis=1)
        print(f"  {b}: {len(b_train)} train items, profile mean {bench_profile[:, b_i].mean():.3f}", flush=True)

    # For reference also keep the scalar from check 1.
    scalar = R[:, train_items].mean(axis=1)

    rows = []
    for k in range(K):
        skill_ho = test_items[Q[test_items, k] > 0]
        if len(skill_ho) < MIN_HELDOUT:
            continue
        target = R[:, skill_ho].mean(axis=1)
        # dominant benchmark of this skill's held-out items (for reporting)
        b_counts = {b: int((bench_of[skill_ho] == b).sum()) for b in benches}
        dom_b = max(b_counts, key=b_counts.get)
        rows.append({
            "skill": int(k),
            "n_heldout_items": int(len(skill_ho)),
            "dominant_benchmark": dom_b,
            "dom_bench_share": float(b_counts[dom_b] / len(skill_ho)),
            "r_theta_raw": float(pearsonr(theta[:, k], target)[0]),
            "r_theta_partial_scalar": float(partial_corr_multi(theta[:, k], target, scalar[:, None])),
            "r_theta_partial_benchprofile": float(partial_corr_multi(theta[:, k], target, bench_profile)),
        })

    n = len(rows)
    p_scalar = np.array([r["r_theta_partial_scalar"] for r in rows])
    p_bench = np.array([r["r_theta_partial_benchprofile"] for r in rows])
    print(f"\nskills evaluated (>= {MIN_HELDOUT} held-out items): {n}", flush=True)
    print(f"  mean partial r | scalar          : {p_scalar.mean():.4f}  (positive on {(p_scalar>0).sum()}/{n})", flush=True)
    print(f"  mean partial r | 5-bench profile : {p_bench.mean():.4f}  (positive on {(p_bench>0).sum()}/{n})", flush=True)
    print(f"  median partial r | 5-bench profile: {np.median(p_bench):.4f}", flush=True)
    print(f"  partial r | bench > 0.2 on {(p_bench>0.2).sum()}/{n} skills", flush=True)

    rows_sorted = sorted(rows, key=lambda r: -r["r_theta_partial_benchprofile"])
    print("\ntop 5 (most sub-benchmark signal):", flush=True)
    for r in rows_sorted[:5]:
        print(f"  skill {r['skill']:3d} [{r['dominant_benchmark']:6s} {r['dom_bench_share']:.0%}] "
              f"n_ho={r['n_heldout_items']:3d}  partial_bench={r['r_theta_partial_benchprofile']:+.3f}", flush=True)
    print("bottom 5:", flush=True)
    for r in rows_sorted[-5:]:
        print(f"  skill {r['skill']:3d} [{r['dominant_benchmark']:6s} {r['dom_bench_share']:.0%}] "
              f"n_ho={r['n_heldout_items']:3d}  partial_bench={r['r_theta_partial_benchprofile']:+.3f}", flush=True)

    out = {
        "experiment": "skill_theta_beyond_benchmark_profile_heldout",
        "ckpt": str(CKPT.relative_to(REPO)),
        "design": "partial corr of theta_k with held-out skill-k acc, controlling for 5 per-benchmark train accuracies",
        "min_heldout_items": MIN_HELDOUT,
        "n_skills_evaluated": n,
        "mean_partial_r_given_scalar": float(p_scalar.mean()),
        "mean_partial_r_given_benchprofile": float(p_bench.mean()),
        "median_partial_r_given_benchprofile": float(np.median(p_bench)),
        "n_positive_given_benchprofile": int((p_bench > 0).sum()),
        "n_above_0.2_given_benchprofile": int((p_bench > 0.2).sum()),
        "per_skill": rows,
        "verified": True,
    }
    outp = EXP / "v2_skill_beyond_benchprofile_heldout.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nWrote {outp}", flush=True)


if __name__ == "__main__":
    main()
