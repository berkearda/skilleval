#!/usr/bin/env python3
"""Robustness of the profile-agreement measure in run_profiling_bayes.py (T-137, 24 Sep 2026).

Concern (A5): the reference profile (all 7,618 answers) was estimated WITH the prior. For skills with few items the
reference then partly reflects the prior itself, which could favour the prior-based methods over the old recipe.
This diagnostic recomputes profile agreement (per skill, Pearson r across the new LLMs with the reference, averaged over
skills) against (i) the prior-based reference, as reported, and (ii) a reference estimated WITHOUT the prior, each over
(a) all 100 skills and (b) only skills with at least 20 pool items, where the answers dominate the estimate.
No new selection is run; the N-answer profiles are read from v2_profiling_bayes.npz.

Output: cdm_exploration/experiments/v2_profiling_bayes_robustness.json

    python3 tools/diag_profiling_bayes_robustness.py [--smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
from run_profiling_bayes import CKPT, DATA, EXP, SHRINK, Frozen, colcorr, fit_bayes

MIN_ITEMS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seed_everything(42)
    torch.set_num_threads(12)
    z = np.load(EXP / "v2_profiling_bayes.npz")
    res_main = json.load(open(EXP / "v2_profiling_bayes.json"))
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q = np.load(DATA / "qmatrix_v2_K100.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    n_llms, n_items = R.shape
    K = q.shape[1]
    pool, test = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    train_llms, _ = train_test_split(np.arange(n_llms), test_size=0.2, random_state=42)
    ok = verify_splits(pool, test, expected_seed=42, label="profiling_bayes_robustness")
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(CKPT, net, "cpu")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    model = Frozen(net, emb, q)
    raw_train = net.student_emb.weight[torch.as_tensor(train_llms)].detach().double().numpy()
    mu = torch.tensor(raw_train.mean(axis=0), dtype=torch.float32)
    S = np.cov(raw_train.T)
    S = (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S))
    P = torch.tensor(np.linalg.inv(S), dtype=torch.float32)

    new_llms = z["new_llms"]
    L = 3 if a.smoke else len(new_llms)
    t0 = time.time()
    theta_np = np.zeros((L, K))
    for li in range(L):
        l = new_llms[li]
        r = fit_bayes(model, pool, R[l, pool], mu, P, mu, max_iter=200, prior=False)
        theta_np[li] = torch.sigmoid(r).numpy()
        if li == 0 or (li + 1) % 25 == 0:
            print(f"  {li + 1}/{L}: {(time.time() - t0) / 60:.1f} min", flush=True)
    theta_pr = z["theta_full"][:L]
    n_pool_items = q[pool].sum(axis=0)
    big = n_pool_items >= MIN_ITEMS
    budgets = [int(b) for b in z["budgets"]]
    out = {"n_llms": L, "min_items": MIN_ITEMS, "n_skills_with_min_items": int(big.sum()), "rows": []}
    for bi, n in enumerate(budgets):
        row = {"N": n}
        for ref_name, ref in (("prior_ref", theta_pr), ("noprior_ref", theta_np)):
            for sk_name, sk in (("all", np.ones(K, bool)), ("big", big)):
                def agr(th):
                    return float(colcorr(th[:L][:, sk], ref[:, sk]).mean())
                row[f"random_{ref_name}_{sk_name}"] = float(np.mean([agr(z["theta_random"][s, bi]) for s in
                                                                   range(z["theta_random"].shape[0])]))
                row[f"old_{ref_name}_{sk_name}"] = agr(z["theta_old"][bi])
                if not np.isnan(z["theta_adaptive"][bi]).any():
                    row[f"adaptive_{ref_name}_{sk_name}"] = agr(z["theta_adaptive"][bi])
        out["rows"].append(row)
    if not a.smoke:   # the prior-based, all-skills variant must reproduce the reported numbers
        for row, main in zip(out["rows"], res_main["summary"]):
            assert abs(row["random_prior_ref_all"] - main["random_profile_agreement"]) < 1e-9
            assert abs(row["old_prior_ref_all"] - main["old_profile_agreement"]) < 1e-9
    # adaptive minus random against the no-prior reference, all skills: paired bootstrap over LLMs
    rng = np.random.default_rng(42)
    for bi, n in enumerate(budgets):
        if n == 0 or np.isnan(z["theta_adaptive"][bi]).any():
            continue
        th_r, th_a = z["theta_random"][:, bi, :L], z["theta_adaptive"][bi][:L]
        diffs = []
        for _ in range(2000):
            ix = rng.integers(0, L, L)
            a_ = colcorr(th_a[ix], theta_np[ix]).mean()
            r_ = np.mean([colcorr(th_r[s][ix], theta_np[ix]).mean() for s in range(th_r.shape[0])])
            diffs.append(a_ - r_)
        row = out["rows"][bi]
        row["adaptive_minus_random_noprior_ref_all"] = row["adaptive_noprior_ref_all"] - row["random_noprior_ref_all"]
        row["adaptive_minus_random_noprior_ref_all_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    np.savez_compressed(EXP / f"v2_profiling_bayes_robustness{'_smoke' if a.smoke else ''}.npz", theta_noprior_full=theta_np)
    out["verified"] = bool(ok)
    tag = "_smoke" if a.smoke else ""
    (EXP / f"v2_profiling_bayes_robustness{tag}.json").write_text(json.dumps(out, indent=2))
    for row in out["rows"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print(f"{(time.time() - t0) / 60:.1f} min")
    if not a.smoke:
        log_experiment(name="diag_profiling_bayes_robustness", config={"min_items": MIN_ITEMS},
                       results=out, split_info={"n_pool_items": len(pool), "n_test_items": len(test),
                                                "random_state": 42}, verified=bool(ok))


if __name__ == "__main__":
    main()
