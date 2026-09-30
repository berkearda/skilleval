#!/usr/bin/env python3
"""Profiling unseen LLMs, redesigned (T-137, 24 Sep 2026): start from the average training profile, let one answer move
related skills, and measure what the answers teach us about the specific LLM.

Why: with the frozen network trained on the 3,048 training LLMs, the average training profile with zero answers already
gives per-LLM test AUC 0.675 (diag_cdcat_prior_probe.py), while the old recipe (every skill at 0.5, no prior) needs
500 answers for 0.678. Per-LLM AUC is dominated by item difficulty, so it cannot show what a profile adds.

Pipeline (all with the frozen network text_conditioned_protocolB_llmsplit3048.pt; item parameters come from the item
text and never change; only the new LLM's 100-dimensional raw profile r, theta = sigmoid(r), is estimated):
  0. From the 3,048 training LLMs: prior mean mu = average raw profile; prior covariance = their covariance, shrunk 10%
     toward its diagonal ("related-skills table").
  1. New LLMs: the first --n_llms of the 763 held-out LLMs (same ones for every method).
  2. Start at mu. Pool = the 7,618 training items; held-out check set = the 1,905 test items (never asked).
  3. Ask one pool item at a time, chosen at random (3 seeds) or by the adaptive heuristic p(1-p)*|q_j| (deterministic),
     and update r by minimising  sum of Bernoulli negative log-likelihoods of all answers so far  +  0.5 (r-mu)' P (r-mu).
     The likelihood is SUMMED, so the prior fades as answers accumulate.
  4. At each budget N, refit from mu to convergence and measure on the held-out items:
       (a) benchmark scores: predicted accuracy (mean predicted probability) vs real accuracy per benchmark;
       (b) profile agreement: per skill, Pearson r across the new LLMs between the N-answer theta and the theta from all
           7,618 answers (same estimator), averaged over skills (a skill whose estimate is identical for all LLMs counts 0);
       (c) per-LLM AUC (the old measure).
  5. Comparators: average profile with zero answers (= N 0); simple count (share correct among the asked items of each
     benchmark, random arm only; a benchmark with no asked item falls back to the average-profile prediction); old
     method (start 0.5, mean BCE, Adam lr 0.05, 500 steps, no prior; random arm, seed 0), which must reproduce 0.547 at 0.
  6. Adaptive minus random (random averaged over seeds): paired per LLM, bootstrap 95% CI over LLMs.
Checks: N 0 gives exactly mu; with all 7,618 answers the prior no longer matters (prior vs no-prior fits agree on a
subset); old method at 0 answers reproduces the registered 0.547 on the same LLMs to +-0.005 of the probe.

Pre-run expectations (design_system N10 B4), written before the full run:
  zero answers, average profile: per-LLM AUC ~0.675; benchmark error about the spread of LLM scores (~0.1); profile
  agreement 0.  Random, 500 answers: AUC ~0.68-0.69, benchmark error ~0.03-0.05, profile agreement ~0.7-0.9.  Simple count:
  error ~0.09 at 100 answers (about 20 per benchmark), ~0.04 at 500.  Old method at 0 answers: AUC 0.547.
Cost (B5): CPU only, about 30-40 min for 200 LLMs; abort and re-plan if the smoke test implies more than 80 min.

    python3 tools/run_profiling_bayes.py --smoke            # 3 LLMs, budgets 0,5,20
    python3 tools/run_profiling_bayes.py                    # 200 LLMs, 3 seeds
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits

DATA = REPO / "cdm_exploration/data/cdm_ready"
EXP = REPO / "cdm_exploration/experiments"
CKPT = REPO / "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB_llmsplit3048.pt"
BENCH = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
SHRINK = 0.10


def colcorr(A, Z):
    """Pearson r per column between A and Z (n x K); a column that is constant in A counts 0."""
    A = A - A.mean(0); Z = Z - Z.mean(0)
    sa, sz = np.sqrt((A ** 2).sum(0)), np.sqrt((Z ** 2).sum(0))
    r = (A * Z).sum(0) / np.where(sa * sz > 0, sa * sz, 1.0)
    return np.where(sa < 1e-9, 0.0, r)


class Frozen:
    """The frozen network, with the text-derived item parameters computed once."""

    def __init__(self, net, emb, q):
        with torch.no_grad():
            te = torch.tensor(emb, dtype=torch.float32)
            self.kd = torch.sigmoid(net.k_difficulty_proj(te))
            self.ed = torch.sigmoid(net.e_difficulty_proj(te))
        self.q = torch.tensor(q, dtype=torch.float32)
        self.l1, self.l2, self.l3 = net.prednet_full1, net.prednet_full2, net.prednet_full3

    def prob(self, raw, idx):
        x = self.ed[idx] * (torch.sigmoid(raw) - self.kd[idx]) * self.q[idx]
        h = torch.sigmoid(self.l2(torch.sigmoid(self.l1(x))))
        return torch.sigmoid(self.l3(h)).squeeze(-1)


def fit_bayes(model, idx, y, mu, P, init, max_iter, prior=True):
    """Minimise summed Bernoulli NLL (+ Gaussian prior on the raw profile). Returns the raw profile."""
    if len(idx) == 0:
        return mu.clone()
    idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    r = init.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([r], lr=1.0, max_iter=max_iter, history_size=20, tolerance_grad=1e-7,
                            tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = model.prob(r, idx_t).clamp(1e-6, 1 - 1e-6)
        loss = -(y_t * torch.log(p) + (1 - y_t) * torch.log(1 - p)).sum()
        if prior:
            d = r - mu
            loss = loss + 0.5 * d @ P @ d
        loss.backward()
        return loss
    opt.step(closure)
    return r.detach()


def fit_old(model, idx, y, K):
    """Old recipe (tools/run_adaptive_testing_v2_fixed.py::fit_theta_fixed): start 0, mean BCE, Adam 0.05, 500 steps."""
    raw = torch.zeros(K, requires_grad=True)
    if len(idx) == 0:
        return raw.detach()
    idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    opt = torch.optim.Adam([raw], lr=0.05)
    bce = torch.nn.BCELoss()
    for _ in range(500):
        loss = bce(model.prob(raw, idx_t), y_t)
        opt.zero_grad(); loss.backward(); opt.step()
    return raw.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_llms", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", default="0,5,10,20,50,100,200,500")
    ap.add_argument("--adaptive_max", type=int, default=100)
    ap.add_argument("--n_check_noprior", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_llms, a.seeds, a.budgets, a.adaptive_max, a.n_check_noprior = 3, 1, "0,5,20", 20, 3
    if a.timing:
        a.n_llms, a.n_check_noprior = 2, 1
    budgets = [int(b) for b in a.budgets.split(",")]
    seed_everything(42)
    torch.set_num_threads(12)

    # input contracts (B2)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q = np.load(DATA / "qmatrix_v2_K100.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    assert R.shape == (3811, 9523) and q.shape == (9523, 100) and emb.shape == (9523, 768) and len(items) == 9523
    assert set(np.unique(R)) <= {0, 1}
    n_llms, n_items = R.shape
    K = q.shape[1]
    pool, test = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    train_llms, test_llms = train_test_split(np.arange(n_llms), test_size=0.2, random_state=42)
    ok = verify_splits(pool, test, expected_seed=42, label="profiling_bayes")
    assert len(pool) == 7618 and len(test) == 1905 and len(train_llms) == 3048 and len(test_llms) == 763
    new_llms = test_llms[:a.n_llms]
    bench_te = np.array([items[i]["benchmark"] for i in test])
    bench_pool = np.array([items[i]["benchmark"] for i in pool])

    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(CKPT, net, "cpu")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    model = Frozen(net, emb, q)
    test_t = torch.as_tensor(test, dtype=torch.long)
    nsk = q[pool].sum(axis=1)

    # step 0: prior from the training LLMs
    raw_train = net.student_emb.weight[torch.as_tensor(train_llms)].detach().double().numpy()
    mu = torch.tensor(raw_train.mean(axis=0), dtype=torch.float32)
    S = np.cov(raw_train.T)
    S = (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S))
    P = torch.tensor(np.linalg.inv(S), dtype=torch.float32)

    def measure(raw, l):
        with torch.no_grad():
            p = model.prob(raw, test_t).numpy()
        yl = R[l, test]
        return (roc_auc_score(yl, p), np.array([p[bench_te == b].mean() for b in BENCH]), torch.sigmoid(raw).numpy())

    real = np.array([[R[l, test][bench_te == b].mean() for b in BENCH] for l in new_llms])
    L, B = len(new_llms), len(budgets)
    out = {k: np.full((a.seeds, B, L), np.nan) for k in ("auc_random",)}
    auc = {"random": np.full((a.seeds, B, L), np.nan), "adaptive": np.full((B, L), np.nan), "old": np.full((B, L), np.nan)}
    bpred = {"random": np.full((a.seeds, B, L, 5), np.nan), "adaptive": np.full((B, L, 5), np.nan),
             "old": np.full((B, L, 5), np.nan), "count": np.full((a.seeds, B, L, 5), np.nan)}
    theta = {"random": np.full((a.seeds, B, L, K), np.nan), "adaptive": np.full((B, L, K), np.nan),
             "old": np.full((B, L, K), np.nan)}
    auc_full, bpred_full, theta_full = np.zeros(L), np.zeros((L, 5)), np.zeros((L, K))
    checks = {}
    t0 = time.time()
    for li, l in enumerate(new_llms):
        # reference profile from all 7,618 pool answers (same estimator)
        r_full = fit_bayes(model, pool, R[l, pool], mu, P, mu, max_iter=200)
        auc_full[li], bpred_full[li], theta_full[li] = measure(r_full, l)
        if li < a.n_check_noprior:
            r_np = fit_bayes(model, pool, R[l, pool], mu, P, mu, max_iter=200, prior=False)
            with torch.no_grad():
                pa, pb = model.prob(r_full, test_t).numpy(), model.prob(r_np, test_t).numpy()
            checks.setdefault("prior_vs_noprior_all_answers_mean_abs_pred_diff", []).append(float(np.abs(pa - pb).mean()))
            checks.setdefault("prior_vs_noprior_all_answers_share_items_diff_above_005", []).append(
                float((np.abs(pa - pb) > 0.05).mean()))
            checks.setdefault("prior_vs_noprior_all_answers_auc_diff", []).append(
                float(roc_auc_score(R[l, test], pa) - roc_auc_score(R[l, test], pb)))
        # random order, with prior (3 seeds) and the simple count
        for s in range(a.seeds):
            order = np.random.default_rng([s, int(l)]).permutation(pool)
            for bi, n in enumerate(budgets):
                asked = order[:n]
                r = fit_bayes(model, asked, R[l, asked], mu, P, mu, max_iter=100)
                if n == 0:
                    assert torch.equal(r, mu)
                auc["random"][s, bi, li], bpred["random"][s, bi, li], theta["random"][s, bi, li] = measure(r, l)
                ab = np.array([items[i]["benchmark"] for i in asked]) if n else np.array([])
                for k, b in enumerate(BENCH):
                    m = ab == b
                    bpred["count"][s, bi, li, k] = R[l, asked][m].mean() if m.any() else bpred["random"][0, 0, li, k]
                if s == 0:   # old method on the same asked items
                    r_old = fit_old(model, asked, R[l, asked], K)
                    auc["old"][bi, li], bpred["old"][bi, li], theta["old"][bi, li] = measure(r_old, l)
        # adaptive heuristic, with prior (deterministic)
        remaining = np.ones(len(pool), bool)
        asked, r = [], mu.clone()
        if 0 in budgets:
            bi = budgets.index(0)
            auc["adaptive"][bi, li], bpred["adaptive"][bi, li], theta["adaptive"][bi, li] = measure(mu, l)
        for step in range(1, a.adaptive_max + 1):
            cand = pool[remaining]
            with torch.no_grad():
                p = model.prob(r, torch.as_tensor(cand, dtype=torch.long)).numpy()
            j = int(np.argmax(p * (1 - p) * np.maximum(nsk[remaining], 1)))
            asked.append(int(cand[j]))
            remaining[np.flatnonzero(remaining)[j]] = False
            r = fit_bayes(model, asked, R[l, asked], mu, P, r, max_iter=10)
            if step in budgets:
                bi = budgets.index(step)
                r_fin = fit_bayes(model, asked, R[l, asked], mu, P, mu, max_iter=100)
                auc["adaptive"][bi, li], bpred["adaptive"][bi, li], theta["adaptive"][bi, li] = measure(r_fin, l)
        if li == 0 or (li + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  LLM {li + 1}/{L}: {el / 60:.1f} min elapsed, about {el / (li + 1) * (L - li - 1) / 60:.0f} min left",
                  flush=True)

    # summaries
    def agree(th, ix=None):  # mean over skills of Pearson r across LLMs with the full-answer profile
        ix = np.arange(L) if ix is None else ix
        return float(colcorr(th[ix], theta_full[ix]).mean())

    def bench_err(bp):      # mean absolute error over LLMs and benchmarks
        return float(np.mean(np.abs(bp - real)))

    def bench_r(bp):
        rs = [0.0 if np.std(bp[:, k]) < 1e-9 else float(np.corrcoef(bp[:, k], real[:, k])[0, 1]) for k in range(5)]
        return float(np.mean(rs))

    rng = np.random.default_rng(42)
    summary = []
    for bi, n in enumerate(budgets):
        row = {"N": n}
        row["random_auc"] = float(np.nanmean(auc["random"][:, bi]))
        row["random_bench_err"] = float(np.mean([bench_err(bpred["random"][s, bi]) for s in range(a.seeds)]))
        row["random_bench_r"] = float(np.mean([bench_r(bpred["random"][s, bi]) for s in range(a.seeds)]))
        row["random_profile_agreement"] = float(np.mean([agree(theta["random"][s, bi]) for s in range(a.seeds)]))
        row["count_bench_err"] = float(np.mean([bench_err(bpred["count"][s, bi]) for s in range(a.seeds)]))
        row["count_bench_r"] = float(np.mean([bench_r(bpred["count"][s, bi]) for s in range(a.seeds)]))
        row["old_auc"] = float(np.mean(auc["old"][bi]))
        row["old_bench_err"] = bench_err(bpred["old"][bi])
        row["old_bench_r"] = bench_r(bpred["old"][bi])
        row["old_profile_agreement"] = agree(theta["old"][bi])
        if not np.isnan(auc["adaptive"][bi]).any():
            row["adaptive_auc"] = float(np.mean(auc["adaptive"][bi]))
            row["adaptive_bench_err"] = bench_err(bpred["adaptive"][bi])
            row["adaptive_bench_r"] = bench_r(bpred["adaptive"][bi])
            row["adaptive_profile_agreement"] = agree(theta["adaptive"][bi])
            # paired differences, adaptive minus random (random averaged over seeds), bootstrap over LLMs
            d_auc = auc["adaptive"][bi] - auc["random"][:, bi].mean(0)
            e_ad = np.abs(bpred["adaptive"][bi] - real).mean(1)
            e_rd = np.abs(bpred["random"][:, bi] - real).mean(2).mean(0)
            boots_auc, boots_err, boots_agr = [], [], []
            for _ in range(2000):
                ix = rng.integers(0, L, L)
                boots_auc.append(d_auc[ix].mean())
                boots_err.append((e_ad[ix] - e_rd[ix]).mean())
                agr_r = np.mean([agree(theta["random"][s, bi], ix) for s in range(a.seeds)])
                agr_a = agree(theta["adaptive"][bi], ix)
                boots_agr.append(agr_a - agr_r)
            for name, arr, point in (("auc", boots_auc, d_auc.mean()), ("bench_err", boots_err, (e_ad - e_rd).mean()),
                                     ("profile_agreement", boots_agr,
                                      row["adaptive_profile_agreement"] - row["random_profile_agreement"])):
                row[f"adaptive_minus_random_{name}"] = float(point)
                row[f"adaptive_minus_random_{name}_ci95"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        summary.append(row)

    for key in ("prior_vs_noprior_all_answers_mean_abs_pred_diff", "prior_vs_noprior_all_answers_share_items_diff_above_005"):
        checks[key] = float(np.mean(checks[key]))
    checks["prior_vs_noprior_all_answers_auc_diff_max_abs"] = float(np.max(np.abs(checks.pop("prior_vs_noprior_all_answers_auc_diff"))))
    checks["old_method_zero_answers_auc"] = summary[budgets.index(0)]["old_auc"] if 0 in budgets else None
    probe = json.load(open(EXP / "v2_cdcat_prior_probe.json"))
    if 0 in budgets and not (a.smoke or a.timing):
        assert abs(checks["old_method_zero_answers_auc"] - probe["starts"]["all_0.5"]["auc_mean_all_763"]) < 0.01
        assert abs(summary[budgets.index(0)]["random_auc"] - probe["starts"]["population_mean"]["auc_mean_all_763"]) < 0.01
    assert checks["prior_vs_noprior_all_answers_auc_diff_max_abs"] < 0.005, checks
    res = {"experiment": "profiling_bayes", "t_id": "T-137", "checkpoint": CKPT.name, "n_new_llms": L,
           "seeds": a.seeds, "budgets": budgets, "adaptive_max": a.adaptive_max, "shrink": SHRINK,
           "full_answers": {"auc": float(auc_full.mean()), "bench_err": bench_err(bpred_full), "bench_r": bench_r(bpred_full)},
           "checks": checks, "summary": summary, "minutes": (time.time() - t0) / 60, "verified": bool(ok)}
    tag = "_smoke" if a.smoke else "_timing" if a.timing else ""
    (EXP / f"v2_profiling_bayes{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(EXP / f"v2_profiling_bayes{tag}.npz", new_llms=new_llms, budgets=np.array(budgets), real=real,
                        auc_random=auc["random"], auc_adaptive=auc["adaptive"], auc_old=auc["old"],
                        bpred_random=bpred["random"], bpred_adaptive=bpred["adaptive"], bpred_old=bpred["old"],
                        bpred_count=bpred["count"], theta_random=theta["random"], theta_adaptive=theta["adaptive"],
                        theta_old=theta["old"], theta_full=theta_full, auc_full=auc_full, bpred_full=bpred_full)
    for row in summary:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if "ci95" not in k})
    print("full answers:", res["full_answers"], "\nchecks:", checks, f"\n{res['minutes']:.1f} min")
    if not (a.smoke or a.timing):
        log_experiment(name="run_profiling_bayes", config={k: res[k] for k in ("n_new_llms", "seeds", "budgets",
                                                                             "adaptive_max", "shrink", "checkpoint")},
                       results={"summary": summary, "full_answers": res["full_answers"], "checks": checks},
                       split_info={"n_pool_items": len(pool), "n_test_items": len(test), "n_train_llms": 3048,
                                   "n_test_llms": 763, "random_state": 42}, verified=bool(ok))


if __name__ == "__main__":
    main()
