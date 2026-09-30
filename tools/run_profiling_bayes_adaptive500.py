#!/usr/bin/env python3
"""Extend adaptive selection in the new-LLM profiling experiment from 100 to 500 answers (T-137, 24 Sep 2026).

Berke: the figure shows random selection up to 500 answers but adaptive only up to 100, which invites the question why.
This reruns the adaptive heuristic (Equation 4) for the same 200 held-out LLMs with the same frozen network, prior and
estimator as tools/run_profiling_bayes.py, now up to 500 answers, and measures at 200 and 500 answers as well.
Adaptive selection is deterministic, so budgets 5-100 must reproduce the stored results (checked below).
Reads v2_profiling_bayes.npz (random-selection results, the prior-based all-answers profiles) and
v2_profiling_bayes_robustness.npz (the all-answers profiles estimated without the prior, the strict reference).

Pre-run expectations (N10 B4): adaptive error about 0.050 at 200 and 0.046 at 500 answers (random: 0.055, 0.049);
strict agreement about 0.58 and 0.63 (random: 0.50, 0.57); reproduction of budgets 5-100 within 0.001 on the means.
Cost (B5): the smoke test took 4 min per LLM on one core (13 h serially, far above the first estimate of 45 min), so
the run is split into 10 parallel parts of 20 LLMs (--part i, one thread each, about 80-90 min), then merged (--merge).

Output: cdm_exploration/experiments/v2_profiling_bayes_adaptive500.{json,npz}

    python3 tools/run_profiling_bayes_adaptive500.py --smoke     # 2 LLMs
    python3 tools/run_profiling_bayes_adaptive500.py --part 0 ... --part 9     # 20 LLMs each, run in parallel
    python3 tools/run_profiling_bayes_adaptive500.py --merge
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
sys.path.insert(0, str(REPO / "tools"))
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
from run_profiling_bayes import BENCH, CKPT, DATA, EXP, SHRINK, Frozen, colcorr, fit_bayes

BUDGETS = [5, 10, 20, 50, 100, 200, 500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    seed_everything(42)
    torch.set_num_threads(1 if a.part is not None else 12)
    z = np.load(EXP / "v2_profiling_bayes.npz")
    zr = np.load(EXP / "v2_profiling_bayes_robustness.npz")
    main_res = json.load(open(EXP / "v2_profiling_bayes.json"))
    R = np.load(DATA / "response_matrix_v2_full.npy")
    assert set(np.unique(R)) <= {0, 1}
    R = R.astype(np.int8)          # 0/1 only; keeps ten parallel parts small in memory
    q = np.load(DATA / "qmatrix_v2_K100.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    n_llms, n_items = R.shape
    K = q.shape[1]
    pool, test = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    train_llms, _ = train_test_split(np.arange(n_llms), test_size=0.2, random_state=42)
    ok = verify_splits(pool, test, expected_seed=42, label="profiling_bayes_adaptive500")
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
    test_t = torch.as_tensor(test, dtype=torch.long)
    bench_te = np.array([items[i]["benchmark"] for i in test])
    nsk = q[pool].sum(axis=1)

    new_llms = z["new_llms"]
    L = 2 if a.smoke else len(new_llms)
    real = z["real"][:L]
    B = len(BUDGETS)
    auc, bpred, theta = np.zeros((B, L)), np.zeros((B, L, 5)), np.zeros((B, L, K))
    t0 = time.time()
    if a.merge:
        for i in range(10):
            part = np.load(EXP / f"v2_profiling_bayes_adaptive500_part{i}.npz")
            idx = part["llm_index"]
            assert np.array_equal(new_llms[idx], part["new_llms"])
            auc[:, idx], bpred[:, idx], theta[:, idx] = part["auc"], part["bpred"], part["theta"]
        assert (auc > 0).all(), "a part is missing"
    todo = [] if a.merge else (range(20 * a.part, 20 * a.part + 20) if a.part is not None else range(L))
    for li in todo:
        l = int(new_llms[li])
        remaining = np.ones(len(pool), bool)
        asked, r = [], mu.clone()
        for step in range(1, BUDGETS[-1] + 1):
            cand = pool[remaining]
            with torch.no_grad():
                p = model.prob(r, torch.as_tensor(cand, dtype=torch.long)).numpy()
            j = int(np.argmax(p * (1 - p) * np.maximum(nsk[remaining], 1)))
            asked.append(int(cand[j]))
            remaining[np.flatnonzero(remaining)[j]] = False
            r = fit_bayes(model, asked, R[l, asked], mu, P, r, max_iter=10)
            if step in BUDGETS:
                bi = BUDGETS.index(step)
                r_fin = fit_bayes(model, asked, R[l, asked], mu, P, mu, max_iter=100)
                with torch.no_grad():
                    pt = model.prob(r_fin, test_t).numpy()
                auc[bi, li] = roc_auc_score(R[l, test], pt)
                bpred[bi, li] = [pt[bench_te == b].mean() for b in BENCH]
                theta[bi, li] = torch.sigmoid(r_fin).numpy()
        if li == 0 or (li + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  LLM {li + 1}/{L}: {el / 60:.1f} min elapsed, about {el / (li + 1) * (L - li - 1) / 60:.0f} min left",
                  flush=True)

    if a.part is not None:
        idx = np.array(list(todo))
        np.savez_compressed(EXP / f"v2_profiling_bayes_adaptive500_part{a.part}.npz", auc=auc[:, idx],
                            bpred=bpred[:, idx], theta=theta[:, idx], llm_index=idx, new_llms=new_llms[idx])
        print(f"part {a.part} saved, {(time.time() - t0) / 60:.1f} min")
        return
    # reproduction check: budgets 5-100 must match the stored adaptive results
    old_budgets = [int(b) for b in z["budgets"]]
    repro = {}
    for n in (5, 10, 20, 50, 100):
        bo, bn = old_budgets.index(n), BUDGETS.index(n)
        old_err = np.abs(z["bpred_adaptive"][bo][:L] - real).mean()
        new_err = np.abs(bpred[bn] - real).mean()
        repro[n] = {"stored_err": float(old_err), "rerun_err": float(new_err),
                    "max_abs_llm_err_diff": float(np.abs(np.abs(z["bpred_adaptive"][bo][:L] - real).mean(1)
                                                         - np.abs(bpred[bn] - real).mean(1)).max())}
    print("reproduction:", json.dumps(repro, indent=1))
    if not a.smoke:
        assert all(abs(v["stored_err"] - v["rerun_err"]) < 0.001 for v in repro.values()), repro

    # measures at every budget, with paired differences against random (random averaged over its seeds)
    theta_np = zr["theta_noprior_full"][:L]
    theta_pr = z["theta_full"][:L]
    rng = np.random.default_rng(42)
    rows = []
    for bn, n in enumerate(BUDGETS):
        bo = old_budgets.index(n)
        e_ad = np.abs(bpred[bn] - real).mean(1)
        e_rd = np.abs(z["bpred_random"][:, bo, :L] - real).mean(2).mean(0)
        row = {"N": n, "adaptive_bench_err": float(e_ad.mean()), "adaptive_auc": float(auc[bn].mean()),
               "adaptive_agree_strict": float(colcorr(theta[bn], theta_np).mean()),
               "adaptive_agree_prior_ref": float(colcorr(theta[bn], theta_pr).mean()),
               "random_bench_err": float(e_rd.mean()),
               "random_agree_strict": float(np.mean([colcorr(z["theta_random"][s, bo, :L], theta_np).mean()
                                                     for s in range(z["theta_random"].shape[0])]))}
        de, da = [], []
        for _ in range(2000):
            ix = rng.integers(0, L, L)
            de.append((e_ad[ix] - e_rd[ix]).mean())
            da.append(colcorr(theta[bn][ix], theta_np[ix]).mean() -
                      np.mean([colcorr(z["theta_random"][s, bo, :L][ix], theta_np[ix]).mean()
                               for s in range(z["theta_random"].shape[0])]))
        row["adaptive_minus_random_bench_err"] = row["adaptive_bench_err"] - row["random_bench_err"]
        row["adaptive_minus_random_bench_err_ci95"] = [float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))]
        row["adaptive_minus_random_agree_strict"] = row["adaptive_agree_strict"] - row["random_agree_strict"]
        row["adaptive_minus_random_agree_strict_ci95"] = [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]
        rows.append(row)
    res = {"experiment": "profiling_bayes_adaptive500", "t_id": "T-137", "n_new_llms": L, "budgets": BUDGETS,
           "reproduction_of_stored_budgets": repro, "rows": rows, "minutes": (time.time() - t0) / 60,
           "verified": bool(ok)}
    tag = "_smoke" if a.smoke else ""
    (EXP / f"v2_profiling_bayes_adaptive500{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(EXP / f"v2_profiling_bayes_adaptive500{tag}.npz", auc=auc, bpred=bpred, theta=theta,
                        budgets=np.array(BUDGETS), new_llms=new_llms[:L])
    for row in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    print(f"{res['minutes']:.1f} min")
    if not a.smoke:
        log_experiment(name="run_profiling_bayes_adaptive500", config={"budgets": BUDGETS, "shrink": SHRINK},
                       results={"rows": rows, "reproduction": repro},
                       split_info={"n_pool_items": len(pool), "n_test_items": len(test), "random_state": 42},
                       verified=bool(ok))


if __name__ == "__main__":
    main()
