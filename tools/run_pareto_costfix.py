#!/usr/bin/env python3
"""Cost-accuracy curves with corrected LLM prices (T-131). Nothing is trained; existing models are only evaluated.

Problem: `compute_flops_cost` reads the parameter count from the LLM name. Four names fool it, so four
7B/8B-class LLMs were priced as if they had a few million parameters. The cheapest-above-threshold router prefers
cheap LLMs, so it sent many questions to these "free" LLMs.

How the four are found (rule, not by eye): priced below 1,000 GFLOPs (under 1B parameters) AND overall accuracy above
0.39. Genuinely small LLMs score about 0.26 (90th percentile 0.30).
How they are re-priced (rule): the most common price among the 15 LLMs with the most similar right/wrong pattern over
all 9,523 questions.

Everything else is identical to the submitted protocol: split, thresholds, policy (threshold_sweep from
run_pareto_multibaseline.py), strongest single LLM chosen on training items.

Output: cdm_exploration/experiments/v2_pareto_costfix.json   (the submitted result files are not touched)

    python3 tools/run_pareto_costfix.py
"""
import collections
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
for _m in ("matplotlib", "matplotlib.pyplot"):            # not needed here; some tool modules import it
    try:
        __import__(_m)
    except ModuleNotFoundError:
        sys.modules[_m] = types.ModuleType(_m); sys.modules[_m].use = lambda *a, **k: None
from cdmeval.evaluation.cost_analysis import compute_flops_cost
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits
from tools.run_pareto_multibaseline import threshold_sweep, knn_predictions, irtnet_predictions

D = REPO / "cdm_exploration/data/cdm_ready"
CK = REPO / "cdm_exploration/checkpoints"
EXP = REPO / "cdm_exploration/experiments"
SEEDS = [42, 43, 44, 45, 46]
PRICE_FLOOR, ACC_FLAG, N_NEIGHBOURS = 1000.0, 0.39, 15


def predictions(ckpt_path, test_idx, n_llms, K, emb, Q):
    net = TextConditionedNet(K, n_llms, text_dim=768)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"])
    net.eval()
    te, tq, ids = torch.tensor(emb[test_idx], dtype=torch.float32), torch.tensor(Q[test_idx], dtype=torch.float32), torch.arange(n_llms)
    out = np.zeros((n_llms, len(test_idx)), dtype=np.float32)
    with torch.no_grad():
        for j in range(len(test_idx)):
            out[:, j] = net(ids, te[j].unsqueeze(0).expand(n_llms, -1), tq[j].unsqueeze(0).expand(n_llms, -1)).numpy()
    return out


def corrected_prices(names, flops, R):
    acc = R.mean(axis=1)
    flagged = [i for i in range(len(names)) if flops[i] < PRICE_FLOOR and acc[i] > ACC_FLAG]
    fixed, table = flops.copy(), []
    R8 = R.astype(np.int8)
    for i in flagged:
        agree = (R8 == R8[i]).mean(axis=1)
        nb = [j for j in np.argsort(-agree) if j != i and j not in flagged][:N_NEIGHBOURS]
        price, votes = collections.Counter(float(flops[j]) for j in nb).most_common(1)[0]
        fixed[i] = price
        table.append({"llm": names[i], "accuracy": float(acc[i]), "price_before": float(flops[i]), "price_after": price,
                      "neighbours_with_that_price": int(votes), "most_similar_llm": names[nb[0]],
                      "agreement_with_it": float(agree[nb[0]])})
    return fixed, table


def headline(sweep, strongest_acc, strongest_cost):
    pts = [{"threshold": r["threshold"], "accuracy": r["accuracy"], "cost_pct": 100 * r["mean_cost"] / strongest_cost,
            "acc_pct_of_strongest": 100 * r["accuracy"] / strongest_acc} for r in sweep]
    parity = [p for p in pts if p["accuracy"] >= strongest_acc]
    return {"points": pts, "cheapest_point_at_or_above_strongest": min(parity, key=lambda p: p["cost_pct"]) if parity else None}


def main():
    seed_everything(42)
    R = np.load(D / "response_matrix_v2_full.npy"); Q = np.load(D / "qmatrix_v2_K100.npy")
    emb = np.load(D / "item_text_embeddings_v2_full.npz")["embeddings"]
    names = json.load(open(D / "response_matrix_v2_full_llms.json"))
    n_llms, n_items = R.shape; K = Q.shape[1]
    train_idx, test_idx = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    ok = verify_splits(train_idx, test_idx, expected_seed=42, label="pareto_costfix")
    fl = compute_flops_cost(names, seq_length=512)
    flops = np.array([fl[n] for n in names], dtype=float)
    fixed, table = corrected_prices(names, flops, R)
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05).tolist()]
    R_test = R[:, test_idx]
    strongest = int(np.argmax(R[:, train_idx].mean(axis=1)))
    s_acc, s_cost = float(R_test[strongest].mean()), float(flops[strongest])
    assert fixed[strongest] == flops[strongest]

    preds = {"skilleval_main": predictions(CK / "expanded/text_conditioned_protocolB.pt", test_idx, n_llms, K, emb, Q)}
    for s in SEEDS:
        preds[f"skilleval_seed{s}"] = predictions(CK / f"multi_seed/text_conditioned_seed_{s}.pt", test_idx, n_llms, K, emb, Q)
    preds["knn_k10"] = knn_predictions(emb, R, train_idx, test_idx, K=10)
    try:
        preds["irtnet_d232"] = irtnet_predictions(232, 42, test_idx, n_llms, n_items, emb, "cpu")
    except Exception as e:                                   # the curve is only needed for the figure
        print("IrtNet predictions unavailable:", type(e).__name__, str(e)[:120], flush=True)

    res = {"experiment": "pareto_costfix", "thresholds": thresholds, "n_test_items": int(len(test_idx)),
           "strongest": {"llm": names[strongest], "accuracy": s_acc, "cost_gflops": s_cost},
           "flag_rule": {"price_below_gflops": PRICE_FLOOR, "accuracy_above": ACC_FLAG, "n_neighbours": N_NEIGHBOURS},
           "price_corrections": table, "methods": {}}
    for name, p in preds.items():
        before, after = threshold_sweep(p, R_test, flops, thresholds), threshold_sweep(p, R_test, fixed, thresholds)
        res["methods"][name] = {"submitted_prices": headline(before, s_acc, s_cost), "corrected_prices": headline(after, s_acc, s_cost)}
        print(f"{name}: done", flush=True)
    for key in ("submitted_prices", "corrected_prices"):
        agg = []
        for i, t in enumerate(thresholds):
            a = [res["methods"][f"skilleval_seed{s}"][key]["points"][i]["accuracy"] for s in SEEDS]
            c = [res["methods"][f"skilleval_seed{s}"][key]["points"][i]["cost_pct"] for s in SEEDS]
            agg.append({"threshold": t, "acc_mean": float(np.mean(a)), "acc_sd_sample": float(np.std(a, ddof=1)),
                        "cost_pct_mean": float(np.mean(c)), "cost_pct_sd_sample": float(np.std(c, ddof=1)),
                        "acc_pct_of_strongest": float(100 * np.mean(a) / s_acc)})
        par = [r for r in agg if r["acc_mean"] >= s_acc]
        res.setdefault("five_seed", {})[key] = {"points": agg, "cheapest_point_at_or_above_strongest": min(par, key=lambda r: r["cost_pct_mean"]) if par else None}
    res["verified"] = bool(ok)
    (EXP / "v2_pareto_costfix.json").write_text(json.dumps(res, indent=2))
    log_experiment(name="run_pareto_costfix", config={"seed": 42, "seeds": SEEDS, "flag_rule": res["flag_rule"]},
                   results={"price_corrections": table, "strongest": res["strongest"],
                            "main_parity_before": res["methods"]["skilleval_main"]["submitted_prices"]["cheapest_point_at_or_above_strongest"],
                            "main_parity_after": res["methods"]["skilleval_main"]["corrected_prices"]["cheapest_point_at_or_above_strongest"],
                            "five_seed_parity_before": res["five_seed"]["submitted_prices"]["cheapest_point_at_or_above_strongest"],
                            "five_seed_parity_after": res["five_seed"]["corrected_prices"]["cheapest_point_at_or_above_strongest"]},
                   split_info={"n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "random_state": 42}, verified=bool(ok))
    print(json.dumps({"price_corrections": table}, indent=1))


if __name__ == "__main__":
    main()
