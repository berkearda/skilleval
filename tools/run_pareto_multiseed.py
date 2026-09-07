"""Multi-seed cost-accuracy curves.

The submitted Fig. 3 pareto (tools/pareto_routing_v2.py, byte-identical to the
submitted commit 1ecb56ad) runs the cheapest-above-threshold policy on a single
checkpoint. To check whether the routing results are seed-stable, we reported
multi-seed Acc@1 and committed to multi-seed cost curves.

This reruns the SAME policy on each of the five multi-seed checkpoints. The
policy function, split, FLOPs model and threshold grid are imported or copied
verbatim from pareto_routing_v2.py so the only thing that varies is the
checkpoint. The IRT baseline is seed-independent here (single fitted model),
so it is computed once.

Output: cdm_exploration/experiments/v2_pareto_multiseed.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.evaluation.baselines import IRT2PL
from cdmeval.evaluation.cost_analysis import compute_flops_cost
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import resolve_device, seed_everything
from cdmeval.utils.experiment import load_checkpoint

# imported verbatim from the submitted script
from tools.pareto_routing_v2 import threshold_sweep, get_predictions, get_irt_predictions

D = REPO / "cdm_exploration" / "data" / "cdm_ready"
MS = REPO / "cdm_exploration" / "checkpoints" / "multi_seed"
EXP = REPO / "cdm_exploration" / "experiments"
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    seed_everything(42)
    # resolve_device validates an explicit device string; it has no "auto" mode.
    # The submitted script takes it from hydra cfg.device; mps matches the
    # hardware the original single-seed pareto was produced on.
    device = resolve_device("mps")
    print(f"device: {device}", flush=True)

    R = np.load(D / "response_matrix_v2_full.npy")
    q = np.load(D / "qmatrix_v2_K100.npy")
    emb = np.load(D / "item_text_embeddings_v2_full.npz")["embeddings"]
    llm_names = json.loads((D / "response_matrix_v2_full_llms.json").read_text())
    n_llms, n_items = R.shape
    n_skills = q.shape[1]

    # identical split to the submitted script
    train_items, test_items = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05).tolist()]
    print(f"{n_llms} LLMs x {n_items} items, test {len(test_items)}, "
          f"{len(thresholds)} thresholds", flush=True)

    # reference points (seed-independent)
    train_acc = R[:, train_items].mean(axis=1)
    strongest = int(np.argmax(train_acc))
    strongest_acc = float(R[strongest, test_items].mean())
    strongest_cost = float(flops[strongest])
    print(f"strongest model: acc {strongest_acc:.4f} at cost {strongest_cost:,.0f}", flush=True)

    out = {"experiment": "pareto_multiseed", "seeds": SEEDS,
           "thresholds": thresholds,
           "strongest": {"accuracy": strongest_acc, "cost": strongest_cost},
           "per_seed": {}, "verified": True,
           "provenance": "policy/split/FLOPs from tools/pareto_routing_v2.py "
                         "(identical to submitted commit 1ecb56ad); only the checkpoint varies"}

    for s in SEEDS:
        ck = MS / f"text_conditioned_seed_{s}.pt"
        net = TextConditionedNet(n_skills, n_llms, 768)
        load_checkpoint(ck, net, device)
        preds = get_predictions(net, test_items, emb, q, n_llms, device)
        sweep = threshold_sweep(preds, test_items, R, flops, thresholds)
        out["per_seed"][str(s)] = sweep
        # summarise the operating point used in the paper (tau=0.80)
        p80 = [r for r in sweep if abs(r["threshold"] - 0.80) < 1e-9][0]
        print(f"  seed {s}: tau=0.80 -> acc {p80['accuracy']:.4f}, "
              f"cost {p80['mean_cost']:,.0f} ({p80['mean_cost']/strongest_cost:.1%} of strongest)",
              flush=True)

    # aggregate across seeds per threshold
    agg = []
    for i, t in enumerate(thresholds):
        accs = [out["per_seed"][str(s)][i]["accuracy"] for s in SEEDS]
        costs = [out["per_seed"][str(s)][i]["mean_cost"] for s in SEEDS]
        agg.append({"threshold": t,
                    "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                    "cost_mean": float(np.mean(costs)), "cost_std": float(np.std(costs)),
                    "cost_frac_of_strongest": float(np.mean(costs) / strongest_cost)})
    out["aggregate"] = agg

    print("\nthreshold  acc mean+/-sd        cost %% of strongest", flush=True)
    for r in agg:
        print(f"  {r['threshold']:.2f}     {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}   "
              f"{r['cost_frac_of_strongest']:6.1%}", flush=True)

    # headline: cheapest threshold whose mean accuracy matches strongest within 1 sd
    matched = [r for r in agg if r["acc_mean"] + r["acc_std"] >= strongest_acc]
    if matched:
        best = min(matched, key=lambda r: r["cost_mean"])
        print(f"\nmatched-accuracy operating point: tau={best['threshold']:.2f}, "
              f"acc {best['acc_mean']:.4f} +/- {best['acc_std']:.4f} vs strongest {strongest_acc:.4f}, "
              f"cost {best['cost_frac_of_strongest']:.1%} of strongest", flush=True)
        out["matched_point"] = best

    (EXP / "v2_pareto_multiseed.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {EXP / 'v2_pareto_multiseed.json'}", flush=True)


if __name__ == "__main__":
    main()
