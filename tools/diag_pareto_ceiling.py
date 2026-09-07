"""Explore what's achievable beyond CDMEval's tau=0.95 ceiling.

Computes three additional reference points on the same test split:
  (1) Extended tau sweep: tau in {0.97, 0.99, 1.0}
  (2) Argmax-per-query router: ignore threshold, always pick argmax P
  (3) Oracle upper bound: at-least-one-correct per query

Logs to experiment_log.json with split verification.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q = np.load(data_dir / "qmatrix_v2_K100.npy")
    emb = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    llm_names = json.load(open(data_dir / "response_matrix_v2_full_llms.json"))
    n_llms, n_items = R.shape
    n_skills = q.shape[1]
    print(f"  {n_llms} LLMs x {n_items} items, K={n_skills}", flush=True)

    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_items)}, Test: {len(test_items)}", flush=True)

    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])

    net = TextConditionedNet(n_skills, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, device,
    )
    net.eval().to(device)

    # Strongest reference (on train)
    strongest_idx = int(np.argmax(R[:, train_items.astype(int)].mean(axis=1)))
    strongest_name = llm_names[strongest_idx]
    strongest_test_acc = float(R[strongest_idx, test_items.astype(int)].mean())
    strongest_cost = float(flops[strongest_idx])
    print(f"\n  Strongest: {strongest_name}", flush=True)
    print(f"    test_acc={strongest_test_acc:.4f}  cost={strongest_cost:.0f}",
          flush=True)

    # Predict on full test set
    print("\nGenerating CDM predictions...", flush=True)
    all_ids = torch.arange(n_llms, device=device)
    P_list = []
    with torch.no_grad():
        for i, item_idx in enumerate(test_items):
            em = torch.tensor(emb[int(item_idx)], dtype=torch.float32, device=device)
            qr = torch.tensor(q[int(item_idx)], dtype=torch.float32, device=device)
            p = net(all_ids,
                    em.unsqueeze(0).expand(n_llms, -1),
                    qr.unsqueeze(0).expand(n_llms, -1)).cpu().numpy()
            P_list.append(p)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(test_items)}", flush=True)
    P = np.array(P_list)  # (n_test, n_llms)
    GT = R[:, test_items.astype(int)].T  # (n_test, n_llms)
    n_test = len(test_items)

    def route_at_threshold(t):
        """Same logic as pareto_routing_v2.py."""
        correct = 0
        total_cost = 0.0
        for i in range(n_test):
            above = np.where(P[i] > t)[0]
            if len(above) > 0:
                chosen = above[np.argmin(flops[above])]
            else:
                chosen = int(np.argmax(P[i]))
            total_cost += flops[chosen]
            if GT[i, chosen] > 0:
                correct += 1
        return correct / n_test, total_cost / n_test

    # (1) Extended tau sweep
    print("\n== Extended tau sweep ==", flush=True)
    ext_points = []
    for t in [0.95, 0.97, 0.99, 1.0, 1.01]:
        acc, cost = route_at_threshold(t)
        acc_pct = acc / strongest_test_acc * 100
        cost_pct = cost / strongest_cost * 100
        ext_points.append({"t": t, "acc": acc, "cost": cost,
                           "acc_pct": acc_pct, "cost_pct": cost_pct})
        print(f"  t={t:.2f}  acc={acc:.4f} ({acc_pct:6.2f}% of strongest)  "
              f"cost={cost:.1f} ({cost_pct:6.2f}% of strongest)", flush=True)

    # (2) Argmax-per-query router: for each item, pick argmax P (ignore threshold)
    print("\n== Argmax-per-query router (no threshold) ==", flush=True)
    argmax_idx = np.argmax(P, axis=1)
    argmax_correct = int(sum(GT[i, argmax_idx[i]] for i in range(n_test)))
    argmax_cost = float(flops[argmax_idx].mean())
    argmax_acc = argmax_correct / n_test
    argmax_acc_pct = argmax_acc / strongest_test_acc * 100
    argmax_cost_pct = argmax_cost / strongest_cost * 100
    print(f"  acc={argmax_acc:.4f} ({argmax_acc_pct:.2f}% of strongest)  "
          f"cost={argmax_cost:.1f} ({argmax_cost_pct:.2f}% of strongest)",
          flush=True)

    # (3) Oracle upper bound: any LLM gets it right
    print("\n== Oracle ceiling (at-least-one-correct) ==", flush=True)
    oracle_correct = int((GT.sum(axis=1) > 0).sum())
    oracle_acc = oracle_correct / n_test
    oracle_acc_pct = oracle_acc / strongest_test_acc * 100
    print(f"  acc={oracle_acc:.4f} ({oracle_acc_pct:.2f}% of strongest)",
          flush=True)
    print(f"    (cost undefined: oracle is a theoretical bound)", flush=True)

    # How close does CDMEval get to oracle?
    cdm_ceiling_acc_pct = ext_points[0]["acc_pct"]  # tau=0.95
    fraction_of_oracle = cdm_ceiling_acc_pct / oracle_acc_pct
    gap_to_oracle = oracle_acc_pct - cdm_ceiling_acc_pct
    print(f"\n  CDMEval tau=0.95 captures "
          f"{fraction_of_oracle:.1%} of oracle", flush=True)
    print(f"  Gap to oracle: {gap_to_oracle:.1f} pp", flush=True)

    # Oracle-cost under "cheapest correct LLM" per query (a tighter oracle)
    # for items where at least one LLM is correct, use cheapest correct
    print("\n== Oracle (cheapest-correct-per-query) ==", flush=True)
    o_cost = 0.0
    o_corr = 0
    for i in range(n_test):
        correct_llms = np.where(GT[i] > 0)[0]
        if len(correct_llms):
            cheapest_correct = correct_llms[np.argmin(flops[correct_llms])]
            o_cost += flops[cheapest_correct]
            o_corr += 1
        else:
            # no LLM gets it right; router can't win -- pick argmax flops-wise
            chosen = int(np.argmax(P[i]))
            o_cost += flops[chosen]
    o_acc = o_corr / n_test
    o_acc_pct = o_acc / strongest_test_acc * 100
    o_cost_avg = o_cost / n_test
    o_cost_pct = o_cost_avg / strongest_cost * 100
    print(f"  acc={o_acc:.4f} ({o_acc_pct:.2f}% of strongest)  "
          f"cost={o_cost_avg:.1f} ({o_cost_pct:.2f}% of strongest)", flush=True)

    # Verify + log
    verified = verify_splits(
        np.array(train_items), np.array(test_items),
        label="pareto_ceiling_diagnostic",
    )

    log_experiment(
        name="pareto_ceiling_diagnostic",
        config={
            "n_llms": n_llms, "n_items": n_items, "K": n_skills,
            "seed": 42, "device": device,
        },
        results={
            "strongest_test_acc": strongest_test_acc,
            "strongest_cost_gflops": strongest_cost,
            "extended_tau_sweep": ext_points,
            "argmax_router": {
                "acc": argmax_acc, "cost": argmax_cost,
                "acc_pct": argmax_acc_pct, "cost_pct": argmax_cost_pct,
            },
            "oracle_at_least_one_correct": {
                "acc": oracle_acc, "acc_pct": oracle_acc_pct,
            },
            "oracle_cheapest_correct": {
                "acc": o_acc, "cost": o_cost_avg,
                "acc_pct": o_acc_pct, "cost_pct": o_cost_pct,
            },
            "cdm_ceiling_fraction_of_oracle": fraction_of_oracle,
            "cdm_ceiling_gap_to_oracle_pp": gap_to_oracle,
        },
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
