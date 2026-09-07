"""Diagnose IRT vs CDM prediction distributions and validate the claim
that IRT's 'low-cost Pareto' is an artifact of degenerate single-model
routing rather than smart per-query routing.

Runs on the full v2 test split with verify_splits() and log_experiment().
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
    from cdmeval.evaluation.baselines import IRT2PL
    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load data (same as pareto_routing_v2.py) ──
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q = np.load(data_dir / "qmatrix_v2_K100.npy")
    emb = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    llm_names = json.load(open(data_dir / "response_matrix_v2_full_llms.json"))
    n_llms, n_items = R.shape
    n_skills = q.shape[1]
    print(f"  {n_llms} LLMs x {n_items} items, K={n_skills}", flush=True)

    # ── Split (identical to pareto_routing_v2.py) ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_items)}, Test: {len(test_items)}", flush=True)

    # ── FLOPs ──
    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])

    # ── Load models ──
    net = TextConditionedNet(n_skills, n_llms, 768)
    cdm_ckpt = load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        net, device,
    )
    net.eval().to(device)

    irt = IRT2PL(n_llms, n_items)
    irt_ckpt = load_checkpoint(
        "cdm_exploration/checkpoints/expanded/irt_2pl_v2.pt",
        irt, device,
    )
    irt.eval().to(device)

    cdm_auc = float(cdm_ckpt.get("val_auc", float("nan")))
    irt_auc = float(irt_ckpt.get("val_auc", float("nan")))

    # ── Strongest model on TRAIN items ──
    strongest_idx = int(np.argmax(R[:, train_items.astype(int)].mean(axis=1)))
    strongest_name = llm_names[strongest_idx]
    strongest_test_acc = float(R[strongest_idx, test_items.astype(int)].mean())
    strongest_cost = float(flops[strongest_idx])
    print(f"\n  Strongest (on train): {strongest_name}", flush=True)
    print(f"    Test acc = {strongest_test_acc:.4f}  Cost = {strongest_cost:.0f} GFLOPs",
          flush=True)

    # ── Predictions on FULL test set ──
    print("\nGenerating predictions on full test set...", flush=True)
    all_ids = torch.arange(n_llms, device=device)
    C_list, I_list = [], []
    with torch.no_grad():
        for i, item_idx in enumerate(test_items):
            em = torch.tensor(emb[int(item_idx)], dtype=torch.float32, device=device)
            qr = torch.tensor(q[int(item_idx)], dtype=torch.float32, device=device)
            p_cdm = net(all_ids,
                        em.unsqueeze(0).expand(n_llms, -1),
                        qr.unsqueeze(0).expand(n_llms, -1)).cpu().numpy()
            it_t = torch.full((n_llms,), int(item_idx), dtype=torch.int64, device=device)
            p_irt = irt(all_ids, it_t).cpu().numpy()
            C_list.append(p_cdm)
            I_list.append(p_irt)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(test_items)}", flush=True)
    C = np.array(C_list)
    I = np.array(I_list)
    GT = R[:, test_items.astype(int)].T  # (n_test, n_llms)

    # ── Distribution & calibration ──
    gt_rate = float(GT.mean())
    cdm_mean = float(C.mean())
    irt_mean = float(I.mean())
    print(f"\nP(correct) distributions (n_pairs={C.size:,}):")
    print(f"  Ground-truth rate:  {gt_rate:.4f}")
    print(f"  CDM mean:           {cdm_mean:.4f}   bias={cdm_mean-gt_rate:+.4f}")
    print(f"  IRT mean:           {irt_mean:.4f}   bias={irt_mean-gt_rate:+.4f}")

    # ── Routing diversity: at each threshold, how many unique LLMs are chosen? ──
    print("\nRouting-diversity audit:")
    print(f"  {'t':>5}  {'CDM unique':>10}  {'CDM top-freq':>12}  "
          f"{'IRT unique':>10}  {'IRT top-freq':>12}")
    diversity = {}
    for t in [0.30, 0.50, 0.55, 0.65, 0.70, 0.90]:
        def pick(P, t):
            choices = []
            for i in range(len(P)):
                above = np.where(P[i] > t)[0]
                if len(above):
                    choices.append(above[np.argmin(flops[above])])
                else:
                    choices.append(int(np.argmax(P[i])))
            return np.array(choices)

        c_c = pick(C, t)
        c_i = pick(I, t)
        uc, cc = np.unique(c_c, return_counts=True)
        ui, ci = np.unique(c_i, return_counts=True)
        cdm_top = cc.max() / len(c_c)
        irt_top = ci.max() / len(c_i)
        print(f"  {t:5.2f}  {len(uc):>10d}  {cdm_top:>12.1%}  "
              f"{len(ui):>10d}  {irt_top:>12.1%}")
        diversity[f"{t:.2f}"] = {
            "cdm_unique_llms": int(len(uc)),
            "cdm_top_llm_share": float(cdm_top),
            "irt_unique_llms": int(len(ui)),
            "irt_top_llm_share": float(irt_top),
        }

    # ── Key claim: IRT's "cheap Pareto point" == single-model selection ──
    t_irt = 0.65
    choices_irt = []
    for i in range(len(I)):
        above = np.where(I[i] > t_irt)[0]
        if len(above):
            choices_irt.append(above[np.argmin(flops[above])])
        else:
            choices_irt.append(int(np.argmax(I[i])))
    choices_irt = np.array(choices_irt)
    u, c = np.unique(choices_irt, return_counts=True)
    top_llm = int(u[np.argmax(c)])
    top_share = float(c.max() / len(choices_irt))
    top_name = llm_names[top_llm]
    top_cost = float(flops[top_llm])
    top_test_acc = float(R[top_llm, test_items.astype(int)].mean())
    top_acc_pct = top_test_acc / strongest_test_acc * 100
    top_cost_pct = top_cost / strongest_cost * 100
    print(f"\n  IRT t={t_irt} routes {top_share:.1%} of test items to a single LLM:",
          flush=True)
    print(f"    {top_name}", flush=True)
    print(f"    Test acc:       {top_test_acc:.4f}  ({top_acc_pct:.1f}% of strongest)",
          flush=True)
    print(f"    Cost:           {top_cost:.0f} GFLOPs ({top_cost_pct:.2f}% of strongest)",
          flush=True)
    print(f"    -> IRT's 'Pareto' point IS just this single model.", flush=True)

    # ── Cheapest single-model Pareto baseline: for each LLM, compute (cost, acc) ──
    per_llm_test_acc = R[:, test_items.astype(int)].mean(axis=1)
    pareto_individual = []
    for idx in range(n_llms):
        pareto_individual.append({
            "llm": llm_names[idx],
            "cost_gflops": float(flops[idx]),
            "test_acc": float(per_llm_test_acc[idx]),
            "cost_pct": float(flops[idx] / strongest_cost * 100),
            "acc_pct": float(per_llm_test_acc[idx] / strongest_test_acc * 100),
        })

    # Find single models that dominate the IRT claim: acc >= 96% at cost <= 3%
    dominators = [p for p in pareto_individual
                  if p["acc_pct"] >= 96.0 and p["cost_pct"] <= 3.0]
    dominators.sort(key=lambda p: (-p["acc_pct"], p["cost_pct"]))
    print(f"\n  Single LLMs achieving >=96% acc at <=3% cost: {len(dominators)}",
          flush=True)
    for p in dominators[:5]:
        print(f"    {p['llm']:<60s}  acc={p['acc_pct']:.1f}%  cost={p['cost_pct']:.2f}%",
              flush=True)

    # ── Validate splits ──
    verified = verify_splits(
        np.array(train_items), np.array(test_items),
        label="pareto_routing_diagnostic",
    )

    # ── Log experiment ──
    results = {
        "cdm_val_auc": cdm_auc,
        "irt_val_auc": irt_auc,
        "ground_truth_correct_rate": gt_rate,
        "cdm_pred_mean": cdm_mean,
        "cdm_calibration_bias": cdm_mean - gt_rate,
        "irt_pred_mean": irt_mean,
        "irt_calibration_bias": irt_mean - gt_rate,
        "routing_diversity": diversity,
        "irt_t065_degenerate_routing": {
            "threshold": t_irt,
            "dominant_llm": top_name,
            "share_routed_to_dominant": top_share,
            "dominant_llm_test_acc": top_test_acc,
            "dominant_llm_acc_pct_of_strongest": top_acc_pct,
            "dominant_llm_cost_pct_of_strongest": top_cost_pct,
        },
        "strongest": {
            "llm": strongest_name,
            "test_acc": strongest_test_acc,
            "cost_gflops": strongest_cost,
        },
        "single_model_dominators_at_irt_pareto_point": dominators[:10],
    }
    log_experiment(
        name="pareto_routing_diagnostic",
        config={
            "n_llms": n_llms, "n_items": n_items, "K": n_skills,
            "seed": 42, "device": device,
            "cdm_checkpoint": "text_conditioned_protocolB.pt",
            "irt_checkpoint": "irt_2pl_v2.pt",
        },
        results=results,
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=verified,
    )

    # ── Save standalone JSON too ──
    out = Path("cdm_exploration/experiments/v2_pareto_diagnostic.json")
    json.dump({"config": {"n_llms": n_llms, "n_items": n_items, "K": n_skills},
               "results": results}, open(out, "w"), indent=2, default=str)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
