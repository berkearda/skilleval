"""Pareto routing on v2 full dataset (3811 LLMs x 9523 items).

Usage:
    python tools/pareto_routing_v2.py device=cpu
    python tools/pareto_routing_v2.py device=mps
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def get_predictions(net, test_items, text_embeddings, q_matrix, n_llms, device):
    """Get P(correct) for all LLMs on all test items."""
    net.eval()
    net = net.to(device)
    all_ids = torch.arange(n_llms, device=device)
    preds = {}
    for i, item_idx in enumerate(test_items):
        emb = torch.tensor(text_embeddings[int(item_idx)], dtype=torch.float32, device=device)
        q_row = torch.tensor(q_matrix[int(item_idx)], dtype=torch.float32, device=device)
        with torch.no_grad():
            p = net(all_ids, emb.unsqueeze(0).expand(n_llms, -1),
                    q_row.unsqueeze(0).expand(n_llms, -1)).cpu().numpy()
        preds[int(item_idx)] = p
        if (i + 1) % 500 == 0:
            print(f"  CDM predictions: {i+1}/{len(test_items)}", flush=True)
    return preds


def get_irt_predictions(model, test_items, n_llms, device):
    """Get IRT 2PL predictions for all test items."""
    model.eval()
    model = model.to(device)
    all_ids = torch.arange(n_llms, device=device)
    preds = {}
    with torch.no_grad():
        for item_idx in test_items:
            item_t = torch.full((n_llms,), int(item_idx), dtype=torch.int64, device=device)
            p = model(all_ids, item_t).cpu().numpy()
            preds[int(item_idx)] = p
    return preds


def threshold_sweep(preds, test_items, R, flops, thresholds):
    """Threshold routing: pick cheapest confident model, fallback to best-pred."""
    n_total = len(test_items)
    results = []
    for t in thresholds:
        correct = 0
        total_cost = 0.0
        for item_idx in test_items:
            p = preds[int(item_idx)]
            gt = R[:, int(item_idx)]
            above = np.where(p > t)[0]
            if len(above) > 0:
                chosen = above[np.argmin(flops[above])]
            else:
                chosen = int(np.argmax(p))
            total_cost += flops[chosen]
            if gt[chosen] > 0:
                correct += 1
        results.append({
            "threshold": float(t),
            "accuracy": correct / n_total,
            "mean_cost": total_cost / n_total,
        })
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.baselines import IRT2PL
    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
    from cdmeval.utils.visualization import SAVE_KW, setup_style

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load data ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embeddings = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)

    n_llms, n_items = R.shape
    n_skills = q_matrix.shape[1]
    print(f"  {n_llms} LLMs x {n_items} items, K={n_skills}", flush=True)

    # ── Split ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  Train: {len(train_items)}, Test: {len(test_items)}", flush=True)

    # ── FLOPs ──
    print("\nComputing FLOPs...", flush=True)
    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])

    # ── Load CDM checkpoint ──
    ckpt_path = Path("cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt")
    print(f"\nLoading CDM checkpoint: {ckpt_path}", flush=True)
    net = TextConditionedNet(n_skills, n_llms, 768)
    ckpt = load_checkpoint(ckpt_path, net, device)

    # ── CDM predictions ──
    print("\nGenerating CDM predictions...", flush=True)
    cdm_preds = get_predictions(net, test_items, text_embeddings, q_matrix, n_llms, device)
    print(f"  Done: {len(cdm_preds)} items", flush=True)

    # ── IRT predictions ──
    irt_path = Path("cdm_exploration/checkpoints/expanded/irt_2pl_v2.pt")
    print(f"\nLoading IRT checkpoint: {irt_path}", flush=True)
    irt_model = IRT2PL(n_llms, n_items)
    load_checkpoint(irt_path, irt_model, device)

    print("Generating IRT predictions...", flush=True)
    irt_preds = get_irt_predictions(irt_model, test_items, n_llms, device)
    print(f"  Done: {len(irt_preds)} items", flush=True)

    # ── Threshold sweep ──
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05).tolist()]
    print(f"\nThreshold sweep ({len(thresholds)} values)...", flush=True)

    cdm_sweep = threshold_sweep(cdm_preds, test_items, R, flops, thresholds)
    irt_sweep = threshold_sweep(irt_preds, test_items, R, flops, thresholds)

    # ── Baselines ──
    n_total = len(test_items)
    train_acc = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    strongest_correct = sum(R[strongest_idx, int(i)] for i in test_items)
    strongest_acc = strongest_correct / n_total
    strongest_cost = float(flops[strongest_idx])
    strongest_name = llm_names[strongest_idx]

    cheapest_idx = int(np.argmin(flops))
    cheapest_correct = sum(R[cheapest_idx, int(i)] for i in test_items)
    cheapest_acc = cheapest_correct / n_total
    cheapest_cost = float(flops[cheapest_idx])

    # ── Normalize ──
    cdm_rel = [{"t": p["threshold"],
                "cost_pct": p["mean_cost"] / strongest_cost * 100,
                "acc_pct": p["accuracy"] / strongest_acc * 100} for p in cdm_sweep]
    irt_rel = [{"t": p["threshold"],
                "cost_pct": p["mean_cost"] / strongest_cost * 100,
                "acc_pct": p["accuracy"] / strongest_acc * 100} for p in irt_sweep]

    # ── Find 90% accuracy threshold ──
    sweet_spot = None
    for p in cdm_rel:
        if p["acc_pct"] >= 90:
            sweet_spot = p
            break

    # ── Summary table ──
    print(f"\n{'='*85}", flush=True)
    print(f"PARETO ROUTING RESULTS (v2 dataset)", flush=True)
    print(f"{'='*85}", flush=True)
    print(f"{'Thresh':>7} {'CDM Acc%':>9} {'CDM Cost%':>10} {'IRT Acc%':>9} {'IRT Cost%':>10}", flush=True)
    print("-" * 85, flush=True)
    for c, i in zip(cdm_rel, irt_rel):
        print(f"  {c['t']:>5.2f} {c['acc_pct']:>9.1f} {c['cost_pct']:>10.1f} "
              f"{i['acc_pct']:>9.1f} {i['cost_pct']:>10.1f}", flush=True)

    print(f"\n  Strongest: {strongest_name}", flush=True)
    print(f"    Acc={strongest_acc:.4f}, Cost={strongest_cost:.0f} GFLOPs", flush=True)
    if sweet_spot:
        print(f"\n  KEY FINDING: {sweet_spot['acc_pct']:.0f}% accuracy at "
              f"{sweet_spot['cost_pct']:.0f}% cost (t={sweet_spot['t']:.2f})", flush=True)

    # ── Plot ──
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    cdm_x = [p["cost_pct"] for p in cdm_rel]
    cdm_y = [p["acc_pct"] for p in cdm_rel]
    irt_x = [p["cost_pct"] for p in irt_rel]
    irt_y = [p["acc_pct"] for p in irt_rel]

    # Sweet spot region
    ax.add_patch(mpatches.Rectangle((0, 90), 60, 10, alpha=0.08, color="#4C72B0", zorder=0))

    ax.plot(cdm_x, cdm_y, "o-", color="#4C72B0", lw=2, markersize=7,
            label="CDMEval (K=100)", zorder=4)
    ax.plot(irt_x, irt_y, "s--", color="#DD8452", lw=1.5, markersize=5,
            label="IRT 2PL", zorder=3)
    ax.plot(100, 100, "*", color="#C44E52", markersize=18, zorder=5,
            label=f"Strongest ({strongest_name.split('__')[-1][:20]})")

    # Annotate key CDM points
    txt_bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none")
    for p in cdm_rel:
        if p["t"] in {0.30, 0.50, 0.70, 0.90}:
            ax.annotate(f"t={p['t']:.1f}", xy=(p["cost_pct"], p["acc_pct"]),
                        xytext=(8, -12), textcoords="offset points",
                        fontsize=8, color="#4C72B0", bbox=txt_bbox)

    # 90% line
    ax.axhline(90, ls="--", color="#999999", lw=0.8, zorder=1)
    if sweet_spot:
        ax.axvline(sweet_spot["cost_pct"], ls="--", color="#999999", lw=0.8,
                   ymax=90/110, zorder=1)
        ax.annotate(
            f"{sweet_spot['acc_pct']:.0f}% accuracy\nat {sweet_spot['cost_pct']:.0f}% cost",
            xy=(sweet_spot["cost_pct"], 90), xytext=(75, 30),
            fontsize=11, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        )

    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Cost relative to strongest model (%)", fontsize=12)
    ax.set_ylabel("Accuracy relative to strongest model (%)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    out_fig = fig_dir / "fig_pareto_v2.pdf"
    fig.savefig(out_fig, **SAVE_KW)
    plt.close()
    print(f"\nSaved: {out_fig}", flush=True)

    # ── Save results ──
    save_data = {
        "dataset": "v2_full",
        "n_llms": n_llms, "n_items": n_items, "n_test": n_total,
        "strongest": {"name": strongest_name, "acc": strongest_acc, "cost": strongest_cost},
        "cheapest": {"acc": cheapest_acc, "cost": cheapest_cost},
        "cdm_sweep": cdm_sweep,
        "irt_sweep": irt_sweep,
        "cdm_normalized": cdm_rel,
        "irt_normalized": irt_rel,
        "sweet_spot": sweet_spot,
    }
    out_json = Path("cdm_exploration/experiments/v2_pareto_routing.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Saved: {out_json}", flush=True)

    # ── Verify and log ──
    verified = verify_splits(train_items, test_items, label="pareto_v2")
    log_experiment(
        name="pareto_routing_v2",
        config={"n_llms": n_llms, "n_items": n_items, "K": n_skills,
                "thresholds": thresholds, "device": device},
        results={"sweet_spot": sweet_spot,
                 "strongest_acc": strongest_acc,
                 "cdm_acc1_at_t50": cdm_sweep[8]["accuracy"] if len(cdm_sweep) > 8 else None,
                 "irt_acc1_at_t50": irt_sweep[8]["accuracy"] if len(irt_sweep) > 8 else None},
        split_info={"n_train": len(train_items), "n_test": n_total},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
