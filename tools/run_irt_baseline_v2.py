"""IRT 2PL baseline on the v2 full dataset (3811 LLMs x 9523 items).

Answers: does a single scalar ability per LLM compete with 100-skill CDM?

Usage:
    python tools/run_irt_baseline_v2.py device=cuda
    python tools/run_irt_baseline_v2.py device=cpu model.epochs=5
"""

import json
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.baselines import IRT2PL, _train_irt
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint, verify_splits

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # ── Load v2 data ──
    print("\nLoading v2 full dataset...")
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)

    n_llms, n_items = R.shape
    print(f"  Response matrix: {n_llms} LLMs x {n_items} items")

    # ── Protocol B split ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(
        all_items, test_size=0.2, random_state=42
    )
    print(f"  Evaluating on {len(test_items)} test items, {len(train_items)} train items")

    # ── Build triplets from training items only ──
    print("\nBuilding training triplets...")
    student_ids = np.repeat(np.arange(n_llms), len(train_items))
    item_ids = np.tile(train_items.astype(int), n_llms)
    scores = R[
        np.repeat(np.arange(n_llms), len(train_items)),
        np.tile(train_items.astype(int), n_llms),
    ].astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])
    print(f"  Training triplets: {len(triplets):,}")

    # Train/val split
    idx = np.arange(len(triplets))
    tr_idx, va_idx = train_test_split(idx, test_size=0.1, random_state=42)
    print(f"  Train: {len(tr_idx):,}, Val: {len(va_idx):,}")

    # ── Train IRT 2PL ──
    epochs = cfg.model.epochs
    lr = cfg.model.lr
    bs = cfg.model.batch_size
    print(f"\n{'='*60}")
    print(f"Training IRT 2PL ({n_llms} LLMs, {n_items} items)")
    print(f"  P(correct) = sigmoid(a_j * (theta_s - b_j))")
    print(f"  theta_s: scalar ability per LLM (1-dim)")
    print(f"  a_j, b_j: per-item discrimination and difficulty")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {bs}")
    print(f"{'='*60}")

    model = IRT2PL(n_llms, n_items)
    model = _train_irt(
        model, triplets[tr_idx], triplets[va_idx],
        epochs=epochs, lr=lr, batch_size=bs, device=device,
    )

    # ── Evaluate: AUC on test triplets ──
    print("\nEvaluating on test items...")
    from sklearn.metrics import roc_auc_score, accuracy_score

    model.eval()
    model = model.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    # Test AUC: predict all (LLM, test_item) pairs
    y_true_all = []
    y_pred_all = []
    batch_size_eval = 4096

    test_student_ids = np.repeat(np.arange(n_llms), len(test_items))
    test_item_ids = np.tile(test_items.astype(int), n_llms)
    test_scores = R[test_student_ids, test_item_ids].astype(float)

    with torch.no_grad():
        for start in range(0, len(test_student_ids), batch_size_eval):
            end = min(start + batch_size_eval, len(test_student_ids))
            s = torch.tensor(test_student_ids[start:end], dtype=torch.int64, device=device)
            i = torch.tensor(test_item_ids[start:end], dtype=torch.int64, device=device)
            p = model(s, i).cpu().numpy()
            y_pred_all.extend(p.tolist())
            y_true_all.extend(test_scores[start:end].tolist())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    test_auc = roc_auc_score(y_true_all, y_pred_all)
    test_acc = accuracy_score(y_true_all, (y_pred_all >= 0.5).astype(int))
    test_rmse = float(np.sqrt(((y_true_all - y_pred_all) ** 2).mean()))
    print(f"  Test AUC={test_auc:.4f}, Acc={test_acc:.4f}, RMSE={test_rmse:.4f}")

    # ── Routing evaluation ──
    print("\nRouting evaluation...")
    n_test = len(test_items)
    accs_at = {1: 0, 3: 0, 5: 0, 10: 0}

    with torch.no_grad():
        for item_idx in test_items:
            gt = R[:, int(item_idx)]
            if gt.sum() == 0:
                continue

            item_t = torch.full(
                (n_llms,), int(item_idx), dtype=torch.int64, device=device
            )
            preds = model(all_llm_ids, item_t).cpu().numpy()
            ranking = np.argsort(-preds)

            for k in accs_at:
                if gt[ranking[:k]].sum() > 0:
                    accs_at[k] += 1

    routing = {f"acc@{k}": accs_at[k] / n_test for k in accs_at}
    print(f"  Routing: " + ", ".join(f"@{k}={routing[f'acc@{k}']:.4f}" for k in accs_at))

    # Strongest model baseline
    train_acc = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    strongest_correct = sum(R[strongest_idx, int(i)] for i in test_items)
    strongest_acc1 = strongest_correct / n_test
    print(f"  Strongest model: {llm_names[strongest_idx]}")
    print(f"  Strongest Acc@1: {strongest_acc1:.4f}")

    # ── Comparison table ──
    print(f"\n{'='*75}")
    print(f"COMPARISON: IRT 2PL vs CDMEval vs Strongest")
    print(f"{'='*75}")
    print(f"{'Method':<30} {'AUC':>8} {'Acc@1':>8} {'Acc@5':>8} {'Acc@10':>8}")
    print("-" * 75)
    print(f"{'Strongest model':<30} {'--':>8} {strongest_acc1:>8.4f} {'--':>8} {'--':>8}")
    print(f"{'IRT 2PL (1-dim ability)':<30} {test_auc:>8.4f} {routing['acc@1']:>8.4f} {routing['acc@5']:>8.4f} {routing['acc@10']:>8.4f}")
    print(f"{'CDMEval (K=100 skills)':<30} {'0.7170':>8} {'0.6567':>8} {'0.8294':>8} {'0.8751':>8}")
    print()

    cdm_acc1 = 0.6567
    irt_acc1 = routing["acc@1"]
    if cdm_acc1 > irt_acc1:
        print(f"  CDMEval wins by {cdm_acc1 - irt_acc1:+.4f} on Acc@1")
        print(f"  Multi-skill decomposition adds value over single-dimension IRT")
    else:
        print(f"  IRT 2PL matches or beats CDMEval by {irt_acc1 - cdm_acc1:+.4f}")
        print(f"  Single ability dimension may suffice for this dataset")

    # ── Save results ──
    results = {
        "dataset": "v2_full",
        "model": "IRT_2PL",
        "n_llms": n_llms,
        "n_items": n_items,
        "n_test_items": n_test,
        "epochs": epochs,
        "lr": lr,
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "test_rmse": float(test_rmse),
        **{f"routing_{k}": v for k, v in routing.items()},
        "strongest_model": llm_names[strongest_idx],
        "strongest_acc1": float(strongest_acc1),
        "cdmeval_acc1": cdm_acc1,
        "advantage_over_irt": float(cdm_acc1 - irt_acc1),
    }

    out_path = Path("cdm_exploration/experiments/v2_irt_baseline.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Save checkpoint ──
    ckpt_dir = Path("cdm_exploration/checkpoints/expanded")
    save_checkpoint(
        model, ckpt_dir / "irt_2pl_v2.pt",
        config={"n_llms": n_llms, "n_items": n_items, "epochs": epochs, "lr": lr},
        train_items=train_items, test_items=test_items,
        val_auc=test_auc, epoch=epochs,
    )

    # ── Verify and log ──
    verified = verify_splits(train_items, test_items, label="irt_baseline_v2")
    log_experiment(
        name="irt_baseline_v2",
        config={"model": "IRT_2PL", "n_llms": n_llms, "n_items": n_items,
                "epochs": epochs, "lr": lr, "device": device},
        results=results,
        split_info={"n_train_items": len(train_items),
                    "n_test_items": len(test_items), "n_llms": n_llms},
        verified=verified,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
