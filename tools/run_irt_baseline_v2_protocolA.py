"""IRT 2PL baseline under Protocol A (interaction-wise random triplet split).

Fills the ID column of Table 1 for the IRT 2PL row. The OOD column is
intentionally left blank because standard 2PL has no item features and so
cannot generalise to held-out items.

Training set: 80% of (LLM, item) triplets, sampled at random (seed 42).
Val set: 10% of triplets (early-stopping signal).
Test set: 10% of triplets.

Routing Acc@k is *not* computed here. Per the 2026-04-25 request, routing
under Protocol A is undefined (every item has both train and test triplets);
the routing IRT 2PL number reported in Table 3 is the Protocol B / OOD
result already in v2_irt_baseline.json (acc@1=0.611), kept with a
"theta-ranking only" caveat.

Usage:
    python tools/run_irt_baseline_v2_protocolA.py device=mps
    python tools/run_irt_baseline_v2_protocolA.py device=cuda \\
        model.epochs=15 model.batch_size=16384
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.baselines import IRT2PL
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint

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

    # ── Protocol A: 80/10/10 random split over ALL triplets ──
    print("\nBuilding all triplets...")
    all_llm = np.repeat(np.arange(n_llms, dtype=np.int32), n_items)
    all_item = np.tile(np.arange(n_items, dtype=np.int32), n_llms)
    all_label = R[all_llm, all_item].astype(np.float32)
    n_total = len(all_label)
    print(f"  Total triplets: {n_total:,}")

    all_idx = np.arange(n_total, dtype=np.int64)
    trvl_idx, te_idx = train_test_split(all_idx, test_size=0.10, random_state=42)
    tr_idx, va_idx = train_test_split(trvl_idx, test_size=10.0 / 90.0,
                                      random_state=42)
    print(f"  Protocol A split: train={len(tr_idx):,}  "
          f"val={len(va_idx):,}  test={len(te_idx):,}")

    # Materialise tensors once (avoid per-epoch numpy slicing).
    s_all = torch.from_numpy(all_llm.astype(np.int64))
    i_all = torch.from_numpy(all_item.astype(np.int64))
    y_all = torch.from_numpy(all_label)
    tr_t = torch.from_numpy(tr_idx)
    va_t = torch.from_numpy(va_idx)
    te_t = torch.from_numpy(te_idx)

    s_tr, i_tr, y_tr = s_all[tr_t], i_all[tr_t], y_all[tr_t]
    s_va, i_va, y_va = s_all[va_t], i_all[va_t], y_all[va_t]
    s_te, i_te, y_te = s_all[te_t], i_all[te_t], y_all[te_t]
    del s_all, i_all, y_all, all_llm, all_item, all_label

    # ── Model ──
    epochs = int(cfg.model.epochs)
    lr = float(cfg.model.lr)
    bs = int(cfg.model.batch_size)
    patience = int(getattr(cfg.model, "early_stopping_patience", 3))
    print(f"\nIRT 2PL config: epochs={epochs}, lr={lr}, batch={bs}, "
          f"patience={patience}")

    model = IRT2PL(n_llms, n_items).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    n_train = len(tr_t)

    def eval_loader(s_, i_, y_, bs_eval: int = 65536):
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(y_), bs_eval):
                end = min(start + bs_eval, len(y_))
                p = model(s_[start:end].to(device),
                           i_[start:end].to(device)).cpu().numpy()
                preds.append(p)
        preds = np.concatenate(preds)
        y_np = y_.numpy()
        auc = roc_auc_score(y_np, preds)
        acc = accuracy_score(y_np, (preds >= 0.5).astype(int))
        rmse = float(np.sqrt(((y_np - preds) ** 2).mean()))
        return auc, acc, rmse, preds

    best_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train)
        losses = []
        for start in range(0, n_train, bs):
            idx = perm[start:start + bs]
            s_b = s_tr[idx].to(device)
            i_b = i_tr[idx].to(device)
            y_b = y_tr[idx].to(device)
            pred = model(s_b, i_b)
            loss = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        va_auc, va_acc, _, _ = eval_loader(s_va, i_va, y_va)
        avg_loss = float(np.mean(losses))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch + 1:>2}/{epochs}: loss={avg_loss:.4f} "
              f"val_auc={va_auc:.4f} val_acc={va_acc:.4f} "
              f"({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "loss": avg_loss,
                         "val_auc": va_auc, "val_acc": va_acc,
                         "elapsed_s": elapsed})

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stop at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Restored best (val_auc={best_auc:.4f})")

    # ── Final test evaluation ──
    print("\nEvaluating on Protocol A test triplets...")
    test_auc, test_acc, test_rmse, _ = eval_loader(s_te, i_te, y_te)
    print(f"  test_auc={test_auc:.4f}  test_acc={test_acc:.4f}  "
          f"test_rmse={test_rmse:.4f}")

    # ── Save ──
    results = {
        "dataset": "v2_full",
        "model": "IRT_2PL",
        "protocol": "A",
        "split": "interaction_wise_80_10_10",
        "seed": 42,
        "n_llms": n_llms,
        "n_items": n_items,
        "n_total_triplets": int(n_total),
        "n_train_triplets": int(len(tr_t)),
        "n_val_triplets": int(len(va_t)),
        "n_test_triplets": int(len(te_t)),
        "epochs_trained": len(history),
        "epochs_max": epochs,
        "lr": lr,
        "batch_size": bs,
        "best_val_auc": float(best_auc),
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "test_rmse": float(test_rmse),
        "trainable_params": n_params,
        "history": history,
    }

    out_path = Path("cdm_exploration/experiments/v2_irt_baseline_protocolA.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    ckpt_dir = Path("cdm_exploration/checkpoints/expanded")
    save_checkpoint(
        model, ckpt_dir / "irt_2pl_v2_protocolA.pt",
        config={"protocol": "A", "epochs": len(history), "lr": lr,
                 "batch_size": bs, "n_llms": n_llms, "n_items": n_items},
        train_items=np.arange(n_items),  # all items participate
        test_items=np.arange(n_items),
        val_auc=best_auc, epoch=len(history),
    )

    log_experiment(
        name="irt_baseline_v2_protocolA",
        config={"model": "IRT_2PL", "protocol": "A", "n_llms": n_llms,
                 "n_items": n_items, "epochs": len(history), "lr": lr,
                 "batch_size": bs, "device": device},
        results=results,
        split_info={"split": "interaction_wise_80_10_10",
                     "n_train_triplets": len(tr_t),
                     "n_val_triplets": len(va_t),
                     "n_test_triplets": len(te_t)},
        verified=True,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
