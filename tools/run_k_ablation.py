"""K ablation: train NCDM with different K values on v2 dataset.

Usage:
    python tools/run_k_ablation.py device=cuda '+ablation.K=50'
    python tools/run_k_ablation.py device=cuda '+ablation.K=150'
    python tools/run_k_ablation.py device=cuda '+ablation.K=200'
    python tools/run_k_ablation.py device=cuda '+ablation.K=300'
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
    from cdmeval.data.dataloader import make_text_dataloader
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint, verify_splits

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)

    K = cfg.ablation.K if hasattr(cfg, "ablation") else 100
    print(f"Device: {device}, K={K}", flush=True)

    # ── Load data ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / f"qmatrix_v2_K{K}.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)

    n_llms, n_items = R.shape
    n_skills = q_matrix.shape[1]
    text_dim = text_embs.shape[1]
    print(f"  {n_llms} LLMs x {n_items} items, K={n_skills}", flush=True)
    assert n_skills == K, f"Q-matrix K={n_skills} != requested K={K}"

    # ── Split ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  Evaluating on {len(test_items)} test items, {len(train_items)} train items", flush=True)

    # ── Build triplets ──
    print("Building triplets...", flush=True)
    student_ids = np.repeat(np.arange(n_llms), n_items)
    item_ids = np.tile(np.arange(n_items), n_llms)
    scores = R.ravel().astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])

    train_mask = np.isin(triplets[:, 1].astype(int), train_items)
    train_triplets = triplets[train_mask]
    test_triplets = triplets[~train_mask]

    tv_idx = np.arange(len(train_triplets))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)
    print(f"  Train: {len(train_triplets[tr_idx]):,}, Val: {len(train_triplets[va_idx]):,}, Test: {len(test_triplets):,}", flush=True)

    bs = cfg.model.batch_size
    train_loader = make_text_dataloader(train_triplets[tr_idx], text_embs, q_matrix, bs, shuffle=True)
    val_loader = make_text_dataloader(train_triplets[va_idx], text_embs, q_matrix, bs, shuffle=False)
    test_loader = make_text_dataloader(test_triplets, text_embs, q_matrix, bs, shuffle=False)

    # ── Train ──
    epochs = cfg.model.epochs
    lr = cfg.model.lr
    batches = len(train_triplets[tr_idx]) // bs
    print(f"\nTraining TextConditionedNet (K={K}, {n_llms} LLMs)", flush=True)
    print(f"  {batches:,} batches/epoch, {epochs} epochs", flush=True)

    net = TextConditionedNet(n_skills, n_llms, text_dim)
    net = train_text_model(net, train_loader, val_loader,
                           epochs=epochs, lr=lr, device=device)

    # ── Evaluate AUC ──
    print("\nEvaluating...", flush=True)
    auc, acc, rmse = eval_text_model(net, test_loader, device)
    print(f"  Test AUC={auc:.4f}, Acc={acc:.4f}, RMSE={rmse:.4f}", flush=True)

    # ── Routing ──
    print("Routing evaluation...", flush=True)
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    accs_at = {1: 0, 3: 0, 5: 0, 10: 0}
    n_total = len(test_items)

    for item_idx in test_items:
        gt = R[:, int(item_idx)]
        if gt.sum() == 0:
            continue
        emb = torch.tensor(text_embs[int(item_idx)], dtype=torch.float32, device=device)
        qr = torch.tensor(q_matrix[int(item_idx)], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = net(all_llm_ids, emb.unsqueeze(0).expand(n_llms, -1),
                        qr.unsqueeze(0).expand(n_llms, -1)).cpu().numpy()
        ranking = np.argsort(-preds)
        for k in accs_at:
            if gt[ranking[:k]].sum() > 0:
                accs_at[k] += 1

    routing = {f"acc@{k}": accs_at[k] / n_total for k in accs_at}
    print(f"  Routing: " + ", ".join(f"@{k}={routing[f'acc@{k}']:.4f}" for k in accs_at), flush=True)

    # Strongest baseline
    train_acc = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    strongest_acc1 = sum(R[strongest_idx, int(i)] for i in test_items) / n_total
    print(f"  Strongest @1: {strongest_acc1:.4f}", flush=True)

    # ── Save checkpoint ──
    ckpt_dir = Path("cdm_exploration/checkpoints/expanded")
    save_checkpoint(
        net, ckpt_dir / f"text_conditioned_K{K}.pt",
        config={"K": K, "n_llms": n_llms, "n_items": n_items,
                "epochs": epochs, "lr": lr, "text_dim": text_dim},
        train_items=train_items, test_items=test_items,
        val_auc=auc, epoch=epochs,
    )

    # ── Save results ──
    results = {
        "K": K, "test_auc": float(auc), "test_acc": float(acc),
        "test_rmse": float(rmse), **routing,
        "strongest_acc1": float(strongest_acc1),
        "n_llms": n_llms, "n_items": n_items, "epochs": epochs,
    }
    out = Path(f"cdm_exploration/experiments/v2_K{K}_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}", flush=True)

    # ── Verify and log ──
    verified = verify_splits(train_items, test_items, label=f"k_ablation_K{K}")
    log_experiment(
        name=f"k_ablation_K{K}",
        config={"K": K, "epochs": epochs, "lr": lr, "device": device},
        results=results,
        split_info={"n_train": len(train_items), "n_test": len(test_items), "n_llms": n_llms},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
