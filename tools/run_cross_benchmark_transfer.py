"""Cross-benchmark transfer: train on MATH+BBH+GPQA, test on MuSR+IFEval.

The model NEVER sees MuSR or IFEval items during training.
Tests whether skill profiles learned from math/reasoning/science
transfer to multi-step reasoning and instruction following.

Usage:
    python tools/run_cross_benchmark_transfer.py device=cuda
    python tools/run_cross_benchmark_transfer.py device=cpu model.epochs=3
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
    from cdmeval.utils.experiment import log_experiment, save_checkpoint
    from cdmeval.validation import (
        validate_checkpoint, validate_data, validate_metrics, validate_split,
    )

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)

    # ── Load data ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    text_dim = text_embs.shape[1]

    # Validate data
    validate_data(R, q_matrix, text_embs, items_data, llm_names)

    # ── Benchmark-based split ──
    print("\nSplitting by benchmark...", flush=True)
    train_benchmarks = {"MATH", "BBH", "GPQA"}
    test_benchmarks = {"MuSR", "IFEval"}

    train_items = np.array([i for i, it in enumerate(items_data)
                            if it.get("benchmark") in train_benchmarks])
    test_items = np.array([i for i, it in enumerate(items_data)
                           if it.get("benchmark") in test_benchmarks])

    # Per-benchmark test item indices
    musr_items = np.array([i for i, it in enumerate(items_data)
                           if it.get("benchmark") == "MuSR"])
    ifeval_items = np.array([i for i, it in enumerate(items_data)
                             if it.get("benchmark") == "IFEval"])

    print(f"  Train benchmarks: {train_benchmarks} = {len(train_items)} items", flush=True)
    print(f"  Test benchmarks: MuSR={len(musr_items)}, IFEval={len(ifeval_items)}, "
          f"total={len(test_items)}", flush=True)

    # Verify no overlap
    overlap = np.intersect1d(train_items, test_items)
    assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} items!"
    print(f"  No overlap: PASS", flush=True)

    # ── Build triplets ──
    print("\nBuilding triplets...", flush=True)
    student_ids = np.repeat(np.arange(n_llms), n_items)
    item_ids = np.tile(np.arange(n_items), n_llms)
    scores = R.ravel().astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])

    train_mask = np.isin(triplets[:, 1].astype(int), train_items)
    train_triplets = triplets[train_mask]
    test_triplets = triplets[~train_mask]

    tv_idx = np.arange(len(train_triplets))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)
    print(f"  Train: {len(train_triplets[tr_idx]):,}, Val: {len(train_triplets[va_idx]):,}, "
          f"Test: {len(test_triplets):,}", flush=True)

    bs = cfg.model.batch_size
    train_loader = make_text_dataloader(train_triplets[tr_idx], text_embs, q_matrix, bs, shuffle=True)
    val_loader = make_text_dataloader(train_triplets[va_idx], text_embs, q_matrix, bs, shuffle=False)
    test_loader = make_text_dataloader(test_triplets, text_embs, q_matrix, bs, shuffle=False)

    # ── Train ──
    epochs = cfg.model.epochs
    lr = cfg.model.lr
    batches = len(train_triplets[tr_idx]) // bs
    print(f"\nTraining (K={K}, {n_llms} LLMs, train on MATH+BBH+GPQA only)", flush=True)
    print(f"  {batches:,} batches/epoch, {epochs} epochs", flush=True)

    net = TextConditionedNet(K, n_llms, text_dim)
    net = train_text_model(net, train_loader, val_loader,
                           epochs=epochs, lr=lr, device=device)

    # ── Evaluate combined AUC ──
    print("\nEvaluating on held-out benchmarks (MuSR + IFEval)...", flush=True)
    combined_auc, combined_acc, combined_rmse = eval_text_model(net, test_loader, device)
    print(f"  Combined AUC={combined_auc:.4f}, Acc={combined_acc:.4f}", flush=True)

    # ── Per-benchmark routing ──
    print("\nRouting evaluation...", flush=True)
    net.eval()
    net = net.to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    # Strongest model (by train benchmark accuracy)
    train_acc = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    rng = np.random.RandomState(42)

    def compute_routing(items, label):
        n = len(items)
        cdm_correct = 0
        strong_correct = 0
        rand_correct = 0

        for item_idx in items:
            ii = int(item_idx)
            gt = R[:, ii]
            if gt.sum() == 0:
                continue

            emb = torch.tensor(text_embs[ii], dtype=torch.float32, device=device)
            qr = torch.tensor(q_matrix[ii], dtype=torch.float32, device=device)
            with torch.no_grad():
                preds = net(all_llm_ids, emb.unsqueeze(0).expand(n_llms, -1),
                            qr.unsqueeze(0).expand(n_llms, -1)).cpu().numpy()
            if gt[np.argmax(preds)] > 0:
                cdm_correct += 1
            if gt[strongest_idx] > 0:
                strong_correct += 1
            if gt[rng.randint(n_llms)] > 0:
                rand_correct += 1

        result = {
            "n_items": n,
            "cdm_acc1": cdm_correct / n,
            "strongest_acc1": strong_correct / n,
            "random_acc1": rand_correct / n,
        }
        print(f"  {label}: CDM @1={result['cdm_acc1']:.4f}, "
              f"Strongest={result['strongest_acc1']:.4f}, "
              f"Random={result['random_acc1']:.4f} (n={n})", flush=True)
        return result

    musr_results = compute_routing(musr_items, "MuSR")
    ifeval_results = compute_routing(ifeval_items, "IFEval")
    combined_routing = compute_routing(test_items, "Combined")

    # ── Comparison table ──
    # Full-training results from per_benchmark_routing
    print(f"\n{'='*80}", flush=True)
    print(f"CROSS-BENCHMARK TRANSFER RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Setting':<40} {'MuSR @1':>9} {'IFEval @1':>10} {'Comb @1':>9} {'AUC':>8}", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Full training (saw all benchmarks)':<40} {'0.5441':>9} {'0.8200':>10} {'—':>9} {'0.7170':>8}", flush=True)
    print(f"{'Transfer (never saw MuSR/IFEval)':<40} "
          f"{musr_results['cdm_acc1']:>9.4f} {ifeval_results['cdm_acc1']:>10.4f} "
          f"{combined_routing['cdm_acc1']:>9.4f} {combined_auc:>8.4f}", flush=True)
    print(f"{'Strongest model':<40} "
          f"{musr_results['strongest_acc1']:>9.4f} {ifeval_results['strongest_acc1']:>10.4f} "
          f"{combined_routing['strongest_acc1']:>9.4f} {'—':>8}", flush=True)
    print(f"{'Random':<40} "
          f"{musr_results['random_acc1']:>9.4f} {ifeval_results['random_acc1']:>10.4f} "
          f"{combined_routing['random_acc1']:>9.4f} {'—':>8}", flush=True)

    # Transfer gap
    musr_gap = musr_results["cdm_acc1"] - 0.5441
    ifeval_gap = ifeval_results["cdm_acc1"] - 0.8200
    print(f"\n  Transfer gap vs full training:", flush=True)
    print(f"    MuSR:  {musr_gap:+.4f}", flush=True)
    print(f"    IFEval: {ifeval_gap:+.4f}", flush=True)

    # ── Save checkpoint ──
    ckpt_dir = Path("cdm_exploration/checkpoints/expanded")
    save_checkpoint(
        net, ckpt_dir / "text_conditioned_cross_benchmark.pt",
        config={"K": K, "n_llms": n_llms, "n_items": n_items,
                "epochs": epochs, "lr": lr, "text_dim": text_dim,
                "train_benchmarks": list(train_benchmarks),
                "test_benchmarks": list(test_benchmarks)},
        train_items=train_items, test_items=test_items,
        val_auc=combined_auc, epoch=epochs,
    )
    validate_checkpoint(ckpt_dir / "text_conditioned_cross_benchmark.pt",
                        expected_K=K, expected_n_llms=n_llms)

    # ── Save results ──
    results = {
        "experiment": "cross_benchmark_transfer",
        "train_benchmarks": list(train_benchmarks),
        "test_benchmarks": list(test_benchmarks),
        "n_train_items": len(train_items),
        "n_test_items": len(test_items),
        "K": K, "epochs": epochs, "n_llms": n_llms,
        "combined_auc": float(combined_auc),
        "combined_acc": float(combined_acc),
        "combined_routing": combined_routing,
        "musr": musr_results,
        "ifeval": ifeval_results,
        "full_training_reference": {
            "musr_acc1": 0.5441, "ifeval_acc1": 0.8200, "auc": 0.7170,
        },
    }
    validate_metrics(results)

    out = Path("cdm_exploration/experiments/v2_cross_benchmark_transfer.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}", flush=True)

    # ── Log ──
    log_experiment(
        name="cross_benchmark_transfer",
        config={"K": K, "epochs": epochs, "lr": lr, "device": device,
                "train_benchmarks": list(train_benchmarks),
                "test_benchmarks": list(test_benchmarks)},
        results=results,
        split_info={"n_train_items": len(train_items), "n_test_items": len(test_items),
                    "n_llms": n_llms},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
