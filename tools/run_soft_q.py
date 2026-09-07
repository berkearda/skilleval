"""Train text-conditioned NCDM with a SOFT Q-matrix (continuous-valued
per-item-skill weights).

Mirrors tools/run_multi_seed.py exactly for splits, RNG, model, and
metric protocol -- the only differences are:
  (a) load qmatrix from cfg.soft_q.qmatrix_file (a soft Q .npy),
  (b) skip the fix_zero_skill_items binary-only fix (soft rows already
      sum to 1 by construction; nothing to fix),
  (c) write per-bench Acc@1 and routing metrics to a tau-tagged JSON,
  (d) save a tau-tagged checkpoint.

Usage examples:
    # full smoke (5 epochs):
    python tools/run_soft_q.py device=cuda +seed=42 \
        +soft_q.tau=0.3 +soft_q.qmatrix_file=qmatrix_v2_K100_soft_tau0.3.npy \
        +soft_q.epochs=5 +soft_q.tag=smoke

    # full sweep entry (uses cfg.model.epochs):
    python tools/run_soft_q.py device=cuda +seed=42 \
        +soft_q.tau=0.3 +soft_q.qmatrix_file=qmatrix_v2_K100_soft_tau0.3.npy
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


class TripletIndexedDataset(Dataset):
    def __init__(self, triplets, text_t, q_t):
        self.s = triplets[:, 0].astype(np.int32, copy=False)
        self.i = triplets[:, 1].astype(np.int32, copy=False)
        self.r = triplets[:, 2].astype(np.float32, copy=False)
        self.text = text_t
        self.q = q_t

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        item = int(self.i[idx])
        return (
            torch.tensor(int(self.s[idx]), dtype=torch.int64),
            self.text[item],
            self.q[item],
            torch.tensor(float(self.r[idx]), dtype=torch.float32),
        )


def _require(cfg, dotted: str):
    obj = cfg
    for part in dotted.split("."):
        if not hasattr(obj, part):
            raise SystemExit(
                f"ERROR: missing required override +{dotted} (e.g. "
                f"+soft_q.qmatrix_file=qmatrix_v2_K100_soft_tau0.3.npy)"
            )
        obj = getattr(obj, part)
    return obj


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint
    from cdmeval.validation import validate_metrics

    qmatrix_file = str(_require(cfg, "soft_q.qmatrix_file"))
    tau = float(_require(cfg, "soft_q.tau"))
    tag = str(getattr(cfg.soft_q, "tag", "")) if hasattr(cfg, "soft_q") else ""
    seed = int(getattr(cfg, "seed", 42))
    epochs_override = getattr(cfg.soft_q, "epochs", None) if hasattr(cfg, "soft_q") else None

    seed_everything(seed)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print(f"Soft Q file: {qmatrix_file}  tau={tau}  tag='{tag}'", flush=True)

    # ── Load data ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_path = data_dir / qmatrix_file
    if not q_path.exists():
        raise SystemExit(f"FAIL: soft Q file not found: {q_path}")
    q_matrix = np.load(q_path)
    if q_matrix.dtype != np.float32:
        q_matrix = q_matrix.astype(np.float32, copy=False)
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)

    n_llms, n_items = R.shape

    # Optional LLM subset (smoke-test only). Matches train_expanded.py
    # convention: rng is independent of cfg.seed so subset is reproducible.
    n_llms_sub = None
    if hasattr(cfg, "subset") and hasattr(cfg.subset, "n_llms"):
        n_llms_sub = int(cfg.subset.n_llms)
        rng_sub = np.random.RandomState(42)
        llm_indices = rng_sub.choice(n_llms, size=n_llms_sub, replace=False)
        R = R[llm_indices]
        llm_names = [llm_names[i] for i in llm_indices]
        n_llms = n_llms_sub
        print(f"  LLM subset: using {n_llms_sub} of {len(llm_indices)} sampled LLMs",
              flush=True)

    K = q_matrix.shape[1]
    text_dim = text_embs.shape[1]
    print(
        f"  R: {R.shape}  Q: {q_matrix.shape} (dtype={q_matrix.dtype} "
        f"sum_row_mean={float(q_matrix.sum(axis=1).mean()):.4f})  "
        f"text_dim={text_dim}",
        flush=True,
    )
    # Inline data checks (skip validate_data because it enforces a binary Q,
    # which is exactly what we relax here). All other checks preserved.
    if q_matrix.shape[0] != n_items:
        raise SystemExit(f"FAIL: Q rows {q_matrix.shape[0]} != n_items {n_items}")
    if text_embs.shape[0] != n_items:
        raise SystemExit(f"FAIL: text rows {text_embs.shape[0]} != n_items {n_items}")
    if len(items_data) != n_items:
        raise SystemExit(f"FAIL: items_data {len(items_data)} != n_items {n_items}")
    if len(llm_names) != n_llms:
        raise SystemExit(f"FAIL: llm_names {len(llm_names)} != n_llms {n_llms}")
    r_unique = np.unique(R)
    if not np.all(np.isin(r_unique, [0, 1])):
        raise SystemExit(f"FAIL: response matrix not binary: {r_unique[:10]}")
    if np.isnan(R.astype(float)).any():
        raise SystemExit("FAIL: NaN in response matrix")
    if np.isnan(text_embs).any():
        raise SystemExit("FAIL: NaN in text embeddings")
    print(
        f"  [VALIDATE] Data OK (soft Q): {n_llms} LLMs x {n_items} items, "
        f"K={K}, emb_dim={text_dim}",
        flush=True,
    )

    if np.isnan(q_matrix).any():
        raise SystemExit("FAIL (F14): NaN in soft Q-matrix")
    sums = q_matrix.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise SystemExit(
            f"FAIL: soft Q rows do not sum to 1 "
            f"(max dev {float(np.max(np.abs(sums - 1.0))):.3e})"
        )

    R_f32 = R.astype(np.float32, copy=False)
    text_t = torch.from_numpy(text_embs.astype(np.float32, copy=False))
    q_t = torch.from_numpy(q_matrix)

    # ── Splits (verbatim from run_multi_seed.py) ──
    print("\nBuilding triplets...", flush=True)
    student_ids = np.repeat(np.arange(n_llms), n_items)
    item_ids = np.tile(np.arange(n_items), n_llms)
    scores = R.ravel().astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])

    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(
        f"  Train items: {len(train_items)}, Test items: {len(test_items)}",
        flush=True,
    )

    train_mask = np.isin(triplets[:, 1].astype(int), train_items)
    train_triplets = triplets[train_mask]
    test_triplets = triplets[~train_mask]

    tv_idx = np.arange(len(train_triplets))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)
    print(
        f"  Train triplets: {len(train_triplets[tr_idx]):,}, "
        f"Val: {len(train_triplets[va_idx]):,}, "
        f"Test: {len(test_triplets):,}",
        flush=True,
    )

    bs = cfg.model.batch_size
    train_ds = TripletIndexedDataset(train_triplets[tr_idx], text_t, q_t)
    val_ds = TripletIndexedDataset(train_triplets[va_idx], text_t, q_t)
    test_ds = TripletIndexedDataset(test_triplets, text_t, q_t)

    del student_ids, item_ids, scores, triplets, train_triplets, test_triplets
    del train_mask, tv_idx, tr_idx, va_idx

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    epochs = int(epochs_override) if epochs_override is not None else int(cfg.model.epochs)
    lr = float(cfg.model.lr)
    print(
        f"\nTraining (K={K}, {n_llms} LLMs, seed={seed}, epochs={epochs}, lr={lr})",
        flush=True,
    )

    net = TextConditionedNet(K, n_llms, text_dim)
    net = train_text_model(net, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    print("\nEvaluating on test items...", flush=True)
    test_auc, test_acc, test_rmse = eval_text_model(net, test_loader, device)
    print(
        f"  Test AUC={test_auc:.4f}, Acc={test_acc:.4f}, RMSE={test_rmse:.4f}",
        flush=True,
    )

    # ── Routing Acc@1 (and per-benchmark) on test items ──
    print("\nRouting Acc@1 on test items...", flush=True)
    net.eval().to(device)
    all_llm_ids = torch.arange(n_llms, device=device)
    rng = np.random.RandomState(42)
    cdm_correct = strong_correct = rand_correct = 0
    train_acc_per_llm = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc_per_llm))
    used = 0

    bench_correct = defaultdict(int)
    bench_used = defaultdict(int)

    for ii in test_items:
        i = int(ii)
        gt = R[:, i]
        if gt.sum() == 0:
            continue
        used += 1
        emb = torch.tensor(text_embs[i], dtype=torch.float32, device=device)
        qr = torch.tensor(q_matrix[i], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = net(
                all_llm_ids,
                emb.unsqueeze(0).expand(n_llms, -1),
                qr.unsqueeze(0).expand(n_llms, -1),
            ).cpu().numpy()
        chosen = int(np.argmax(preds))
        bench = items_data[i].get("benchmark", "?") if i < len(items_data) else "?"
        bench_used[bench] += 1
        if gt[chosen] > 0:
            cdm_correct += 1
            bench_correct[bench] += 1
        if gt[strongest_idx] > 0:
            strong_correct += 1
        if gt[int(rng.randint(n_llms))] > 0:
            rand_correct += 1

    routing_acc1 = cdm_correct / max(used, 1)
    strongest_acc1 = strong_correct / max(used, 1)
    random_acc1 = rand_correct / max(used, 1)
    per_bench_acc1 = {
        b: {
            "n_items": int(bench_used[b]),
            "acc1": float(bench_correct[b] / bench_used[b]) if bench_used[b] else None,
        }
        for b in sorted(bench_used)
    }
    print(
        f"  CDM @1={routing_acc1:.4f}, Strongest={strongest_acc1:.4f}, "
        f"Random={random_acc1:.4f}",
        flush=True,
    )
    for b, d in per_bench_acc1.items():
        print(f"    {b}: n={d['n_items']:4d}  acc1={d['acc1']:.4f}",
              flush=True)

    # ── Save artefacts ──
    out_dir = Path("cdm_exploration/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"tau{tau}_s{seed}"
    if tag:
        suffix = f"{tag}_{suffix}"
    out_json = out_dir / f"v2_soft_q_{suffix}.json"

    fold_result = {
        "experiment": "soft_q_ncdm",
        "soft_q_file": qmatrix_file,
        "tau": tau,
        "tag": tag,
        "seed": seed,
        "K": int(K),
        "n_llms": int(n_llms),
        "n_items": int(n_items),
        "epochs": int(epochs),
        "lr": float(lr),
        "n_train_items": int(len(train_items)),
        "n_test_items": int(len(test_items)),
        "val_protocol": "triplet_level_90_10",
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "test_rmse": float(test_rmse),
        "routing_acc1": float(routing_acc1),
        "strongest_acc1": float(strongest_acc1),
        "random_acc1": float(random_acc1),
        "per_benchmark_acc1": per_bench_acc1,
    }
    validate_metrics(fold_result)
    with open(out_json, "w") as f:
        json.dump(fold_result, f, indent=2)
    print(f"\nSaved per-run file: {out_json}", flush=True)

    ckpt_dir = Path("cdm_exploration/checkpoints/expanded")
    ckpt_name = f"text_conditioned_K{K}_soft_{suffix}.pt"
    save_checkpoint(
        net,
        ckpt_dir / ckpt_name,
        config={
            "K": K, "n_llms": n_llms, "n_items": n_items,
            "epochs": epochs, "lr": lr, "text_dim": text_dim,
            "seed": seed, "soft_q_file": qmatrix_file, "tau": tau, "tag": tag,
        },
        train_items=train_items,
        test_items=test_items,
        val_auc=test_auc,
        epoch=epochs,
    )
    log_experiment(
        name=f"soft_q_{suffix}",
        config={
            "K": K, "epochs": epochs, "lr": lr, "device": device,
            "seed": seed, "soft_q_file": qmatrix_file, "tau": tau, "tag": tag,
        },
        results=fold_result,
        split_info={
            "n_train_items": int(len(train_items)),
            "n_test_items": int(len(test_items)),
            "n_llms": int(n_llms),
        },
        verified=True,
    )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
