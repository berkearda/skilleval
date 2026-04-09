"""Multi-seed stability analysis of CDMEval theta vectors.

For each seed in {42,43,44,45,46}:
  1. Train a fresh K=100 text-conditioned NCDM on the standard 80/20 item split.
  2. Save the learned theta matrix (sigmoid(student_emb.weight)) and routing
     Acc@1 to a per-seed file.

After all 5 per-seed files exist, the last finishing task computes:
  * Pairwise Pearson correlation between flattened theta matrices.
  * Mean per-LLM Pearson correlation across seed pairs.
  * Per-skill correlation across seeds (mean over LLMs and seed pairs).
  * Routing Acc@1 mean / std across seeds.
and writes experiments/v2_multi_seed.json.

Designed for SLURM array job (--array=0-4) where SLURM_ARRAY_TASK_ID picks
the seed. Can also run a single seed locally via:
    python tools/run_multi_seed.py device=cuda +seed=42
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


SEEDS = [42, 43, 44, 45, 46]


class TripletIndexedDataset(Dataset):
    """Lazy dataset over an explicit (student, item, score) triplet array.

    Avoids the per-triplet text/q materialisation that caused the OOM in
    make_text_dataloader: stores only the (s, i, r) columns of the triplet
    array, plus shared text and Q tensors that are indexed at __getitem__
    time.
    """

    def __init__(self, triplets, text_t, q_t):
        # triplets: (N, 3) array with [student_id, item_id, score]
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


class ItemEvalDataset(Dataset):
    """Lazy (student, text, q, score) over a held-out item set, for evaluation.

    Used only for the test loader where we want every (student, item) pair on
    the test items but no triplet array — same shape as TripletIndexedDataset
    output.
    """

    def __init__(self, R, item_ids, text_t, q_t):
        self.R = R
        self.item_ids = np.ascontiguousarray(item_ids, dtype=np.int64)
        self.text = text_t
        self.q = q_t
        self.n_llms = R.shape[0]
        self.n_local = int(len(self.item_ids))

    def __len__(self):
        return self.n_llms * self.n_local

    def __getitem__(self, idx):
        s, li = divmod(idx, self.n_local)
        i = int(self.item_ids[li])
        return (
            torch.tensor(s, dtype=torch.int64),
            self.text[i],
            self.q[i],
            torch.tensor(float(self.R[s, i]), dtype=torch.float32),
        )


def select_seed(cfg) -> int:
    if hasattr(cfg, "seed"):
        s = int(cfg.seed)
        if s not in SEEDS:
            raise ValueError(f"seed={s} not in {SEEDS}")
        return s
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id)
        if not (0 <= idx < len(SEEDS)):
            raise ValueError(f"SLURM_ARRAY_TASK_ID={idx} out of range 0..4")
        return SEEDS[idx]
    raise SystemExit(
        "ERROR: no seed specified. Pass +seed=42 or set SLURM_ARRAY_TASK_ID in [0,4]."
    )


def maybe_aggregate(out_dir: Path):
    """If all 5 per-seed files are present, compute the aggregate metrics
    and write experiments/v2_multi_seed.json. Idempotent."""
    per_seed_paths = [out_dir / f"v2_multi_seed_{s}.json" for s in SEEDS]
    if not all(p.exists() for p in per_seed_paths):
        missing = [p.name for p in per_seed_paths if not p.exists()]
        print(f"\nAggregation skipped: still waiting on {missing}", flush=True)
        return

    print("\nAll 5 per-seed files present — computing aggregate metrics.", flush=True)
    per_seed = [json.load(open(p)) for p in per_seed_paths]

    # Stack thetas: (5, n_llms, K)
    thetas = np.stack(
        [np.load(out_dir / f"v2_multi_seed_{s}_theta.npy") for s in SEEDS],
        axis=0,
    )
    n_seeds, n_llms, K = thetas.shape
    print(f"  thetas shape: {thetas.shape}", flush=True)

    # ── Flattened pairwise Pearson r ──
    pair_r_flat = {}
    for i, j in combinations(range(n_seeds), 2):
        r, _ = pearsonr(thetas[i].ravel(), thetas[j].ravel())
        pair_r_flat[f"{SEEDS[i]}-{SEEDS[j]}"] = float(r)
    flat_mean = float(np.mean(list(pair_r_flat.values())))
    flat_std = float(np.std(list(pair_r_flat.values())))

    # ── Per-LLM correlation, averaged over LLMs and pairs ──
    per_llm_pair_means = []
    for i, j in combinations(range(n_seeds), 2):
        rs = np.array([
            pearsonr(thetas[i, l], thetas[j, l])[0] for l in range(n_llms)
        ])
        rs = rs[~np.isnan(rs)]
        per_llm_pair_means.append(float(np.mean(rs)))
    per_llm_mean = float(np.mean(per_llm_pair_means))
    per_llm_std = float(np.std(per_llm_pair_means))

    # ── Per-skill correlation across seeds: for each skill k, treat the
    # length-n_llms vector across LLMs as the signal, correlate across seed
    # pairs, then average over pairs and report per skill.
    per_skill_r = np.zeros(K)
    for k in range(K):
        rs = []
        for i, j in combinations(range(n_seeds), 2):
            r, _ = pearsonr(thetas[i, :, k], thetas[j, :, k])
            if not np.isnan(r):
                rs.append(r)
        per_skill_r[k] = np.mean(rs) if rs else float("nan")
    per_skill_mean = float(np.nanmean(per_skill_r))
    per_skill_std = float(np.nanstd(per_skill_r))
    per_skill_min = float(np.nanmin(per_skill_r))
    per_skill_max = float(np.nanmax(per_skill_r))

    # ── Routing Acc@1 stats ──
    acc1s = np.array([d["routing_acc1"] for d in per_seed], dtype=float)
    aucs = np.array([d["test_auc"] for d in per_seed], dtype=float)

    summary = {
        "experiment": "multi_seed_stability",
        "seeds": SEEDS,
        "K": int(K),
        "n_llms": int(n_llms),
        "theta_pairwise_pearson_flat": pair_r_flat,
        "theta_pairwise_pearson_flat_mean": flat_mean,
        "theta_pairwise_pearson_flat_std": flat_std,
        "theta_per_llm_pearson_mean": per_llm_mean,
        "theta_per_llm_pearson_std": per_llm_std,
        "theta_per_skill_pearson_mean": per_skill_mean,
        "theta_per_skill_pearson_std": per_skill_std,
        "theta_per_skill_pearson_min": per_skill_min,
        "theta_per_skill_pearson_max": per_skill_max,
        "theta_per_skill_pearson": per_skill_r.tolist(),
        "routing_acc1_per_seed": acc1s.tolist(),
        "routing_acc1_mean": float(np.mean(acc1s)),
        "routing_acc1_std": float(np.std(acc1s)),
        "test_auc_per_seed": aucs.tolist(),
        "test_auc_mean": float(np.mean(aucs)),
        "test_auc_std": float(np.std(aucs)),
        "per_seed": {str(s): d for s, d in zip(SEEDS, per_seed)},
    }
    out_path = out_dir / "v2_multi_seed.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregate written: {out_path}", flush=True)
    print(f"  flat pairwise r mean = {flat_mean:.4f} ± {flat_std:.4f}", flush=True)
    print(f"  per-LLM pairwise r mean = {per_llm_mean:.4f} ± {per_llm_std:.4f}", flush=True)
    print(f"  per-skill r mean = {per_skill_mean:.4f} (min {per_skill_min:.4f}, "
          f"max {per_skill_max:.4f})", flush=True)
    print(f"  routing Acc@1 = {np.mean(acc1s):.4f} ± {np.std(acc1s):.4f}", flush=True)
    print(f"  test AUC      = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint
    from cdmeval.validation import validate_checkpoint, validate_data, validate_metrics

    seed = select_seed(cfg)
    seed_everything(seed)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {seed}", flush=True)

    # ── Load data ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy").copy()
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    text_dim = text_embs.shape[1]
    validate_data(R, q_matrix, text_embs, items_data, llm_names)

    # ── Fix zero-skill items (copied from train_expanded.py) ──
    # Items with all-zero Q-rows produce zero PosLinear input → constant
    # predictions → routing degrades to random on those items.
    text_emb_path = data_dir / "item_text_embeddings_v2_full.npz"
    skill_emb_path = data_dir / "skill_embeddings_v2_K100.npz"
    n_zero = int((q_matrix.sum(axis=1) == 0).sum())
    if n_zero > 0:
        from tools.train_expanded import fix_zero_skill_items
        q_matrix = fix_zero_skill_items(
            q_matrix, skill_emb_path, items_data, text_emb_path,
        )
        print(f"  Fixed {n_zero} zero-skill items in Q-matrix", flush=True)
    else:
        print(f"  No zero-skill items to fix", flush=True)

    R_f32 = R.astype(np.float32, copy=False)
    text_t = torch.from_numpy(text_embs.astype(np.float32, copy=False))
    q_t = torch.from_numpy(q_matrix.astype(np.float32, copy=False))

    # ─────────────────────────────────────────────────────────────────────
    # SPLIT PROTOCOL: copied verbatim from tools/train_expanded.py so that
    # this script's main-run results match the published numbers when
    # seed=42. The val split is at the TRIPLET level (90/10 of train_triplets)
    # NOT at the item level — every train item is seen by the model during
    # training. The 80/20 outer item split AND the 90/10 inner triplet split
    # both use random_state=42, independent of the per-run seed; only model
    # init / shuffle / dropout are reseeded by `seed`.
    # ─────────────────────────────────────────────────────────────────────

    # ── Build triplets ──
    print("\nBuilding triplets...", flush=True)
    student_ids = np.repeat(np.arange(n_llms), n_items)
    item_ids = np.tile(np.arange(n_items), n_llms)
    scores = R.ravel().astype(float)
    triplets = np.column_stack([student_ids, item_ids, scores])
    print(f"  Total triplets: {len(triplets):,}", flush=True)

    # ── Protocol B split ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"  Train items: {len(train_items)}, Test items: {len(test_items)}",
          flush=True)

    train_mask = np.isin(triplets[:, 1].astype(int), train_items)
    train_triplets = triplets[train_mask]
    test_triplets = triplets[~train_mask]

    tv_idx = np.arange(len(train_triplets))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)
    print(f"  Train triplets: {len(train_triplets[tr_idx]):,}, "
          f"Val: {len(train_triplets[va_idx]):,}, "
          f"Test: {len(test_triplets):,}", flush=True)

    bs = cfg.model.batch_size

    train_ds = TripletIndexedDataset(train_triplets[tr_idx], text_t, q_t)
    val_ds = TripletIndexedDataset(train_triplets[va_idx], text_t, q_t)
    test_ds = TripletIndexedDataset(test_triplets, text_t, q_t)

    # Free the large intermediate arrays now that the datasets own slim
    # int32/float32 columns.
    del student_ids, item_ids, scores, triplets, train_triplets, test_triplets
    del train_mask, tv_idx, tr_idx, va_idx

    # IMPORTANT: num_workers=0 to match make_text_dataloader's default in
    # train_expanded.py. With num_workers>0, worker RNG states diverge from
    # the single-threaded path, producing different batch orderings and a
    # model with similar AUC but different per-item rankings (Acc@1 drops
    # from 0.657 to 0.617).
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # ── Train ──
    epochs = cfg.model.epochs
    lr = cfg.model.lr
    print(f"\nTraining (K={K}, {n_llms} LLMs, seed={seed})", flush=True)
    print(f"  {len(train_ds)//bs:,} batches/epoch, {epochs} epochs", flush=True)

    net = TextConditionedNet(K, n_llms, text_dim)
    net = train_text_model(net, train_loader, val_loader,
                           epochs=epochs, lr=lr, device=device)

    # ── Evaluate ──
    print("\nEvaluating on test items...", flush=True)
    test_auc, test_acc, test_rmse = eval_text_model(net, test_loader, device)
    print(f"  Test AUC={test_auc:.4f}, Acc={test_acc:.4f}, RMSE={test_rmse:.4f}",
          flush=True)

    # ── Routing Acc@1 on test items ──
    print("\nRouting Acc@1 on test items...", flush=True)
    net.eval().to(device)
    all_llm_ids = torch.arange(n_llms, device=device)
    rng = np.random.RandomState(42)
    cdm_correct = strong_correct = rand_correct = 0
    train_acc_per_llm = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc_per_llm))
    used = 0
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
        if gt[int(np.argmax(preds))] > 0:
            cdm_correct += 1
        if gt[strongest_idx] > 0:
            strong_correct += 1
        if gt[int(rng.randint(n_llms))] > 0:
            rand_correct += 1
    routing_acc1 = cdm_correct / max(used, 1)
    strongest_acc1 = strong_correct / max(used, 1)
    random_acc1 = rand_correct / max(used, 1)
    print(f"  CDM @1={routing_acc1:.4f}, Strongest={strongest_acc1:.4f}, "
          f"Random={random_acc1:.4f}", flush=True)

    # ── Sanity check: seed=42 must reproduce main-paper numbers ──
    if seed == 42:
        MAIN_AUC = 0.717
        MAIN_ACC1 = 0.657
        TOL = 0.005
        d_auc = abs(test_auc - MAIN_AUC)
        d_acc = abs(routing_acc1 - MAIN_ACC1)
        if d_auc > TOL or d_acc > TOL:
            print("\n  WARNING: seed=42 does not match main-paper numbers", flush=True)
            print(f"           AUC    : got {test_auc:.4f}, expected {MAIN_AUC:.4f} "
                  f"(|delta|={d_auc:.4f}, tol={TOL})", flush=True)
            print(f"           Acc@1  : got {routing_acc1:.4f}, expected {MAIN_ACC1:.4f} "
                  f"(|delta|={d_acc:.4f}, tol={TOL})", flush=True)
            print("           Investigate the val protocol or RNG state before "
                  "trusting the multi-seed aggregate.", flush=True)
        else:
            print(f"\n  seed=42 sanity check PASS: "
                  f"AUC {test_auc:.4f} (target {MAIN_AUC:.4f}, |d|={d_auc:.4f}), "
                  f"Acc@1 {routing_acc1:.4f} (target {MAIN_ACC1:.4f}, |d|={d_acc:.4f})",
                  flush=True)

    # ── Extract theta = sigmoid(student_emb.weight) ──
    with torch.no_grad():
        theta = torch.sigmoid(net.student_emb.weight.detach()).cpu().numpy()
    print(f"  theta shape: {theta.shape}, range "
          f"[{theta.min():.3f}, {theta.max():.3f}]", flush=True)

    # ── Save per-seed artefacts ──
    out_dir = Path("cdm_exploration/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"v2_multi_seed_{seed}_theta.npy", theta)

    fold_result = {
        "seed": seed,
        "K": int(K), "n_llms": int(n_llms), "n_items": int(n_items),
        "epochs": int(epochs), "lr": float(lr),
        "n_train_items": int(len(train_items)),
        "n_test_items": int(len(test_items)),
        "val_protocol": "triplet_level_90_10",
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "test_rmse": float(test_rmse),
        "routing_acc1": float(routing_acc1),
        "strongest_acc1": float(strongest_acc1),
        "random_acc1": float(random_acc1),
    }
    validate_metrics(fold_result)
    with open(out_dir / f"v2_multi_seed_{seed}.json", "w") as f:
        json.dump(fold_result, f, indent=2)
    print(f"\nSaved per-seed file: v2_multi_seed_{seed}.json", flush=True)

    # ── Checkpoint + log ──
    ckpt_dir = Path("cdm_exploration/checkpoints/multi_seed")
    save_checkpoint(
        net, ckpt_dir / f"text_conditioned_seed_{seed}.pt",
        config={"K": K, "n_llms": n_llms, "n_items": n_items,
                "epochs": epochs, "lr": lr, "text_dim": text_dim, "seed": seed},
        train_items=train_items, test_items=test_items,
        val_auc=test_auc, epoch=epochs,
    )
    validate_checkpoint(
        ckpt_dir / f"text_conditioned_seed_{seed}.pt",
        expected_K=K, expected_n_llms=n_llms,
    )
    log_experiment(
        name=f"multi_seed_{seed}",
        config={"K": K, "epochs": epochs, "lr": lr, "device": device, "seed": seed},
        results=fold_result,
        split_info={"n_train_items": int(len(train_items)),
                    "n_test_items": int(len(test_items)), "n_llms": int(n_llms)},
        verified=True,
    )

    # ── Try aggregation (last finisher wins) ──
    maybe_aggregate(out_dir)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
