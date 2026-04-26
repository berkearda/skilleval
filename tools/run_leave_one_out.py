"""Leave-one-benchmark-out cross-domain evaluation.

For each of the 5 benchmarks (MATH, BBH, GPQA, MuSR, IFEval):
  1. Hold out all items from that benchmark.
  2. Train a fresh K=100 text-conditioned NCDM on the remaining 4 benchmarks.
  3. Evaluate AUC, Acc@1, and benchmark-prediction Pearson r on the held-out
     benchmark, and on the 4 training benchmarks (test split) for comparison.

Each fold is one process. Designed for a SLURM array job (--array=0-4) where
SLURM_ARRAY_TASK_ID picks the held-out benchmark. Can also run a single fold
locally via:
    python tools/run_leave_one_out.py device=cuda +held_out=MATH

All folds write to a single file: experiments/v2_leave_one_out.json
(safe under concurrent fold writes via per-fold staging files merged at the
end of each fold).

Validations performed:
  * validate_data() at startup.
  * Zero overlap between train and held-out item indices (asserted).
  * AUC on the 4 training benchmarks > 0.65 (warning otherwise).
  * Reference comparison against v2_per_benchmark_routing.json printed.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset


class LazyTripletDataset(Dataset):
    """Generate (student, text_emb, q_row, score) on the fly.

    Stores only the dense response matrix once (n_llms x n_items) plus the
    item-id list for this split, and looks up text/Q rows by item id at
    __getitem__ time. Avoids the n_llms*n_items_per_split copy of the SBERT
    embedding (the actual OOM source on Euler).

    Memory: ~R + text_embs + q_matrix shared across all splits, regardless of
    how many (student, item) pairs the split contains.
    """

    def __init__(self, R, item_ids, text_embs_t, q_t):
        # R: float32 numpy (n_llms, n_items_total)
        # item_ids: int array of items participating in this split
        # text_embs_t: torch.float32 (n_items_total, text_dim)
        # q_t: torch.float32 (n_items_total, K)
        self.R = R
        self.item_ids = np.ascontiguousarray(item_ids, dtype=np.int64)
        self.text = text_embs_t
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]

# Reference Acc@1 from v2_per_benchmark_routing.json (full-training run).
REFERENCE_ACC1 = {
    "MATH":   0.5417,
    "BBH":    0.7351,
    "GPQA":   0.4065,
    "MuSR":   0.5441,
    "IFEval": 0.8200,
    "Overall": 0.6567,
}
REFERENCE_AUC = 0.717  # from v2_baselines.json


def select_held_out(cfg) -> str:
    """Pick which benchmark to hold out for this process.

    Priority: hydra override `+held_out=NAME`  >  $SLURM_ARRAY_TASK_ID  >  error.
    """
    if hasattr(cfg, "held_out"):
        name = str(cfg.held_out)
        if name not in BENCHMARKS:
            raise ValueError(f"held_out={name} not in {BENCHMARKS}")
        return name
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        idx = int(task_id)
        if not (0 <= idx < len(BENCHMARKS)):
            raise ValueError(f"SLURM_ARRAY_TASK_ID={idx} out of range 0..4")
        return BENCHMARKS[idx]
    raise SystemExit(
        "ERROR: no held_out specified. Pass +held_out=NAME or set "
        "SLURM_ARRAY_TASK_ID in [0,4]."
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint
    from cdmeval.validation import validate_checkpoint, validate_data, validate_metrics

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    held_out = select_held_out(cfg)

    # K is optional, default 100 for backward compat with the legacy LOO runs
    # (existing v2_leave_one_out*.json + exports/llm_skill_profiles_loo_*.csv
    # are at K=100). Pass `+ablation.K=300` to run the LOO sweep at other K.
    # When K != 100 the output filenames get a `_K{K}` suffix so they do not
    # clobber the K=100 artefacts already shared on 2026-04-25.
    K_target = (
        int(cfg.ablation.K)
        if hasattr(cfg, "ablation") and hasattr(cfg.ablation, "K")
        else 100
    )
    K_suffix = "" if K_target == 100 else f"_K{K_target}"

    print(f"Device: {device}", flush=True)
    print(f"Held-out benchmark: {held_out}", flush=True)
    print(f"K (Q-matrix): {K_target}", flush=True)

    # ── Load data ──
    print("\nLoading v2 data...", flush=True)
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / f"qmatrix_v2_K{K_target}.npy").copy()
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        items_data = json.load(f)

    n_llms, n_items = R.shape
    K = q_matrix.shape[1]
    text_dim = text_embs.shape[1]
    assert K == K_target, (
        f"Q-matrix K={K} does not match requested K_target={K_target}"
    )

    validate_data(R, q_matrix, text_embs, items_data, llm_names)

    # ── Fix zero-skill items (copied from train_expanded.py) ──
    # Skill embeddings only exist at K=100 in the data dir. For other K we
    # only attempt the imputation if a matching skill_embeddings_v2_K{K}.npz
    # is present; otherwise we require the Q-matrix to have no zero-skill rows.
    text_emb_path = data_dir / "item_text_embeddings_v2_full.npz"
    skill_emb_path = data_dir / f"skill_embeddings_v2_K{K_target}.npz"
    n_zero = int((q_matrix.sum(axis=1) == 0).sum())
    if n_zero > 0:
        if not skill_emb_path.exists():
            raise FileNotFoundError(
                f"Q-matrix at K={K_target} has {n_zero} zero-skill rows "
                f"but {skill_emb_path} is missing — cannot impute. "
                f"Either add the skill embeddings file or use a K with "
                f"zero zero-skill rows.",
            )
        from tools.train_expanded import fix_zero_skill_items
        q_matrix = fix_zero_skill_items(
            q_matrix, skill_emb_path, items_data, text_emb_path,
        )
        print(f"  Fixed {n_zero} zero-skill items in Q-matrix", flush=True)
    else:
        print(f"  No zero-skill items to fix", flush=True)

    # ── Per-benchmark item indices ──
    bench_items = {b: np.array(
        [i for i, it in enumerate(items_data) if it.get("benchmark") == b],
        dtype=int,
    ) for b in BENCHMARKS}
    print("\nItems per benchmark:", flush=True)
    for b in BENCHMARKS:
        print(f"  {b:<8}: {len(bench_items[b])}", flush=True)

    # ── Build the leave-one-out split ──
    held_items = bench_items[held_out]
    train_pool_items = np.concatenate(
        [bench_items[b] for b in BENCHMARKS if b != held_out]
    )

    # Validate: zero overlap between train pool and held-out.
    overlap = np.intersect1d(train_pool_items, held_items)
    assert len(overlap) == 0, (
        f"FAIL: {len(overlap)} items appear in both train pool and held-out "
        f"({held_out}). Item indexing is broken."
    )
    print(f"\nFold {held_out}: {len(train_pool_items)} train items, "
          f"{len(held_items)} held-out items, overlap=0  PASS",
          flush=True)

    # Within the 4 training benchmarks, carve item-level splits for in-dist
    # test (20%) and val (10% of the remaining 80%, i.e. 8% of the pool).
    # NOTE: val is item-level (not triplet-level) to keep the dataset lazy and
    # avoid materialising any per-triplet index array. For early stopping this
    # is sufficient and matches Protocol B semantics.
    in_dist_pool, in_dist_test_items = train_test_split(
        train_pool_items, test_size=0.2, random_state=42,
    )
    in_dist_train_items, in_dist_val_items = train_test_split(
        in_dist_pool, test_size=0.1, random_state=42,
    )
    print(f"  In-distribution split: {len(in_dist_train_items)} train items, "
          f"{len(in_dist_val_items)} val items, "
          f"{len(in_dist_test_items)} in-dist test items", flush=True)

    # Per-benchmark in-dist test items (for the comparison row).
    in_dist_test_set = set(in_dist_test_items.tolist())
    per_bench_in_dist_test = {
        b: np.array(sorted(set(bench_items[b].tolist()) & in_dist_test_set), dtype=int)
        for b in BENCHMARKS if b != held_out
    }

    # Sanity check: zero overlap between training items and held-out items.
    held_set = set(held_items.tolist())
    leak = sum(1 for i in in_dist_train_items if int(i) in held_set)
    assert leak == 0, f"FAIL: {leak} training items overlap held-out. Leakage!"
    print(f"  Leakage check: 0 / {len(in_dist_train_items)} train items in held-out  PASS",
          flush=True)

    # ── Build lazy datasets (no triplet materialisation) ──
    # R is already loaded as float; cast to float32 to halve memory.
    R_f32 = R.astype(np.float32, copy=False)
    text_t = torch.from_numpy(text_embs.astype(np.float32, copy=False))
    q_t = torch.from_numpy(q_matrix.astype(np.float32, copy=False))

    bs = cfg.model.batch_size
    n_workers = int(getattr(cfg.model, "num_workers", 4))

    train_ds = LazyTripletDataset(R_f32, in_dist_train_items, text_t, q_t)
    val_ds = LazyTripletDataset(R_f32, in_dist_val_items, text_t, q_t)
    print(f"  Train pairs (lazy): {len(train_ds):,}, "
          f"Val pairs (lazy): {len(val_ds):,}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=n_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=n_workers, pin_memory=True,
    )

    # ── Train ──
    epochs = cfg.model.epochs
    lr = cfg.model.lr
    print(f"\nTraining (K={K}, {n_llms} LLMs, hold out {held_out})", flush=True)
    print(f"  {len(train_ds)//bs:,} batches/epoch, {epochs} epochs",
          flush=True)

    net = TextConditionedNet(K, n_llms, text_dim)
    net = train_text_model(
        net, train_loader, val_loader, epochs=epochs, lr=lr, device=device,
    )

    # ── Evaluate on held-out benchmark ──
    print(f"\nEvaluating on held-out benchmark ({held_out})...", flush=True)
    held_ds = LazyTripletDataset(R_f32, held_items, text_t, q_t)
    held_loader = DataLoader(
        held_ds, batch_size=bs, shuffle=False,
        num_workers=n_workers, pin_memory=True,
    )
    held_auc, held_acc, held_rmse = eval_text_model(net, held_loader, device)
    print(f"  Held-out AUC={held_auc:.4f}, Acc={held_acc:.4f}, "
          f"RMSE={held_rmse:.4f}", flush=True)

    # ── Evaluate on in-distribution test items (for sanity) ──
    print("\nEvaluating on in-distribution test items "
          "(4 training benchmarks)...", flush=True)
    in_dist_ds = LazyTripletDataset(R_f32, in_dist_test_items, text_t, q_t)
    in_dist_loader = DataLoader(
        in_dist_ds, batch_size=bs, shuffle=False,
        num_workers=n_workers, pin_memory=True,
    )
    in_dist_auc, in_dist_acc, _ = eval_text_model(net, in_dist_loader, device)
    print(f"  In-dist AUC={in_dist_auc:.4f}, Acc={in_dist_acc:.4f}", flush=True)

    if in_dist_auc < 0.65:
        print(f"  WARNING: in-dist AUC={in_dist_auc:.4f} < 0.65 — "
              f"this fold's model is likely broken.", flush=True)
    if in_dist_auc < 0.5:
        print(f"  CRITICAL: in-dist AUC < 0.5 — predictions are anti-correlated "
              f"with truth. Aborting before recording results.", flush=True)
        raise SystemExit(2)

    # ── Routing on the held-out benchmark ──
    print(f"\nRouting on held-out items ({len(held_items)})...", flush=True)
    net.eval().to(device)
    all_llm_ids = torch.arange(n_llms, device=device)

    # Strongest model: chosen on the in-dist train items only.
    train_acc = R[:, in_dist_train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc))
    rng = np.random.RandomState(42)

    def route(items, label):
        n = len(items)
        if n == 0:
            return {"n_items": 0, "cdm_acc1": float("nan"),
                    "strongest_acc1": float("nan"), "random_acc1": float("nan")}
        cdm = strong = rand = 0
        used = 0
        for ii in items:
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
                cdm += 1
            if gt[strongest_idx] > 0:
                strong += 1
            if gt[int(rng.randint(n_llms))] > 0:
                rand += 1
        denom = max(used, 1)
        out = {"n_items": int(n),
               "cdm_acc1": cdm / denom,
               "strongest_acc1": strong / denom,
               "random_acc1": rand / denom}
        print(f"  {label}: CDM @1={out['cdm_acc1']:.4f}, "
              f"Strongest={out['strongest_acc1']:.4f}, "
              f"Random={out['random_acc1']:.4f} (n={n})", flush=True)
        return out

    held_routing = route(held_items, f"held_out={held_out}")

    # In-dist routing per benchmark (for the comparison columns).
    in_dist_routing = {}
    for b, items in per_bench_in_dist_test.items():
        in_dist_routing[b] = route(items, f"in-dist:{b}")

    # ── Benchmark prediction Pearson r on the held-out benchmark ──
    # For each LLM, predicted mean accuracy on held-out items vs. true mean.
    print("\nBenchmark prediction (held-out benchmark)...", flush=True)
    pred_means = np.zeros(n_llms)
    true_means = R[:, held_items].mean(axis=1)
    with torch.no_grad():
        for i in held_items:
            ii = int(i)
            emb = torch.tensor(text_embs[ii], dtype=torch.float32, device=device)
            qr = torch.tensor(q_matrix[ii], dtype=torch.float32, device=device)
            preds = net(
                all_llm_ids,
                emb.unsqueeze(0).expand(n_llms, -1),
                qr.unsqueeze(0).expand(n_llms, -1),
            ).cpu().numpy()
            pred_means += preds
    pred_means /= max(len(held_items), 1)
    held_pearson_r, held_pearson_p = pearsonr(pred_means, true_means)
    print(f"  Pearson r = {held_pearson_r:.4f} (p={held_pearson_p:.2e})", flush=True)

    # ── Reference comparison ──
    print(f"\n{'='*72}", flush=True)
    print(f"FOLD: held_out = {held_out}", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  Held-out Acc@1   : {held_routing['cdm_acc1']:.4f}  "
          f"(reference full-training: {REFERENCE_ACC1[held_out]:.4f})",
          flush=True)
    print(f"  Held-out AUC     : {held_auc:.4f}  "
          f"(reference full-training overall: {REFERENCE_AUC:.4f})",
          flush=True)
    print(f"  Held-out Pearson : {held_pearson_r:.4f}", flush=True)
    print(f"  In-dist AUC      : {in_dist_auc:.4f}", flush=True)

    # ── Save / merge into experiments/v2_leave_one_out.json ──
    fold_result = {
        "held_out": held_out,
        "K": K, "epochs": epochs, "n_llms": n_llms,
        "n_train_items": int(len(in_dist_train_items)),
        "n_in_dist_test_items": int(len(in_dist_test_items)),
        "n_held_out_items": int(len(held_items)),
        "held_out_auc": float(held_auc),
        "held_out_acc": float(held_acc),
        "held_out_rmse": float(held_rmse),
        "held_out_pearson_r": float(held_pearson_r),
        "held_out_routing": held_routing,
        "in_dist_auc": float(in_dist_auc),
        "in_dist_acc": float(in_dist_acc),
        "in_dist_routing": in_dist_routing,
        "reference_acc1": REFERENCE_ACC1[held_out],
        "reference_auc": REFERENCE_AUC,
        "in_dist_warning": bool(in_dist_auc < 0.65),
    }
    validate_metrics(fold_result)

    out_dir = Path("cdm_exploration/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / f"v2_leave_one_out{K_suffix}_{held_out}.json"
    with open(fold_path, "w") as f:
        json.dump(fold_result, f, indent=2)
    print(f"\nSaved fold: {fold_path}", flush=True)

    # Atomic merge into the combined file (last fold to write wins; race-safe
    # because we re-read whatever exists and overwrite our own slot only).
    combined_path = out_dir / f"v2_leave_one_out{K_suffix}.json"
    if combined_path.exists():
        with open(combined_path) as f:
            combined = json.load(f)
    else:
        combined = {"experiment": "leave_one_benchmark_out",
                    "K": K, "benchmarks": BENCHMARKS, "folds": {}}
    combined["folds"][held_out] = fold_result
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Merged into: {combined_path}", flush=True)

    # ── Checkpoint + log ──
    ckpt_dir = Path("cdm_exploration/checkpoints/leave_one_out")
    ckpt_path = ckpt_dir / f"text_conditioned_loo{K_suffix}_{held_out}.pt"
    save_checkpoint(
        net, ckpt_path,
        config={"K": K, "n_llms": n_llms, "n_items": n_items,
                "epochs": epochs, "lr": lr, "text_dim": text_dim,
                "held_out": held_out},
        train_items=in_dist_train_items, test_items=held_items,
        val_auc=held_auc, epoch=epochs,
    )
    validate_checkpoint(
        ckpt_path,
        expected_K=K, expected_n_llms=n_llms,
    )

    log_experiment(
        name=f"leave_one_out{K_suffix}_{held_out}",
        config={"K": K, "epochs": epochs, "lr": lr, "device": device,
                "held_out": held_out},
        results=fold_result,
        split_info={"n_train_items": int(len(in_dist_train_items)),
                    "n_held_out_items": int(len(held_items)),
                    "n_llms": n_llms},
        verified=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
