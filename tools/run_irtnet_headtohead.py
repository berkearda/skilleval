"""IrtNet head-to-head baseline on CDMEval's v2 dataset.

Reproduces IrtNet (Chen et al., arXiv:2510.00844) on the same
3811 x 9523 response matrix and the same 1905 held-out test items used by
CDMEval and the IRT 2PL baseline. Results are directly comparable with
tools/run_irt_baseline_v2.py and Table 1 of the paper.

Design choices (kept strict to avoid any accusation of baseline handicap):
  * IrtNet hyperparameters taken verbatim from the published pipeline.sh
    (`num_experts=39`, `top_k=39`, `expert_hidden=512`, `shared_hidden=512`,
    `expert_output=256`, `dropout=0.5`, `embedding_noise=0.05`, `lr=1e-4`,
    `weight_decay=1e-4`, `batch_size=2048`, `epochs=30`, `early_stop=5`,
    `bias_update_speed=0.01`). Only `model_embed_dim` is swept because
    IrtNet was tuned at d=232 for 112 LLMs; our 3811-LLM population calls
    for a capacity sensitivity check.
  * Data split uses `train_test_split(np.arange(9523), test_size=0.2,
    random_state=42)` — identical to Protocol B used by every other
    CDMEval baseline. The val split is the standard 10% of train
    triplets at `random_state=42`.
  * SBERT prompt embeddings are reused from
    `item_text_embeddings_v2_full.npz` (same `all-mpnet-base-v2` model
    IrtNet uses), indexed in the same item_idx order as the response
    matrix columns.
  * Training loop mirrors IrtNet's `train_and_eval.train` (including
    the unassisted expert-bias load-balancing update). Early stopping
    watches validation accuracy only; the test set is evaluated once at
    the end.

SLURM array layout (--array=0-8):
    idx 0-2: d_model=232, seed in {42, 43, 44}  # IrtNet-as-published
    idx 3-5: d_model=512, seed in {42, 43, 44}  # 2x capacity sensitivity
    idx 6-8: d_model=1024, seed in {42, 43, 44} # 4x capacity sensitivity

Usage (Euler, via SLURM array):
    sbatch tools/run_irtnet_headtohead.sbatch

Usage (single run, local or interactive):
    python tools/run_irtnet_headtohead.py device=cuda \
        +irtnet.d_model=232 +irtnet.seed=42
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


D_MODEL_GRID = [232, 512, 1024]
SEED_GRID = [42, 43, 44]


def resolve_sweep_config(cfg: DictConfig) -> tuple[int, int, str]:
    """Map SLURM array index or CLI override to a (d_model, seed) pair."""
    cli_d = cfg.get("irtnet", {}).get("d_model") if "irtnet" in cfg else None
    cli_seed = cfg.get("irtnet", {}).get("seed") if "irtnet" in cfg else None
    if cli_d is not None and cli_seed is not None:
        return int(cli_d), int(cli_seed), f"cli(d_model={cli_d}, seed={cli_seed})"

    task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_env is None:
        raise RuntimeError(
            "Must provide SLURM_ARRAY_TASK_ID (0..8) or CLI overrides "
            "`+irtnet.d_model=X +irtnet.seed=Y`."
        )
    idx = int(task_id_env)
    if not (0 <= idx < len(D_MODEL_GRID) * len(SEED_GRID)):
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={idx} out of range 0..{len(D_MODEL_GRID) * len(SEED_GRID) - 1}"
        )
    d_model = D_MODEL_GRID[idx // len(SEED_GRID)]
    seed = SEED_GRID[idx % len(SEED_GRID)]
    return d_model, seed, f"slurm_array_task_id={idx} -> (d_model={d_model}, seed={seed})"


def import_irtnet_model_class() -> type:
    """Import MoEClassifier from the cloned IrtNet repo."""
    irtnet_src = Path(__file__).resolve().parent.parent / "cdm_exploration" / "repos" / "IrtNet" / "src"
    if not irtnet_src.exists():
        raise FileNotFoundError(
            f"IrtNet source not found at {irtnet_src}. "
            "Clone it with: "
            "git clone https://github.com/JianhaoChen-nju/IrtNet.git "
            f"{irtnet_src.parent}"
        )
    sys.path.insert(0, str(irtnet_src))
    from modules import MoEClassifier  # type: ignore[import-not-found]
    return MoEClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.baselines import evaluate_rankings
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import (
        log_experiment,
        save_checkpoint,
        verify_splits,
    )

    d_model, seed, run_source = resolve_sweep_config(cfg)
    seed_everything(seed)
    device = resolve_device(cfg.device)
    print(f"Run config source: {run_source}")
    print(f"Device: {device}, seed: {seed}, d_model: {d_model}")

    MoEClassifier = import_irtnet_model_class()

    # ── Load v2 data ──
    data_dir = Path(cfg.paths.cdm_ready)
    print(f"\nLoading data from {data_dir}")
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names: list[str] = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        item_ids_global = json.load(f)

    n_llms, n_items = R.shape
    assert len(llm_names) == n_llms, f"LLM name count {len(llm_names)} != {n_llms}"
    assert len(item_ids_global) == n_items, f"Item id count {len(item_ids_global)} != {n_items}"
    data_checksum = hashlib.sha256(R.tobytes()).hexdigest()[:12]
    print(f"  Response matrix: {n_llms} LLMs x {n_items} items (checksum {data_checksum})")

    # ── SBERT prompt embeddings (all-mpnet-base-v2, same model IrtNet uses) ──
    emb_path = data_dir / "item_text_embeddings_v2_full.npz"
    emb_data = np.load(emb_path)
    embeddings_np = emb_data["embeddings"]
    assert embeddings_np.shape == (n_items, 768), (
        f"Embedding shape {embeddings_np.shape} does not match (n_items, 768) "
        f"= ({n_items}, 768). Index misalignment risk — aborting."
    )
    prompt_embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
    print(f"  SBERT embeddings: {tuple(prompt_embeddings.shape)}")

    # ── Protocol B split (identical to all other v2 baselines) ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(
        all_items, test_size=0.2, random_state=42,
    )
    print(f"  Train items: {len(train_items)}, Test items: {len(test_items)}")

    # ── Build triplets ──
    tr_llm = np.repeat(np.arange(n_llms), len(train_items))
    tr_item = np.tile(train_items.astype(np.int64), n_llms)
    tr_label = R[tr_llm, tr_item].astype(np.float32)

    tr_all = np.arange(len(tr_label))
    tr_idx, va_idx = train_test_split(tr_all, test_size=0.1, random_state=42)

    te_llm = np.repeat(np.arange(n_llms), len(test_items))
    te_item = np.tile(test_items.astype(np.int64), n_llms)
    te_label = R[te_llm, te_item].astype(np.float32)
    print(f"  Train triplets: {len(tr_idx):,}, val: {len(va_idx):,}, test: {len(te_label):,}")

    verified = verify_splits(
        train_items, test_items, label=f"irtnet_d{d_model}_s{seed}",
    )
    if not verified:
        raise RuntimeError("Split verification failed — aborting.")

    # ── Build model with IrtNet's published hyperparameters ──
    model_hp = {
        "model_embed_dim": d_model,
        "num_experts": 39,
        "top_k_experts": 39,
        "expert_hidden_dim": 512,
        "shared_expert_hidden_dim": 512,
        "expert_output_dim": 256,
        "dropout_rate": 0.5,
        "embedding_noise": 0.05,
    }
    train_hp = {
        "epochs": 30,
        "batch_size": 2048,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "bias_update_speed": 0.01,
        "early_stopping_patience": 5,
    }
    print(f"\nIrtNet config (d_model={d_model}):")
    for k, v in {**model_hp, **train_hp}.items():
        print(f"    {k}: {v}")

    model = MoEClassifier(
        num_models=n_llms,
        num_prompts=n_items,
        prompt_embeddings=prompt_embeddings,
        **model_hp,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ── Train (mirrors IrtNet/src/train_and_eval.py:train, with our seeding & logging) ──
    from torch import nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset

    pin = device.startswith("cuda")
    train_ds = TensorDataset(
        torch.from_numpy(tr_llm[tr_idx]).long(),
        torch.from_numpy(tr_item[tr_idx]).long(),
        torch.from_numpy(tr_label[tr_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(tr_llm[va_idx]).long(),
        torch.from_numpy(tr_item[va_idx]).long(),
        torch.from_numpy(tr_label[va_idx]),
    )
    train_loader = DataLoader(
        train_ds, batch_size=train_hp["batch_size"], shuffle=True,
        num_workers=0, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_hp["batch_size"], shuffle=False,
        num_workers=0, pin_memory=pin, drop_last=False,
    )

    optimizer = Adam(
        model.parameters(), lr=train_hp["lr"],
        weight_decay=train_hp["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    best_val_acc = 0.0
    best_epoch = 0
    best_state: dict | None = None
    epochs_no_improve = 0

    print("\nTraining...")
    for epoch in range(train_hp["epochs"]):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for model_ids, prompt_ids, labels in train_loader:
            model_ids = model_ids.to(device)
            prompt_ids = prompt_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            a_q, b_q, gating_logits, _ = model.analyze_prompt(prompt_ids)
            theta = model.model_embedder(model_ids)
            ability_term = torch.sum(a_q * theta, dim=1, keepdim=True)
            logit = (ability_term - b_q).squeeze(-1)
            loss = loss_fn(logit, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

            # IrtNet's unassisted expert load-balancing bias update.
            with torch.no_grad():
                mean_expert_logits = gating_logits.mean(dim=0)
                ideal = mean_expert_logits.mean()
                over = mean_expert_logits > ideal
                under = mean_expert_logits < ideal
                upd = torch.zeros_like(model.routed_moe.bias)
                upd.masked_fill_(over, -train_hp["bias_update_speed"])
                upd.masked_fill_(under, train_hp["bias_update_speed"])
                model.routed_moe.bias.add_(upd)

        avg_train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum, val_true, val_pred = 0.0, [], []
        nb_val = 0
        with torch.no_grad():
            for model_ids, prompt_ids, labels in val_loader:
                model_ids = model_ids.to(device)
                prompt_ids = prompt_ids.to(device)
                labels = labels.to(device)
                logits = model(model_ids, prompt_ids)
                val_loss_sum += loss_fn(logits, labels).item()
                probs = torch.sigmoid(logits).cpu().numpy()
                val_pred.extend((probs > 0.5).astype(int).tolist())
                val_true.extend(labels.cpu().numpy().astype(int).tolist())
                nb_val += 1
        avg_val_loss = val_loss_sum / max(nb_val, 1)
        val_acc = float(accuracy_score(val_true, val_pred))
        scheduler.step(avg_val_loss)

        print(
            f"  Epoch {epoch + 1:2d} | train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= train_hp["early_stopping_patience"]:
            print(
                f"  Early stop at epoch {epoch + 1} "
                f"(no improvement for {epochs_no_improve} epochs)",
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"\nBest val_acc={best_val_acc:.4f} at epoch {best_epoch}")

    # ── Test triplet AUC ──
    test_ds = TensorDataset(
        torch.from_numpy(te_llm).long(),
        torch.from_numpy(te_item).long(),
        torch.from_numpy(te_label),
    )
    test_loader = DataLoader(
        test_ds, batch_size=4096, shuffle=False, num_workers=0, pin_memory=pin,
    )
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for model_ids, prompt_ids, labels in test_loader:
            model_ids = model_ids.to(device)
            prompt_ids = prompt_ids.to(device)
            logits = model(model_ids, prompt_ids)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred_all.extend(probs.tolist())
            y_true_all.extend(labels.cpu().numpy().astype(int).tolist())
    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    test_auc = float(roc_auc_score(y_true_all, y_pred_all))
    test_acc = float(accuracy_score(y_true_all, (y_pred_all >= 0.5).astype(int)))
    test_rmse = float(np.sqrt(((y_true_all - y_pred_all) ** 2).mean()))
    print(f"\nTest triplet: AUC={test_auc:.4f}, Acc={test_acc:.4f}, RMSE={test_rmse:.4f}")

    # ── Routing Acc@k (reuses cdmeval.evaluation.baselines.evaluate_rankings) ──
    all_llm_ids = torch.arange(n_llms, device=device)
    rankings: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for raw_idx in test_items:
            item_idx = int(raw_idx)
            item_t = torch.full((n_llms,), item_idx, dtype=torch.int64, device=device)
            logits = model(all_llm_ids, item_t).cpu().numpy()
            rankings[item_idx] = np.argsort(-logits)
    routing = evaluate_rankings(rankings, R, ks=(1, 3, 5, 10))
    print(
        "Routing: "
        + ", ".join(
            f"@{k}={routing[f'accuracy@{k}']:.4f}" for k in (1, 3, 5, 10)
        )
        + f" (n_items={routing['n_items']})",
    )

    # ── Strongest model baseline for reporting consistency ──
    train_acc_per_llm = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_acc_per_llm))
    strongest_correct = float(sum(R[strongest_idx, int(i)] for i in test_items))
    strongest_acc1 = strongest_correct / len(test_items)

    # ── Consolidate and persist ──
    results = {
        "dataset": "v2_full",
        "data_checksum": data_checksum,
        "model": "IrtNet",
        "d_model": d_model,
        "seed": seed,
        "n_llms": n_llms,
        "n_items": n_items,
        "n_test_items": int(len(test_items)),
        "n_test_items_evaluated": int(routing["n_items"]),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_rmse": test_rmse,
        "routing_acc@1": float(routing["accuracy@1"]),
        "routing_acc@3": float(routing["accuracy@3"]),
        "routing_acc@5": float(routing["accuracy@5"]),
        "routing_acc@10": float(routing["accuracy@10"]),
        "strongest_acc1": float(strongest_acc1),
        "strongest_model": llm_names[strongest_idx],
        "hyperparams": {**model_hp, **train_hp},
        "trainable_params": int(n_params),
    }

    out_path = Path("cdm_exploration/experiments") / f"v2_irtnet_d{d_model}_s{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-run results: {out_path}")

    ckpt_path = Path("cdm_exploration/checkpoints/irtnet") / f"d{d_model}_s{seed}.pt"
    save_checkpoint(
        model,
        ckpt_path,
        config={"d_model": d_model, "seed": seed, **model_hp, **train_hp},
        train_items=train_items,
        test_items=test_items,
        val_auc=best_val_acc,
        epoch=best_epoch,
    )

    log_experiment(
        name=f"irtnet_d{d_model}_s{seed}",
        config={
            "model": "IrtNet",
            "d_model": d_model,
            "seed": seed,
            **model_hp,
            **train_hp,
            "device": str(device),
            "hydra_overrides": OmegaConf.to_container(cfg, resolve=True),
        },
        results=results,
        split_info={
            "n_train_items": int(len(train_items)),
            "n_test_items": int(len(test_items)),
            "n_llms": int(n_llms),
            "n_train_triplets": int(len(tr_idx)),
            "n_val_triplets": int(len(va_idx)),
            "data_checksum": data_checksum,
        },
        verified=verified,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
