"""IrtNet under Protocol A (interaction-wise random triplet split) on v2.

Companion to run_irtnet_headtohead.py. Same model (MoEClassifier from
JianhaoChen-nju/IrtNet), same hyperparameters (Birnbaum 2PL likelihood,
their published pipeline.sh defaults), same v2 dataset (3811 LLMs x 9523
items, SBERT all-mpnet-base-v2 embeddings). The only difference is the
split: 80/10/10 random over individual triplets instead of 80/20 over
items.

Reports test AUC + Test Acc on the held-out 10% triplets. Skips routing
Acc@k because routing is a Protocol B concept (per-item argmax over
LLMs); under interaction-wise splitting every item has both train and
test cells, so the routing comparison is not directly meaningful.

SLURM array layout (--array=0-2):
    idx 0: d_model=232  (IrtNet as published)
    idx 1: d_model=512
    idx 2: d_model=1024

Usage (Euler, via SLURM array):
    sbatch tools/run_protocolA.sbatch

Usage (single run, local or interactive):
    python tools/run_irtnet_protocolA.py device=cuda \
        +irtnet.d_model=232 +irtnet.seed=42

Output:
    cdm_exploration/experiments/v2_irtnet_protocolA_d{d_model}_s{seed}.json
    cdm_exploration/checkpoints/irtnet/protocolA_d{d_model}_s{seed}.pt
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


def resolve_sweep_config(cfg: DictConfig) -> tuple[int, int, str]:
    """Map SLURM_ARRAY_TASK_ID or CLI override to (d_model, seed)."""
    cli_d = cfg.get("irtnet", {}).get("d_model") if "irtnet" in cfg else None
    cli_seed = cfg.get("irtnet", {}).get("seed") if "irtnet" in cfg else None
    if cli_d is not None and cli_seed is not None:
        return int(cli_d), int(cli_seed), f"cli(d_model={cli_d}, seed={cli_seed})"

    task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_env is None:
        raise RuntimeError(
            "Must provide SLURM_ARRAY_TASK_ID (0..2) or CLI overrides "
            "`+irtnet.d_model=X +irtnet.seed=Y`.",
        )
    idx = int(task_id_env)
    if not (0 <= idx < len(D_MODEL_GRID)):
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={idx} out of range 0..{len(D_MODEL_GRID) - 1}",
        )
    d_model = D_MODEL_GRID[idx]
    seed = 42  # Protocol A scoped to single seed per the 2026-04-25 request
    return d_model, seed, f"slurm_array_task_id={idx} -> (d_model={d_model}, seed={seed})"


def import_irtnet_model_class() -> type:
    """Import MoEClassifier from the cloned IrtNet repo."""
    irtnet_src = (
        Path(__file__).resolve().parent.parent
        / "cdm_exploration" / "repos" / "IrtNet" / "src"
    )
    if not irtnet_src.exists():
        raise FileNotFoundError(
            f"IrtNet source not found at {irtnet_src}. Clone it with: "
            "git clone https://github.com/JianhaoChen-nju/IrtNet.git "
            f"{irtnet_src.parent}",
        )
    sys.path.insert(0, str(irtnet_src))
    from modules import MoEClassifier  # type: ignore[import-not-found]
    return MoEClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, save_checkpoint

    d_model, seed, run_source = resolve_sweep_config(cfg)
    seed_everything(seed)
    device = resolve_device(cfg.device)
    print(f"Run config source: {run_source}")
    print(
        f"Device: {device}, seed: {seed}, d_model: {d_model}, protocol: A",
    )

    MoEClassifier = import_irtnet_model_class()
    # IrtNet/src/modules.py has module-level `torch.manual_seed(42)` that
    # silently overrides our seed at import. Re-seed here so our seed wins.
    seed_everything(seed)

    # ── Load v2 data ──
    data_dir = Path(cfg.paths.cdm_ready)
    print(f"\nLoading data from {data_dir}")
    R = np.load(data_dir / "response_matrix_v2_full.npy")
    with open(data_dir / "response_matrix_v2_full_llms.json") as f:
        llm_names: list[str] = json.load(f)
    with open(data_dir / "response_matrix_v2_full_items.json") as f:
        item_meta = json.load(f)

    n_llms, n_items = R.shape
    assert len(llm_names) == n_llms, f"LLM name count {len(llm_names)} != {n_llms}"
    assert len(item_meta) == n_items, f"Item meta count {len(item_meta)} != {n_items}"
    data_checksum = hashlib.sha256(R.tobytes()).hexdigest()[:12]
    print(
        f"  Response matrix: {n_llms} LLMs x {n_items} items "
        f"(checksum {data_checksum})",
    )

    # ── SBERT prompt embeddings (same model IrtNet uses) ──
    emb_data = np.load(data_dir / "item_text_embeddings_v2_full.npz")
    embeddings_np = emb_data["embeddings"]
    assert embeddings_np.shape == (n_items, 768), (
        f"Embedding shape {embeddings_np.shape} does not match (n_items, 768) "
        f"= ({n_items}, 768). Index misalignment risk; aborting."
    )
    prompt_embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
    print(f"  SBERT embeddings: {tuple(prompt_embeddings.shape)}")

    # ── Protocol A split: 80/10/10 random over ALL triplets ──
    print("\nBuilding all triplets...")
    all_llm = np.repeat(np.arange(n_llms), n_items)
    all_item = np.tile(np.arange(n_items, dtype=np.int64), n_llms)
    all_label = R[all_llm, all_item].astype(np.float32)
    n_total = len(all_label)
    print(f"  Total triplets: {n_total:,}")

    all_idx = np.arange(n_total)
    trvl_idx, te_idx = train_test_split(
        all_idx, test_size=0.10, random_state=42,
    )
    tr_idx, va_idx = train_test_split(
        trvl_idx, test_size=10.0 / 90.0, random_state=42,
    )
    print(
        f"  Protocol A split: train={len(tr_idx):,}, val={len(va_idx):,}, "
        f"test={len(te_idx):,}",
    )

    # ── Build IrtNet model with published hyperparameters ──
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
    print(f"\nIrtNet config (d_model={d_model}, protocol A):")
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

    # ── Train (mirrors IrtNet/src/train_and_eval.py:train) ──
    from torch import nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, TensorDataset

    pin = device.startswith("cuda")
    train_ds = TensorDataset(
        torch.from_numpy(all_llm[tr_idx]).long(),
        torch.from_numpy(all_item[tr_idx]).long(),
        torch.from_numpy(all_label[tr_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(all_llm[va_idx]).long(),
        torch.from_numpy(all_item[va_idx]).long(),
        torch.from_numpy(all_label[va_idx]),
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

            # IrtNet's expert load-balancing bias update
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
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
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

    # ── Test triplet AUC + Acc ──
    test_ds = TensorDataset(
        torch.from_numpy(all_llm[te_idx]).long(),
        torch.from_numpy(all_item[te_idx]).long(),
        torch.from_numpy(all_label[te_idx]),
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
    print(
        f"\nTest triplet (Protocol A): AUC={test_auc:.4f}, "
        f"Acc={test_acc:.4f}, RMSE={test_rmse:.4f}",
    )

    # ── Persist ──
    results = {
        "dataset": "v2_full",
        "data_checksum": data_checksum,
        "model": "IrtNet",
        "protocol": "A",
        "split": "interaction_wise_80_10_10",
        "d_model": d_model,
        "seed": seed,
        "n_llms": n_llms,
        "n_items": n_items,
        "n_total_triplets": int(n_total),
        "n_train_triplets": int(len(tr_idx)),
        "n_val_triplets": int(len(va_idx)),
        "n_test_triplets": int(len(te_idx)),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "test_auc": test_auc,
        "test_acc": test_acc,
        "test_rmse": test_rmse,
        "hyperparams": {**model_hp, **train_hp},
        "trainable_params": int(n_params),
    }

    out_path = (
        Path("cdm_exploration/experiments")
        / f"v2_irtnet_protocolA_d{d_model}_s{seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-run results: {out_path}")

    ckpt_path = (
        Path("cdm_exploration/checkpoints/irtnet")
        / f"protocolA_d{d_model}_s{seed}.pt"
    )
    save_checkpoint(
        model,
        ckpt_path,
        config={
            "d_model": d_model, "seed": seed, "protocol": "A",
            **model_hp, **train_hp,
        },
        train_items=np.array([], dtype=np.int64),  # protocol A is triplet-wise
        test_items=np.array([], dtype=np.int64),
        val_auc=best_val_acc,
        epoch=best_epoch,
    )

    log_experiment(
        name=f"irtnet_protocolA_d{d_model}_s{seed}",
        config={
            "model": "IrtNet",
            "protocol": "A",
            "d_model": d_model,
            "seed": seed,
            **model_hp, **train_hp,
            "device": str(device),
            "hydra_overrides": OmegaConf.to_container(cfg, resolve=True),
        },
        results=results,
        split_info={
            "split_type": "interaction_wise_80_10_10",
            "n_total_triplets": int(n_total),
            "n_train_triplets": int(len(tr_idx)),
            "n_val_triplets": int(len(va_idx)),
            "n_test_triplets": int(len(te_idx)),
            "n_llms": int(n_llms),
            "n_items": int(n_items),
            "data_checksum": data_checksum,
        },
        verified=True,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
