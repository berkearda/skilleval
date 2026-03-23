"""LLM cold-start evaluation: profile new LLMs from few responses.

Usage:
    python tools/llm_cold_start.py device=mps
    python tools/llm_cold_start.py device=mps model.epochs=1 '+cold_start.calibration_sizes=[50,200]' '+cold_start.n_repeats=2'
"""

import json
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.dataloader import make_text_dataloader
    from cdmeval.data.response_matrix import build_triplets, load_q_matrix, load_response_matrix
    from cdmeval.evaluation.llm_cold_start import (
        evaluate_llm_cold_start,
        llm_cold_start_split,
        plot_cold_start_curve,
    )
    from cdmeval.evaluation.metrics import eval_text_model
    from cdmeval.evaluation.training import train_text_model
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    K = cfg.skills.n_clusters
    print(f"Device: {device}")

    # Config overrides for cold-start params
    if hasattr(cfg, "cold_start"):
        cal_sizes = list(cfg.cold_start.calibration_sizes)
        n_repeats = cfg.cold_start.n_repeats
    else:
        cal_sizes = [0, 1, 3, 5, 10, 25, 50, 100, 200, 500]
        n_repeats = 5
    print(f"Calibration sizes: {cal_sizes}")
    print(f"Repeats per size: {n_repeats}")

    # ── Load data ──
    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    q_matrix, skill_cols = load_q_matrix(data_dir / f"q_matrix_hac{K}.csv")
    n_skills = q_matrix.shape[1]
    triplets = build_triplets(response_df)
    response_vals = response_df.values

    text_embeddings = np.load(data_dir / "item_text_embeddings.npz")["embeddings"]
    text_dim = text_embeddings.shape[1]

    print(f"Data: {n_llms} LLMs, {n_items} items, {n_skills} skills")

    # ── LLM-level split ──
    train_llms, test_llms = llm_cold_start_split(n_llms, test_fraction=0.2, seed=42)
    print(f"LLM split: {len(train_llms)} train, {len(test_llms)} test")

    # ── Item-level split (Protocol B, same as other experiments) ──
    all_items = np.arange(n_items)
    train_items, test_item_set = train_test_split(
        all_items, test_size=0.2, random_state=42
    )
    print(f"Item split: {len(train_items)} train, {len(test_item_set)} test")

    # ── Train text-conditioned NCDM on training LLMs only ──
    print("\n" + "=" * 60)
    print("Training TextConditionedNet on training LLMs")
    print("=" * 60)

    # Filter triplets: only training LLMs and training items
    train_mask = (
        np.isin(triplets[:, 0].astype(int), train_llms)
        & np.isin(triplets[:, 1].astype(int), train_items)
    )
    train_triplets = triplets[train_mask]

    tv_idx = np.arange(len(train_triplets))
    tr_idx, va_idx = train_test_split(tv_idx, test_size=0.1, random_state=42)

    bs = cfg.model.batch_size
    train_loader = make_text_dataloader(
        train_triplets[tr_idx], text_embeddings, q_matrix, bs, shuffle=True
    )
    val_loader = make_text_dataloader(
        train_triplets[va_idx], text_embeddings, q_matrix, bs, shuffle=False
    )

    net = TextConditionedNet(n_skills, n_llms, text_dim)
    net = train_text_model(
        net, train_loader, val_loader,
        epochs=cfg.model.epochs, lr=cfg.model.lr, device=device,
    )

    # ── Compute full-training baseline AUC ──
    # Evaluate trained model on training LLMs + test items (seen LLMs, unseen items)
    full_test_mask = (
        np.isin(triplets[:, 0].astype(int), train_llms)
        & np.isin(triplets[:, 1].astype(int), test_item_set)
    )
    full_test_trips = triplets[full_test_mask]
    full_test_loader = make_text_dataloader(
        full_test_trips, text_embeddings, q_matrix, bs, shuffle=False
    )
    full_auc, full_acc, full_rmse = eval_text_model(net, full_test_loader, device)
    print(f"\nFull-training baseline (seen LLMs, unseen items):")
    print(f"  AUC={full_auc:.4f}, Acc={full_acc:.4f}, RMSE={full_rmse:.4f}")

    # ── Cold-start evaluation ──
    print("\n" + "=" * 60)
    print("Cold-Start Evaluation on Held-Out LLMs")
    print("=" * 60)

    # Calibrate on training items only, evaluate on test items only
    results_df = evaluate_llm_cold_start(
        trained_net=net,
        test_llm_indices=test_llms,
        response_matrix=response_vals,
        q_matrix=q_matrix,
        text_embeddings=text_embeddings,
        calibration_pool=train_items,
        eval_items=test_item_set,
        calibration_sizes=cal_sizes,
        device=device,
        n_repeats=n_repeats,
        seed=42,
        lr=0.01,
        fit_epochs=100,
    )

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("LLM COLD-START SUMMARY")
    print("=" * 70)
    header = f"{'N cal':>8}  {'AUC mean':>10}  {'AUC std':>10}  {'Acc mean':>10}  {'Acc std':>10}"
    print(header)
    print("-" * 70)

    agg = results_df.groupby("n_calibration").agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
    ).reset_index()

    for _, row in agg.iterrows():
        print(
            f"{int(row['n_calibration']):>8}  "
            f"{row['auc_mean']:>10.4f}  {row['auc_std']:>10.4f}  "
            f"{row['acc_mean']:>10.4f}  {row['acc_std']:>10.4f}"
        )

    print(f"\nFull-training AUC: {full_auc:.4f}")
    threshold_90 = full_auc * 0.9
    for _, row in agg.iterrows():
        if row["auc_mean"] >= threshold_90:
            print(
                f"90% threshold ({threshold_90:.4f}) reached at "
                f"N={int(row['n_calibration'])} items "
                f"(AUC={row['auc_mean']:.4f})"
            )
            break

    # ── Save results ──
    results_path = data_dir / "llm_cold_start_results.json"
    save_data = {
        "full_training_auc": float(full_auc),
        "n_train_llms": len(train_llms),
        "n_test_llms": len(test_llms),
        "calibration_sizes": cal_sizes,
        "n_repeats": n_repeats,
        "summary": agg.to_dict(orient="records"),
        "detailed": results_df.to_dict(orient="records"),
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    csv_path = data_dir / "llm_cold_start_results.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"CSV saved: {csv_path}")

    # ── Figure ──
    plot_cold_start_curve(results_df, full_auc, fig_dir)

    # ── Verify and log ──
    from cdmeval.utils.experiment import verify_splits, log_experiment
    verified = verify_splits(
        train_items, test_item_set,
        eval_items=test_item_set,
        calibration_items=train_items,
        label="cold_start",
    )
    log_experiment(
        name="llm_cold_start",
        config={"K": K, "epochs": cfg.model.epochs, "lr": cfg.model.lr,
                "device": device, "seed": 42, "cal_sizes": cal_sizes,
                "n_repeats": n_repeats},
        results={"full_auc": float(full_auc),
                 "summary": agg.to_dict(orient="records")},
        split_info={"n_train_items": len(train_items),
                    "n_test_items": len(test_item_set),
                    "n_train_llms": len(train_llms),
                    "n_test_llms": len(test_llms)},
        verified=verified,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
