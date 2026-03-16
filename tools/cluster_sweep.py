"""Cluster sweep: evaluate NCDM performance across K values.

Trains ID-based (Protocol A) and text-conditioned (Protocol B) NCDMs for each K,
producing a summary table, results JSON, and two analysis figures.

Usage:
    python tools/cluster_sweep.py
    python tools/cluster_sweep.py device=mps
    python tools/cluster_sweep.py device=cpu model.epochs=1 'sweep.k_values=[30,50,70]'
"""

import json
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.data.response_matrix import load_response_matrix
    from cdmeval.skills.cluster_sweep import run_cluster_sweep
    from cdmeval.skills.clustering import load_skills
    from cdmeval.utils.cluster_analysis import plot_cluster_examples, plot_k_vs_metrics
    from cdmeval.utils.device import resolve_device

    data_dir = Path(cfg.paths.cdm_ready)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # K values from config override or default
    if OmegaConf.is_missing(cfg, "sweep") or not hasattr(cfg, "sweep"):
        k_values = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]
    else:
        k_values = list(cfg.sweep.k_values)
    print(f"K values: {k_values}")

    # ── Load data ──
    emb_data = np.load(data_dir / "skill_embeddings.npz", allow_pickle=True)
    skill_embeddings = emb_data["embeddings"]
    unique_skills = list(emb_data["skills"])
    print(f"Skill embeddings: {skill_embeddings.shape}")

    data, all_skills, _, counts = load_skills(data_dir / "skills_extracted.json")

    response_df, n_llms, n_items, llm_names = load_response_matrix(
        data_dir / "response_matrix.csv"
    )
    print(f"Response matrix: {n_llms} LLMs x {n_items} items")

    # Load cached item text embeddings
    text_emb_path = data_dir / "item_text_embeddings.npz"
    if text_emb_path.exists():
        text_embeddings = np.load(text_emb_path)["embeddings"]
        print(f"Item text embeddings: {text_embeddings.shape}")
    else:
        from sentence_transformers import SentenceTransformer

        texts = [item["problem"] for item in data]
        print(f"Encoding {len(texts)} item texts with all-mpnet-base-v2...")
        model = SentenceTransformer("all-mpnet-base-v2")
        text_embeddings = model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True, batch_size=64
        )
        np.savez(text_emb_path, embeddings=text_embeddings)
        print(f"Saved item text embeddings: {text_embeddings.shape}")

    # ── Run sweep ──
    sweep_df = run_cluster_sweep(
        k_values=k_values,
        skill_embeddings=skill_embeddings,
        unique_skills=unique_skills,
        all_skills=all_skills,
        skill_counts=counts,
        response_df=response_df,
        text_embeddings=text_embeddings,
        n_llms=n_llms,
        n_items=n_items,
        llm_names=llm_names,
        epochs=cfg.model.epochs,
        lr=cfg.model.lr,
        batch_size=cfg.model.batch_size,
        device=device,
        seed=42,
    )

    # ── Save results ──
    results_path = data_dir / "cluster_sweep_results.json"
    sweep_df.to_json(results_path, orient="records", indent=2)
    print(f"\nResults saved: {results_path}")

    csv_path = data_dir / "cluster_sweep_results.csv"
    sweep_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"CSV saved: {csv_path}")

    # ── Generate figures ──
    print("\nGenerating figures...")
    plot_k_vs_metrics(sweep_df, fig_dir)

    example_ks = [20, 50, 100]
    available = [k for k in example_ks if k in k_values]
    if len(available) >= 2:
        plot_cluster_examples(available, skill_embeddings, unique_skills, counts, fig_dir)
    else:
        # Fall back to min, median, max from sweep
        sorted_ks = sorted(k_values)
        fallback = [sorted_ks[0], sorted_ks[len(sorted_ks) // 2], sorted_ks[-1]]
        plot_cluster_examples(fallback, skill_embeddings, unique_skills, counts, fig_dir)

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("CLUSTER SWEEP SUMMARY")
    print("=" * 90)
    header = (
        f"{'K':>5}  {'AUC(A)':>8}  {'Acc(A)':>8}  "
        f"{'AUC(B)':>8}  {'Rt@1':>7}  {'Rt@5':>7}  "
        f"{'Silh':>7}  {'IntraSim':>8}  {'Sk/Item':>7}"
    )
    print(header)
    print("-" * 90)
    for _, row in sweep_df.iterrows():
        print(
            f"{int(row['K']):>5}  "
            f"{row['proto_a_auc']:>8.4f}  {row['proto_a_acc']:>8.4f}  "
            f"{row['proto_b_auc']:>8.4f}  "
            f"{row['routing_acc1']:>7.4f}  {row['routing_acc5']:>7.4f}  "
            f"{row['silhouette']:>7.4f}  "
            f"{row['mean_intra_cosine_sim']:>8.4f}  "
            f"{row['mean_skills_per_item']:>7.2f}"
        )

    # Highlight best
    best_a = sweep_df.loc[sweep_df["proto_a_auc"].idxmax()]
    best_b = sweep_df.loc[sweep_df["proto_b_auc"].idxmax()]
    best_rt = sweep_df.loc[sweep_df["routing_acc1"].idxmax()]
    print(f"\nBest Protocol A AUC:   K={int(best_a['K'])} ({best_a['proto_a_auc']:.4f})")
    print(f"Best Protocol B AUC:   K={int(best_b['K'])} ({best_b['proto_b_auc']:.4f})")
    print(f"Best Routing Acc@1:    K={int(best_rt['K'])} ({best_rt['routing_acc1']:.4f})")
    print("\nDone.")


if __name__ == "__main__":
    main()
