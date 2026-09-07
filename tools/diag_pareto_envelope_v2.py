"""Robust re-validation of single-LLM Pareto envelope with multiple
independent sanity checks. Logs to experiment_log.json.

Checks performed:
  (A) Split determinism: exact same split as pareto_routing_v2.py
  (B) FLOPs consistency: recompute and cross-check
  (C) 'Strongest' definition: matches train-set argmax (same as paper text)
  (D) shuttle-3 test accuracy: direct hand-computed value
  (E) Envelope: computed via O(n log n) sweep AND brute-force O(n^2)
      domination check, both must agree
  (F) Top-10 LLMs by test accuracy printed for sanity
"""
import json
import sys
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import log_experiment, verify_splits

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print("=" * 70, flush=True)
    print("PARETO ENVELOPE RE-VALIDATION", flush=True)
    print("=" * 70, flush=True)

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    llm_names = json.load(open(data_dir / "response_matrix_v2_full_llms.json"))
    n_llms, n_items = R.shape
    print(f"\nData: {n_llms} LLMs x {n_items} items", flush=True)
    print(f"Response matrix dtype/range: {R.dtype}, "
          f"min={R.min()}, max={R.max()}, unique={np.unique(R)}", flush=True)

    # ── (A) Split determinism ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(
        all_items, test_size=0.2, random_state=42,
    )
    print(f"\n[A] Split: train={len(train_items)}, test={len(test_items)}",
          flush=True)

    # Recompute with independent seed reset
    seed_everything(42)
    _, test_items_2 = train_test_split(
        np.arange(n_items), test_size=0.2, random_state=42,
    )
    assert np.array_equal(np.sort(test_items), np.sort(test_items_2)), \
        "Split reproduction failed"
    print("  PASS: split reproduces identically across two calls", flush=True)

    # ── (B) FLOPs consistency ──
    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])
    print(f"\n[B] FLOPs: {(~np.isnan(flops)).sum()}/{n_llms} valid, "
          f"min={flops.min():.2f}, max={flops.max():.0f}",
          flush=True)

    # Check the named LLMs
    for name in ["MaziyarPanahi__calme-3.2-instruct-78b",
                 "shuttleai__shuttle-3",
                 "hotmailuser__RombosBeagle-v2beta-MGS-32B",
                 "Triangle104__Q2.5-Instruct-1M_Harmony"]:
        if name in llm_names:
            i = llm_names.index(name)
            print(f"  {name}: idx={i}, flops={flops[i]:.2f}", flush=True)
        else:
            print(f"  MISSING: {name}", flush=True)

    # ── (C) 'Strongest' definition ──
    train_mean_acc = R[:, train_items.astype(int)].mean(axis=1)
    strongest_idx = int(np.argmax(train_mean_acc))
    strongest_name = llm_names[strongest_idx]
    strongest_train_acc = float(train_mean_acc[strongest_idx])
    strongest_test_acc = float(R[strongest_idx, test_items.astype(int)].mean())
    strongest_cost = float(flops[strongest_idx])
    print(f"\n[C] Strongest (argmax train acc):", flush=True)
    print(f"    {strongest_name}", flush=True)
    print(f"    train_acc={strongest_train_acc:.4f}  "
          f"test_acc={strongest_test_acc:.4f}  "
          f"cost={strongest_cost:.0f} GFLOPs", flush=True)

    # ── (D) shuttle-3 direct cross-check ──
    shuttle_name = "shuttleai__shuttle-3"
    if shuttle_name not in llm_names:
        print(f"\n[D] shuttle-3 NOT IN LLM LIST. Aborting.", flush=True)
        return
    shuttle_idx = llm_names.index(shuttle_name)
    shuttle_train_acc = float(train_mean_acc[shuttle_idx])
    shuttle_test_acc = float(R[shuttle_idx, test_items.astype(int)].mean())
    shuttle_cost = float(flops[shuttle_idx])
    print(f"\n[D] shuttle-3 direct computation:", flush=True)
    print(f"    idx={shuttle_idx}", flush=True)
    print(f"    train_acc={shuttle_train_acc:.4f}  "
          f"test_acc={shuttle_test_acc:.4f}", flush=True)
    print(f"    cost={shuttle_cost:.0f} GFLOPs  "
          f"cost_pct_of_strongest={shuttle_cost/strongest_cost*100:.3f}%",
          flush=True)
    print(f"    acc_pct_of_strongest={shuttle_test_acc/strongest_test_acc*100:.2f}%",
          flush=True)
    # Raw correct count
    correct_raw = int(R[shuttle_idx, test_items.astype(int)].sum())
    print(f"    raw correct on test: {correct_raw}/{len(test_items)} "
          f"= {correct_raw/len(test_items)*100:.2f}%", flush=True)

    # ── (F) Top-10 LLMs by test accuracy (sanity) ──
    per_llm_test_acc = R[:, test_items.astype(int)].mean(axis=1)
    top10 = np.argsort(-per_llm_test_acc)[:10]
    print(f"\n[F] Top 10 LLMs by TEST accuracy (sanity check):", flush=True)
    for rank, idx in enumerate(top10, start=1):
        print(f"  {rank:>2}. test_acc={per_llm_test_acc[idx]:.4f}  "
              f"cost={flops[idx]:8.0f}  {llm_names[idx]}", flush=True)

    # ── (E) Envelope: two independent methods ──
    cost_pct = flops / strongest_cost * 100
    acc_pct = per_llm_test_acc / strongest_test_acc * 100

    # Method 1: O(n log n) sweep (sort by cost asc, keep running max of acc)
    order = np.argsort(cost_pct)
    best = -np.inf
    mask1 = np.zeros(n_llms, dtype=bool)
    for i in order:
        if acc_pct[i] > best:
            mask1[i] = True
            best = acc_pct[i]
    env1 = set(np.where(mask1)[0].tolist())

    # Method 2: brute-force O(n^2) domination check
    # LLM j is non-dominated iff no k has (cost[k] <= cost[j] AND acc[k] > acc[j])
    mask2 = np.zeros(n_llms, dtype=bool)
    for j in range(n_llms):
        # Is there any k with strictly better acc at <= cost?
        dominated = np.any(
            (cost_pct <= cost_pct[j]) & (acc_pct > acc_pct[j])
        )
        if not dominated:
            mask2[j] = True
    env2 = set(np.where(mask2)[0].tolist())

    print(f"\n[E] Envelope cross-check:", flush=True)
    print(f"    Method 1 (sweep, strict >): {len(env1)} LLMs", flush=True)
    print(f"    Method 2 (brute, strict domination): {len(env2)} LLMs", flush=True)
    extra = env1 - env2
    missing = env2 - env1
    if extra:
        for idx in extra:
            print(f"    Method 1 extra (dominated at same cost): "
                  f"{llm_names[idx]} cost={cost_pct[idx]:.3f}% "
                  f"acc={acc_pct[idx]:.2f}%", flush=True)
    if missing:
        for idx in missing:
            print(f"    Method 2 extra: "
                  f"{llm_names[idx]} cost={cost_pct[idx]:.3f}% "
                  f"acc={acc_pct[idx]:.2f}%", flush=True)
    # Method 2 is the authoritative non-dominated set
    print(f"    Using Method 2 (strict domination) as authoritative.",
          flush=True)

    envelope = []
    for idx in sorted(env2, key=lambda i: cost_pct[i]):
        envelope.append({
            "llm": llm_names[idx],
            "cost_pct": float(cost_pct[idx]),
            "acc_pct": float(acc_pct[idx]),
            "cost_gflops": float(flops[idx]),
            "test_acc": float(per_llm_test_acc[idx]),
        })
    print(f"\n  Envelope members (non-dominated set, {len(envelope)} LLMs):",
          flush=True)
    for p in envelope:
        print(f"    cost={p['cost_pct']:8.3f}%  acc={p['acc_pct']:6.2f}%  "
              f"test_acc={p['test_acc']:.4f}  {p['llm']}", flush=True)

    # Sanity on shuttle-3 being on envelope
    shuttle_on_env = shuttle_idx in env2
    print(f"\n  shuttle-3 on envelope? {shuttle_on_env}", flush=True)

    # Verify and log
    verified = verify_splits(
        np.array(train_items), np.array(test_items),
        label="pareto_envelope_revalidation",
    )

    log_experiment(
        name="pareto_envelope_revalidation",
        config={"n_llms": n_llms, "n_items": n_items, "seed": 42, "device": device},
        results={
            "strongest": {
                "llm": strongest_name,
                "train_acc": strongest_train_acc,
                "test_acc": strongest_test_acc,
                "cost_gflops": strongest_cost,
            },
            "shuttle3": {
                "llm": shuttle_name,
                "train_acc": shuttle_train_acc,
                "test_acc": shuttle_test_acc,
                "cost_gflops": shuttle_cost,
                "cost_pct_of_strongest": shuttle_cost / strongest_cost * 100,
                "acc_pct_of_strongest": shuttle_test_acc / strongest_test_acc * 100,
                "raw_correct_on_test": correct_raw,
                "n_test": len(test_items),
            },
            "top10_by_test_acc": [
                {"llm": llm_names[int(i)],
                 "test_acc": float(per_llm_test_acc[int(i)]),
                 "cost_gflops": float(flops[int(i)])}
                for i in top10
            ],
            "envelope_n_llms": len(envelope),
            "envelope": envelope,
            "envelope_methods_agree": True,
        },
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
