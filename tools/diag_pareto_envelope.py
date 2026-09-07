"""Compute the single-LLM Pareto envelope on the v2 test split.

For each of the 3,811 LLMs in the pool, (cost, accuracy) is known directly.
The Pareto-optimal subset = LLMs not dominated by any other (no other LLM
has both lower-or-equal cost AND higher-or-equal accuracy, with at least one
strict). Compare CDMEval's router curve to this envelope at matching cost
levels.

Logs to experiment_log.json with split verification.
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

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    llm_names = json.load(open(data_dir / "response_matrix_v2_full_llms.json"))
    n_llms, n_items = R.shape

    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)

    flops_dict = compute_flops_cost(llm_names, seq_length=512)
    flops = np.array([flops_dict[n] for n in llm_names])

    strongest_idx = int(np.argmax(R[:, train_items.astype(int)].mean(axis=1)))
    strongest_test_acc = float(R[strongest_idx, test_items.astype(int)].mean())
    strongest_cost = float(flops[strongest_idx])

    # (cost_pct, acc_pct) for every LLM
    per_llm_acc = R[:, test_items.astype(int)].mean(axis=1)
    cost_pct = flops / strongest_cost * 100
    acc_pct = per_llm_acc / strongest_test_acc * 100
    print(f"  {n_llms} LLMs on {len(test_items)} test items", flush=True)
    print(f"  Strongest: {llm_names[strongest_idx]}  "
          f"(test_acc={strongest_test_acc:.4f}, cost={strongest_cost:.0f})",
          flush=True)

    # ── Pareto-optimal subset (non-dominated) ──
    # An LLM j is dominated if there exists k such that:
    #   cost[k] <= cost[j] AND acc[k] >= acc[j], with strict inequality somewhere.
    # Efficient sweep: sort by cost asc, keep a running max of acc.
    order = np.argsort(cost_pct)
    best_acc_so_far = -np.inf
    pareto_mask = np.zeros(n_llms, dtype=bool)
    for idx in order:
        if acc_pct[idx] > best_acc_so_far:
            pareto_mask[idx] = True
            best_acc_so_far = acc_pct[idx]

    pareto_idx = np.where(pareto_mask)[0]
    print(f"\n  Single-LLM Pareto envelope: {len(pareto_idx)} LLMs",
          flush=True)
    envelope = []
    # sort by cost
    for idx in sorted(pareto_idx, key=lambda i: cost_pct[i]):
        envelope.append({
            "llm": llm_names[idx],
            "cost_pct": float(cost_pct[idx]),
            "acc_pct": float(acc_pct[idx]),
            "cost_gflops": float(flops[idx]),
        })

    # Print some representative envelope points
    print("\n  Envelope samples (sorted by cost):")
    for p in envelope[::max(1, len(envelope) // 15)][:15]:
        print(f"    cost={p['cost_pct']:7.3f}%  acc={p['acc_pct']:6.2f}%  "
              f"{p['llm'][:55]}", flush=True)
    # Always show endpoints
    print(f"    -- max acc point --")
    best_acc = max(envelope, key=lambda p: p["acc_pct"])
    print(f"    cost={best_acc['cost_pct']:7.3f}%  acc={best_acc['acc_pct']:6.2f}%  "
          f"{best_acc['llm']}", flush=True)

    # ── Compare CDMEval at matching cost levels ──
    cdm_data = json.load(open("cdm_exploration/experiments/v2_pareto_routing.json"))
    cdm_points = cdm_data["cdm_normalized"]

    # Envelope interpolated at each CDM cost level
    env_costs = np.array([p["cost_pct"] for p in envelope])
    env_accs = np.array([p["acc_pct"] for p in envelope])

    def envelope_at(c):
        """Best single-LLM accuracy at cost <= c (step function)."""
        mask = env_costs <= c
        if mask.sum() == 0:
            return 0.0
        return float(env_accs[mask].max())

    print("\n  CDMEval vs single-LLM envelope (at matched cost):")
    print(f"  {'tau':>5}  {'cost%':>7}  {'CDM acc%':>9}  "
          f"{'env acc%':>9}  {'delta':>7}")
    comparisons = []
    for p in cdm_points:
        env_here = envelope_at(p["cost_pct"])
        delta = p["acc_pct"] - env_here
        comparisons.append({
            "t": p["t"], "cost_pct": p["cost_pct"],
            "cdm_acc_pct": p["acc_pct"],
            "envelope_acc_pct": env_here,
            "delta_pp": delta,
        })
        marker = " WIN" if delta > 0.5 else ("=EQ=" if abs(delta) <= 0.5 else "LOSE")
        print(f"  {p['t']:>5.2f}  {p['cost_pct']:7.2f}  "
              f"{p['acc_pct']:9.2f}  {env_here:9.2f}  "
              f"{delta:+7.2f}  {marker}",
              flush=True)

    # Summary counts
    n_win = sum(1 for c in comparisons if c["delta_pp"] > 0.5)
    n_eq = sum(1 for c in comparisons if abs(c["delta_pp"]) <= 0.5)
    n_lose = sum(1 for c in comparisons if c["delta_pp"] < -0.5)
    print(f"\n  Summary across {len(comparisons)} CDMEval operating points:")
    print(f"    CDMEval above envelope (>0.5pp): {n_win}", flush=True)
    print(f"    Within 0.5pp:                    {n_eq}", flush=True)
    print(f"    Below envelope (>0.5pp):         {n_lose}", flush=True)

    # Max delta
    best_c = max(comparisons, key=lambda c: c["delta_pp"])
    worst_c = min(comparisons, key=lambda c: c["delta_pp"])
    print(f"\n  Best CDM advantage: {best_c['delta_pp']:+.2f}pp at "
          f"t={best_c['t']:.2f} (cost={best_c['cost_pct']:.1f}%)", flush=True)
    print(f"  Worst:              {worst_c['delta_pp']:+.2f}pp at "
          f"t={worst_c['t']:.2f} (cost={worst_c['cost_pct']:.1f}%)", flush=True)

    # Verify and log
    verified = verify_splits(
        np.array(train_items), np.array(test_items),
        label="pareto_envelope_diagnostic",
    )

    log_experiment(
        name="pareto_envelope_diagnostic",
        config={"n_llms": n_llms, "n_items": n_items, "seed": 42, "device": device},
        results={
            "n_envelope_llms": int(len(envelope)),
            "envelope": envelope[:100],  # truncate for log size
            "comparisons": comparisons,
            "n_cdm_above_envelope": n_win,
            "n_cdm_equal_envelope": n_eq,
            "n_cdm_below_envelope": n_lose,
            "best_cdm_advantage_pp": best_c["delta_pp"],
            "worst_cdm_gap_pp": worst_c["delta_pp"],
        },
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=verified,
    )

    # Save full envelope for plotting later
    out = Path("cdm_exploration/experiments/v2_pareto_envelope.json")
    json.dump({
        "envelope": envelope,
        "comparisons": comparisons,
        "all_llms": [
            {"llm": llm_names[i],
             "cost_pct": float(cost_pct[i]),
             "acc_pct": float(acc_pct[i])}
            for i in range(n_llms)
        ],
    }, open(out, "w"), indent=2)
    print(f"\n  Saved full scatter + envelope to: {out}", flush=True)


if __name__ == "__main__":
    main()
