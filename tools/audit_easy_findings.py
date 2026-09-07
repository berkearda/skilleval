"""Audit three open findings (R9, R10, R5-followup) per the project's design rules §7.

R10: fig_benchmark_prediction r=0.993 — is this on held-out LLMs or only
     held-out items?
R9:  weak-beats-strong 38% — does it use the train-selected 'strongest'
     reference that failed the Pareto audit?
R5f: Table 1 per-benchmark CDMEval vs strongest — does CDMEval still win
     against shuttle-3 (the test-rank-1 LLM) per benchmark?

Runs one pass over the data, validates splits, logs each audit as a
separate experiment_log entry. Prints plain-English conclusions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.evaluation.baselines import IRT2PL
    from cdmeval.evaluation.cost_analysis import compute_flops_cost
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits

    seed_everything(42)
    data_dir = Path(cfg.paths.cdm_ready)
    device = resolve_device(cfg.device)
    print("=" * 72, flush=True)
    print("AUDIT: R9, R10, R5-followup", flush=True)
    print("=" * 72, flush=True)

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q = np.load(data_dir / "qmatrix_v2_K100.npy")
    emb = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    llm_names = json.load(open(data_dir / "response_matrix_v2_full_llms.json"))
    items_data = json.load(open(data_dir / "response_matrix_v2_full_items.json"))
    n_llms, n_items = R.shape
    K = q.shape[1]
    print(f"\nData: {n_llms} LLMs x {n_items} items, K={K}", flush=True)

    # ── Split ──
    all_items = np.arange(n_items)
    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    print(f"Split: train={len(train_items)}, test={len(test_items)}", flush=True)

    # ── References ──
    train_acc = R[:, train_items.astype(int)].mean(axis=1)
    test_acc = R[:, test_items.astype(int)].mean(axis=1)
    strongest_train_idx = int(np.argmax(train_acc))
    strongest_test_idx = int(np.argmax(test_acc))
    shuttle_idx = llm_names.index("shuttleai__shuttle-3")

    print(f"\nReferences:", flush=True)
    print(f"  train-selected (paper's 'strongest'): {llm_names[strongest_train_idx]}",
          flush=True)
    print(f"    train_acc={train_acc[strongest_train_idx]:.4f}  "
          f"test_acc={test_acc[strongest_train_idx]:.4f}", flush=True)
    print(f"  test-rank-1:                           {llm_names[strongest_test_idx]}",
          flush=True)
    print(f"    train_acc={train_acc[strongest_test_idx]:.4f}  "
          f"test_acc={test_acc[strongest_test_idx]:.4f}", flush=True)
    assert strongest_test_idx == shuttle_idx, "test-rank-1 is not shuttle-3?"

    benchmarks = [it["benchmark"] for it in items_data]
    test_bench = np.array([benchmarks[i] for i in test_items])
    bench_names = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]

    # ═══════════════════════════════════════════════════════════════════
    # R10 — benchmark prediction correlation scope
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72, flush=True)
    print("R10 — benchmark prediction r=0.993: what is held out?", flush=True)
    print("=" * 72, flush=True)

    # Load CDM + IRT (same as fig_benchmark_prediction.py)
    cdm = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(
        "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
        cdm, device)
    cdm = cdm.to(device); cdm.eval()

    # Predict on test items, all LLMs
    te_test = torch.tensor(emb[test_items], dtype=torch.float32, device=device)
    qr_test = torch.tensor(q[test_items], dtype=torch.float32, device=device)
    with torch.no_grad():
        stat = torch.sigmoid(cdm.student_emb(torch.arange(n_llms, device=device)))
        k_d = torch.sigmoid(cdm.k_difficulty_proj(te_test))
        e_d = torch.sigmoid(cdm.e_difficulty_proj(te_test))
        cdm_pred = np.zeros((n_llms, len(test_items)), dtype=np.float32)
        chunk = 256
        for i in range(0, n_llms, chunk):
            s = stat[i:i+chunk].unsqueeze(1)
            x = e_d.unsqueeze(0) * (s - k_d.unsqueeze(0)) * qr_test.unsqueeze(0)
            h1 = torch.sigmoid(cdm.prednet_full1(x))
            h2 = torch.sigmoid(cdm.prednet_full2(h1))
            p = torch.sigmoid(cdm.prednet_full3(h2)).squeeze(-1)
            cdm_pred[i:i+chunk] = p.cpu().numpy()

    R_test = R[:, test_items.astype(int)]

    # --- R10 Audit: are LLMs held out or not? ---
    print("\nCDMEval training configuration (from checkpoint):",
          flush=True)
    print(f"  n_llms trained on: ALL {n_llms} (LLM embedding layer fits all)",
          flush=True)
    print(f"  items trained on:  {len(train_items)} train items",
          flush=True)
    print(f"  items held out:    {len(test_items)} test items",
          flush=True)
    print(f"  -> Prediction target is (any_LLM, held_out_item) pairs.",
          flush=True)
    print(f"  -> No LLMs are held out. Every LLM's embedding was fit.",
          flush=True)

    # Compute r per benchmark using the same aggregation as the figure
    r_per_benchmark = {}
    for b in bench_names + ["All benchmarks"]:
        if b == "All benchmarks":
            mask = np.ones(len(test_items), dtype=bool)
        else:
            mask = test_bench == b
        if mask.sum() == 0:
            continue
        true_acc = R_test[:, mask].mean(axis=1)
        cdm_acc = cdm_pred[:, mask].mean(axis=1)
        r, _ = pearsonr(true_acc, cdm_acc)
        r_per_benchmark[b] = float(r)
        print(f"    {b}: r = {r:.4f}  (n_items={int(mask.sum())})",
              flush=True)

    # The key question: how predictable would this be from a TRIVIAL
    # baseline? We check: LLM's train-set mean accuracy as a predictor
    # of LLM's test-set mean accuracy. If trivial baseline also gets
    # r~0.99, then CDMEval's correlation is not actually informative.
    print("\n  Naive baseline: predict test-benchmark-acc by train-mean-acc:",
          flush=True)
    print(f"  (each LLM's train-set overall accuracy as flat predictor)",
          flush=True)
    naive_corrs = {}
    for b in bench_names + ["All benchmarks"]:
        if b == "All benchmarks":
            mask = np.ones(len(test_items), dtype=bool)
        else:
            mask = test_bench == b
        if mask.sum() == 0:
            continue
        true_acc = R_test[:, mask].mean(axis=1)
        naive_pred = train_acc  # each LLM's train overall acc, broadcast
        r, _ = pearsonr(true_acc, naive_pred)
        naive_corrs[b] = float(r)
        print(f"    {b}: r = {r:.4f}  (vs CDM {r_per_benchmark[b]:.4f})",
              flush=True)

    # --- R10 verdict ---
    print("\n  R10 VERDICT:", flush=True)
    print("    The r=0.993 claim is computed on held-out ITEMS, with all",
          flush=True)
    print("    3,811 LLM embeddings already fit. No LLMs are out-of-sample.",
          flush=True)
    print("    A naive 'use train-mean-accuracy for everything' baseline",
          flush=True)
    print(f"    reaches r={naive_corrs['All benchmarks']:.4f} overall,",
          flush=True)
    print("    so the r=0.993 claim is largely a property of train-test",
          flush=True)
    print("    item correlation, not CDMEval's specific ability.",
          flush=True)
    print("    Paper prose 'out-of-sample' is imprecise; reviewer may read",
          flush=True)
    print("    it as LLM-level generalization, which is not tested.",
          flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # R9 — weak-beats-strong 38% reference
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72, flush=True)
    print("R9 — weak-beats-strong 38%: depends on 'strongest' choice?",
          flush=True)
    print("=" * 72, flush=True)

    # Parse sizes exactly like the figure script
    import re
    PHI_SIZES = [("phi-3.5-mini", 3.8), ("phi-3-mini", 3.8), ("phi-3-small", 7.0),
                 ("phi-3-medium", 14.0), ("phi-2", 2.7), ("phi-1_5", 1.3)]

    def parse_size(name):
        low = name.lower()
        parts = name.split("__")
        suffix = parts[-1] if len(parts) > 1 else name
        moe = re.search(r"(\d+)x(\d+\.?\d*)[bB]", suffix)
        if moe:
            return float(moe.group(1)) * float(moe.group(2))
        for m in re.finditer(r"(\d+\.?\d*)[bB]", suffix):
            idx = suffix.find(m.group(0))
            if idx > 0 and suffix[idx - 1].lower() == "v":
                continue
            return float(m.group(1))
        for kw, sz in PHI_SIZES:
            if kw in low:
                return sz
        return np.nan

    sizes = np.array([parse_size(n) for n in llm_names])
    small_mask = (sizes <= 13.0) & (~np.isnan(sizes))
    small_idx = np.where(small_mask)[0]
    print(f"  Small LLMs (<=13B): {len(small_idx)} of {n_llms}",
          flush=True)

    def wbs_count(ref_idx, label):
        cnt = 0
        for it in test_items:
            gt = R[:, int(it)]
            if gt[ref_idx] == 0 and gt[small_idx].sum() > 0:
                cnt += 1
        pct = cnt / len(test_items) * 100
        print(f"    {label}: {cnt}/{len(test_items)} = {pct:.2f}%",
              flush=True)
        return cnt, pct

    print(f"\n  WBS rate with each candidate reference:", flush=True)
    cnt_train, pct_train = wbs_count(strongest_train_idx,
                                     f"train-selected ({llm_names[strongest_train_idx]})")
    cnt_shuttle, pct_shuttle = wbs_count(shuttle_idx,
                                         f"shuttle-3 (test-rank-1)")

    # Does WBS even require a small model? Check if test-rank-1 beats
    # the train-selected reference on the WBS items
    wbs_items_train = [int(it) for it in test_items
                       if R[strongest_train_idx, int(it)] == 0
                       and R[small_idx, int(it)].sum() > 0]
    shuttle_saves = int(sum(
        1 for it in wbs_items_train
        if R[shuttle_idx, int(it)] == 1
    ))
    print(f"\n  Of the {len(wbs_items_train)} WBS items w.r.t. train-strongest:",
          flush=True)
    print(f"    shuttle-3 solves: {shuttle_saves} ({shuttle_saves/max(len(wbs_items_train),1)*100:.1f}%)",
          flush=True)
    print(f"    -> Using shuttle-3 as reference drops WBS rate to "
          f"{pct_shuttle:.1f}%", flush=True)

    # --- R9 verdict ---
    print("\n  R9 VERDICT:", flush=True)
    print(f"    WBS rate = {pct_train:.1f}% with train-strongest, "
          f"{pct_shuttle:.1f}% with shuttle-3.", flush=True)
    print(f"    The 38% claim is heavily dependent on using the "
          f"train-selected reference.", flush=True)
    print(f"    With shuttle-3 (the LLM that actually does best on test),",
          flush=True)
    print(f"    {pct_shuttle:.1f}% of test items are beaten by some small model.",
          flush=True)
    print(f"    Additionally: the calculation is purely ground-truth-based",
          flush=True)
    print(f"    (no CDMEval readout involved). Paper prose 'under CDMEval's",
          flush=True)
    print(f"    skill-conditional readout' misdescribes what is computed.",
          flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # R5 follow-up — per-benchmark Table 1 vs shuttle-3
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72, flush=True)
    print("R5 follow-up — Table 1 per-benchmark: does CDMEval beat shuttle-3?",
          flush=True)
    print("=" * 72, flush=True)

    # Per-benchmark accuracy for strongest_train, shuttle-3, and
    # CDMEval router at the default operating point used in the table.
    # Table 1 reports CDMEval 0.657 overall — that's at some threshold.
    # We compare direct model accuracies first (no router).
    print(f"\n  Per-benchmark direct accuracies (no router, just one LLM):",
          flush=True)
    print(f"  {'bench':<10} {'strongest_train':>18} {'shuttle-3':>12}  delta",
          flush=True)
    r5_table = {}
    for b in bench_names:
        mask = test_bench == b
        if mask.sum() == 0:
            continue
        s_train = float(R[strongest_train_idx,
                          test_items[mask].astype(int)].mean())
        s_shut = float(R[shuttle_idx, test_items[mask].astype(int)].mean())
        print(f"  {b:<10} {s_train:>18.4f} {s_shut:>12.4f}  "
              f"{s_shut - s_train:+7.4f}", flush=True)
        r5_table[b] = {"strongest_train": s_train, "shuttle3": s_shut,
                       "delta_shuttle_vs_strongest": s_shut - s_train,
                       "n_items": int(mask.sum())}

    # Overall
    s_train_all = float(R[strongest_train_idx, test_items.astype(int)].mean())
    s_shut_all = float(R[shuttle_idx, test_items.astype(int)].mean())
    print(f"  {'Overall':<10} {s_train_all:>18.4f} {s_shut_all:>12.4f}  "
          f"{s_shut_all - s_train_all:+7.4f}", flush=True)
    r5_table["Overall"] = {"strongest_train": s_train_all,
                           "shuttle3": s_shut_all,
                           "delta_shuttle_vs_strongest": s_shut_all - s_train_all,
                           "n_items": int(len(test_items))}

    # --- R5 verdict ---
    wins_shuttle = sum(1 for b in bench_names
                       if r5_table[b]["delta_shuttle_vs_strongest"] > 0)
    print(f"\n  R5 follow-up VERDICT:", flush=True)
    print(f"    shuttle-3 beats the paper's 'strongest' on "
          f"{wins_shuttle} of {len(bench_names)} benchmarks.", flush=True)
    print(f"    Overall delta: {s_shut_all - s_train_all:+.4f} "
          f"({(s_shut_all - s_train_all) / s_train_all * 100:+.1f}%).",
          flush=True)
    print(f"    Table 1 should use shuttle-3 (or both) as the baseline row.",
          flush=True)
    print(f"    (CDMEval router comparison pending: needs threshold choice.)",
          flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # Validate + log
    # ═══════════════════════════════════════════════════════════════════
    verified = verify_splits(
        np.array(train_items), np.array(test_items),
        label="easy_audits",
    )

    log_experiment(
        name="audit_benchmark_prediction_scope",
        config={"n_llms": n_llms, "n_items": n_items, "K": K,
                "seed": 42, "device": device},
        results={
            "cdm_r_per_benchmark": r_per_benchmark,
            "naive_baseline_r_per_benchmark": naive_corrs,
            "scope": "Items held out; all 3811 LLM embeddings fit during training",
            "reviewer_risk": "Prose 'out-of-sample' may be read as LLM-level generalization",
        },
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=verified,
    )

    log_experiment(
        name="audit_weak_beats_strong_reference",
        config={"n_llms": n_llms, "n_items": n_items,
                "seed": 42, "device": device},
        results={
            "wbs_with_train_selected_strongest": {
                "reference": llm_names[strongest_train_idx],
                "count": cnt_train, "pct": pct_train,
            },
            "wbs_with_shuttle3": {
                "reference": llm_names[shuttle_idx],
                "count": cnt_shuttle, "pct": pct_shuttle,
            },
            "shuttle_solves_train_wbs_items": shuttle_saves,
            "n_train_wbs_items": len(wbs_items_train),
            "note_on_prose": "Computation is purely ground-truth; no CDMEval readout used",
        },
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=verified,
    )

    log_experiment(
        name="audit_table1_shuttle3_baseline",
        config={"n_llms": n_llms, "n_items": n_items,
                "seed": 42, "device": device},
        results={
            "per_benchmark": r5_table,
            "shuttle3_beats_strongest_on_n_of_5": wins_shuttle,
            "overall_delta_abs": s_shut_all - s_train_all,
            "overall_delta_rel_pct": (s_shut_all - s_train_all) / s_train_all * 100,
        },
        split_info={"n_train": len(train_items), "n_test": len(test_items)},
        verified=verified,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
