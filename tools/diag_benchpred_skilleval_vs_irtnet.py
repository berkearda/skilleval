#!/usr/bin/env python3
"""Benchmark-level prediction on held-out items: SkillEval against IrtNet d=232 (note on Figure 6, 23 Sep 2026).

The submitted Figure 6 compared SkillEval with an IRT 2PL on the item-wise (OOD) split, but the 2PL has no trained
parameters for held-out items (all 1,905 keep their initial values), so that series is not a 2PL prediction. IrtNet, like
SkillEval, derives item parameters from the item text and is a genuine OOD comparator (Table 2).

For each LLM and benchmark: predicted accuracy = mean predicted P(correct) over the benchmark's held-out items; true
accuracy = the LLM's observed accuracy on them. Pearson r across the 3,811 LLMs per benchmark, and "all benchmarks" =
mean over all 1,905 held-out items per LLM (the definition of the registered benchpred.overall_r = 0.9889).

Models: SkillEval main model (text_conditioned_protocolB.pt); IrtNet d=232 seed 42 (checkpoints/irtnet/d232_s42.pt,
the Table 2 size; the only IrtNet d=232 checkpoint kept locally). Checks: both models reproduce their recorded test AUC
(0.7170 and 0.7486), and SkillEval reproduces the registered overall r 0.9889.

Output: cdm_exploration/experiments/v2_benchpred_skilleval_vs_irtnet.json (+ .npz with per-LLM values for the figure)

    python3 tools/diag_benchpred_skilleval_vs_irtnet.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.modeling.text_conditioned import TextConditionedNet
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import load_checkpoint, log_experiment, verify_splits
from tools.run_irtnet_headtohead import import_irtnet_model_class

DATA = REPO / "cdm_exploration/data/cdm_ready"
EXP = REPO / "cdm_exploration/experiments"
BENCHMARKS = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]
EXPECTED = {"skilleval_test_auc": 0.7170448, "irtnet_test_auc": 0.7485832343376343, "skilleval_overall_r": 0.9889}


def summarise(pred, R_te, bench_te):
    out = {}
    for b in BENCHMARKS:
        m = bench_te == b
        out[b] = float(pearsonr(R_te[:, m].mean(1), pred[:, m].mean(1))[0])
    out["all"] = float(pearsonr(R_te.mean(1), pred.mean(1))[0])
    return out


def main():
    seed_everything(42)
    R = np.load(DATA / "response_matrix_v2_full.npy")
    q = np.load(DATA / "qmatrix_v2_K100.npy")
    emb = np.load(DATA / "item_text_embeddings_v2_full.npz")["embeddings"]
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    bench = np.array([it.get("benchmark") for it in items])
    n_llms, n_items = R.shape
    train_items, test_items = train_test_split(np.arange(n_items), test_size=0.2, random_state=42)
    ok = verify_splits(train_items, test_items, expected_seed=42, label="benchpred_skilleval_vs_irtnet")
    R_te, bench_te = R[:, test_items], bench[test_items]

    # SkillEval main model
    t0 = time.time()
    net = TextConditionedNet(q.shape[1], n_llms, emb.shape[1])
    load_checkpoint(REPO / "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt", net, "cpu")
    net.eval()
    te_emb = torch.tensor(emb[test_items], dtype=torch.float32)
    te_q = torch.tensor(q[test_items], dtype=torch.float32)
    p_skill = np.zeros((n_llms, len(test_items)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_llms, 16):
            ids = torch.arange(s, min(s + 16, n_llms))
            c = len(ids)
            out = net(ids.repeat_interleave(len(test_items)), te_emb.repeat(c, 1), te_q.repeat(c, 1))
            p_skill[s:s + c] = out.view(c, -1).numpy()
    print(f"SkillEval predictions in {time.time() - t0:.0f}s", flush=True)

    # IrtNet d=232 seed 42, loaded as in tools/run_irtnet_benchmark_prediction.py
    t0 = time.time()
    MoE = import_irtnet_model_class()
    ck = torch.load(REPO / "cdm_exploration/checkpoints/irtnet/d232_s42.pt", map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    hp = {"model_embed_dim": cfg.get("model_embed_dim", 232), "num_experts": cfg.get("num_experts", 39),
          "top_k_experts": cfg.get("top_k_experts", 39), "expert_hidden_dim": cfg.get("expert_hidden_dim", 512),
          "shared_expert_hidden_dim": cfg.get("shared_expert_hidden_dim", 512),
          "expert_output_dim": cfg.get("expert_output_dim", 256), "dropout_rate": cfg.get("dropout_rate", 0.5),
          "embedding_noise": cfg.get("embedding_noise", 0.05)}
    irt = MoE(num_models=n_llms, num_prompts=n_items, prompt_embeddings=torch.tensor(emb, dtype=torch.float32), **hp)
    irt.load_state_dict(ck["model_state_dict"])
    irt.eval()
    p_irt = np.zeros((n_llms, len(test_items)), dtype=np.float32)
    all_llms = torch.arange(n_llms)
    with torch.no_grad():
        for j, it in enumerate(test_items):
            # IrtNet returns logits; tools/run_irtnet_headtohead.py applies the sigmoid before scoring, and so must we.
            # (tools/run_irtnet_benchmark_prediction.py averages the raw logits: its r values are not accuracy-level.)
            p_irt[:, j] = torch.sigmoid(irt(all_llms, torch.full((n_llms,), int(it), dtype=torch.long))).numpy()
    print(f"IrtNet d=232 predictions in {time.time() - t0:.0f}s", flush=True)

    assert 0 <= p_skill.min() and p_skill.max() <= 1 and 0 <= p_irt.min() and p_irt.max() <= 1, "predictions must be probabilities"
    y = R_te.ravel()
    res = {"experiment": "benchpred_skilleval_vs_irtnet", "split": "item-wise 80/20, random_state 42",
           "n_llms": n_llms, "n_test_items": int(len(test_items)),
           "n_test_items_per_benchmark": {b: int((bench_te == b).sum()) for b in BENCHMARKS},
           "skilleval": {"checkpoint": "expanded/text_conditioned_protocolB.pt",
                         "test_auc": float(roc_auc_score(y, p_skill.ravel())), "r": summarise(p_skill, R_te, bench_te)},
           "irtnet_d232_s42": {"checkpoint": "irtnet/d232_s42.pt",
                               "test_auc": float(roc_auc_score(y, p_irt.ravel())), "r": summarise(p_irt, R_te, bench_te)},
           "expected": EXPECTED}
    checks = {"skilleval_auc": abs(res["skilleval"]["test_auc"] - EXPECTED["skilleval_test_auc"]) < 5e-4,
              "irtnet_auc": abs(res["irtnet_d232_s42"]["test_auc"] - EXPECTED["irtnet_test_auc"]) < 5e-4,
              "skilleval_overall_r": abs(res["skilleval"]["r"]["all"] - EXPECTED["skilleval_overall_r"]) < 5e-4}
    res["checks"] = checks
    res["verified"] = bool(ok and all(checks.values()))
    print(json.dumps({k: res[k] for k in ("skilleval", "irtnet_d232_s42", "checks")}, indent=1))
    (EXP / "v2_benchpred_skilleval_vs_irtnet.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(EXP / "v2_benchpred_skilleval_vs_irtnet.npz", test_items=test_items, bench_te=bench_te,
                        true=R_te.astype(np.int8),
                        pred_skilleval=p_skill, pred_irtnet=p_irt)
    log_experiment(name="diag_benchpred_skilleval_vs_irtnet", config={"seed": 42, "irtnet": "d232_s42"},
                   results={k: res[k] for k in ("skilleval", "irtnet_d232_s42", "checks")},
                   split_info={"n_train_items": len(train_items), "n_test_items": len(test_items), "random_state": 42},
                   verified=res["verified"])
    if not res["verified"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
