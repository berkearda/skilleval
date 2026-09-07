"""Diagnose v1 fixed CD-CAT: is the heuristic picking degenerate items?

Three checks:
  1. Skill coverage of heuristic vs random picks
  2. Item overlap between heuristic and random
  3. Inner-loop theta scale at end of selector run
  4. Top items picked by heuristic across multiple LLMs (do all LLMs pick the
     same item first?)

Reuses the heuristic from tools/run_adaptive_testing.py.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> None:
    from cdmeval.modeling.text_conditioned import TextConditionedNet
    from cdmeval.utils.device import resolve_device, seed_everything
    from cdmeval.utils.experiment import load_checkpoint
    from tools.run_adaptive_testing import (
        compute_fisher_information, fit_theta_single_step,
    )

    seed_everything(42)
    data_dir = REPO / "cdm_exploration" / "data" / "cdm_ready"
    device = resolve_device("cpu")

    R = np.load(data_dir / "response_matrix_v2_full.npy")
    q_matrix = np.load(data_dir / "qmatrix_v2_K100.npy")
    text_embs = np.load(data_dir / "item_text_embeddings_v2_full.npz")["embeddings"]
    n_llms, n_items = R.shape
    K = q_matrix.shape[1]

    train_items, test_items = train_test_split(np.arange(n_items), test_size=0.2,
                                                  random_state=42)
    _, test_llms = train_test_split(np.arange(n_llms), test_size=0.2, random_state=42)

    ckpt_path = REPO / "cdm_exploration" / "checkpoints" / "expanded" / "text_conditioned_protocolB.pt"
    net = TextConditionedNet(K, n_llms, 768)
    load_checkpoint(ckpt_path, net, device)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False

    # Reference theta scale
    with torch.no_grad():
        ref = net.student_emb.weight[:200].abs().mean().item()
    print(f"Trained-LLM |theta_raw| reference: {ref:.4f}")
    print()

    # Run heuristic selector for 5 LLMs at N=100 to get item lists
    N = 100
    n_diag = 5
    diag_llms = test_llms[:n_diag]

    heuristic_picks = []
    inner_theta_norms = []
    rng_random = np.random.RandomState(42)
    random_picks = [rng_random.choice(train_items, size=N, replace=False)
                    for _ in range(n_diag)]

    for li, llm_idx in enumerate(diag_llms):
        responses = R[llm_idx]
        theta_raw = nn.Parameter(torch.zeros(1, K, device=device))
        optimizer = torch.optim.Adam([theta_raw], lr=0.05)
        remaining_set = set(train_items.tolist())
        remaining_arr = train_items.copy()
        items_chosen = []

        for step in range(N):
            info = compute_fisher_information(
                net, theta_raw, remaining_arr, q_matrix, text_embs, device)
            best_local = int(np.argmax(info))
            best_item = int(remaining_arr[best_local])
            items_chosen.append(best_item)
            fit_theta_single_step(
                net, theta_raw, optimizer,
                best_item, responses[best_item],
                q_matrix, text_embs, device,
            )
            remaining_set.discard(best_item)
            remaining_arr = np.array(sorted(remaining_set))

        heuristic_picks.append(items_chosen)
        inner_norm = theta_raw.detach().abs().mean().item()
        inner_theta_norms.append(inner_norm)
        print(f"  LLM {li+1}/{n_diag}: |inner theta_raw|={inner_norm:.4f}  "
              f"(ref {ref:.4f})", flush=True)

    print()
    print("=" * 60)
    print("DIAGNOSTIC 1: Skill coverage")
    print("=" * 60)
    for li in range(n_diag):
        h_skills = q_matrix[heuristic_picks[li]].sum(axis=0) > 0
        r_skills = q_matrix[random_picks[li]].sum(axis=0) > 0
        print(f"  LLM {li+1}: heuristic covers {int(h_skills.sum())}/{K} skills, "
              f"random covers {int(r_skills.sum())}/{K} skills")

    print()
    print("=" * 60)
    print("DIAGNOSTIC 2: Mean skills/item picked")
    print("=" * 60)
    for li in range(n_diag):
        h_mean = q_matrix[heuristic_picks[li]].sum(axis=1).mean()
        r_mean = q_matrix[random_picks[li]].sum(axis=1).mean()
        print(f"  LLM {li+1}: heuristic {h_mean:.2f} skills/item, "
              f"random {r_mean:.2f}")

    print()
    print("=" * 60)
    print("DIAGNOSTIC 3: Item overlap heuristic vs random (per LLM)")
    print("=" * 60)
    for li in range(n_diag):
        ovr = len(set(heuristic_picks[li]) & set(int(x) for x in random_picks[li]))
        print(f"  LLM {li+1}: {ovr}/{N} overlap (random expectation ~{N*N/len(train_items):.1f})")

    print()
    print("=" * 60)
    print("DIAGNOSTIC 4: Are LLMs picking the same items?")
    print("=" * 60)
    flat = [it for lst in heuristic_picks for it in lst]
    counts = Counter(flat)
    top10 = counts.most_common(10)
    print(f"  Total slots: {n_diag*N}, unique items: {len(counts)}")
    print(f"  Top-10 items by repeat across LLMs:")
    for it, c in top10:
        skills = int(q_matrix[it].sum())
        print(f"    item={it:>5}  picked by {c}/{n_diag} LLMs  "
              f"({skills} skills active)")

    print()
    print("=" * 60)
    print("DIAGNOSTIC 5: Inner-theta scale summary")
    print("=" * 60)
    print(f"  mean |inner theta_raw|: {np.mean(inner_theta_norms):.4f}")
    print(f"  reference trained:      {ref:.4f}")
    print(f"  ratio: {np.mean(inner_theta_norms)/ref:.2%}")
    if np.mean(inner_theta_norms) < 0.3 * ref:
        print(f"  -> WARN: inner theta is undertrained, heuristic computes")
        print(f"     uncertainty at a wrong point (Hypothesis C).")

    print()
    print("=" * 60)
    print("DIAGNOSTIC 6: First 5 heuristic picks per LLM (do they agree?)")
    print("=" * 60)
    for li in range(n_diag):
        print(f"  LLM {li+1} first 5: {heuristic_picks[li][:5]}")


if __name__ == "__main__":
    main()
