"""Recover full question text for the 9,523 v2 items from RouterBench prompts.

Background: the v2 pipeline stored only a 200-char `question_preview`, which
for few-shot-formatted items is the SHARED preamble, not the item's real
question. Both skill extraction and item embeddings were built on that
truncated text. The real question is the FINAL example in each RouterBench
prompt string (the one the model must answer).

This script:
  1. aligns each v2 item to its source prompt by matching the per-item
     correctness fingerprint (3,811-dim binary vector) -- content-based,
     not just order, and it cleanly handles the ~51 zero-solver items that
     were filtered out of v2;
  2. parses the real (final) question per benchmark format;
  3. validates and writes a NEW file (does not touch any existing artifact).

Read-only w.r.t. all existing files; output -> item_full_text_recovered.json
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

DATA = Path("cdm_exploration/data/cdm_ready")
RR = Path("cdm_exploration/data/routereval")
PROMPT_PKL = RR / "leaderboard_prompt" / "leaderboard_prompt" / "leaderboard_new_prompt.pkl"
SCORE_PKL = RR / "leaderboard_score" / "leaderboard_score" / "leaderboard_new.pkl"
OUT = DATA / "item_full_text_recovered.json"


def parse_question(prompt: str, benchmark: str) -> str:
    """Extract the final (real) question from a few-shot prompt string."""
    s = str(prompt)
    if benchmark == "BBH":
        seg = s.rsplit("Q:", 1)[-1]
        seg = seg.rsplit("A:", 1)[0]
    elif benchmark == "MATH":
        seg = s.rsplit("Problem:", 1)[-1]
        seg = seg.rsplit("Solution:", 1)[0]
    elif benchmark in ("GPQA", "MuSR"):
        # examples end with "Answer: <x>"; the real question is the text
        # after the last completed answer, before the trailing "Answer:".
        before_final = s.rsplit("Answer:", 1)[0]
        seg = before_final.split("Answer:")[-1]
        # strip a leading prev-answer token (e.g. " A", " 1") + separator
        seg = re.sub(r"^\s*[A-D0-9]+\s*", "", seg, count=1) if "Answer:" in s else seg
    else:  # IFEval -- zero-shot, the whole string is the instruction
        seg = s
    return seg.strip()


def main() -> int:
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    R = np.load(DATA / "response_matrix_v2_full.npy")  # (3811, 9523)
    prompts = pickle.load(open(PROMPT_PKL, "rb"))
    score = pickle.load(open(SCORE_PKL, "rb"))["data"]
    n_models = R.shape[0]
    print(f"items={len(items)}  R={R.shape}  prompt_keys={len(prompts)}", flush=True)

    # group our items by key, preserving order (verified contiguous)
    by_key: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_key.setdefault(it["key"], []).append(i)

    out = []
    report = {}
    for key, idxs in by_key.items():
        bench = items[idxs[0]]["benchmark"]
        src_corr = np.asarray(score[key]["correctness"])      # (n_src, 3811)
        src_prompts = prompts[key]                            # (n_src,)
        our_corr = R[:, idxs].T                               # (n_our, 3811)

        # drop zero-solver source items (these were filtered out of v2)
        src_keep = np.where(src_corr.sum(axis=1) > 0)[0]
        matched, ambiguous, unmatched = 0, 0, 0
        # align each of our items to a source row by exact fingerprint
        # (fall back to order among kept rows if fingerprints collide)
        src_fp = {}
        for r in src_keep:
            src_fp.setdefault(src_corr[r].tobytes(), []).append(r)
        used = set()
        order_kept = list(src_keep)
        for pos, i in enumerate(idxs):
            fp = our_corr[pos].astype(src_corr.dtype).tobytes()
            cand = [r for r in src_fp.get(fp, []) if r not in used]
            if len(cand) == 1:
                r = cand[0]; matched += 1
            elif len(cand) > 1:
                r = cand[0]; used.add(r); ambiguous += 1
            else:
                # fallback: positional among kept rows
                r = order_kept[pos] if pos < len(order_kept) else None
                unmatched += 1
            if r is not None:
                used.add(r)
                q = parse_question(src_prompts[r], bench)
            else:
                q = ""
            out.append({
                "item_idx": items[i]["item_idx"], "benchmark": bench,
                "subtask": items[i]["subtask"], "key": key,
                "question_full_text": q, "n_chars": len(q),
                "recovered": bool(q),
            })
        report[key] = {
            "bench": bench, "n_our": len(idxs), "n_src": len(src_prompts),
            "n_src_kept": len(src_keep), "matched": matched,
            "ambiguous": ambiguous, "fallback": unmatched,
        }

    out.sort(key=lambda d: d["item_idx"])
    # ---- validation summary ----
    n = len(out)
    rec = sum(d["recovered"] for d in out)
    lens = np.array([d["n_chars"] for d in out])
    longer = sum(1 for d in out if d["n_chars"] > 200)
    print("\n=== RECOVERY REPORT ===", flush=True)
    print(f"items: {n} | recovered non-empty: {rec} ({100*rec/n:.1f}%)")
    print(f"length: min {lens.min()} median {int(np.median(lens))} max {lens.max()} | >200 chars: {longer} ({100*longer/n:.1f}%)")
    tot = {k: sum(r[k] for r in report.values()) for k in ("matched", "ambiguous", "fallback")}
    print(f"alignment: exact-fingerprint {tot['matched']} | ambiguous {tot['ambiguous']} | order-fallback {tot['fallback']}")
    by_b = {}
    for d in out:
        by_b.setdefault(d["benchmark"], []).append(d["n_chars"])
    print("\nper-benchmark median recovered length:")
    for b, L in by_b.items():
        print(f"  {b:7s} n={len(L):5d} median={int(np.median(L)):5d} empty={sum(1 for x in L if x==0)}")

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
