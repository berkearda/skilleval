#!/usr/bin/env python3
"""Human evaluation of skill-question assignments: acceptance and inter-annotator agreement.

Three annotators judged the same (skill, question) pairs with yes/no: "is this skill needed to answer
this question?". This script reads the three returned sheets as they are, aligns them on
(skill_id, item_idx), and reports

  - acceptance per annotator (micro over pairs, macro over skills);
  - acceptance by majority vote of the three annotators (micro and macro);
  - share of pairs on which all three agree;
  - Cohen's kappa for each annotator pair and their mean;
  - Fleiss' kappa and Krippendorff's alpha (nominal) over the three annotators.

Output: cdm_exploration/experiments/v2_human_eval_agreement.json

    python3 tools/score_human_eval_agreement.py
"""
import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment, verify_splits

RAW = REPO / "cdm_exploration/experiments/human_eval/raw"
BLANK = REPO / "cdm_exploration/experiments/human_eval_skills.csv"
OUT = REPO / "cdm_exploration/experiments/v2_human_eval_agreement.json"
SHEETS = {"A": ("annotator_A.csv", "cp1252", ";"),
          "B": ("annotator_B.csv", "utf-8-sig", ","),
          "C": ("annotator_C.csv", "utf-8-sig", ",")}
YES = {"1", "y", "yes"}
NO = {"0", "n", "no", "mo", "noo"}          # "mo" is an observed typo for "no" in sheet A


def load(name, enc, delim):
    rows = list(csv.DictReader(open(RAW / name, encoding=enc, newline=""), delimiter=delim))
    out, unreadable = {}, []
    for r in rows:
        key = (int(r["skill_id"]), int(r["item_idx"]))
        v = (r.get("skill_needed_yes_no") or "").strip().lower()
        val = 1 if v in YES else 0 if v in NO else None
        if val is None:
            unreadable.append({"skill_id": key[0], "item_idx": key[1], "raw": v})
        assert key not in out, f"duplicate key {key} in {name}"
        out[key] = val
    return out, unreadable, {k: r["skill_name"] for r in rows for k in [int(r["skill_id"])]}


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe)


def fleiss_kappa(M):
    """M: (n_items, n_raters) of 0/1, no missing."""
    n, r = M.shape
    yes = M.sum(1)
    counts = np.column_stack([r - yes, yes]).astype(float)
    P_i = ((counts ** 2).sum(1) - r) / (r * (r - 1))
    p_j = counts.sum(0) / (n * r)
    return float((P_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum()))


def krippendorff_alpha_nominal(units):
    """units: list of lists with the available 0/1 values per unit (missing values left out)."""
    units = [u for u in units if len(u) >= 2]
    o = Counter()
    for u in units:
        m = len(u)
        for x, y in itertools.permutations(range(m), 2):
            o[(u[x], u[y])] += 1.0 / (m - 1)
    n_c = Counter()
    for (c, _), w in o.items():
        n_c[c] += w
    n = sum(n_c.values())
    d_o = sum(w for (c, k), w in o.items() if c != k)
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n - 1)
    return float(1 - d_o / d_e)


def main():
    seed_everything(42)
    ck = torch.load(REPO / "cdm_exploration/checkpoints/expanded/text_conditioned_protocolB.pt",
                    map_location="cpu", weights_only=False)
    ok = verify_splits(np.array(ck["train_items"]), np.array(ck["test_items"]), expected_seed=42,
                       label="human_eval_agreement")

    data, unreadable, names = {}, {}, {}
    for tag, (name, enc, delim) in SHEETS.items():
        data[tag], unreadable[tag], nm = load(name, enc, delim)
        names.update(nm)
    blank_keys = {(int(r["skill_id"]), int(r["item_idx"]))
                  for r in csv.DictReader(open(BLANK, encoding="utf-8-sig", newline=""))}
    keys = sorted(blank_keys)
    same_keys = all(set(d) == blank_keys for d in data.values())

    per_annotator = {}
    for tag, d in data.items():
        vals = [(k, v) for k, v in d.items() if v is not None]
        by_skill = defaultdict(list)
        for (s, _), v in vals:
            by_skill[s].append(v)
        per_annotator[tag] = {"n_judged": len(vals), "n_yes": int(sum(v for _, v in vals)),
                              "accept_micro": float(np.mean([v for _, v in vals])),
                              "accept_macro_per_skill": float(np.mean([np.mean(v) for v in by_skill.values()])),
                              "unreadable": unreadable[tag]}

    complete = [k for k in keys if all(data[t][k] is not None for t in SHEETS)]
    M = np.array([[data[t][k] for t in SHEETS] for k in complete])
    # majority vote over the available votes; pairs with a single vote or a 1-1 tie are left out
    maj, maj_by_skill = [], defaultdict(list)
    for k in keys:
        votes = [data[t][k] for t in SHEETS if data[t][k] is not None]
        if len(votes) >= 2 and sum(votes) * 2 != len(votes):
            m = int(sum(votes) * 2 > len(votes))
            maj.append(m)
            maj_by_skill[k[0]].append(m)
    pair_kappa = {f"{a}-{b}": cohen_kappa(*zip(*[(data[a][k], data[b][k]) for k in keys
                                                  if data[a][k] is not None and data[b][k] is not None]))
                  for a, b in itertools.combinations(SHEETS, 2)}
    per_skill_major = {s: float(np.mean(v)) for s, v in maj_by_skill.items()}
    res = {
        "experiment": "human_eval_agreement", "n_pairs": len(keys), "n_skills": len({k[0] for k in keys}),
        "n_annotators": len(SHEETS), "all_sheets_have_the_same_pairs_as_the_blank_sheet": bool(same_keys),
        "n_pairs_judged_by_all_three": len(complete),
        "per_annotator": per_annotator,
        "majority": {"n_pairs": len(maj), "n_accepted": int(sum(maj)), "accept_micro": float(np.mean(maj)),
                     "accept_macro_per_skill": float(np.mean(list(per_skill_major.values()))),
                     "n_skills_below_50pct": int(sum(v < 0.5 for v in per_skill_major.values()))},
        "all_three_agree_pct": float(100 * (M.min(1) == M.max(1)).mean()),
        "cohen_kappa_pairwise": pair_kappa, "cohen_kappa_mean": float(np.mean(list(pair_kappa.values()))),
        "fleiss_kappa": fleiss_kappa(M),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(
            [[data[t][k] for t in SHEETS if data[t][k] is not None] for k in keys]),
        "lowest_skills_by_majority": [{"skill_id": s, "name": names[s], "accept": round(v, 3)}
                                      for s, v in sorted(per_skill_major.items(), key=lambda x: x[1])[:5]],
        "verified": bool(ok and same_keys),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_annotator"}, indent=2))
    for t, v in per_annotator.items():
        print(f"  annotator {t}: judged {v['n_judged']}, accept micro {v['accept_micro']:.4f}, "
              f"macro {v['accept_macro_per_skill']:.4f}, unreadable {len(v['unreadable'])}")
    log_experiment(name="score_human_eval_agreement",
                   config={"seed": 42, "sheets": {t: s[0] for t, s in SHEETS.items()}},
                   results={k: res[k] for k in ("n_pairs", "n_pairs_judged_by_all_three", "majority",
                                                "all_three_agree_pct", "cohen_kappa_pairwise", "cohen_kappa_mean",
                                                "fleiss_kappa", "krippendorff_alpha_nominal")},
                   split_info={"n_train": len(ck["train_items"]), "n_test": len(ck["test_items"]), "random_state": 42},
                   verified=res["verified"])


if __name__ == "__main__":
    main()
