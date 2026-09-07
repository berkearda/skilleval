"""Human-evaluation CSV: 100 skills (new names) x up to 10 questions.

Sampling is UNIFORM RANDOM with seed 42 per cluster (not curated, not
fit-ranked) so the human evaluation is unbiased and directly comparable to
the LLM name-fit number (0.83), which was also measured on sampled own
questions. Clusters with fewer than 10 questions contribute all of them.

Memberships are the FROZEN originals: Q-matrix column nonzeros keyed by
cluster_labels_v2_K100.json, identical to tools/relabel_original.py::load().
New names from oldtax_final.json (rows + small_rows, matched on old name).

Outputs:
  cdm_exploration/experiments/human_eval_skills.csv
  cdm_exploration/experiments/human_eval_skills_manifest.json
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"

SEED = 42
N_PER_SKILL = 10
MAX_CHARS = 700


def main() -> None:
    q = np.load(D / "qmatrix_v2_K100.npy")
    lab = json.load(open(D / "cluster_labels_v2_K100.json"))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    meta = {pi["item_idx"]: (pi["benchmark"], pi.get("subtask") or "")
            for pi in json.load(open(E / "oldtax_repaired_FINAL.json"))["per_item"]}
    clusters = {}
    for k in range(q.shape[1]):
        clusters[lab[str(k)]] = sorted(np.nonzero(q[:, k])[0].tolist())

    tax = json.load(open(E / "oldtax_final.json"))
    entries = tax["rows"] + tax["small_rows"]
    assert len(entries) == 100, f"expected 100 skills, got {len(entries)}"
    names = [e["name"] for e in entries]
    assert len(set(names)) == 100, "names not distinct"

    rows_out = []
    n_truncated = 0
    n_missing_text = 0
    for sid, e in enumerate(entries, start=1):
        old = e["old"]
        if old not in clusters:
            raise KeyError(f"old name not found in cluster labels: {old!r}")
        items = clusters[old]
        if len(items) != e["n"]:
            print(f"WARN skill {sid}: taxonomy n={e['n']} vs members {len(items)}")
        rng = random.Random(SEED + sid)
        sample = sorted(rng.sample(items, min(N_PER_SKILL, len(items))))
        for i in sample:
            text = qtext.get(i, "")
            if not text:
                n_missing_text += 1
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS].rstrip() + " [...]"
                n_truncated += 1
            b, st = meta.get(i, ("?", "?"))
            rows_out.append({
                # annotation columns — EXACTLY what the LLM judge saw
                # (final_leg.py L307: item text + skill name, nothing else),
                # plus neutral identifiers for joining. benchmark/subtask/n
                # are hints and live in the key file only.
                "skill_id": sid,
                "item_idx": i,
                "skill_name": e["name"],
                "question_text": text,
                "skill_needed_yes_no": "",
                "notes": "",
                "_benchmark": b,
                "_subtask": st,
                "_n_in_skill": len(items),
            })

    ann_fields = ["skill_id", "item_idx", "skill_name", "question_text",
                  "skill_needed_yes_no", "notes"]
    key_fields = ["skill_id", "item_idx", "_benchmark", "_subtask", "_n_in_skill"]

    out_csv = E / "human_eval_skills.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ann_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    with open(E / "human_eval_skills_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    manifest = {
        "artifact": "human_eval_skills_csv",
        "date": "2026-07-26",
        "seed": SEED,
        "sampling": "uniform random per cluster, seed 42+skill_id, no curation",
        "n_skills": 100,
        "n_rows": len(rows_out),
        "n_truncated_texts": n_truncated,
        "n_missing_texts": n_missing_text,
        "max_chars": MAX_CHARS,
        "judging_question": ("Does solving this question genuinely require the "
                             "named skill? Answer yes or no in skill_needed_yes_no."),
        "comparable_llm_number": "wmean_all100=0.834 (gpt-4o-mini), wmean_big~0.86 (gpt-4.1-mini)",
        "sources": ["qmatrix_v2_K100.npy", "cluster_labels_v2_K100.json",
                    "item_full_text_recovered.json", "oldtax_repaired_FINAL.json",
                    "oldtax_final.json"],
    }
    (E / "human_eval_skills_manifest.json").write_text(json.dumps(manifest, indent=2))

    from collections import Counter
    per_skill = Counter(r["skill_id"] for r in rows_out)
    print(f"skills: {len(per_skill)}, rows: {len(rows_out)}")
    print(f"skills with 10 questions: {sum(1 for v in per_skill.values() if v==10)}, "
          f"fewer: {sum(1 for v in per_skill.values() if v<10)}")
    print(f"truncated texts: {n_truncated}, missing texts: {n_missing_text}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
