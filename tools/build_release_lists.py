"""Build the two release artifacts.

  skill_list.csv  - the 100 skills, with the name as published, the relabeled
                    name, question count and per-benchmark composition
  llm_list.csv    - the 3,811 evaluated models

Both names are included on purpose. The paper's Q-matrix carries labels
produced by cdmeval/skills/clustering.py::label_clusters, which takes the most
frequent member skill phrase, title-cases it and truncates at 50 characters
(hence entries ending mid-word). The relabeled column is the July naming pass
with gpt-4o-mini over the SAME frozen clusters, and is the version the human
evaluation scored. Memberships are identical in both columns.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration" / "data" / "cdm_ready"
E = REPO / "cdm_exploration" / "experiments"
OUT = REPO / "release"
BENCHES = ["MATH", "BBH", "GPQA", "MuSR", "IFEval"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    q = np.load(D / "qmatrix_v2_K100.npy")
    lab = json.loads((D / "cluster_labels_v2_K100.json").read_text())
    items = json.loads((D / "response_matrix_v2_full_items.json").read_text())
    bench_of = {it["item_idx"]: it["benchmark"] for it in items}
    tax = json.loads((E / "oldtax_final.json").read_text())
    entries = tax["rows"] + tax["small_rows"]
    relabel = {e["old"]: e["name"] for e in entries}

    rows = []
    for k in range(q.shape[1]):
        members = np.nonzero(q[:, k])[0]
        c = Counter(bench_of[int(i)] for i in members)
        published = lab[str(k)]
        rows.append({
            "skill_id": k,
            "name_as_published": published,
            "name_relabeled": relabel.get(published, ""),
            "n_questions": len(members),
            **{f"n_{b}": c.get(b, 0) for b in BENCHES},
        })

    f = OUT / "skill_list.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    missing = sum(1 for r in rows if not r["name_relabeled"])
    print(f"wrote {f}  ({len(rows)} skills, {missing} without a relabeled name)")
    print(f"  question counts: min {min(r['n_questions'] for r in rows)}, "
          f"median {int(np.median([r['n_questions'] for r in rows]))}, "
          f"max {max(r['n_questions'] for r in rows)}, "
          f"total assignments {sum(r['n_questions'] for r in rows)}")

    llms = json.loads((D / "response_matrix_v2_full_llms.json").read_text())
    f2 = OUT / "llm_list.csv"
    with open(f2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "model"])
        for i, n in enumerate(llms):
            w.writerow([i, n])
    print(f"wrote {f2}  ({len(llms)} models)")


if __name__ == "__main__":
    main()
