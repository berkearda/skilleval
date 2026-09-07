"""Print human-readable cards for the new full-text K=100 taxonomy.

No scores -- just label, size, benchmark/subtask breakdown, the actual skill
phrases inside, and a couple of real questions. For reading by eye.

Usage:
    python tools/show_clusters.py            # biggest + cross-benchmark + small
    python tools/show_clusters.py 12 47 5    # specific cluster ids
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("cdm_exploration/data/cdm_ready")


def preview(text: str, limit: int = 800) -> str:
    """Readable single-line text; for long items keep head + tail so the
    actual question (often at the end, e.g. MuSR) stays visible."""
    t = " ".join(text.split())
    if len(t) <= limit:
        return t
    return t[: limit * 2 // 3] + "   […]   " + t[-limit // 3:]


def main() -> int:
    Q = np.load(DATA / "qmatrix_v2_K100_fulltext.npy")
    labels = json.load(open(DATA / "cluster_labels_v2_K100_fulltext.json"))
    items = json.load(open(DATA / "response_matrix_v2_full_items.json"))
    skills = {r["item_idx"]: r for r in
              json.load(open(DATA / "skills_extracted_v2_fulltext.json"))}
    full = {r["item_idx"]: r["question_full_text"] for r in
            json.load(open(DATA / "item_full_text_recovered.json"))}
    sizes = Q.sum(0)

    def phrase_to_cluster_members(c):
        rows = np.where(Q[:, c] == 1)[0]
        return [items[i]["item_idx"] for i in rows]

    def card(c):
        member_idx = phrase_to_cluster_members(c)
        benches = Counter(items[i]["benchmark"] for i in member_idx)
        subs = Counter(items[i]["subtask"] for i in member_idx)
        phrases = Counter(s.lower() for i in member_idx
                          for s in (skills[i].get("skills") or []))
        n = len(member_idx)
        print(f"\n{'='*78}")
        print(f"Cluster {c}  —  \"{labels[str(c)]}\"   (size: {n})")
        bs = ", ".join(f"{b} {100*v//n}%" for b, v in benches.most_common())
        print(f"  benchmarks: {bs}")
        print("  benchmark subtasks (original dataset categories): "
              + ", ".join(f"{s}:{v}" for s, v in subs.most_common(5)))
        print("  top extracted skill-labels (LLM-generated for these items, with counts):")
        for ph, v in phrases.most_common(6):
            print(f"     {v:4d}  {ph}")
        print("  example questions (one representative item per top subtask):")
        by_sub = {}
        for i in member_idx:
            by_sub.setdefault(items[i]["subtask"], []).append(i)
        for sub, _ in subs.most_common(3):
            i = by_sub[sub][len(by_sub[sub]) // 2]  # middle member of that subtask
            print(f"     [{sub}]")
            for line in textwrap.wrap(preview(full.get(i, "")), width=92):
                print(f"        {line}")

    if len(sys.argv) > 1:
        for c in map(int, sys.argv[1:]):
            card(c)
        return 0

    order = np.argsort(sizes)[::-1]
    # cross-benchmark clusters: span >1 benchmark meaningfully
    def n_benches(c):
        idx = phrase_to_cluster_members(c)
        b = Counter(items[i]["benchmark"] for i in idx)
        return sum(1 for _, v in b.items() if v >= max(3, 0.1 * len(idx)))
    xbench = [c for c in order if n_benches(c) > 1][:3]
    small = [c for c in order if sizes[c] > 0][-3:]

    print("########## 3 LARGEST ##########")
    for c in order[:3]:
        card(c)
    print("\n\n########## 3 CROSS-BENCHMARK ##########")
    for c in xbench:
        card(c)
    print("\n\n########## 3 SMALLEST ##########")
    for c in small:
        card(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
