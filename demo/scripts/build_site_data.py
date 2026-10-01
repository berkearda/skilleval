#!/usr/bin/env python3
"""Rebuild the site's example items and the data behind its weak-beats-strong chart.

Run from the repository root, after `python tools/download_data.py`:

    python demo/scripts/build_site_data.py

1. Example items (public/data/skills.json, field example_items). The texts in
   response_matrix_v2_full_items.json are 200-character previews, and for BBH they often show a
   few-shot example from the prompt rather than the item itself. Each item's own question is taken
   from the RouterEval prompt instead: the text after the last "Problem:" for MATH, the text after
   the last "Q:" for BBH, the instruction for IFEval, and the question with the opening of the story
   for MuSR. GPQA items are never shown (the GPQA authors ask that its questions not be posted in
   plain text), nor is any item that shares a 30-character passage with a GPQA prompt; MATH items with
   Asymptote drawings are skipped. Up to three items per skill.

   Each skill also gets items_by_benchmark: how many of the items needing it come from each benchmark.

3. Home page (public/data/home.json): the 12 models with the highest mean mastery, the pair the
   Compare link opens (that model and the best one with at most 13B parameters) and the 14 skills of the
   grid picture, read from the site's own theta_matrix.json and skills.json. It spares the Home page the
   full 3.6 MB matrix.

2. Weak-beats-strong (public/data/weak_beats_strong.json): the held-out items that some model with
   at most 13B parameters answers while the strongest single model fails, overall and per skill.
   Same split, strongest model, size rule and 10-item minimum as the paper's figure
   (tools/fig_weak_beats_strong_v2.py).
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_response_matrix_v2 import math_question, question_preview  # noqa: E402

CDM = ROOT / "cdm_exploration/data/cdm_ready"
ROUTEREVAL = ROOT / "cdm_exploration/data/routereval"
SITE = ROOT / "demo/public/data"
PER_SKILL = 3
MAX_CHARS = 900


def load_prompts(items):
    """The raw prompt of every item, in the order and with the filter of build_response_matrix_v2.py."""
    with open(ROUTEREVAL / "leaderboard_score/leaderboard_score/leaderboard_new.pkl", "rb") as f:
        lb = pickle.load(f)
    with open(ROUTEREVAL / "leaderboard_prompt/leaderboard_prompt/leaderboard_new_prompt.pkl", "rb") as f:
        prompts = pickle.load(f)
    keys = sorted(k for k in lb["data"] if k.startswith("math_"))
    keys += sorted(k for k in lb["data"] if k.startswith("bbh_"))
    keys += sorted(k for k in lb["data"] if k not in keys)
    raw, blocks = [], []
    for k in keys:
        correctness = lb["data"][k]["correctness"]
        texts = [str(p) for p in prompts.get(k, [])]
        raw += [texts[i] if i < len(texts) else "" for i in range(correctness.shape[0])]
        blocks.append(correctness)
    keep = np.vstack(blocks).T.sum(axis=0) > 0
    raw = [r for r, k in zip(raw, keep) if k]
    assert len(raw) == len(items)
    # the previews must come out exactly as in the items file, or the prompts are misaligned
    assert all(question_preview(r, it["benchmark"]) == it["question_preview"] for r, it in zip(raw, items))
    return raw


def shorten(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + " …"


def item_text(raw, benchmark):
    """The item's own question, or None if it should not be shown."""
    if benchmark == "MATH":
        q = math_question(raw)
        if not q or "[asy]" in q:
            return None
    elif benchmark == "BBH":
        start = raw.rfind("Q: ")
        if start == -1:
            return None
        q = raw[start + 3:].rstrip()
        q = re.sub(r"\nA:\s*$", "", q).rstrip()
        if "\nA:" in q:
            return None
    elif benchmark == "IFEval":
        q = raw.strip()
    elif benchmark == "MuSR":
        text = re.sub(r"\n*Answer:\s*$", "", raw.rstrip())
        choices = text.rfind("\n\n1 - ")
        if choices == -1:
            return None
        start = text.rfind("\n\n", 0, choices) + 2
        story, ask = text[:start].strip(), text[start:].strip()
        q = shorten(story, 320) + "\n\n" + ask
        return q if len(q) <= MAX_CHARS + 200 else None
    else:
        return None  # GPQA: never shown
    q = re.sub(r"\n{3,}", "\n\n", q.strip())
    return shorten(q, MAX_CHARS) if q else None


def gpqa_windows(raw, items, width=30):
    """30-character passages of the GPQA prompts, every 10 characters, mostly letters."""
    out = set()
    for r, it in zip(raw, items):
        if it["benchmark"] != "GPQA":
            continue
        t = re.sub(r"\s+", " ", r).strip()
        for i in range(0, max(1, len(t) - width + 1), 10):
            w = t[i:i + width]
            if len(w) == width and sum(c.isalpha() for c in w) >= 18:
                out.add(w)
    return out


def shares_gpqa_text(text, windows, width=30):
    t = re.sub(r"\s+", " ", text).strip()
    return any(t[i:i + width] in windows for i in range(len(t) - width + 1))


# --- weak beats strong: identical rules to tools/fig_weak_beats_strong_v2.py ---
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


def weak_beats_strong(R, Q, llms):
    sizes = np.array([parse_size(n) for n in llms])
    train, test = train_test_split(np.arange(R.shape[1]), test_size=0.2, random_state=42)
    strongest = int(np.argmax(R[:, train].mean(axis=1)))
    small = np.where((sizes <= 13.0) & ~np.isnan(sizes))[0]
    hit = (R[strongest] == 0) & (R[small].max(axis=0) > 0)
    per_skill = []
    for k in range(Q.shape[1]):
        items = test[Q[test, k] > 0]
        if len(items) < 10:
            continue
        w = int(hit[items].sum())
        per_skill.append({"id": k, "n": int(len(items)), "wbs": w, "pct": round(100 * w / len(items), 2)})
    per_skill.sort(key=lambda s: -s["pct"])
    return {
        "rule": "held-out items answered correctly by at least one model with at most 13B parameters "
                "while the strongest single model (highest training accuracy) fails",
        "n_test_items": int(len(test)),
        "n_items": int(hit[test].sum()),
        "pct": round(100 * float(hit[test].mean()), 2),
        "strongest_model": llms[strongest],
        "n_small_models": int(len(small)),
        "min_test_items_per_skill": 10,
        "skills": per_skill,
    }


def home_data(models, skills):
    """What the Home page shows, computed exactly as HomePage.tsx used to compute it."""
    means = [sum(m["theta"]) / len(m["theta"]) for m in models]
    order = sorted(range(len(models)), key=lambda i: -means[i])  # stable, like Array.prototype.sort
    keys = ("id", "name", "family", "tier", "params", "accuracy", "theta")
    top = [dict({k: models[i][k] for k in keys}, meanTheta=means[i]) for i in order[:12]]
    best_small = None
    for m, mu in zip(models, means):
        if m["params"] is None or m["params"] > 13:
            continue
        if best_small is None or mu > best_small[1]:
            best_small = (m, mu)
    compare = None
    if best_small and best_small[0]["id"] != top[0]["id"]:
        compare = f'{top[0]["id"]},{best_small[0]["id"]}'
    label = {s["id"]: s["label"] for s in skills}
    mosaic = [{"id": c, "label": label[c]} for c in (i * 7 + 2 for i in range(14))]
    return {"top": top, "compare": compare, "mosaic_skills": mosaic}


def main():
    items = json.loads((CDM / "response_matrix_v2_full_items.json").read_text())
    llms = json.loads((CDM / "response_matrix_v2_full_llms.json").read_text())
    Q = np.load(CDM / "qmatrix_v2_K100.npy")
    R = np.load(CDM / "response_matrix_v2_full.npy")
    raw = load_prompts(items)
    texts = [item_text(r, it["benchmark"]) for r, it in zip(raw, items)]
    windows = gpqa_windows(raw, items)
    texts = [None if t and shares_gpqa_text(t, windows) else t for t in texts]

    skills_path = SITE / "skills.json"
    skills = json.loads(skills_path.read_text())
    for s in skills:
        k = s["id"]
        shown = [e["item_idx"] for e in s["example_items"] if texts[e["item_idx"]] and Q[e["item_idx"], k]]
        pool = [i for i in np.random.default_rng(42 + k).permutation(np.where(Q[:, k] > 0)[0])
                if texts[i] and i not in shown]
        chosen = (shown + [int(i) for i in pool])[:PER_SKILL]
        s["example_items"] = [{"item_idx": int(i), "benchmark": items[i]["benchmark"],
                               "subtask": items[i]["subtask"], "text": texts[i]} for i in chosen]
        need = np.where(Q[:, k] > 0)[0]
        counts = {}
        for i in need:
            counts[items[i]["benchmark"]] = counts.get(items[i]["benchmark"], 0) + 1
        s["items_by_benchmark"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    skills_path.write_text(json.dumps(skills, ensure_ascii=False))  # same layout as before: one line
    n = sum(len(s["example_items"]) for s in skills)
    print(f"example items: {n} across {sum(1 for s in skills if s['example_items'])} skills "
          f"(none for {sum(1 for s in skills if not s['example_items'])} skills whose items are all GPQA "
          f"or unusable)")

    models = json.loads((SITE / "theta_matrix.json").read_text())
    home = home_data(models, skills)
    (SITE / "home.json").write_text(json.dumps(home, ensure_ascii=False))
    print(f"home page data: top {len(home['top'])} models, compare pair {home['compare']}")

    wbs = weak_beats_strong(R, Q, llms)
    (SITE / "weak_beats_strong.json").write_text(json.dumps(wbs, indent=1) + "\n")
    print(f"weak beats strong: {wbs['n_items']} of {wbs['n_test_items']} held-out items ({wbs['pct']}%), "
          f"{sum(1 for s in wbs['skills'] if s['wbs'] > 0)} skills with at least one")


if __name__ == "__main__":
    main()
