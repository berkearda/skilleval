"""Phase 2: intervention ladder on the old taxonomy's validity, stratified by
Phase 1 validity band. Judges the SAME fixed pairs under four conditions:

  A  cluster name only                      (baseline, judge v3)
  B  name + one-line definition generated from OTHER member items   (tests H1)
  C  a regenerated name written from OTHER member items, original name hidden (H2)
  D  name-blind: category description inferred from OTHER member items (ceiling; gap to 100 ~ mixedness, H3)

Generation evidence never includes the judged items (no circularity).

    .venv312/bin/python tools/phase2_ladder.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
LABF = REPO / "cdm_exploration/experiments/oldtax_full_format.json"
PROFF = REPO / "cdm_exploration/experiments/oldtax_percluster_v3.json"
OUTF = REPO / "cdm_exploration/experiments/phase2_ladder.json"
PER_BAND = 12       # clusters per validity band
PAIRS = 8           # judged pairs per cluster (fixed across conditions)
EVID = 6            # member items used as generation evidence (disjoint from pairs)
MINSZ = PAIRS + EVID
random.seed(42)

VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(LABF))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    sk_items = defaultdict(list)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])

    prof = json.load(open(PROFF))["per_cluster"]
    def band(v):
        return "low" if v <= .2 else ("mid" if v < .6 else "high")
    by_band = defaultdict(list)
    for c in prof:
        if c["size"] >= MINSZ:
            by_band[band(c["validity"])].append(c["skill"])
    chosen = {b: random.sample(v, min(PER_BAND, len(v))) for b, v in by_band.items()}
    for b in ("low", "mid", "high"):
        print(f"band {b}: {len(chosen.get(b, []))} clusters")

    # fixed judged pairs + disjoint generation evidence per cluster
    plan = {}
    for b, skills in chosen.items():
        for sk in skills:
            items = random.sample(sk_items[sk], MINSZ)
            plan[sk] = {"band": b, "judge": items[:PAIRS], "evid": items[PAIRS:]}

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def call(system, user):
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return json.loads(r.choices[0].message.content)
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return {}
        return {}

    def ev_block(sk, n=EVID, cut=300):
        return "\n".join(f"- {qtext.get(i,'')[:cut]}" for i in plan[sk]["evid"][:n])

    # ---- generation steps (definitions, new names, blind categories) ----
    def gen_def(sk):
        u = (f"Example test items all tagged with the skill category '{sk}':\n{ev_block(sk)}\n\n"
             'Write ONE line defining the skill this group measures. Return JSON {"definition": "<one line>"}')
        return sk, str(call("You write concise skill definitions for a test taxonomy.", u).get("definition", ""))

    def gen_name(sk):
        u = (f"These test items form one group in a skill taxonomy:\n{ev_block(sk)}\n\n"
             "Write a concise name (at most 8 words) for the skill this group measures. "
             'Return JSON {"name": "<concise name>"}')
        return sk, str(call("You name skill groups from their member items. You are not shown any existing name.", u).get("name", ""))

    def gen_cat(sk):
        u = (f"Read these test items:\n{ev_block(sk, n=4)}\n\n"
             "In one or two sentences, describe the category of skill they share. "
             'Return JSON {"category": "<description>"}')
        return sk, str(call("You infer what connects a group of test items, as a human annotator would.", u).get("category", ""))

    skills = list(plan)
    with ThreadPoolExecutor(max_workers=8) as ex:
        defs = dict(ex.map(gen_def, skills))
        names = dict(ex.map(gen_name, skills))
        cats = dict(ex.map(gen_cat, skills))

    # ---- the four judged conditions on identical pairs ----
    def make_tasks(cond):
        t = []
        for sk in skills:
            for i in plan[sk]["judge"]:
                t.append((cond, sk, i))
        return t

    def render(cond, sk):
        if cond == "A":
            return f"Skill: {sk}"
        if cond == "B":
            return f"Skill: {sk}\nDefinition: {defs[sk]}"
        if cond == "C":
            return f"Skill: {names[sk]}"
        return f"Skill (described): {cats[sk]}"

    def judge(t):
        cond, sk, i = t
        u = (f"Item:\n{qtext.get(i,'')}\n\n{render(cond, sk)}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return (cond, sk, call(VSYS, u).get("verdict") == "yes")

    tasks = [t for c in "ABCD" for t in make_tasks(c)]
    print(f"judged calls: {len(tasks)}  (+{3*len(skills)} generation calls)")
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, tasks))

    agg = defaultdict(lambda: [0, 0])   # (band, cond) -> [yes, n]
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for cond, sk, ok in res:
        b = plan[sk]["band"]
        agg[(b, cond)][0] += ok
        agg[(b, cond)][1] += 1
        per[sk][cond][0] += ok
        per[sk][cond][1] += 1

    print(f"\n{'band':6s} {'A name':>8s} {'B +def':>8s} {'C rename':>9s} {'D blind':>8s}")
    for b in ("low", "mid", "high"):
        row = [agg[(b, c)][0] / max(agg[(b, c)][1], 1) for c in "ABCD"]
        print(f"{b:6s} " + " ".join(f"{v:8.0%}" for v in row))
    tot = [sum(agg[(b, c)][0] for b in ("low", "mid", "high")) /
           max(sum(agg[(b, c)][1] for b in ("low", "mid", "high")), 1) for c in "ABCD"]
    print(f"{'all':6s} " + " ".join(f"{v:8.0%}" for v in tot))

    json.dump({"plan": {s: plan[s]["band"] for s in plan},
               "definitions": defs, "new_names": names, "blind_categories": cats,
               "per_cluster": {s: {c: per[s][c] for c in "ABCD"} for s in per},
               "bands": {f"{b}|{c}": agg[(b, c)] for b in ("low", "mid", "high") for c in "ABCD"}},
              open(OUTF, "w"))
    print(f"\nwrote {OUTF}")


if __name__ == "__main__":
    main()
