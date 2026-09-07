"""Pre-flight A (T-074): does the generic dimension-hint sentence fix the
instruction-type group-test blind spot WITHOUT hurting normal skills?

Arms (both run WITH the hint; no-hint scores come from stored metrics):
  instr   - substantive skills with group<=50% and validity>=70%
  control - substantive skills with group>=83% (currently passing)

Keep the hint only if instr scores rise and control scores hold.

    .venv312/bin/python tools/diag_hint_ab.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
E = REPO / "cdm_exploration/experiments"
LABF = E / "oldtax_repaired_FINAL.json"
METF = E / "oldtax_repaired_FINAL_metrics.json"
OUTF = E / "preflight_hint_ab.json"
TRIALS = 6
N_CONTROL = 15
random.seed(42)

ISYS = "You find the one item that does NOT belong with the others."
HINT = ("Note: the questions are grouped by the operation or instruction they "
        "require, which may have nothing to do with their topic or subject area.")


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(LABF))
    met = json.load(open(METF))
    g, v = met["group_test"], met["validity"]
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}

    sk_items = defaultdict(list)
    item_skills = defaultdict(set)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            sk_items[s].append(pi["item_idx"])
            item_skills[pi["item_idx"]].add(s)
    items_all = sorted(item_skills)
    subst = {s for s, it in sk_items.items() if len(set(it)) >= 4}

    instr = sorted(s for s in subst
                   if g.get(s) is not None and g[s] <= 0.5
                   and v.get(s) is not None and v[s] >= 0.7)
    passing = sorted(s for s in subst if g.get(s) is not None and g[s] >= 5 / 6)
    control = random.sample(passing, min(N_CONTROL, len(passing)))
    print(f"instr arm: {len(instr)}  control arm: {len(control)}")

    tasks = []
    for arm, skills in (("instr", instr), ("control", control)):
        for sk in skills:
            uniq = list(set(sk_items[sk]))
            for _ in range(TRIALS):
                own = random.sample(uniq, 4)
                pool = [i for i in items_all if sk not in item_skills[i]]
                intr = random.choice(pool)
                five = own + [intr]
                random.shuffle(five)
                tasks.append((arm, sk, five, five.index(intr) + 1))
    print(f"trials={len(tasks)}")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(t):
        arm, sk, five, truth = t
        body = "\n".join(f"{k+1}. {qtext.get(i,'')[:400]}" for k, i in enumerate(five))
        u = ("These five test questions should all belong to one group that tests the same "
             f"skill. {HINT} Exactly one does NOT belong. Which number is the odd one out?"
             f"\n\n{body}\n\n"
             'Return JSON {"intruder": <number 1-5>}.')
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": ISYS},
                              {"role": "user", "content": u}])
                return (arm, sk, int(json.loads(r.choices[0].message.content).get("intruder", -1)) == truth)
            except Exception:  # noqa: BLE001
                if a == 2:
                    return (arm, sk, False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, tasks))

    hit = defaultdict(int); n = defaultdict(int); arm_of = {}
    for arm, sk, ok in res:
        n[sk] += 1; hit[sk] += ok; arm_of[sk] = arm

    out = {"hint": HINT, "trials_per_skill": TRIALS, "arms": {}}
    for arm in ("instr", "control"):
        rows = []
        for sk in sorted(a for a in arm_of if arm_of[a] == arm):
            rows.append({"skill": sk, "no_hint": g.get(sk),
                         "with_hint": hit[sk] / n[sk], "trials": n[sk]})
        mean_no = sum(r["no_hint"] for r in rows) / len(rows)
        mean_hint = sum(r["with_hint"] for r in rows) / len(rows)
        out["arms"][arm] = {"skills": rows, "mean_no_hint": mean_no,
                            "mean_with_hint": mean_hint}
        print(f"\n{arm}: mean no-hint {mean_no:.0%} -> with-hint {mean_hint:.0%}")
        for r in sorted(rows, key=lambda r: r["with_hint"]):
            print(f"  {r['no_hint']:4.0%} -> {r['with_hint']:4.0%}  {r['skill'][:58]}")

    json.dump(out, open(OUTF, "w"), indent=1)
    print(f"\nwrote {OUTF}")


if __name__ == "__main__":
    main()
