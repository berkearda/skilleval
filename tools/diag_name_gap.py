"""Pre-flight B (T-075): calibrate the own-vs-strangers gap threshold for
name acceptance.

gap = validity on the skill's OWN questions minus validity on questions
drawn from OTHER skills. Specific names have a big gap (accept own,
reject strangers); vague names have a small gap (accept everything).

Arms:
  trusted - 10 contract-clean names with group>=83% and validity>=80%
  vague   - "Evaluate" (the real catch-all) + 4 synthetic vague names
            attached to real trusted clusters

Threshold = midpoint between max vague gap and min trusted gap.

    python3 tools/diag_name_gap.py
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
OUTF = E / "preflight_name_gap.json"
N_OWN = 10
N_STRANGER = 10
random.seed(42)

VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill. Be strict."

SYNTH_VAGUE = [
    ("follow the given instructions", "Follow whatever instructions the question gives."),
    ("solve the given problem", "Work out the answer to the problem that is asked."),
    ("apply reasoning to find the answer", "Use reasoning to arrive at the correct answer."),
    ("understand the question and respond", "Read the question carefully and produce a response."),
]

BANNED = ("problem solving", "reasoning", "analysis", "skills", "advanced")


def compliant(name):
    w = name.split()
    return (name == name.lower() and 3 <= len(w) <= 8 and " and " not in f" {name} "
            and " or " not in f" {name} " and "/" not in name
            and not any(b in name for b in BANNED))


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def main():
    data = json.load(open(LABF))
    met = json.load(open(METF))
    g, v = met["group_test"], met["validity"]
    defn = {b["name"]: b.get("definition", "") for b in data["bank"]}
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}

    sk_items = defaultdict(set)
    for pi in data["per_item"]:
        for s in pi["skills"]:
            sk_items[s].add(pi["item_idx"])
    items_all = sorted({i for it in sk_items.values() for i in it})

    trusted_pool = sorted(s for s, it in sk_items.items() if len(it) >= 10
                          and (g.get(s) or 0) >= 5 / 6 and (v.get(s) or 0) >= 0.8
                          and compliant(s))
    trusted = random.sample(trusted_pool, min(10, len(trusted_pool)))

    # each name = (label, definition, own item pool, arm)
    probes = [(s, defn.get(s, ""), sk_items[s], "trusted") for s in trusted]
    if "Evaluate" in sk_items:
        probes.append(("Evaluate", defn.get("Evaluate", ""), sk_items["Evaluate"], "vague"))
    host_pool = [s for s in trusted_pool if s not in trusted] or trusted
    for k, (name, d) in enumerate(SYNTH_VAGUE):
        host = host_pool[k % len(host_pool)]
        probes.append((name, d, sk_items[host], "vague"))

    tasks = []
    for name, d, own_pool, arm in probes:
        own = random.sample(sorted(own_pool), min(N_OWN, len(own_pool)))
        strangers = random.sample([i for i in items_all if i not in own_pool], N_STRANGER)
        for i in own:
            tasks.append((name, d, arm, "own", i))
        for i in strangers:
            tasks.append((name, d, arm, "stranger", i))
    print(f"names={len(probes)}  judgments={len(tasks)}")

    from openai import OpenAI
    client = OpenAI(api_key=load_key())

    def judge(t):
        name, d, arm, side, i = t
        dline = f"\nDefinition: {d}" if d else ""
        u = (f"Item:\n{qtext.get(i, '')}\n\nSkill: {name}{dline}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini", temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": VSYS},
                              {"role": "user", "content": u}])
                return (name, arm, side, json.loads(r.choices[0].message.content).get("verdict") == "yes")
            except Exception:  # noqa: BLE001
                if a == 2:
                    return (name, arm, side, False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(judge, tasks))

    yes = defaultdict(int); n = defaultdict(int); arm_of = {}
    for name, arm, side, ok in res:
        yes[(name, side)] += ok; n[(name, side)] += 1; arm_of[name] = arm

    rows = []
    for name in sorted(arm_of):
        own_v = yes[(name, "own")] / n[(name, "own")]
        str_v = yes[(name, "stranger")] / n[(name, "stranger")]
        rows.append({"name": name, "arm": arm_of[name], "own": own_v,
                     "stranger": str_v, "gap": own_v - str_v})
    for r in sorted(rows, key=lambda r: r["gap"]):
        print(f"  {r['arm']:7s} own {r['own']:4.0%}  stranger {r['stranger']:4.0%}  "
              f"gap {r['gap']:+5.0%}  {r['name'][:50]}")

    tg = [r["gap"] for r in rows if r["arm"] == "trusted"]
    vg = [r["gap"] for r in rows if r["arm"] == "vague"]
    thr = (min(tg) + max(vg)) / 2
    print(f"\ntrusted gaps min {min(tg):.0%}  vague gaps max {max(vg):.0%}  -> threshold {thr:.0%}")
    if max(vg) >= min(tg):
        print("WARNING: arms overlap - threshold not clean, inspect before adopting")

    json.dump({"rows": rows, "threshold": thr, "n_own": N_OWN, "n_stranger": N_STRANGER,
               "overlap": max(vg) >= min(tg)}, open(OUTF, "w"), indent=1)
    print(f"wrote {OUTF}")


if __name__ == "__main__":
    main()
