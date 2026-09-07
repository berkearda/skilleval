"""Final assignment pass (TopicGPT/TnT-LLM stage): re-label ALL 9,523 questions
against the finished skill bank, from scratch.

Bank = 150 certified + 31 coherent-but-misnamed (group test passed, name off).
Flagged grab-bags excluded (would re-pollute). Each question gets 1-2 skills or
"none" (0). After assignment, the name-free group test is re-run on the new
piles (the non-circular check; validity is circular after assignment and is NOT
reported as a headline).

    python3 tools/assign_all.py assign     # the labeling pass (resumable)
    python3 tools/assign_all.py finalize   # build v6 + re-run group test + HTML
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.taxonomy_pipeline as tp

tp.P = tp.E / "pipeline_v51"
FRESH = tp.E / "pipeline_v51"
MULTI = True                       # allow every genuinely-required skill (not capped at 2)
ASSIGN_F = FRESH / "assignments_v6b.jsonl"   # against the deduped v5.1 bank (round7_dedup)
MAX_SKILLS = 4                     # safety ceiling against runaway over-tagging
BATCH = 10

ASSIGN_SYS = ("You label test questions with the cognitive skill(s) each one requires, "
              "choosing from a fixed catalogue. A skill is the smallest named mental "
              "operation a solver must master to answer a question correctly. Judge by the "
              "operation the question requires, never by its surface topic.")
ASSIGN_U = """Skill catalogue (each line: number) name | definition):
{bank}

Below are test questions. For EACH question id, list EVERY catalogue skill whose operation the question genuinely requires to be solved. Many questions need one skill; some need two or more. Do NOT add a skill the question does not truly require, and do NOT list a skill just because the topic is related. If NO catalogue skill fits, output 0.

Questions:
{questions}

Output format: PLAIN TEXT, one line per question, exactly:
<question id> -> <number> [number ...]
Use 0 when nothing fits. No prose, no other text."""


def build_bank():
    # latest snapshot = round7_dedup after the v5.1 dedup; bank = non-flagged substantive
    snaps = sorted(FRESH.glob("round*.json"))
    snap = json.load(open(snaps[-1]))
    names = [s for s, d in snap["skills"].items()
             if len(d["items"]) >= 4 and not d.get("flag")]
    names.sort(key=lambda s: -len(snap["skills"][s]["items"]))
    bank = [(s, snap["skills"][s]["definition"] or "") for s in names]
    return snap, bank


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "assign"
    snap, bank = build_bank()
    num2name = {i + 1: s for i, (s, _) in enumerate(bank)}
    bank_txt = "\n".join(f"{i+1}) {s} | {d}" for i, (s, d) in enumerate(bank))
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(tp.D / "item_full_text_recovered.json"))}
    all_q = sorted(qtext)

    if mode == "assign":
        done = {}
        if ASSIGN_F.exists():
            for line in open(ASSIGN_F):
                d = json.loads(line)
                done[d["q"]] = d["skills"]
            print(f"resuming: {len(done)} already assigned")
        todo = [q for q in all_q if q not in done]
        print(f"bank {len(bank)} skills ({len(bank_txt)//4} tok); assigning {len(todo)} questions")
        jd = tp.Judges(tp.make_client(), qtext)

        def do_batch(chunk):
            qs = "\n".join(f"{q}. {qtext.get(q,'')[:900]}" for q in chunk)
            txt = jd._text(tp.SMALL, ASSIGN_SYS,
                           ASSIGN_U.format(bank=bank_txt, questions=qs), max_out=800)
            out = {}
            for line in (txt or "").splitlines():
                m = re.match(r"^\s*(\d+)\s*->\s*(.+)$", line)
                if not m:
                    continue
                q = int(m.group(1))
                if q not in chunk:
                    continue
                nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
                seen, skills = set(), []
                for n in nums:
                    if n in num2name and num2name[n] not in seen:
                        seen.add(num2name[n]); skills.append(num2name[n])
                out[q] = skills[:MAX_SKILLS]
            return out

        chunks = [todo[i:i+BATCH] for i in range(0, len(todo), BATCH)]
        with open(ASSIGN_F, "a") as f:
            with ThreadPoolExecutor(max_workers=8) as ex:
                for k, res in enumerate(ex.map(do_batch, chunks)):
                    for q in chunks[k]:
                        f.write(json.dumps({"q": q, "skills": res.get(q, [])}) + "\n")
                    f.flush()
                    if k % 20 == 0:
                        print(f"  {k+1}/{len(chunks)} batches  spent ${tp.spend['usd']:.2f}",
                              flush=True)
        print(f"assignment done, spent ${tp.spend['usd']:.2f}")
        return

    # ---- finalize
    assign = {}
    for line in open(ASSIGN_F):
        d = json.loads(line)
        assign[d["q"]] = d["skills"]
    old_prim = {}
    for s, dd in snap["skills"].items():
        for i in dd["items"]:
            old_prim.setdefault(i, s)

    new_items = defaultdict(set)
    memb = []
    unassigned = []
    for q in all_q:
        sk = assign.get(q, [])
        if not sk:
            unassigned.append(q)
        for s in sk:
            new_items[s].add(q)
        memb.append(len(sk))
    moved = sum(1 for q in all_q if assign.get(q) and old_prim.get(q) not in assign.get(q, []))
    subst = {s: it for s, it in new_items.items() if len(it) >= 4}
    print(f"assigned {len(all_q)-len(unassigned)}/{len(all_q)} "
          f"({len(unassigned)} unassigned, {len(unassigned)/len(all_q):.0%})")
    print(f"mean skills per assigned question: "
          f"{sum(m for m in memb if m)/max(1,len(all_q)-len(unassigned)):.2f}")
    print(f"skills surviving with >=4 questions: {len(subst)} / {len(bank)}")
    print(f"questions whose primary skill changed: {moved} ({moved/len(all_q):.0%})")

    # build state and re-run the name-free group test (non-circular check)
    st = tp.State({"skills": {s: {"definition": snap["skills"][s]["definition"],
                                  "items": sorted(it)} for s, it in new_items.items()}})
    st.round = 500
    jd = tp.Judges(tp.make_client(), qtext)
    targets = sorted(subst)
    print(f"re-running group test on {len(targets)} new piles (gpt-4.1-mini, 12 trials)...")
    tp.measure(st, jd, targets, model=tp.BIG, val_n=1, grp_n=tp.CERT_GRP_N, topup=False)
    gscores = [st.metrics["group"][s] for s in targets if s in st.metrics["group"]]
    passed = sum(1 for g in gscores if g > (0.75 if False else 0.5))
    import statistics
    print(f"group-test PASS (>50%): {passed}/{len(gscores)}  "
          f"mean {statistics.mean(gscores):.0%}  median {statistics.median(gscores):.0%}")

    out = {"bank_size": len(bank), "assigned": len(all_q)-len(unassigned),
           "unassigned": len(unassigned), "mean_skills_per_q":
           sum(m for m in memb if m)/max(1,len(all_q)-len(unassigned)),
           "surviving_skills": len(subst), "moved": moved,
           "group_pass": passed, "group_total": len(gscores),
           "group_mean": statistics.mean(gscores),
           "per_skill": {s: {"n": len(new_items[s]), "group": st.metrics["group"].get(s)}
                         for s in targets}}
    json.dump(out, open(FRESH / "assign_report.json", "w"), indent=1)

    # export v6 taxonomy
    bench = {pi["item_idx"]: pi["benchmark"] for pi in
             json.load(open(tp.E / "oldtax_repaired_FINAL.json"))["per_item"]}
    subtask = {pi["item_idx"]: pi.get("subtask") for pi in
               json.load(open(tp.E / "oldtax_repaired_FINAL.json"))["per_item"]}
    per_item = [{"item_idx": q, "benchmark": bench.get(q), "subtask": subtask.get(q),
                 "skills": assign.get(q, [])} for q in all_q]
    v6 = {"bank": [{"name": s, "definition": snap["skills"][s]["definition"],
                    "n_items": len(new_items[s]),
                    "group": st.metrics["group"].get(s)} for s in sorted(new_items)],
          "per_item": per_item, "unassigned_count": len(unassigned),
          "provenance": "assign_all.py: global reassignment of all 9523 questions vs the "
                        "v5.1 certified+coherent bank, name-free group test re-run, 2026-07-23"}
    vfile = "oldtax_pipeline_v6b.json"
    json.dump(v6, open(tp.E / vfile, "w"))
    print(f"wrote {vfile}  spent ${tp.spend['usd']:.2f}")


if __name__ == "__main__":
    main()
