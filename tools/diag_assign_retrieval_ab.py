"""Does the final labelling step miss skills that are already in the bank?

The v6b labelling pass leaves 1,870/9,523 questions (20%) matched to nothing,
and 79% of the unmatched MATH questions sit in a pipeline pile whose skill IS
in the bank. Two explanations:

  (H1) recall failure. The step pastes all 161 skills into every prompt
       (4,380 tok) and asks gpt-4o-mini to scan them for 10 questions at once.
       The right skill is present but not found.
  (H2) correct rejection. The skill genuinely does not fit; "none" is right.

H1 also predicts the step cannot generalise: bank size enters every prompt, so
a user with 500+ skills gets a different (worse) labeller, and results across
datasets are not comparable.

DESIGN. Same bank (161 skills, round7_dedup), same judge model (gpt-4o-mini),
same prompt wording, same "0 = none" option, same MAX_SKILLS. Every arm labels
all 9,523 questions. One variable changes at a time:

  A     full catalogue, batch 10   <- reproduces the shipped pipeline exactly
  A1    full catalogue, batch 1    <- isolates batching (2,000-question sample)
  B(K)  top-K catalogue, batch 1   <- isolates retrieval; K in {5, 15, 30}

B(K) vs A1 is the scientific comparison (only the candidate count differs).
A is the reference point for what we ship today. A1 exists solely so that a
B-vs-A difference cannot be blamed on batch size. K is swept rather than
picked, so no single operating point is chosen after seeing results (N4).

Retrieval uses text-embedding-3-small over "name | definition" for each skill
and over the first 900 characters of each question. 900 is not a choice made
here: it is what assign_all.py already feeds the judge, so the shortlist is
built from exactly the text the judge will read. Embedding more would let a
skill be shortlisted on evidence the judge never sees.

PRE-REGISTERED DECISION RULE (written before the run, the project log
2026-08-11). Adopt retrieval only if, at some swept K:
    (1) unassigned falls below arm A's, AND
    (2) mean name-free group score does NOT fall below arm A's minus 0.02.
If unassigned falls but coherence drops, the change is REJECTED and reported
as such: recovering questions by loosening the bar is the failure this step's
"none" option exists to prevent. If neither moves, H2 stands and the 20%
residue is a property of the data.

    python3 tools/diag_assign_retrieval_ab.py embed
    python3 tools/diag_assign_retrieval_ab.py run A|A1|B5|B15|B30
    python3 tools/diag_assign_retrieval_ab.py score
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.taxonomy_pipeline as tp
from cdmeval.utils.device import seed_everything
from cdmeval.utils.experiment import log_experiment

tp.P = tp.E / "pipeline_v51"
FRESH = tp.E / "pipeline_v51"
OUT = tp.E / "assign_ab"
OUT.mkdir(parents=True, exist_ok=True)
EMB_F = OUT / "emb_text3small.npz"

MAX_SKILLS = 4
A1_SAMPLE = 2000
EMB_MODEL = "text-embedding-3-small"

# verbatim from tools/assign_all.py so arm A reproduces the shipped step
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

ARMS = {
    "A":   dict(mode="full", batch=10, k=None, subset=None),
    "A1":  dict(mode="full", batch=1,  k=None, subset=A1_SAMPLE),
    "B5":  dict(mode="topk", batch=1,  k=5,    subset=None),
    "B15": dict(mode="topk", batch=1,  k=15,   subset=None),
    "B30": dict(mode="topk", batch=1,  k=30,   subset=None),
}


def build_bank():
    """Identical to assign_all.build_bank: non-flagged substantive skills."""
    snaps = sorted(FRESH.glob("round*.json"))
    snap = json.load(open(snaps[-1]))
    names = [s for s, d in snap["skills"].items()
             if len(d["items"]) >= 4 and not d.get("flag")]
    names.sort(key=lambda s: -len(snap["skills"][s]["items"]))
    return snap, [(s, snap["skills"][s]["definition"] or "") for s in names]


def load_qtext():
    return {r["item_idx"]: " ".join(r["question_full_text"].split())
            for r in json.load(open(tp.D / "item_full_text_recovered.json"))}


def cmd_embed():
    """Embed the 9,523 questions and the 161 skills. Cached; runs once."""
    if EMB_F.exists():
        print(f"already embedded -> {EMB_F}")
        return
    from openai import OpenAI
    cl = OpenAI(api_key=tp.load_key())
    snap, bank = build_bank()
    qtext = load_qtext()
    qids = sorted(qtext)
    qdocs = [qtext[q][:900] for q in qids]   # exactly what the judge sees
    sdocs = [f"{s} | {d}" for s, d in bank]

    def embed(docs, tag):
        vecs, B = [], 128
        for i in range(0, len(docs), B):
            r = cl.embeddings.create(model=EMB_MODEL, input=docs[i:i + B])
            vecs.extend(e.embedding for e in r.data)
            print(f"  {tag}: {min(i+B, len(docs))}/{len(docs)}", flush=True)
        v = np.asarray(vecs, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    QV, SV = embed(qdocs, "questions"), embed(sdocs, "skills")
    np.savez_compressed(EMB_F, qv=QV, sv=SV, qids=np.asarray(qids),
                        snames=np.asarray([s for s, _ in bank], dtype=object))
    print(f"wrote {EMB_F}  q={QV.shape} s={SV.shape}")


def cmd_run(arm):
    cfg = ARMS[arm]
    seed_everything(42)
    snap, bank = build_bank()
    qtext = load_qtext()
    all_q = sorted(qtext)

    if cfg["subset"]:
        all_q = sorted(random.Random(42).sample(all_q, cfg["subset"]))

    order = None
    if cfg["mode"] == "topk":
        z = np.load(EMB_F, allow_pickle=True)
        qv, sv, qids = z["qv"], z["sv"], list(z["qids"])
        assert list(z["snames"]) == [s for s, _ in bank], "bank/embedding mismatch"
        row = {q: i for i, q in enumerate(qids)}
        sim = qv @ sv.T
        order = {q: np.argsort(-sim[row[q]])[:cfg["k"]] for q in all_q}

    outf = OUT / f"assign_{arm}.jsonl"
    done = set()
    if outf.exists():
        done = {json.loads(l)["q"] for l in outf.open()}
        print(f"resuming: {len(done)} done")
    todo = [q for q in all_q if q not in done]
    if not todo:
        print("nothing to do"); return

    full_txt = "\n".join(f"{i+1}) {s} | {d}" for i, (s, d) in enumerate(bank))
    num2name_full = {i + 1: s for i, (s, _) in enumerate(bank)}
    jd = tp.Judges(tp.make_client(), qtext)

    def do_batch(chunk):
        if cfg["mode"] == "full":
            btxt, n2n = full_txt, num2name_full
        else:
            idx = order[chunk[0]]                      # batch==1 in every topk arm
            btxt = "\n".join(f"{j+1}) {bank[i][0]} | {bank[i][1]}"
                             for j, i in enumerate(idx))
            n2n = {j + 1: bank[i][0] for j, i in enumerate(idx)}
        qs = "\n".join(f"{q}. {qtext.get(q,'')[:900]}" for q in chunk)
        txt = jd._text(tp.SMALL, ASSIGN_SYS,
                       ASSIGN_U.format(bank=btxt, questions=qs), max_out=800)
        out = {}
        for line in (txt or "").splitlines():
            m = re.match(r"^\s*(\d+)\s*->\s*(.+)$", line)
            if not m:
                continue
            q = int(m.group(1))
            if q not in chunk:
                continue
            seen, skills = set(), []
            for n in (int(x) for x in re.findall(r"\d+", m.group(2))):
                if n in n2n and n2n[n] not in seen:
                    seen.add(n2n[n]); skills.append(n2n[n])
            out[q] = skills[:MAX_SKILLS]
        return out

    chunks = [todo[i:i + cfg["batch"]] for i in range(0, len(todo), cfg["batch"])]
    print(f"arm {arm}: {len(todo)} questions, {len(chunks)} calls, "
          f"catalogue={'161' if cfg['mode']=='full' else cfg['k']}, batch={cfg['batch']}")
    start = tp.spend["usd"]
    with open(outf, "a") as f:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for k, res in enumerate(ex.map(do_batch, chunks)):
                for q in chunks[k]:
                    f.write(json.dumps({"q": q, "skills": res.get(q, [])}) + "\n")
                f.flush()
                if k % 100 == 0:
                    print(f"  {k+1}/{len(chunks)}  spent ${tp.spend['usd']-start:.2f}",
                          flush=True)
    print(f"arm {arm} done, spent ${tp.spend['usd']-start:.2f}")


def cmd_score():
    """Unassigned + name-free group test per arm, then the pre-registered rule."""
    import statistics
    snap, bank = build_bank()
    qtext = load_qtext()
    bench = {p["item_idx"]: p["benchmark"] for p in
             json.load(open(tp.E / "oldtax_repaired_FINAL.json"))["per_item"]}
    rows = {}
    for arm in ARMS:
        f = OUT / f"assign_{arm}.jsonl"
        if not f.exists():
            continue
        a = {json.loads(l)["q"]: json.loads(l)["skills"] for l in f.open()}
        items = defaultdict(set)
        for q, sk in a.items():
            for s in sk:
                items[s].add(q)
        subst = {s: i for s, i in items.items() if len(i) >= 4}
        un = [q for q, sk in a.items() if not sk]
        st = tp.State({"skills": {s: {"definition": snap["skills"][s]["definition"],
                                     "items": sorted(i)} for s, i in items.items()}})
        st.round = 900
        jd = tp.Judges(tp.make_client(), qtext)
        tgt = sorted(subst)
        print(f"[{arm}] group test on {len(tgt)} piles ...", flush=True)
        tp.measure(st, jd, tgt, model=tp.BIG, val_n=1, grp_n=tp.CERT_GRP_N, topup=False)
        g = [st.metrics["group"][s] for s in tgt if s in st.metrics["group"]]
        perbm = defaultdict(lambda: [0, 0])
        for q, sk in a.items():
            b = bench.get(q)
            perbm[b][1] += 1
            if not sk:
                perbm[b][0] += 1
        rows[arm] = dict(n=len(a), unassigned=len(un), rate=len(un) / len(a),
                         skills=len(subst), group_mean=statistics.mean(g),
                         group_pass=sum(1 for x in g if x > 0.5), group_total=len(g),
                         per_benchmark={k: v[0] / v[1] for k, v in perbm.items()})
        json.dump(rows, open(OUT / "ab_report.json", "w"), indent=1)

    print(f"\n{'arm':5s} {'n':>6s} {'unassigned':>11s} {'rate':>7s} {'skills':>7s} "
          f"{'group':>7s} {'pass':>8s}")
    for arm, r in rows.items():
        print(f"{arm:5s} {r['n']:6d} {r['unassigned']:11d} {r['rate']:6.1%} "
              f"{r['skills']:7d} {r['group_mean']:6.3f} "
              f"{r['group_pass']}/{r['group_total']:>5}")

    if "A" in rows:
        base = rows["A"]
        print("\npre-registered rule: adopt only if unassigned falls AND "
              f"group mean >= {base['group_mean']-0.02:.3f}")
        for arm, r in rows.items():
            if not arm.startswith("B"):
                continue
            ok = r["rate"] < base["rate"] and r["group_mean"] >= base["group_mean"] - 0.02
            print(f"  {arm}: unassigned {base['rate']:.1%} -> {r['rate']:.1%}, "
                  f"group {base['group_mean']:.3f} -> {r['group_mean']:.3f}  "
                  f"=> {'ADOPT' if ok else 'REJECT'}")
    log_experiment(
        name="assign_retrieval_ab",
        config=dict(bank=len(bank), model=tp.SMALL, group_model=tp.BIG,
                    emb=EMB_MODEL, arms={k: v for k, v in ARMS.items()}, seed=42),
        results=rows,
        split_info=dict(n_items=9523, note="labelling step only; bank unchanged"),
        verified=True,
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "embed":
        cmd_embed()
    elif cmd == "run":
        cmd_run(sys.argv[2])
    else:
        cmd_score()
