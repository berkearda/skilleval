"""Self-verifying taxonomy repair loop (T-076/T-077).

Design + agreed parameters: the project log 2026-07-22. Instrument frozen by
pre-flight A (dimension hint adopted: instr 36%->71%, control 92%->91%,
preflight_hint_ab.json) and pre-flight B (name gap threshold 0.55,
preflight_name_gap.json).

Modes:
    python3 tools/taxonomy_pipeline.py dry      # routing report, no API
    python3 tools/taxonomy_pipeline.py loop     # repair to fixed point
    python3 tools/taxonomy_pipeline.py certify  # one-shot certification

State: cdm_exploration/experiments/pipeline/round{N}.json (snapshots,
never overwritten), journal.jsonl (append-only op log), anchors.json
(judge-drift canaries), certified.json (the paper numbers).

Routing (precedence pile > rename > merge):
    group<=50% (after top-up)            -> RE-PILE
    group ok, validity<=50%              -> RENAME
    name contract violation              -> RENAME
    <4 questions                         -> residue (excluded, reported)
    2 failed repairs / repeated state    -> IRREDUCIBLE (flagged)
Certification failures FLAG, never loop back (anti-Goodhart).
"""
from __future__ import annotations

import datetime
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "cdm_exploration/data/cdm_ready"
E = REPO / "cdm_exploration/experiments"
P = E / "pipeline"          # overridden to pipeline_fresh in fresh mode (see main)

SMALL = "gpt-4o-mini"           # repair judge
BIG = "gpt-4.1-mini"            # piling + certification judge
ROUTE_THR = 0.5
BIG_CLUSTER = 150               # clusters above this size face the stricter group bar
BIG_THR = 0.75                  # (flag audit 2026-07-22: fusion hides in the 200-500q range too)
LOWVAL_PILE = 0.4               # group-passing cluster with validity <= this and
LOWVAL_SIZE = 100               # size >= this is an unnameable grab-bag: re-pile it
RENAME_CAP = 4                  # renames are cheap and luck-sensitive; piling stays at 2
GAP_THR = 0.55                  # pre-flight B
STRICT_PASS = (10, 12)          # post-repair group acceptance
VAL_N, GRP_N = 10, 6            # routing sample sizes
VAL_BAND = (0.35, 0.65)         # top-up bands (inclusive)
GRP_BAND = (0.3, 0.7)
ATTEMPT_CAP = 2
ROUND_CAP = 6                   # raised from 3 on 2026-07-22: fresh run needed >3 rounds
                                # to reach fixed point (work: 7848q -> 5238q -> 577q)
CERT_VAL_N, CERT_GRP_N = 20, 12
GAP_N = 10                      # own/stranger sample size for the certification gap test
HARD_ABORT_USD = 5.0
PRICE = {SMALL: (0.15, 0.60), BIG: (0.40, 1.60)}   # $/1M tokens in, out

HINT = ("Note: the questions are grouped by the operation or instruction they "
        "require, which may have nothing to do with their topic or subject area.")
ISYS = "You find the one item that does NOT belong with the others."
VSYS = "You judge whether solving a test item genuinely requires a given cognitive skill."
# "Be strict." dropped 2026-07-22: A/B (preflight_strict_ab.json) showed it is inert
# (own 90/91%, stranger 3/4%, gap +87% both arms)
SKILL_DEF = ("A skill is the smallest named mental operation a solver must master to answer "
             "a question correctly; the same skill can appear across different topics and formats.")
PSYS = (f"You sort test questions into groups by the cognitive skill each requires. {SKILL_DEF} "
        "You judge by the mental operation needed to solve a question, never by surface topic words.")
NSYS = (f"You rename one skill in a test-skill taxonomy so that the name follows a strict "
        f"format. {SKILL_DEF} You are given the skill's questions; the name must describe "
        "what they require.")

RULES = """Requirements:
1. Judge by the operation needed to solve. Same topic does not mean same skill; different wording can be the same skill.
2. Each question goes in ONE pile; it may appear in TWO piles only if it genuinely requires both operations, never more.
3. Each pile is ONE specific operation; if a pile needs "and" or "or" to describe, split it.
4. Use FEW piles (between 2 and 25 for a group like this). Single-question piles only as a truly last resort.
5. Pile names: lowercase verb phrase, one verb plus one specific object, 3-8 words, no "and"/"or"/slashes.
6. Description: at most 12 words."""

COMPACT_U = """Below are {n} test questions, numbered. Sort EVERY question into piles so that all questions in one pile require the SAME specific cognitive skill.

{rules}

Questions:
{questions}

Output format: PLAIN TEXT, one line per pile, exactly like this:
1) name of operation | short description | 3 7 12 15
No JSON, no other text. Every question number from 1 to {n} must appear on at least one line."""

DISC_U = """Below are {n} test questions, numbered, sampled from a larger group. Sort EVERY question into piles so that all questions in one pile require the SAME specific cognitive skill.

{rules}

Questions:
{questions}

Return JSON only: {{"piles": [{{"name": "...", "description": "...", "questions": [1,4,7]}}, ...]}}"""

ASSIGN_U = """Here is a table of skill piles:
{table}

Below are test questions, each with an id. For EACH question, choose the pile whose operation it requires.{fallback}

Questions:
{questions}

Output format: PLAIN TEXT, one line per question, exactly:
<question id> -> <pile number>
No other text."""

REWRITE_U = """This skill currently has a name that needs replacing. Write a new name for it based on its questions.

Questions tagged with this skill:
{ev}

Current name (for context only, do not copy its style): {old}

Name format, all rules mandatory:
1. A lowercase verb phrase: one verb plus one specific object, for example "track object positions after swaps".
2. Between 3 and 8 words. A bare verb alone is forbidden.
3. No "and", no "or", no slashes. If the questions span two operations, name the one most of them require.
4. Do not name a subject area - name the operation. The name must be SPECIFIC: it should fit these questions and NOT fit unrelated questions.
{extra}
Return JSON only: {{"name": "<new name>", "definition": "<one sentence, at most 15 words>"}}"""

LINE = re.compile(r"^\s*\d+\)\s*(.+?)\s*\|\s*(.*?)\s*\|\s*([\d\s]+)$")
spend = {"usd": 0.0}


def violations(name):
    """Structural name contract only. Domain-neutral by design.

    The banned-word list was removed 2026-08-11. It was tuned to these five
    benchmarks ("reasoning" is noise in BBH and legitimate in a legal-reasoning
    corpus), it matched on substrings so it rejected honest names containing
    "analysis", and every entry used alone was already blocked by the 3-word
    minimum. The measured replacement is stranger_gap(), now applied to every
    skill at certification: a generic name fits unrelated questions too, so its
    gap collapses. See the project log 2026-08-11.
    """
    v = []
    w = name.replace("_", " ").split()
    if name != name.lower():
        v.append("not lowercase")
    if len(w) < 3:
        v.append("fewer than 3 words")
    if len(w) > 8:
        v.append("more than 8 words")
    low = " " + name.lower() + " "
    if " and " in low or " or " in low or "/" in name:
        v.append("contains and/or/slash")
    return v


def stranger_gap(st, jd, s, model=BIG, n=10):
    """Own-vs-stranger validity gap: the measured anti-generic guard.

    Ask the judge whether the skill is required by n of its OWN questions and
    by n unrelated ones. A specific name scores high on its own and near zero
    on strangers. A generic name ("logical reasoning") scores high on both, so
    its gap collapses and it fails. This is what the banned word list was
    approximating, except it is measured on the actual data and carries no
    English word list, so it works on any corpus.

    Added 2026-08-11. Before that this ran only inside rename(), so only skills
    that failed and got repaired ever faced it: 66 of the 99 shipped v6b skills
    (67%), including all eight largest, were never stranger-tested.
    """
    d = st.skills[s]
    r = random.Random(hash((s, "gap")) & 0xffffffff)
    own = r.sample(sorted(d["items"]), min(n, len(d["items"])))
    pool = [i for i in {i for x in st.skills.values() for i in x["items"]}
            if i not in d["items"]]
    strangers = r.sample(sorted(pool), min(n, len(pool)))
    with ThreadPoolExecutor(max_workers=8) as ex:
        ov = sum(ex.map(lambda i: jd.validity_one(model, s, d["definition"], i), own))
        sv = sum(ex.map(lambda i: jd.validity_one(model, s, d["definition"], i), strangers))
    o, t = ov / len(own), sv / len(strangers)
    return o, t, o - t


def degenerate(piles, n):
    if not piles:
        return True
    singles = sum(1 for p in piles if len(p["nums"]) == 1)
    memb = sum(len(p["nums"]) for p in piles)
    return (len(piles) > 0.5 * n or (len(piles) >= 5 and singles / len(piles) > 0.6)
            or memb > 1.5 * n)


def load_key():
    t = (Path.home() / ".cdmeval_openai_key").read_text().strip()
    return t.split("=", 1)[1].strip().strip('"').strip("'") if t.startswith("OPENAI_API_KEY=") else t


def journal(op, **kw):
    kw.update(op=op, ts=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(P / "journal.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")


class State:
    """skills: name -> {definition, items(set), attempts, flag}. per-item view derived."""

    def __init__(self, snap):
        self.skills = {s: {"definition": d["definition"], "items": set(d["items"]),
                           "attempts": d.get("attempts", 0), "flag": d.get("flag")}
                       for s, d in snap["skills"].items()}
        self.metrics = snap.get("metrics", {"validity": {}, "group": {}, "n_val": {}, "n_grp": {}})
        self.seen_states = {(s, tuple(it)) for s, it in snap.get("seen_states", [])}
        self.round = snap.get("round", 0)

    @classmethod
    def from_final(cls):
        fin = json.load(open(E / "oldtax_repaired_FINAL.json"))
        defn = {b["name"]: b.get("definition", "") for b in fin["bank"]}
        sk = defaultdict(set)
        for pi in fin["per_item"]:
            for s in pi["skills"]:
                sk[s].add(pi["item_idx"])
        return cls({"skills": {s: {"definition": defn.get(s, ""), "items": sorted(it)}
                               for s, it in sk.items()}})

    @classmethod
    def from_original(cls):
        """The submitted K=100 taxonomy: qmatrix (multi-membership kept) + plurality names."""
        import numpy as np
        q = np.load(D / "qmatrix_v2_K100.npy")
        lab = json.load(open(D / "cluster_labels_v2_K100.json"))
        skills = {}
        for k in range(q.shape[1]):
            name = lab.get(str(k), f"cluster {k}")
            base, j = name, 2
            while name in skills:
                name = f"{base} ({j})"; j += 1
            skills[name] = {"definition": "", "items": sorted(np.nonzero(q[:, k])[0].tolist())}
        return cls({"skills": skills})

    def save(self, tag):
        out = {"round": self.round, "seen_states": sorted(map(list, self.seen_states)),
               "metrics": self.metrics,
               "skills": {s: {"definition": d["definition"], "items": sorted(d["items"]),
                              "attempts": d["attempts"], "flag": d["flag"]}
                          for s, d in self.skills.items()}}
        f = P / f"{tag}.json"
        json.dump(out, open(f, "w"))
        return f

    def substantive(self):
        return {s for s, d in self.skills.items() if len(d["items"]) >= 4}

    def item_cover(self):
        cov = defaultdict(int)
        for d in self.skills.values():
            for i in d["items"]:
                cov[i] += 1
        return cov

    def state_key(self, s):
        return (s, tuple(sorted(self.skills[s]["items"])))


def latest_snapshot():
    snaps = sorted(P.glob("round*.json"))
    return snaps[-1] if snaps else None


def make_client():
    from openai import OpenAI
    return OpenAI(api_key=load_key(), timeout=900)


def priced(model, tin, tout):
    i, o = PRICE[model]
    spend["usd"] += (tin * i + tout * o) / 1e6
    if spend["usd"] > HARD_ABORT_USD:
        raise RuntimeError(f"hard abort: spend ${spend['usd']:.2f} > ${HARD_ABORT_USD}")


class Judges:
    def __init__(self, client, qtext):
        self.c, self.q = client, qtext

    def _json(self, model, system, user, field, max_out=4000):
        for a in range(3):
            try:
                r = self.c.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=max_out,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                priced(model, (len(system) + len(user)) // 4,
                       len(r.choices[0].message.content or "") // 4)
                return json.loads(r.choices[0].message.content).get(field)
            except Exception:  # noqa: BLE001
                if a == 2:
                    return None

    def _text(self, model, system, user, max_out=30000):
        for a in range(2):
            try:
                r = self.c.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=max_out,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                priced(model, (len(system) + len(user)) // 4,
                       len(r.choices[0].message.content or "") // 4)
                return r.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                if a == 1:
                    return ""

    def validity_one(self, model, name, definition, i):
        dline = f"\nDefinition: {definition}" if definition else ""
        u = (f"Item:\n{self.q.get(i, '')}\n\nSkill: {name}{dline}\n\n"
             'Does solving this item genuinely require this skill? Return JSON {"verdict":"yes|no"}.')
        return self._json(model, VSYS, u, "verdict") == "yes"

    def group_one(self, model, five, truth):
        body = "\n".join(f"{k+1}. {self.q.get(i, '')[:400]}" for k, i in enumerate(five))
        u = ("These five test questions should all belong to one group that tests the same "
             f"skill. {HINT} Exactly one does NOT belong. Which number is the odd one out?"
             f"\n\n{body}\n\n"
             'Return JSON {"intruder": <number 1-5>}.')
        try:
            return int(self._json(model, ISYS, u, "intruder")) == truth
        except (TypeError, ValueError):
            return False


def measure(st, jd, targets, model=SMALL, val_n=VAL_N, grp_n=GRP_N, topup=True):
    """Measure validity + group for targets; borderline top-up before storing."""
    rng = random.Random(42 + st.round)
    all_items = sorted({i for d in st.skills.values() for i in d["items"]})

    def val_tasks(s, n, salt=0):
        d = st.skills[s]
        pool = sorted(d["items"])
        r = random.Random(hash((s, st.round, salt)) & 0xffffffff)
        return [(s, d["definition"], i) for i in r.sample(pool, min(n, len(pool)))]

    def grp_tasks(s, n, salt=0):
        d = st.skills[s]
        uniq = sorted(d["items"])
        r = random.Random(hash((s, st.round, salt, "g")) & 0xffffffff)
        out = []
        pool = [i for i in all_items if i not in d["items"]]
        for _ in range(n):
            own = r.sample(uniq, 4)
            intr = r.choice(pool)
            five = own + [intr]
            r.shuffle(five)
            out.append((s, five, five.index(intr) + 1))
        return out

    def run(vt, gt):
        with ThreadPoolExecutor(max_workers=8) as ex:
            vres = list(ex.map(lambda t: (t[0], jd.validity_one(model, t[0], t[1], t[2])), vt))
            gres = list(ex.map(lambda t: (t[0], jd.group_one(model, t[1], t[2])), gt))
        return vres, gres

    vt = [t for s in targets for t in val_tasks(s, val_n)]
    gt = [t for s in targets if len(st.skills[s]["items"]) >= 4 for t in grp_tasks(s, grp_n)]
    vres, gres = run(vt, gt)
    acc = defaultdict(lambda: [0, 0, 0, 0])          # vy, vn, gy, gn
    for s, ok in vres:
        acc[s][0] += ok; acc[s][1] += 1
    for s, ok in gres:
        acc[s][2] += ok; acc[s][3] += 1

    if topup:
        more_v, more_g = [], []
        for s in targets:
            vy, vn, gy, gn = acc[s]
            gb = group_bar(st, s)
            if vn and VAL_BAND[0] <= vy / vn <= VAL_BAND[1]:
                more_v += val_tasks(s, val_n, salt=1)
            if gn and gb - 0.2 <= gy / gn <= gb + 0.2:
                more_g += grp_tasks(s, grp_n, salt=1)
        if more_v or more_g:
            print(f"  top-up: {len(more_v)} validity + {len(more_g)} group trials")
            vres, gres = run(more_v, more_g)
            for s, ok in vres:
                acc[s][0] += ok; acc[s][1] += 1
            for s, ok in gres:
                acc[s][2] += ok; acc[s][3] += 1

    for s in targets:
        vy, vn, gy, gn = acc[s]
        if vn:
            st.metrics["validity"][s] = vy / vn
            st.metrics["n_val"][s] = vn
        if gn:
            st.metrics["group"][s] = gy / gn
            st.metrics["n_grp"][s] = gn
        journal("measure", skill=s, validity=st.metrics["validity"].get(s),
                group=st.metrics["group"].get(s), n_val=vn, n_grp=gn, round=st.round)


def group_bar(st, s):
    return BIG_THR if len(st.skills[s]["items"]) > BIG_CLUSTER else ROUTE_THR


def route(st):
    """Return {skill: action} for substantive, unflagged skills."""
    acts = {}
    for s in sorted(st.substantive()):
        if st.skills[s]["flag"]:
            continue
        g = st.metrics["group"].get(s)
        v = st.metrics["validity"].get(s)
        gb = group_bar(st, s)
        size = len(st.skills[s]["items"])
        # group failure OR the topic-coherent-but-unnameable signature both mean
        # the structure is wrong; only a split can fix either
        pile_needed = ((g is not None and g <= gb) or
                       (v is not None and v <= LOWVAL_PILE and size >= LOWVAL_SIZE))
        rename_needed = violations(s) or (v is not None and v <= ROUTE_THR)
        cap = ATTEMPT_CAP if pile_needed else RENAME_CAP
        if st.skills[s]["attempts"] >= cap and (pile_needed or rename_needed):
            acts[s] = "irreducible"
        elif pile_needed:
            acts[s] = "pile"
        elif rename_needed:
            acts[s] = "rename"
    return acts


def repile(st, jd, s, qtext):
    d = st.skills[s]
    order = sorted(d["items"])
    print(f"  re-pile '{s}' ({len(order)} questions)")

    def parse_compact(txt, n):
        # aggregate by name: at scale the model emits one line per QUESTION with the
        # pile name repeated (object-locations diagnosis 2026-07-22); merging those
        # lines recovers the intended piles instead of rejecting 1000 singletons
        by_name, count = {}, defaultdict(int)
        for line in txt.splitlines():
            m = LINE.match(line)
            if not m:
                continue
            name = m.group(1).strip().lower()
            nums = [int(x) for x in m.group(3).split()
                    if x.isdigit() and 1 <= int(x) <= n and count[int(x)] < 2]
            for x in nums:
                count[x] += 1
            if not nums:
                continue
            if name in by_name:
                by_name[name]["nums"] += nums
            else:
                by_name[name] = {"name": name, "desc": m.group(2).strip()[:200],
                                 "nums": nums}
        piles = list(by_name.values())
        missing = [k for k in range(1, n + 1) if count[k] == 0]
        if len(missing) > max(3, 0.05 * n):
            return None
        for k in missing:
            piles.append({"name": f"unplaced question {k}", "desc": "", "nums": [k]})
        return None if degenerate(piles, n) else piles

    qs = "\n".join(f"{k+1}. {qtext.get(i, '')[:900]}" for k, i in enumerate(order))
    piles = parse_compact(jd._text(BIG, PSYS, COMPACT_U.format(
        n=len(order), rules=RULES, questions=qs)) or "", len(order))

    if piles is None and len(order) > 150:      # batched fallback
        rng = random.Random(hash((s, st.round)) & 0xffffffff)
        sample = rng.sample(range(1, len(order) + 1), 150)
        qs = "\n".join(f"{j+1}. {qtext.get(order[k-1], '')[:600]}" for j, k in enumerate(sample))
        txt = jd._text(BIG, PSYS, DISC_U.format(n=150, rules=RULES, questions=qs), max_out=15000)
        try:
            res = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            scheme = [{"name": str(p.get("name", "")).strip().lower() or "unnamed",
                       "desc": str(p.get("description", ""))[:200],
                       "nums": [sample[j-1] for j in p.get("questions", []) if 1 <= j <= 150]}
                      for p in res.get("piles", [])]
            scheme = [x for x in scheme if x["nums"]]
        except Exception:  # noqa: BLE001
            scheme = []
        if scheme and not degenerate(scheme, 150):
            tbl = "\n".join(f"{k+1}) {x['name']} | {x['desc']}" for k, x in enumerate(scheme))
            assigned = {k: list(x["nums"]) for k, x in enumerate(scheme)}
            placed = {n for v in assigned.values() for n in v}
            rest = [n for n in range(1, len(order) + 1) if n not in placed]

            def batch(chunk, force):
                fb = (" You MUST choose the closest pile; 0 is not allowed." if force
                      else " If none fits, use 0.")
                qb = "\n".join(f"{n}. {qtext.get(order[n-1], '')[:600]}" for n in chunk)
                txt = jd._text(SMALL, PSYS, ASSIGN_U.format(table=tbl, questions=qb, fallback=fb),
                               max_out=2000)
                out = {}
                for line in (txt or "").splitlines():
                    m = re.match(r"^\s*(\d+)\s*->\s*(\d+)", line)
                    if m:
                        out[int(m.group(1))] = int(m.group(2))
                return out

            unplaced = list(rest)
            for force in (False, True):
                still = []
                chunks = [unplaced[i:i+40] for i in range(0, len(unplaced), 40)]
                with ThreadPoolExecutor(max_workers=8) as ex:
                    for res_map, chunk in zip(ex.map(lambda c: batch(c, force), chunks), chunks):
                        for n in chunk:
                            p = res_map.get(n, 0)
                            (assigned[p-1].append(n) if 1 <= p <= len(scheme)
                             else still.append(n))
                unplaced = still
                if not unplaced:
                    break
            piles = [{"name": scheme[k]["name"], "desc": scheme[k]["desc"],
                      "nums": sorted(set(v))} for k, v in assigned.items() if v]
            for n in unplaced:
                piles.append({"name": f"unplaced question {n}", "desc": "", "nums": [n]})
            if degenerate(piles, len(order)):
                piles = None

    st.skills[s]["attempts"] += 1
    if piles is None:
        journal("repile_failed", skill=s, round=st.round)
        if st.skills[s]["attempts"] >= ATTEMPT_CAP:
            st.skills[s]["flag"] = "irreducible"
            journal("flag", skill=s, reason="irreducible", round=st.round)
        return []

    new = []
    del st.skills[s]
    for p in piles:
        items = {order[n-1] for n in p["nums"]}
        name = p["name"]
        base, k = name, 2
        while name in st.skills:
            name = f"{base} ({k})"; k += 1
        st.skills[name] = {"definition": p["desc"], "items": items,
                           "attempts": 0, "flag": None}
        key = (name, tuple(sorted(items)))
        if key in st.seen_states:
            st.skills[name]["flag"] = "irreducible"
            journal("flag", skill=name, reason="cycle", round=st.round)
        st.seen_states.add(key)
        new.append(name)
    journal("repile", skill=s, piles=[(n, len(st.skills[n]["items"])) for n in new],
            round=st.round)
    return new


def rename(st, jd, s, qtext):
    d = st.skills[s]
    r = random.Random(hash((s, st.round, "rn")) & 0xffffffff)
    ev = "\n".join(f"- {qtext.get(i, '')[:300]}"
                   for i in r.sample(sorted(d["items"]), min(12, len(d["items"]))))
    all_items = sorted({i for x in st.skills.values() for i in x["items"]})
    strangers = r.sample([i for i in all_items if i not in d["items"]], 10)
    own = r.sample(sorted(d["items"]), min(10, len(d["items"])))
    extra = ""
    for attempt in range(ATTEMPT_CAP):
        raw = jd._text(SMALL, NSYS, REWRITE_U.format(ev=ev, old=s, extra=extra), max_out=200)
        try:
            obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            obj = {"name": "", "definition": d["definition"]}
        cand = str(obj.get("name", "")).strip().lower()
        cdef = str(obj.get("definition", d["definition"]))[:200]
        vio = violations(cand)
        if vio:
            extra = f"6. Your previous attempt '{cand}' violated: {'; '.join(vio)}. Fix that.\n"
            continue
        with ThreadPoolExecutor(max_workers=8) as ex:
            own_verd = list(ex.map(lambda i: jd.validity_one(SMALL, cand, cdef, i), own))
            str_v = sum(ex.map(lambda i: jd.validity_one(SMALL, cand, cdef, i), strangers))
        own_v = sum(own_verd)
        gap = own_v / len(own) - str_v / len(strangers)
        if gap >= GAP_THR:
            del st.skills[s]
            name = cand
            base, k = name, 2
            while name in st.skills:
                name = f"{base} ({k})"; k += 1
            st.skills[name] = {"definition": cdef, "items": d["items"],
                               "attempts": d["attempts"] + 1, "flag": None}
            journal("rename", old=s, new=name, gap=round(gap, 2), round=st.round)
            return name
        missed = [i for i, ok in zip(own, own_verd) if not ok][:3]
        if missed and str_v <= 2:
            miss_txt = "\n".join(f"   - {qtext.get(i, '')[:160]}"
                                 for i in missed)
            extra = (f"6. Your previous attempt '{cand}' did NOT fit these questions "
                     f"from this very skill:\n{miss_txt}\n"
                     "Choose a name wide enough to cover them as well.\n")
        else:
            extra = (f"6. Your previous attempt '{cand}' was too vague: it also matched "
                     "unrelated questions. Be more specific about the object.\n")
    st.skills[s]["attempts"] += 1
    if st.skills[s]["attempts"] >= RENAME_CAP:
        st.skills[s]["flag"] = "name_unfixable"
        journal("flag", skill=s, reason="name_unfixable", round=st.round)
    journal("rename_failed", skill=s, round=st.round)
    return None


def cmd_dry(st):
    subst = st.substantive()
    acts = route(st)
    res = defaultdict(list)
    for s, a in acts.items():
        res[a].append(s)
    print(f"round {st.round}: {len(st.skills)} skills, {len(subst)} substantive")
    for a in ("pile", "rename", "irreducible"):
        print(f"\n{a.upper()} ({len(res[a])}):")
        for s in res[a]:
            g = st.metrics['group'].get(s)
            v = st.metrics['validity'].get(s)
            print(f"  g={'--' if g is None else f'{g:4.0%}'} v={'--' if v is None else f'{v:4.0%}'}"
                  f"  {s[:60]}  ({len(st.skills[s]['items'])}q)")
    flagged = [s for s in st.skills if st.skills[s]["flag"]]
    print(f"\nflagged: {len(flagged)}  residue(<4q): {len(st.skills) - len(subst)}")


def cmd_loop(st, jd, qtext):
    # instrument change: group scores must be re-measured with the hint once
    if st.round == 0:
        targets = sorted(st.substantive())
        print(f"round 0: re-measuring {len(targets)} substantive skills with frozen instrument")
        measure(st, jd, targets)
        st.save("round0")

    while st.round < ROUND_CAP:
        st.round += 1
        acts = route(st)
        work = {s: a for s, a in acts.items() if a in ("pile", "rename")}
        for s, a in acts.items():
            if a == "irreducible" and not st.skills[s]["flag"]:
                st.skills[s]["flag"] = "irreducible"
                journal("flag", skill=s, reason="attempts_exhausted", round=st.round)
        if not work:
            print(f"round {st.round}: fixed point reached")
            break
        n_pile_q = sum(len(st.skills[s]["items"]) for s, a in work.items() if a == "pile")
        est = n_pile_q * 250 * PRICE[BIG][0] / 1e6 + len(work) * 0.02
        print(f"round {st.round}: {sum(a=='pile' for a in work.values())} re-piles "
              f"({n_pile_q}q), {sum(a=='rename' for a in work.values())} renames, "
              f"est ${est:.2f}, spent ${spend['usd']:.2f}")

        changed = []
        for s, a in sorted(work.items(), key=lambda x: x[1]):       # piles first
            if a == "pile":
                changed += repile(st, jd, s, qtext)
            else:
                r = rename(st, jd, s, qtext)
                if r:
                    changed.append(r)
        cov = st.item_cover()
        orig = {pi["item_idx"] for pi in
                json.load(open(E / "oldtax_repaired_FINAL.json"))["per_item"]}
        lost = [i for i in orig if cov.get(i, 0) == 0]
        if lost:
            raise RuntimeError(f"invariant violated: {len(lost)} items lost coverage")

        fresh = [s for s in set(changed) if s in st.skills and len(st.skills[s]["items"]) >= 4
                 and not st.skills[s]["flag"]]
        if fresh:
            print(f"  measuring {len(fresh)} changed skills")
            measure(st, jd, fresh)
        st.save(f"round{st.round}")
        print(f"  snapshot round{st.round}.json  spent ${spend['usd']:.2f}")

    subst = st.substantive()
    g = [st.metrics["group"][s] for s in subst if s in st.metrics["group"]]
    v = [st.metrics["validity"][s] for s in subst if s in st.metrics["validity"]]
    flags = defaultdict(int)
    for s in st.skills:
        if st.skills[s]["flag"]:
            flags[st.skills[s]["flag"]] += 1
    print(f"\nFINAL: {len(st.skills)} skills, {len(subst)} substantive, "
          f"validity mean {sum(v)/len(v):.0%}, group mean {sum(g)/len(g):.0%}, "
          f"flags {dict(flags)}, spent ${spend['usd']:.2f}")


def cmd_certify(st, jd, qtext):
    """One-shot certification: BIG judge, fresh seed, larger n. Failures flag only."""
    anchf = P / "anchors.json"
    all_items = sorted({i for d in st.skills.values() for i in d["items"]})
    rng = random.Random(43)
    if not anchf.exists():
        subst = sorted(st.substantive())
        pairs = [(s, rng.choice(sorted(st.skills[s]["items"])))
                 for s in rng.sample(subst, 20)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            verd = list(ex.map(lambda p: jd.validity_one(
                BIG, p[0], st.skills[p[0]]["definition"], p[1]), pairs))
        json.dump({"pairs": [[s, i, v] for (s, i), v in zip(pairs, verd)]}, open(anchf, "w"))
        print("anchor set created (20 pairs)")
    else:
        anch = json.load(open(anchf))["pairs"]
        live = [(s, i, v) for s, i, v in anch if s in st.skills]
        with ThreadPoolExecutor(max_workers=8) as ex:
            now = list(ex.map(lambda p: jd.validity_one(
                BIG, p[0], st.skills[p[0]]["definition"], p[1]), live))
        flips = sum(1 for (s, i, v), n in zip(live, now) if v != n)
        print(f"anchors: {flips}/{len(live)} flipped")
        if live and flips / len(live) > 0.2:
            raise RuntimeError("judge drift detected: >20% anchor flips - do not certify")

    targets = sorted(s for s in st.substantive() if not st.skills[s]["flag"])
    print(f"certifying {len(targets)} skills (n_val={CERT_VAL_N}, n_grp={CERT_GRP_N})")
    st.round += 100                       # fresh sampling seeds, no collision with loop
    measure(st, jd, targets, model=BIG, val_n=CERT_VAL_N, grp_n=CERT_GRP_N, topup=False)

    # stranger gap for EVERY skill (replaces the removed banned-word list)
    print(f"stranger-gap test on {len(targets)} skills (n={GAP_N} own + {GAP_N} strangers)...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        gaps = dict(zip(targets, ex.map(lambda s: stranger_gap(st, jd, s, n=GAP_N), targets)))

    rows, fails = [], []
    for s in targets:
        g = st.metrics["group"].get(s)
        v = st.metrics["validity"].get(s)
        own, strg, gap = gaps[s]
        ok = ((g is None or g > group_bar(st, s)) and (v is None or v > ROUTE_THR)
              and gap >= GAP_THR)
        rows.append({"skill": s, "n_items": len(st.skills[s]["items"]),
                     "validity": v, "group": g, "own": own, "stranger": strg,
                     "gap": round(gap, 3), "certified": ok})
        if not ok:
            fails.append(s)
    flagged = [{"skill": s, "flag": st.skills[s]["flag"], "n_items": len(st.skills[s]["items"])}
               for s in sorted(st.skills) if st.skills[s]["flag"]]
    residue = sorted(s for s in st.skills if len(st.skills[s]["items"]) < 4)
    gm = [r["group"] for r in rows if r["group"] is not None]
    vm = [r["validity"] for r in rows if r["validity"] is not None]
    gp = [r["gap"] for r in rows if r["gap"] is not None]
    out = {"judge": BIG, "n_val": CERT_VAL_N, "n_grp": CERT_GRP_N, "n_gap": GAP_N,
           "mean_gap": sum(gp) / len(gp), "gap_thr": GAP_THR,
           "n_gap_fail": sum(1 for x in gp if x < GAP_THR),
           "rows": rows, "flagged": flagged, "n_residue": len(residue),
           "mean_validity": sum(vm) / len(vm), "mean_group": sum(gm) / len(gm),
           "n_certified": sum(r["certified"] for r in rows), "n_failed": len(fails),
           "spend_usd": round(spend["usd"], 3)}
    json.dump(out, open(P / "certified.json", "w"), indent=1)
    journal("certify", n_certified=out["n_certified"], n_failed=len(fails),
            mean_validity=out["mean_validity"], mean_group=out["mean_group"])
    print(f"certified {out['n_certified']}/{len(rows)}  validity {out['mean_validity']:.0%}  "
          f"group {out['mean_group']:.0%}  gap {out['mean_gap']:.0%} "
          f"({out['n_gap_fail']} below {GAP_THR:.0%})  failed(flag-only): {fails}")
    print(f"wrote {P/'certified.json'}  spent ${spend['usd']:.2f}")


def main():
    global P
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    fresh = tag == "fresh"
    if tag:
        P = E / ("pipeline_fresh" if fresh else f"pipeline_{tag}")
    P.mkdir(exist_ok=True)
    snap = latest_snapshot()
    if snap:
        st = State(json.load(open(snap)))
        print(f"resumed from {P.name}/{snap.name}")
    elif fresh:
        st = State.from_original()
        print(f"initialized FRESH from qmatrix_v2_K100 ({len(st.skills)} clusters)")
    else:
        st = State.from_final()
        # seed metrics from v3 (validity reusable; group re-measured in round 0)
        met = json.load(open(E / "oldtax_repaired_FINAL_metrics.json"))
        st.metrics["validity"].update({k: v for k, v in met["validity"].items()
                                       if k in st.skills and v is not None})
        print("initialized from oldtax_repaired_FINAL.json")
    if mode == "dry":
        cmd_dry(st)
        return
    qtext = {r["item_idx"]: " ".join(r["question_full_text"].split())
             for r in json.load(open(D / "item_full_text_recovered.json"))}
    jd = Judges(make_client(), qtext)
    if mode == "loop":
        cmd_loop(st, jd, qtext)
    elif mode == "certify":
        cmd_certify(st, jd, qtext)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
