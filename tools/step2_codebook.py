"""Step 2: build the codebook incrementally from the raw labels.

Follows the pipeline doc. Constant comparison: each raw skill is either mapped
to an existing code or becomes a new one, with embeddings used for RETRIEVAL
(pick the candidate codes to show) rather than for clustering.

Two permission levels, as the doc specifies:
  in-batch  append-only. map to an existing code, or add a new one. nothing
            destructive, so the codebook is not a moving target and the
            new-code rate stays interpretable.
  audit     every AUDIT_EVERY batches. destructive edits allowed, emitted as an
            explicit edit script rather than a rewritten codebook, so every
            operation carries a justification, is reversible, and implies an
            old_id -> new_id mapping that migrates assignments automatically.

Deviations from the doc, all agreed and recorded in
skill-induction-pipeline-annotated.md:

  1. Batches are STRATIFIED, not frequency-sorted. The doc's SGD framing needs
     each batch to resemble the corpus; sorting by frequency builds the codebook
     from head concepts and lets the tail arrive after the structure has set.
  2. The audit also receives definition-embedding neighbours as merge
     candidates. It still never sees raw data, so the blinding holds.
  3. The MERGE criterion requires the discriminating rule to be about the
     OPERATION. "One is about dancing, one about gifts" is a topic rule and does
     not count. Without this, v6's three swap-tracking skills survive forever.

SPLIT is recorded but not applied. A text-level audit cannot see member counts,
and the doc says the trustworthy split decisions happen after Step 4 when real
counts, coherence scores and a co-assignment matrix exist. Proposals are logged
for Step 5.

    python3 tools/step2_codebook.py smoke   # one batch, measures cost
    python3 tools/step2_codebook.py run     # full, resumable
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gemini import Gemini, GeminiError, BULK, CODEBOOK

REPO = Path(__file__).resolve().parent.parent
P = REPO / "cdm_exploration/experiments/pipeline_v7"

BATCH = int(os.environ.get("STEP2_BATCH", 150))   # doc says 100-200. Was 50, which
                           # made 313 API calls where ~105 do, with no recorded reason.

TOPK_MIN, TOPK_MAX = 6, 16   # candidate codes retrieved per raw label, scaled by how big
TOPK_PER = 80                # the codebook is: max(MIN, live // PER), capped at MAX.
                             # Measured over three 25-batch trials: codes CREATED was
                             # 285 / 284 / 285 at TOPK 6 / 6 / 12. Doubling retrieval
                             # depth changed creation not at all, and cost 27% more
                             # tokens and 29% more time. The model opens a new code
                             # when it judges no candidate matches, not because the
                             # match was absent from the list. But those trials only
                             # reach ~285 codes, where top-6 recall is already 96%;
                             # the full run reaches thousands, where it will not be.
                             # So stay cheap while small, widen as it grows.

AUDIT_EVERY = 10          # doc says every 5-10 batches; 12 sat outside it. Audits are cheap
                          # (one pro call) and merging duplicates sooner keeps them out of
                          # later retrieval, where they compete as candidates
AUDIT_OPS_MIN = 15        # the doc's flat cap, now a floor
AUDIT_OPS_PER = 1         # one operation per code created since the last audit.
                          # Was 10, which made the scaling dead code: ~56 codes were
                          # created per audit, 56//10 = 5, below the floor of 15, so
                          # the cap was 15 on all 31 audits and every one of them hit
                          # it. 1,727 codes were created against 465 operations.
CHURN_ABORT = 0.15        # if an audit moves this much, it is thrashing: stop the run
MERGE_SIM = 0.65          # definition-embedding threshold for nominating merge pairs.
                          # Was 0.80. Of 21 merges the first audit made, 10 were below
                          # 0.80 (min 0.66), i.e. the nomination list missed half of them
                          # and the audit had to find them by reading the codebook.
MERGE_PAIRS_SHOWN = 60    # lowering the threshold only helps if more pairs are shown
AUDIT_TIMEOUT = 900       # one audit call, up to `cap` operations with reasons. The
                          # default 180s fits cap=15 and not cap=150.
SAT_RATE = 0.02           # doc: new-code rate under ~2% ...
SAT_RUNS = 8              # ... averaged over this many batches. The doc says "consecutive
                          # batches", but at 50 labels the rate is far too noisy for that
                          # (trial: 2, 6, 0, 20, 4, 4, 18, 14, 24, 0 percent), so an
                          # all-of-N rule would never fire. Mean over the window instead.
SAT_CHURN = 0.05          # ... and low churn from the last audit
BATCH_CAP = 340           # so it cannot run forever; hitting this is reported

SYS = ("You maintain a codebook of cognitive skills for a test-item taxonomy. A skill is the "
       "smallest named mental operation a solver must master to answer a question correctly; "
       "the same skill appears across different topics and formats. You judge by the operation "
       "a solver performs, never by the subject matter or cover story.")

BATCH_U = """CURRENT CODEBOOK (the candidate codes retrieved for this batch):
{codes}

RAW SKILLS TO PROCESS:
{items}

For EACH raw skill, do exactly one of:
  - map it to one of ITS OWN candidate codes, if that code's operation is the same
  - propose a NEW code, if none of its candidates is the same operation

Two raw skills describing the same operation in different words are the same code.
Two raw skills on the same topic are NOT the same code unless the operation matches.

Return JSON only:
{{"decisions": [
  {{"i": 1, "action": "map", "code": "c_007"}},
  {{"i": 2, "action": "new", "name": "<lowercase verb phrase, 3-8 words>",
    "definition": "<one sentence, what the solver must do>",
    "include": ["<when this code applies>"],
    "exclude": ["<when it does not, and which code instead>"]}}
]}}"""

AUDIT_U = """Here is the current codebook. Each line: id | name | uses | definition.

{codes}

These pairs are near-identical by definition embedding and are the most likely merges:
{pairs}

Propose at most {cap} operations that most improve this codebook. Rules:

- MERGE two codes if you cannot write a rule that tells them apart BY THE OPERATION
  a solver performs. A rule based on subject matter, topic, or cover story does NOT
  count: "one is about dancing, one about gifts" is not a discriminator, because the
  solver does the same thing in both.
- RENAME a code whose name does not describe its definition.
- EDIT_DEF to add a missing exclusion clause.
- SPLIT a code that clearly covers two different operations, or whose share of
  total uses exceeds ~10% (an over-broad code that has swallowed the corpus).

Do not touch codes marked STABLE unless you state an explicit override reason.

Return JSON only:
{{"ops": [
  {{"op": "MERGE", "from": "c_012", "into": "c_007", "reason": "..."}},
  {{"op": "RENAME", "code": "c_003", "new_name": "...", "reason": "..."}},
  {{"op": "EDIT_DEF", "code": "c_005", "add_exclude": "...", "reason": "..."}},
  {{"op": "SPLIT", "code": "c_031", "discriminator": "...", "reason": "..."}}
]}}"""


def norm(x: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (x or "").lower().replace("_", " ")).split())


def load_labels():
    """Unique normalised raw labels with frequencies and an example source label."""
    rows = [json.loads(l) for l in (P / "raw_labels.jsonl").open()]
    freq, example = Counter(), {}
    for r in rows:
        for s in r.get("skills", []):
            k = norm(s.get("canonical_form") or s.get("label"))
            if not k:
                continue
            freq[k] += 1
            example.setdefault(k, s.get("label"))
    return freq, example


def stratified_batches(freq: Counter, size: int) -> list[list[str]]:
    """Deal frequency-sorted labels round-robin so every batch spans the range."""
    ordered = [k for k, _ in freq.most_common()]
    n = max(1, -(-len(ordered) // size))
    buckets: list[list[str]] = [[] for _ in range(n)]
    for i, k in enumerate(ordered):
        buckets[i % n].append(k)
    return buckets


class Codebook:
    def __init__(self):
        self.codes: dict[str, dict] = {}
        self.alias: dict[str, str] = {}      # old_id -> new_id, chained
        self.assign: dict[str, str] = {}     # raw label -> code id
        self.journal: list[dict] = []
        self.version = 0
        self._next = 1

    def resolve(self, cid: str) -> str:
        seen = set()
        while cid in self.alias and cid not in seen:
            seen.add(cid)
            cid = self.alias[cid]
        return cid

    def add(self, name, definition, include, exclude, batch):
        cid = f"c_{self._next:04d}"
        self._next += 1
        self.codes[cid] = {"name": name, "definition": definition,
                           "include": include or [], "exclude": exclude or [],
                           "exemplars": [], "created_batch": batch,
                           "unchanged_audits": 0, "stable": False}
        return cid

    def log(self, op, **kw):
        self.journal.append({"op": op, "version": self.version, **kw})

    def uses(self) -> Counter:
        c = Counter()
        for lab, cid in self.assign.items():
            c[self.resolve(cid)] += 1
        return c

    def save(self, tag, state=None):
        """`state` carries run progress so an interrupted run can resume.
        The journal is persisted because the run log's per-audit MERGE/SPLIT/
        RENAME counts are otherwise recoverable only from stdout."""
        blob = {"version": self.version, "codes": self.codes, "alias": self.alias,
                "assign": self.assign, "next": self._next, "journal": self.journal}
        if state:
            blob["state"] = state
        (P / f"codebook_{tag}.json").write_text(json.dumps(blob, indent=1))

    def load(self, tag):
        d = json.loads((P / f"codebook_{tag}.json").read_text())
        self.codes = d["codes"]; self.alias = d["alias"]; self.assign = d["assign"]
        self._next = d["next"]; self.version = d["version"]
        self.journal = d.get("journal", [])
        return d.get("state", {})


EX_KEEP = 6      # exemplars stored per code; the doc shows 2-3, extra give merges material


def code_line(cb, cid, full=True):
    c = cb.codes[cid]
    if not full:
        return f"{cid} | {c['name']}"
    inc = "; ".join(c["include"][:2]) or "-"
    exc = "; ".join(c["exclude"][:2]) or "-"
    ex = "; ".join(c.get("exemplars", [])[:3]) or "-"
    return (f"{cid} | {c['name']} | {c['definition']} | include: {inc} | "
            f"exclude: {exc} | examples: {ex}")


def embed_new_codes(g, cb, code_vec):
    """Embed every code that has no vector yet, in ONE call.

    Was a call per code. Eleven new codes meant eleven HTTP requests fired back
    to back, which is what tripped the embedding endpoint's rate limit and
    killed the 2026-09-07 run at batch 40. A 429 here now defers instead of
    raising: the code keeps no vector, so the next batch retries it, and the
    only cost is that it cannot be retrieved as a candidate until then.
    """
    todo = [c for c in cb.codes if c not in code_vec and c not in cb.alias]
    if not todo:
        return 0
    try:
        V = g.embed([f"{cb.codes[c]['name']}. {cb.codes[c]['definition']}" for c in todo])
    except GeminiError as e:
        print(f"    embed deferred for {len(todo)} codes: {e}", flush=True)
        return 0
    for c, v in zip(todo, V):
        code_vec[c] = v
    return len(todo)


def _batch_prompt(cb, labels, freq, lab_vec, code_vec):
    # code_vec can be empty while cb.codes is not: embed_new_codes defers the
    # whole batch on a 429, and np.stack([]) raises rather than returning empty
    if cb.codes and code_vec:
        ids = list(code_vec)
        M = np.stack([code_vec[i] for i in ids])
        sims = np.stack([lab_vec[l] for l in labels]) @ M.T
        k = min(TOPK_MAX, max(TOPK_MIN, len(ids) // TOPK_PER))
        cands = [[ids[j] for j in np.argsort(-sims[r])[:k]] for r in range(len(labels))]
    else:
        cands = [[] for _ in labels]

    shown = sorted({c for cs in cands for c in cs})
    codes_txt = "\n".join(code_line(cb, c) for c in shown) or "(empty, this is the first batch)"
    items_txt = "\n".join(
        f"{i+1}. {lab}  (appears {freq[lab]}x)  candidates: {', '.join(cands[i]) or 'none'}"
        for i, lab in enumerate(labels))
    return cands, codes_txt, items_txt


def run_batch(g, cb, labels, freq, bidx, lab_vec, code_vec, batch_model=CODEBOOK, pre=None):
    """One append-only batch: map to an existing code, or create a new one.

    `pre` is a response already obtained by plan_batch against an earlier
    snapshot. Applying it is still sequential, so codes created by an earlier
    batch in the same wave are visible to `cb.resolve` here even though the
    model did not see them. That is the whole trade: concurrency buys speed and
    costs some duplicate creation, which the audit removes.
    """
    cands, codes_txt, items_txt = _batch_prompt(cb, labels, freq, lab_vec, code_vec)
    obj = pre if pre is not None else g.json_obj(
        SYS, BATCH_U.format(codes=codes_txt, items=items_txt),
        model=batch_model, max_out=60000)

    n_new = n_map = 0
    for d in obj.get("decisions", []):
        i = int(d.get("i", 0)) - 1
        if not 0 <= i < len(labels):
            continue
        lab = labels[i]
        if d.get("action") == "map" and cb.resolve(d.get("code", "")) in cb.codes:
            cid = cb.resolve(d["code"])
            cb.assign[lab] = cid
            if len(cb.codes[cid]["exemplars"]) < EX_KEEP:
                cb.codes[cid]["exemplars"].append(lab)
            n_map += 1
        elif d.get("action") == "new" and d.get("name"):
            cid = cb.add(d["name"], d.get("definition", ""), d.get("include"),
                         d.get("exclude"), bidx)
            cb.assign[lab] = cid
            cb.codes[cid]["exemplars"].append(lab)
            n_new += 1
    # Labels the model simply did not mention must not be dropped. Flash omitted
    # 32 of 150 on the first real batch; without this they would silently never
    # be assigned to any code and would vanish from the taxonomy.
    missing = [l for l in labels if l not in cb.assign]
    if missing:
        items2 = "\n".join(
            f"{i+1}. {lab}  (appears {freq[lab]}x)  candidates: "
            f"{', '.join(cands[labels.index(lab)]) or 'none'}"
            for i, lab in enumerate(missing))
        try:
            obj2 = g.json_obj(SYS, BATCH_U.format(codes=codes_txt, items=items2),
                              model=batch_model, max_out=60000)
            for d in obj2.get("decisions", []):
                i = int(d.get("i", 0)) - 1
                if not 0 <= i < len(missing):
                    continue
                lab = missing[i]
                if d.get("action") == "map" and cb.resolve(d.get("code", "")) in cb.codes:
                    cid = cb.resolve(d["code"]); cb.assign[lab] = cid; n_map += 1
                    if len(cb.codes[cid]["exemplars"]) < EX_KEEP:
                        cb.codes[cid]["exemplars"].append(lab)
                elif d.get("action") == "new" and d.get("name"):
                    cid = cb.add(d["name"], d.get("definition", ""), d.get("include"),
                                 d.get("exclude"), bidx)
                    cb.assign[lab] = cid
                    cb.codes[cid]["exemplars"].append(lab)
                    n_new += 1
        except GeminiError as e:
            print(f"    retry for {len(missing)} omitted labels failed: {e}")
    still = [l for l in labels if l not in cb.assign]
    return n_new, n_map, still


def run_audit(g, cb, code_vec, bidx, created_since, cap=None):
    """Destructive edits, emitted as an explicit edit script. Returns churn.

    The doc caps this at a flat 15 operations. On our corpus that is far too
    small: the first audit hit exactly 15 (12 MERGE + 3 RENAME), i.e. it was cut
    off rather than finished, while ~144 codes had been created since the
    previous one. Creation outran consolidation ten to one. So the cap now
    scales with the number of codes created since the last audit.

    The doc gives anti-oscillation as the reason for the cap, but the real guard
    is the stable-code rule (a code unmodified across two audits cannot be
    changed without an explicit override), which is unchanged. Churn is the
    alarm: above CHURN_ABORT the audit is thrashing and the run stops.
    """
    cap = cap or max(AUDIT_OPS_MIN, created_since // AUDIT_OPS_PER)
    uses = cb.uses()
    # a 429-deferred embed leaves a live code with no vector; indexing it below
    # would raise KeyError and kill the run, which is what deferring exists to avoid
    live = [c for c in cb.codes if c not in cb.alias and c in code_vec]
    for c in live:
        cb.codes[c]["stable"] = cb.codes[c]["unchanged_audits"] >= 2 and uses[c] >= 4

    # our change 2: definition-embedding neighbours as merge candidates
    pairs = []
    if len(live) > 1:
        M = np.stack([code_vec[c] for c in live])
        S = M @ M.T
        np.fill_diagonal(S, 0)
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                if S[a, b] >= MERGE_SIM:
                    pairs.append((float(S[a, b]), live[a], live[b]))
    pairs.sort(reverse=True)
    pairs_txt = "\n".join(f"{x:.2f}  {a} ({cb.codes[a]['name']})  ~  {b} ({cb.codes[b]['name']})"
                          for x, a, b in pairs[:MERGE_PAIRS_SHOWN]) or "(none above threshold)"
    codes_txt = "\n".join(
        f"{c} | {cb.codes[c]['name']}{' [STABLE]' if cb.codes[c]['stable'] else ''} "
        f"| {uses[c]} | {cb.codes[c]['definition']} "
        f"| e.g. {'; '.join(cb.codes[c].get('exemplars', [])[:3]) or '-'}" for c in live)

    before = {lab: cb.resolve(cid) for lab, cid in cb.assign.items()}
    try:
        # a large cap means a long generation: at cap 150 this timed out five
        # times over 935s against the default 180s. The audit is one call per
        # round and there is no point retrying a request that cannot finish.
        obj = g.json_obj(SYS, AUDIT_U.format(codes=codes_txt, pairs=pairs_txt, cap=cap),
                         model=CODEBOOK, max_out=40000, timeout=AUDIT_TIMEOUT)
    except GeminiError as e:
        print(f"    audit failed, skipped: {e}")
        return None, Counter()      # None, never 0.0: a failed audit must not read
                                    # as zero churn and satisfy the saturation test

    cb.version += 1
    applied = Counter()
    for op in obj.get("ops", [])[:cap]:
        kind = op.get("op")
        try:
            if kind == "MERGE":
                a, b = cb.resolve(op["from"]), cb.resolve(op["into"])
                if a in cb.codes and b in cb.codes and a != b:
                    if cb.codes[a]["stable"] and not op.get("reason"):
                        continue
                    cb.alias[a] = b
                    cb.codes[b]["exemplars"] += cb.codes[a]["exemplars"][:2]
                    cb.log("MERGE", **{k: op.get(k) for k in ("from", "into", "reason")})
                    applied["MERGE"] += 1
            elif kind == "RENAME":
                c = cb.resolve(op["code"])
                if c in cb.codes and op.get("new_name"):
                    if cb.codes[c]["stable"] and not op.get("reason"):
                        continue          # "changing it after that requires an explicit override"
                    cb.log("RENAME", code=c, old=cb.codes[c]["name"],
                           new=op["new_name"], reason=op.get("reason"))
                    cb.codes[c]["name"] = op["new_name"]
                    applied["RENAME"] += 1
            elif kind == "EDIT_DEF":
                c = cb.resolve(op["code"])
                if c in cb.codes and op.get("add_exclude"):
                    if cb.codes[c]["stable"] and not op.get("reason"):
                        continue
                    cb.codes[c]["exclude"].append(op["add_exclude"])
                    cb.log("EDIT_DEF", code=c, add_exclude=op["add_exclude"],
                           reason=op.get("reason"))
                    applied["EDIT_DEF"] += 1
            elif kind == "SPLIT":
                # recorded only: a text-level audit cannot see member counts (doc, Step 5)
                cb.log("SPLIT_PROPOSED", code=cb.resolve(op.get("code", "")),
                       discriminator=op.get("discriminator"), reason=op.get("reason"))
                applied["SPLIT_proposed"] += 1
        except (KeyError, TypeError):
            continue

    after = {lab: cb.resolve(cid) for lab, cid in cb.assign.items()}
    churn = sum(1 for k in before if before[k] != after.get(k)) / max(1, len(before))
    for c in cb.codes:
        touched = any(j.get("code") == c or j.get("into") == c or j.get("from") == c
                      for j in cb.journal if j["version"] == cb.version)
        cb.codes[c]["unchanged_audits"] = 0 if touched else cb.codes[c]["unchanged_audits"] + 1
    return churn, applied


def main(mode="smoke", batch_model=CODEBOOK, resume=False):
    freq, example = load_labels()
    batches = stratified_batches(freq, BATCH)
    print(f"step 2: {len(freq):,} unique raw labels -> {len(batches)} stratified batches of ~{BATCH}")
    if mode == "smoke":
        batches = batches[:1]
        print("SMOKE: one batch only, measuring cost before committing to the rest")
    elif mode == "trial":
        n = int(os.environ.get("STEP2_TRIAL_BATCHES", 25))
        batches = batches[:n]
        print(f"TRIAL: {n} batches (~{n*BATCH:,} labels), runs "
              f"{n // AUDIT_EVERY} audit(s), then stops")

    g = Gemini()
    cb = Codebook()

    # Resume: a 314-batch run must not restart from zero if it is interrupted.
    start_batch, resumed_hist, resumed_carry, resumed_created = 0, [], [], 0
    if resume:
        try:
            st = cb.load("run")
            start_batch = st.get("next_batch", 0)
            resumed_hist = st.get("hist", [])
            resumed_carry = st.get("carry", [])
            resumed_created = st.get("created_since", 0)
            saved_batch = st.get("batch_size")
            if saved_batch is not None and saved_batch != BATCH:
                raise SystemExit(
                    f"cannot resume: this run was batched at {saved_batch} and BATCH is "
                    f"now {BATCH}. next_batch is an index into a partition that depends "
                    f"on BATCH, so resuming would skip a different set of labels "
                    f"entirely and they would never be processed.")
            print(f"  RESUME: {len(cb.codes)-len(cb.alias)} live codes, "
                  f"{len(cb.assign):,} labels assigned, continuing at batch {start_batch}")
        except FileNotFoundError:
            print("  RESUME requested but no codebook_run.json; starting fresh")

    # embed every unique label once; codebook entries are embedded as they appear
    cache = P / "label_emb.npz"
    keys = [k for k, _ in freq.most_common()]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        lab_vec = {k: v for k, v in zip(list(z["keys"]), z["vecs"])}
        print(f"  loaded {len(lab_vec):,} cached label embeddings")
    else:
        print(f"  embedding {len(keys):,} labels ...")
        V = g.embed(keys)
        np.savez_compressed(cache, keys=np.array(keys, dtype=object), vecs=V)
        lab_vec = {k: v for k, v in zip(keys, V)}
    code_vec: dict[str, np.ndarray] = {}

    hist, carry, t0 = resumed_hist, resumed_carry, time.time()
    created_since = resumed_created
    stop_reason = f"batch cap {BATCH_CAP}"
    if start_batch:                    # re-embed the codebook we just loaded
        embed_new_codes(g, cb, code_vec)
    # Parallel batch planning was tried and reverted on 2026-09-10. Two
    # independent defects, both found by review before it ran:
    #
    #   - a batch that picked up carried labels had its plan discarded and then
    #     immediately re-planned from the ORIGINAL label list, so every decision
    #     index was shifted by len(carry) and labels were bound to codes chosen
    #     for different labels. Silent corruption, not loss, and carry is
    #     non-empty on most batches.
    #   - the executor joined inside the loop, so planning never overlapped
    #     application. Steady-state concurrency was 1 while paying for
    #     PARALLEL-1 discarded pro calls per audit cycle: ~+70% tokens for zero
    #     wall-clock gain.
    #
    # Real prefetching needs futures held across iterations plus a lock around
    # the codebook snapshot, which is a redesign rather than an addition.
    for bi, labels in enumerate(batches):
        if bi < start_batch:
            continue
        if bi >= BATCH_CAP:
            break
        if carry:                      # labels the previous batch omitted
            labels = carry + labels
            carry = []
        try:
            n_new, n_map, still = run_batch(g, cb, labels, freq, bi, lab_vec,
                                            code_vec, batch_model)
            carry.extend(still)
        except GeminiError as e:
            print(f"  batch {bi}: FAILED {e}")
            carry.extend(labels)          # a failed batch must not lose its labels either
            continue
        created_since += n_new
        rate = n_new / max(1, len(labels))
        hist.append(rate)
        embed_new_codes(g, cb, code_vec)
        print(f"  batch {bi:3d}: {n_map:3d} mapped, {n_new:3d} new ({rate:5.1%}), "
              f"{len(still):2d} carried | codes {len(cb.codes)-len(cb.alias):4d} | "
              f"{g.total_tokens:,} tok | {time.time()-t0:.0f}s", flush=True)

        if mode == "run":          # only a full run leaves resumable state
            cb.save("run", {"next_batch": bi + 1, "hist": hist, "carry": carry,
                            "created_since": created_since, "mode": mode,
                            "batch_size": BATCH})

        churn = None
        if (bi + 1) % AUDIT_EVERY == 0:
            churn, applied = run_audit(g, cb, code_vec, bi, created_since)
            if churn is not None:
                created_since = 0      # only a successful audit consumes the backlog
            for cid in list(code_vec):
                if cid in cb.alias:
                    code_vec.pop(cid, None)
            embed_new_codes(g, cb, code_vec)
            if churn is None:
                print(f"    audit v{cb.version}: FAILED, skipped (no churn measured); "
                      f"saturation cannot be judged this cycle")
                continue          # no save, no saturation test, keep created_since
            print(f"    audit v{cb.version}: {dict(applied)} churn {churn:.1%} "
                  f"| codes {len(cb.codes)-len(cb.alias)}")
            if churn > CHURN_ABORT:
                stop_reason = f"audit thrashing: churn {churn:.1%} > {CHURN_ABORT:.0%}"
                cb.save(f"v{cb.version}")
                break
            cb.save(f"v{cb.version}")

            # the doc's rule: new-code rate low for a few batches AND low churn
            window = hist[-SAT_RUNS:]
            if len(hist) >= SAT_RUNS and sum(window) / len(window) < SAT_RATE \
               and churn < SAT_CHURN:
                stop_reason = (f"saturated: mean new-code rate "
                               f"{sum(window)/len(window):.1%} over {SAT_RUNS} batches "
                               f"< {SAT_RATE:.0%}, churn {churn:.1%}")
                break

    if stop_reason == f"batch cap {BATCH_CAP}" and len(hist) < BATCH_CAP:
        stop_reason = f"corpus exhausted after {len(batches)} batches"
    cb.save({"run": "final", "trial": "trial", "smoke": "smoke"}[mode])
    uses = cb.uses()
    live = [c for c in cb.codes if c not in cb.alias]
    sizes = sorted((uses[c] for c in live), reverse=True)
    print(f"\nstopped: {stop_reason}")
    print(f"batches run: {len(hist)}   codes: {len(live)}   "
          f"labels assigned: {len(cb.assign):,}/{len(freq):,}")
    if sizes:
        print(f"code sizes (raw-label mentions): max {sizes[0]}, median {sizes[len(sizes)//2]}, "
              f"singletons {sum(1 for s in sizes if s == 1)}")
    print(f"new-code rate by batch: {[f'{r:.0%}' for r in hist]}")
    print(f"journal ops: {dict(Counter(j['op'] for j in cb.journal))}")
    print(f"usage: {g.report()}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--resume"]
    main(args[0] if args else "smoke",
         args[1] if len(args) > 1 else CODEBOOK,
         resume="--resume" in sys.argv)
