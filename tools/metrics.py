#!/usr/bin/env python3
"""Pure functions for every number this pipeline reports.

Extracted so they can be unit-tested against hand-computed values. Previously
these calculations lived inline inside the step scripts, where the only thing a
test could do was assert that a line of source existed. That is how a missing
`import os` passed a green suite and then killed a run.

Nothing here does I/O or calls a model.
"""
from collections import Counter


# ---------- coherence ----------
def coherence_precision(verdicts, n_sent):
    """Share of SENT questions the judge said match.

    Indexed by the echoed `i`, never by position: a judge that skips question 3
    would otherwise shift every later verdict onto the wrong question. An
    unreturned verdict counts against the code rather than shrinking the
    denominator, which is what made a code sent 10 and given 3 matches score 1.0.

    Returns (precision, n_returned, n_missing).
    """
    got = {}
    for v in verdicts or []:
        if not isinstance(v, dict):
            continue
        try:
            i = int(v.get("i", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_sent:
            got[i] = bool(v.get("match"))
    if n_sent <= 0:
        return None, 0, 0
    return sum(got.values()) / n_sent, len(got), n_sent - len(got)


# ---------- stability ----------
def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------- null control ----------
def separation(accept_real, n_real, accept_decoy, n_decoy):
    """Accept rate on real minus accept rate on decoys.

    Near zero means the answers do not distinguish the two, so a high raw accept
    rate would be agreeableness rather than skill.
    """
    if not n_real or not n_decoy:
        return None
    return accept_real / n_real - accept_decoy / n_decoy


# ---------- merge application ----------
def apply_merges(accepted, rejected, sizes, ceiling, order="smallest"):
    """Union-find over accepted pairs. Returns (alias, stats).

    Merging is NOT transitive: A~B and B~C accepted does not license A~C, and if
    a judge explicitly rejected A~C then joining them is a contradiction. Applying
    accepted pairs greedily did exactly that, through chains up to 8 hops.

    A union is refused if it would breach `ceiling`, and blocked if it would put
    an explicitly rejected pair into one component.

    `order`: "smallest" merges small pairs first so the undersized tail is
    consolidated before the ceiling budget is spent; "largest" is the old
    behaviour and is kept only so the difference can be tested.
    """
    parent = {c: c for c in sizes}
    members = {c: {c} for c in sizes}
    size = dict(sizes)
    rej = {tuple(sorted(p)) for p in rejected}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    key = (lambda p: max(sizes.get(p[0], 0), sizes.get(p[1], 0)))
    pairs = sorted(accepted, key=key, reverse=(order == "largest"))

    applied = refused = blocked = 0
    for x, y in pairs:
        if x not in parent or y not in parent:
            continue
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        if size[rx] + size[ry] > ceiling:
            refused += 1
            continue
        if any(tuple(sorted((u, v))) in rej for u in members[rx] for v in members[ry]):
            blocked += 1
            continue
        big, small = (rx, ry) if size[rx] >= size[ry] else (ry, rx)
        parent[small] = big
        members[big] |= members[small]
        members.pop(small, None)
        size[big] += size[small]
        size[small] = 0
        applied += 1

    alias = {c: find(c) for c in sizes if find(c) != c}
    return alias, {"applied": applied, "refused_ceiling": refused, "blocked_rejected": blocked,
                   "components": sum(1 for v in Counter(find(c) for c in sizes).values() if v > 1)}


def resolve(alias, cid, _max=10_000):
    """Follow an alias chain to its end. Cycle-safe."""
    seen = set()
    while cid in alias and cid not in seen and len(seen) < _max:
        seen.add(cid)
        cid = alias[cid]
    return cid


# ---------- distribution against Step 0's decisions ----------
def size_stats(counts, floor=20, ceiling_frac=0.05, n_items=None):
    sz = sorted(counts.values(), reverse=True)
    if not sz:
        return {}
    ceil_n = ceiling_frac * n_items if n_items else float("inf")
    return {"codes": len(sz), "max": sz[0], "median": sz[len(sz) // 2],
            "below_floor": sum(1 for s in sz if s < floor),
            "above_ceiling": sum(1 for s in sz if s > ceil_n),
            "total_assignments": sum(sz)}


def identifiable(rows):
    """Codes with at least one question that measures them ALONE.

    A skill that never appears by itself is not separately identifiable: if a
    two-skill question is failed, the responses cannot say which skill was missing.
    """
    alone = set()
    for r in rows:
        cs = [x["code"] for x in r.get("assigned", [])]
        if len(cs) == 1:
            alone.add(cs[0])
    return alone
