#!/usr/bin/env python3
"""One place that decides whether a cached code-definition vector may be used.

Six steps read `code_def_emb.npz` and only Step 3 ever checked whether the
vectors still describe the definitions on disk. Step 8 rewrites definitions, so
every step running after it was measuring similarity on superseded text. The
concrete damage: Step 9 nominated duplicate pairs over 379 codes of which 271
(72%) had been re-worded, and missed six pairs a later judge called identical,
including two codes that Step 8 had renamed to the *same string*. One of them
had been re-defined from "assigning biological organisms to taxonomic ranks" to
"comparing a source text to an altered translation"; Step 9 hunted its
duplicates using the biology sentence.

The cache carried a `digest` field that said `stale-after-step5` at the time.
Nothing outside Step 3 read it.

Vectors are keyed per code on a hash of the exact text embedded, so a definition
that changed re-embeds only itself and the rest of the bank is reused.
"""
import hashlib

import numpy as np

CACHE = "code_def_emb.npz"


def code_text(cb, c):
    """The doc retrieves on "the code's definition + exemplars", not the
    definition alone. Pass 1 has no item exemplars yet, so it uses the
    raw-label ones."""
    e = cb["codes"][c]
    ex = "; ".join(e.get("exemplars", [])[:3])
    return f"{e['name']}. {e['definition']}" + (f" Examples: {ex}" if ex else "")


def text_hash(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def read_cache(path):
    """(ids, vecs, hashes) or None. A cache written before per-code hashing
    cannot be vouched for and is reported as absent rather than trusted."""
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    if "hashes" not in z.files:
        return None
    return [str(x) for x in z["ids"]], z["vecs"], [str(x) for x in z["hashes"]]


def load(path, cb, live, g=None, verbose=True):
    """Vectors for `live`, in that order, guaranteed to match the current text.

    Any cached vector whose text hash still matches is reused; the rest are
    embedded. Without a client, a stale or incomplete cache raises instead of
    returning vectors the caller cannot vouch for. That is the whole point: the
    previous failure was silent reuse, not a crash.
    """
    texts = [code_text(cb, c) for c in live]
    want = [text_hash(t) for t in texts]
    have = {}
    cached = read_cache(path)
    if cached:
        ids, vecs, hashes = cached
        have = {c: (h, vecs[i]) for i, (c, h) in enumerate(zip(ids, hashes))}

    fresh = [i for i, c in enumerate(live) if c in have and have[c][0] == want[i]]
    missing = [i for i in range(len(live)) if i not in set(fresh)]
    if missing and g is None:
        raise RuntimeError(
            f"{len(missing)} of {len(live)} code embeddings are stale or absent and "
            f"no embedding client was given. Pass one, or re-run Step 3. Refusing to "
            f"return vectors that do not describe the definitions on disk.")
    if verbose:
        print(f"  code embeddings: {len(fresh):,} reused, {len(missing):,} to embed")

    dim = have[live[fresh[0]]][1].shape[0] if fresh else None
    if missing:
        newV = np.stack(g.embed([texts[i] for i in missing]))
        dim = newV.shape[1]
    V = np.zeros((len(live), dim), dtype="float32")
    for i in fresh:
        V[i] = have[live[i]][1]
    for k, i in enumerate(missing):
        V[i] = newV[k]

    merged = dict(zip(live, zip(want, V)))
    for c, (h, v) in have.items():                 # keep vectors for codes not in `live`
        merged.setdefault(c, (h, v))
    ids = list(merged)
    np.savez_compressed(
        path,
        ids=np.array(ids, dtype=object),
        vecs=np.stack([merged[c][1] for c in ids]),
        hashes=np.array([merged[c][0] for c in ids], dtype=object),
    )
    return V


def normed(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)
