"""Gemini client for the skill-induction pipeline.

Replaces the OpenAI client in taxonomy_pipeline.py. gemini-2.5-* are closed to
new keys, so the usable models are the 3.x line (checked 2026-09-01).

DESIGN NOTE, and the reason this is not a thin wrapper. The old client's
`_text` caught every exception and returned "". A dead API call was therefore
written to disk as a real answer meaning "no skill fits", indistinguishable
from a genuine rejection. That is how an out-of-credit account silently
produced a 100%-unassigned arm (the task list T-100). Here a failed call raises
GeminiError. Callers must decide what to record; they cannot mistake a failure
for an answer by accident.

Accounting is in tokens, not dollars: no pricing table is available for these
models, and a guessed one would be worse than none. Thinking tokens are counted
separately because on the flash models they dominate (gemini-3.6-flash spent
664 thinking tokens on a 58-token prompt; gemini-3.5-flash-lite spent 0).
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta/models"

# roles, not model names, so a model swap is one edit here
BULK = "gemini-3.5-flash-lite"      # extraction + labelling, ~19k calls, no thinking overhead
CODEBOOK = "gemini-3.1-pro-preview"  # step 2, the hard reasoning
JUDGE = "gemini-3.6-flash"           # step 6, must differ from CODEBOOK
EMBED = "gemini-embedding-2"         # 3072-dim

TOKEN_BUDGET = 40_000_000            # hard stop; ~2x a full expected run


class GeminiError(RuntimeError):
    """A call did not produce an answer. Never conflate with an empty answer."""


class BudgetExceeded(Exception):
    """The token budget is spent. Deliberately NOT a GeminiError: callers catch
    GeminiError and record a failed item, so raising one here would quietly turn
    the rest of the corpus into recorded failures instead of halting the run."""


def load_key() -> str:
    t = (Path.home() / ".cdmeval_gemini_key").read_text().strip()
    for pre in ("GEMINI_API_KEY=", "GOOGLE_API_KEY="):
        if t.startswith(pre):
            t = t[len(pre):]
    return t.strip().strip('"').strip("'")


class Gemini:
    def __init__(self, key: str | None = None, budget: int = TOKEN_BUDGET):
        self.key = key or load_key()
        self.budget = budget
        self.usage: dict[str, dict[str, int]] = {}
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()

    # ---------- accounting ----------
    def _record(self, model: str, u: dict) -> None:
        with self._lock:
            d = self.usage.setdefault(model, {"in": 0, "think": 0, "out": 0, "calls": 0})
            d["in"] += u.get("promptTokenCount", 0) or 0
            d["think"] += u.get("thoughtsTokenCount", 0) or 0
            d["out"] += u.get("candidatesTokenCount", 0) or 0
            d["calls"] += 1
            self.calls += 1
            if self.total_tokens > self.budget:
                raise BudgetExceeded(f"token budget exceeded: {self.total_tokens:,} > {self.budget:,}")

    @property
    def total_tokens(self) -> int:
        return sum(d["in"] + d["think"] + d["out"] for d in self.usage.values())

    def report(self) -> str:
        rows = [f"{m}: {d['calls']:,} calls, in {d['in']:,} think {d['think']:,} out {d['out']:,}"
                for m, d in sorted(self.usage.items())]
        return " | ".join(rows) + f" | total {self.total_tokens:,} tokens, {self.failures} failures"

    # ---------- transport ----------
    def _post(self, path: str, body: dict, timeout: int = 180) -> dict:
        req = urllib.request.Request(f"{API}/{path}?key={self.key}",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                try:
                    msg = json.loads(raw).get("error", {}).get("message", raw)[:200]
                except Exception:  # noqa: BLE001
                    msg = raw[:200]
                last = f"HTTP {e.code}: {msg}"
                if e.code not in (429, 500, 502, 503, 504):
                    break                       # not retryable: bad model, bad key, bad request
            except Exception as e:              # noqa: BLE001  timeouts, connection resets
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20) + random.random())
        with self._lock:
            self.failures += 1
        raise GeminiError(last or "unknown failure")

    # ---------- generation ----------
    def text(self, system: str, user: str, model: str = BULK, max_out: int = 2000,
             temperature: float = 0.0) -> str:
        body = {"contents": [{"parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_out}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        d = self._post(f"{model}:generateContent", body)
        self._record(model, d.get("usageMetadata", {}))
        cands = d.get("candidates") or []
        if not cands:
            raise GeminiError(f"no candidates (promptFeedback={d.get('promptFeedback')})")
        c = cands[0]
        parts = c.get("content", {}).get("parts", []) or []
        txt = "".join(p["text"] for p in parts if "text" in p)
        fin = c.get("finishReason")
        if not txt:
            # MAX_TOKENS with all budget spent on thinking lands here. Raising is
            # correct: an empty string is not an answer, and must not be stored as one.
            raise GeminiError(f"empty response (finishReason={fin})")
        if fin == "MAX_TOKENS":
            # Thinking tokens count against maxOutputTokens on the pro models, so a
            # large reasoning budget can leave the answer truncated mid-sentence.
            # Say so, rather than letting the caller report a parse error.
            u = d.get("usageMetadata", {})
            raise GeminiError(
                f"truncated at max_out: thinking={u.get('thoughtsTokenCount',0)} "
                f"output={u.get('candidatesTokenCount',0)}; raise max_out or shrink the batch")
        return txt

    @staticmethod
    def _repair(s: str) -> str:
        """Make LaTeX inside JSON strings survive json.loads.

        Models quote LaTeX in the rationale ("solve \\sqrt{x} via \\frac{1}{2}").
        43 of 9,523 Step 1 items failed on this, all MATH. Escaping only the
        *invalid* escapes is not enough: \\f and \\n are legal JSON escapes, so
        \\frac silently becomes a formfeed plus "rac" and the text is corrupted.

        So escape every backslash except the two that are structural inside a
        JSON string: \\" (a quoted quote) and \\\\ (an already-escaped backslash).
        A genuine \\n line break degrades to a literal backslash-n, which is
        acceptable in these one-line fields and much better than losing the item.
        """
        def fix(m):
            nxt = m.group(1)
            return m.group(0) if nxt in '"\\' else '\\\\' + nxt

        return re.sub(r'\\(.)', fix, s, flags=re.S)

    def json_obj(self, system: str, user: str, model: str = BULK, max_out: int = 2000) -> dict:
        raw = self.text(system, user, model=model, max_out=max_out)
        try:
            sub = raw[raw.index("{"):raw.rindex("}") + 1]
        except ValueError as e:
            raise GeminiError(f"no JSON object in response: {raw[:160]!r}") from e
        try:
            return json.loads(sub)
        except json.JSONDecodeError:
            try:
                return json.loads(self._repair(sub))
            except Exception as e:  # noqa: BLE001
                raise GeminiError(f"unparseable JSON: {raw[:160]!r}") from e

    # ---------- embeddings ----------
    def embed(self, texts: list[str], model: str = EMBED, dim: int | None = None):
        import numpy as np
        out = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            body = {"requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": x}]},
                 **({"outputDimensionality": dim} if dim else {})} for x in chunk]}
            d = self._post(f"{model}:batchEmbedContents", body)
            # embeddings bypassed accounting entirely, so report() understated the
            # run and the budget could never fire on this endpoint
            self._record(model, d.get("usageMetadata") or
                         {"promptTokenCount": sum(len(x) // 4 for x in chunk)})
            embs = d.get("embeddings")
            if not embs or len(embs) != len(chunk):
                raise GeminiError(f"embedding count mismatch: got {len(embs or [])} for {len(chunk)}")
            try:
                out.extend(e["values"] for e in embs)
            except (KeyError, TypeError) as exc:   # a malformed element must raise
                raise GeminiError(f"malformed embedding element: {exc}") from exc
        v = np.asarray(out, dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)
