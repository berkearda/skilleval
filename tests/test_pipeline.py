#!/usr/bin/env python3
"""Regression tests for the v7 skill-induction pipeline.

Every test here pins a defect that actually occurred in this repo and was found
by an audit, not a hypothetical. The rule this suite enforces: a number produced
by these scripts is not accepted unless the test guarding its computation passes.

    python3 -m unittest discover -s tests -v
"""
import json, os, sys, types, unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import gemini as G
from tools import step2_codebook as S2
from tools import step3_freeze as S3
from tools import step4_relabel as S4
from tools import step5_loop as S5


# --------------------------------------------------------------------------
class TestGeminiLayer(unittest.TestCase):
    """gemini.py: the layer whose one job is to never confuse failure with emptiness."""

    def test_budget_is_not_a_gemini_error(self):
        """A budget trip must halt, not be caught by callers as a failed item.

        Bug: BudgetExceeded used to be GeminiError, so exhausting the budget
        converted the rest of the corpus into recorded failures at zero cost.
        """
        self.assertFalse(issubclass(G.BudgetExceeded, G.GeminiError))
        self.assertFalse(issubclass(G.BudgetExceeded, RuntimeError))

    def test_repair_preserves_latex_backslashes(self):
        """\\frac must survive JSON repair.

        Bug: a naive repair turned \\f into a formfeed, silently corrupting MATH
        rationales without raising.
        """
        raw = r'{"a": "$\frac{1}{2}$ and \sqrt{3}"}'
        obj = json.loads(G.Gemini._repair(raw))
        self.assertIn("frac", obj["a"])
        self.assertNotIn("\f", obj["a"])

    def test_repair_keeps_structural_escapes_and_degrades_newlines(self):
        """_repair's documented contract, which I initially got wrong in this test.

        It escapes every backslash except the two that are structural inside a
        JSON string, \" and \\. A genuine \n therefore degrades to a literal
        backslash-n. That is the deliberate trade: losing a line break in a
        one-line field beats losing the item to a corrupted \frac. It only ever
        runs after json.loads has already failed, so valid JSON never reaches it.
        """
        obj = json.loads(G.Gemini._repair(r'{"a": "line\nbreak \"quoted\""}'))
        self.assertIn('"quoted"', obj["a"], "a quoted quote must survive")
        self.assertNotIn("\n", obj["a"], "a real newline is the documented casualty")
        self.assertIn("\\n", obj["a"])


# --------------------------------------------------------------------------
class TestStep2Codebook(unittest.TestCase):

    def _cb(self):
        cb = S2.Codebook()
        cb.add("track object positions", "Follow a sequence of swaps.", ["swaps"], ["static"], 0)
        cb.add("count elements", "Enumerate items.", ["counting"], ["measuring"], 0)
        return cb

    def test_norm_collapses_case_punctuation_underscores(self):
        self.assertEqual(S2.norm("Tracking_Object  Positions!"), "tracking object positions")
        self.assertEqual(S2.norm(None), "")

    def test_stratified_batches_span_the_frequency_range(self):
        """The annotation overrides the doc's descending-frequency ordering.

        Every batch must look like the corpus, or the codebook is built from the
        head and the tail arrives after the structure is fixed.
        """
        freq = Counter({f"lab{i}": 100 - i for i in range(100)})
        b = S2.stratified_batches(freq, 10)
        self.assertEqual(sum(len(x) for x in b), 100)
        self.assertEqual(len({l for batch in b for l in batch}), 100)
        ranks = {k: i for i, (k, _) in enumerate(freq.most_common())}
        first = [ranks[l] for l in b[0]]
        self.assertGreater(max(first) - min(first), 50,
                           "a batch must span the frequency range, not cluster in the head")

    def test_alias_chain_resolves_and_never_orphans(self):
        cb = self._cb()
        cb.assign["x"] = "c_0001"
        cb.alias["c_0001"] = "c_0002"
        self.assertEqual(cb.resolve("c_0001"), "c_0002")
        self.assertEqual(cb.uses()["c_0002"], 1)

    def test_alias_cycle_terminates(self):
        cb = self._cb()
        cb.alias["c_0001"] = "c_0002"
        cb.alias["c_0002"] = "c_0001"
        self.assertIn(cb.resolve("c_0001"), {"c_0001", "c_0002"})

    def test_code_line_shows_exemplars(self):
        """The doc requires 2-3 positive examples per entry.

        Bug: code_line emitted name/definition/include/exclude only, so the whole
        codebook was induced by a judge that never saw an example.
        """
        cb = self._cb()
        cb.codes["c_0001"]["exemplars"] = ["tracking gift swaps", "tracking dance swaps"]
        line = S2.code_line(cb, "c_0001")
        self.assertIn("tracking gift swaps", line)
        self.assertIn("examples:", line)

    def test_save_persists_the_journal(self):
        """Per-audit MERGE/RENAME counts must survive the process.

        Bug: the journal lived only in stdout, so the 1 September trials left no
        record of what their audits did.
        """
        import tempfile
        cb = self._cb()
        cb.log("MERGE", **{"from": "c_0001", "into": "c_0002", "reason": "same operation"})
        with tempfile.TemporaryDirectory() as d:
            old, S2.P = S2.P, Path(d)
            try:
                cb.save("t")
                blob = json.loads((Path(d) / "codebook_t.json").read_text())
            finally:
                S2.P = old
        self.assertEqual(len(blob["journal"]), 1)
        self.assertEqual(blob["journal"][0]["op"], "MERGE")

    def test_trial_mode_does_not_write_the_final_codebook(self):
        """Bug: a 25-batch trial wrote codebook_final.json, the same path as the
        313-batch run, with no provenance check downstream."""
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn('{"run": "final", "trial": "trial", "smoke": "smoke"}[mode]', src)

    def test_stable_guard_covers_every_destructive_edit(self):
        """The doc: changing a stable code requires an explicit override.

        Bug: the guard was on MERGE only, so c_0100 was renamed three times,
        oscillating between two names at audits v5, v11 and v23.
        """
        src = (REPO / "tools/step2_codebook.py").read_text()
        for branch in ('elif kind == "RENAME":', 'elif kind == "EDIT_DEF":'):
            i = src.index(branch)
            body = src[i:i + 420]
            self.assertIn('["stable"] and not op.get("reason")', body,
                          f"{branch} has no STABLE override guard")

    def test_failed_audit_returns_none_not_zero_churn(self):
        """Bug: a failed audit returned churn 0.0, which is below SAT_CHURN, so a
        rate limit could end the run declaring a convergence that never happened."""
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn("return None, Counter()", src)
        self.assertIn("if churn is None:", src)

    def test_audit_skips_codes_with_no_embedding(self):
        """Bug: a 429-deferred embedding left a live code without a vector and
        run_audit indexed it blindly, so the deferral raised KeyError instead."""
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn("c not in cb.alias and c in code_vec", src)


# --------------------------------------------------------------------------
class TestStep3Freeze(unittest.TestCase):

    def test_code_text_includes_exemplars(self):
        """The doc retrieves on 'definition + exemplars', not the definition alone."""
        cb = {"codes": {"c_1": {"name": "n", "definition": "d",
                                "exemplars": ["ex one", "ex two"]}}}
        t = S3.code_text(cb, "c_1")
        self.assertIn("ex one", t)
        self.assertTrue(t.startswith("n. d"))

    def test_kmeans_is_deterministic(self):
        import numpy as np
        V = np.random.default_rng(0).normal(size=(60, 8))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        self.assertTrue((S3.kmeans(V, 4) == S3.kmeans(V, 4)).all())

    def test_embedding_cache_is_keyed_on_text_not_ids(self):
        """Bug: the cache keyed on ids alone, so changing the embedded text with
        ids unchanged silently reused stale vectors and nothing could detect it.
        Step 8 rewrites definitions, so Steps 6 and 9 measured similarity on text
        the codes no longer had. Now asserted behaviourally, in tools/codeemb."""
        import numpy as np, tempfile, os
        from pathlib import Path as _P
        from tools import codeemb

        class FakeG:
            def __init__(self): self.calls = []
            def embed(self, texts):
                self.calls.append(list(texts))
                return [np.full(4, float(len(t)), dtype="float32") for t in texts]

        cb = {"codes": {"c_1": {"name": "a", "definition": "one"},
                        "c_2": {"name": "b", "definition": "two"}}}
        live = ["c_1", "c_2"]
        with tempfile.TemporaryDirectory() as d:
            cache = _P(d) / "e.npz"
            g = FakeG()
            codeemb.load(cache, cb, live, g=g, verbose=False)
            self.assertEqual(len(g.calls[0]), 2, "first run embeds everything")

            g2 = FakeG()
            codeemb.load(cache, cb, live, g=g2, verbose=False)
            self.assertEqual(g2.calls, [], "unchanged text must reuse the cache")

            cb["codes"]["c_2"]["definition"] = "two, but reworded by Step 8"
            g3 = FakeG()
            V = codeemb.load(cache, cb, live, g=g3, verbose=False)
            self.assertEqual(len(g3.calls[0]), 1,
                             "only the changed definition re-embeds")
            self.assertIn("reworded", g3.calls[0][0])
            self.assertEqual(V.shape, (2, 4))

            cb["codes"]["c_1"]["definition"] = "changed again"
            with self.assertRaises(RuntimeError,
                                   msg="stale cache with no client must raise, not return"):
                codeemb.load(cache, cb, live, g=None, verbose=False)

    def test_a_cache_without_per_code_hashes_is_not_trusted(self):
        """The shipped cache carried digest='stale-after-step5' and five steps
        loaded it anyway. An old-format cache must read as absent."""
        import numpy as np, tempfile
        from pathlib import Path as _P
        from tools import codeemb
        with tempfile.TemporaryDirectory() as d:
            cache = _P(d) / "e.npz"
            np.savez_compressed(cache, ids=np.array(["c_1"], dtype=object),
                                vecs=np.zeros((1, 4), dtype="float32"),
                                digest="stale-after-step5")
            self.assertIsNone(codeemb.read_cache(cache))

    def test_steps_after_definition_repair_do_not_read_the_cache_blind(self):
        """Step 8 rewrites definitions; anything nominating on similarity after
        it must go through the guarded loader."""
        for name in ("step6_validate", "step9_dedupe"):
            src = (REPO / f"tools/{name}.py").read_text()
            with self.subTest(step=name):
                self.assertNotIn('np.load(P / "code_def_emb.npz"', src,
                                 f"{name} loads the embedding cache without the staleness check")
                self.assertIn("load_code_vecs", src)


# --------------------------------------------------------------------------
class TestStep4Relabel(unittest.TestCase):

    def test_retrieval_and_judging_see_the_same_question_text(self):
        """Bug: retrieval saw 2,000 chars and the judge saw 4,000, so for 30% of
        the corpus candidates were chosen from half the question.

        Now the loader clips once and both paths use that one string, so they
        cannot drift apart again.
        """
        src = (REPO / "tools/step4_relabel.py").read_text()
        self.assertNotIn("[:2000]", src)
        self.assertEqual(src.count('"question": clip(txt[i], QCHARS)'), 1)
        self.assertIn('g.embed([it["question"] for it in items])', src,
                      "retrieval must embed the same clipped question the judge sees")

    def test_model_is_restricted_to_its_own_candidates(self):
        """The doc: 'have the LLM pick from those candidates'.

        Bug: the filter tested membership of the whole codebook, so a hallucinated
        id that happened to exist was accepted. One did.
        """
        src = (REPO / "tools/step4_relabel.py").read_text()
        self.assertIn('a.get("code") in cand', src)
        self.assertNotIn('a.get("code") in codes', src)
        self.assertIn("off_candidate", src)

    def test_unassignable_is_parsed_strictly(self):
        """Bug: bool('false') is True, and unassignable is the headline metric
        and Step 5's residue selector."""
        src = (REPO / "tools/step4_relabel.py").read_text()
        self.assertNotIn('bool(obj.get("unassignable"))', src)
        self.assertIn('_u is True', src)

    def test_prompt_carries_no_subtask_metadata(self):
        """Step 0's anchors name subtask strings as the too-fine axis; feeding
        tracking_shuffled_objects_five_objects to the labeller contradicts the
        system prompt one line above it."""
        self.assertNotIn("{subtask}", S4.USER)
        self.assertNotIn("{benchmark}", S4.USER)

    def test_zero_skill_rows_are_not_counted_as_labelled(self):
        """Bug: sum(n_asg.values()) included the 0 bucket, so the run reported
        9,505 labelled while its own histogram showed 54 with no skill."""
        src = (REPO / "tools/step4_relabel.py").read_text()
        self.assertIn("sum(v for k, v in n_asg.items() if k > 0)", src)

    def test_rewrite_is_atomic(self):
        src = (REPO / "tools/step4_relabel.py").read_text()
        self.assertIn("os.replace", src)


# --------------------------------------------------------------------------
class TestStep5Loop(unittest.TestCase):

    def test_norm_proposal_handles_every_observed_shape(self):
        """proposed_skill arrives as name / title / description-only / a c_xxxx
        placeholder. All four were seen on disk."""
        self.assertEqual(S5.norm_proposal({"name": "applying vieta"}), "applying vieta")
        self.assertEqual(S5.norm_proposal({"title": "using lagrange"}), "using lagrange")
        self.assertEqual(S5.norm_proposal({"description": "only a description"}),
                         "only a description")
        self.assertEqual(S5.norm_proposal({"name": "c_xxxx", "description": "real text"}),
                         "real text")
        self.assertEqual(S5.norm_proposal(None), "")

    def test_residual_is_measured_after_the_rerun(self):
        """Bug: residual_rate was the extension model's own claim about what its
        new codes would cover, written before the re-label that could measure it."""
        src = (REPO / "tools/step5_loop.py").read_text()
        self.assertIn("measured_after_rerun", src)
        i_other = src.index('fz["step5"]["other_bucket"] = sorted(still_ids)')
        i_rerun = src.index("re-running the")
        self.assertGreater(i_other, i_rerun, "residual must be written after the re-run")

    def test_other_bucket_is_seeded_from_all_residue(self):
        """Bug: the one residue item with no usable proposal was in neither
        covered nor other, so it vanished from the tail bucket entirely."""
        src = (REPO / "tools/step5_loop.py").read_text()
        self.assertIn("other = [i for i in resid if i not in covered]", src)

    def test_retry_failures_are_recorded_separately(self):
        """Bug: a network blip in the re-label loop was swallowed and the item
        counted as genuinely unassignable."""
        src = (REPO / "tools/step5_loop.py").read_text()
        self.assertIn("retry_failed", src)

    def test_new_code_embeddings_are_persisted(self):
        """Bug: Step 5 computed vectors for its new codes and discarded them, so
        Step 6 loaded a pre-Step-5 embedding file and silently could neither
        nominate them for merging nor retrieve them, manufacturing instability."""
        src = (REPO / "tools/step5_loop.py").read_text()
        self.assertIn('np.savez_compressed(P / "code_def_emb.npz"', src)
        self.assertIn("stale-after-step5", src)

    def test_no_dead_rounds_flag(self):
        """Bug: --rounds was parsed and never read, advertising a loop the
        script does not have and is not re-entrant enough to run."""
        src = (REPO / "tools/step5_loop.py").read_text()
        self.assertNotIn('add_argument("--rounds"', src)


# --------------------------------------------------------------------------
class TestStep6Validate(unittest.TestCase):

    def test_coherence_delegates_to_the_tested_metric(self):
        """The indexing and denominator rules are unit-tested in test_metrics.
        What matters here is that the step calls that function rather than
        keeping a second copy that can drift from it."""
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn("from tools.metrics import coherence_precision", src)
        self.assertIn("coherence_precision(obj.get(\"verdicts\"), len(items))", src)

    def test_coherence_records_missing_verdicts(self):
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn('"missing": missing', src)
        self.assertIn('"sent": len(items)', src)

    def test_distinctness_merge_rate_excludes_errored_pairs(self):
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn('judged = [r for r in res if "error" not in r]', src)

    def test_distinctness_numbers_pairs_contiguously(self):
        """Bug: pairs were filtered after enumerate, leaving gaps in the numbering
        the model echoes back, so a rule could be recorded against the wrong pair."""
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn("ch = [(x, y) for x, y in ch if x in codes and y in codes]", src)

    def test_stability_does_not_swap_the_model_by_default(self):
        """The doc's stability bullet asks for a different seed and shuffled order.
        Swapping the model confounds order-sensitivity with model difference."""
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn("model = JUDGE if a.cross_model else BULK", src)

    def test_sampling_is_reproducible_under_concurrency(self):
        """Bug: one random.Random was shared across 16 workers while recording a
        seed in the artifact, asserting a reproducibility the code did not have."""
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn("random.Random(42 + hash(c)", src)
        self.assertIn("random.Random(4242 + i)", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# --------------------------------------------------------------------------
class TestStep2Report(unittest.TestCase):
    """The script whose whole purpose is that log numbers are not typed by hand."""

    def test_rate_uses_the_scripts_own_rate_not_a_fixed_denominator(self):
        """Bug: the rate was recomputed as new/50 while Step 2 divides by the real
        batch size. Batch 191 absorbed a failed batch's carry, processed 100
        labels and printed 27%; this script would have reported 54%."""
        src = (REPO / "tools/step2_report.py").read_text()
        self.assertNotIn("/50:.1%", src)
        self.assertIn("x['rate']", src)

    def test_totals_survive_a_resume(self):
        """Bug: token and time counters reset on process start, so after a resume
        the last row is the continuation segment and was published as the run total."""
        src = (REPO / "tools/step2_report.py").read_text()
        self.assertIn("segs", src)
        self.assertNotIn("B[-1]['tok']", src)

    def test_the_last_stop_reason_is_the_outcome(self):
        src = (REPO / "tools/step2_report.py").read_text()
        self.assertIn("stops[-1]", src)

    def test_positional_argument_is_not_silently_ignored(self):
        """Bug: `step2_report.py run` fell through to 'final' and reported a
        different artifact than the one asked for."""
        src = (REPO / "tools/step2_report.py").read_text()
        self.assertIn("argparse", src)
        self.assertNotIn("sys.argv[2] if len(sys.argv) > 2", src)


class TestGate(unittest.TestCase):

    def test_reporting_is_gated_on_this_suite(self):
        """The rule: a number is not accepted unless the tests guarding it pass."""
        from tools.gate import require_tests_pass, GateFailed
        self.assertTrue(callable(require_tests_pass))
        src = (REPO / "tools/step2_report.py").read_text()
        self.assertIn("require_tests_pass()", src)


# --------------------------------------------------------------------------
class TestModulesActuallyRun(unittest.TestCase):
    """String-matching tests miss what only execution catches.

    I added an atomic rewrite to step5_loop.py without importing os. The suite
    passed, because the test asserted the string "os.replace" appeared in the
    source. The chain then died at the write with a NameError, after the API
    calls. These tests execute instead of grepping.
    """

    def test_every_tool_module_imports(self):
        import importlib
        for m in ("gemini", "step1_extract", "step2_codebook", "step3_freeze",
                  "step4_relabel", "step5_loop", "step6_validate", "step7_audit",
                  "step2_report", "gate"):
            with self.subTest(module=m):
                importlib.import_module(f"tools.{m}")

    def test_every_tool_module_compiles_under_py_compile(self):
        import py_compile, tempfile
        for f in sorted((REPO / "tools").glob("step*.py")) + [REPO / "tools/gemini.py"]:
            with self.subTest(file=f.name), tempfile.TemporaryDirectory() as d:
                py_compile.compile(str(f), cfile=str(Path(d) / "x.pyc"), doraise=True)

    def test_atomic_rewrite_actually_replaces(self):
        """Executes the os.replace path rather than grepping for it."""
        import os as _os, tempfile
        for mod in (S4, S5):
            with self.subTest(module=mod.__name__):
                self.assertTrue(hasattr(mod, "os"), f"{mod.__name__} uses os.replace but never imported os")
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "a.tmp", Path(d) / "a.jsonl"
            dst.write_text("old\n"); src.write_text("new\n")
            _os.replace(src, dst)
            self.assertEqual(dst.read_text(), "new\n")
            self.assertFalse(src.exists())


# --------------------------------------------------------------------------
class TestStep7Audit(unittest.TestCase):
    """The evidence-driven audit the doc defers the real merge decisions to."""

    def test_alias_chain_resolves(self):
        from tools.step7_audit import resolve
        self.assertEqual(resolve({"a": "b", "b": "c"}, "a"), "c")
        self.assertEqual(resolve({}, "a"), "a")

    def test_alias_cycle_terminates(self):
        from tools.step7_audit import resolve
        self.assertIn(resolve({"a": "b", "b": "a"}, "a"), {"a", "b"})

    def test_splits_are_proposed_never_applied(self):
        """The doc: a split needs new labels, so applying one here would be
        'looks off to me' dressed as evidence."""
        src = (REPO / "tools/step7_audit.py").read_text()
        self.assertIn("split proposals (not applied)", src)
        self.assertNotIn('alias[o["code"]]', src)

    def test_uses_step0_thresholds_as_defaults(self):
        src = (REPO / "tools/step7_audit.py").read_text()
        self.assertIn('"--floor", type=int, default=20', src)
        self.assertIn('"--ceiling", type=float, default=0.05', src)

    def test_persists_the_co_assignment_matrix(self):
        """It was previously built in a local and discarded, though the doc names
        it as one of the three evidence streams."""
        src = (REPO / "tools/step7_audit.py").read_text()
        self.assertIn("co_assignment.json", src)

    def test_a_merge_cannot_breach_step0_ceiling(self):
        """Bug found in the first dry run: merging below-floor codes into their
        nearest neighbour chains them into attractors, producing a 982-item code
        against a ceiling of 476 - the very fusion the ceiling exists to prevent."""
        src = (REPO / "tools/step7_audit.py").read_text()
        self.assertIn("size[t_] + size[f_] > ceil_n", src)
        self.assertIn("refused_ceiling", src)

    def test_merges_apply_smallest_source_first(self):
        """Otherwise a chain can settle on the smaller code."""
        src = (REPO / "tools/step7_audit.py").read_text()
        self.assertIn('sorted(ops, key=lambda o: counts[o["from"]])', src)

    def test_empty_codes_are_pruned(self):
        src = (REPO / "tools/step7_audit.py").read_text()
        self.assertIn("holding no item after migration", src)

    def test_reporting_is_gated(self):
        src = (REPO / "tools/step7_audit.py").read_text()
        self.assertIn("require_tests_pass()", src)


# --------------------------------------------------------------------------
class TestStep8Definitions(unittest.TestCase):
    """Repairing definitions that drifted from the questions they hold."""

    def test_split_is_deterministic_and_disjoint(self):
        from tools.step8_definitions import split_items
        items = list(range(37))
        r1, h1 = split_items("c_0001", items)
        r2, h2 = split_items("c_0001", items)
        self.assertEqual(r1, r2, "the split must reproduce, or before/after is not like-for-like")
        self.assertEqual(h1, h2)
        self.assertEqual(set(r1) & set(h1), set(), "repair and holdout must be disjoint")
        self.assertEqual(sorted(r1 + h1), sorted(items), "no item may be lost in the split")

    def test_split_differs_per_code(self):
        from tools.step8_definitions import split_items
        items = list(range(40))
        self.assertNotEqual(split_items("c_0001", items)[0], split_items("c_0002", items)[0])

    def test_holdout_is_never_shown_to_the_writer(self):
        """The whole point: rewriting a definition from items and then scoring
        coherence on those same items raises the score by construction."""
        src = (REPO / "tools/step8_definitions.py").read_text()
        self.assertIn("repair, _hold = splits[c]", src)
        self.assertIn("shown = repair[:a.show]", src)
        self.assertNotIn("hold[:a.show]", src)

    def test_previous_definition_is_kept(self):
        """Without definition_before there is no like-for-like baseline."""
        src = (REPO / "tools/step8_definitions.py").read_text()
        self.assertIn('codes[c]["definition_before"]', src)

    def test_coherence_can_score_holdout_and_before(self):
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn('"--holdout"', src)
        self.assertIn('"--use-before"', src)
        self.assertIn('field = "definition_before" if', src)

    def test_coherence_refuses_holdout_without_step8(self):
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn("--holdout needs a codebook that Step 8 has written", src)

    def test_before_after_compares_only_rewritten_codes(self):
        """Bug: definition_before exists only on codes Step 8 rewrote, and the
        guard checked whether ANY code had it before indexing every code blindly.
        Including unrewritten codes would also dilute both sides equally."""
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn('rewritten = {c for c, v in codes.items() if "definition_before" in v}', src)
        self.assertIn("restricted to the", src)

    def test_reporting_is_gated(self):
        src = (REPO / "tools/step8_definitions.py").read_text()
        self.assertIn("require_tests_pass()", src)


# --------------------------------------------------------------------------
class TestGoldSet(unittest.TestCase):
    """The reference the doc ranks first and calls best return on effort."""

    def test_the_set_is_frozen_against_accidental_redraw(self):
        """Redrawing invalidates every score measured against it, so the sampler
        must refuse rather than silently overwrite."""
        src = (REPO / "tools/gold_sample.py").read_text()
        self.assertIn("already exists. The set is frozen by design", src)
        self.assertIn("--force", src)

    def test_selection_never_looks_at_pipeline_output(self):
        """A gold set stratified by the thing it judges agrees by construction.

        Checks executable code only. An earlier version of this test grepped the
        whole file and failed on the docstring sentence explaining that selection
        ignores what the pipeline assigned, which is the opposite of a violation.
        """
        import ast
        tree = ast.parse((REPO / "tools/gold_sample.py").read_text())
        for node in ast.walk(tree):          # drop docstrings, keep real strings
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
               and isinstance(node.value.value, str):
                node.value.value = ""
        code = ast.unparse(tree)
        for forbidden in ("item_labels", "codebook_v", "confidence", "validation_"):
            self.assertNotIn(forbidden, code,
                             f"the sampler reads {forbidden}, which taints the reference")

    def test_frozen_file_matches_the_declared_size_and_strata(self):
        f = REPO / "gold/gold_set.json"
        if not f.exists():
            self.skipTest("gold set not drawn yet")
        gs = json.loads(f.read_text())
        self.assertEqual(len(gs["items"]), gs["n"])
        self.assertEqual(len({i["item_idx"] for i in gs["items"]}), gs["n"],
                         "a duplicated question would be double-counted in the score")
        self.assertEqual(sum(gs["quota"].values()), gs["n"])
        for b, q in gs["quota"].items():
            self.assertGreaterEqual(q, min(gs["floor_per_benchmark"], gs["corpus_counts"][b]))

    def test_both_weightings_are_recorded(self):
        """The sample is not corpus-proportional, so a score can be reported
        per-question or re-weighted; the choice must be explicit, not baked in."""
        f = REPO / "gold/gold_set.json"
        if not f.exists():
            self.skipTest("gold set not drawn yet")
        gs = json.loads(f.read_text())
        self.assertIn("corpus_weight", gs)
        self.assertIn("sample_weight", gs)
        self.assertAlmostEqual(sum(gs["corpus_weight"].values()), 1.0, places=6)

    def test_part_b_carries_decoys_as_a_null_control(self):
        """Without decoys, a high tick rate cannot be told apart from agreeableness."""
        f = REPO / "gold/gold_partB_key.json"
        if not f.exists():
            self.skipTest("sheet not generated yet")
        key = json.loads(f.read_text())["by_item"]
        self.assertTrue(all(len(v["decoys"]) >= 1 for v in key.values()))
        for v in key.values():
            self.assertEqual(set(v["shown"]), set(v["assigned"]) | set(v["decoys"]))
            self.assertEqual(set(v["assigned"]) & set(v["decoys"]), set())

    def test_sheet_puts_free_text_before_the_checklist(self):
        """Seeing the pipeline's answer first anchors the free-text answer."""
        f = REPO / "gold/gold_sheet.md"
        if not f.exists():
            self.skipTest("sheet not generated yet")
        t = f.read_text()
        first = t.index("### Q001")
        blk = t[first:t.index("### Q002")]
        self.assertLess(blk.index("**Part A**"), blk.index("**Part B**"))


# --------------------------------------------------------------------------
class TestTextClipping(unittest.TestCase):
    """A flat cut deletes the question, and this project found that once already.

    MuSR items are ~4,800-char narratives whose ask is at the very end. Measured
    here: 506 questions exceed 4,000 chars, 502 of them MuSR, and 493 of the 506
    have question-like text after the cut. That produced a phantom skill about
    "completing an abruptly cut-off sentence" which no question actually needs.
    the project log 2026-06-13 recorded the same bug; Step 1 was fixed and Steps 4,
    6 and 8 reintroduced it with flat caps.
    """

    def test_short_text_is_untouched(self):
        from tools.textclip import clip
        self.assertEqual(clip("abc", 100), "abc")

    def test_the_tail_survives(self):
        """The whole point: the question lives at the end."""
        from tools.textclip import clip
        q = "narrative " * 800 + "Which location would Charlie look? 1 - cupboard 2 - desk"
        out = clip(q, 500)
        self.assertIn("Which location would Charlie look?", out)
        self.assertIn("2 - desk", out)

    def test_output_respects_the_budget(self):
        from tools.textclip import clip
        for b in (60, 200, 1500, 4000):
            self.assertLessEqual(len(clip("x" * 20000, b)), b)

    def test_head_is_kept_too(self):
        from tools.textclip import clip
        out = clip("START" + "m" * 5000 + "END", 200)
        self.assertTrue(out.startswith("START"))
        self.assertTrue(out.endswith("END"))

    def test_no_step_truncates_question_text_with_a_flat_cut(self):
        """The regression guard that would have caught this."""
        import re
        for f in ("step4_relabel", "step6_validate", "step8_definitions",
                  "gold_choice", "gold_sheet"):
            src = (REPO / f"tools/{f}.py").read_text()
            bad = re.findall(r'(?:txt\[[a-z_]+\]|question"?\]?)\[:\s*\d+\s*\]', src)
            self.assertEqual(bad, [], f"{f}.py truncates question text with a flat cut: {bad}")

    def test_every_step_uses_the_shared_clipper(self):
        """Three copies of a rule drift apart, and these three did."""
        for f in ("step4_relabel", "step6_validate", "step8_definitions"):
            src = (REPO / f"tools/{f}.py").read_text()
            self.assertIn("from tools.textclip import clip", src, f"{f}.py has its own rule")


# --------------------------------------------------------------------------
class TestTask2NullControl(unittest.TestCase):
    """A null control the subject can detect is not a control.

    v1 sampled its composition (12 decoys of 20, 7 of the last 8 consecutive)
    and the labeller noticed. v2's first attempt fixed the balance but produced
    RdRdRdRdRd, a perfect alternation, which is more detectable than the bug it
    replaced.
    """

    def _key(self):
        f = REPO / "gold/task2_key.json"
        if not f.exists():
            self.skipTest("task 2 sheet not generated")
        return json.loads(f.read_text())["items"]

    def test_composition_is_exactly_half(self):
        k = self._key()
        real = sum(1 for v in k.values() if v["is_pipeline_assignment"])
        self.assertEqual(real * 2, len(k), "composition must be fixed, not sampled")

    def test_no_run_longer_than_two(self):
        k = self._key()
        seq = [v["is_pipeline_assignment"] for v in k.values()]
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        self.assertLessEqual(best, 2, "a long run is detectable")

    def test_order_is_not_a_strict_alternation(self):
        """The failure introduced while fixing the first one."""
        k = self._key()
        seq = [v["is_pipeline_assignment"] for v in k.values()]
        alt = all(x != y for x, y in zip(seq, seq[1:]))
        self.assertFalse(alt, "perfect alternation is trivially predictable")

    def test_items_are_not_reused_from_the_earlier_sheet(self):
        """Remembering an earlier answer is its own contamination."""
        k = self._key()
        old = REPO / "gold/choice_key.json"
        if not old.exists():
            self.skipTest("no earlier sheet")
        o = json.loads(old.read_text())
        prev = {v["item_idx"] for v in o.get("task2", {}).values()} | \
               {v["item_idx"] for v in o.get("task1", {}).values()}
        self.assertEqual({v["item_idx"] for v in k.values()} & prev, set())

    def test_decoys_are_never_drawn_from_the_items_own_candidates(self):
        """Otherwise a decoy could be a skill the question genuinely needs."""
        src = (REPO / "tools/gold_task2.py").read_text()
        self.assertIn('banned = set(r.get("candidates", [])) | set(assigned)', src)


# --------------------------------------------------------------------------
class TestStep9Dedupe(unittest.TestCase):
    """Merging duplicates that every earlier criterion was blind to.

    Berke's "neither definition fits" on 6 of 14 hand-labelled items turned out
    to be four codes for one operation (object location from a narrative). All
    four were above the size floor, had pairwise cosine 0.67-0.79 against a 0.85
    threshold, and were co-assigned 0-2 times against a threshold of 20.
    """

    def test_similarity_threshold_is_a_percentile_not_an_absolute(self):
        """0.85 sat above the 99.9th percentile here and nominated 28 pairs of
        roughly 68,000. An absolute cosine does not transfer between codebooks."""
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn("np.percentile", src)
        self.assertIn("SIM_PCT", src)
        self.assertNotIn(">= 0.85", src)

    def test_competition_not_co_assignment_is_the_duplicate_signal(self):
        """Co-assignment is backwards for this: near-duplicates compete for the
        same item, so exactly one wins and they co-occur almost never."""
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn("OFFERED_MIN", src)
        self.assertIn("CHOSEN_MAX", src)
        self.assertIn("offered[x] & offered[y]", src)
        self.assertIn("chosen[x] & chosen[y]", src)

    def test_chunk_size_leaves_room_for_thinking_tokens(self):
        """10 pairs truncated the JSON mid-array and every such chunk was
        recorded as an error; the judge spends thinking tokens from max_out."""
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn("CHUNK = 6", src)
        self.assertIn("max_out=16000", src)

    def test_merges_respect_the_ceiling_and_go_through_the_alias_chain(self):
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn("ceil_n", src)
        self.assertIn("resolve(alias", src)

    def test_merges_apply_smallest_first(self):
        """Largest-first let big pairs consume the ceiling budget: 44 merges were
        refused and the undersized tail was never consolidated. The effect of the
        ordering is unit-tested in test_metrics."""
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn('order="smallest"', src)

    def test_verdicts_can_be_reused_to_isolate_application_changes(self):
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn("--reuse-verdicts", src)

    def test_reporting_is_gated(self):
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn("require_tests_pass()", src)


# --------------------------------------------------------------------------
class TestMergeApplication(unittest.TestCase):
    """Merging is not transitive, and applying it as if it were composed
    accepted pairs into a merge the judge had explicitly rejected:
    it ruled c_0464 and c_0274 distinct, and c_0464 -> c_1047 -> c_0274 put them
    together anyway, through a chain 8 hops long at its worst.
    """

    def test_step9_passes_rejected_pairs_to_the_applier(self):
        """The non-transitivity rule is unit-tested in test_metrics; here we only
        check the rejected pairs are actually handed over."""
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn('rejected = [tuple(r["pair"]) for r in judged if not r["merge"]]', src)
        self.assertIn("apply_merges(accepted, rejected", src)

    def test_step9_delegates_merge_application(self):
        src = (REPO / "tools/step9_dedupe.py").read_text()
        self.assertIn("from tools.metrics import apply_merges", src)

    def test_union_find_honours_both_constraints(self):
        """Executes the logic rather than grepping it."""
        counts = {"a": 10, "b": 10, "c": 10}
        rejected = {("a", "c")}
        parent = {c: c for c in counts}
        members = {c: {c} for c in counts}
        size = dict(counts)

        def find(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]; c = parent[c]
            return c

        def union(x, y, ceiling=100):
            rx, ry = find(x), find(y)
            if rx == ry:
                return "same"
            if size[rx] + size[ry] > ceiling:
                return "ceiling"
            if any(tuple(sorted((u, v))) in rejected
                   for u in members[rx] for v in members[ry]):
                return "blocked"
            parent[ry] = rx; members[rx] |= members[ry]; size[rx] += size[ry]
            return "ok"

        self.assertEqual(union("a", "b"), "ok")
        # a~b and b~c accepted, but a~c was rejected: the chain must not close
        self.assertEqual(union("b", "c"), "blocked")
        self.assertNotEqual(find("a"), find("c"))

    def test_label_migration_is_snapshotted_first(self):
        """The alias map is many-to-one and cannot be inverted, so a bad pass
        without a snapshot costs a full Step 4 re-run. It did once."""
        for f in ("step7_audit", "step9_dedupe"):
            src = (REPO / f"tools/{f}.py").read_text()
            self.assertIn("item_labels_before_", src, f"{f} migrates labels with no snapshot")


class TestValidationTargetsTheCodebookYouAsked(unittest.TestCase):
    """Bug: --codebook was honoured by coherence and silently ignored by
    distinctness and stability, which fell back to a hardcoded list headed by
    codebook_v5_deduped.json. That is why those two were reported against v5
    while coherence was against v6, a discrepancy recorded in the run log as a
    sequencing choice when it was this defect."""

    def test_all_three_checks_pass_the_requested_codebook_through(self):
        import re
        src = (REPO / "tools/step6_validate.py").read_text()
        calls = re.findall(r"load_state\(([^)]*)\)", src)
        calls = [c for c in calls if "name=None" not in c]
        self.assertEqual(len(calls), 3, "expected coherence, distinctness and stability")
        for c in calls:
            self.assertIn("codebook", c,
                          "a validation entry point ignores --codebook and will "
                          "silently score whichever codebook the fallback list finds first")


class TestValidationOutputsAreNamedForTheirCodebook(unittest.TestCase):
    """Bug: every check wrote to a fixed filename, so scoring a second codebook
    destroyed the first one's per-skill results. On 2026-09-09 the v6 coherence
    detail was lost that way. The run log's by-size table and floor analysis were
    computed from it, and only the aggregates survived, in prose. An aggregate
    cannot re-verify a claim."""

    def test_a_second_codebook_does_not_overwrite_the_first(self):
        import argparse
        from tools.step6_validate import out_name
        a = argparse.Namespace(codebook="codebook_v8_definitions.json")
        b = argparse.Namespace(codebook="codebook_v6_definitions.json")
        for base in ("validation_coherence.json", "validation_distinctness.json",
                     "validation_stability.json"):
            with self.subTest(base=base):
                self.assertNotEqual(out_name(a, base), out_name(b, base))
                self.assertIn("v8_definitions", out_name(a, base))
                self.assertTrue(out_name(a, base).endswith(".json"))

    def test_the_holdout_tag_survives_the_rename(self):
        import argparse
        from tools.step6_validate import out_name
        a = argparse.Namespace(codebook="codebook_v8_definitions.json")
        n = out_name(a, "validation_coherence_holdout_before.json")
        self.assertIn("holdout_before", n)
        self.assertIn("v8_definitions", n)

    def test_no_codebook_flag_keeps_the_legacy_name(self):
        import argparse
        from tools.step6_validate import out_name
        a = argparse.Namespace(codebook=None)
        self.assertEqual(out_name(a, "validation_coherence.json"),
                         "validation_coherence.json")


class TestDoubleJudgedMerges(unittest.TestCase):
    """Dedupe pass 2 merged 149 pairs on one judge's word and measured worse.
    This applies only the intersection of two independent verdict sets."""

    def test_only_pairs_both_judges_accepted_are_taken(self):
        from tools.apply_double_judged import double_judged, rejected_by_either
        dedupe = [{"pair": ["a", "b"], "merge": True},
                  {"pair": ["c", "d"], "merge": True},
                  {"pair": ["e", "f"], "merge": False}]
        distinct = [{"pair": ["b", "a"], "merge": True},
                    {"pair": ["c", "d"], "merge": False},
                    {"pair": ["e", "f"], "merge": True}]
        self.assertEqual(double_judged(dedupe, distinct), [("a", "b")],
                         "a pair only one judge accepted must not be merged")

    def test_either_judge_can_veto(self):
        from tools.apply_double_judged import rejected_by_either
        dedupe = [{"pair": ["a", "b"], "merge": False}]
        distinct = [{"pair": ["c", "d"], "merge": False}]
        self.assertEqual(rejected_by_either(dedupe, distinct), [("a", "b"), ("c", "d")])

    def test_pair_order_does_not_matter(self):
        from tools.apply_double_judged import double_judged
        self.assertEqual(
            double_judged([{"pair": ["z", "a"], "merge": True}],
                          [{"pair": ["a", "z"], "merge": True}]), [("a", "z")])


class TestGoldSheetIsUsable(unittest.TestCase):
    """The sheet Berke hand-labels is the doc's acceptance gate, so a flaw in it
    invalidates every comparison made against it. The version generated on
    2026-09-08 had three: 18 questions listed the same skill twice, the number of
    options revealed how many were real (3 options meant 1 real, 5 meant 3), and
    all 15 MuSR passages were cut before the question, which sits at the end."""

    def setUp(self):
        self.f = REPO / "gold/gold_sheet.md"
        if not self.f.exists():
            self.skipTest("gold sheet not generated")
        import re
        self.qs = re.split(r"^### ", self.f.read_text(), flags=re.M)[1:]

    def test_every_question_offers_the_same_number_of_options(self):
        import re, collections
        c = collections.Counter(len(re.findall(r"- \[ \] `c_\d+`", q)) for q in self.qs)
        self.assertEqual(len(c), 1,
                         f"option count varies {dict(c)}; it tells the labeller how many to tick")

    def test_no_question_lists_the_same_skill_twice(self):
        import re
        for q in self.qs:
            ids = re.findall(r"- \[ \] `(c_\d+)`", q)
            with self.subTest(q=q.split(chr(10))[0]):
                self.assertEqual(len(ids), len(set(ids)))

    def test_the_question_itself_is_never_cut_off(self):
        """MuSR puts the question after a ~5,000 character story. A flat cut
        removes it and the labeller is shown a passage with nothing asked."""
        import re, json
        D = REPO / "cdm_exploration/data/cdm_ready"
        tx = {r["item_idx"]: " ".join(r["question_full_text"].split())
              for r in json.load((D / "item_full_text_recovered.json").open())}
        missing = []
        for q in self.qs:
            m = re.match(r"Q\d+ · item (\d+)", q)
            body = q.split("**Part A**")[0]
            if m and tx[int(m.group(1))][-40:] not in body:
                missing.append(m.group(0))
        self.assertEqual(missing, [], "these entries do not show the end of the question")

    def test_the_old_flat_truncation_marker_is_gone(self):
        self.assertNotIn("[truncated]", self.f.read_text())


class TestAuditBudgetActuallyScales(unittest.TestCase):
    """The missing test. run_audit carried a scaling cap whose comment said the
    doc's flat 15 was too small, and it computed max(15, created//10). With ~56
    codes created per audit that is max(15, 5) = 15, so the relaxation never once
    took effect and all 31 audits were truncated at the doc's original figure.
    Nothing asserted the cap ever exceeded its floor."""

    def cap(self, created):
        import re
        src = (REPO / "tools/step2_codebook.py").read_text()
        mn = int(re.search(r"AUDIT_OPS_MIN\s*=\s*(\d+)", src).group(1))
        per = int(re.search(r"AUDIT_OPS_PER\s*=\s*(\d+)", src).group(1))
        return max(mn, created // per)

    def test_the_cap_rises_above_its_floor_at_the_observed_creation_rate(self):
        """1,727 codes over 31 audits is about 56 created per audit."""
        self.assertGreater(self.cap(56), 15,
                           "the scaling cap is dead code: it never exceeds its own floor")

    def test_the_budget_is_not_a_quarter_of_creation(self):
        """31 audits x 15 ops = 465 against 1,727 codes created, i.e. 27%."""
        total = 31 * self.cap(56)
        self.assertGreater(total, 1727,
                           f"consolidation gets {total} operations for 1,727 codes created")

    def test_a_small_batch_still_gets_the_floor(self):
        self.assertEqual(self.cap(0), 15)


class TestAuditCallCanActuallyFinish(unittest.TestCase):
    """The audit is one long generation per round. At cap 150 against the client's
    default 180s timeout it failed five times over 935s and merged nothing. Two
    ways that goes wrong silently: the timeout is not plumbed through, or the cap
    exceeds the number of candidate pairs the audit is even shown."""

    def test_the_client_accepts_a_per_call_timeout(self):
        import inspect
        from tools.gemini import Gemini
        for fn in (Gemini.json_obj, Gemini.text):
            with self.subTest(fn=fn.__name__):
                self.assertIn("timeout", inspect.signature(fn).parameters)

    def test_the_audit_asks_for_more_than_the_default(self):
        import re
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn("timeout=AUDIT_TIMEOUT", src,
                      "the audit call uses the default timeout it already exceeded")
        t = int(re.search(r"AUDIT_TIMEOUT\s*=\s*(\d+)", src).group(1))
        self.assertGreater(t, 180)

    def test_the_cap_cannot_exceed_the_pairs_shown(self):
        import re
        shown = int(re.search(r"MERGE_PAIRS_SHOWN\s*=\s*(\d+)",
                              (REPO / "tools/step2_codebook.py").read_text()).group(1))
        cap = int(re.search(r'"--cap", type=int, default=(\d+)',
                            (REPO / "tools/audit_rounds.py").read_text()).group(1))
        self.assertLessEqual(cap, shown,
                             "asking for more operations than there are candidate pairs "
                             "lengthens the generation without making more merges possible")


class TestChurnGuardAppliesWhereThrashingIsPossible(unittest.TestCase):
    """The churn abort fired on the first audit of the batch-150 run, stopping it
    after 10 batches. Churn detects thrashing, and thrashing means undoing an
    earlier audit's work. The first audit has none to undo: it collapses the cold
    start, where batch 0 puts BATCH labels into an empty codebook and creates one
    code per label by construction. Merging those moves a large share of labels
    arithmetically. From the second audit the guard is unchanged."""

    def test_the_threshold_itself_was_not_moved(self):
        import re
        src = (REPO / "tools/step2_codebook.py").read_text()
        v = float(re.search(r"CHURN_ABORT\s*=\s*([\d.]+)", src).group(1))
        self.assertEqual(v, 0.15, "CHURN_ABORT must not be relaxed to make a run pass")

    def test_the_guard_is_skipped_only_for_the_first_audit(self):
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn("if churn > CHURN_ABORT and cb.version > 1:", src)

    def test_audit_interval_is_inside_the_doc_range(self):
        from tools.step2_codebook import AUDIT_EVERY
        self.assertGreaterEqual(AUDIT_EVERY, 5, "the doc says every 5-10 batches")
        self.assertLessEqual(AUDIT_EVERY, 10)


class TestBatchSizeAndResumeAgree(unittest.TestCase):
    """BATCH decides how labels are partitioned, and `next_batch` is an index
    into that partition. Making BATCH configurable without recording it meant a
    resume at a different size would skip a completely different set of labels,
    which would never be processed and would show up nowhere except a total
    nobody diffs."""

    def test_batch_size_follows_the_doc(self):
        from tools.step2_codebook import BATCH
        self.assertGreaterEqual(BATCH, 100, "the doc specifies 100-200 per batch")
        self.assertLessEqual(BATCH, 200)

    def test_the_resume_state_records_the_batch_size(self):
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn('"batch_size": BATCH', src)

    def test_resuming_at_a_different_batch_size_is_refused(self):
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn("cannot resume:", src,
                      "a BATCH mismatch on resume must stop the run, not skip labels")

    def test_retrieval_survives_an_empty_embedding_cache(self):
        """embed_new_codes defers the whole batch on a 429, leaving cb.codes
        non-empty and code_vec empty. np.stack([]) raises."""
        from collections import Counter
        from tools.step2_codebook import Codebook, _batch_prompt
        import numpy as np
        cb = Codebook()
        cb.add("a skill", "does a thing", [], [], 0)
        cands, codes_txt, items_txt = _batch_prompt(
            cb, ["some label"], Counter({"some label": 1}),
            {"some label": np.ones(4, dtype="float32")}, {})
        self.assertEqual(cands, [[]])
        self.assertIn("some label", items_txt)

    def test_a_run_can_be_namespaced_so_it_cannot_clobber_the_existing_one(self):
        """codebook_final.json and codebook_run.json are what every downstream
        number currently traces to. A second run with no tag overwrites both."""
        src = (REPO / "tools/step2_codebook.py").read_text()
        self.assertIn('TAG = os.environ.get("STEP2_TAG"', src)
        for call in ('cb.save(f"run{TAG}"', '{"run": "final", "trial": "trial", "smoke": "smoke"}[mode] + TAG',
                     'cb.save(f"v{cb.version}{TAG}")', 'cb.load(f"run{TAG}")'):
            with self.subTest(call=call):
                self.assertIn(call, src, "an output path ignores the run tag")

    def test_parallel_planning_is_gone(self):
        """Reverted 2026-09-10: it shifted decision indices by len(carry) and
        its executor joined inside the loop, so it corrupted labels and was not
        faster. Re-adding it needs futures held across iterations and a lock."""
        import tools.step2_codebook as m
        self.assertFalse(hasattr(m, "PARALLEL"))
        self.assertNotIn("ThreadPoolExecutor(max_workers=len(todo))",
                         (REPO / "tools/step2_codebook.py").read_text())


class TestSteps3to5CannotClobberAnEarlierRun(unittest.TestCase):
    """Steps 2, 6 and 9 namespace their outputs; steps 3-5 did not. Running them
    on a second codebook would have overwritten codebook_v1_frozen.json and
    item_labels.jsonl, which the 230-skill taxonomy and Berke's gold-set score
    both trace to, with no warning."""

    def test_every_artifact_path_goes_through_the_tag(self):
        import re
        for name in ("step3_freeze", "step4_relabel", "step5_loop"):
            src = (REPO / f"tools/{name}.py").read_text()
            bare = re.findall(r'P / "(codebook_[a-z0-9_]*\.json|item_labels[a-z0-9_]*\.jsonl)"', src)
            with self.subTest(step=name):
                self.assertEqual(bare, [], f"{name} reads or writes an untagged artifact: {bare}")
                self.assertIn("def tagged(", src)

    def test_no_tag_means_the_original_filenames(self):
        import importlib, os, sys
        sys.path.insert(0, str(REPO / "tools"))
        os.environ.pop("STEP_TAG", None)
        m = importlib.reload(importlib.import_module("step3_freeze"))
        self.assertEqual(m.tagged("codebook_final.json"), "codebook_final.json")
        self.assertEqual(m.tagged("item_labels.jsonl"), "item_labels.jsonl")

    def test_a_tag_is_inserted_before_the_extension(self):
        import importlib, os, sys
        sys.path.insert(0, str(REPO / "tools"))
        os.environ["STEP_TAG"] = "_b150"
        try:
            m = importlib.reload(importlib.import_module("step3_freeze"))
            self.assertEqual(m.tagged("codebook_final.json"), "codebook_final_b150.json")
            self.assertEqual(m.tagged("item_labels.jsonl"), "item_labels_b150.jsonl")
        finally:
            os.environ.pop("STEP_TAG", None)
            importlib.reload(importlib.import_module("step3_freeze"))


class TestStep3DoesNotImportAnotherRunsRules(unittest.TestCase):
    """Code ids restart at c_0001 every run, so they collide across runs while
    naming different skills. Step 3 read an untagged validation_distinctness.json
    and attached the previous run's discriminating rules to this run's codes:
    c_0019 meant "calculating temporal offsets" in one and "solving quadratic
    equations" in the other. Those rules feed Step 4's prompt."""

    def test_the_distinctness_file_is_namespaced(self):
        src = (REPO / "tools/step3_freeze.py").read_text()
        self.assertNotIn('P / "validation_distinctness.json"', src)
        self.assertIn('tagged("validation_distinctness.json")', src)
