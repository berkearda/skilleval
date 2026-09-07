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
        ids unchanged silently reused stale vectors and nothing could detect it."""
        src = (REPO / "tools/step3_freeze.py").read_text()
        self.assertIn("hashlib.sha256", src)
        self.assertIn('z["digest"]', src)


# --------------------------------------------------------------------------
class TestStep4Relabel(unittest.TestCase):

    def test_retrieval_and_judging_use_one_truncation(self):
        """Bug: retrieval saw 2,000 chars and the judge saw 4,000, so for 30% of
        the corpus candidates were chosen from half the question."""
        src = (REPO / "tools/step4_relabel.py").read_text()
        self.assertIn("QCHARS", src)
        self.assertNotIn("[:2000]", src)
        self.assertEqual(src.count("txt[i][:QCHARS]"), 1)

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

    def test_no_dead_rounds_flag(self):
        """Bug: --rounds was parsed and never read, advertising a loop the
        script does not have and is not re-entrant enough to run."""
        src = (REPO / "tools/step5_loop.py").read_text()
        self.assertNotIn('add_argument("--rounds"', src)


# --------------------------------------------------------------------------
class TestStep6Validate(unittest.TestCase):

    def test_coherence_indexes_verdicts_by_i(self):
        """Bug: verdicts were consumed positionally while the prompt asks the
        judge to echo i, so one skipped question shifted every later verdict."""
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn('int(x.get("i", 0)) - 1', src)
        self.assertNotIn('[bool(x.get("match")) for x in obj.get("verdicts", [])]', src)

    def test_coherence_denominator_is_what_was_sent(self):
        """Bug: precision divided by verdicts returned, so a code sent 10 and
        given 3 matching verdicts scored 1.0."""
        src = (REPO / "tools/step6_validate.py").read_text()
        self.assertIn("sum(got.values()) / len(items)", src)
        self.assertIn('"missing"', src)

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
