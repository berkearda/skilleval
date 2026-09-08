#!/usr/bin/env python3
"""Unit tests for every number the pipeline reports, on hand-computed values."""
import sys, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.metrics import (coherence_precision, jaccard, separation, apply_merges,
                           resolve, size_stats, identifiable)


class TestCoherencePrecision(unittest.TestCase):

    def test_all_returned_all_match(self):
        v = [{"i": i, "match": True} for i in range(1, 11)]
        self.assertEqual(coherence_precision(v, 10), (1.0, 10, 0))

    def test_denominator_is_what_was_sent(self):
        """The bug: 10 sent, 3 returned all matching, scored 1.0."""
        v = [{"i": 1, "match": True}, {"i": 2, "match": True}, {"i": 3, "match": True}]
        p, got, missing = coherence_precision(v, 10)
        self.assertAlmostEqual(p, 0.3)
        self.assertEqual((got, missing), (3, 7))

    def test_verdicts_are_indexed_by_i_not_position(self):
        """A skipped question must not shift later verdicts onto wrong questions.
        Here questions 1 and 3 match, 5 does not. Positionally this would read
        as 1,2 matching and 3 not."""
        v = [{"i": 1, "match": True}, {"i": 3, "match": True}, {"i": 5, "match": False}]
        p, got, missing = coherence_precision(v, 5)
        self.assertAlmostEqual(p, 0.4)
        self.assertEqual((got, missing), (3, 2))

    def test_out_of_range_and_malformed_indices_are_dropped(self):
        v = [{"i": 99, "match": True}, {"i": "x", "match": True}, "notadict",
             {"i": 1, "match": True}]
        self.assertEqual(coherence_precision(v, 3), (1 / 3, 1, 2))

    def test_no_verdicts_scores_zero_not_none(self):
        p, got, missing = coherence_precision([], 4)
        self.assertEqual((p, got, missing), (0.0, 0, 4))

    def test_zero_sent_is_none(self):
        self.assertEqual(coherence_precision([], 0), (None, 0, 0))


class TestJaccard(unittest.TestCase):

    def test_identical(self):
        self.assertEqual(jaccard(["a", "b"], ["b", "a"]), 1.0)

    def test_disjoint(self):
        self.assertEqual(jaccard(["a"], ["b"]), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(jaccard(["a", "b"], ["b", "c"]), 1 / 3)

    def test_both_empty_is_agreement(self):
        self.assertEqual(jaccard([], []), 1.0)


class TestSeparation(unittest.TestCase):

    def test_perfect(self):
        self.assertEqual(separation(10, 10, 0, 10), 1.0)

    def test_no_discrimination(self):
        """Accepting everything scores zero: the null control's whole purpose."""
        self.assertEqual(separation(10, 10, 10, 10), 0.0)

    def test_matches_the_measured_run(self):
        """Berke's clean sheet: 10/10 real accepted, 0/10 decoys."""
        self.assertEqual(separation(10, 10, 0, 10), 1.0)

    def test_undefined_without_both_arms(self):
        self.assertIsNone(separation(5, 5, 0, 0))


class TestApplyMerges(unittest.TestCase):

    def test_transitive_chain_is_blocked_when_the_ends_were_rejected(self):
        """The real failure: a~b and b~c accepted, a~c explicitly rejected, and
        greedy application merged a with c anyway through the chain."""
        alias, st = apply_merges(accepted=[("a", "b"), ("b", "c")], rejected=[("a", "c")],
                                 sizes={"a": 10, "b": 10, "c": 10}, ceiling=100)
        self.assertNotEqual(resolve(alias, "a"), resolve(alias, "c"))
        self.assertEqual(st["blocked_rejected"], 1)
        self.assertEqual(st["applied"], 1)

    def test_chain_closes_when_nothing_was_rejected(self):
        alias, st = apply_merges([("a", "b"), ("b", "c")], [], {"a": 5, "b": 5, "c": 5}, 100)
        self.assertEqual(resolve(alias, "a"), resolve(alias, "c"))
        self.assertEqual(st["applied"], 2)

    def test_ceiling_refuses_a_union(self):
        alias, st = apply_merges([("a", "b")], [], {"a": 60, "b": 60}, ceiling=100)
        self.assertEqual(alias, {})
        self.assertEqual(st["refused_ceiling"], 1)

    def test_merge_order_changes_what_fits_under_the_ceiling(self):
        """Largest-first spends the budget on big pairs and leaves the small tail
        unmerged; smallest-first consolidates the tail. This is the change that
        moved below-floor from 60% to 56%."""
        sizes = {"big1": 50, "big2": 45, "s1": 3, "s2": 3}
        acc = [("big1", "big2"), ("s1", "s2")]
        _, big_first = apply_merges(acc, [], sizes, ceiling=100, order="largest")
        _, small_first = apply_merges(acc, [], sizes, ceiling=100, order="smallest")
        self.assertEqual(big_first["applied"], 2)
        self.assertEqual(small_first["applied"], 2)
        # with a tighter ceiling only one union fits, and the order decides which
        _, bf = apply_merges(acc, [], sizes, ceiling=90, order="largest")
        _, sf = apply_merges(acc, [], sizes, ceiling=90, order="smallest")
        self.assertEqual(bf["refused_ceiling"], 1)
        self.assertEqual(sf["applied"], 1)

    def test_labels_are_never_orphaned(self):
        alias, _ = apply_merges([("a", "b"), ("b", "c")], [], {"a": 1, "b": 2, "c": 3}, 100)
        for c in ("a", "b", "c"):
            self.assertIn(resolve(alias, c), {"a", "b", "c"})

    def test_alias_cycle_terminates(self):
        self.assertIn(resolve({"a": "b", "b": "a"}, "a"), {"a", "b"})

    def test_component_size_is_conserved(self):
        sizes = {"a": 4, "b": 6, "c": 5}
        alias, _ = apply_merges([("a", "b"), ("b", "c")], [], sizes, 100)
        merged = {}
        for c, n in sizes.items():
            merged[resolve(alias, c)] = merged.get(resolve(alias, c), 0) + n
        self.assertEqual(sum(merged.values()), sum(sizes.values()))


class TestSizeStats(unittest.TestCase):

    def test_hand_computed(self):
        counts = {"a": 100, "b": 30, "c": 10, "d": 1}
        st = size_stats(counts, floor=20, ceiling_frac=0.05, n_items=1000)
        self.assertEqual(st["codes"], 4)
        self.assertEqual(st["max"], 100)
        self.assertEqual(st["median"], 10)
        self.assertEqual(st["below_floor"], 2)
        self.assertEqual(st["above_ceiling"], 1)     # 100 > 0.05 * 1000
        self.assertEqual(st["total_assignments"], 141)

    def test_empty(self):
        self.assertEqual(size_stats({}), {})


class TestIdentifiable(unittest.TestCase):

    def test_only_single_skill_questions_identify(self):
        rows = [{"assigned": [{"code": "a"}]},
                {"assigned": [{"code": "b"}, {"code": "c"}]},
                {"assigned": []}]
        self.assertEqual(identifiable(rows), {"a"})

    def test_a_skill_never_seen_alone_is_not_identifiable(self):
        rows = [{"assigned": [{"code": "x"}, {"code": "y"}]}] * 50
        self.assertEqual(identifiable(rows), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
