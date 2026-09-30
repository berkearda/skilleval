#!/usr/bin/env python3
"""Regression tests for the training script and the experiment log.

    python3 -m unittest discover -s tests -v
"""
import json, sys, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class TestTrainerCannotClobberACheckpoint(unittest.TestCase):
    """save_checkpoint has no existence check, and the names carry no taxonomy, so
    training any other Q at seed 42 overwrote the checkpoint behind the paper's
    table AUC, and seeds 43-46 overwrote the multi-seed appendix checkpoints."""

    def test_an_existing_checkpoint_is_moved_aside_not_destroyed(self):
        """Overwriting lost the paper's checkpoint silently. Refusing was worse for a
        reproducibility artifact: train_v2.sh would abort the moment a checkpoint
        existed, so the submitted repo would no longer reproduce its own number.
        Moving it aside keeps both properties."""
        src = (REPO / "tools/train_expanded.py").read_text()
        self.assertIn("shutil.move", src)
        self.assertIn("superseded_", src)
        self.assertIn("moved existing checkpoint aside", src)
        self.assertNotIn("already exists and nothing in its name says which", src,
                         "the refusal is back, and it breaks reproduction")
        # The move-aside path only runs when a checkpoint already exists, which is
        # exactly the reproduction path, so a missing import would raise NameError
        # there and nowhere else. A source-string test cannot see that; pin the
        # import, and compile the module so a syntax or name error fails here.
        self.assertIn("from datetime import datetime", src)
        import py_compile, tempfile
        with tempfile.TemporaryDirectory() as d:
            py_compile.compile(str(REPO / "tools/train_expanded.py"),
                               cfile=str(Path(d) / "t.pyc"), doraise=True)

    def test_the_arm_can_be_named(self):
        src = (REPO / "tools/train_expanded.py").read_text()
        self.assertIn("ckpt_tag", src)
        self.assertIn('ckpt_name = f"{base}{tag}.pt"', src)

    def test_the_log_entry_records_which_qmatrix_it_trained_on(self):
        """K alone does not identify a taxonomy: two 274-skill banks differ."""
        src = (REPO / "tools/train_expanded.py").read_text()
        self.assertIn('"qmatrix": (str(cfg.data.qmatrix)', src)
        self.assertIn('name=f"train_expanded_seed{seed}{tag}"', src)


class TestExperimentLogMerge(unittest.TestCase):
    """The cluster's log holds only its own runs, because scratch was purged. Copying
    it back would destroy the local history that every paper number traces to."""

    def _run(self, cur, inc, apply=False):
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as d:
            t, i = Path(d) / "target.json", Path(d) / "inc.json"
            t.write_text(json.dumps(cur))
            i.write_text(json.dumps(inc))
            cmd = [sys.executable, str(REPO / "tools/merge_experiment_log.py"),
                   "--incoming", str(i), "--target", str(t)]
            if apply:
                cmd.append("--apply")
            p = subprocess.run(cmd, capture_output=True, text=True)
            return p, json.loads(t.read_text())

    def test_existing_entries_are_never_lost(self):
        cur = [{"experiment": "a", "timestamp": "t1", "results": {"test_auc": 0.5}}]
        inc = [{"experiment": "b", "timestamp": "t2", "results": {"test_auc": 0.6}}]
        p, after = self._run(cur, inc, apply=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(cur[0], after)
        self.assertEqual(len(after), 2)

    def test_a_dry_run_writes_nothing(self):
        cur = [{"experiment": "a", "timestamp": "t1"}]
        inc = [{"experiment": "b", "timestamp": "t2"}]
        p, after = self._run(cur, inc, apply=False)
        self.assertEqual(after, cur)
        self.assertIn("dry run", p.stdout)

    def test_reimporting_the_same_log_adds_nothing(self):
        cur = [{"experiment": "a", "timestamp": "t1", "results": {"test_auc": 0.5}}]
        p, after = self._run(cur, list(cur), apply=True)
        self.assertEqual(after, cur, "an idempotent merge must not duplicate")

    def test_a_conflicting_entry_is_refused_not_overwritten(self):
        """Same run identity, different numbers: guessing would silently replace a
        result."""
        cur = [{"experiment": "a", "timestamp": "t1", "results": {"test_auc": 0.5}}]
        inc = [{"experiment": "a", "timestamp": "t1", "results": {"test_auc": 0.9}}]
        p, after = self._run(cur, inc, apply=True)
        self.assertEqual(after, cur, "the local result was overwritten")
        self.assertIn("CONFLICTS        1", p.stdout)


if __name__ == "__main__":
    unittest.main()
