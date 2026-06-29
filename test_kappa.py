"""Tests for kappa. Run: python -m unittest test_kappa -v   (stdlib only)."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import kappa


class TestCohenKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        pairs = [("a", "a"), ("b", "b"), ("a", "a")]
        self.assertAlmostEqual(kappa.cohen_kappa(pairs), 1.0)

    def test_known_negative_case(self):
        # judge=pass,revise,pass ; human=pass,pass,revise -> kappa = -0.5 (hand-computed)
        pairs = [("pass", "pass"), ("revise", "pass"), ("pass", "revise")]
        self.assertAlmostEqual(kappa.cohen_kappa(pairs), -0.5, places=6)

    def test_known_2x2_case(self):
        # 7 (yes,yes), 1 (no,no), 1 (yes,no), 1 (no,yes) -> po=0.8, pe=0.68, kappa=0.375
        pairs = ([("yes", "yes")] * 7 + [("no", "no")] + [("yes", "no")] + [("no", "yes")])
        self.assertAlmostEqual(kappa.cohen_kappa(pairs), 0.375, places=6)

    def test_empty(self):
        self.assertIsNone(kappa.cohen_kappa([]))

    def test_single_label_both_sides(self):
        # everyone says "pass" -> perfect agreement, pe==1 guard returns 1.0
        self.assertAlmostEqual(kappa.cohen_kappa([("pass", "pass")] * 5), 1.0)

    def test_multiclass(self):
        pairs = [("x", "x"), ("y", "y"), ("z", "z"), ("x", "y")]
        k = kappa.cohen_kappa(pairs)
        self.assertTrue(-1.0 <= k <= 1.0)
        self.assertLess(k, 1.0)


class TestBand(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(kappa.band(1.0), "almost perfect")
        self.assertEqual(kappa.band(0.9), "almost perfect")
        self.assertEqual(kappa.band(0.7), "substantial")
        self.assertEqual(kappa.band(0.5), "moderate")
        self.assertEqual(kappa.band(0.3), "fair")
        self.assertEqual(kappa.band(0.1), "slight")
        self.assertEqual(kappa.band(-0.2), "poor (worse than chance)")


class TestConfusion(unittest.TestCase):
    def test_counts(self):
        pairs = [("pass", "pass"), ("revise", "pass"), ("pass", "revise")]
        labels, m = kappa.confusion(pairs)
        self.assertEqual(labels, ["pass", "revise"])
        self.assertEqual(m["pass"]["pass"], 1)     # human pass, judge pass
        self.assertEqual(m["pass"]["revise"], 1)    # human pass, judge revise
        self.assertEqual(m["revise"]["pass"], 1)    # human revise, judge pass


class TestLoadPairs(unittest.TestCase):
    def _tmp(self, lines):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.addCleanup(os.remove, path)
        return path

    def test_roundtrip_and_blank_lines(self):
        path = self._tmp([
            json.dumps({"judge": "pass", "human": "pass"}),
            "",
            "# a comment",
            json.dumps({"judge": "pass", "human": "revise"}),
        ])
        pairs, rows = kappa.load_pairs(path, "judge", "human")
        self.assertEqual(pairs, [("pass", "pass"), ("pass", "revise")])
        self.assertEqual(len(rows), 2)


class TestScoreCLI(unittest.TestCase):
    def _tmp(self, objs):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(o) for o in objs))
        self.addCleanup(os.remove, path)
        return path

    def test_score_runs_and_reports(self):
        objs = [{"judge": "pass", "human": "pass"}] * 8 + [{"judge": "pass", "human": "revise"}] * 2
        path = self._tmp(objs)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kappa.main(["score", path])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Cohen's kappa", out)
        self.assertIn("agreement: 8/10", out)
        self.assertIn("disagreements", out)


class TestLintCLI(unittest.TestCase):
    def _tmp(self, text):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_good_prompt_passes_all(self):
        good = (
            "Score the answer against this rubric on a scale of 1-5 for each "
            "dimension (correctness, completeness). First explain your reasoning "
            "and cite evidence before giving the score. Do not favor longer answers; "
            "judge concise and verbose equally."
        )
        path = self._tmp(good)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kappa.main(["lint", path])
        self.assertEqual(rc, 0)              # all checks passed
        self.assertIn("4/4 checks passed", buf.getvalue())

    def test_bad_prompt_warns(self):
        bad = "Is this a good answer? Reply pass or fail."
        path = self._tmp(bad)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kappa.main(["lint", path])
        self.assertEqual(rc, 1)              # some checks failed
        self.assertIn("fix:", buf.getvalue())

    def test_numeric_scale_passes_rubric_check(self):
        # regression: "score 0-100" is an explicit scale and must NOT warn on rubric
        p = "Score each segment 0-100 for quality. Reply with JSON."
        path = self._tmp(p)
        buf = io.StringIO()
        with redirect_stdout(buf):
            kappa.main(["lint", path])
        out = buf.getvalue()
        self.assertIn("✓ explicit rubric / scoring scale", out)

    def test_pairwise_without_order_safeguard_warns(self):
        pw = "Compare response A and response B. Which one is better? Use a rubric, scale 1-5, explain reasoning, ignore length."
        path = self._tmp(pw)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kappa.main(["lint", path])
        out = buf.getvalue()
        self.assertIn("position-bias", out)
        self.assertIn("5", out)             # total is now 5 checks (pairwise added)


class TestDriftCLI(unittest.TestCase):
    def _tmp(self, objs):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(o) for o in objs))
        self.addCleanup(os.remove, path)
        return path

    def test_drift_alerts_on_drop(self):
        # baseline: perfect; current: lots of disagreement
        base = self._tmp([{"judge": "p", "human": "p"}] * 10)
        cur = self._tmp([{"judge": "p", "human": "p"}] * 5 + [{"judge": "p", "human": "f"}] * 5)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = kappa.main(["drift", base, cur])
        self.assertEqual(rc, 1)             # alert
        self.assertIn("ALERT", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
