"""Tests for kappa. Run: python -m unittest test_kappa -v   (stdlib only).

Golden values in TestGoldenVectors were cross-checked once, offline, against
scikit-learn's cohen_kappa_score and R's irr::kappa2, then frozen as literals
here so the test suite itself stays zero-dependency.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import kappa


def _tmp(text, suffix=".jsonl"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class _Tmp(unittest.TestCase):
    def tmp(self, text, suffix=".jsonl"):
        path = _tmp(text, suffix)
        self.addCleanup(os.remove, path)
        return path

    def jsonl(self, objs):
        return self.tmp("\n".join(json.dumps(o) for o in objs))

    def run_cli(self, argv):
        buf, err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            rc = kappa.main(argv)
        return rc, buf.getvalue(), err.getvalue()


# ---------------------------------------------------------------- statistics


class TestCohenKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertAlmostEqual(kappa.cohen_kappa([("a", "a"), ("b", "b"), ("a", "a")]), 1.0)

    def test_known_negative_case(self):
        pairs = [("pass", "pass"), ("revise", "pass"), ("pass", "revise")]
        self.assertAlmostEqual(kappa.cohen_kappa(pairs), -0.5, places=6)

    def test_known_2x2_case(self):
        pairs = [("yes", "yes")] * 7 + [("no", "no")] + [("yes", "no")] + [("no", "yes")]
        self.assertAlmostEqual(kappa.cohen_kappa(pairs), 0.375, places=6)

    def test_empty(self):
        self.assertIsNone(kappa.cohen_kappa([]))

    def test_multiclass(self):
        pairs = [("x", "x"), ("y", "y"), ("z", "z"), ("x", "y")]
        k = kappa.cohen_kappa(pairs)
        self.assertTrue(-1.0 <= k <= 1.0)
        self.assertLess(k, 1.0)

    def test_label_permutation_invariance(self):
        pairs = [("a", "a")] * 6 + [("b", "b")] * 3 + [("a", "b")] * 2
        renamed = [({"a": "x", "b": "y"}[j], {"a": "x", "b": "y"}[h]) for j, h in pairs]
        self.assertAlmostEqual(kappa.cohen_kappa(pairs), kappa.cohen_kappa(renamed))

    def test_bounded(self):
        for pairs in (
            [("a", "b")] * 5 + [("b", "a")] * 5,
            [("a", "a")] * 9 + [("b", "a")],
            [("1", "5"), ("5", "1"), ("3", "3")],
        ):
            k = kappa.cohen_kappa(pairs)
            if k is not None:
                self.assertTrue(-1.0 <= k <= 1.0, pairs)


class TestGoldenVectors(unittest.TestCase):
    """Frozen values cross-checked against scikit-learn and R's irr package."""

    def test_unweighted(self):
        pairs = [("yes", "yes")] * 7 + [("no", "no")] + [("yes", "no")] + [("no", "yes")]
        self.assertAlmostEqual(kappa.cohen_kappa(pairs), 0.375, places=6)

    def test_linear_and_quadratic_ordinal(self):
        # 3-level ordinal. Recomputed independently in exact rational arithmetic:
        # linear = 17/22, quadratic = 47/67.
        pairs = [("1", "1")] * 4 + [("2", "2")] * 3 + [("3", "3")] * 2 + [("1", "3")]
        self.assertAlmostEqual(kappa.cohen_kappa(pairs, "linear"), 17 / 22, places=12)
        self.assertAlmostEqual(kappa.cohen_kappa(pairs, "quadratic"), 47 / 67, places=12)

    def test_weighted_equals_unweighted_on_two_levels(self):
        # With two ordinal levels, linear weighting reduces to plain kappa.
        pairs = [("1", "1")] * 7 + [("2", "2")] + [("1", "2")] + [("2", "1")]
        self.assertAlmostEqual(
            kappa.cohen_kappa(pairs, "linear"), kappa.cohen_kappa(pairs), places=9
        )

    def test_quadratic_punishes_distance_less_than_linear(self):
        pairs = [("1", "1")] * 5 + [("5", "5")] * 4 + [("1", "2")]
        self.assertGreater(
            kappa.cohen_kappa(pairs, "quadratic"), kappa.cohen_kappa(pairs, "linear")
        )

    def test_pabak(self):
        pairs = [("a", "a")] * 8 + [("a", "b")] * 2
        self.assertAlmostEqual(kappa.pabak(pairs), 0.6, places=9)

    def test_spearman_perfect_and_none(self):
        pairs = [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")]
        self.assertAlmostEqual(kappa.spearman(pairs), 1.0, places=9)
        self.assertIsNone(kappa.spearman([("a", "b"), ("c", "d"), ("e", "f")]))

    def test_mcnemar_exact_matches_binomial(self):
        # b=8, c=1 -> two-sided exact p = 2 * sum_{i<=1} C(9,i) / 2^9 = 0.0390625
        self.assertAlmostEqual(kappa._mcnemar_exact(8, 1), 0.0390625, places=9)
        self.assertEqual(kappa._mcnemar_exact(0, 0), 1.0)


class TestBand(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(kappa.band(1.0), "almost perfect")
        self.assertEqual(kappa.band(0.9), "almost perfect")
        self.assertEqual(kappa.band(0.7), "substantial")
        self.assertEqual(kappa.band(0.5), "moderate")
        self.assertEqual(kappa.band(0.3), "fair")
        self.assertEqual(kappa.band(0.1), "slight")
        self.assertEqual(kappa.band(-0.2), "poor (worse than chance)")
        self.assertEqual(kappa.band(None), "undefined")


class TestBootstrap(unittest.TestCase):
    def test_seeded_and_reproducible(self):
        pairs = [("a", "a")] * 8 + [("a", "b")] * 2 + [("b", "b")] * 5
        a = kappa.bootstrap_ci(pairs, kappa.cohen_kappa, n_boot=200, seed=7)
        b = kappa.bootstrap_ci(pairs, kappa.cohen_kappa, n_boot=200, seed=7)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], a[1])

    def test_brackets_point_estimate_on_clean_data(self):
        pairs = [("a", "a")] * 20 + [("b", "b")] * 20
        ci = kappa.bootstrap_ci(pairs, kappa.cohen_kappa, n_boot=300, seed=1)
        self.assertIsNotNone(ci)
        self.assertLessEqual(ci[0], 1.0)


class TestConfusion(unittest.TestCase):
    def test_counts(self):
        pairs = [("pass", "pass"), ("revise", "pass"), ("pass", "revise")]
        labels, m = kappa.confusion(pairs)
        self.assertEqual(labels, ["pass", "revise"])
        self.assertEqual(m["pass"]["pass"], 1)
        self.assertEqual(m["pass"]["revise"], 1)
        self.assertEqual(m["revise"]["pass"], 1)

    def test_numeric_labels_sort_numerically(self):
        pairs = [("10", "10"), ("2", "2"), ("1", "1")]
        labels, _ = kappa.confusion(pairs)
        self.assertEqual(labels, ["1", "2", "10"])

    def test_per_label_recall_precision(self):
        # human pass=10 (judge got 9), human revise=5 (judge got 3)
        pairs = (
            [("pass", "pass")] * 9
            + [("revise", "pass")]
            + [("revise", "revise")] * 3
            + [("pass", "revise")] * 2
        )
        rows = {r["label"]: r for r in kappa.per_label_stats(pairs)}
        self.assertAlmostEqual(rows["pass"]["recall"], 0.9)
        self.assertAlmostEqual(rows["revise"]["recall"], 0.6)
        self.assertAlmostEqual(rows["pass"]["precision"], 9 / 11)
        self.assertAlmostEqual(rows["revise"]["precision"], 0.75)


# ---------------------------------------------------------------- regressions
# One class per defect found in the v0.1 audit. Each fails on the old code.


class TestRegressionHelpCrash(_Tmp):
    """--help %-formatted the subcommand help strings and raised ValueError."""

    def test_parent_help_does_not_crash(self):
        for argv in (["--help"], ["-h"]):
            with self.assertRaises(SystemExit) as cm:
                self.run_cli(argv)
            self.assertEqual(cm.exception.code, 0)

    def test_every_subparser_help_renders(self):
        for cmd in ("score", "drift", "lint", "sample", "gold"):
            with self.assertRaises(SystemExit) as cm:
                self.run_cli([cmd, "--help"])
            self.assertEqual(cm.exception.code, 0, cmd)

    def test_help_text_actually_formats(self):
        # Render through the formatter directly - that is where the crash lived.
        text = kappa.build_parser().format_help()
        self.assertIn("score", text)
        self.assertIn("drift", text)


class TestRegressionHtmlEscaping(_Tmp):
    """Judge labels are model output and were interpolated into HTML raw."""

    def test_script_tag_is_escaped(self):
        path = self.jsonl(
            [{"judge": "<script>alert(1)</script>", "human": "pass", "id": "<img src=x onerror=alert(2)>"}] * 4
            + [{"judge": "pass", "human": "pass"}] * 4
        )
        out = self.tmp("", suffix=".html")
        rc, _, _ = self.run_cli(["score", path, "--html", out, "--boot", "0", "--quiet"])
        self.assertEqual(rc, 0)
        with open(out, encoding="utf-8") as f:
            body = f.read()
        self.assertNotIn("<script>alert(1)</script>", body)
        self.assertNotIn("<img src=x onerror=alert(2)>", body)
        self.assertIn("&lt;script&gt;", body)

    def test_closing_tag_label_cannot_break_the_table(self):
        path = self.jsonl([{"judge": "</table>", "human": "pass"}] * 3 + [{"judge": "pass", "human": "pass"}] * 3)
        out = self.tmp("", suffix=".html")
        self.run_cli(["score", path, "--html", out, "--boot", "0", "--quiet"])
        with open(out, encoding="utf-8") as f:
            body = f.read()
        self.assertNotIn("</table>1", body)
        self.assertIn("&lt;/table&gt;", body)


class TestRegressionEncoding(_Tmp):
    """lint printed non-cp1252 glyphs and crashed whenever stdout was a pipe."""

    def test_ascii_fallback_when_stream_cannot_encode(self):
        # A stream that cannot be upgraded to UTF-8 (no reconfigure) and reports
        # a codepage the glyphs do not fit in must fall back to ASCII markers.
        class Legacy:
            encoding = "cp1252"

        marks = kappa._init_stream(Legacy())
        self.assertEqual(marks["ok"], "OK ")
        for value in marks.values():
            value.encode("cp1252")  # must not raise

    def test_reconfigurable_stream_is_upgraded_to_utf8(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        marks = kappa._init_stream(stream)
        self.assertEqual(stream.encoding, "utf-8")
        self.assertEqual(marks["ok"], "✓ ")

    def test_lint_survives_a_cp1252_pipe_end_to_end(self):
        prompt = self.tmp("Is this a good answer? Reply pass or fail.", suffix=".txt")
        env = dict(os.environ)
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
        env["PYTHONLEGACYWINDOWSSTDIO"] = "1"  # forces the ANSI codepage on Windows
        proc = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kappa.py"),
             "lint", prompt],
            capture_output=True, env=env,
        )
        self.assertNotIn(b"UnicodeEncodeError", proc.stderr)
        self.assertIn(b"checks passed", proc.stdout)


class TestRegressionLintStems(_Tmp):
    """Truncated stems carried a trailing \\b, so they could never match."""

    def _lint(self, text):
        path = self.tmp(text, suffix=".txt")
        rc, out, _ = self.run_cli(["lint", path, "--json"])
        return rc, json.loads(out)

    def _status(self, res, check_id):
        for f in res["findings"]:
            if f["id"] == check_id:
                return f["status"]
        return "skipped"

    def test_verbosity_word_passes(self):
        for text in ("Do not reward verbosity in the reply.", "Ignore how verbose the answer is."):
            _, res = self._lint(text)
            self.assertEqual(self._status(res, "verbosity"), "pass", text)

    def test_decide_and_scoring_pass_evidence_check(self):
        for text in (
            "Quote the exact sentence before you decide PASS or FAIL.",
            "List the errors before scoring the reply.",
        ):
            _, res = self._lint(text)
            self.assertEqual(self._status(res, "evidence-order"), "pass", text)

    def test_randomizing_passes_position_check(self):
        _, res = self._lint(
            "Compare response A and response B, randomizing their order each time. Which is better?"
        )
        self.assertEqual(self._status(res, "position-bias"), "pass")

    def test_scoring_passes_rubric_check(self):
        _, res = self._lint("Follow the scoring guide below.")
        self.assertEqual(self._status(res, "rubric"), "pass")


class TestRegressionPairwiseDetection(_Tmp):
    def _ctx(self, text):
        return kappa.lint(text)["context"]["pairwise"]

    def test_numeric_scale_is_not_pairwise(self):
        self.assertFalse(self._ctx("Rate the answer 1-5 against the rubric."))
        self.assertFalse(self._ctx("Score each segment 0-100 for quality."))
        self.assertFalse(self._ctx("Pick option 2 from the menu below."))

    def test_real_pairwise_still_detected(self):
        self.assertTrue(self._ctx("Compare response A and response B. Which one is better?"))
        self.assertTrue(self._ctx("You get answer 1 and answer 2; choose the stronger."))
        self.assertTrue(self._ctx("Compare the two and explain."))

    def test_single_answer_rubric_prompt_passes_cleanly(self):
        path = self.tmp(
            "Rate the answer 1-5 against the rubric. 1 = wrong, 3 = partial, 5 = complete. "
            "Explain your reasoning and cite evidence first, then give the score. "
            "Judge each dimension separately. Do not favor longer answers. "
            "Respond only with JSON.",
            suffix=".txt",
        )
        rc, out, _ = self.run_cli(["lint", path])
        self.assertEqual(rc, 0, out)
        self.assertNotIn("position-bias", out)


class TestRegressionUndefinedKappa(_Tmp):
    """pe==1 means kappa is 0/0; the old code returned 1.0 and shipped it."""

    def test_single_label_both_sides_is_undefined(self):
        self.assertIsNone(kappa.cohen_kappa([("pass", "pass")] * 5))

    def test_score_refuses_to_certify_zero_signal_data(self):
        path = self.jsonl([{"judge": "pass", "human": "pass"}] * 20)
        rc, out, _ = self.run_cli(["score", path, "--boot", "0"])
        self.assertIn("UNDEFINED", out)
        self.assertNotIn("SHIP-WORTHY", out)
        res = kappa.score([("pass", "pass")] * 20, boot=0)
        self.assertIsNone(res["kappa"])
        self.assertEqual(res["verdict"], "undefined")
        self.assertIsNone(res["ship_bar_met"])

    def test_fail_under_treats_undefined_as_failure(self):
        path = self.jsonl([{"judge": "pass", "human": "pass"}] * 20)
        rc, _, _ = self.run_cli(["score", path, "--boot", "0", "--fail-under", "0.6", "--quiet"])
        self.assertEqual(rc, 1)

    def test_drift_refuses_a_degenerate_baseline(self):
        base = self.jsonl([{"judge": "p", "human": "p"}] * 10)
        cur = self.jsonl([{"judge": "p", "human": "p"}] * 5 + [{"judge": "p", "human": "f"}] * 5)
        rc, out, _ = self.run_cli(["drift", base, cur, "--boot", "0"])
        self.assertEqual(rc, 1)
        self.assertIn("undefined", out)


class TestRegressionLabelNormalization(_Tmp):
    """JSON booleans/numbers must land on the same label as the typed word."""

    def test_bool_matches_string(self):
        pairs, _ = kappa.load_pairs(
            self.jsonl([{"judge": True, "human": "true"}] * 12), "judge", "human"
        )
        self.assertEqual(set(pairs), {("true", "true")})
        res = kappa.score(pairs, boot=0)
        self.assertEqual(res["agreement"], 12)

    def test_float_matches_int_string(self):
        pairs, _ = kappa.load_pairs(
            self.jsonl([{"judge": 1.0, "human": "1"}] * 5), "judge", "human"
        )
        self.assertEqual(set(pairs), {("1", "1")})

    def test_whitespace_and_case_of_bools(self):
        pairs, _ = kappa.load_pairs(
            self.jsonl([{"judge": "  True ", "human": False}]), "judge", "human"
        )
        self.assertEqual(pairs, [("true", "false")])

    def test_null_label_is_a_clean_error_not_a_category(self):
        path = self.jsonl([{"judge": None, "human": "pass"}])
        with self.assertRaises(kappa.KappaError) as cm:
            kappa.load_pairs(path, "judge", "human")
        self.assertIn("null", str(cm.exception))
        rc, _, err = self.run_cli(["score", path])
        self.assertEqual(rc, 2)
        self.assertIn("error:", err)

    def test_nonscalar_label_rejected(self):
        path = self.jsonl([{"judge": {"a": 1}, "human": "pass"}])
        with self.assertRaises(kappa.KappaError):
            kappa.load_pairs(path, "judge", "human")


class TestRegressionStdin(_Tmp):
    """load_pairs('-') closed the process's real stdin."""

    def test_stdin_is_not_closed(self):
        data = "\n".join(json.dumps({"judge": "a", "human": "a"}) for _ in range(3))
        real = sys.stdin
        sys.stdin = io.StringIO(data)
        try:
            kappa.load_pairs("-", "judge", "human")
            self.assertFalse(sys.stdin.closed)
        finally:
            sys.stdin = real

    def test_second_stdin_read_gives_a_clean_error_not_a_traceback(self):
        data = "\n".join(json.dumps({"judge": "a", "human": "b"}) for _ in range(3))
        real = sys.stdin
        sys.stdin = io.StringIO(data)
        try:
            rc, _, err = self.run_cli(["drift", "-", "-", "--boot", "0"])
        finally:
            sys.stdin = real
        self.assertIn(rc, (1, 2))
        self.assertNotIn("Traceback", err)


# ---------------------------------------------------------------- api + cli


class TestLoadPairs(_Tmp):
    def test_roundtrip_and_blank_lines(self):
        path = self.tmp(
            "\n".join(
                [
                    json.dumps({"judge": "pass", "human": "pass"}),
                    "",
                    "# a comment",
                    json.dumps({"judge": "pass", "human": "revise"}),
                ]
            )
        )
        pairs, rows = kappa.load_pairs(path, "judge", "human")
        self.assertEqual(pairs, [("pass", "pass"), ("pass", "revise")])
        self.assertEqual(len(rows), 2)

    def test_missing_key_names_the_file_and_line(self):
        path = self.jsonl([{"judge": "pass"}])
        with self.assertRaises(kappa.KappaError) as cm:
            kappa.load_pairs(path, "judge", "human")
        self.assertIn(":1:", str(cm.exception))

    def test_bad_json_names_the_line(self):
        path = self.tmp('{"judge": "a", "human": "a"}\nnot json\n')
        with self.assertRaises(kappa.KappaError) as cm:
            kappa.load_pairs(path, "judge", "human")
        self.assertIn(":2:", str(cm.exception))

    def test_csv_input(self):
        path = self.tmp("judge,human,id\npass,pass,t1\npass,revise,t2\n", suffix=".csv")
        pairs, rows = kappa.load_pairs(path, "judge", "human")
        self.assertEqual(pairs, [("pass", "pass"), ("pass", "revise")])
        self.assertEqual(rows[1]["id"], "t2")

    def test_nested_dotted_keys(self):
        path = self.jsonl([{"eval": {"verdict": "pass"}, "gold": "revise"}])
        pairs, _ = kappa.load_pairs(path, "eval.verdict", "gold")
        self.assertEqual(pairs, [("pass", "revise")])

    def test_literal_dotted_key_wins_over_path(self):
        path = self.jsonl([{"a.b": "pass", "human": "pass"}])
        pairs, _ = kappa.load_pairs(path, "a.b", "human")
        self.assertEqual(pairs, [("pass", "pass")])


class TestScoreAPI(unittest.TestCase):
    def test_returns_data_and_does_not_print(self):
        pairs = [("pass", "pass")] * 8 + [("pass", "revise")] * 2
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = kappa.score(pairs, boot=0)
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(res["n"], 10)
        self.assertEqual(res["agreement"], 8)
        self.assertEqual(res["schema_version"], kappa.SCHEMA_VERSION)

    def test_empty_raises(self):
        with self.assertRaises(kappa.KappaError):
            kappa.score([])

    def test_warns_on_small_n_and_skew(self):
        res = kappa.score([("a", "a")] * 19 + [("b", "b")], boot=0)
        joined = " ".join(res["warnings"])
        self.assertIn("below 30", joined)
        self.assertIn("prevalence paradox", joined)

    def test_inconclusive_when_ci_straddles_the_bar(self):
        # 15 rows, kappa ~0.53 with a very wide interval -> not a ship decision
        pairs = (
            [("pass", "pass")] * 9
            + [("revise", "pass")]
            + [("revise", "revise")] * 3
            + [("pass", "revise")] * 2
        )
        res = kappa.score(pairs, boot=500, seed=7)
        self.assertEqual(res["verdict"], "inconclusive")
        self.assertIsNotNone(res["kappa_ci"])
        self.assertLess(res["kappa_ci"][0], 0.6)

    def test_ordinal_labels_get_weighted_by_default(self):
        pairs = [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "4"), ("2", "3")] * 4
        res = kappa.score(pairs, boot=0)
        self.assertEqual(res["weights"], "linear")
        self.assertIsNotNone(res["weighted_kappa"])
        self.assertIsNotNone(res["spearman"])

    def test_binary_labels_stay_unweighted(self):
        res = kappa.score([("pass", "pass")] * 5 + [("pass", "revise")] * 5, boot=0)
        self.assertEqual(res["weights"], "none")


class TestDriftAPI(unittest.TestCase):
    def _mix(self, good, bad):
        return [("p", "p")] * good + [("f", "f")] * good + [("p", "f")] * bad

    def test_alerts_on_a_real_drop(self):
        base = self._mix(20, 1)
        cur = self._mix(10, 15)
        res = kappa.drift(base, cur, boot=200, seed=1)
        self.assertTrue(res["alerts"])
        self.assertLess(res["delta"], 0)

    def test_quiet_when_stable(self):
        res = kappa.drift(self._mix(20, 2), self._mix(20, 3), boot=200, seed=1)
        self.assertEqual(res["alerts"], [])
        self.assertEqual(res["verdict"], "ok")

    def test_strict_withholds_an_alert_that_is_noise(self):
        base = [("p", "p")] * 5 + [("f", "f")] * 4 + [("p", "f")]
        cur = [("p", "p")] * 5 + [("f", "f")] * 3 + [("p", "f")] * 2
        loose = kappa.drift(base, cur, threshold=0.05, boot=400, seed=3)
        strict = kappa.drift(base, cur, threshold=0.05, boot=400, seed=3, strict=True)
        if loose["alerts"] and loose["significant"] is False:
            self.assertEqual(strict["alerts"], [])

    def test_mcnemar_runs_on_shared_ids(self):
        base_rows = [{"id": i} for i in range(10)]
        cur_rows = [{"id": i} for i in range(10)]
        base = [("p", "p")] * 9 + [("p", "f")]
        cur = [("p", "p")] * 3 + [("p", "f")] * 7
        res = kappa.drift(base, cur, boot=0, base_rows=base_rows, cur_rows=cur_rows)
        self.assertIsNotNone(res["mcnemar"])
        self.assertEqual(res["mcnemar"]["paired"], 10)
        self.assertGreater(res["mcnemar"]["lost"], 0)


class TestLintAPI(unittest.TestCase):
    def test_good_prompt_passes_everything(self):
        good = (
            "Score the answer against this rubric on a scale of 1-5 for each dimension "
            "(correctness, completeness). 1 = wrong, 3 = partial, 5 = complete. "
            "First explain your reasoning and cite evidence before giving the score. "
            "Do not favor longer answers; judge concise and verbose equally. "
            "Respond only with JSON."
        )
        res = kappa.lint(good)
        failed = [f["id"] for f in res["findings"] if f["status"] == "fail" and f["severity"] == "warn"]
        self.assertEqual(failed, [])
        self.assertEqual(res["verdict"], "pass")

    def test_bad_prompt_fails(self):
        res = kappa.lint("Is this a good answer? Reply pass or fail.")
        self.assertEqual(res["verdict"], "fail")
        self.assertGreater(res["failed"], 0)

    def test_numeric_scale_passes_rubric_check(self):
        res = kappa.lint("Score each segment 0-100 for quality. Reply with JSON.")
        status = {f["id"]: f["status"] for f in res["findings"]}
        self.assertEqual(status["rubric"], "pass")

    def test_model_identity_leak_is_flagged(self):
        res = kappa.lint("Decide whether GPT-4o or Claude wrote the better response A or response B.")
        status = {f["id"]: f["status"] for f in res["findings"]}
        self.assertEqual(status["model-identity"], "fail")

    def test_score_first_ordering_is_flagged(self):
        res = kappa.lint("First give the score, then explain why. Use the rubric 1-5.")
        status = {f["id"]: f["status"] for f in res["findings"]}
        self.assertEqual(status["score-first"], "fail")

    def test_pairwise_gates_add_two_checks(self):
        plain = kappa.lint("Rate the answer 1-5 using the rubric.")
        pw = kappa.lint("Compare response A and response B. Which one is better?")
        ids_plain = set(f["id"] for f in plain["findings"])
        ids_pw = set(f["id"] for f in pw["findings"])
        self.assertNotIn("tie-option", ids_plain)
        self.assertIn("tie-option", ids_pw)
        self.assertIn("position-bias", ids_pw)

    def test_strict_enables_extra_checks_and_promotes_info(self):
        text = "I think this is an excellent response. Rate it 1-5 with a rubric."
        loose = kappa.lint(text)
        strict = kappa.lint(text, strict=True)
        self.assertNotIn("sycophancy", [f["id"] for f in loose["findings"]])
        self.assertIn("sycophancy", [f["id"] for f in strict["findings"]])
        self.assertGreater(strict["total"], loose["total"])

    def test_ignore_skips_a_check(self):
        res = kappa.lint("Is this good?", ignore=("rubric",))
        self.assertNotIn("rubric", [f["id"] for f in res["findings"]])

    def test_stable_ids_are_unique(self):
        ids = [c["id"] for c in kappa._CHECKS]
        self.assertEqual(len(ids), len(set(ids)))


class TestCLI(_Tmp):
    def test_version(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_cli(["--version"])
        self.assertEqual(cm.exception.code, 0)

    def test_score_json_is_parseable_and_stable(self):
        path = self.jsonl([{"judge": "pass", "human": "pass"}] * 8 + [{"judge": "pass", "human": "revise"}] * 2)
        rc, out, _ = self.run_cli(["score", path, "--json", "--boot", "0"])
        self.assertEqual(rc, 0)
        res = json.loads(out)
        for key in ("kappa", "n", "agreement", "confusion", "per_label", "verdict", "schema_version"):
            self.assertIn(key, res)

    def test_score_text_report(self):
        path = self.jsonl([{"judge": "pass", "human": "pass"}] * 8 + [{"judge": "pass", "human": "revise"}] * 2)
        rc, out, _ = self.run_cli(["score", path, "--boot", "0"])
        self.assertEqual(rc, 0)
        self.assertIn("agreement: 8/10", out)
        self.assertIn("per label", out)
        self.assertIn("disagreements", out)

    def test_fail_under_gates(self):
        path = self.jsonl([{"judge": "pass", "human": "pass"}] * 8 + [{"judge": "revise", "human": "revise"}] * 8)
        rc_ok, _, _ = self.run_cli(["score", path, "--fail-under", "0.6", "--quiet", "--boot", "0"])
        self.assertEqual(rc_ok, 0)
        noisy = self.jsonl(
            [{"judge": "pass", "human": "pass"}] * 5
            + [{"judge": "pass", "human": "revise"}] * 5
            + [{"judge": "revise", "human": "pass"}] * 5
        )
        rc_bad, _, _ = self.run_cli(["score", noisy, "--fail-under", "0.6", "--quiet", "--boot", "0"])
        self.assertEqual(rc_bad, 1)

    def test_multiple_files_are_pooled(self):
        a = self.jsonl([{"judge": "pass", "human": "pass"}] * 4)
        b = self.jsonl([{"judge": "revise", "human": "revise"}] * 4)
        rc, out, _ = self.run_cli(["score", a, b, "--boot", "0"])
        self.assertEqual(rc, 0)
        self.assertIn("rows: 8", out)

    def test_quiet_prints_nothing(self):
        path = self.jsonl([{"judge": "pass", "human": "pass"}] * 5 + [{"judge": "pass", "human": "revise"}] * 5)
        rc, out, _ = self.run_cli(["score", path, "--quiet", "--boot", "0"])
        self.assertEqual(out, "")

    def test_usage_error_exits_2(self):
        rc, _, err = self.run_cli(["score", "no_such_file_xyz.jsonl"])
        self.assertEqual(rc, 2)
        self.assertIn("error:", err)

    def test_drift_cli_alert_exit_code(self):
        base = self.jsonl([{"judge": "p", "human": "p"}] * 15 + [{"judge": "f", "human": "f"}] * 15)
        cur = self.jsonl(
            [{"judge": "p", "human": "p"}] * 8 + [{"judge": "f", "human": "f"}] * 8
            + [{"judge": "p", "human": "f"}] * 14
        )
        rc, out, _ = self.run_cli(["drift", base, cur, "--boot", "200"])
        self.assertEqual(rc, 1)
        self.assertIn("ALERT", out)

    def test_lint_json_and_exit_codes(self):
        bad = self.tmp("Is this a good answer?", suffix=".txt")
        rc, out, _ = self.run_cli(["lint", bad, "--json"])
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads(out)["verdict"], "fail")

    def test_lint_unknown_ignore_id_errors(self):
        p = self.tmp("anything", suffix=".txt")
        rc, _, err = self.run_cli(["lint", p, "--ignore", "nope"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown check id", err)

    def test_github_step_summary_is_written(self):
        path = self.jsonl([{"judge": "pass", "human": "pass"}] * 6 + [{"judge": "pass", "human": "revise"}] * 4)
        summary = self.tmp("", suffix=".md")
        os.environ["GITHUB_STEP_SUMMARY"] = summary
        try:
            self.run_cli(["score", path, "--quiet", "--boot", "0"])
        finally:
            del os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("kappa score", body)
        self.assertIn("| kappa |", body)
        self.assertIn("| rows | 10 |", body)

    def test_step_summary_kappa_cell_is_readable(self):
        # regression: the CI was concatenated onto the value ("+0.52695% CI")
        path = self.jsonl(
            [{"judge": "pass", "human": "pass"}] * 12 + [{"judge": "pass", "human": "revise"}] * 8
        )
        summary = self.tmp("", suffix=".md")
        os.environ["GITHUB_STEP_SUMMARY"] = summary
        try:
            self.run_cli(["score", path, "--quiet", "--boot", "200", "--seed", "1"])
        finally:
            del os.environ["GITHUB_STEP_SUMMARY"]
        with open(summary, encoding="utf-8") as f:
            cell = [line for line in f if line.startswith("| kappa |")][0]
        self.assertNotIn("95%", cell.split("95%")[0].rstrip()[-3:])
        self.assertRegex(cell, r"\| kappa \| [+-]\d\.\d{3} 95% CI \[")


class TestSampleAndGold(_Tmp):
    def test_sample_is_balanced_and_seeded(self):
        rows = [{"id": i, "judge": "pass"} for i in range(90)] + [
            {"id": 100 + i, "judge": "revise"} for i in range(10)
        ]
        path = self.jsonl(rows)
        rc, out, _ = self.run_cli(["sample", path, "-n", "10", "--seed", "1"])
        self.assertEqual(rc, 0)
        picked = [json.loads(line) for line in out.strip().splitlines()]
        self.assertEqual(len(picked), 10)
        counts = {}
        for r in picked:
            counts[r["judge"]] = counts.get(r["judge"], 0) + 1
        self.assertEqual(counts.get("revise"), 5)  # balanced, not 1-in-10
        rc2, out2, _ = self.run_cli(["sample", path, "-n", "10", "--seed", "1"])
        self.assertEqual(out, out2)

    def test_sample_proportional_mode(self):
        rows = [{"id": i, "judge": "pass"} for i in range(90)] + [
            {"id": 100 + i, "judge": "revise"} for i in range(10)
        ]
        path = self.jsonl(rows)
        _, out, _ = self.run_cli(["sample", path, "-n", "10", "--proportional", "--seed", "1"])
        picked = [json.loads(line) for line in out.strip().splitlines()]
        counts = {}
        for r in picked:
            counts[r["judge"]] = counts.get(r["judge"], 0) + 1
        self.assertEqual(counts.get("pass"), 9)

    def test_sample_cannot_exceed_a_stratum(self):
        path = self.jsonl([{"id": i, "judge": "pass"} for i in range(3)])
        _, out, _ = self.run_cli(["sample", path, "-n", "50"])
        self.assertEqual(len(out.strip().splitlines()), 3)

    def test_gold_resumes_and_defaults_to_agreement(self):
        src = self.jsonl([{"id": "a", "judge": "pass"}, {"id": "b", "judge": "revise"}])
        out = self.tmp("", suffix=".jsonl")
        real = sys.stdin
        sys.stdin = io.StringIO("\nrevise\n")  # enter = agree, then an explicit label
        try:
            rc, _, _ = self.run_cli(["gold", src, "-o", out])
        finally:
            sys.stdin = real
        self.assertEqual(rc, 0)
        with open(out, encoding="utf-8") as f:
            got = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(got[0]["human"], "pass")
        self.assertEqual(got[1]["human"], "revise")

        # second run: both ids are already present, so nothing is left to do
        rc2, out2, _ = self.run_cli(["gold", src, "-o", out])
        self.assertEqual(rc2, 0)
        self.assertIn("nothing left to label", out2)


class TestBundledExamples(unittest.TestCase):
    """The shipped examples must keep working - they are the first thing anyone runs."""

    def setUp(self):
        self.here = os.path.dirname(os.path.abspath(__file__))

    def test_score_examples(self):
        path = os.path.join(self.here, "examples", "verdicts.jsonl")
        pairs, rows = kappa.load_pairs(path, "judge", "human")
        res = kappa.score(pairs, rows, boot=500, seed=7)
        self.assertEqual(res["n"], 15)
        self.assertEqual(res["agreement"], 12)
        self.assertAlmostEqual(res["kappa"], 0.5263157894, places=6)
        self.assertEqual(res["verdict"], "inconclusive")

    def test_lint_examples_prompt_still_warns(self):
        path = os.path.join(self.here, "examples", "judge_prompt.txt")
        with open(path, encoding="utf-8") as f:
            res = kappa.lint(f.read())
        self.assertEqual(res["verdict"], "fail")

    def test_good_prompt_example_passes(self):
        path = os.path.join(self.here, "examples", "judge_prompt_good.txt")
        with open(path, encoding="utf-8") as f:
            res = kappa.lint(f.read())
        self.assertEqual(res["verdict"], "pass", [f for f in res["findings"] if f["status"] == "fail"])


if __name__ == "__main__":
    unittest.main()
