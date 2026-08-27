# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-08-27

The first release after a full audit. Eight confirmed defects are fixed, the
statistics gained the uncertainty they were always missing, and the tool is now
usable as a library and from any CI system.

### Fixed

Every item here was reproduced from a command line against v0.1.1 before being
fixed, and each has a regression test that fails on the old code.

- **`--help` crashed.** A bare `%` in the `score` subparser's help string made
  argparse raise `ValueError: unsupported format character` when the parent
  parser rendered its subcommand list. The first command a new user runs died
  with a traceback.
- **The HTML report interpolated user data unescaped.** Judge labels are model
  output; a label containing `<script>` executed when the report was opened, and
  a label containing `</table>` corrupted the layout. Everything is now passed
  through `html.escape`.
- **`lint` crashed on Windows whenever stdout was not a console.** The `✓`/`⚠`
  glyphs cannot encode in cp1252, so any redirect, pipe or CI runner produced
  `UnicodeEncodeError` — in exactly the context the command exists for. Output
  is now upgraded to UTF-8 where possible, with ASCII markers as a fallback.
- **Four lint patterns could never match.** Truncated stems carried a trailing
  `\b` (`verbos\b`, `decid\b`, `scor\b`, `random(ize)?\b`), so prompts that
  correctly implemented a mitigation were warned for lacking it, and CI failed.
- **`answer 1-5` was misread as a pairwise comparison,** so single-answer rubric
  prompts collected a spurious position-bias warning and exited 1. A digit now
  only signals a side when its twin is present.
- **Degenerate data was certified ship-worthy.** When both raters use a single
  identical label, kappa is 0/0 — undefined. The old code returned `1.0` and
  printed `SHIP-WORTHY` for a gold set containing no negative examples at all.
  It now reports the statistic as undefined and withholds the verdict.
- **Label coercion manufactured disagreements.** JSON `true` became `"True"` and
  never matched a human's `"true"`; `1.0` never matched `"1"`; `null` silently
  became a label category named `"None"`. Booleans and integral floats now
  normalize, and a null label is a clean error naming the file and line.
- **Reading stdin closed the process's stdin.** `kappa drift - -` died with a
  raw traceback at exit 1, indistinguishable from a real drift alert.

### Added

- **Bootstrap confidence intervals** on kappa, reported by default and seeded so
  CI output is reproducible. When the interval straddles the ship bar, the
  verdict is `inconclusive` rather than a false yes or no.
- **Per-label recall and precision**, plus a plain-language callout naming the
  label the judge misses most.
- **Weighted kappa** (linear and quadratic) for ordinal labels, applied
  automatically to numeric scales with more than two levels, and **Spearman's
  rho** as a ranking complement.
- **PABAK** and a skew warning for the kappa prevalence paradox, so a pass-heavy
  gold set is not mistaken for a bad judge.
- **Small-sample warnings** — below 30 rows, thin classes, or an interval wider
  than 0.3.
- **Statistical significance in `drift`** — a bootstrap interval on the delta,
  exact McNemar on rows shared between the two files, and `--strict` to withhold
  alerts that cannot be distinguished from sampling noise.
- **A pure-function library API.** `score()`, `drift()` and `lint()` return
  dicts and never print or exit; `load_pairs` raises `KappaError`. `from kappa
  import score` now works.
- **`--json` on every subcommand**, one versioned schema, treated as a contract.
- **A documented exit-code contract**: 0 clean, 1 gate failed, 2 usage error.
  Plus `--fail-under`, `--ship-bar`, `--quiet`, `--version`.
- **`kappa sample`** — stratified (balanced by default) sampling of rows to
  hand-label, and **`kappa gold`** — a resumable interactive labeling loop.
  Together with `score` they close the loop with no platform involved.
- **Six new lint checks**: output format specified, tie/abstain allowed in
  pairwise prompts, no model identities in the prompt, per-level scale anchors,
  no score-before-reasoning ordering, and reference-guided correctness grading.
  Plus severity levels, stable check ids, `--ignore` and `--strict`.
- **CSV input, dotted nested keys, multiple input files and glob expansion**
  (Windows shells do not expand globs for you).
- **GitHub Actions integration** — a markdown summary is appended to
  `$GITHUB_STEP_SUMMARY` automatically when one is present.
- **Packaging** as `kappa-eval` on PyPI (the name `kappa` is taken by an
  unrelated 2017 tool) with a `kappa` console entry point, a PEP 723 header so
  `uv run kappa.py` works on the raw file, and CI across Python 3.8–3.13 on
  Linux and Windows.

### Changed

Two fixes change numbers that users may have recorded, and are called out here
because no CHANGELOG entry should bury that:

- Degenerate single-label data now reports kappa as **undefined** instead of
  `1.0`. Any recorded `1.0` from such a set was never a real measurement.
- Label normalization changes what counts as agreement for pipelines that logged
  JSON booleans or numbers. Agreement should go **up**, because it was wrong
  before — but the number will move.

## [0.1.1] - 2026-06-30

- Fixed rubric/scale detection for numeric ranges (e.g. `0-100`).

## [0.1.0] - 2026-06-30

- Initial release: `score`, `drift` and `lint`.
