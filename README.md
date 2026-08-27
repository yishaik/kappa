# kappa

A tiny, zero-dependency evaluation kit for **LLM-as-a-Judge**.

Every eval platform tells you to calibrate your judge against human labels.
Almost none of them does it for you — it's either a notebook you assemble
yourself or a feature inside a hosted annotation UI. `kappa` is the offline
post-hoc layer: the thirty-second, no-install, no-API-key way to actually do it
on a JSONL file you already have.

| command | what it does |
|---|---|
| `kappa score`  | agreement %, **Cohen's κ with a confidence interval**, confusion matrix, per-label recall, disagreements |
| `kappa drift`  | compare two batches and alert when agreement drops — and say whether the drop is real |
| `kappa lint`   | scan your judge **prompt** for the well-known bias pitfalls |
| `kappa sample` | draw a stratified set of rows to hand-label |
| `kappa gold`   | resumable interactive labeling loop |

No pip install, no API keys, no network, no LLM calls. Python 3.8+, one file.

## Why Cohen's κ, not accuracy

When 90% of your outputs are fine, a judge that blindly says "pass" scores 90%
accuracy and is useless. Cohen's κ discounts the agreement you'd get by chance,
so it stays honest under class imbalance. A common ship bar is **κ ≥ 0.60**
(substantial agreement).

You build the gold set the cheap way: log the judge's verdict, and whenever you
*disagree* with it, record your label. `kappa sample` and `kappa gold` automate
that loop if you want them.

## Why the interval matters more than the number

Run `score` on this repo's own example file and you get κ = 0.526 — with a 95%
confidence interval of **[−0.10, +1.00]**. Fifteen rows cannot tell a judge worse
than a coin flip apart from a perfect one. Any tool that prints a bare `0.526`
and a ship verdict there is lying to you by omission, so this one says
`INCONCLUSIVE` and tells you to collect more labels.

Rough guidance: tens of rows are directional, ~200 rows make a κ near the ship
bar trustworthy, and you want at least 10 examples of every label.

## Usage

```bash
# 1. Score the judge against human labels (JSONL of {judge, human, [id|text]})
python kappa.py score examples/verdicts.jsonl
python kappa.py score verdicts.jsonl --html report.html   # self-contained HTML report
python kappa.py score verdicts.jsonl --json               # machine-readable

# 2. Watch for drift between last month's batch and this month's
python kappa.py drift baseline.jsonl current.jsonl

# 3. Lint a judge prompt for bias pitfalls
python kappa.py lint examples/judge_prompt.txt
cat my_prompt.txt | python kappa.py lint -

# 4. Build a gold set: draw a balanced sample, then label it
python kappa.py sample judged.jsonl -n 50 -o to_label.jsonl
python kappa.py gold to_label.jsonl -o labels.jsonl
```

```
kappa score - examples/verdicts.jsonl
  rows: 15    agreement: 12/15 = 80.0%
  Cohen's kappa: +0.526  95% CI [-0.10, +1.00]  [moderate]
    -> INCONCLUSIVE: interval spans the 0.60 ship bar
  PABAK: +0.600   (prevalence-adjusted companion)
  confusion (rows=human, cols=judge):
              pass  revise
      pass       9       1
    revise       2       3
  per label:
     label  human  judge  recall  precision
      pass     10     11    0.90       0.82
    revise      5      4    0.60       0.75
    -> judge misses 40% of items a human marked 'revise'
  disagreements (3) - gold for few-shot examples:
    judge='pass' human='revise' - t-003
    judge='revise' human='pass' - t-007
    judge='pass' human='revise' - t-012
  warnings:
    ! n=15 is below 30; treat as directional only
    ! 'revise' has 5 human examples; label at least 10 per class
    ! confidence interval is 1.10 wide; collect more labels before trusting the bar
```

The per-label block is the part you act on. "κ = 0.526" says something is wrong;
"the judge misses 40% of the items a human marked revise" says what, in the
direction that costs you.

Custom key names: `--judge-key model_verdict --human-key gold`. Nested keys work
too (`--judge-key eval.verdict`), as do CSV files and multiple inputs.

## Ordinal scores

If your judge emits 1–5 rather than pass/fail, `score` detects it and switches to
**linear-weighted κ**, where a 2-vs-3 disagreement counts less than a 1-vs-5 one,
and adds Spearman's ρ. Override with `--weights {none,linear,quadratic}`.

## Is the drift real?

A κ drop of 0.10 on 50 rows is usually inside the noise. `drift` reports a
bootstrap interval on the change, runs an exact McNemar test on rows the two
files share (matched by `id`), and tells you plainly whether the drop is
distinguishable from zero. `--strict` withholds the alert when it isn't.

## What `lint` checks (and why)

Each check encodes a documented LLM-judge failure mode and has a stable id you
can `--ignore`:

| id | check |
|---|---|
| `rubric` | explicit criteria and a scale — vague "is this good?" judges are noisy |
| `evidence-order` | cite specifics *before* the verdict; kills rubber-stamping |
| `decomposed` | score one dimension at a time, not one global number |
| `verbosity` | judges over-reward longer answers unless told not to |
| `output-format` | unparseable verdicts silently corrupt an eval run |
| `position-bias` | pairwise only — randomize A/B order or score each side alone |
| `tie-option` | pairwise only — forced choices amplify position bias |
| `scale-anchors` | define each scale point, or judges drift lenient |
| `model-identity` | model names in the prompt shift verdicts; anonymize |
| `score-first` | asking for the score first makes the reasoning post-hoc |
| `reference` | grading correctness without a reference answer |
| `few-shot` | 2–3 calibration examples raise judge consistency |
| `abstain` | a binary verdict with no escape hatch for ambiguous items |

`--strict` promotes suggestions to warnings and adds two weaker heuristics
(leading phrasing, citation bias). `lint` is regex over your prompt text, so
treat warnings as prompts to think, not gospel. Passing every check ≠ a
calibrated judge — that's what `score` is for.

## Input format

JSON Lines, one object per line:

```json
{"id": "t-003", "judge": "pass", "human": "revise", "text": "judge missed a factual error"}
```

`judge` and `human` are the two labels (any categorical or numeric values).
`id` or `text` is optional, shown next to disagreements and used to pair rows in
`drift`. CSV works too, with the header row supplying the keys.

Labels are normalized so a JSON-logging pipeline lines up with typed human
labels: `true` and `"True"` are the same label, as are `1`, `1.0` and `"1"`.
A `null` label is an error naming the file and line, not a silent category.

## CI

Exit codes are the contract: **0** clean, **1** a gate failed, **2** usage or
input error. That's what makes kappa usable from any stack's CI.

```yaml
- uses: actions/setup-python@v5
  with: { python-version: "3.x" }

- name: Judge is still calibrated
  run: pipx run kappa-eval score labels.jsonl --fail-under 0.6

- name: Judge has not drifted
  run: pipx run kappa-eval drift baseline.jsonl current.jsonl

- name: Judge prompt has no known bias pitfalls
  run: pipx run kappa-eval lint prompts/judge.txt
```

When `GITHUB_STEP_SUMMARY` is set, `score` and `drift` append a markdown table to
it automatically, so results render on the Actions run page instead of hiding in
the logs. No wrapper action needed.

For pre-commit:

```yaml
- repo: https://github.com/yishaik/kappa
  rev: v0.4.0
  hooks: [{ id: kappa-lint }]
```

## Not a Python shop?

Your judge harness can be in any language — kappa only reads a text file.

```js
// append one line per judged item; that's the whole integration
fs.appendFileSync("verdicts.jsonl", JSON.stringify({ id, judge, human }) + "\n");
```

Then pick a run path:

1. **No install:** `uvx --from kappa-eval kappa score labels.jsonl`
   (or `pipx run kappa-eval ...`). Needs any Python 3.8+ on PATH.
2. **Vendor it:** `curl -LO https://raw.githubusercontent.com/yishaik/kappa/master/kappa.py`
   — one MIT-licensed file, commit it next to your eval scripts, run
   `python3 kappa.py`. `uv run kappa.py` works too, via the PEP 723 header.
3. **Install it:** `pipx install kappa-eval`.

## As a library

`score`, `drift` and `lint` are pure functions that return dicts and never print
or exit:

```python
from kappa import load_pairs, score

pairs, rows = load_pairs("labels.jsonl", "judge", "human")
res = score(pairs, rows)
assert res["kappa"] >= 0.6, res["verdict"]
```

`--json` emits the same structure from the CLI, under a versioned
`schema_version` that's treated as a contract.

## Non-goals

- **Never calls an LLM.** No API keys, no network. That's the whole point.
- **No multi-rater support.** Exactly two raters, no missing data. If that ever
  changes it'll be Krippendorff's α, not Fleiss' κ, since one coefficient then
  covers every case.
- **No config files.** Flags cover it, and the pitch is no setup.
- **No runtime dependencies, ever.**

If you need multi-annotator agreement, a hosted annotation UI, or an eval runner
that calls models for you, kappa is the wrong tool — reach for promptfoo, Ragas,
Inspect AI, or a platform.

## Test

```bash
python -m unittest discover -v
```

## License

MIT.
