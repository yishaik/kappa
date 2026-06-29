# kappa

A tiny, zero-dependency evaluation kit for **LLM-as-a-Judge**.

If you use an LLM to grade open-ended output (a support reply, a draft, a RAG
answer), you have a judge — and a judge you haven't measured is just a vibe. `kappa`
gives you the three things you actually need, in one stdlib-only Python file:

| command | what it does |
|---|---|
| `kappa score`  | agreement %, **Cohen's κ**, a confusion matrix, and the list of disagreements |
| `kappa drift`  | compare two batches and alert when agreement drops |
| `kappa lint`   | scan your judge **prompt** for the well-known bias pitfalls |

No pip install, no API keys, no network. Python 3.8+.

## Why Cohen's κ, not accuracy

When 90% of your outputs are fine, a judge that blindly says "pass" scores 90%
accuracy and is useless. Cohen's κ discounts the agreement you'd get by chance, so
it stays honest under class imbalance. A common ship bar is **κ ≥ 0.60** (substantial
agreement); recent rubric-grading studies put good judges around κ 0.77–0.87.

You build the gold set the cheap way: log the judge's verdict, and whenever you
*disagree* with it, record your label. A few dozen labeled items is enough to start.

## Usage

```bash
# 1. Score the judge against human labels (JSONL of {judge, human, [id|text]})
python kappa.py score examples/verdicts.jsonl
python kappa.py score verdicts.jsonl --html report.html   # also write an HTML report

# 2. Watch for drift between last month's batch and this month's
python kappa.py drift baseline.jsonl current.jsonl

# 3. Lint a judge prompt for bias pitfalls
python kappa.py lint examples/judge_prompt.txt
cat my_prompt.txt | python kappa.py lint -
```

Custom key names: `--judge-key model_verdict --human-key gold`.

## What `lint` checks (and why)

Each check encodes a known LLM-judge failure mode:

- **explicit rubric / scale** — vague "is this good?" judges are noisy; give criteria + a scale.
- **evidence-before-verdict** — make the judge cite specifics *before* scoring; kills rubber-stamping.
- **decomposed judging** — score one dimension at a time, not one global number.
- **verbosity-bias control** — judges over-reward longer answers unless told not to.
- **position-bias control** — for pairwise prompts, randomize A/B order or score each side independently.

`lint` is heuristic (regex over your prompt text), so treat warnings as prompts to
think, not gospel. Passing all checks ≠ a calibrated judge — that's what `score` is for.

## Input format

JSON Lines, one object per line:

```json
{"id": "t-003", "judge": "pass", "human": "revise", "text": "judge missed a factual error"}
```

`judge` and `human` are the two labels (any categorical values; binary or multi-class).
`id` or `text` is optional and shown next to disagreements for context.

## Test

```bash
python -m unittest test_kappa -v
```

## License

MIT.
