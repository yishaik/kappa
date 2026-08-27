#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
"""kappa - a tiny, zero-dependency evaluation kit for LLM-as-a-Judge.

Three things every team running an LLM judge needs, and there's no clean little
tool for: measure whether the judge agrees with humans, watch that agreement for
drift, and catch the well-known bias pitfalls in the judge prompt itself.

Subcommands
-----------
  kappa score  <labels.jsonl>                  agreement, Cohen's kappa + CI, confusion, per-label
  kappa drift  <baseline.jsonl> <new.jsonl>    compare two batches; flag a kappa drop
  kappa lint   <judge-prompt.txt | ->          heuristic linter for judge-prompt bias pitfalls
  kappa sample <judged.jsonl>                  stratified sample of rows to hand-label
  kappa gold   <judged.jsonl>                  interactive labeling loop, resumable

Input format for score/drift: JSON Lines (or CSV), one record per line, each with
a judge label and a human label (defaults: keys "judge" and "human"; override with
flags). An optional "id" or "text" field is shown next to disagreements for context.

Why kappa, not raw accuracy: Cohen's kappa discounts agreement that would happen
by chance, so it stays honest under class imbalance (the usual "90% of replies are
fine" case). A common ship bar is kappa >= 0.6 (substantial agreement).

Why the interval matters: on a few dozen rows a point estimate of kappa is noisy
enough to be meaningless on its own, so score reports a bootstrap confidence
interval and says "inconclusive" when that interval straddles the ship bar.

Exit codes: 0 = clean, 1 = a gate failed (lint warning, drift alert, --fail-under),
2 = usage or input error.

Zero dependencies - Python 3.8+ standard library only.
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import html
import json
import math
import os
import random
import re
import sys
from collections import Counter
from contextlib import nullcontext

__version__ = "0.4.0"

SCHEMA_VERSION = 1

__all__ = [
    "KappaError",
    "cohen_kappa",
    "band",
    "confusion",
    "pabak",
    "spearman",
    "bootstrap_ci",
    "per_label_stats",
    "load_pairs",
    "load_rows",
    "score",
    "drift",
    "lint",
    "main",
    "__version__",
]

# Landis & Koch (1977) interpretation bands for Cohen's kappa.
_BANDS = [
    (0.0, "poor (worse than chance)"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
]
SHIP_BAR = 0.60
DEFAULT_BOOT = 2000
MIN_ROWS = 30
MIN_PER_CLASS = 10
SKEW_LIMIT = 0.85
WIDE_CI = 0.30


class KappaError(Exception):
    """Bad input or unusable data. main() turns this into a clean exit 2."""


# ---------------------------------------------------------------- output plumbing

# Resolved in main(); ASCII fallbacks keep lint usable when stdout is a cp1252
# pipe (a redirect or CI runner on Windows), where the glyphs cannot encode.
MARKS = {"ok": "OK ", "warn": "!  ", "info": "-  ", "tip": "-  "}
_UNICODE_MARKS = {"ok": "✓ ", "warn": "⚠ ", "info": "· ", "tip": "· "}


def _init_stream(stream=None):
    """Prefer UTF-8 output; fall back to ASCII markers when it cannot encode."""
    stream = stream if stream is not None else sys.stdout
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        "✓⚠·".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return dict(MARKS)
    return dict(_UNICODE_MARKS)


# ---------------------------------------------------------------- statistics


def band(k):
    """Landis & Koch band name for a kappa value."""
    if k is None:
        return "undefined"
    for hi, name in _BANDS:
        if k < hi:
            return name
    return "almost perfect"


def _numeric_order(labels):
    """Labels sorted by numeric value, or None if they are not all numeric."""
    try:
        return sorted(labels, key=float)
    except (TypeError, ValueError):
        return None


def cohen_kappa(pairs, weights=None):
    """Cohen's kappa for (judge, human) label pairs.

    weights=None/"none" gives the unweighted (nominal) coefficient; "linear" or
    "quadratic" give the ordinal variants, which only apply to numeric labels.
    Returns None when kappa is undefined: no rows, or no label variation at all
    (both raters used a single identical label, where kappa is 0/0).
    """
    n = len(pairs)
    if n == 0:
        return None
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)

    if weights in (None, "none"):
        po = sum(1 for a, b in pairs if a == b) / n
        pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
        if pe >= 1.0:
            return None
        return (po - pe) / (1.0 - pe)

    if weights not in ("linear", "quadratic"):
        raise ValueError("weights must be one of: none, linear, quadratic")
    order = _numeric_order(set(ca) | set(cb))
    if order is None or len(order) < 2:
        return None
    idx = {lab: i for i, lab in enumerate(order)}
    span = len(order) - 1

    def w(i, j):
        d = abs(i - j) / span
        return d if weights == "linear" else d * d

    obs = sum(w(idx[a], idx[b]) for a, b in pairs) / n
    exp = sum(
        w(idx[x], idx[y]) * (ca[x] / n) * (cb[y] / n) for x in order for y in order
    )
    if exp <= 0:
        return None
    return 1.0 - obs / exp


def pabak(pairs):
    """Prevalence-adjusted bias-adjusted kappa: 2*agreement - 1."""
    if not pairs:
        return None
    po = sum(1 for a, b in pairs if a == b) / len(pairs)
    return 2 * po - 1


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(pairs):
    """Spearman's rho for numeric labels; None if labels are not numeric."""
    if len(pairs) < 3:
        return None
    try:
        xs = [float(a) for a, _ in pairs]
        ys = [float(b) for _, b in pairs]
    except (TypeError, ValueError):
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def bootstrap_ci(pairs, stat_fn, n_boot=DEFAULT_BOOT, seed=0, conf=0.95):
    """Percentile bootstrap interval for any statistic over label pairs.

    Returns (lo, hi), or None when the statistic is undefined on too many
    resamples to be meaningful. Seeded, so CI output is reproducible in CI.
    """
    n = len(pairs)
    if n < 2 or n_boot <= 0:
        return None
    rnd = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [pairs[rnd.randrange(n)] for _ in range(n)]
        v = stat_fn(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < max(20, n_boot // 10):
        return None
    vals.sort()
    lo_i = int((1.0 - conf) / 2.0 * len(vals))
    hi_i = min(len(vals) - 1, int((1.0 + conf) / 2.0 * len(vals)))
    return (vals[lo_i], vals[hi_i])


def confusion(pairs):
    """Return (labels, matrix) where matrix[h][j] counts human=h judged as j."""
    labels = sorted(set(x for p in pairs for x in p))
    order = _numeric_order(labels)
    if order is not None:
        labels = order
    m = {h: {j: 0 for j in labels} for h in labels}
    for j, h in pairs:
        m[h][j] += 1
    return labels, m


def per_label_stats(pairs):
    """Per-label counts plus the judge's recall and precision for that label."""
    labels, _ = confusion(pairs)
    out = []
    for lab in labels:
        human_n = sum(1 for _, h in pairs if h == lab)
        judge_n = sum(1 for j, _ in pairs if j == lab)
        tp = sum(1 for j, h in pairs if j == lab and h == lab)
        out.append(
            {
                "label": lab,
                "human": human_n,
                "judge": judge_n,
                "recall": (tp / human_n) if human_n else None,
                "precision": (tp / judge_n) if judge_n else None,
            }
        )
    return out


def _mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    if n > 1000:  # exact term count explodes; normal approximation is fine here
        z = abs(b - c) / math.sqrt(n)
        return math.erfc(z / math.sqrt(2))
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


# ---------------------------------------------------------------- input


def _norm_label(value, where, key):
    """Coerce a JSON/CSV scalar to a comparable label string.

    JSON pipelines emit booleans and numbers where humans type words, so `true`
    and "True" must land on the same label, as must 1, 1.0 and "1". Without this
    a judge that agrees perfectly can report 0% agreement.
    """
    if value is None:
        raise KappaError(
            "%s: '%s' is null - drop the row, or map unanswered rows to a label" % (where, key)
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise KappaError("%s: '%s' is not a finite number" % (where, key))
        if value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, (dict, list)):
        raise KappaError(
            "%s: '%s' must be a scalar, got %s" % (where, key, type(value).__name__)
        )
    s = str(value).strip()
    if not s:
        raise KappaError("%s: '%s' is empty" % (where, key))
    low = s.lower()
    if low in ("true", "false"):
        return low
    return s


def _lookup(obj, key, where):
    """Fetch a key, supporting dotted paths. A literal dotted key wins."""
    if key in obj:
        return obj[key]
    if "." in key:
        cur = obj
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KappaError("%s: missing '%s' (no segment '%s')" % (where, key, part))
            cur = cur[part]
        return cur
    raise KappaError("%s: missing '%s'" % (where, key))


def _open(path):
    if path == "-":
        return sys.stdin
    try:
        return open(path, encoding="utf-8", newline="")
    except OSError as e:
        raise KappaError("cannot read %s (%s)" % (path, e.strerror or e)) from e


def _reader(path, fmt=None):
    """Yield (line_number, record) for a JSONL or CSV source."""
    fmt = fmt or ("csv" if str(path).lower().endswith(".csv") else "jsonl")
    handle = _open(path)
    # Never close the real stdin: the process may need it again (kappa drift - -).
    ctx = nullcontext(handle) if handle is sys.stdin else handle
    with ctx as f:
        if fmt == "csv":
            rd = csv.DictReader(f)
            for row in rd:
                yield rd.line_num, {k: v for k, v in row.items() if k is not None}
        else:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise KappaError("%s:%d: invalid JSON (%s)" % (path, ln, e)) from e
                if not isinstance(obj, dict):
                    raise KappaError("%s:%d: expected a JSON object" % (path, ln))
                yield ln, obj


def load_rows(path, fmt=None):
    """Read a JSONL/CSV file into a list of records, unlabeled."""
    return [obj for _, obj in _reader(path, fmt)]


def load_pairs(path, judge_key="judge", human_key="human", fmt=None):
    """Read a JSONL/CSV file into (pairs, rows). Raises KappaError on bad input."""
    pairs, rows = [], []
    for ln, obj in _reader(path, fmt):
        where = "%s:%d" % (path, ln)
        j = _norm_label(_lookup(obj, judge_key, where), where, judge_key)
        h = _norm_label(_lookup(obj, human_key, where), where, human_key)
        pairs.append((j, h))
        rows.append(obj)
    return pairs, rows


def expand_paths(patterns):
    """Expand globs ourselves - cmd.exe and PowerShell do not do it for us."""
    out = []
    for pat in patterns:
        if pat == "-":
            out.append(pat)
            continue
        hits = sorted(globmod.glob(pat))
        out.extend(hits or [pat])
    return out


# ---------------------------------------------------------------- score


def _resolve_weights(weights, labels):
    if weights and weights != "auto":
        return weights
    order = _numeric_order(labels)
    # With two levels linear weighting is identical to unweighted kappa, so it
    # only earns its keep on genuinely ordinal scales.
    if order is not None and len(order) > 2:
        return "linear"
    return "none"


def score(
    pairs,
    rows=None,
    ship_bar=SHIP_BAR,
    weights="auto",
    boot=DEFAULT_BOOT,
    seed=0,
    conf=0.95,
    max_show=20,
):
    """Measure a judge against human labels. Pure: returns data, never prints."""
    n = len(pairs)
    if n == 0:
        raise KappaError("no labeled rows found")
    rows = rows if rows is not None else [{} for _ in pairs]

    agree = sum(1 for a, b in pairs if a == b)
    labels, matrix = confusion(pairs)
    used = _resolve_weights(weights, labels)

    k = cohen_kappa(pairs)
    kw = cohen_kappa(pairs, used) if used != "none" else None
    headline = kw if kw is not None else k

    ci = bootstrap_ci(
        pairs, (lambda p: cohen_kappa(p, used)) if used != "none" else cohen_kappa,
        n_boot=boot, seed=seed, conf=conf,
    )

    per_label = per_label_stats(pairs)
    warnings = []
    if n < MIN_ROWS:
        warnings.append(
            "n=%d is below %d; treat as directional only" % (n, MIN_ROWS)
        )
    for row in per_label:
        if row["human"] < MIN_PER_CLASS:
            warnings.append(
                "'%s' has %d human examples; label at least %d per class"
                % (row["label"], row["human"], MIN_PER_CLASS)
            )
    human_counts = Counter(h for _, h in pairs)
    top_label, top_n = human_counts.most_common(1)[0]
    skew = top_n / n
    if skew > SKEW_LIMIT:
        warnings.append(
            "labels are %.0f%% '%s'; kappa reads low under skew (prevalence paradox) "
            "- compare against PABAK and the raw agreement" % (100 * skew, top_label)
        )
    if ci is not None and (ci[1] - ci[0]) > WIDE_CI:
        warnings.append(
            "confidence interval is %.2f wide; collect more labels before trusting the bar"
            % (ci[1] - ci[0])
        )
    if k is None:
        warnings.append(
            "kappa is undefined: both raters used the single label '%s', so there is "
            "no signal to measure" % pairs[0][1]
        )

    if headline is None:
        verdict = "undefined"
    elif ci is not None and ci[0] < ship_bar <= ci[1]:
        verdict = "inconclusive"
    elif headline >= ship_bar:
        verdict = "ship-worthy"
    else:
        verdict = "below-bar"

    disagreements = [
        {
            "judge": j,
            "human": h,
            "id": r.get("id"),
            "text": r.get("text"),
        }
        for r, (j, h) in zip(rows, pairs)
        if j != h
    ]

    worst = None
    scored = [r for r in per_label if r["recall"] is not None and r["human"] >= 5]
    if scored:
        cand = min(scored, key=lambda r: r["recall"])
        if cand["recall"] < 1.0:
            worst = cand

    return {
        "schema_version": SCHEMA_VERSION,
        "kappa_version": __version__,
        "n": n,
        "agreement": agree,
        "agreement_rate": agree / n,
        "kappa": k,
        "kappa_band": band(k),
        "kappa_ci": list(ci) if ci else None,
        "weights": used,
        "weighted_kappa": kw,
        "headline_kappa": headline,
        "pabak": pabak(pairs),
        "spearman": spearman(pairs),
        "labels": labels,
        "confusion": matrix,
        "per_label": per_label,
        "worst_label": worst,
        "disagreements": disagreements[:max_show],
        "disagreement_count": len(disagreements),
        "ship_bar": ship_bar,
        "ship_bar_met": (headline >= ship_bar) if headline is not None else None,
        "verdict": verdict,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- drift


def drift(
    base_pairs,
    cur_pairs,
    threshold=0.10,
    ship_bar=SHIP_BAR,
    weights="auto",
    boot=DEFAULT_BOOT,
    seed=0,
    conf=0.95,
    base_rows=None,
    cur_rows=None,
    id_key="id",
    strict=False,
):
    """Compare judge-human agreement between two batches. Pure: returns data."""
    if not base_pairs or not cur_pairs:
        raise KappaError("a file had no labeled rows")

    labels = set(x for p in base_pairs + cur_pairs for x in p)
    used = _resolve_weights(weights, labels)
    stat = (lambda p: cohen_kappa(p, used)) if used != "none" else cohen_kappa

    kb, kc = stat(base_pairs), stat(cur_pairs)
    alerts, notes = [], []

    if kb is None or kc is None:
        which = "baseline" if kb is None else "current"
        alerts.append(
            "%s kappa is undefined (no label variation) - cannot compare" % which
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "kappa_version": __version__,
            "weights": used,
            "baseline": {"kappa": kb, "band": band(kb), "n": len(base_pairs)},
            "current": {"kappa": kc, "band": band(kc), "n": len(cur_pairs)},
            "delta": None,
            "delta_ci": None,
            "significant": None,
            "mcnemar": None,
            "threshold": threshold,
            "ship_bar": ship_bar,
            "alerts": alerts,
            "notes": notes,
            "verdict": "undefined",
        }

    delta = kc - kb

    delta_ci = None
    if boot > 0 and len(base_pairs) >= 2 and len(cur_pairs) >= 2:
        rnd = random.Random(seed)
        nb, nc = len(base_pairs), len(cur_pairs)
        vals = []
        for _ in range(boot):
            b = stat([base_pairs[rnd.randrange(nb)] for _ in range(nb)])
            c = stat([cur_pairs[rnd.randrange(nc)] for _ in range(nc)])
            if b is not None and c is not None:
                vals.append(c - b)
        if len(vals) >= max(20, boot // 10):
            vals.sort()
            delta_ci = [
                vals[int((1.0 - conf) / 2.0 * len(vals))],
                vals[min(len(vals) - 1, int((1.0 + conf) / 2.0 * len(vals)))],
            ]

    significant = None
    if delta_ci is not None:
        significant = not (delta_ci[0] <= 0.0 <= delta_ci[1])

    mcnemar = None
    if base_rows and cur_rows:
        base_flag, cur_flag = {}, {}
        for r, (j, h) in zip(base_rows, base_pairs):
            rid = r.get(id_key)
            if rid is not None:
                base_flag[rid] = j == h
        for r, (j, h) in zip(cur_rows, cur_pairs):
            rid = r.get(id_key)
            if rid is not None:
                cur_flag[rid] = j == h
        shared = set(base_flag) & set(cur_flag)
        if shared:
            b = sum(1 for i in shared if base_flag[i] and not cur_flag[i])
            c = sum(1 for i in shared if not base_flag[i] and cur_flag[i])
            mcnemar = {
                "paired": len(shared),
                "lost": b,
                "gained": c,
                "p": _mcnemar_exact(b, c),
            }

    if delta <= -threshold:
        alerts.append(
            "kappa dropped by %.3f (>= %.2f threshold)" % (-delta, threshold)
        )
    if kc < ship_bar <= kb:
        alerts.append("current fell below the %.2f ship bar" % ship_bar)

    if alerts and significant is False:
        notes.append(
            "the drop is not distinguishable from sampling noise at these sample "
            "sizes; collect more labels before acting on it"
        )
        if strict:
            alerts = []
            notes.append("--strict: alert withheld (statistically inconclusive)")
    if not alerts and significant and delta < 0:
        notes.append(
            "drop is below the alert threshold but statistically distinguishable from zero"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kappa_version": __version__,
        "weights": used,
        "baseline": {"kappa": kb, "band": band(kb), "n": len(base_pairs)},
        "current": {"kappa": kc, "band": band(kc), "n": len(cur_pairs)},
        "delta": delta,
        "delta_ci": delta_ci,
        "significant": significant,
        "mcnemar": mcnemar,
        "threshold": threshold,
        "ship_bar": ship_bar,
        "alerts": alerts,
        "notes": notes,
        "verdict": "alert" if alerts else "ok",
    }


# ---------------------------------------------------------------- lint

# Each check encodes a documented LLM-as-a-Judge failure mode. mode="require"
# means the pattern SHOULD appear; mode="forbid" means its presence is the
# problem. gate names a precondition - the check is skipped when it does not hold.
_CHECKS = [
    {
        "id": "rubric",
        "label": "explicit rubric / scoring scale",
        "severity": "warn",
        "mode": "require",
        "pattern": r"\brubric\b|\bcriteri(a|on)\b|\bscale\b|\bscor\w*\b|\brat(e|ing)\b"
                   r"|\bpoints?\b|\b\d{1,2}\s*[-–]\s*\d{1,3}\b|\bout of \d",
        "fix": "Define explicit criteria and a scale - don't ask 'is this good?'. Vague judges are noisy.",
    },
    {
        "id": "evidence-order",
        "label": "evidence-before-verdict",
        "severity": "warn",
        "mode": "require",
        "pattern": r"\b(step[- ]by[- ]step|explain\w*|reason\w*|justif\w*|cite|citing|quot\w*"
                   r"|evidence|because"
                   r"|before (you )?(decid\w*|scor\w*|answer\w*|rat\w*|giv\w*|stat\w*|writ\w*))\b",
        "fix": "Make the judge cite specifics BEFORE the verdict - it kills rubber-stamping.",
    },
    {
        "id": "decomposed",
        "label": "decomposed judging",
        "severity": "warn",
        "mode": "require",
        "pattern": r"\b(for each|per (dimension|criterion|aspect)|dimensions?|separately|one at a time)\b",
        "fix": "Judge one dimension at a time (correctness / safety / completeness), not one global score.",
    },
    {
        "id": "verbosity",
        "label": "verbosity-bias control",
        "severity": "warn",
        "mode": "require",
        "pattern": r"\b(length|verbos\w*|concise\w*|word count|token count"
                   r"|do not (favor|favour|prefer|reward) (longer|long))\b",
        "fix": "Say longer != better, or normalize for length - judges over-reward verbose answers.",
    },
    {
        "id": "output-format",
        "label": "output format specified",
        "severity": "warn",
        "mode": "require",
        "pattern": r"\b(json|yaml|xml|output format|respond (only|exactly) with"
                   r"|answer (only )?with|reply (only )?with|verdict:|rating:|score:)"
                   r"|\[\[|<verdict>|final answer:",
        "fix": "Specify an exact output format (a one-line JSON object, or a [[A]]/[[B]] marker) "
               "- unparseable verdicts silently corrupt an eval run.",
    },
    {
        "id": "position-bias",
        "label": "position-bias control (pairwise)",
        "severity": "warn",
        "mode": "require",
        "gate": "pairwise",
        "pattern": r"\b(randomiz\w*|randomis\w*|both orders?|swap\w*|order[- ]independent"
                   r"|either order|positional bias|position bias)\b",
        "fix": "Pairwise prompt detected - randomize A/B order or score each side independently.",
    },
    {
        "id": "tie-option",
        "label": "tie / abstain allowed (pairwise)",
        "severity": "warn",
        "mode": "require",
        "gate": "pairwise",
        "pattern": r"\b(tie|both (are )?equal(ly)?|equally good|no (clear )?winner"
                   r"|too close to call|cannot (decide|determine)|abstain"
                   r"|insufficient (information|evidence|context))\b",
        "fix": "Forced choice on near-equal responses amplifies position bias - allow a tie or abstain verdict.",
    },
    {
        "id": "scale-anchors",
        "label": "per-level scale anchors",
        "severity": "warn",
        "mode": "require",
        "gate": "numeric_scale",
        "pattern": r"(^|\n)\s*\d{1,2}\s*[=:).-]\s+\S|\b\d (means|indicates)\b|\b\d\s*=\s*\w",
        "fix": "Numeric scale with no per-level definitions - anchor each point (1 = ..., 5 = ...) "
               "or judges drift lenient and compress the scale.",
    },
    {
        "id": "model-identity",
        "label": "no model identities in the prompt",
        "severity": "warn",
        "mode": "forbid",
        "pattern": r"\b(gpt-?[3-5](\.\d)?|gpt-?4o|chatgpt|openai|claude|anthropic|gemini|bard"
                   r"|llama|mistral|qwen|deepseek|grok)\b",
        "fix": "The prompt names model identities - anonymize the responses. Judges shift verdicts "
               "when provenance is visible, and favor their own family.",
    },
    {
        "id": "score-first",
        "label": "no score-before-reasoning ordering",
        "severity": "warn",
        "mode": "forbid",
        "pattern": r"\b(first,? (give|state|output|provide|return) (the |your )?(score|rating|verdict|answer)"
                   r"|(score|verdict|rating) first,? then (explain|justify))\b",
        "fix": "Asking for the score first turns the explanation into post-hoc rationalization "
               "- put the analysis before the verdict.",
    },
    {
        "id": "reference",
        "label": "reference answer for correctness grading",
        "severity": "info",
        "mode": "require",
        "gate": "objective",
        "pattern": r"\b(reference (answer|solution|output)|gold (answer|label|standard)"
                   r"|expected (answer|output)|answer key|ground truth)\b",
        "fix": "Grading correctness with no reference answer - supply one. Reference-guided grading "
               "cut math-judging failures from ~70% to ~15% (Zheng et al., 2023).",
    },
    {
        "id": "few-shot",
        "label": "calibration examples",
        "severity": "info",
        "mode": "require",
        "pattern": r"(^|\n)\s*#*\s*examples?\b|<example|\bfor example\b"
                   r"|\be\.g\.,? (input|response)|\bsample (input|response|evaluation)\b",
        "fix": "No calibration examples found - 2-3 graded examples raise judge consistency. "
               "Your own judge/human disagreements are the best source.",
    },
    {
        "id": "abstain",
        "label": "abstain option on a binary verdict",
        "severity": "info",
        "mode": "require",
        "gate": "binary",
        "pattern": r"\b(unsure|uncertain|maybe|borderline|cannot (determine|tell)|insufficient"
                   r"|abstain|not enough (information|context)|n/?a)\b",
        "fix": "Binary verdict with no escape hatch - genuinely ambiguous items get forced into a "
               "side, which shows up later as judge-human disagreement you cannot fix.",
    },
    {
        "id": "sycophancy",
        "label": "neutral, non-leading phrasing",
        "severity": "info",
        "mode": "forbid",
        "strict_only": True,
        "pattern": r"\b(i|we) (think|believe|feel|expect)\b"
                   r"|\b(obviously|clearly) (correct|wrong|better|the best)\b"
                   r"|\bthis (excellent|great|impressive|poor|terrible|bad) (response|answer|output)\b",
        "fix": "Opinionated framing leaks into the verdict - judges are sycophantic to stated opinions.",
    },
    {
        "id": "authority",
        "label": "citation-bias guard",
        "severity": "info",
        "mode": "require",
        "strict_only": True,
        "gate": "citations",
        "pattern": r"\b(verify|do not (trust|assume)|regardless of (citations?|sources?)|fabricat\w*)\b",
        "fix": "The prompt weighs citations - say they may be fabricated and must not raise "
               "credibility unverified (authority bias).",
    },
]

_TIPS = [
    "pin the judge's temperature (or sample 3x and take the majority) - verdict flicker inflates drift alarms.",
    "a panel of small cross-family judges beats one big same-family judge.",
]

_PAIRWISE_PHRASE = re.compile(
    r"\bwhich (one |response |answer )?is better\b|\bcompare the two\b"
    r"|\bpick the better\b|\bprefer (response|answer|option|candidate)\b"
    r"|\bthe two (responses|answers|outputs)\b"
)
_SIDE_LETTER = r"\b(response|answer|option|candidate)\s*%s\b"
# A digit only signals a side when its twin is present too, so "the answer 1-5"
# (a scale) is not mistaken for "answer 1 vs answer 2" (a comparison).
_SIDE_DIGIT = r"\b(response|answer|option|candidate)\s*%d\b(?!\s*[-–]\s*\d)"
_NUMERIC_SCALE = re.compile(r"\b\d{1,2}\s*[-–]\s*\d{1,3}\b|\bout of \d|\bscale of \d")
_OBJECTIVE = re.compile(
    r"\b(correct(ness)?|accura(te|cy)|factual|math|arithmetic|solution|right answer)\b"
)
_BINARY = re.compile(
    r"\b(pass or fail|fail or pass|yes or no|true or false|good or bad"
    r"|pass/fail|yes/no|accept or reject)\b"
)
_CITATIONS = re.compile(r"\b(citations?|cited|references?|sources?)\b")


def _is_pairwise(low):
    if _PAIRWISE_PHRASE.search(low):
        return True
    if re.search(_SIDE_LETTER % "a", low) and re.search(_SIDE_LETTER % "b", low):
        return True
    if re.search(_SIDE_DIGIT % 1, low) and re.search(_SIDE_DIGIT % 2, low):
        return True
    return False


def lint(text, strict=False, ignore=()):
    """Heuristically check a judge prompt for known bias pitfalls.

    Pure: returns data. Findings carry stable ids so callers can ignore rules.
    """
    low = text.lower()
    ctx = {
        "pairwise": _is_pairwise(low),
        "numeric_scale": bool(_NUMERIC_SCALE.search(low)),
        "objective": bool(_OBJECTIVE.search(low)),
        "binary": bool(_BINARY.search(low)),
        "citations": bool(_CITATIONS.search(low)),
    }

    findings = []
    for chk in _CHECKS:
        if chk["id"] in ignore:
            continue
        if chk.get("strict_only") and not strict:
            continue
        gate = chk.get("gate")
        if gate and not ctx[gate]:
            continue
        hit = bool(re.search(chk["pattern"], low))
        passed = hit if chk["mode"] == "require" else not hit
        severity = chk["severity"]
        if strict and severity == "info":
            severity = "warn"
        findings.append(
            {
                "id": chk["id"],
                "label": chk["label"],
                "severity": severity,
                "status": "pass" if passed else "fail",
                "fix": "" if passed else chk["fix"],
            }
        )

    counted = [f for f in findings if f["severity"] == "warn"]
    passed_n = sum(1 for f in counted if f["status"] == "pass")
    failed = [f for f in findings if f["status"] == "fail" and f["severity"] == "warn"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kappa_version": __version__,
        "context": ctx,
        "findings": findings,
        "passed": passed_n,
        "total": len(counted),
        "failed": len(failed),
        "tips": _TIPS,
        "verdict": "fail" if failed else "pass",
    }


# ---------------------------------------------------------------- rendering


def _fmt_k(v):
    return "undefined" if v is None else "%+.3f" % v


def _fmt_ci(ci):
    return "" if not ci else "  95%% CI [%+.2f, %+.2f]" % (ci[0], ci[1])


_VERDICT_TEXT = {
    "ship-worthy": "SHIP-WORTHY",
    "below-bar": "BELOW the %.2f ship bar",
    "inconclusive": "INCONCLUSIVE: interval spans the %.2f ship bar",
    "undefined": "UNDEFINED: no label variation to measure",
}


def render_score(res, src, marks):
    out = ["kappa score - %s" % src]
    out.append(
        "  rows: %d    agreement: %d/%d = %.1f%%"
        % (res["n"], res["agreement"], res["n"], 100 * res["agreement_rate"])
    )
    label = "Cohen's kappa"
    if res["weights"] != "none":
        label = "Cohen's kappa (%s-weighted)" % res["weights"]
        out.append("  unweighted kappa: %s  [%s]" % (_fmt_k(res["kappa"]), res["kappa_band"]))
    out.append(
        "  %s: %s%s  [%s]"
        % (label, _fmt_k(res["headline_kappa"]), _fmt_ci(res["kappa_ci"]), band(res["headline_kappa"]))
    )
    vt = _VERDICT_TEXT[res["verdict"]]
    if "%.2f" in vt:
        vt = vt % res["ship_bar"]
    out.append("    -> %s" % vt)
    if res["pabak"] is not None:
        out.append("  PABAK: %+.3f   (prevalence-adjusted companion)" % res["pabak"])
    if res["spearman"] is not None:
        out.append("  Spearman rho: %+.3f   (rank agreement on the ordinal scale)" % res["spearman"])

    labels = res["labels"]
    shown = [lab[:12] for lab in labels]
    w = max([len(s) for s in shown] + [5])
    out.append("  confusion (rows=human, cols=judge):")
    out.append("    " + " " * w + "  " + "  ".join(s.rjust(w) for s in shown))
    for h, hs in zip(labels, shown):
        cells = "  ".join(str(res["confusion"][h][j]).rjust(w) for j in labels)
        out.append("    " + hs.rjust(w) + "  " + cells)

    out.append("  per label:")
    head = "%s  %5s  %5s  %6s  %9s" % ("label".rjust(w), "human", "judge", "recall", "precision")
    out.append("    " + head)
    for row in res["per_label"]:
        rec = "-" if row["recall"] is None else "%.2f" % row["recall"]
        pre = "-" if row["precision"] is None else "%.2f" % row["precision"]
        out.append(
            "    %s  %5d  %5d  %6s  %9s"
            % (row["label"][:12].rjust(w), row["human"], row["judge"], rec, pre)
        )
    if res["worst_label"]:
        wl = res["worst_label"]
        out.append(
            "    -> judge misses %.0f%% of items a human marked '%s'"
            % (100 * (1 - wl["recall"]), wl["label"])
        )

    if res["disagreement_count"]:
        out.append(
            "  disagreements (%d) - gold for few-shot examples:" % res["disagreement_count"]
        )
        for d in res["disagreements"]:
            ctx = d.get("id") or d.get("text") or ""
            ctx = (" - " + str(ctx)[:60]) if ctx else ""
            out.append("    judge=%r human=%r%s" % (d["judge"], d["human"], ctx))
        if res["disagreement_count"] > len(res["disagreements"]):
            out.append(
                "    ... and %d more" % (res["disagreement_count"] - len(res["disagreements"]))
            )
    if res["warnings"]:
        out.append("  warnings:")
        for wmsg in res["warnings"]:
            out.append("    %s%s" % (marks["warn"], wmsg))
    return "\n".join(out)


def render_drift(res, marks):
    out = ["kappa drift"]
    if res["weights"] != "none":
        out.append("  weighting: %s (ordinal labels)" % res["weights"])
    for name in ("baseline", "current"):
        side = res[name]
        out.append(
            "  %-9s kappa %s  [%s]  (n=%d)"
            % (name + ":", _fmt_k(side["kappa"]), side["band"], side["n"])
        )
    if res["delta"] is not None:
        out.append("  delta:    %+.3f%s" % (res["delta"], _fmt_ci(res["delta_ci"])))
        if res["significant"] is not None:
            out.append(
                "            %s from zero at these sample sizes"
                % ("distinguishable" if res["significant"] else "NOT distinguishable")
            )
    if res["mcnemar"]:
        m = res["mcnemar"]
        out.append(
            "  paired:   %d shared ids; %d agreements lost, %d gained (McNemar p=%.3f)"
            % (m["paired"], m["lost"], m["gained"], m["p"])
        )
    for note in res["notes"]:
        out.append("  note: %s" % note)
    if res["alerts"]:
        out.append("  ALERT:")
        for a in res["alerts"]:
            out.append("    %s%s" % (marks["warn"], a))
    else:
        out.append("  %sok: no significant drift" % marks["ok"])
    return "\n".join(out)


def render_lint(res, src, marks):
    out = ["kappa lint - %s" % src]
    for f in res["findings"]:
        if f["status"] == "pass":
            out.append("  %s%s" % (marks["ok"], f["label"]))
        else:
            mark = marks["warn"] if f["severity"] == "warn" else marks["info"]
            out.append("  %s%s  [%s]" % (mark, f["label"], f["id"]))
            out.append("      fix: %s" % f["fix"])
    out.append("  ---")
    out.append("  %d/%d checks passed" % (res["passed"], res["total"]))
    for tip in res["tips"]:
        out.append("  %stip: %s" % (marks["tip"], tip))
    return "\n".join(out)


def _write_html(path, src, res):
    """Self-contained HTML report. Every interpolated value is escaped: labels
    and text fields are model output, and this report gets shared."""
    e = html.escape
    labels = res["labels"]
    head = "".join("<th>%s</th>" % e(j) for j in labels)
    rows_html = ""
    for h in labels:
        cells = "".join("<td>%d</td>" % res["confusion"][h][j] for j in labels)
        rows_html += "<tr><th>%s</th>%s</tr>" % (e(h), cells)
    per = "".join(
        "<tr><th>%s</th><td>%d</td><td>%d</td><td>%s</td><td>%s</td></tr>"
        % (
            e(r["label"]),
            r["human"],
            r["judge"],
            "-" if r["recall"] is None else "%.2f" % r["recall"],
            "-" if r["precision"] is None else "%.2f" % r["precision"],
        )
        for r in res["per_label"]
    )
    dis = "".join(
        "<li><code>judge=%s</code> vs <code>human=%s</code>%s</li>"
        % (e(d["judge"]), e(d["human"]), e((" - " + str(d.get("id") or d.get("text") or ""))[:80]))
        for d in res["disagreements"]
    )
    warn = "".join("<li>%s</li>" % e(w) for w in res["warnings"])
    k = res["headline_kappa"]
    color = "#1a7f37" if res["verdict"] == "ship-worthy" else "#b35900"
    ci = ""
    if res["kappa_ci"]:
        ci = " <span class=muted>95%% CI [%+.2f, %+.2f]</span>" % tuple(res["kappa_ci"])
    html_doc = """<!doctype html><meta charset=utf-8><title>kappa report</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#111}}
.k{{font-size:2.4rem;font-weight:700;color:{color}}} table{{border-collapse:collapse;margin:1rem 0}}
td,th{{border:1px solid #ddd;padding:.3rem .6rem;text-align:center}} code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}}
.muted{{color:#666;font-size:.85em}}</style>
<h1>kappa report</h1><p class=muted>source: {src} &middot; {n} rows &middot; {agree}/{n} agree ({rate:.1f}%)</p>
<p>Cohen's &kappa; <span class=k>{k}</span>{ci} <span class=muted>[{bandname}] &middot; ship bar {bar}</span></p>
<p><strong>{verdict}</strong></p>
<h2>Confusion (rows = human, cols = judge)</h2><table><tr><th></th>{head}</tr>{rows}</table>
<h2>Per label</h2><table><tr><th>label</th><th>human</th><th>judge</th><th>recall</th><th>precision</th></tr>{per}</table>
<h2>Disagreements ({dn})</h2><ul>{dis}</ul>
{warnblock}
<p class=muted>Generated by kappa {ver} - zero-dependency LLM-as-a-Judge eval kit.</p>""".format(
        color=color,
        src=e(str(src)),
        n=res["n"],
        agree=res["agreement"],
        rate=100 * res["agreement_rate"],
        k=e(_fmt_k(k)),
        ci=ci,
        bandname=e(band(k)),
        bar=res["ship_bar"],
        verdict=e(res["verdict"]),
        head=head,
        rows=rows_html,
        per=per,
        dn=res["disagreement_count"],
        dis=dis or "<li>none</li>",
        warnblock=("<h2>Warnings</h2><ul>%s</ul>" % warn) if warn else "",
        ver=e(__version__),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def _step_summary(markdown):
    """Append a markdown block to the GitHub Actions run summary, if we're in one."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
    except OSError:
        pass


def _score_markdown(res, src):
    lines = [
        "### kappa score - `%s`" % src,
        "",
        "| metric | value |",
        "| --- | --- |",
        "| rows | %d |" % res["n"],
        "| agreement | %d/%d (%.1f%%) |" % (res["agreement"], res["n"], 100 * res["agreement_rate"]),
        "| kappa | %s %s |" % (_fmt_k(res["headline_kappa"]), _fmt_ci(res["kappa_ci"]).strip()),
        "| band | %s |" % band(res["headline_kappa"]),
        "| verdict | **%s** |" % res["verdict"],
        "",
    ]
    if res["warnings"]:
        lines.append("Warnings: " + "; ".join(res["warnings"]))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- commands


def _emit(args, res, text):
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2, sort_keys=True))
    elif not getattr(args, "quiet", False):
        print(text)


def cmd_score(args):
    pairs, rows = [], []
    for path in expand_paths(args.file):
        p, r = load_pairs(path, args.judge_key, args.human_key, args.format)
        pairs.extend(p)
        rows.extend(r)
    src = ", ".join(args.file)
    res = score(
        pairs,
        rows,
        ship_bar=args.ship_bar,
        weights=args.weights,
        boot=args.boot,
        seed=args.seed,
        max_show=args.max_show,
    )
    _emit(args, res, render_score(res, src, args._marks))
    if args.html:
        _write_html(args.html, src, res)
        if not args.json and not args.quiet:
            print("  html report -> %s" % args.html)
    _step_summary(_score_markdown(res, src))

    if args.fail_under is not None:
        k = res["headline_kappa"]
        if k is None:
            return 1
        if args.strict and res["kappa_ci"]:
            return 1 if res["kappa_ci"][0] < args.fail_under else 0
        return 1 if k < args.fail_under else 0
    return 0


def cmd_drift(args):
    bp, br = load_pairs(args.baseline, args.judge_key, args.human_key, args.format)
    cp, cr = load_pairs(args.current, args.judge_key, args.human_key, args.format)
    res = drift(
        bp,
        cp,
        threshold=args.threshold,
        ship_bar=args.ship_bar,
        weights=args.weights,
        boot=args.boot,
        seed=args.seed,
        base_rows=br,
        cur_rows=cr,
        id_key=args.id_key,
        strict=args.strict,
    )
    _emit(args, res, render_drift(res, args._marks))
    _step_summary(
        "### kappa drift\n\nbaseline %s -> current %s (delta %s)\n\n%s\n"
        % (
            _fmt_k(res["baseline"]["kappa"]),
            _fmt_k(res["current"]["kappa"]),
            _fmt_k(res["delta"]),
            ("**ALERT** " + "; ".join(res["alerts"])) if res["alerts"] else "ok",
        )
    )
    return 1 if res["alerts"] else 0


def cmd_lint(args):
    if args.file == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            raise KappaError("cannot read %s (%s)" % (args.file, e.strerror or e)) from e
    ignore = set(i.strip() for i in (args.ignore or "").split(",") if i.strip())
    unknown = ignore - set(c["id"] for c in _CHECKS)
    if unknown:
        raise KappaError("unknown check id(s): %s" % ", ".join(sorted(unknown)))
    res = lint(text, strict=args.strict, ignore=ignore)
    _emit(args, res, render_lint(res, args.file, args._marks))
    return 1 if res["failed"] else 0


def cmd_sample(args):
    rows = load_rows(args.file, args.format)
    if not rows:
        raise KappaError("no rows found in %s" % args.file)
    key = args.by or args.judge_key
    strata = {}
    skipped = 0
    for i, r in enumerate(rows):
        try:
            lab = _norm_label(_lookup(r, key, "%s:%d" % (args.file, i + 1)), "row", key)
        except KappaError:
            skipped += 1
            continue
        strata.setdefault(lab, []).append(r)
    if not strata:
        raise KappaError("no rows carried a '%s' value to stratify on" % key)

    rnd = random.Random(args.seed)
    names = sorted(strata)
    picked = []
    if args.proportional:
        total = sum(len(v) for v in strata.values())
        for name in names:
            take = int(round(args.n * len(strata[name]) / total))
            picked.append((name, min(take, len(strata[name]))))
    else:
        base, extra = divmod(args.n, len(names))
        # Give the remainder to the smallest strata: minority labels carry the
        # information, and a 95%-pass set wastes an afternoon if sampled by share.
        by_size = sorted(names, key=lambda s: len(strata[s]))
        quota = {name: base for name in names}
        for name in by_size[:extra]:
            quota[name] += 1
        picked = [(name, min(quota[name], len(strata[name]))) for name in names]

    out_rows = []
    for name, take in picked:
        pool = list(strata[name])
        rnd.shuffle(pool)
        out_rows.extend(pool[:take])
    rnd.shuffle(out_rows)

    payload = "\n".join(json.dumps(r, sort_keys=True) for r in out_rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + ("\n" if payload else ""))
        if not args.quiet:
            summary = ", ".join("%s=%d" % (n, t) for n, t in picked)
            print("kappa sample -> %s   %d rows (%s)" % (args.out, len(out_rows), summary))
            if skipped:
                print("  skipped %d rows with no usable '%s'" % (skipped, key))
    else:
        print(payload)
    return 0


def cmd_gold(args):
    rows = load_rows(args.file, args.format)
    done = set()
    if os.path.exists(args.out):
        for _, obj in _reader(args.out):
            rid = obj.get(args.id_key)
            if rid is not None:
                done.add(rid)
    todo = [r for r in rows if r.get(args.id_key) not in done or r.get(args.id_key) is None]
    if not todo:
        print("kappa gold - nothing left to label (%d already in %s)" % (len(done), args.out))
        return 0

    print("kappa gold - %d rows to label, %d already done" % (len(todo), len(done)))
    print("  enter = agree with the judge, a label = disagree, s = skip, q = save and quit\n")
    written = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for r in todo:
            where = "row"
            try:
                jv = _norm_label(_lookup(r, args.judge_key, where), where, args.judge_key)
            except KappaError as e:
                print("  skipping: %s" % e)
                continue
            rid = r.get(args.id_key)
            text = str(r.get("text") or "")[:300]
            print("  id=%s" % (rid if rid is not None else "-"))
            if text:
                print("  %s" % text)
            print("  judge says: %s" % jv)
            try:
                reply = input("  human label [%s]: " % jv).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  stopping.")
                break
            if reply.lower() == "q":
                break
            if reply.lower() == "s":
                print()
                continue
            human = jv if not reply else reply
            rec = {args.judge_key: jv, args.human_key: human}
            if rid is not None:
                rec[args.id_key] = rid
            if r.get("text"):
                rec["text"] = r["text"]
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            out.flush()
            written += 1
            print()
    print("kappa gold -> wrote %d labeled rows to %s" % (written, args.out))
    return 0


# ---------------------------------------------------------------- cli


def build_parser():
    p = argparse.ArgumentParser(
        prog="kappa",
        description="LLM-as-a-Judge evaluation kit (zero-dependency).",
        epilog="exit codes: 0 clean, 1 gate failed, 2 usage or input error",
    )
    p.add_argument("--version", action="version", version="kappa %s" % __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--judge-key", default="judge", help="key holding the judge label")
        sp.add_argument("--human-key", default="human", help="key holding the human label")
        sp.add_argument("--format", choices=["jsonl", "csv"], help="input format (default: by extension)")
        sp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
        sp.add_argument("--quiet", action="store_true", help="suppress the report; rely on the exit code")

    def add_stats(sp):
        sp.add_argument("--ship-bar", type=float, default=SHIP_BAR, help="agreement bar (default 0.60)")
        sp.add_argument(
            "--weights", choices=["auto", "none", "linear", "quadratic"], default="auto",
            help="ordinal weighting for numeric labels (default: auto)",
        )
        sp.add_argument("--boot", type=int, default=DEFAULT_BOOT, help="bootstrap resamples (0 disables)")
        sp.add_argument("--seed", type=int, default=0, help="bootstrap seed, for reproducible intervals")

    # Note: percent signs in help strings must be doubled - argparse %%-formats
    # them when the parent parser renders its subcommand list.
    s = sub.add_parser("score", help="agreement %%, Cohen's kappa, confusion, per-label recall")
    s.add_argument("file", nargs="+", help="JSONL/CSV of {judge, human, [id|text]} (- for stdin)")
    add_common(s)
    add_stats(s)
    s.add_argument("--max-show", type=int, default=20, help="max disagreements to list")
    s.add_argument("--html", metavar="OUT", help="also write a self-contained HTML report")
    s.add_argument("--fail-under", type=float, metavar="K", help="exit 1 when kappa is below K")
    s.add_argument("--strict", action="store_true", help="with --fail-under, test the CI lower bound")
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("drift", help="compare agreement between two batches")
    d.add_argument("baseline")
    d.add_argument("current")
    add_common(d)
    add_stats(d)
    d.add_argument("--threshold", type=float, default=0.10, help="kappa-drop alert threshold (default 0.10)")
    d.add_argument("--id-key", default="id", help="key used to pair rows across files")
    d.add_argument("--strict", action="store_true", help="only alert when the drop clears sampling noise")
    d.set_defaults(func=cmd_drift)

    li = sub.add_parser("lint", help="heuristic linter for judge-prompt bias pitfalls")
    li.add_argument("file", help="judge prompt text file (- for stdin)")
    li.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    li.add_argument("--quiet", action="store_true", help="suppress the report; rely on the exit code")
    li.add_argument("--strict", action="store_true", help="promote suggestions to warnings, add weak heuristics")
    li.add_argument("--ignore", metavar="IDS", help="comma-separated check ids to skip")
    li.set_defaults(func=cmd_lint)

    sa = sub.add_parser("sample", help="stratified sample of rows to hand-label")
    sa.add_argument("file", help="JSONL/CSV of judged rows")
    sa.add_argument("-n", type=int, default=50, help="how many rows to draw (default 50)")
    sa.add_argument("--by", help="key to stratify on (default: the judge key)")
    sa.add_argument("--judge-key", default="judge", help="key holding the judge label")
    sa.add_argument("--format", choices=["jsonl", "csv"], help="input format (default: by extension)")
    sa.add_argument("--proportional", action="store_true", help="sample by class share instead of evenly")
    sa.add_argument("--seed", type=int, default=0, help="sampling seed")
    sa.add_argument("-o", "--out", help="write here instead of stdout")
    sa.add_argument("--quiet", action="store_true", help="suppress the summary line")
    sa.set_defaults(func=cmd_sample)

    g = sub.add_parser("gold", help="interactive labeling loop, resumable")
    g.add_argument("file", help="JSONL/CSV of judged rows")
    g.add_argument("-o", "--out", default="labels.jsonl", help="labels file to append to")
    g.add_argument("--judge-key", default="judge", help="key holding the judge label")
    g.add_argument("--human-key", default="human", help="key to write the human label to")
    g.add_argument("--id-key", default="id", help="key identifying a row (used to resume)")
    g.add_argument("--format", choices=["jsonl", "csv"], help="input format (default: by extension)")
    g.set_defaults(func=cmd_gold)
    return p


def main(argv=None):
    marks = _init_stream()
    args = build_parser().parse_args(argv)
    args._marks = marks
    try:
        return args.func(args)
    except KappaError as e:
        sys.stderr.write("error: %s\n" % e)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
