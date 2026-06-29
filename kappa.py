#!/usr/bin/env python3
"""kappa — a tiny, zero-dependency evaluation kit for LLM-as-a-Judge.

Three things every team running an LLM judge needs, and there's no clean little
tool for: measure whether the judge agrees with humans, watch that agreement for
drift, and catch the well-known bias pitfalls in the judge prompt itself.

Subcommands
-----------
  kappa score  <labels.jsonl>            agreement %, Cohen's kappa, confusion, disagreements
  kappa drift  <baseline.jsonl> <new.jsonl>   compare two batches; flag a kappa drop
  kappa lint   <judge-prompt.txt | ->    heuristic linter for judge-prompt bias pitfalls

Input format for score/drift: JSON Lines, one object per line, each with a judge
label and a human label (defaults: keys "judge" and "human"; override with flags).
An optional "id" or "text" field is shown next to disagreements for context.

Why kappa, not raw accuracy: Cohen's kappa discounts agreement that would happen
by chance, so it stays honest under class imbalance (the usual "90% of replies are
fine" case). A common ship bar is kappa >= 0.6 (substantial agreement).

Zero dependencies — Python 3.8+ standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

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


def band(k: float) -> str:
    for hi, name in _BANDS:
        if k < hi:
            return name
    return "almost perfect"


def cohen_kappa(pairs):
    """Cohen's kappa for a list of (rater_a, rater_b) categorical label pairs."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    labels = set(ca) | set(cb)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in labels)
    if pe >= 1.0:                       # only one label in play on a side
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def load_pairs(path, judge_key, human_key):
    """Read a JSONL file into (pairs, rows). Tolerates blank/comment lines."""
    pairs, rows = [], []
    with _open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"error: {path}:{ln}: invalid JSON ({e})")
            if judge_key not in obj or human_key not in obj:
                sys.exit(f"error: {path}:{ln}: missing '{judge_key}' or '{human_key}'")
            j, h = str(obj[judge_key]), str(obj[human_key])
            pairs.append((j, h))
            rows.append(obj)
    return pairs, rows


def _open(path):
    if path == "-":
        return sys.stdin
    return open(path, "r", encoding="utf-8")


def confusion(pairs):
    """Return (labels, matrix) where matrix[h][j] counts human=h judged as j."""
    labels = sorted(set(x for p in pairs for x in p))
    m = {h: {j: 0 for j in labels} for h in labels}
    for j, h in pairs:
        m[h][j] += 1
    return labels, m


# ---------------------------------------------------------------- score

def cmd_score(args):
    pairs, rows = load_pairs(args.file, args.judge_key, args.human_key)
    n = len(pairs)
    if n == 0:
        sys.exit("error: no labeled rows found")
    agree = sum(1 for a, b in pairs if a == b)
    k = cohen_kappa(pairs)
    labels, m = confusion(pairs)

    out = []
    out.append(f"kappa score — {args.file}")
    out.append(f"  rows: {n}    agreement: {agree}/{n} = {100*agree/n:.1f}%")
    if n < 10:
        out.append(f"  Cohen's kappa: {k:+.3f}  [{band(k)}]   (n<10 — treat as indicative)")
    else:
        verdict = "SHIP-WORTHY" if k >= SHIP_BAR else f"BELOW {SHIP_BAR:.2f} ship bar"
        out.append(f"  Cohen's kappa: {k:+.3f}  [{band(k)}]  -> {verdict}")
    # confusion matrix (human rows x judge cols)
    out.append("  confusion (rows=human, cols=judge):")
    w = max([len(s) for s in labels] + [5])
    out.append("    " + " " * w + "  " + "  ".join(s.rjust(w) for s in labels))
    for h in labels:
        cells = "  ".join(str(m[h][j]).rjust(w) for j in labels)
        out.append("    " + h.rjust(w) + "  " + cells)
    # disagreements
    disagreements = [(r, j, h) for r, (j, h) in zip(rows, pairs) if j != h]
    if disagreements:
        out.append(f"  disagreements ({len(disagreements)}) — gold for few-shot examples:")
        for r, j, h in disagreements[: args.max_show]:
            ctx = r.get("id") or r.get("text") or ""
            ctx = (" — " + str(ctx)[:60]) if ctx else ""
            out.append(f"    judge={j!r} human={h!r}{ctx}")
        if len(disagreements) > args.max_show:
            out.append(f"    ... and {len(disagreements) - args.max_show} more")
    report = "\n".join(out)
    print(report)
    if args.html:
        _write_html(args.html, args.file, n, agree, k, labels, m, disagreements)
        print(f"  html report -> {args.html}")
    return 0


def _write_html(path, src, n, agree, k, labels, m, disagreements):
    rows_html = ""
    for h in labels:
        cells = "".join(f"<td>{m[h][j]}</td>" for j in labels)
        rows_html += f"<tr><th>{h}</th>{cells}</tr>"
    head = "".join(f"<th>{j}</th>" for j in labels)
    dis = "".join(
        f"<li><code>judge={j}</code> vs <code>human={h}</code>"
        f"{(' — ' + str(r.get('id') or r.get('text') or ''))[:80]}</li>"
        for r, j, h in disagreements[:50]
    )
    color = "#1a7f37" if k is not None and k >= SHIP_BAR else "#b35900"
    html = f"""<!doctype html><meta charset=utf-8><title>kappa report</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#111}}
.k{{font-size:2.4rem;font-weight:700;color:{color}}} table{{border-collapse:collapse;margin:1rem 0}}
td,th{{border:1px solid #ddd;padding:.3rem .6rem;text-align:center}} code{{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}}
.muted{{color:#666}}</style>
<h1>kappa report</h1><p class=muted>source: {src} · {n} rows · {agree}/{n} agree ({100*agree/n:.1f}%)</p>
<p>Cohen's &kappa; <span class=k>{k:+.3f}</span> <span class=muted>[{band(k)}] · ship bar {SHIP_BAR}</span></p>
<h2>Confusion (rows = human, cols = judge)</h2><table><tr><th></th>{head}</tr>{rows_html}</table>
<h2>Disagreements ({len(disagreements)})</h2><ul>{dis or '<li>none</li>'}</ul>
<p class=muted>Generated by kappa — zero-dependency LLM-as-a-Judge eval kit.</p>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------- drift

def cmd_drift(args):
    bp, _ = load_pairs(args.baseline, args.judge_key, args.human_key)
    cp, _ = load_pairs(args.current, args.judge_key, args.human_key)
    kb, kc = cohen_kappa(bp), cohen_kappa(cp)
    if kb is None or kc is None:
        sys.exit("error: a file had no labeled rows")
    delta = kc - kb
    print(f"kappa drift")
    print(f"  baseline: kappa {kb:+.3f}  [{band(kb)}]  (n={len(bp)})")
    print(f"  current:  kappa {kc:+.3f}  [{band(kc)}]  (n={len(cp)})")
    print(f"  delta:    {delta:+.3f}")
    alerts = []
    if delta <= -args.threshold:
        alerts.append(f"kappa dropped by {-delta:.3f} (>= {args.threshold} threshold)")
    if kc < SHIP_BAR <= kb:
        alerts.append(f"current fell below the {SHIP_BAR} ship bar")
    if alerts:
        print("  ALERT:")
        for a in alerts:
            print(f"    - {a}")
        return 1
    print("  ok: no significant drift")
    return 0


# ---------------------------------------------------------------- lint

# (regex, human_label, fix). Each encodes a known LLM-as-a-Judge failure mode.
_LINT_CHECKS = [
    (
        r"\brubric\b|\bcriteria\b|\bscale\b|\bscores?\b|\brate\b|\brating\b|\bpoints?\b|\d\s*[-–]\s*\d{1,3}|out of \d",
        "explicit rubric / scoring scale",
        "Define explicit criteria and a scale — don't ask 'is this good?'. Vague judges are noisy.",
    ),
    (
        r"\b(step[- ]by[- ]step|explain|reasoning|justify|cite|evidence|because|before (you )?(decid|scor|answer))\b",
        "evidence-before-verdict",
        "Make the judge cite specifics BEFORE the verdict — it kills rubber-stamping.",
    ),
    (
        r"\b(for each|per (dimension|criterion|aspect)|dimensions?|separately|one at a time)\b",
        "decomposed judging",
        "Judge one dimension at a time (correctness / safety / completeness), not one global score.",
    ),
    (
        r"\b(length|verbos|concise|word count|do not (favor|prefer) longer|token count)\b",
        "verbosity-bias control",
        "Say longer != better, or normalize for length — judges over-reward verbose answers.",
    ),
]
# Pairwise comparison without an order-randomization safeguard => position bias.
_PAIRWISE = r"\b(response|answer|option|candidate)\s*(a|b|1|2)\b|\bwhich (one )?is better\b|\bcompare the two\b"
_ORDER_SAFE = r"\b(random(ize|ised|ized)?|both orders?|swap(ped)?|order[- ]independent|either order|positional? bias)\b"


def cmd_lint(args):
    if args.file == "-":
        text = sys.stdin.read()
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    low = text.lower()
    print(f"kappa lint — {args.file}")
    findings, passed = [], 0
    for rx, label, fix in _LINT_CHECKS:
        if re.search(rx, low):
            findings.append(("ok", label, ""))
            passed += 1
        else:
            findings.append(("warn", label, fix))
    # pairwise position-bias check (conditional)
    if re.search(_PAIRWISE, low):
        if re.search(_ORDER_SAFE, low):
            findings.append(("ok", "position-bias control (pairwise)", ""))
            passed += 1
        else:
            findings.append((
                "warn", "position-bias control (pairwise)",
                "Pairwise prompt detected — randomize A/B order or score each side independently.",
            ))
        total = len(_LINT_CHECKS) + 1
    else:
        total = len(_LINT_CHECKS)

    for status, label, fix in findings:
        if status == "ok":
            print(f"  ✓ {label}")
        else:
            print(f"  ⚠ {label}\n      fix: {fix}")
    print(f"  ---\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


# ---------------------------------------------------------------- cli

def build_parser():
    p = argparse.ArgumentParser(prog="kappa", description="LLM-as-a-Judge evaluation kit (zero-dependency).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_keys(sp):
        sp.add_argument("--judge-key", default="judge", help="JSON key for the judge label (default: judge)")
        sp.add_argument("--human-key", default="human", help="JSON key for the human label (default: human)")

    s = sub.add_parser("score", help="agreement %, Cohen's kappa, confusion, disagreements")
    s.add_argument("file", help="JSONL of {judge, human, [id|text]} (use - for stdin)")
    add_keys(s)
    s.add_argument("--max-show", type=int, default=20, help="max disagreements to list")
    s.add_argument("--html", metavar="OUT", help="also write a self-contained HTML report")
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("drift", help="compare agreement between two batches")
    d.add_argument("baseline")
    d.add_argument("current")
    add_keys(d)
    d.add_argument("--threshold", type=float, default=0.10, help="kappa-drop alert threshold (default 0.10)")
    d.set_defaults(func=cmd_drift)

    l = sub.add_parser("lint", help="heuristic linter for judge-prompt bias pitfalls")
    l.add_argument("file", help="judge prompt text file (use - for stdin)")
    l.set_defaults(func=cmd_lint)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
