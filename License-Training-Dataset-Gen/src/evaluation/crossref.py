"""Cross-rendering eval — a non-circular benchmark for the verbatim regime.

The Atarashi dataset's test split cannot measure identification: ~86% of its
fragments are verbatim substrings of the very reference documents they are scored
against, so Recall@k there is a substring lookup. Cleaning does not fix that — the
defect is between query and index, not inside the dataset.

This removes the circularity at the source. ScanCode and FOSSology each maintain
their own text for the same licenses, transcribed independently:

    index = ScanCode license texts   (all of them, so distractors are realistic)
    query = FOSSology license text   for the same license

A hit means the matcher recognised *the license* across two independent
renderings, not that it retrieved a string it had already been shown.

Queries are tiered by how far the renderings actually diverge, because the tiers
measure different things and blending them misleads:

    identical   normalized texts equal        -> exact-match check only
    substring   one contains the other        -> containment check
    divergent   genuinely different wording   -> THE benchmark number

``--query-mode head`` truncates queries to a header-sized excerpt, testing the
notice regime from an independent source.

  uv run src/run_eval.py crossref --query-mode full
  uv run src/run_eval.py crossref --query-mode head --refs both
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from evaluation.references import (load_fossology_texts, load_references,
                                   load_scancode_texts, reference_units)
from evaluation.retrieval import encode_lexical, metrics, norm, topk_keys

RESULTS = Path(__file__).resolve().parents[2] / "output" / "crossref_eval.json"

_PUNCT = re.compile(r"[^a-z0-9 ]+")
MIN_QUERY_WORDS = 12


def canon(text: str) -> str:
    """Punctuation-insensitive form, for deciding how far two renderings diverge.

    Whitespace is re-collapsed *after* punctuation is blanked: dropping a comma
    leaves two adjacent spaces, and without this two renderings differing only in
    punctuation compare unequal and get scored as `divergent` — inflating the
    headline tier with pairs that are not actually reworded.
    """
    return " ".join(_PUNCT.sub(" ", norm(text)).split())


def tier_of(sc_text: str, fo_text: str) -> str:
    a, b = canon(sc_text), canon(fo_text)
    if a == b:
        return "identical"
    if a in b or b in a:
        return "substring"
    return "divergent"


def build_queries(query_mode: str, head_chars: int) -> list[dict]:
    """One query per license carried by both sources, tagged with its tier."""
    sc, fo = load_scancode_texts(), load_fossology_texts()
    out = []
    for key in sorted(set(sc) & set(fo)):
        text = fo[key] if query_mode == "full" else fo[key][:head_chars]
        if len(text.split()) < MIN_QUERY_WORDS:
            continue
        out.append({"text": text, "gt": {key}, "tier": tier_of(sc[key], fo[key]), "key": key})
    return out


def evaluate(units, queries, label: str) -> dict:
    ref_keys = [k for k, _ in units]
    Q, R = encode_lexical([t for _, t in units], [q["text"] for q in queries])
    tk = topk_keys(Q, R, ref_keys, k=5)
    gts = [q["gt"] for q in queries]

    overall = metrics(tk, gts)
    print(f"\n=== {label} — {len(units)} reference units, {len(queries)} queries ===")
    per_tier = {}
    for tier in ("divergent", "substring", "identical"):
        idx = [i for i, q in enumerate(queries) if q["tier"] == tier]
        if not idx:
            continue
        m = metrics([tk[i] for i in idx], [gts[i] for i in idx])
        per_tier[tier] = m
        star = "  <-- headline" if tier == "divergent" else ""
        print(f"  {tier:<10} n={m['n']:<4} R@1={m['recall_at_1']:.3f}  "
              f"R@5={m['recall_at_5']:.3f}  MRR={m['mrr']:.3f}{star}")
    print(f"  {'ALL':<10} n={overall['n']:<4} R@1={overall['recall_at_1']:.3f}  "
          f"R@5={overall['recall_at_5']:.3f}  MRR={overall['mrr']:.3f}")

    misses = [queries[i]["key"] for i in range(len(queries))
              if queries[i]["tier"] == "divergent" and tk[i][0] not in gts[i]]
    if misses:
        print(f"  divergent misses ({len(misses)}): {', '.join(misses[:12])}"
              + (" ..." if len(misses) > 12 else ""))
    return {"overall": overall, "per_tier": per_tier, "divergent_misses": misses}


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-mode", choices=["full", "head"], default="full",
                    help="full license text as query, or a header-sized excerpt")
    ap.add_argument("--head-chars", type=int, default=1200)
    ap.add_argument("--refs", choices=["fulltext", "both"], default="fulltext",
                    help="index full texts only, or add the ScanCode notice layer")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    queries = build_queries(args.query_mode, args.head_chars)
    tiers = dict(Counter(q["tier"] for q in queries))
    print(f"cross-rendering queries: {len(queries)}  tiers: {tiers}")

    units = reference_units(load_references(), args.refs)
    res = evaluate(units, queries, f"refs={args.refs} query={args.query_mode}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"config": vars(args), "n_queries": len(queries),
                                   "tiers": tiers, "results": res}, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
