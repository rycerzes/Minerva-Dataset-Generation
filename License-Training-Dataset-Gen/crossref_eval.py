"""Held-out-RENDERING eval — a non-circular benchmark for the verbatim regime.

The Atarashi dataset's test split cannot measure identification: ~86% of its
fragments are verbatim substrings of the very reference documents they are matched
against, so Recall@k there is a substring lookup. Cleaning cannot fix that — the
defect is between the query and the index, not inside the dataset.

This eval removes the circularity at the source instead. Both ScanCode and
FOSSology maintain their own text for the same licenses, transcribed independently.
So:

    index  = ScanCode  license texts   (all of them, so distractors are realistic)
    query  = FOSSology license text    for the same license

A hit means the matcher recognised *the license* across two independent
renderings, not that it found a string it had already been shown.

Queries are tiered by how far the two renderings actually diverge, because the
tiers measure different things and blending them is misleading:

    identical   normalized texts are equal      -> exact-match check only
    substring   one contains the other          -> containment check
    divergent   genuinely different wording     -> THE benchmark number

Report the divergent tier as the headline. `--query-mode head` truncates queries to
a header-sized excerpt, which tests the notice regime from an independent source.

  python crossref_eval.py --query-mode full
  python crossref_eval.py --query-mode head --refs both
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from atarashi_real_corpus_eval import load_references, encode_lexical, topk_keys, metrics

SC_CACHE = Path("cache/scancode_licenses.json")
FO_CACHE = Path("cache/fossology_licenses.json")
RESULTS = Path("output/crossref_eval.json")

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def canon(t: str) -> str:
    """Punctuation-insensitive form, for deciding how far two renderings diverge."""
    return _PUNCT.sub(" ", norm(t)).strip()


def load_pairs():
    """(license_key, scancode_text, fossology_text) for licenses both sources carry."""
    sc = {}
    for d in json.load(open(SC_CACHE)):
        t = (d.get("license_text") or "").strip()
        if t:
            sc[d["license_key"].lower()] = t
    fo = {}
    for d in json.load(open(FO_CACHE)):
        t = (d.get("rf_text") or "").strip()
        if t:
            fo.setdefault(d["rf_shortname"].lower(), t)
    return [(k, sc[k], fo[k]) for k in sorted(set(sc) & set(fo))], sc


def tier_of(sc_text: str, fo_text: str) -> str:
    a, b = canon(sc_text), canon(fo_text)
    if a == b:
        return "identical"
    if a in b or b in a:
        return "substring"
    return "divergent"


def build_queries(query_mode: str, head_chars: int):
    pairs, _ = load_pairs()
    out = []
    for key, sc_text, fo_text in pairs:
        tier = tier_of(sc_text, fo_text)
        text = fo_text if query_mode == "full" else fo_text[:head_chars]
        if len(text.split()) < 12:      # too short to identify from
            continue
        out.append({"text": text, "gt": {key}, "tier": tier, "key": key})
    return out


def evaluate(units, queries, label):
    ref_keys = [k for k, _ in units]
    ref_texts = [t for _, t in units]
    Q, R = encode_lexical(ref_texts, [q["text"] for q in queries])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-mode", choices=["full", "head"], default="full",
                    help="full license text as query, or a header-sized excerpt")
    ap.add_argument("--head-chars", type=int, default=1200)
    ap.add_argument("--refs", choices=["fulltext", "both"], default="fulltext",
                    help="index full texts only, or add the ScanCode notice layer")
    args = ap.parse_args()

    queries = build_queries(args.query_mode, args.head_chars)
    print(f"cross-rendering queries: {len(queries)}  "
          f"tiers: {dict(Counter(q['tier'] for q in queries))}")

    ref = load_references()
    units = [(k, ref[k]) for k in sorted(ref)]
    if args.refs == "both":
        from notice_refs import load_notice_refs
        units += [(k, t) for k, t in load_notice_refs() if k in ref]

    res = evaluate(units, queries, f"refs={args.refs} query={args.query_mode}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "config": vars(args), "n_queries": len(queries),
        "tiers": dict(Counter(q["tier"] for q in queries)), "results": res}, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
