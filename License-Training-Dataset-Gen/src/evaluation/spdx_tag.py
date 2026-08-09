"""SPDX-tag eval — the non-circular headline metric.

Ground truth is the author-declared ``SPDX-License-Identifier:`` in real source
files, a label source independent of any text matcher. The query is the file
header with the tag line(s) stripped, so identification happens from the prose —
mirroring what Nirjas forwards to Atarashi.

Three properties make this a benchmark rather than a spot check:

* **Scale** — ``--per-lang 0`` reads every file in the snapshot (30 x 10k), not
  the first 1,500 per language: 445 -> ~2,985 labelled queries.
* **Stratification** — ``--max-per-license`` caps any one license. Uncapped,
  apache-2.0 is ~64% of queries and micro-R@1 tracks the license mix rather than
  quality (it swings 0.605 -> 0.782 with the cap alone). Macro is stable at ~0.65;
  report macro, with the cap stated.
* **Regime split** — a file carrying only a bare tag has no license prose once the
  tag is stripped. ~72% of tagged files are like this, and the correct answer for
  them is UNKNOWN, not a guess. Scoring them as misses is what made the original
  blended figure (0.317) look like an engine failure. They are now reported
  separately, and their top-1 similarity distribution is printed so an abstention
  threshold can be calibrated against real negatives.

  uv run src/run_eval.py spdx-tag --per-lang 0 --max-per-license 150
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.corpus import iter_stack_files
from evaluation.references import load_references, reference_units, spdx_to_key_map
from evaluation.retrieval import encode_lexical, metrics, topk_keys, top1_scores

RESULTS = Path(__file__).resolve().parents[2] / "output" / "spdx_tag_eval.json"

TAG = re.compile(
    r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+(?:\s+(?:WITH|AND|OR)\s+[A-Za-z0-9.\-+]+)*)",
    re.I)
TAGLINE = re.compile(r"(?im)^.*SPDX-License-Identifier:.*$")
_SPLIT = re.compile(r"\s+(?:AND|OR|WITH)\s+", re.IGNORECASE)

# Markers of an actual license grant/terms register. Deliberately excludes bare
# "copyright": a copyright line with no grant is precisely the no-signal case
# (~64% of misses in the original error analysis), so counting it as prose would
# hide the regime we most need to measure.
_MARKERS = (
    "license", "licence", "licensed", "warranty", "permission", "redistribut",
    "terms and conditions", "free software", "merchantability", "sublicense",
    "disclaim", "public license", "gnu", "apache", "mozilla", "without restriction",
)
MIN_MARKERS = 2
HEADER_CHARS = 1500


def gt_keys(tag: str, s2k: dict[str, str], refs: dict[str, str]) -> set[str]:
    """Map an SPDX tag (possibly compound) to reference keys we can score against."""
    out = set()
    for part in _SPLIT.split(tag.strip().lower()):
        key = part.strip().rstrip(".")
        if key.startswith("licenseref-"):
            continue
        resolved = s2k.get(key, key)      # SPDX id -> ScanCode key, else the raw token
        if resolved in refs:
            out.add(resolved)
    return out


def classify_regime(text: str) -> str:
    """``notice`` if the tag-stripped header carries license prose, else ``no-signal``."""
    low = text.lower()
    return "notice" if sum(1 for m in _MARKERS if m in low) >= MIN_MARKERS else "no-signal"


def build_queries(per_lang: int, s2k: dict[str, str], refs: dict[str, str],
                  snapshot=None, keep_no_signal: bool = True) -> list[dict]:
    """Query set from SPDX-tagged files; ``per_lang=0`` reads every file."""
    seen, out = set(), []
    for lang, content in iter_stack_files(per_lang, snapshot, min_chars=0):
        found = TAG.search(content)
        if not found:
            continue
        gt = gt_keys(found.group(1), s2k, refs)
        if not gt:
            continue
        query = " ".join(TAGLINE.sub(" ", content[:HEADER_CHARS]).split())
        regime = classify_regime(query)
        if regime == "no-signal" and not keep_no_signal:
            continue
        if regime == "notice" and len(query) < 40:
            continue
        sig = (query, frozenset(gt))
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"text": query, "gt": gt, "tag": found.group(1),
                    "regime": regime, "lang": lang})
    return out


def stratify(queries: list[dict], max_per_license: int) -> list[dict]:
    """Cap each primary ground-truth license's contribution (deterministic order)."""
    if not max_per_license:
        return queries
    kept, counts = [], Counter()
    for q in queries:
        primary = sorted(q["gt"])[0]
        if counts[primary] >= max_per_license:
            continue
        counts[primary] += 1
        kept.append(q)
    return kept


def run(label: str, units, queries) -> dict:
    ref_keys = [k for k, _ in units]
    Q, R = encode_lexical([t for _, t in units], [q["text"] for q in queries])
    m = metrics(topk_keys(Q, R, ref_keys, k=5), [q["gt"] for q in queries])
    print(f"\n=== lexical / refs={label} ({len(units)} units) ===")
    print(f"  n={m['n']}  R@1={m['recall_at_1']:.3f}  R@5={m['recall_at_5']:.3f}  "
          f"MRR={m['mrr']:.3f}  MACRO-R@1={m['macro_recall_at_1']:.3f}")
    return m


def no_signal_report(units, queries) -> dict | None:
    """Top-1 similarity on no-signal queries — the abstention calibration set.

    These carry no license prose, so every answer is wrong by construction. The
    similarity a matcher assigns them is the false-positive distribution an UNKNOWN
    threshold must sit above.
    """
    if not queries:
        return None
    Q, R = encode_lexical([t for _, t in units], [q["text"] for q in queries])
    top1 = top1_scores(Q, R)
    pct = {f"p{p}": round(float(np.percentile(top1, p)), 4) for p in (50, 75, 90, 95, 99)}
    print(f"\n=== no-signal queries (n={len(queries)}) — correct answer is UNKNOWN ===")
    print(f"  top-1 similarity: mean={top1.mean():.4f} " +
          " ".join(f"{k}={v}" for k, v in pct.items()))
    print("  -> an abstention threshold must exceed these to keep precision.")
    return {"n": len(queries), "top1_mean": round(float(top1.mean()), 4), **pct}


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-lang", type=int, default=0, help="files per language; 0 = all")
    ap.add_argument("--max-per-license", type=int, default=150, help="cap per license; 0 = off")
    ap.add_argument("--snapshot", default=None, help="override the-stack-smol data dir")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    refs = load_references()
    allq = build_queries(args.per_lang, spdx_to_key_map(), refs, args.snapshot)
    notice_all = [q for q in allq if q["regime"] == "notice"]
    nosig = [q for q in allq if q["regime"] == "no-signal"]
    queries = stratify(notice_all, args.max_per_license)

    by_lic = Counter(sorted(q["gt"])[0] for q in queries)
    print(f"SPDX-tagged files with a resolvable label: {len(allq)}")
    print(f"  notice regime : {len(notice_all)}  -> {len(queries)} after stratification "
          f"(cap={args.max_per_license or 'none'})")
    print(f"  no-signal     : {len(nosig)}  (tag only, no prose -> expect UNKNOWN)")
    print(f"distinct licenses in the scored set: {len(by_lic)}")
    print("gt histogram:", by_lic.most_common(15))

    full = reference_units(refs, "fulltext")
    both = reference_units(refs, "both")
    results = {"fulltext": run("fulltext", full, queries),
               "both": run("both", both, queries)}
    ns = no_signal_report(both, nosig)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "n_queries": len(queries), "n_no_signal": len(nosig), "n_licenses": len(by_lic),
        "config": {"per_lang": args.per_lang, "max_per_license": args.max_per_license},
        "results": results, "no_signal": ns}, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
