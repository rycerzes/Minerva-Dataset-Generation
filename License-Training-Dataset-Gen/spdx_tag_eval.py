"""INDEPENDENT eval of the notice-reference layer (non-circular headline metric).

The main real-corpus eval labels queries with ScanCode, so adding ScanCode notice
rules as references makes it circular (optimistic). Here the ground truth is the
author-declared `SPDX-License-Identifier:` tag found in real source files — a label
source independent of ScanCode's text matching. Query = the file's header notice
with the SPDX tag line(s) stripped (so we identify from the prose, not the tag),
mirroring what Nirjas forwards to Atarashi.

Three things make this a usable benchmark rather than a spot check:

* **Scale** — `--per-lang 0` scans every file in the local the-stack-smol snapshot
  (30 languages x 10k files) instead of the first 1,500 per language.
* **Stratification** — `--max-per-license` caps any one license's contribution.
  Un-capped, apache-2.0 is ~64% of queries and the macro average is dominated by
  whichever handful of licenses happen to be frequent.
* **Regime split** — a file carrying only a bare SPDX tag has *no license prose*
  once the tag is stripped; the correct answer there is UNKNOWN, not a guess.
  Those queries are reported separately (`no-signal`) instead of being silently
  dropped, and their top-1 similarity distribution is printed so an abstention
  threshold can be calibrated against real negatives.

  python spdx_tag_eval.py --per-lang 0 --max-per-license 150
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from atarashi_real_corpus_eval import load_references, encode_lexical, topk_keys, metrics

TAG = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+(?:\s+(?:WITH|AND|OR)\s+[A-Za-z0-9.\-+]+)*)", re.I)
TAGLINE = re.compile(r"(?im)^.*SPDX-License-Identifier:.*$")
_SPLIT = re.compile(r"\s+(?:AND|OR|WITH)\s+", re.IGNORECASE)

# Markers of an actual license *grant/terms* register. Deliberately excludes bare
# "copyright": a copyright line with no grant is the no-signal case (the eval
# findings attribute ~64% of misses to exactly these), so counting it as prose
# would hide the regime we most need to measure.
_MARKERS = (
    "license", "licence", "licensed", "warranty", "permission", "redistribut",
    "terms and conditions", "free software", "merchantability", "sublicense",
    "disclaim", "public license", "gnu", "apache", "mozilla", "without restriction",
)
MIN_MARKERS = 2  # distinct markers required to count as license prose


def find_snapshot() -> Path:
    """Locate the local the-stack-smol data dir (env var, then the HF cache)."""
    env = os.environ.get("STACK_SNAPSHOT")
    if env:
        return Path(env)
    bases = [Path(os.environ["HF_HOME"]) / "hub"] if os.environ.get("HF_HOME") else []
    bases.append(Path.home() / ".cache" / "huggingface" / "hub")
    for base in bases:
        snaps = sorted((base / "datasets--bigcode--the-stack-smol" / "snapshots").glob("*/data"))
        if snaps:
            return snaps[-1]
    raise SystemExit(
        "the-stack-smol snapshot not found. Set STACK_SNAPSHOT=/path/to/snapshots/<rev>/data "
        "or download it once with `datasets.load_dataset('bigcode/the-stack-smol')`."
    )


def spdx_to_key_map():
    """SPDX id (lowercased) -> ScanCode reference key, from the scancode cache."""
    m = {}
    for d in json.load(open("cache/scancode_licenses.json")):
        spdx = (d.get("spdx_license_key") or "").strip().lower()
        if spdx:
            m[spdx] = d["license_key"].lower()
    return m


def gt_keys(tag, s2k, ref):
    """Map an SPDX tag (maybe compound) to reference keys we can score against."""
    out = set()
    for part in _SPLIT.split(tag.strip().lower()):
        p = part.strip().rstrip(".")
        if p.startswith("licenseref-"):
            continue
        k = s2k.get(p, p)              # SPDX->scancode key, else try the raw token
        if k in ref:
            out.add(k)
    return out


def classify_regime(text: str) -> str:
    """`notice` if the tag-stripped header carries license prose, else `no-signal`."""
    low = text.lower()
    hits = sum(1 for m in _MARKERS if m in low)
    return "notice" if hits >= MIN_MARKERS else "no-signal"


def build_queries(per_lang, s2k, ref, snapshot=None, keep_no_signal=True):
    """Query set from SPDX-tagged files. ``per_lang=0`` reads every file.

    Returns rows of {text, gt, tag, regime, lang}. ``no-signal`` rows are files whose
    header holds a tag but no license prose — kept so abstention can be measured.
    """
    snap = Path(snapshot) if snapshot else find_snapshot()
    seen, out = set(), []
    for d in sorted(snap.iterdir()):
        f = d / "data.json"
        if not f.exists():
            continue
        for i, line in enumerate(f.open()):
            if per_lang and i >= per_lang:
                break
            try:
                c = json.loads(line).get("content", "") or ""
            except Exception:
                continue
            m = TAG.search(c)
            if not m:
                continue
            gt = gt_keys(m.group(1), s2k, ref)
            if not gt:
                continue
            header = TAGLINE.sub(" ", c[:1500])         # strip the tag line(s)
            q = " ".join(header.split())
            regime = classify_regime(q)
            if regime == "no-signal" and not keep_no_signal:
                continue
            if len(q) < 40 and regime == "notice":
                continue                                 # too short to carry prose
            sig = (q, frozenset(gt))
            if sig in seen:
                continue
            seen.add(sig)
            out.append({"text": q, "gt": gt, "tag": m.group(1),
                        "regime": regime, "lang": d.name})
    return out


def stratify(queries, max_per_license):
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


def run(units_label, units, queries):
    ref_keys = [k for k, _ in units]
    ref_texts = [t for _, t in units]
    Q, R = encode_lexical(ref_texts, [q["text"] for q in queries])
    tk = topk_keys(Q, R, ref_keys, k=5)
    m = metrics(tk, [q["gt"] for q in queries])
    print(f"\n=== lexical / refs={units_label} ({len(units)} units) ===")
    print(f"  n={m['n']}  R@1={m['recall_at_1']:.3f}  R@5={m['recall_at_5']:.3f}  "
          f"MRR={m['mrr']:.3f}  MACRO-R@1={m['macro_recall_at_1']:.3f}")
    return m


def no_signal_report(units, queries):
    """Top-1 similarity distribution on no-signal queries -> abstention calibration.

    These carry no license prose, so every answer is wrong by construction. The
    similarity a matcher assigns them is the false-positive score distribution an
    UNKNOWN threshold has to sit above.
    """
    if not queries:
        return None
    ref_texts = [t for _, t in units]
    Q, R = encode_lexical(ref_texts, [q["text"] for q in queries])
    sims = Q @ R.T
    if hasattr(sims, "toarray"):
        sims = sims.toarray()
    top1 = np.asarray(sims).max(axis=1)
    pct = {f"p{p}": round(float(np.percentile(top1, p)), 4) for p in (50, 75, 90, 95, 99)}
    print(f"\n=== no-signal queries (n={len(queries)}) — correct answer is UNKNOWN ===")
    print(f"  top-1 similarity: mean={top1.mean():.4f} " +
          " ".join(f"{k}={v}" for k, v in pct.items()))
    print("  -> an abstention threshold must exceed these to keep precision.")
    return {"n": len(queries), "top1_mean": round(float(top1.mean()), 4), **pct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=0,
                    help="files per language; 0 = all (default)")
    ap.add_argument("--max-per-license", type=int, default=150,
                    help="cap per ground-truth license; 0 = uncapped")
    ap.add_argument("--snapshot", default=None, help="override the-stack-smol data dir")
    args = ap.parse_args()

    ref = load_references()
    s2k = spdx_to_key_map()
    allq = build_queries(args.per_lang, s2k, ref, snapshot=args.snapshot)

    notice_all = [q for q in allq if q["regime"] == "notice"]
    nosig = [q for q in allq if q["regime"] == "no-signal"]
    queries = stratify(notice_all, args.max_per_license)

    by_lic = Counter(sorted(q["gt"])[0] for q in queries)
    print(f"SPDX-tagged files with a resolvable label: {len(allq)}")
    print(f"  notice regime : {len(notice_all)}  -> {len(queries)} after "
          f"stratification (cap={args.max_per_license or 'none'})")
    print(f"  no-signal     : {len(nosig)}  (tag only, no prose -> expect UNKNOWN)")
    print(f"distinct licenses in the scored set: {len(by_lic)}")
    print("gt histogram:", by_lic.most_common(15))
    print("languages:", Counter(q["lang"] for q in queries).most_common(8))

    from notice_refs import load_notice_refs
    full = [(k, ref[k]) for k in sorted(ref)]
    notices = [(k, t) for k, t in load_notice_refs() if k in ref]
    res = {"fulltext": run("fulltext", full, queries),
           "both": run("both", full + notices, queries)}
    ns = no_signal_report(full + notices, nosig)

    out = {"n_queries": len(queries), "n_no_signal": len(nosig),
           "n_licenses": len(by_lic),
           "config": {"per_lang": args.per_lang, "max_per_license": args.max_per_license},
           "results": res, "no_signal": ns}
    Path("output/spdx_tag_eval.json").write_text(json.dumps(out, indent=2))
    print("\nwrote output/spdx_tag_eval.json")


if __name__ == "__main__":
    main()
