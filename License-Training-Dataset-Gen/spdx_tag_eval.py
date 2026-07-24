"""INDEPENDENT eval of the notice-reference layer.

The main real-corpus eval labels queries with ScanCode, so adding ScanCode notice
rules as references makes it circular (optimistic). Here the ground truth is the
author-declared `SPDX-License-Identifier:` tag found in real source files — a label
source independent of ScanCode's text matching. Query = the file's header notice
with the SPDX tag line(s) stripped (so we identify from the prose, not the tag),
mirroring what Nirjas forwards to Atarashi.

Compares fulltext-only vs +notice-layer reference indexes with the lexical matcher.

  python spdx_tag_eval.py --per-lang 1500
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from atarashi_real_corpus_eval import load_references, encode_lexical, topk_keys, metrics

SNAP = Path("/home/asyin/.cache/huggingface/hub/datasets--bigcode--the-stack-smol/"
            "snapshots/4a6938ce94446f324c6629e7de00ac591710044b/data")
TAG = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+(?:\s+(?:WITH|AND|OR)\s+[A-Za-z0-9.\-+]+)*)", re.I)
TAGLINE = re.compile(r"(?im)^.*SPDX-License-Identifier:.*$")
_SPLIT = re.compile(r"\s+(?:AND|OR|WITH)\s+", re.IGNORECASE)


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


def build_queries(per_lang, s2k, ref):
    seen, out = set(), []
    for d in sorted(SNAP.iterdir()):
        f = d / "data.json"
        if not f.exists():
            continue
        for i, line in enumerate(f.open()):
            if i >= per_lang:
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
            if len(q) < 40:
                continue
            sig = (q, frozenset(gt))
            if sig in seen:
                continue
            seen.add(sig)
            out.append({"text": q, "gt": gt, "tag": m.group(1)})
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=1500)
    args = ap.parse_args()
    ref = load_references()
    s2k = spdx_to_key_map()
    queries = build_queries(args.per_lang, s2k, ref)
    print(f"independent SPDX-tag queries: {len(queries)}  "
          f"distinct gt licenses: {len(set(sorted(q['gt'])[0] for q in queries))}")
    print("gt histogram:", Counter(sorted(q['gt'])[0] for q in queries).most_common(12))

    from notice_refs import load_notice_refs
    full = [(k, ref[k]) for k in sorted(ref)]
    notices = [(k, t) for k, t in load_notice_refs() if k in ref]
    res = {"fulltext": run("fulltext", full, queries),
           "both": run("both", full + notices, queries)}
    Path("output/spdx_tag_eval.json").write_text(json.dumps(
        {"n_queries": len(queries), "results": res}, indent=2))
    print("\nwrote output/spdx_tag_eval.json")


if __name__ == "__main__":
    main()
