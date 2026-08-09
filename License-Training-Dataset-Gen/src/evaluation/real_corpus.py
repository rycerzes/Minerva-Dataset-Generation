#!/usr/bin/env python3
"""Real-corpus license identification eval (ScanCode-labelled — optimistic).

Queries are real license occurrences as they appear in source files, with ground
truth from ScanCode's ``licensedcode``. Broad coverage: ~1,700 unique queries over
~134 licenses after dedup.

**Read the numbers with care.** The labels come from ScanCode, and the notice
reference layer is built from the rules that same matcher uses, so ``--refs
notices/both`` shares provenance with the ground truth and scores optimistically
(R@1 0.799 here vs 0.605-0.78 on the independent SPDX-tag suite). ``--loo`` probes
whether that is mere rule regurgitation; measurement says it is not — the rule
corpus is too redundant for masking to bite. Use ``spdx_tag`` or ``crossref`` for
an independent read.

Scope: real files carry the top ~20 licenses, so this measures HEAD quality — the
production-relevant number. The ~3,000-license tail barely occurs in real code and
is an index *coverage* concern, not something measurable here. Micro is dominated
by Apache/MIT; macro is the honest headline.

  uv run src/run_eval.py real-corpus --method lexical --refs both
  uv run src/run_eval.py real-corpus --method all --refs fulltext
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation.corpus import load_real_corpus
from evaluation.references import gt_components, load_references, reference_units
from evaluation.retrieval import (build_loo_mask, encode_embeddings, encode_lexical,
                                  metrics, norm, topk_keys)

RESULTS = Path(__file__).resolve().parents[2] / "output" / "atarashi_real_corpus_eval.json"


def load_queries(refs: dict[str, str], dedup: bool = True) -> list[dict]:
    """Label==1 rows whose ground truth maps onto a license we hold a reference for."""
    seen, out = set(), []
    dropped_noref = dropped_dup = 0
    for row in load_real_corpus():
        if row.get("label") != 1:
            continue
        keys = {k for k in gt_components(row["license"]) if k in refs}
        if not keys:
            dropped_noref += 1
            continue
        if dedup:
            sig = (norm(row["text"]), frozenset(keys))
            if sig in seen:
                dropped_dup += 1
                continue
            seen.add(sig)
        out.append({"text": row["text"], "gt": keys, "expr": row["license"],
                    "lang": row.get("lang"), "score": row.get("score")})
    print(f"queries: kept={len(out)} dropped_noref={dropped_noref} dropped_dup={dropped_dup}")
    return out


def run_method(name, model, is_m2v, ref_keys, ref_texts, queries, mask=None):
    q_texts = [q["text"] for q in queries]
    t0 = time.time()
    if name == "lexical":
        Q, R = encode_lexical(ref_texts, q_texts)
    else:
        Q, R = encode_embeddings(model, is_m2v, ref_texts, q_texts)
    dt = time.time() - t0
    m = metrics(topk_keys(Q, R, ref_keys, k=5, mask=mask), [q["gt"] for q in queries])
    m["encode_s"] = round(dt, 1)
    m["throughput_qps"] = round(len(q_texts) / dt, 1) if dt else None
    label = name if name == "lexical" else f"{name}:{model}"
    print(f"\n=== {label} ===")
    print(f"  n={m['n']}  R@1={m['recall_at_1']:.3f}  R@5={m['recall_at_5']:.3f}  "
          f"MRR={m['mrr']:.3f}  MACRO-R@1={m['macro_recall_at_1']:.3f}  "
          f"({m['throughput_qps']} q/s incl refs, {m['encode_s']}s)")
    return label, m


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["lexical", "model2vec", "st", "all"], default="lexical")
    ap.add_argument("--model", default=None)
    ap.add_argument("--refs", choices=["fulltext", "notices", "both"], default="fulltext",
                    help="index layer; notices/both share ScanCode provenance with the "
                         "labels and therefore score optimistically")
    ap.add_argument("--loo", choices=["off", "exact", "contain"], default="off",
                    help="leave-one-out probe for label circularity")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    refs = load_references()
    units = reference_units(refs, args.refs)
    ref_keys = [k for k, _ in units]
    ref_texts = [t for _, t in units]
    print(f"reference index: {len(refs)} licenses, {len(units)} units (refs={args.refs})")

    queries = load_queries(refs)
    print(f"distinct ground-truth licenses: {len(set(sorted(q['gt'])[0] for q in queries))}")
    mask = build_loo_mask(args.loo, ref_texts, [q["text"] for q in queries])

    jobs = []
    if args.method in ("lexical", "all"):
        jobs.append(("lexical", None, False))
    if args.method in ("model2vec", "all"):
        jobs.append(("model2vec", args.model or "minishlab/potion-retrieval-32M", True))
    if args.method in ("st", "all"):
        jobs.append(("st", args.model or "sentence-transformers/all-MiniLM-L6-v2", False))

    results = {}
    for name, model, is_m2v in jobs:
        try:
            label, m = run_method(name, model, is_m2v, ref_keys, ref_texts, queries, mask)
            results[label] = m
        except Exception as exc:
            print(f"\n!! {name} failed: {type(exc).__name__}: {exc}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"n_refs": len(refs), "config": vars(args), "results": results}, indent=2))
    print(f"\nresults -> {RESULTS}")


if __name__ == "__main__":
    main()
