#!/usr/bin/env python3
"""Honest real-corpus eval for Atarashi license IDENTIFICATION.

Why this exists
---------------
The on-disk Atarashi test split cannot be trusted as a benchmark: 90.8% of its
fragments are verbatim substrings of their own license reference text (they were
sliced from the very texts a retriever matches against), and 8-9% overlap train
exactly. Any Recall@k on it measures copy-detection, not identification, and it
structurally favours lexical matching. See docs / the data audit.

This eval instead uses REAL license occurrences as they appear in source files
(the-stack-smol), with ground truth from ScanCode's licensedcode. Each query is
matched against one canonical reference text per license (scancode wins,
fossology fills gaps) and we report Recall@1/@5/MRR.

Honest-scope caveat
-------------------
Real source files overwhelmingly carry the top ~20 licenses. This eval therefore
measures HEAD identification quality (the production-relevant number). The
~3000-license tail barely occurs in real code and is an index-COVERAGE concern,
not something measurable on real data. Micro numbers are dominated by Apache/MIT;
the MACRO (per-license-averaged) number is the honest headline.

Queries come from cache/real_corpus_eval.json (label==1 rows), the same corpus
the Nirjas gate eval builds via eval_real_corpus.py. Regenerate/expand that with
`python eval_real_corpus.py --pos 1500 --neg 0` for more license diversity.

Retrievers
----------
  --method lexical    TF-IDF char(3-5) + word cosine (askalono/scancode-style baseline)
  --method model2vec  static embeddings   (default minishlab/potion-retrieval-32M)
  --method st         sentence-transformer (default sentence-transformers/all-MiniLM-L6-v2)

Usage
-----
  P=/home/asyin/swapnil/fossology-gsoc/minerva-dataset-pipeline/.venv/bin/python
  $P atarashi_real_corpus_eval.py --method lexical
  $P atarashi_real_corpus_eval.py --method model2vec --model minishlab/potion-retrieval-32M
  $P atarashi_real_corpus_eval.py --method st --model BAAI/bge-small-en-v1.5
  $P atarashi_real_corpus_eval.py --method all      # lexical + potion-retrieval + bge-small
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

CORPUS_CACHE = Path("cache/real_corpus_eval.json")
SC_CACHE = Path("cache/scancode_licenses.json")
FO_CACHE = Path("cache/fossology_licenses.json")
RESULTS = Path("output/atarashi_real_corpus_eval.json")

# ScanCode license expression -> component keys. Splits AND/OR/WITH; drops operators.
_SPLIT = re.compile(r"\s+(?:AND|OR|WITH)\s+", re.IGNORECASE)
# ground-truth keys that carry no identifiable reference text -> skip the query
_SKIP_KEYS = {"unknown-license-reference", "unknown", "proprietary-license",
              "warranty-disclaimer", "generic-cla", "other-permissive", "other-copyleft"}


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def gt_components(expr: str) -> set[str]:
    """Component license keys of a (possibly compound) ScanCode expression, lowercased."""
    parts = [p.strip().lower() for p in _SPLIT.split(expr) if p.strip()]
    return {p for p in parts if p and p not in _SKIP_KEYS}


def load_references(sc_path=SC_CACHE, fo_path=FO_CACHE) -> dict[str, str]:
    """One canonical license text per license_key (scancode wins, fossology fills gaps)."""
    ref: dict[str, str] = {}
    for d in json.load(open(sc_path)):
        t = (d.get("license_text") or "").strip()
        if t:
            ref[d["license_key"].lower()] = t
    for d in json.load(open(fo_path)):
        t = (d.get("rf_text") or "").strip()
        if t:
            ref.setdefault(d["rf_shortname"].lower(), t)
    return ref


def load_queries(ref: dict[str, str], dedup=True) -> list[dict]:
    """Real-world license occurrences (label==1) whose ground truth has a reference."""
    corpus = json.loads(CORPUS_CACHE.read_text())
    seen, out = set(), []
    dropped_noref = dropped_dup = 0
    for r in corpus:
        if r.get("label") != 1:
            continue
        keys = gt_components(r["license"])
        keys = {k for k in keys if k in ref}
        if not keys:
            dropped_noref += 1
            continue
        if dedup:
            sig = (norm(r["text"]), frozenset(keys))
            if sig in seen:
                dropped_dup += 1
                continue
            seen.add(sig)
        out.append({"text": r["text"], "gt": keys, "expr": r["license"],
                    "lang": r.get("lang"), "score": r.get("score")})
    print(f"queries: kept={len(out)} dropped_noref={dropped_noref} dropped_dup={dropped_dup}")
    return out


# ---- retrievers: each returns (query_matrix, ref_matrix) L2-normalized rows ----

def encode_lexical(ref_texts, q_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from scipy.sparse import hstack
    # char n-grams catch templated license phrasing; word n-grams catch tokens.
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Rc = vc.fit_transform(ref_texts); Rw = vw.fit_transform(ref_texts)
    Qc = vc.transform(q_texts);       Qw = vw.transform(q_texts)
    R = normalize(hstack([Rc, Rw]).tocsr())
    Q = normalize(hstack([Qc, Qw]).tocsr())
    return Q, R


def encode_embeddings(model_name, is_model2vec, ref_texts, q_texts):
    from sentence_transformers import SentenceTransformer
    if is_model2vec:
        from sentence_transformers.models import StaticEmbedding
        model = SentenceTransformer(modules=[StaticEmbedding.from_model2vec(model_name)])
    else:
        model = SentenceTransformer(model_name, device="cpu")
    # Contextual models read only their first ~512 tokens; cap chars to bound CPU
    # time and memory (full license texts run to tens of KB -> OOM otherwise).
    cap = 4000 if is_model2vec else 2000
    clip = lambda xs: [x[:cap] for x in xs]
    bs = 128 if is_model2vec else 16
    enc = lambda xs: model.encode(clip(xs), normalize_embeddings=True, batch_size=bs,
                                  convert_to_numpy=True, show_progress_bar=False)
    return enc(q_texts), enc(ref_texts)


def topk_keys(Q, R, r_keys, k=5):
    """Top-k reference keys per query by cosine. Works for dense or sparse rows."""
    r_keys = np.asarray(r_keys)
    sims = Q @ R.T
    if hasattr(sims, "toarray"):
        sims = sims.toarray()
    sims = np.asarray(sims)
    k = min(k, R.shape[0])
    part = np.argpartition(-sims, range(k), axis=1)[:, :k]
    rows = np.arange(Q.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    return r_keys[part[rows, order]]  # (n_queries, k)


def metrics(topk, gts):
    """Recall@1/@5 and MRR where a hit = any ground-truth component in the top-k."""
    n = len(gts)
    r1 = r5 = 0
    rr = 0.0
    per_lic_hit = defaultdict(lambda: [0, 0])  # primary gt key -> [hit@1, count]
    for row, gt in zip(topk, gts):
        primary = sorted(gt)[0]
        per_lic_hit[primary][1] += 1
        ranks = [i + 1 for i, key in enumerate(row) if key in gt]
        if ranks:
            r5 += 1
            rr += 1.0 / ranks[0]
            if ranks[0] == 1:
                r1 += 1
                per_lic_hit[primary][0] += 1
    macro = np.mean([h / c for h, c in per_lic_hit.values()]) if per_lic_hit else 0.0
    return dict(n=n, recall_at_1=r1 / n, recall_at_5=r5 / n, mrr=rr / n,
                macro_recall_at_1=float(macro),
                per_license={k: dict(hit1=h, n=c) for k, (h, c) in sorted(per_lic_hit.items())})


def run_method(name, model, is_m2v, ref_keys, ref_texts, queries):
    q_texts = [q["text"] for q in queries]
    gts = [q["gt"] for q in queries]
    t0 = time.time()
    if name == "lexical":
        Q, R = encode_lexical(ref_texts, q_texts)
    else:
        Q, R = encode_embeddings(model, is_m2v, ref_texts, q_texts)
    dt = time.time() - t0
    tk = topk_keys(Q, R, ref_keys, k=5)
    m = metrics(tk, gts)
    m["encode_s"] = round(dt, 1)
    m["throughput_qps"] = round(len(q_texts) / dt, 1) if dt else None
    label = name if name == "lexical" else f"{name}:{model}"
    print(f"\n=== {label} ===")
    print(f"  n={m['n']}  R@1={m['recall_at_1']:.3f}  R@5={m['recall_at_5']:.3f}  "
          f"MRR={m['mrr']:.3f}  MACRO-R@1={m['macro_recall_at_1']:.3f}  "
          f"({m['throughput_qps']} q/s incl refs, {m['encode_s']}s)")
    print("  per-license R@1:")
    for lk, s in m["per_license"].items():
        print(f"    {lk:<28} {s['hit1']}/{s['n']}")
    return label, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["lexical", "model2vec", "st", "all"], default="all")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    ref = load_references()
    print(f"reference index: {len(ref)} licenses")
    ref_keys = sorted(ref)
    ref_texts = [ref[k] for k in ref_keys]
    queries = load_queries(ref)
    print(f"distinct ground-truth licenses in eval: "
          f"{len(set(sorted(q['gt'])[0] for q in queries))}")

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
            label, m = run_method(name, model, is_m2v, ref_keys, ref_texts, queries)
            results[label] = m
        except Exception as e:
            print(f"\n!! {name} failed: {type(e).__name__}: {e}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"n_refs": len(ref), "results": results}, indent=2))
    print(f"\nresults -> {RESULTS}")


if __name__ == "__main__":
    main()
