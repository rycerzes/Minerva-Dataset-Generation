"""Lexical-cascade probe: precision@coverage with abstention.

Raw recall is the wrong target for a license scanner — it rewards guessing. The
production comparison is ScanCode, which answers ~28% of a real corpus at ~0.77
precision and abstains on the rest. This measures the same shape: how much is
answered, how precise those answers are, and how often the matcher answers when
the correct output is UNKNOWN.

Two scorers, both over the notice-augmented index:

* **cosine** (baseline) — normalizes by reference length, so a short notice query
  is penalised against a long license body.
* **BM25** — saturating idf-weighted term scoring, length-normalized by document
  rather than by query, which suits short queries against long references.

Evaluated on the independent SPDX-tag corpus, split by regime:

* ``notice`` queries are answerable — precision is measured on those.
* ``no-signal`` queries carry no license prose, so *any* answer is wrong. Their
  answer rate at a given threshold is the false-positive rate.

The default ``--tau`` comes from measurement, not taste: no-signal top-1 cosine
runs mean 0.135 / p95 0.219 / p99 0.303, so ~0.30 abstains on about 99% of true
negatives.

  uv run src/run_eval.py cascade --tau 0.30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.references import load_references, reference_units, spdx_to_key_map
from evaluation.retrieval import encode_lexical, metrics, topk_keys, top1_scores
from evaluation.spdx_tag import build_queries, stratify

RESULTS = Path(__file__).resolve().parents[2] / "output" / "cascade_eval.json"


def bm25_topk(unit_texts, unit_keys, q_texts, k=5, k1=1.5, b=0.75):
    """BM25 ranking as a single sparse product.

    Precomputes the saturated idf-weighted term matrix S so that
    ``score(q, u) = sum_{t in q} S[u, t]`` is one ``Qb @ S.T``.
    """
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", min_df=1)
    U = cv.fit_transform(unit_texts).tocsr().astype(np.float64)
    N = U.shape[0]
    df = np.asarray((U > 0).sum(0)).ravel()
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
    dl = np.asarray(U.sum(1)).ravel()
    avgdl = dl.mean() or 1.0

    U = U.tocoo()
    denom = U.data + k1 * (1 - b + b * dl[U.row] / avgdl)
    S = sp.csr_matrix((idf[U.col] * U.data * (k1 + 1) / denom, (U.row, U.col)),
                      shape=(N, len(idf)))
    Qb = cv.transform(q_texts)
    Qb.data[:] = 1.0                                    # query term presence
    scores = (Qb @ S.T).toarray()

    keys = np.asarray(unit_keys)
    kk = min(k, N)
    part = np.argpartition(-scores, range(kk), axis=1)[:, :kk]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    return keys[part[rows, order]], scores[rows, part[rows, order]][:, 0]


def precision_at_coverage(top, best, gts, tau, n_no_signal_answered=None, n_no_signal=0):
    """Answer only above ``tau``; report coverage and precision on what was answered."""
    answered = np.where(np.asarray(best) >= tau)[0]
    hits = sum(1 for i in answered if top[i][0] in gts[i])
    out = {
        "tau": tau,
        "answered": int(len(answered)),
        "total": len(gts),
        "coverage": round(len(answered) / len(gts), 4) if gts else 0.0,
        "precision_at_answered": round(hits / len(answered), 4) if len(answered) else None,
    }
    if n_no_signal:
        out["no_signal_answered"] = int(n_no_signal_answered)
        out["no_signal_false_answer_rate"] = round(n_no_signal_answered / n_no_signal, 4)
    return out


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=0.30,
                    help="abstain when the top score is below this (cosine scale)")
    ap.add_argument("--per-lang", type=int, default=0)
    ap.add_argument("--max-per-license", type=int, default=150)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    refs = load_references()
    allq = build_queries(args.per_lang, spdx_to_key_map(), refs)
    notice = stratify([q for q in allq if q["regime"] == "notice"], args.max_per_license)
    nosig = [q for q in allq if q["regime"] == "no-signal"]

    units = reference_units(refs, "both")
    ukeys = [k for k, _ in units]
    utexts = [t for _, t in units]
    gts = [q["gt"] for q in notice]
    q_texts = [q["text"] for q in notice]
    print(f"notice queries={len(notice)}  no-signal={len(nosig)}  units={len(units)}")

    # --- cosine baseline, with abstention measured on both regimes -----------
    Q, R = encode_lexical(utexts, q_texts)
    cos_top = topk_keys(Q, R, ukeys, k=5)
    cos_best = top1_scores(Q, R)
    m_cos = metrics(cos_top, gts)
    print(f"\ncosine   R@1={m_cos['recall_at_1']:.3f}  R@5={m_cos['recall_at_5']:.3f}  "
          f"MRR={m_cos['mrr']:.3f}  macro={m_cos['macro_recall_at_1']:.3f}")

    ns_answered = 0
    if nosig:
        Qn, Rn = encode_lexical(utexts, [q["text"] for q in nosig])
        ns_answered = int((top1_scores(Qn, Rn) >= args.tau).sum())
    pac = precision_at_coverage(cos_top, cos_best, gts, args.tau, ns_answered, len(nosig))
    print(f"cosine + UNKNOWN (tau={args.tau}): answered {pac['answered']}/{pac['total']} "
          f"({pac['coverage']:.0%})  precision@answered={pac['precision_at_answered']}")
    if nosig:
        print(f"  no-signal falsely answered: {ns_answered}/{len(nosig)} "
              f"({pac['no_signal_false_answer_rate']:.1%}) — every one is a wrong answer")

    # --- BM25 (scores are unbounded; ranking comparison only) ---------------
    bm_top, _ = bm25_topk(utexts, ukeys, q_texts, k=5)
    m_bm = metrics(bm_top, gts)
    print(f"\nbm25     R@1={m_bm['recall_at_1']:.3f}  R@5={m_bm['recall_at_5']:.3f}  "
          f"MRR={m_bm['mrr']:.3f}  macro={m_bm['macro_recall_at_1']:.3f}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "config": vars(args), "n_notice": len(notice), "n_no_signal": len(nosig),
        "cosine": m_cos, "bm25": m_bm, "precision_at_coverage": pac}, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
