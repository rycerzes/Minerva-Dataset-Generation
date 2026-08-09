"""Shared retrieval machinery for the license-identification evals.

Encoders, top-k selection and metrics used by every suite. This previously lived
inside ``atarashi_real_corpus_eval.py``, so each new eval had to import its shared
helpers from another eval's CLI script. Keeping it separate lets a suite depend on
the machinery without depending on someone else's entry point.

All encoders return ``(query_matrix, reference_matrix)`` with L2-normalized rows,
so a plain dot product is cosine similarity.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def norm(text: str) -> str:
    """Whitespace- and case-normalized form, used for identity comparisons."""
    return " ".join(text.lower().split())


# ---- encoders ---------------------------------------------------------------


def encode_lexical(ref_texts, q_texts):
    """TF-IDF char(3-5) + word(1-2) cosine — the askalono/ScanCode-style baseline.

    Char n-grams catch templated legal phrasing across punctuation and spelling
    drift; word n-grams catch distinctive tokens. Concatenating both beats either.
    """
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Rc, Rw = vc.fit_transform(ref_texts), vw.fit_transform(ref_texts)
    Qc, Qw = vc.transform(q_texts), vw.transform(q_texts)
    return normalize(hstack([Qc, Qw]).tocsr()), normalize(hstack([Rc, Rw]).tocsr())


def encode_embeddings(model_name, is_model2vec, ref_texts, q_texts):
    """Dense retrieval baseline. Kept for head-to-head comparison only.

    Embeddings underperform the lexical baseline on this task (see the findings
    doc); they are measured, not deployed.
    """
    from sentence_transformers import SentenceTransformer

    if is_model2vec:
        from sentence_transformers.models import StaticEmbedding
        model = SentenceTransformer(modules=[StaticEmbedding.from_model2vec(model_name)])
    else:
        model = SentenceTransformer(model_name, device="cpu")
    # Contextual models read only their first ~512 tokens; cap chars to bound CPU
    # time and memory (full license texts run to tens of KB -> OOM otherwise).
    cap = 4000 if is_model2vec else 2000
    bs = 128 if is_model2vec else 16

    def enc(xs):
        return model.encode([x[:cap] for x in xs], normalize_embeddings=True,
                            batch_size=bs, convert_to_numpy=True, show_progress_bar=False)

    return enc(q_texts), enc(ref_texts)


# ---- selection --------------------------------------------------------------


def build_loo_mask(mode, ref_texts, q_texts):
    """Per-query reference units to exclude — leave-one-out against label circularity.

    When ground truth comes from ScanCode's matcher and the reference index is built
    from the rules that matcher used, a query can be "identified" by regurgitating
    the exact rule that produced its label.

      exact    drop units whose normalized text equals the query (provable
               self-matches only; conservative)
      contain  additionally drop units whose text is contained in the query, since
               the rule text is often a span of the matched region. Stricter, and
               slightly pessimistic — a legitimately short notice for the right
               license can also be contained in a longer one.

    Measured result: masking 35,850 query-unit pairs (84% of queries) moved R@1 by
    0.004. The rule corpus is too redundant for this to bite — apache-2.0 alone has
    1,339 notice rules. Retained as a probe, not a fix.
    """
    if mode == "off":
        return None
    by_text = defaultdict(list)
    for i, t in enumerate(ref_texts):
        by_text[norm(t)].append(i)
    mask = []
    for q in q_texts:
        nq = norm(q)
        drop = set(by_text.get(nq, ()))
        if mode == "contain":
            drop |= {i for t, idxs in by_text.items() if t and t in nq for i in idxs}
        mask.append(drop)
    print(f"leave-one-out ({mode}): masked {sum(len(m) for m in mask)} query-unit pairs; "
          f"{sum(1 for m in mask if m)}/{len(mask)} queries lost >=1 unit")
    return mask


def topk_keys(Q, R, r_keys, k=5, mask=None):
    """Top-k reference keys per query by cosine. Works for dense or sparse rows."""
    r_keys = np.asarray(r_keys)
    sims = Q @ R.T
    if hasattr(sims, "toarray"):
        sims = sims.toarray()
    sims = np.asarray(sims, dtype=float)
    if mask:
        for i, drop in enumerate(mask):
            if drop:
                sims[i, list(drop)] = -np.inf
    k = min(k, R.shape[0])
    part = np.argpartition(-sims, range(k), axis=1)[:, :k]
    rows = np.arange(Q.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    return r_keys[part[rows, order]]  # (n_queries, k)


def top1_scores(Q, R, mask=None):
    """Best similarity per query — the distribution an abstention threshold sits on."""
    sims = Q @ R.T
    if hasattr(sims, "toarray"):
        sims = sims.toarray()
    sims = np.asarray(sims, dtype=float)
    if mask:
        for i, drop in enumerate(mask):
            if drop:
                sims[i, list(drop)] = -np.inf
    return sims.max(axis=1)


# ---- metrics ----------------------------------------------------------------


def metrics(topk, gts):
    """Recall@1/@5 and MRR, where a hit = any ground-truth component in the top-k.

    ``macro_recall_at_1`` averages per-license rather than per-query. Report it:
    micro is dominated by whichever licenses happen to be frequent (apache-2.0 is
    ~64% of real-world occurrences), so it moves with the class mix rather than
    with quality.
    """
    n = len(gts)
    r1 = r5 = 0
    rr = 0.0
    per_lic = defaultdict(lambda: [0, 0])  # primary gt key -> [hit@1, count]
    for row, gt in zip(topk, gts):
        primary = sorted(gt)[0]
        per_lic[primary][1] += 1
        ranks = [i + 1 for i, key in enumerate(row) if key in gt]
        if ranks:
            r5 += 1
            rr += 1.0 / ranks[0]
            if ranks[0] == 1:
                r1 += 1
                per_lic[primary][0] += 1
    macro = float(np.mean([h / c for h, c in per_lic.values()])) if per_lic else 0.0
    return dict(n=n, recall_at_1=r1 / n if n else 0.0, recall_at_5=r5 / n if n else 0.0,
                mrr=rr / n if n else 0.0, macro_recall_at_1=macro,
                per_license={k: dict(hit1=h, n=c) for k, (h, c) in sorted(per_lic.items())})
