"""Learned candidate scoring — replacing the hand-ordered ranking key.

The cascade retrieves well and orders badly. On the DEP-5 corpus, 62 of 77 wrong
answers already carry the correct license inside our own top-5, 28 of them at rank 2;
perfect reranking would take precision from 0.795 to 0.960, past ScanCode's 0.834.
The headroom is entirely in the ranking function.

Hand-tuning that function has been tried five times and failed five times —
required-phrase gating, phrases as a tie-break, preferring the larger reference,
diff-derived discriminators as a gate, and the per-candidate confidence bar. Every one
of those signals carries real information and every hard threshold over them cost more
than it bought. That is the case for learning the combination instead.

Deliberately a small model over engineered features, not a text model. Cross-encoders
and embeddings systematically under-weight the small decisive span — under 40% on
high-overlap/different-meaning pairs (PAWS), below chance when overlap disagrees with
the label (HANS) — which is exactly the GPL-2-vs-GPL-3 case. The discriminating
evidence is instead handed to the model as explicit features (`ref_tail`,
`substitution`, `ref_gap`) computed deterministically by the matcher.

    uv run src/run_eval.py rank-model --train
    uv run src/run_eval.py rank-model --cross-corpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.references import load_references, spdx_to_key_map
from evaluation.spdx_tag import build_queries, stratify

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "output" / "rank_model.json"
FEATURES = ROOT / "cache" / "rank_features.json"

# Imported, never redeclared: inference uses the same code, and a silent mismatch
# between training and inference features would poison every score without erroring.
from atarashi.libs.ranker import FEATURE_NAMES, family, featurize  # noqa: E402
from atarashi.libs.references import expression_components  # noqa: E402


def candidate_keys(shortname: str, k2: dict[str, str]) -> set[str] | None:
    """The reference keys a candidate attests to, or None if any component is unmapped.

    A candidate may be a whole expression since the index carries compound rules, so
    the comparison against ground truth is set against set. ``None`` means the
    candidate cannot be scored, which is not the same as being wrong.
    """
    keys = {k2.get(part) for part in expression_components(shortname)}
    return None if None in keys else keys


def extract(agent, rows, k2, corpus: str, regime: str = "notice") -> list[dict]:
    """One record per query: feature rows for its candidates plus the correct index.

    No-signal queries are included with an all-zero label vector. They are what teaches
    the model to say "none of these" — without them its probabilities only ever rank
    candidates against each other and cannot support an accept/abstain decision.

    A candidate is correct when the set of licenses it attests to *equals* the ground
    truth, not when it merely contains one of them. Anything looser rewards adding an
    exception to a file that does not carry one: on a plain GPL-2.0-or-later file,
    ``GPL-2.0-or-later WITH Classpath-exception-2.0`` is the wrong answer, and under
    "any component matches" the model would be taught it is right. Set equality also
    aligns the training target with exact-set — the metric a reviewer consumes —
    without costing R@1, since a correct compound answer still reports its primary
    license first.
    """
    out = []
    for r in rows:
        hits = agent.matcher.match(r["text"], min_run=agent.min_run,
                                   top_k=getattr(agent, "top_k", 5))
        if not hits:
            continue
        labels = [0] * len(hits) if regime == "no-signal" else [
            1 if candidate_keys(h.shortname, k2) == set(r["gt"]) else 0 for h in hits]
        out.append({
            "corpus": corpus,
            "regime": regime,
            "x": featurize(hits),
            "y": labels,
            "names": [h.shortname for h in hits],
            "gt": sorted(r["gt"]),
        })
    return out


def build_dataset(limit_spdx=150, limit_deb=150) -> list[dict]:
    from evaluation.agent import key_map, load_agent, load_debian_corpus

    refs = load_references()
    s2k = spdx_to_key_map()
    agent, df = load_agent()
    k2 = key_map(df["shortname"], s2k, refs)
    reach = set(k2.values())

    corpus_rows = load_debian_corpus(k2)
    deb = stratify([r for r in corpus_rows if r["regime"] == "notice"], limit_deb)
    deb_neg = [r for r in corpus_rows if r["regime"] == "no-signal"]
    allq = build_queries(0, s2k, refs, None)
    spdx = [q for q in stratify([q for q in allq if q["regime"] == "notice"], limit_spdx)
            if q["gt"] & reach]
    # Cap the negatives: they outnumber the positives 2:1 in DEP-5 and 8:1 in the
    # SPDX-tag corpus, and a model swamped by them abstains on everything.
    spdx_neg = [q for q in allq if q["regime"] == "no-signal"][:len(spdx) * 2]
    return (extract(agent, deb, k2, "debian")
            + extract(agent, spdx, k2, "spdx-tag")
            + extract(agent, deb_neg[:len(deb) * 2], k2, "debian", "no-signal")
            + extract(agent, spdx_neg, k2, "spdx-tag", "no-signal"))


def flatten(records):
    X = np.array([row for r in records for row in r["x"]], dtype=float)
    y = np.array([lab for r in records for lab in r["y"]], dtype=int)
    groups = np.array([i for i, r in enumerate(records) for _ in r["y"]])
    return X, y, groups


def top1_accuracy(model, records, scaler) -> tuple[float, float]:
    """Accuracy of argmax over each candidate list, and of the current ranker."""
    learned = baseline = scored = 0
    for r in records:
        if not any(r["y"]):
            continue           # correct answer absent: reranking cannot help
        scored += 1
        p = model.predict_proba(
            scaler.transform(_matrix(np.asarray(r["x"], float)[:, _columns()])))[:, 1]
        learned += r["y"][int(np.argmax(p))]
        baseline += r["y"][0]          # the matcher's own order
    return learned / scored, baseline / scored


# The rank *position* is not a feature: given it, a linear model simply reproduces
# the ordering it was meant to improve (weight -1.6, delta +0.002). It was dropped
# from FEATURE_NAMES entirely, so every column is used.
def _columns():
    return list(range(len(FEATURE_NAMES)))


def _matrix(rows) -> np.ndarray:
    """Contiguous float matrix.

    Fancy-indexing a column list yields a non-contiguous array, and
    ``StandardScaler.transform`` then accumulates in a different order — a 7e-14
    difference. That is invisible until it lands on a histogram bin boundary inside
    the tree ensemble, where it flipped nine of 269 predictions and made two
    supposedly identical evaluations disagree by 0.016 on leave-one-family-out.
    Contiguity is enforced here so results are reproducible rather than
    memory-layout dependent.
    """
    return np.ascontiguousarray(np.asarray(rows, dtype=float))


def fit(records, seed=0):
    """A shallow gradient-boosted tree, chosen by measurement.

    Logistic regression fails here (-0.025 without `rank`): the signals interact —
    an unmatched reference tail only matters when coverage is otherwise high — and a
    linear combination cannot express that. LambdaMART-style LTR was ruled out on
    scale: it is validated at 10,000 queries and 136 features against our ~620 and 15.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    X, y, _ = flatten(records)
    X = _matrix(X[:, _columns()])
    scaler = StandardScaler().fit(X)
    # Depth is the memorisation knob, and it trades in-distribution gain against
    # out-of-distribution safety. Measured on leave-one-license-family-out against
    # grouped CV: depth 3 gives +0.068 in-dist but -0.013 OOD; depth 2 gives +0.059
    # and +0.001; depth 1 gives +0.021 and +0.008. Depth 2 keeps nearly all the gain
    # while no longer harming licenses the model has never seen.
    model = HistGradientBoostingClassifier(max_depth=2, max_iter=200,
                                           min_samples_leaf=10, random_state=seed)
    model.fit(scaler.transform(X), y)
    return model, scaler


def conformal_threshold(records, alpha: float = 0.10, seed: int = 0,
                        folds: int = 5) -> dict:
    """Cross-conformal calibration of the accept threshold.

    Nonconformity for a candidate is ``1 - P(correct)``; the accept threshold is the
    ``ceil((n+1)(1-alpha))/n`` empirical quantile of those scores. The three set sizes
    it produces are the three behaviours the engine already has — empty is an
    abstention, one is an answer, several is an ambiguity.

    **Why cross-conformal and not the split version this replaces.** Splitting once
    calibrates on half the data and inherits the variance of that particular draw.
    Measured here, that variance is not academic: under nested CV the per-fold
    threshold ranged 0.231 to 0.427 at alpha=0.10, and the high draws cost enough
    coverage to put the engine significantly behind ScanCode on R@1 (McNemar p ~ 0.01)
    while the same predictions at a stable threshold were indistinguishable from it.
    Partitioning into ``folds`` and pooling the out-of-fold scores calibrates on every
    query instead of half of them, and averages over the split rather than betting on
    one. Each score still comes from a model that did not see that query, so
    calibration and test scores stay exchangeable.

    Cross-conformal is approximately rather than exactly valid, unlike split conformal
    (Vovk 2015); CV+/jackknife+ buy a proven 1-2*alpha bound at K times the fitting
    cost. At this scale the empirical stability is worth more than the exact
    finite-sample statement, because the split version's instability was itself
    costing coverage.

    The guarantee is **conditional on the correct licence being retrievable at all**.
    Calibration can only use queries whose candidate list contains the truth, so for
    the rest no threshold can help and the statement does not cover them. It is also
    marginal, not per-licence: class-conditional coverage needs roughly 100
    calibration points per class against the ~27 available.
    """
    solvable = [r for r in records if any(r["y"])]
    others = [r for r in records if not any(r["y"])]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(solvable))
    parts = np.array_split(order, folds)

    def score_with(fitted, r):
        model, scaler = fitted
        return model.predict_proba(
            scaler.transform(_matrix(np.asarray(r["x"], float)[:, _columns()])))[:, 1]

    cal, held = [], []
    for f in range(folds):
        out = parts[f]
        rest = [i for g in range(folds) if g != f for i in parts[g]]
        fitted = fit([solvable[i] for i in rest] + others, seed=seed)
        for i in out:
            r = solvable[i]
            p = score_with(fitted, r)
            cal.append(1.0 - p[int(np.argmax(r["y"]))])
            held.append((r, p))

    cal = np.asarray(cal)
    n = len(cal)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    qhat = float(np.sort(cal)[k - 1])

    covered = sizes = 0
    for r, p in held:
        keep = [j for j in range(len(p)) if (1.0 - p[j]) <= qhat]
        sizes += len(keep)
        covered += any(r["y"][j] for j in keep)
    return {"alpha": alpha, "qhat": qhat, "accept": round(1.0 - qhat, 4),
            "calibration_n": n, "folds": folds,
            "realised_coverage": round(covered / len(held), 4),
            "mean_set_size": round(sizes / len(held), 3),
            "retrievable": f"{len(solvable)}/{len(records)}"}


def coverage_threshold(records, target: float = 0.95, seed: int = 0,
                       folds: int = 5) -> dict:
    """The accept threshold that answers ``target`` of answerable queries, held out.

    Conformal alpha answers "is the correct licence inside the reported set", which is
    not the question the accept decision asks — "should we answer at all". Reading the
    operating point off alpha therefore put the engine at 0.86-0.92 coverage while
    ScanCode answered 0.96-0.99, and comparing there read as a deficit that the
    risk-coverage curve shows is a *choice*: over tau in [0.05, 0.20] the engine is
    statistically level with ScanCode on R@1 on both corpora (McNemar p 0.45-1.00)
    at equal or better precision.

    So the threshold is derived from a stated coverage target instead. Probabilities
    come from out-of-fold models, so the target is met on queries the model has not
    seen rather than in sample.
    """
    notice = [r for r in records if r.get("regime") != "no-signal"]
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(notice)), folds)
    others = [r for r in records if r.get("regime") == "no-signal"]
    tops = []
    for f in range(folds):
        out = set(parts[f].tolist())
        model, scaler = fit([notice[i] for i in range(len(notice)) if i not in out]
                            + others, seed=seed)
        for i in out:
            r = notice[i]
            if not r["x"]:
                continue
            p = model.predict_proba(scaler.transform(
                _matrix(np.asarray(r["x"], float)[:, _columns()])))[:, 1]
            tops.append(float(p.max()))
    tops = np.sort(np.asarray(tops))
    # Answer the top `target` share: cut just below the (1-target) quantile.
    k = int(np.floor((1.0 - target) * len(tops)))
    accept = float(tops[k]) if 0 <= k < len(tops) else 0.0
    realised = float((tops >= accept).sum() / len(tops))
    return {"target_coverage": target, "accept": round(accept, 4),
            "realised_coverage": round(realised, 4), "n": len(tops), "folds": folds}


def ambiguity_margin(records, seed: int = 0, folds: int = 5,
                     percentile: float = 1.0) -> dict:
    """The leader-to-runner-up gap below which an answer is reported as ambiguous.

    A property of the model, not a constant. The shipped 0.10 was derived from the
    37-licence model, where no correct answer was ever decided by a margin below
    0.195 — so flagging at 0.10 was free. Under the 152-licence model the score
    distribution is flatter and the same 0.10 flags **14 correct answers instead of
    3**, each of which then reports a set rather than a licence and loses exact-set.

    Derived the way the original was: the level at which flagging costs essentially no
    correct answer, taken as the ``percentile``-th percentile of the margins on
    correct decisions, measured out of fold.
    """
    notice = [r for r in records if r.get("regime") != "no-signal" and any(r["y"])]
    others = [r for r in records if r.get("regime") == "no-signal"]
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(notice)), folds)
    good = []
    for f in range(folds):
        out = set(parts[f].tolist())
        model, scaler = fit([notice[i] for i in range(len(notice)) if i not in out]
                            + others, seed=seed)
        for i in out:
            r = notice[i]
            if len(r["y"]) < 2:
                continue
            p = model.predict_proba(scaler.transform(
                _matrix(np.asarray(r["x"], float)[:, _columns()])))[:, 1]
            j = int(np.argmax(p))
            if not r["y"][j]:
                continue                      # wrong decisions do not set the bar
            order = np.sort(p)[::-1]
            good.append(float(order[0] - order[1]))
    margin = float(np.percentile(good, percentile)) if good else DEFAULT_MARGIN_FALLBACK
    return {"ambiguous_margin": round(margin, 4), "n_correct": len(good),
            "percentile": percentile}


DEFAULT_MARGIN_FALLBACK = 0.10


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="re-extract features")
    ap.add_argument("--folds", type=int, default=5, help="grouped CV folds")
    ap.add_argument("--leave-family-out", action="store_true",
                    help="hold out a whole license family — tests whether the model "
                         "learned a general ranking rule or memorised per-family "
                         "feature signatures")
    ap.add_argument("--accept", type=float, default=None,
                    help="accept the top candidate above this probability. Omit to "
                         "derive it by split-conformal calibration at --alpha.")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="target miscoverage for conformal calibration of --accept")
    ap.add_argument("--target-coverage", type=float, default=0.95,
                    help="share of answerable queries to answer; the accept threshold "
                         "is derived from this on a held-out risk-coverage curve. "
                         "Set to 0 to fall back to conformal --alpha.")
    ap.add_argument("--save", type=Path, default=None,
                    help="fit on everything and write the artifact for Cascade to load")
    ap.add_argument("--cross-corpus", action="store_true",
                    help="train on one corpus, test on the other — the honest test, "
                         "since license identity is heavily imbalanced within each")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.rebuild or not FEATURES.exists():
        records = build_dataset()
        FEATURES.parent.mkdir(parents=True, exist_ok=True)
        FEATURES.write_text(json.dumps(records))
    else:
        records = json.loads(FEATURES.read_text())

    n_cand = sum(len(r["y"]) for r in records)
    n_pos = sum(sum(r["y"]) for r in records)
    solvable = [r for r in records if any(r["y"])]
    print(f"queries {len(records)}  candidates {n_cand}  positives {n_pos}")
    print(f"queries whose correct answer is in the candidate list: {len(solvable)} "
          f"({len(solvable)/len(records):.1%}) — the reranking ceiling\n")

    out = {"n_queries": len(records), "n_candidates": n_cand, "n_positive": n_pos}

    if args.leave_family_out:
        import collections
        solvable = [r for r in records if any(r["y"])]

        # Reuse the same family rule inference gates on, so the held-out split and
        # the deployed gate cannot disagree about what "unseen family" means.
        by_family = collections.defaultdict(list)
        for r in solvable:
            by_family[family(r["gt"][0])].append(r)
        tot_l = tot_b = tot_n = 0
        rows_out = []
        for fam, rows in sorted(by_family.items(), key=lambda kv: -len(kv[1]))[:6]:
            train = [r for r in solvable if family(r["gt"][0]) != fam]
            model, scaler = fit(train)
            learned, base = top1_accuracy(model, rows, scaler)
            tot_l += learned * len(rows); tot_b += base * len(rows); tot_n += len(rows)
            print(f"  hold out {fam:8s} (n={len(rows):3d})  baseline {base:.4f}  "
                  f"learned {learned:.4f}  delta {learned - base:+.4f}")
            rows_out.append({"family": fam, "n": len(rows),
                             "baseline": base, "learned": learned})
        print(f"\n  WEIGHTED  baseline {tot_b/tot_n:.4f}  learned {tot_l/tot_n:.4f}  "
              f"delta {(tot_l - tot_b)/tot_n:+.4f}")
        out["leave_family_out"] = rows_out
        out["lofo_delta"] = (tot_l - tot_b) / tot_n
    elif args.cross_corpus:
        for train_c, test_c in (("spdx-tag", "debian"), ("debian", "spdx-tag")):
            tr = [r for r in records if r["corpus"] == train_c]
            te = [r for r in records if r["corpus"] == test_c]
            model, scaler = fit(tr)
            learned, base = top1_accuracy(model, te, scaler)
            print(f"train {train_c:9s} -> test {test_c:9s} "
                  f"baseline {base:.4f}  learned {learned:.4f}  "
                  f"delta {learned - base:+.4f}   (n={len([r for r in te if any(r['y'])])})")
            out[f"{train_c}->{test_c}"] = {"baseline": base, "learned": learned}
    else:
        from sklearn.model_selection import GroupKFold
        X, y, groups = flatten(records)
        gkf = GroupKFold(n_splits=args.folds)
        deltas = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
            tr_q = sorted({groups[i] for i in tr_idx})
            te_q = sorted({groups[i] for i in te_idx})
            model, scaler = fit([records[i] for i in tr_q])
            learned, base = top1_accuracy(model, [records[i] for i in te_q], scaler)
            deltas.append(learned - base)
            print(f"  fold {fold}: baseline {base:.4f}  learned {learned:.4f}  "
                  f"delta {learned - base:+.4f}")
        print(f"\nmean delta {np.mean(deltas):+.4f}  (std {np.std(deltas):.4f})")
        out["cv_mean_delta"] = float(np.mean(deltas))

        model, scaler = fit(records)
        from sklearn.inspection import permutation_importance
        cols = _columns()
        names = [FEATURE_NAMES[i] for i in cols]
        Xf, yf, _ = flatten(records)
        imp = permutation_importance(model, scaler.transform(Xf[:, cols]), yf,
                                     n_repeats=8, random_state=0,
                                     scoring="average_precision")
        order = np.argsort(-imp.importances_mean)
        print("\npermutation importance (average precision):")
        for i in order[:8]:
            print(f"  {names[i]:16s} {imp.importances_mean[i]:+.4f}")
        print("  -> the margin features dominate; they re-express the existing "
              "ranking rather than adding evidence.")
        out["importance"] = {names[i]: float(imp.importances_mean[i]) for i in order}

    if args.save:
        import joblib
        conformal = None
        accept = args.accept
        if accept is None and args.target_coverage:
            conformal = coverage_threshold(records, args.target_coverage)
            accept = conformal["accept"]
            print(f"\ncoverage-targeted accept >= {accept} "
                  f"(target {args.target_coverage}, realised "
                  f"{conformal['realised_coverage']} on {conformal['n']} held-out queries)")
        elif accept is None:
            conformal = conformal_threshold(records, args.alpha)
            accept = conformal["accept"]
            print(f"\nconformal calibration at alpha={args.alpha}: accept >= {accept} "
                  f"(qhat {conformal['qhat']:.4f}, n={conformal['calibration_n']})")
            print(f"  realised coverage {conformal['realised_coverage']} "
                  f"(target {1 - args.alpha:.2f}), mean set size "
                  f"{conformal['mean_set_size']}")
            print(f"  guarantee is conditional on the licence being retrievable: "
                  f"{conformal['retrievable']} queries")
        model, scaler = fit([r for r in records if any(r["y"])])
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Licences seen in training. Outside them the model is measurably worse than
        # the hand-tuned key, so inference declines rather than guessing.
        #
        # Recorded per *licence*, not per family. Family granularity let the model act
        # on `MIT-advertising` because `MIT` was trained, and it then promoted plain
        # MIT over it — on the Software Heritage tail the learned ranker was worse
        # than the matcher's own ordering, R@1 0.758 -> 0.697. Families are still
        # written for artifacts read by older code.
        trained = sorted({n for r in records
                          for n, lab in zip(r["names"], r["y"]) if lab})
        families = sorted({family(n) for n in trained})
        margin = ambiguity_margin(records)
        print(f"  ambiguity margin {margin['ambiguous_margin']} "
              f"(1st pct of {margin['n_correct']} correct decisions, held out)")
        joblib.dump({"model": model, "scaler": scaler, "families": families,
                     "licenses": trained,
                     "ambiguous_margin": margin["ambiguous_margin"],
                     "accept": accept, "conformal": conformal,
                     "features": FEATURE_NAMES, "n_queries": len(records)}, args.save)
        print(f"  trained licences ({len(trained)}) across "
              f"{len(families)} families: {', '.join(families)}")
        print(f"\nwrote ranker artifact -> {args.save} "
              f"({args.save.stat().st_size/1e3:.0f} KB)")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
