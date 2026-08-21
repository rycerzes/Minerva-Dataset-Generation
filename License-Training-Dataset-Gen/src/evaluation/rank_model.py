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


def extract(agent, rows, k2, corpus: str) -> list[dict]:
    """One record per query: feature rows for its candidates plus the correct index."""
    out = []
    for r in rows:
        hits = agent.matcher.match(r["text"], min_run=agent.min_run)
        if not hits:
            continue
        labels = [1 if k2.get(h.shortname) in r["gt"] else 0 for h in hits]
        out.append({
            "corpus": corpus,
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

    deb = stratify([r for r in load_debian_corpus(k2) if r["regime"] == "notice"], limit_deb)
    allq = build_queries(0, s2k, refs, None)
    spdx = [q for q in stratify([q for q in allq if q["regime"] == "notice"], limit_spdx)
            if q["gt"] & reach]
    return extract(agent, deb, k2, "debian") + extract(agent, spdx, k2, "spdx-tag")


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
            scaler.transform(np.array(r["x"], dtype=float)[:, _columns()]))[:, 1]
        learned += r["y"][int(np.argmax(p))]
        baseline += r["y"][0]          # the matcher's own order
    return learned / scored, baseline / scored


# The rank *position* is not a feature: given it, a linear model simply reproduces
# the ordering it was meant to improve (weight -1.6, delta +0.002). It was dropped
# from FEATURE_NAMES entirely, so every column is used.
def _columns():
    return list(range(len(FEATURE_NAMES)))


def fit(records, seed=0):
    """A shallow gradient-boosted tree, chosen by measurement.

    Logistic regression fails here (-0.025 without `rank`): the signals interact —
    an unmatched reference tail only matters when coverage is otherwise high — and a
    linear combination cannot express that. LambdaMART-style LTR was ruled out on
    scale: it is validated at 10,000 queries and 136 features against our ~620 and 15.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    cols = _columns()
    X, y, _ = flatten(records)
    X = X[:, cols]
    scaler = StandardScaler().fit(X)
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                           min_samples_leaf=10, random_state=seed)
    model.fit(scaler.transform(X), y)
    return model, scaler


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="re-extract features")
    ap.add_argument("--folds", type=int, default=5, help="grouped CV folds")
    ap.add_argument("--leave-family-out", action="store_true",
                    help="hold out a whole license family — tests whether the model "
                         "learned a general ranking rule or memorised per-family "
                         "feature signatures")
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
        model, scaler = fit([r for r in records if any(r["y"])])
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Families seen in training. Outside them the model is measurably worse than
        # the hand-tuned key, so inference declines rather than guessing.
        families = sorted({family(n) for r in records
                           for n, lab in zip(r["names"], r["y"]) if lab})
        joblib.dump({"model": model, "scaler": scaler, "families": families,
                     "features": FEATURE_NAMES, "n_queries": len(records)}, args.save)
        print(f"  trained families ({len(families)}): {', '.join(families)}")
        print(f"\nwrote ranker artifact -> {args.save} "
              f"({args.save.stat().st_size/1e3:.0f} KB)")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
