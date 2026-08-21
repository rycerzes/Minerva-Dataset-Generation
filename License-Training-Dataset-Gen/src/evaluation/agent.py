"""Agent-level eval — scores ``Cascade.scan``, not a bag-of-words stand-in.

Every other suite here scores a retrieval probe: TF-IDF or BM25 over a reference
index. None of them run the agent. That gap meant no engine change could be shown
to help, and the agent's own thresholds had never been calibrated against the
score scale they gate. This suite closes it.

What it reports, all on the non-circular SPDX-tag corpus:

* **Native operating point** — what ``Cascade.scan`` actually does at its shipped
  thresholds: coverage and precision on the answerable notice regime, and the
  false-answer rate on the no-signal regime where every answer is wrong.
* **Threshold sweep** — precision@coverage across ``strong_run``, so the accepted
  operating point is chosen from a curve rather than reasoned about. The Phase 0
  thresholds were carried over from a cosine scale that no longer applies.
* **Index coverage ceiling** — the share of queries whose ground-truth license is
  absent from the agent's license list. Those are unanswerable by construction and
  cap recall regardless of matcher quality; reporting them separately keeps an
  index gap from reading as an accuracy gap.
* **ScanCode head-to-head** — the same queries through ``licensedcode`` on
  identical denominators. Comparing against a published 0.77/28% is not a
  comparison; this is.
* **``--dump-residual``** — the notice queries the agent abstains on, annotated
  with whether ScanCode also abstains. Cases where both abstain and a license is
  genuinely present are Residual B, the only residual an ML stage can address.

  uv run src/run_eval.py agent --per-lang 0 --max-per-license 150
  uv run src/run_eval.py agent --limit 300 --dump-residual output/residual.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation.references import SKIP_KEYS, load_references, spdx_to_key_map
from evaluation.spdx_tag import build_queries, stratify

RESULTS = Path(__file__).resolve().parents[2] / "output" / "agent_eval.json"

# Sweep range for the run-length acceptance rule. The shipped value is 20, chosen
# by reasoning rather than measurement; these bracket it generously.
SWEEP = (8, 12, 16, 20, 24, 30, 40)


def load_agent(strong_run: int | None = None, min_coverage: float | None = None,
               use_ranker: bool = True):
    """The agent under test, on its shipped license list.

    The gate is switched off deliberately. ``libs/gate.py`` imports ``nirjas.gate``,
    which is unreleased, so it already fails open — disabling it explicitly makes
    the suite deterministic instead of dependent on what happens to be installed,
    and the gate is measured on its own in the ``nirjas-gate`` suite.
    """
    from atarashi.agents.cascade import Cascade
    from atarashi.libs.decision import DEFAULT_MIN_COVERAGE, DEFAULT_STRONG_RUN

    csv = Path(__import__("atarashi").__file__).parent / "data" / "licenses" / "licenseList.csv"
    df = pd.read_csv(csv).fillna("").rename(columns={"text": "processed_text"})
    df = df[["shortname", "processed_text"]]
    return Cascade(df, use_gate=False, use_ranker=use_ranker,
                   strong_run=DEFAULT_STRONG_RUN if strong_run is None else strong_run,
                   min_coverage=DEFAULT_MIN_COVERAGE if min_coverage is None else min_coverage), df


def key_map(shortnames, s2k: dict[str, str], refs: dict[str, str]) -> dict[str, str]:
    """Agent shortname -> ScanCode reference key, so verdicts are comparable.

    The agent answers in FOSSology shortnames; ground truth is in ScanCode keys.
    Most shortnames are SPDX ids and map through ``spdx_to_key_map``; the rest are
    tried as keys directly, then as or-later spellings. Anything still unmapped is
    counted and reported rather than silently scored as a miss.
    """
    out: dict[str, str] = {}
    for name in shortnames:
        low = str(name).lower()
        for cand in (s2k.get(low), low, s2k.get(low.rstrip("+") + "-or-later"),
                     low.rstrip("+") + "-plus", low.rstrip("+")):
            if cand and cand in refs:
                out[str(name)] = cand
                break
    return out


def scan_all(agent, queries, suffix=".txt"):
    """Run the agent over every query; returns the top result dict per query.

    Queries are flattened single-line headers (the corpus builder strips the tag
    and collapses whitespace), so comment extraction finds nothing and the agent
    falls back to the raw text. That is the same input every other suite scores,
    which is what makes the numbers comparable.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    out, full = [], []
    try:
        for q in queries:
            Path(path).write_text(q["text"], errors="replace")
            results = agent.scan(path)
            out.append(results[0])
            full.append(results)
    finally:
        os.unlink(path)
    return out, full


def score(tops, queries, k2: dict[str, str], full=None) -> dict:
    """Coverage and precision on what the agent chose to answer.

    ``full`` is the whole reported result list per query, when available. Top-1
    metrics read result[0] and are blind to what else the engine reported
    alongside it — a bug that let the agent return 4.72 licenses per file, 97% of
    them carrying a license that was not the file's, while precision looked fine.
    ``exact_set`` is the metric a compliance reviewer actually consumes.
    """
    answered = hits = unmapped = exact = reported = 0
    for i, (top, q) in enumerate(zip(tops, queries)):
        name = top["shortname"]
        if name == "UNKNOWN":
            continue
        answered += 1
        key = k2.get(name)
        if key is None:
            unmapped += 1
        else:
            hits += key in q["gt"]
        if full is not None:
            names = {k2.get(h["shortname"]) for h in full[i]
                     if h["shortname"] != "UNKNOWN"}
            reported += len(names)
            exact += names == set(q["gt"])
    out_extra = ({"exact_set": round(exact / answered, 4) if answered else None,
                  "licenses_per_file": round(reported / answered, 3) if answered else None}
                 if full is not None else {})
    return {**out_extra,
        "n": len(queries),
        "answered": answered,
        "coverage": round(answered / len(queries), 4) if queries else 0.0,
        "correct": hits,
        "precision_at_answered": round(hits / answered, 4) if answered else None,
        "recall_at_1": round(hits / len(queries), 4) if queries else 0.0,
        "unmapped_answers": unmapped,
    }


def sweep_thresholds(agent, queries, k2, values=SWEEP) -> list[dict]:
    """precision@coverage across ``strong_run``, reusing one pass of the matcher.

    The acceptance rule is ``longest_run >= strong_run OR coverage >= min_coverage``,
    so the ranked hits do not depend on the threshold — only the accept/reject
    decision does. Matching once and re-thresholding keeps the sweep honest (same
    candidates every row) and cheap.
    """
    from atarashi.libs.decision import is_confident

    ranked = [agent.matcher.match(q["text"], min_run=agent.min_run) for q in queries]
    rows = []
    for strong in values:
        answered = hits = 0
        for hits_list, q in zip(ranked, queries):
            conf = [h for h in hits_list
                    if is_confident(h.score, h.longest_run, strong, agent.min_coverage)]
            if not conf:
                continue
            answered += 1
            key = k2.get(conf[0].shortname)
            if key and key in q["gt"]:
                hits += 1
        rows.append({
            "strong_run": strong,
            "answered": answered,
            "coverage": round(answered / len(queries), 4) if queries else 0.0,
            "precision": round(hits / answered, 4) if answered else None,
            "recall_at_1": round(hits / len(queries), 4) if queries else 0.0,
        })
    return rows


def scancode_verdicts(texts: list[str]) -> list[tuple[str | None, float]]:
    """Top ``licensedcode`` expression and score per query, or (None, 0.0)."""
    from licensedcode.cache import get_index

    idx = get_index()
    out = []
    for text in texts:
        matches = idx.match(query_string=text)
        if not matches:
            out.append((None, 0.0))
            continue
        best = max(matches, key=lambda m: m.score())
        out.append((best.rule.license_expression, best.score()))
    return out


def sc_keys(expr: str | None) -> set[str]:
    """Identifying license keys in a ScanCode expression, or empty for an abstention.

    ``unknown-license-reference`` and its siblings are how ScanCode says "there is
    licensing here but I cannot name it" — an abstention, not an answer. Scoring
    them as answers would both inflate its coverage and destroy its precision, which
    happens to flatter the agent, so the same ``SKIP_KEYS`` the ground-truth builder
    uses is applied to its verdicts too.
    """
    if not expr:
        return set()
    parts = {p.strip().lower() for p in expr.replace("(", " ").replace(")", " ").split()}
    return {p for p in parts if p and p not in ("and", "or", "with") and p not in SKIP_KEYS}


def score_scancode(verdicts, queries) -> dict:
    answered = hits = 0
    for (expr, _), q in zip(verdicts, queries):
        keys = sc_keys(expr)
        if not keys:
            continue
        answered += 1
        hits += bool(keys & q["gt"])
    return {
        "n": len(queries),
        "answered": answered,
        "coverage": round(answered / len(queries), 4) if queries else 0.0,
        "correct": hits,
        "precision_at_answered": round(hits / answered, 4) if answered else None,
        "recall_at_1": round(hits / len(queries), 4) if queries else 0.0,
    }


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("spdx-tag", "debian"), default="spdx-tag",
                    help="spdx-tag: in-file author declarations (15 licenses). "
                         "debian: DEP-5 maintainer records, independent and deeper "
                         "into the tail — build it with `debian-dep5 --mode corpus`.")
    ap.add_argument("--per-lang", type=int, default=0, help="files per language; 0 = all")
    ap.add_argument("--max-per-license", type=int, default=150, help="cap per license; 0 = off")
    ap.add_argument("--snapshot", default=None, help="override the-stack-smol data dir")
    ap.add_argument("--limit", type=int, default=0, help="cap scored queries (quick runs)")
    ap.add_argument("--no-scancode", action="store_true", help="skip the head-to-head")
    ap.add_argument("--no-ranker", action="store_true",
                    help="disable the learned scorer and use the hand-tuned rank key")
    ap.add_argument("--dump-residual", type=Path, default=None,
                    help="write abstained notice queries here (Residual A + B)")
    return ap


def load_debian_corpus(k2: dict[str, str]) -> list[dict]:
    """The DEP-5 corpus, reshaped to the query contract the scorer already uses.

    Ground truth arrives as a license shortname; every other corpus here keys on
    ScanCode reference keys. Converting here rather than at build time keeps the
    corpus in Debian's own terms and keeps one scoring path.
    """
    path = Path(__file__).resolve().parents[2] / "cache" / "debian_corpus.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — build it first:\n"
            "  uv run src/run_eval.py debian-dep5 --mode corpus --packages 700")
    out = []
    for row in json.loads(path.read_text()):
        key = k2.get(row["gt"])
        if not key:
            continue          # label outside the agent's list: an index gap, not a miss
        out.append({**row, "gt": {key}, "tag": row["dep5_name"], "lang": row["path"]})
    return out


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    refs = load_references()
    s2k = spdx_to_key_map()
    if args.corpus == "debian":
        agent_probe, df_probe = load_agent(use_ranker=not args.no_ranker)
        rows = load_debian_corpus(key_map(df_probe["shortname"], s2k, refs))
        notice = stratify([r for r in rows if r["regime"] == "notice"], args.max_per_license)
        nosig = [r for r in rows if r["regime"] == "no-signal"]
    else:
        allq = build_queries(args.per_lang, s2k, refs, args.snapshot)
        notice = stratify([q for q in allq if q["regime"] == "notice"], args.max_per_license)
        nosig = [q for q in allq if q["regime"] == "no-signal"]
    if args.limit:
        notice, nosig = notice[:args.limit], nosig[:args.limit]

    agent, df = load_agent(use_ranker=not args.no_ranker)
    k2 = key_map(df["shortname"], s2k, refs)
    reachable = set(k2.values())

    # An index gap is not an accuracy gap. Separate the two before scoring.
    answerable = [q for q in notice if q["gt"] & reachable]
    unreachable = len(notice) - len(answerable)

    print(f"corpus: {args.corpus}")
    print(f"notice queries : {len(notice)}  (no-signal {len(nosig)})")
    print(f"agent license list: {len(df)} licenses, {len(k2)} mapped to reference keys")
    print(f"ground truth outside the agent's list: {unreachable}/{len(notice)} "
          f"({unreachable / max(len(notice), 1):.1%}) — unanswerable by construction")
    print(f"scored notice queries: {len(answerable)}  "
          f"licenses: {len({sorted(q['gt'])[0] for q in answerable})}")

    print("\n=== agent, shipped thresholds "
          f"(strong_run={agent.strong_run}, min_coverage={agent.min_coverage}) ===")
    tops, full = scan_all(agent, answerable)
    native = score(tops, answerable, k2, full)
    print(f"  answered {native['answered']}/{native['n']} ({native['coverage']:.1%})  "
          f"precision {native['precision_at_answered']}  R@1 {native['recall_at_1']}")
    print(f"  exact-set {native['exact_set']}  ({native['licenses_per_file']} licenses "
          f"reported per file) — what a reviewer actually consumes")
    if native["unmapped_answers"]:
        print(f"  answers with no reference key: {native['unmapped_answers']} (scored as misses)")

    ns_tops = scan_all(agent, nosig)[0] if nosig else []
    ns_answered = sum(1 for t in ns_tops if t["shortname"] != "UNKNOWN")
    if nosig:
        print(f"  no-signal falsely answered: {ns_answered}/{len(nosig)} "
              f"({ns_answered / len(nosig):.1%}) — every one is wrong by construction")

    print("\n=== threshold sweep (strong_run) ===")
    rows = sweep_thresholds(agent, answerable, k2)
    print(f"  {'strong_run':>10} {'coverage':>9} {'precision':>10} {'R@1':>7}")
    for r in rows:
        print(f"  {r['strong_run']:>10} {r['coverage']:>9.1%} "
              f"{str(r['precision']):>10} {r['recall_at_1']:>7.3f}")

    sc = sc_verdicts = None
    if not args.no_scancode:
        print("\n=== ScanCode on the same queries, identical denominators ===")
        sc_verdicts = scancode_verdicts([q["text"] for q in answerable])
        sc = score_scancode(sc_verdicts, answerable)
        print(f"  answered {sc['answered']}/{sc['n']} ({sc['coverage']:.1%})  "
              f"precision {sc['precision_at_answered']}  R@1 {sc['recall_at_1']}")
        print(f"  agent    answered {native['answered']}/{native['n']} "
              f"({native['coverage']:.1%})  precision {native['precision_at_answered']}  "
              f"R@1 {native['recall_at_1']}")

    if args.dump_residual:
        residual = []
        for i, (top, q) in enumerate(zip(tops, answerable)):
            if top["shortname"] != "UNKNOWN":
                continue
            sc_expr = sc_verdicts[i][0] if sc_verdicts else None
            sc_answered = bool(sc_keys(sc_expr))
            residual.append({
                "text": q["text"], "gt": sorted(q["gt"]), "tag": q["tag"], "lang": q["lang"],
                "scancode": sc_expr,
                # Both abstained, yet the author declared a license: beyond lexical
                # reach, and the only residual an ML stage can address.
                "residual": "A" if sc_answered else "B",
            })
        args.dump_residual.parent.mkdir(parents=True, exist_ok=True)
        args.dump_residual.write_text(json.dumps(residual, indent=2))
        kinds = Counter(r["residual"] for r in residual)
        print(f"\nresidual: {len(residual)} abstained  A={kinds['A']} (index/matcher gap)  "
              f"B={kinds['B']} (beyond lexical)  -> {args.dump_residual}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "config": {k: str(v) for k, v in vars(args).items()},
        "n_notice": len(notice), "n_scored": len(answerable),
        "n_no_signal": len(nosig), "unreachable_gt": unreachable,
        "agent": native, "no_signal_answered": ns_answered,
        "sweep": rows, "scancode": sc}, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
