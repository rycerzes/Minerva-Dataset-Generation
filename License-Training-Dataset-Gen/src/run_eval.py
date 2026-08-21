#!/usr/bin/env python3
"""Entry point for the evaluation suites — the counterpart to ``src/main.py``.

``main.py`` builds datasets; this measures things. They are deliberately separate:
evaluation reads build outputs plus a real-code corpus, runs on a different
cadence, and needs heavier optional dependencies (scancode-toolkit,
sentence-transformers) that a dataset regeneration should not have to install.

    uv run src/run_eval.py crossref --query-mode full
    uv run src/run_eval.py spdx-tag --per-lang 0 --max-per-license 150
    uv run src/run_eval.py cascade --tau 0.30
    uv run src/run_eval.py agent --per-lang 0 --max-per-license 150
    uv run src/run_eval.py real-corpus --refs both --loo contain
    uv run src/run_eval.py --list

Which number to quote: ``crossref`` and ``spdx-tag`` are non-circular and are the
honest headlines. ``real-corpus`` shares label provenance with its own reference
layer and reads optimistically. Prefer macro over micro everywhere — micro tracks
the license mix, since apache-2.0 alone is ~64% of real-world occurrences.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SUITES = {
    "crossref": ("evaluation.crossref", "cross-rendering eval (non-circular, verbatim regime)"),
    "spdx-tag": ("evaluation.spdx_tag", "author-declared SPDX labels, regime-split (non-circular)"),
    "cascade": ("evaluation.cascade", "precision@coverage with abstention"),
    "agent": ("evaluation.agent", "score Cascade.scan itself, vs ScanCode (non-circular)"),
    "real-corpus": ("evaluation.real_corpus", "broad ScanCode-labelled corpus (optimistic)"),
    "nirjas-gate": ("evaluation.nirjas_gate", "binary gate against real source files"),
    "nomos": ("evaluation.nomos", "binary gate against FOSSology nomos testdata"),
    "build-corpus": ("evaluation.corpus", "(re)build the ScanCode-labelled query corpus"),
    "notice-refs": ("evaluation.references", "summarize the notice reference layer"),
    "debian-dep5": ("evaluation.debian", "Debian DEP-5: independent labels and the licence tail"),
    "rank-model": ("evaluation.rank_model", "learned candidate scoring vs the hand-tuned rank key"),
}


def _usage() -> None:
    print(__doc__)
    print("suites:")
    width = max(len(name) for name in SUITES)
    for name, (_, desc) in SUITES.items():
        print(f"  {name:<{width}}  {desc}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "--list"):
        _usage()
        return
    suite = sys.argv[1]
    if suite not in SUITES:
        print(f"unknown suite: {suite!r}\n")
        _usage()
        raise SystemExit(2)

    module_name, _ = SUITES[suite]
    module = __import__(module_name, fromlist=["main"])
    entry = getattr(module, "main", None) or getattr(module, "_main")

    # Suites expose main(argv); a few older ones parse sys.argv directly. Decide by
    # signature rather than by catching TypeError, which would swallow a genuine
    # TypeError raised inside the suite and then run it a second time.
    import inspect
    argv = sys.argv[2:]
    if inspect.signature(entry).parameters:
        entry(argv)
    else:
        sys.argv = [suite, *argv]
        entry()


if __name__ == "__main__":
    main()
