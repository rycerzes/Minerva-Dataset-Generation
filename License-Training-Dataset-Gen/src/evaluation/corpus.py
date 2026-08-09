"""Query corpora: real license occurrences drawn from real source files.

Two corpora, with different ground-truth provenance — the distinction decides
which eval you can trust:

* **ScanCode-labelled** (:func:`scan_stack`) — every high-confidence
  ``licensedcode`` match in the-stack-smol becomes a query. Broad coverage, but
  the labels come from the same matcher family as the notice reference layer, so
  evals over it are *optimistic*.
* **SPDX-tagged** (:func:`iter_stack_files` + the ``spdx_tag`` suite) — ground
  truth is the author-declared ``SPDX-License-Identifier:``, independent of any
  matcher. Narrower, and the honest headline.

The snapshot path is resolved from ``$STACK_SNAPSHOT`` or the HuggingFace cache
rather than hardcoded, so this runs off the machine it was written on.

  uv run src/evaluation/corpus.py --per-lang 1500 --workers 16
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_CACHE = ROOT / "cache" / "real_corpus_eval.json"

SCORE_THRESHOLD = 50    # licensedcode match score below this is not ground truth
MIN_MATCH_CHARS = 80    # shorter matched spans are generic boilerplate

_IDX = None


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
        "or fetch it once with `datasets.load_dataset('bigcode/the-stack-smol')`."
    )


def iter_stack_files(per_lang: int = 0, snapshot: Path | None = None,
                     min_chars: int = 100) -> Iterator[tuple[str, str]]:
    """Yield ``(language, content)`` from the snapshot; ``per_lang=0`` reads all.

    Round-robins per language so the query set is not dominated by the
    Apache/MIT-heavy Python subset — license-family diversity comes mostly from
    C/C++/Java/Perl.
    """
    snap = Path(snapshot) if snapshot else find_snapshot()
    for lang_dir in sorted(snap.iterdir()):
        data = lang_dir / "data.json"
        if not data.exists():
            continue
        n = 0
        with data.open() as fh:
            for line in fh:
                if per_lang and n >= per_lang:
                    break
                try:
                    content = json.loads(line).get("content", "") or ""
                except Exception:
                    continue
                if len(content) < min_chars:
                    continue
                n += 1
                yield lang_dir.name, content


def load_real_corpus(path: Path = CORPUS_CACHE) -> list[dict]:
    """Read the cached ScanCode-labelled corpus."""
    return json.loads(Path(path).read_text())


# ---- ScanCode-labelled corpus construction ----------------------------------


def _init_worker():
    global _IDX
    from licensedcode.cache import get_index
    _IDX = get_index()


def _scan(task):
    """(lang, content) -> (lang, row) for the best high-confidence match, else None."""
    lang, content = task
    try:
        matches = [m for m in _IDX.match(query_string=content)
                   if m.score() >= SCORE_THRESHOLD and len(m.matched_text()) >= MIN_MATCH_CHARS]
    except Exception:
        return None
    if not matches:
        return None
    best = max(matches, key=lambda x: x.score())
    return lang, {"text": best.matched_text(), "label": 1,
                  "license": best.rule.license_expression, "score": best.score(),
                  "lang": lang, "source": "the-stack-smol"}


def scan_stack(per_lang: int = 1500, workers: int = 16,
               cache: Path = CORPUS_CACHE, snapshot: Path | None = None) -> list[dict]:
    """Scan the snapshot with ``licensedcode`` and cache the positives.

    Existing negatives in the cache are preserved (the Nirjas gate eval uses
    them); positives are replaced by this scan.
    """
    from collections import Counter
    from multiprocessing import Pool

    corpus = json.loads(Path(cache).read_text()) if Path(cache).exists() else []
    negatives = [r for r in corpus if r.get("label") == 0]
    positives: list[dict] = []

    tasks = list(iter_stack_files(per_lang, snapshot))
    print(f"scanning {len(tasks)} files with {workers} workers…", flush=True)

    per_lang_count: Counter = Counter()
    with Pool(workers, initializer=_init_worker) as pool:
        for done, res in enumerate(pool.imap_unordered(_scan, tasks, chunksize=8), 1):
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)}  positives={len(positives)}", flush=True)
            if res is not None:
                lang, row = res
                positives.append(row)
                per_lang_count[lang] += 1

    Path(cache).write_text(json.dumps(negatives + positives, indent=2))
    licenses = Counter(r["license"] for r in positives)
    print(f"\npositives: {len(positives)}  distinct expressions: {len(licenses)}")
    print("per language:", per_lang_count.most_common())
    print("top expressions:", licenses.most_common(20))
    print(f"wrote {cache} ({len(negatives)} negs + {len(positives)} pos)")
    return positives


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild the ScanCode-labelled query corpus.")
    ap.add_argument("--per-lang", type=int, default=1500, help="files per language; 0 = all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()
    scan_stack(args.per_lang, args.workers, snapshot=args.snapshot)


if __name__ == "__main__":
    _main()
