#!/usr/bin/env python3
"""Split-hygiene for exported datasets: cross-split dedup and junk quarantine.

This used to live in two root-level scripts (``fix_atarashi_leakage.py``,
``fix_nirjas_leakage.py``) run by hand *after* an export. That made the documented
pipeline emit a dataset that was not the one actually shipped: ``output/atarashi``
carried cross-split leakage, and only an undocumented post-pass produced a usable
copy alongside it. The export is now cleaned in-process, so what ``main.py``
writes is what ships, and ``output/atarashi`` is the canonical build.

Two defects are removed, in this order:

1. **Junk quarantine** — very short fragments are generic legal boilerplate
   ("without express or implied warranty of any kind.") sliced out of a longer
   license and labelled with that one license, though the phrase recurs across
   many. The ground truth is not inferable from the text, so they are noise for
   training and actively misleading in evaluation.
2. **Cross-split leakage** — the sliding-window splitter emits overlapping
   fragments of one license, which the content-hash split scatters across
   train/validation/test. Identical (or near-identical) text on both sides of the
   split makes evaluation measure memorisation.

Both passes keep train whole and drop only from later splits, so no label is lost
from train — the exporter's learnability invariant (every validation/test label is
seen in train) is preserved.

Also usable standalone, to clean an export produced before this was wired in:

    uv run src/exporter/cleaning.py --dataset output/atarashi --out output/atarashi.cleaned \
        --near --min-chars 60
"""
from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS = 60
DEFAULT_NEAR_THRESHOLD = 0.8
SPLIT_PRIORITY = ("train", "validation", "test")


def norm(text: str) -> str:
    """Whitespace- and case-normalized form used for all identity comparisons."""
    return " ".join(text.lower().split())


def _minhash(text: str, num_perm: int = 128, n: int = 3):
    """Character n-gram MinHash — same shingling as ``src.utils.dedup_near_duplicates``."""
    from datasketch import MinHash  # type: ignore[import-untyped]

    s = re.sub(r"\s+", " ", text.lower().strip())
    m = MinHash(num_perm=num_perm)
    for j in range(max(1, len(s) - n + 1)):
        m.update(s[j : j + n].encode("utf-8"))
    return m


def drop_short_fragments(ds, min_chars: int = DEFAULT_MIN_CHARS,
                         label_column: str | None = "license_key"):
    """Drop fragments shorter than ``min_chars`` normalized characters.

    ``label_column`` enables the learnability guard: a license's *last* remaining
    train row is never dropped, so every label stays present in train. Pass None
    for datasets without a per-class label (e.g. the binary Nirjas set), where the
    guard is unnecessary — both classes have ample samples.
    """
    from datasets import DatasetDict

    if not min_chars:
        return ds, {s: 0 for s in ds}

    guard = bool(label_column) and "train" in ds and label_column in ds["train"].column_names
    counts = Counter(ds["train"][label_column]) if guard else Counter()

    out, dropped = {}, {}
    for split in ds:
        keep, n_dropped = [], 0
        labels = ds[split][label_column] if guard else [None] * len(ds[split])
        for i, (text, label) in enumerate(zip(ds[split]["text"], labels)):
            if len(norm(text)) >= min_chars:
                keep.append(i)
                continue
            if split == "train" and guard:
                if counts[label] <= 1:
                    keep.append(i)  # last train row for this label — keep it
                    continue
                counts[label] -= 1
            n_dropped += 1
        out[split] = ds[split].select(keep)
        dropped[split] = n_dropped
    return DatasetDict(out), dropped


def dedup_across_splits(ds, priority: tuple[str, ...] = SPLIT_PRIORITY):
    """Exact cross-split dedup. A text may appear in only one split.

    Train is kept whole (preserving label coverage); each later split drops rows
    whose text already appeared in an earlier one. Intra-split duplicates are left
    alone — those are redundant, not leakage.
    """
    from datasets import DatasetDict

    splits = [s for s in priority if s in ds]
    seen: set[str] = set()
    out, dropped = {}, {}
    for pos, split in enumerate(splits):
        keep, n_dropped = [], 0
        for i, text in enumerate(ds[split]["text"]):
            if pos > 0 and norm(text) in seen:
                n_dropped += 1
            else:
                keep.append(i)
        out[split] = ds[split].select(keep)
        dropped[split] = n_dropped
        seen.update(norm(t) for t in out[split]["text"])
    return DatasetDict(out), dropped


def dedup_near_across_splits(ds, threshold: float = DEFAULT_NEAR_THRESHOLD,
                             priority: tuple[str, ...] = SPLIT_PRIORITY):
    """MinHash LSH cross-split near-dedup. Subsumes :func:`dedup_across_splits`.

    A later split's row is dropped when it is a >= ``threshold`` Jaccard
    near-duplicate of any train row or any earlier kept row. Train is inserted
    whole and never dropped.
    """
    from datasets import DatasetDict
    from datasketch import MinHashLSH  # type: ignore[import-untyped]

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    splits = [s for s in priority if s in ds]
    out, dropped = {}, {}
    for pos, split in enumerate(splits):
        keep, n_dropped = [], 0
        for i, text in enumerate(ds[split]["text"]):
            m = _minhash(text)
            if pos > 0 and lsh.query(m):
                n_dropped += 1
            else:
                lsh.insert(f"{split}-{i}", m)
                keep.append(i)
        out[split] = ds[split].select(keep)
        dropped[split] = n_dropped
    return DatasetDict(out), dropped


def residual_overlap(ds) -> int:
    """Count texts appearing in more than one split — must be 0 after cleaning."""
    norms = {s: set(map(norm, ds[s]["text"])) for s in ds}
    splits = list(norms)
    return sum(len(norms[a] & norms[b])
               for i, a in enumerate(splits) for b in splits[i + 1:])


def clean(ds, min_chars: int = DEFAULT_MIN_CHARS, near: bool = True,
          threshold: float = DEFAULT_NEAR_THRESHOLD,
          label_column: str | None = "license_key"):
    """Quarantine junk then remove cross-split leakage. Returns (dataset, report).

    Raises AssertionError if the result still leaks or if a label was lost from
    train — these are invariants the exporter depends on, not soft goals.
    """
    before = {s: len(ds[s]) for s in ds}
    train_labels = (set(ds["train"][label_column])
                    if label_column and "train" in ds
                    and label_column in ds["train"].column_names else set())

    ds, short_dropped = drop_short_fragments(ds, min_chars, label_column)
    ds, leak_dropped = (dedup_near_across_splits(ds, threshold) if near
                        else dedup_across_splits(ds))

    leak = residual_overlap(ds)
    lost = train_labels - set(ds["train"][label_column]) if train_labels else set()
    assert leak == 0, f"cross-split leakage remains: {leak}"
    assert not lost, f"{len(lost)} labels lost from train"

    report = {"before": before, "after": {s: len(ds[s]) for s in ds},
              "quarantined": short_dropped, "deduped": leak_dropped,
              "residual_overlap": leak, "labels_lost": len(lost)}
    logger.info("cleaning: %s", report)
    return ds, report


def _main() -> None:
    import argparse
    import json
    import os
    import shutil

    from datasets import load_from_disk

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="output/atarashi")
    ap.add_argument("--out", default="output/atarashi.cleaned")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--near", action="store_true", help="MinHash near-dedup (subsumes exact)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_NEAR_THRESHOLD)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help="quarantine fragments shorter than this; 0 disables")
    ap.add_argument("--label-column", default="license_key",
                    help="per-class column for the learnability guard; '' to disable")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset)
    cleaned, report = clean(ds, args.min_chars, args.near, args.threshold,
                            args.label_column or None)
    print(json.dumps(report, indent=2))

    if not args.in_place:
        cleaned.save_to_disk(args.out)
        print("wrote:", args.out)
        return
    # HF cannot overwrite a dataset dir in place — write beside it, then swap.
    tmp = args.dataset.rstrip("/") + ".tmp_clean"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    cleaned.save_to_disk(tmp)
    shutil.rmtree(args.dataset)
    os.rename(tmp, args.dataset)
    print("wrote:", args.dataset)


if __name__ == "__main__":
    _main()
