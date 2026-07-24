#!/usr/bin/env python3
"""Fix #1 (exact cross-split leakage) in the Atarashi dataset on disk.

The license-aware split kept every label in train but lets near-identical sliding-window
fragments of the same license land in multiple splits. This removes exact (normalized) text
overlap by priority train > validation > test: a fragment text is allowed in exactly one split,
kept in the earliest-priority split it appears in, dropped from the others.

Singletons already live train-only (license-aware split), so this never empties a label from train.
Writes a cleaned copy; does not touch the original.

  uv run fix_atarashi_leakage.py            # -> output/atarashi_clean
  uv run fix_atarashi_leakage.py --in-place # overwrite output/atarashi
"""
import argparse
import re
from datasets import load_from_disk, DatasetDict


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def _minhash(text, num_perm=128, n=3):
    """Same normalization + char-n-gram shingling as src.utils.dedup_near_duplicates."""
    from datasketch import MinHash
    s = re.sub(r"\s+", " ", text.lower().strip())
    m = MinHash(num_perm=num_perm)
    for j in range(max(1, len(s) - n + 1)):
        m.update(s[j : j + n].encode("utf-8"))
    return m


def dedup_near_across_splits(ds, threshold=0.8, priority=("train", "validation", "test")):
    """Cross-split MinHashLSH near-dedup, train-priority. Subsumes exact dedup: a later split's
    row is dropped if it is a >=threshold Jaccard near-duplicate of any train- or earlier-kept row.
    Train is inserted whole and never dropped (preserves label coverage)."""
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    splits = [s for s in priority if s in ds]
    out, dropped = {}, {}
    for pos, split in enumerate(splits):
        keep_idx, d = [], 0
        for i, t in enumerate(ds[split]["text"]):
            m = _minhash(t)
            if pos > 0 and lsh.query(m):       # near-dup of an earlier split -> drop
                d += 1
            else:
                lsh.insert(f"{split}-{i}", m)   # only train + kept rows enter the index
                keep_idx.append(i)
        out[split] = ds[split].select(keep_idx)
        dropped[split] = d
    return DatasetDict(out), dropped


def dedup_across_splits(ds, priority=("train", "validation", "test")):
    """No fragment text may appear in more than one split. Train is kept whole (preserves label
    coverage); each later split drops only rows whose text already appears in an earlier split.
    Intra-split duplicates are left alone — they are redundant, not leakage."""
    splits = [s for s in priority if s in ds]
    seen = set()
    out, dropped = {}, {}
    for pos, split in enumerate(splits):
        keep_idx, d = [], 0
        for i, t in enumerate(ds[split]["text"]):
            n = norm(t)
            if pos > 0 and n in seen:   # leaks from an earlier split -> drop
                d += 1
            else:
                keep_idx.append(i)
        out[split] = ds[split].select(keep_idx)
        dropped[split] = d
        seen.update(norm(t) for t in out[split]["text"])  # only this split's *kept* texts
    return DatasetDict(out), dropped


def drop_short_unattributable(ds, min_chars, protect_train=True):
    """Quarantine very-short, low-signal fragments (< min_chars after normalization).

    These are generic legal boilerplate windows ("without express or implied
    warranty of any kind.") sliced from a longer license and labeled with that one
    license, even though the phrase appears across many licenses — the ground truth
    is not inferable from the text, so they are noise for both training and eval.

    Learnability guard: never drop a license's *last* remaining train row, so every
    label stays present in train (matches the exporter's invariant).
    """
    from collections import Counter
    if not min_chars:
        return ds, {s: 0 for s in ds}
    train_counts = Counter(ds["train"]["license_key"]) if protect_train and "train" in ds else Counter()
    out, dropped = {}, {}
    for split in ds:
        keep_idx, d = [], 0
        for i, (t, lk) in enumerate(zip(ds[split]["text"], ds[split]["license_key"])):
            if len(norm(t)) >= min_chars:
                keep_idx.append(i)
                continue
            if split == "train" and protect_train and train_counts[lk] <= 1:
                keep_idx.append(i)          # last train row for this label -> keep
                continue
            if split == "train":
                train_counts[lk] -= 1
            d += 1
        out[split] = ds[split].select(keep_idx)
        dropped[split] = d
    return DatasetDict(out), dropped


def save_clean(clean, dataset_path, out_path, in_place):
    """HF can't overwrite a dataset in place, so write to a temp dir then swap."""
    import os, shutil
    if not in_place:
        clean.save_to_disk(out_path)
        return out_path
    tmp = dataset_path.rstrip("/") + ".tmp_clean"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    clean.save_to_disk(tmp)
    shutil.rmtree(dataset_path)
    os.rename(tmp, dataset_path)
    return dataset_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/atarashi")
    ap.add_argument("--out", default="output/atarashi_clean")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--near", action="store_true", help="MinHash near-dedup across splits (subsumes exact)")
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard threshold for --near")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="Quarantine fragments shorter than this (normalized chars); 0 = off. Try 60.")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset)
    before = {s: len(ds[s]) for s in ds}
    ds, short_dropped = drop_short_unattributable(ds, args.min_chars)
    clean, dropped = (dedup_near_across_splits(ds, args.threshold) if args.near
                      else dedup_across_splits(ds))
    after = {s: len(clean[s]) for s in clean}

    # verify zero exact overlap remains across splits
    norms = {s: set(map(norm, clean[s]["text"])) for s in clean}
    leak = 0
    splits = list(norms)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            leak += len(norms[splits[i]] & norms[splits[j]])

    # verify every label still present in train
    train_labels = set(clean["train"]["license_key"])
    lost = set(ds["train"]["license_key"]) - train_labels

    print("before:", before)
    print("short-fragment quarantine dropped:", short_dropped)
    print("cross-split dedup dropped:", dropped)
    print("after :", after)
    print("residual cross-split exact overlap:", leak)
    print("labels lost from train:", len(lost))
    assert leak == 0, "leakage remains"
    assert not lost, f"dropped {len(lost)} labels from train"

    dest = save_clean(clean, args.dataset, args.out, args.in_place)
    print("wrote:", dest)


def _selfcheck():
    from datasets import Dataset
    ds = DatasetDict({
        "train": Dataset.from_dict({"license_key": ["mit", "gpl"], "text": ["shared frag", "gpl only"], "source": ["scancode"] * 2}),
        "test": Dataset.from_dict({"license_key": ["mit"], "text": ["SHARED   frag"], "source": ["scancode"]}),  # dup of train
    })
    clean, dropped = dedup_across_splits(ds)
    assert dropped == {"train": 0, "test": 1}, dropped
    assert len(clean["test"]) == 0 and len(clean["train"]) == 2

    # near-dedup must also catch a near (not exact) cross-split dup, train kept whole
    ds2 = DatasetDict({
        "train": Dataset.from_dict({"license_key": ["mit"], "text": ["permission is hereby granted free of charge"], "source": ["scancode"]}),
        "test": Dataset.from_dict({"license_key": ["mit"], "text": ["permission is hereby granted, free of charge."], "source": ["scancode"]}),
    })
    nclean, ndrop = dedup_near_across_splits(ds2, 0.8)
    assert ndrop["test"] == 1 and len(nclean["train"]) == 1, (ndrop, len(nclean["train"]))
    print("selfcheck OK exact=", dropped, "near=", ndrop)


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()
