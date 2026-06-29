#!/usr/bin/env python3
"""Fix cross-split leakage in the Nirjas (binary) dataset on disk.

Nirjas has no license_key, so leakage is text-level: overlapping sliding-window fragments of
the same license land in multiple splits. Reuses the split-priority dedup from
fix_atarashi_leakage (train kept whole, leaked validation/test rows dropped). No per-label
coverage constraint is needed — both binary classes have ample samples.

  uv run fix_nirjas_leakage.py --near            # MinHash near-dedup (threshold 0.8), recommended
  uv run fix_nirjas_leakage.py                   # exact-text dedup only
  uv run fix_nirjas_leakage.py --near --in-place # overwrite output/nirjas
"""
import argparse
from collections import Counter
from datasets import load_from_disk
from fix_atarashi_leakage import dedup_across_splits, dedup_near_across_splits, norm, save_clean


def regen_stats(ds, dropped, near, threshold):
    names = ds["train"].features["label"].names
    def lbl(v):
        return names[v] if isinstance(v, int) else v
    splits = {}
    for s in ds:
        splits[s] = dict(
            samples=len(ds[s]),
            by_label=dict(Counter(lbl(x) for x in ds[s]["label"])),
            by_source=dict(Counter(ds[s]["source"])),
            by_negative_type=dict(Counter(t for t in ds[s]["negative_type"] if t)),
        )
    all_src = [x for s in ds for x in ds[s]["source"]]
    return {
        "splits": splits,
        "total_samples": sum(len(ds[s]) for s in ds),
        "by_source": dict(Counter(all_src)),
        "cross_split_dedup": {
            "method": f"MinHashLSH char-3gram, threshold {threshold} (matches src/utils.py)" if near
                      else "exact normalized text",
            "policy": "no fragment text appears in >1 split; train kept whole, leaked val/test rows dropped",
            "dropped": dropped,
            "script": "fix_nirjas_leakage.py" + (" --near" if near else ""),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/nirjas")
    ap.add_argument("--out", default="output/nirjas_clean")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--near", action="store_true", help="MinHash near-dedup across splits")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    ds = load_from_disk(args.dataset)
    before = {s: len(ds[s]) for s in ds}
    clean, dropped = (dedup_near_across_splits(ds, args.threshold) if args.near
                      else dedup_across_splits(ds))
    after = {s: len(clean[s]) for s in clean}

    # verify zero exact cross-split overlap
    norms = {s: set(map(norm, clean[s]["text"])) for s in clean}
    ss = list(norms)
    leak = sum(len(norms[ss[i]] & norms[ss[j]]) for i in range(len(ss)) for j in range(i + 1, len(ss)))
    # both classes still present in every split
    for s in clean:
        assert len(set(clean[s]["label"])) == 2, f"{s} lost a class"

    print("before:", before)
    print("dropped:", dropped)
    print("after :", after)
    print("residual cross-split exact overlap:", leak)
    assert leak == 0, "leakage remains"

    import json
    stats = regen_stats(clean, dropped, args.near, args.threshold)
    dest = save_clean(clean, args.dataset, args.out, args.in_place)
    json.dump(stats, open(f"{dest}/statistics.json", "w"), indent=2)
    print("wrote:", dest, "+ statistics.json")


if __name__ == "__main__":
    main()
