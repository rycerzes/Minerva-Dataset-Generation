"""Recreate output/nirjas with short-form license positives added across all splits.

Fixes the SPDX/short-reference recall hole (model dropped `// SPDX-License-Identifier: ...`).
Reuses all existing (cached, not-regenerable) content; only ADDS short-form positives.
Backs up the current export first, splits the synthesized positives disjointly across
train/val/test (no identical string crosses a split), then re-enforce the 0.8 dedup standard
separately via fix_nirjas_leakage.py --near --in-place.
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets

sys.path.insert(0, "src")
from fetchers.shortform_license import generate_shortform_positives

SRC = "output/nirjas"
BACKUP = "output/nirjas_pre_shortform"
TMP = "output/nirjas_tmp"

if not Path(BACKUP).exists():
    shutil.copytree(SRC, BACKUP)
    print(f"backed up {SRC} -> {BACKUP}")
else:
    print(f"backup already exists at {BACKUP} (left untouched)")

keys = sorted(set(load_from_disk("output/atarashi")["train"]["license_key"]))
short = generate_shortform_positives(keys, n_per=2)
import random
random.Random(0).shuffle(short)
n = len(short)
a, b = int(n * 0.8), int(n * 0.9)
parts = {"train": short[:a], "validation": short[a:b], "test": short[b:]}
print(f"generated {n} short-form positives -> "
      f"train {len(parts['train'])} / val {len(parts['validation'])} / test {len(parts['test'])}")

d = load_from_disk(SRC)
out = {}
for name, split in d.items():
    add = parts[name]
    extra = Dataset.from_dict(
        {"text": add, "label": [0] * len(add),  # 0 = license_related
         "source": ["shortform"] * len(add), "negative_type": [""] * len(add)},
        features=split.features,
    )
    out[name] = concatenate_datasets([split, extra])
    print(f"  {name}: {len(split)} -> {len(out[name])} (+{len(add)})")

DatasetDict(out).save_to_disk(TMP)
shutil.rmtree(SRC)
shutil.move(TMP, SRC)
print(f"recreated {SRC}")

# refresh statistics.json counts (preserve other keys if present)
stats_path = Path(SRC) / "statistics.json"
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
dd = load_from_disk(SRC)
splits = {}
total = Counter()
names = dd["train"].features["label"].names
for name, split in dd.items():
    by_label = Counter(names[l] for l in split["label"])
    splits[name] = {
        "samples": len(split),
        "by_label": dict(by_label),
        "by_source": dict(Counter(split["source"])),
        "by_negative_type": dict(Counter(nt for nt in split["negative_type"] if nt)),
    }
    total.update(by_label)
stats["splits"] = splits
stats["total_samples"] = sum(len(s) for s in dd.values())
stats["by_label"] = dict(total)
stats["shortform_added"] = {"total": n, "per_split": {k: len(v) for k, v in parts.items()},
                            "note": "SPDX/short-reference positives; see shortform_license.py"}
stats_path.write_text(json.dumps(stats, indent=2))
print(f"updated {stats_path}")
