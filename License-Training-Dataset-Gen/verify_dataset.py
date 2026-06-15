"""Verify exported Atarashi and Nirjas Hugging Face datasets.

Defaults to the current full export under ``output`` and supports
train/validation/test splits.  Usage:

    uv run python verify_dataset.py
    uv run python verify_dataset.py --output-dir output
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


def _split_sizes(ds) -> dict[str, int]:
    return {split: len(ds[split]) for split in ds}


def _label_name(ds, value: int | str) -> str:
    if not isinstance(value, int):
        return value
    return ds.features["label"].names[value]


def verify_atarashi(path: Path) -> None:
    atarashi = load_from_disk(str(path))
    print("=== ATARASHI DATASET ===")
    print(f"Path: {path}")
    print(f"Splits: {_split_sizes(atarashi)}")
    print(f"Columns: {atarashi['train'].column_names}")

    train_keys = set(atarashi["train"]["license_key"])
    for split in atarashi:
        if split == "train":
            continue
        split_keys = set(atarashi[split]["license_key"])
        print(f"{split}-only labels: {len(split_keys - train_keys):,}")

    rows = [row for split in atarashi for row in atarashi[split]]
    source_counts = Counter(row["source"] for row in rows)
    class_counts = Counter(row["license_key"] for row in rows)
    synthetic = [row for row in rows if row["source"] == "synthetic"]
    artifact_re = re.compile(
        r"(?im)^\s*(?:[-*_`~\s]*)?variant\b(?:\s*[-#: ]*\s*\d*)?\s*:?(?:\s*[-*_`~]*)?\s*$"
        r"|^\s*here (?:are|is) .*variants?"
    )
    synthetic_artifacts = [
        row
        for row in synthetic
        if "```" in row["text"]
        or "**" in row["text"]
        or artifact_re.search(row["text"][:500])
        or len(row["text"]) < 100
    ]

    print(f"Total samples: {len(rows):,}")
    print(f"Unique licenses: {len(class_counts):,}")
    print(f"Sources: {dict(source_counts)}")
    print(f"Synthetic samples: {len(synthetic):,}")
    print(f"Synthetic artifact issues: {len(synthetic_artifacts):,}")
    print(f"Classes with 1 sample: {sum(1 for v in class_counts.values() if v == 1):,}")
    print(f"Classes <=3 samples: {sum(1 for v in class_counts.values() if v <= 3):,}")
    print(f"Classes <=5 samples: {sum(1 for v in class_counts.values() if v <= 5):,}")

    print("\nSample row (train[0]):")
    row = atarashi["train"][0]
    print(f"  license_key: {row['license_key']}")
    print(f"  text: {row['text'][:120]}...")
    print(f"  source: {row['source']}")
    print()


def verify_nirjas(path: Path) -> None:
    nirjas = load_from_disk(str(path))
    print("=== NIRJAS DATASET ===")
    print(f"Path: {path}")
    print(f"Splits: {_split_sizes(nirjas)}")
    print(f"Columns: {nirjas['train'].column_names}")
    print(f"Label names: {nirjas['train'].features['label'].names}")

    for split in nirjas:
        labels = [_label_name(nirjas[split], label) for label in nirjas[split]["label"]]
        print(f"Label distribution ({split}): {dict(Counter(labels))}")
    print()

    wanted = {"license_related": False, "not_license_related": False}
    for row in nirjas["train"]:
        label = _label_name(nirjas["train"], row["label"])
        if label in wanted and not wanted[label]:
            print(f"Sample {label}:")
            print(f"  text: {row['text'][:150]}...")
            print(f"  source: {row['source']}")
            if row.get("negative_type"):
                print(f"  negative_type: {row['negative_type']}")
            print()
            wanted[label] = True
        if all(wanted.values()):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exported Minerva datasets")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root export directory containing atarashi/ and nirjas/ (default: output)",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    verify_atarashi(root / "atarashi")
    verify_nirjas(root / "nirjas")


if __name__ == "__main__":
    main()
