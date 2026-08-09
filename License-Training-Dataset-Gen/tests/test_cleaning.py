"""Split-hygiene invariants the exporter depends on.

These are correctness properties, not quality targets: if any fails, the exported
dataset leaks and every downstream evaluation number is inflated.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exporter.cleaning import (  # noqa: E402
    clean, dedup_across_splits, dedup_near_across_splits, drop_short_fragments,
    norm, residual_overlap,
)

datasets = pytest.importorskip("datasets")
DatasetDict, Dataset = datasets.DatasetDict, datasets.Dataset


def _ds(**splits):
    return DatasetDict({
        name: Dataset.from_dict({
            "license_key": [k for k, _ in rows],
            "text": [t for _, t in rows],
            "source": ["scancode"] * len(rows),
        })
        for name, rows in splits.items()
    })


def test_norm_collapses_case_and_whitespace():
    assert norm("  Permission   IS\nhereby ") == "permission is hereby"


def test_exact_dedup_drops_only_the_later_split():
    ds = _ds(train=[("mit", "shared frag"), ("gpl", "gpl only")],
             test=[("mit", "SHARED   frag")])
    clean_ds, dropped = dedup_across_splits(ds)
    assert dropped == {"train": 0, "test": 1}
    assert len(clean_ds["train"]) == 2 and len(clean_ds["test"]) == 0


def test_near_dedup_catches_non_exact_duplicates():
    ds = _ds(train=[("mit", "permission is hereby granted free of charge")],
             test=[("mit", "permission is hereby granted, free of charge.")])
    clean_ds, dropped = dedup_near_across_splits(ds, threshold=0.8)
    assert dropped["test"] == 1
    assert len(clean_ds["train"]) == 1  # train is never dropped from


def test_short_fragments_dropped_but_last_train_row_per_label_survives():
    """The learnability guard: every label must remain present in train."""
    ds = _ds(train=[("mit", "short"), ("gpl", "x" * 200)],
             test=[("mit", "tiny")])
    cleaned, dropped = drop_short_fragments(ds, min_chars=60)
    assert dropped["test"] == 1
    assert dropped["train"] == 0, "sole train row for 'mit' must be protected"
    assert "mit" in set(cleaned["train"]["license_key"])


def test_short_fragments_no_guard_without_label_column():
    ds = DatasetDict({
        "train": Dataset.from_dict({"text": ["short"], "label": ["license_related"]}),
        "test": Dataset.from_dict({"text": ["tiny"], "label": ["license_related"]}),
    })
    cleaned, dropped = drop_short_fragments(ds, min_chars=60, label_column=None)
    assert dropped == {"train": 1, "test": 1}
    assert len(cleaned["train"]) == 0


def test_clean_leaves_no_residual_overlap_and_loses_no_label():
    ds = _ds(
        train=[("mit", "permission is hereby granted free of charge " * 3),
               ("gpl", "this program is free software you can redistribute " * 3)],
        validation=[("mit", "permission is hereby granted, free of charge. " * 3)],
        test=[("gpl", "tiny"), ("mit", "permission is hereby granted free of charge " * 3)],
    )
    cleaned, report = clean(ds, min_chars=60, near=True, label_column="license_key")
    assert residual_overlap(cleaned) == 0
    assert report["labels_lost"] == 0
    assert set(cleaned["train"]["license_key"]) == {"mit", "gpl"}


def test_clean_is_idempotent():
    ds = _ds(train=[("mit", "permission is hereby granted free of charge " * 3)],
             test=[("mit", "permission is hereby granted, free of charge. " * 3)])
    once, _ = clean(ds, min_chars=60, near=True, label_column="license_key")
    twice, report = clean(once, min_chars=60, near=True, label_column="license_key")
    assert report["before"] == report["after"], "cleaning a clean dataset must be a no-op"
