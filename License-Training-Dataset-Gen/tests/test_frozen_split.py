"""The content-hash split must be frozen.

A sample's split is a pure function of its text. This is what makes benchmark
deltas across pipeline re-runs real rather than resampling noise, and it is what
stops an identical (or upsampled) text from landing on both sides of the split.
Random re-splitting silently destroys both properties, so the invariant is
asserted rather than assumed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exporter.dataset_export import _split_of  # noqa: E402

VAL, TEST = 0.1, 0.1


def test_split_is_deterministic():
    for text in ("permission is hereby granted", "GPL v2", "", "x" * 5000):
        assert _split_of(text, VAL, TEST) == _split_of(text, VAL, TEST)


def test_split_depends_only_on_text_not_on_pool():
    """Same text, different surrounding data -> same split."""
    text = "this program is free software"
    assert _split_of(text, VAL, TEST) == _split_of(text, VAL, TEST)


def test_identical_texts_cannot_straddle_a_split():
    duplicated = "licensed under the apache license version 2.0"
    assert len({_split_of(duplicated, VAL, TEST) for _ in range(10)}) == 1


def test_ratios_are_approximately_honoured():
    """The hash is uniform, so a large pool splits near the configured ratios."""
    from collections import Counter

    counts = Counter(_split_of(f"license fragment number {i}", VAL, TEST)
                     for i in range(20_000))
    total = sum(counts.values())
    assert abs(counts["test"] / total - TEST) < 0.02
    assert abs(counts["validation"] / total - VAL) < 0.02
    assert abs(counts["train"] / total - (1 - VAL - TEST)) < 0.03


def test_all_splits_reachable():
    seen = {_split_of(f"frag {i}", VAL, TEST) for i in range(2000)}
    assert seen == {"train", "validation", "test"}
