"""Unit tests for the evaluation machinery.

The regime classifier gets the most attention: it decides which queries are
answerable at all, and ~72% of real SPDX-tagged files fall on the no-signal side.
If it drifts, the headline metric moves for reasons unrelated to matching quality.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("sklearn")

from evaluation.crossref import canon, tier_of  # noqa: E402
from evaluation.references import gt_components  # noqa: E402
from evaluation.retrieval import build_loo_mask, encode_lexical, metrics, norm, topk_keys  # noqa: E402
from evaluation.spdx_tag import TAGLINE, classify_regime, gt_keys, stratify  # noqa: E402


# ---- regime classification --------------------------------------------------

def test_copyright_plus_code_is_no_signal():
    """A copyright line with no grant is the case where UNKNOWN is correct."""
    text = "// Copyright (c) 2012-2013, ARM Limited. All rights reserved. #include <ArmLib.h>"
    assert classify_regime(text) == "no-signal"


def test_bare_copyright_never_counts_as_prose():
    assert classify_regime("Copyright 2019 The Authors. All rights reserved.") == "no-signal"


def test_apache_notice_is_notice_regime():
    text = ("Licensed under the Apache License, Version 2.0; you may not use this "
            "file except in compliance with the License.")
    assert classify_regime(text) == "notice"


def test_single_marker_is_insufficient():
    """One incidental 'license' token (a URL, a variable) must not qualify."""
    assert classify_regime("see https://example.com/license for details") == "no-signal"


def test_tagline_regex_strips_only_the_tag_line():
    src = "/*\n * SPDX-License-Identifier: MIT\n * Licensed under the MIT License.\n */"
    stripped = TAGLINE.sub(" ", src)
    assert "SPDX-License-Identifier" not in stripped
    assert "Licensed under the MIT License" in stripped


# ---- ground-truth mapping ---------------------------------------------------

def test_gt_components_splits_compound_expressions():
    assert gt_components("apache-2.0 AND mit") == {"apache-2.0", "mit"}
    assert gt_components("gpl-2.0 WITH classpath-exception-2.0") == {
        "gpl-2.0", "classpath-exception-2.0"}


def test_gt_components_drops_unidentifiable_keys():
    assert gt_components("unknown") == set()
    assert gt_components("mit OR proprietary-license") == {"mit"}


def test_gt_keys_ignores_licenseref_and_unknown_ids():
    refs = {"mit": "x"}
    assert gt_keys("LicenseRef-Custom", {}, refs) == set()
    assert gt_keys("MIT", {"mit": "mit"}, refs) == {"mit"}
    assert gt_keys("NoSuchLicense", {}, refs) == set()


# ---- stratification ---------------------------------------------------------

def test_stratify_caps_each_license():
    queries = [{"gt": {"apache-2.0"}} for _ in range(10)] + [{"gt": {"mit"}} for _ in range(3)]
    kept = stratify(queries, max_per_license=4)
    assert sum(1 for q in kept if q["gt"] == {"apache-2.0"}) == 4
    assert sum(1 for q in kept if q["gt"] == {"mit"}) == 3


def test_stratify_zero_means_uncapped():
    queries = [{"gt": {"apache-2.0"}} for _ in range(10)]
    assert len(stratify(queries, 0)) == 10


# ---- crossref tiers ---------------------------------------------------------

def test_tier_identical_ignores_punctuation_and_case():
    assert tier_of("Permission is hereby granted.", "permission is hereby granted") == "identical"


def test_tier_substring_and_divergent():
    assert tier_of("permission is hereby granted", "permission is hereby granted freely") == "substring"
    assert tier_of("permission is hereby granted", "this program is free software") == "divergent"


def test_canon_strips_punctuation():
    assert canon("Hello, World!") == "hello world"


# ---- metrics ----------------------------------------------------------------

def test_metrics_counts_hits_at_rank_one_and_five():
    topk = np.array([["mit", "gpl", "a", "b", "c"], ["a", "b", "c", "d", "mit"]])
    m = metrics(topk, [{"mit"}, {"mit"}])
    assert m["recall_at_1"] == 0.5
    assert m["recall_at_5"] == 1.0
    assert m["mrr"] == pytest.approx((1.0 + 0.2) / 2)


def test_macro_averages_per_license_not_per_query():
    """Two apache queries hit, one mit query misses -> macro 0.5, micro 0.67."""
    topk = np.array([["apache-2.0"], ["apache-2.0"], ["gpl"]])
    m = metrics(topk, [{"apache-2.0"}, {"apache-2.0"}, {"mit"}])
    assert m["recall_at_1"] == pytest.approx(2 / 3)
    assert m["macro_recall_at_1"] == pytest.approx(0.5)


# ---- leave-one-out ----------------------------------------------------------

def test_loo_exact_masks_only_identical_units():
    mask = build_loo_mask("exact", ["the query text", "other"], ["The  Query   Text"])
    assert mask == [{0}]


def test_loo_contain_masks_contained_units():
    mask = build_loo_mask("contain", ["query", "unrelated"], ["a query here"])
    assert 0 in mask[0] and 1 not in mask[0]


def test_loo_off_returns_none():
    assert build_loo_mask("off", ["a"], ["b"]) is None


def test_masked_unit_is_never_returned_as_top1():
    refs = ["permission is hereby granted free of charge",
            "this program is free software you can redistribute it"]
    Q, R = encode_lexical(refs, ["permission is hereby granted free of charge"])
    assert topk_keys(Q, R, ["mit", "gpl"], k=1)[0][0] == "mit"
    masked = topk_keys(Q, R, ["mit", "gpl"], k=1, mask=[{0}])
    assert masked[0][0] == "gpl", "masked unit must be excluded from the ranking"


def test_norm_matches_cleaning_norm():
    """Eval and exporter must agree on text identity, else 'leak-free' means two things."""
    from exporter.cleaning import norm as export_norm
    for text in ("  A  B ", "Permission\nIS granted", ""):
        assert norm(text) == export_norm(text)
