"""Unit tests for the agent-level eval.

Two things here are easy to get wrong in the direction that flatters the agent,
so both are pinned: how a ScanCode abstention is scored, and how the agent's
shortnames are mapped onto ground-truth keys.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("pandas")

from evaluation.agent import key_map, sc_keys, score, score_scancode  # noqa: E402

REFS = {"apache-2.0": "...", "mit": "...", "gpl-2.0-plus": "...", "bsd-new": "..."}
S2K = {"apache-2.0": "apache-2.0", "mit": "mit", "bsd-3-clause": "bsd-new"}


# --- ScanCode verdicts --------------------------------------------------------

def test_unknown_license_reference_is_an_abstention_not_an_answer():
    """ScanCode's way of saying "licensing here, cannot name it". Scoring it as an
    answer would inflate its coverage and destroy its precision — in our favour."""
    assert sc_keys("unknown-license-reference") == set()


def test_compound_expression_yields_every_component():
    assert sc_keys("apache-2.0 AND mit") == {"apache-2.0", "mit"}


def test_skip_keys_are_dropped_but_real_components_survive():
    assert sc_keys("apache-2.0 AND unknown-license-reference") == {"apache-2.0"}


def test_no_match_is_an_abstention():
    assert sc_keys(None) == set()


def test_scancode_abstentions_are_excluded_from_coverage_and_precision():
    queries = [{"gt": {"apache-2.0"}}, {"gt": {"mit"}}]
    verdicts = [("apache-2.0", 100.0), ("unknown-license-reference", 100.0)]
    out = score_scancode(verdicts, queries)
    assert out["answered"] == 1
    assert out["precision_at_answered"] == 1.0
    assert out["recall_at_1"] == 0.5


# --- shortname -> reference key ----------------------------------------------

def test_maps_spdx_shortnames_through_the_spdx_table():
    assert key_map(["BSD-3-Clause"], S2K, REFS)["BSD-3-Clause"] == "bsd-new"


def test_falls_back_to_the_shortname_as_a_key():
    assert key_map(["gpl-2.0-plus"], {}, REFS)["gpl-2.0-plus"] == "gpl-2.0-plus"


def test_or_later_suffix_resolves_to_the_plus_key():
    assert key_map(["GPL-2.0+"], {}, REFS)["GPL-2.0+"] == "gpl-2.0-plus"


def test_unmappable_shortname_is_omitted_rather_than_guessed():
    assert key_map(["Adaptec"], S2K, REFS) == {}


# --- agent scoring ------------------------------------------------------------

def test_unknown_is_an_abstention_not_a_wrong_answer():
    queries = [{"gt": {"apache-2.0"}}, {"gt": {"mit"}}]
    tops = [{"shortname": "Apache-2.0"}, {"shortname": "UNKNOWN"}]
    out = score(tops, queries, {"Apache-2.0": "apache-2.0"})
    assert out["answered"] == 1
    assert out["precision_at_answered"] == 1.0
    assert out["coverage"] == 0.5


def test_answer_with_no_reference_key_counts_as_a_miss_and_is_reported():
    """An unmappable answer is still an answer — it must cost precision, and be
    visible, rather than being quietly dropped from the denominator."""
    queries = [{"gt": {"apache-2.0"}}]
    out = score([{"shortname": "Adaptec"}], queries, {})
    assert out["answered"] == 1
    assert out["correct"] == 0
    assert out["unmapped_answers"] == 1
    assert out["precision_at_answered"] == 0.0
