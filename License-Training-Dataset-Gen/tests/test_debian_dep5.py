"""Unit tests for the Debian DEP-5 label mapping.

DEP-5 short names predate SPDX and are written by hand, so the mapping from them to
license shortnames is where this corpus is won or lost. Raw name counts overstate
diversity badly — `Expat` and `MIT` are the same license, as are `Apache-2.0`,
`APACHE-2.0`, `Apache 2.0` and `Apache2`.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("atarashi")

from evaluation.debian import canonical_license, dep5_licenses  # noqa: E402

INDEX = {s.lower(): s for s in
         ["MIT", "Apache-2.0", "GPL-2.0", "GPL-3.0", "BSD-3-Clause", "Artistic-1.0",
          "Zlib", "LGPL-2.1", "CC0-1.0"]}


def _canon(name):
    return canonical_license(name, INDEX)


# --- name mapping -------------------------------------------------------------

def test_debian_calls_mit_expat():
    assert _canon("Expat") == "MIT"


def test_or_later_marker_maps_to_the_base_license():
    """Debian writes `GPL-2+`; the shortname list has no `+` form."""
    assert _canon("GPL-2+") == "GPL-2.0"


def test_bare_version_gains_a_minor():
    """Debian writes GPL-2 and Apache-2 where SPDX writes GPL-2.0 and Apache-2.0."""
    assert _canon("GPL-2") == "GPL-2.0"
    assert _canon("Apache-2") == "Apache-2.0"


def test_clause_casing_is_normalized():
    assert _canon("BSD-3-clause") == "BSD-3-Clause"


def test_exception_suffix_falls_back_to_the_base_license():
    assert _canon("GPL-2+ with Autoconf exception") == "GPL-2.0"


def test_case_insensitive():
    assert _canon("APACHE-2.0") == "Apache-2.0"


def test_names_identifying_no_license_are_distinguished_from_failures():
    """A non-identifying name and an unresolvable one are different facts: merging
    them would hide the error rate that decides whether this corpus is usable."""
    assert _canon("public-domain") is None
    assert _canon("Ruby's") is False


def test_unresolvable_name_is_not_guessed():
    assert _canon("BSD-2-cluase") is False    # a real typo in the archive


# --- DEP-5 parsing ------------------------------------------------------------

DEP5 = """Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: example

Files: *
Copyright: 2020 Example
License: Expat

Files: src/vendor/*
Copyright: 2019 Other
License: Apache-2.0 or GPL-2+

License: Expat
 Permission is hereby granted...
"""


def test_parses_license_names_including_compound_fields():
    assert dep5_licenses(DEP5) == {"Expat", "Apache-2.0", "GPL-2+"}


def test_non_dep5_copyright_is_reported_as_unparseable_not_empty():
    """About a fifth of packages ship prose. Returning an empty set would count them
    as packages that declare nothing, rather than ones we cannot read."""
    assert dep5_licenses("This package is free software.\nSee /usr/share/...") is None
    assert dep5_licenses(None) is None
