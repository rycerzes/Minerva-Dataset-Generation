"""Reference index construction and license-key mapping.

Everything an eval matches *against*, plus the mapping from a license expression
or SPDX id to a reference key. This is index-building, not evaluation — the evals
import it, never the reverse.

Two reference layers, and the difference between them is the central finding of
the Atarashi eval work:

* **full texts** — one canonical body per license. Real-world queries are short
  *notices*, and a notice sits at ~95% depth of an 11k-character license body, so
  whole-document scoring drowns it. On real headers this layer scores R@1 ~0.004:
  it cannot identify them at all.
* **notice / short-form** — ScanCode ships ~30k short rule texts (notice,
  reference, tag, intro), each tagged with a license expression: exactly the
  register real files carry. Adding it lifts R@1 by roughly two orders of
  magnitude. ScanCode succeeds in the wild precisely because it indexes these.

  uv run src/evaluation/references.py       # summarize the notice layer
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Repo root, so evals work regardless of the caller's working directory.
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache"
SC_CACHE = CACHE_DIR / "scancode_licenses.json"
FO_CACHE = CACHE_DIR / "fossology_licenses.json"
NOTICE_CACHE = CACHE_DIR / "notice_refs.json"

_MARK = re.compile(r"\{\{.*?\}\}", re.DOTALL)      # ScanCode rule template markers
_COMPOUND = re.compile(r"\s+(?:AND|OR|WITH)\s+", re.IGNORECASE)
# Rule kinds that represent the short-form register we want as references.
_KINDS = ("is_license_notice", "is_license_reference", "is_license_tag", "is_license_intro")

# Ground-truth keys carrying no identifiable reference text — a query labelled
# only with these has nothing to match, so it is excluded rather than counted
# as a miss.
SKIP_KEYS = frozenset({
    "unknown-license-reference", "unknown", "proprietary-license",
    "warranty-disclaimer", "generic-cla", "other-permissive", "other-copyleft",
})


def load_references(sc_path: Path = SC_CACHE, fo_path: Path = FO_CACHE) -> dict[str, str]:
    """One canonical license text per key — ScanCode wins, FOSSology fills gaps."""
    ref: dict[str, str] = {}
    for d in json.load(open(sc_path)):
        text = (d.get("license_text") or "").strip()
        if text:
            ref[d["license_key"].lower()] = text
    for d in json.load(open(fo_path)):
        text = (d.get("rf_text") or "").strip()
        if text:
            ref.setdefault(d["rf_shortname"].lower(), text)
    return ref


def load_fossology_texts(fo_path: Path = FO_CACHE) -> dict[str, str]:
    """FOSSology's own rendering per license — the query side of the crossref eval."""
    out: dict[str, str] = {}
    for d in json.load(open(fo_path)):
        text = (d.get("rf_text") or "").strip()
        if text:
            out.setdefault(d["rf_shortname"].lower(), text)
    return out


def load_scancode_texts(sc_path: Path = SC_CACHE) -> dict[str, str]:
    """ScanCode's rendering per license — the index side of the crossref eval."""
    out: dict[str, str] = {}
    for d in json.load(open(sc_path)):
        text = (d.get("license_text") or "").strip()
        if text:
            out[d["license_key"].lower()] = text
    return out


def spdx_to_key_map(sc_path: Path = SC_CACHE) -> dict[str, str]:
    """SPDX id (lowercased) -> ScanCode reference key."""
    out: dict[str, str] = {}
    for d in json.load(open(sc_path)):
        spdx = (d.get("spdx_license_key") or "").strip().lower()
        if spdx:
            out[spdx] = d["license_key"].lower()
    return out


def gt_components(expr: str) -> set[str]:
    """Component license keys of a (possibly compound) expression, lowercased."""
    parts = [p.strip().lower() for p in _COMPOUND.split(expr) if p.strip()]
    return {p for p in parts if p and p not in SKIP_KEYS}


def build_notice_refs(min_chars: int = 15, max_chars: int = 3000) -> list[tuple[str, str]]:
    """Extract single-license short-form rule texts from the installed scancode-toolkit.

    Compound (AND/OR/WITH) rules are skipped: they key to an expression rather
    than one license, so they cannot be scored against a single-license label.
    """
    import licensedcode
    from licensedcode.models import load_rules

    base = Path(licensedcode.__file__).parent / "data" / "rules"
    out: list[tuple[str, str]] = []
    for rule in load_rules(base):
        expr = (rule.license_expression or "").strip().lower()
        if not expr or _COMPOUND.search(expr):
            continue
        if not any(getattr(rule, kind, False) for kind in _KINDS):
            continue
        raw = rule.text() if callable(getattr(rule, "text", None)) else getattr(rule, "text", "")
        text = " ".join(_MARK.sub(" ", raw).split())
        if min_chars <= len(text) <= max_chars:
            out.append((expr, text))
    return out


def load_notice_refs() -> list[tuple[str, str]]:
    """Cached notice layer; builds it from scancode-toolkit on first use."""
    if NOTICE_CACHE.exists():
        return [tuple(x) for x in json.loads(NOTICE_CACHE.read_text())]
    units = build_notice_refs()
    NOTICE_CACHE.parent.mkdir(exist_ok=True)
    NOTICE_CACHE.write_text(json.dumps(units))
    return units


def reference_units(refs: dict[str, str], layer: str) -> list[tuple[str, str]]:
    """Assemble the index as (license_key, text) units for ``layer``.

    ``notices``/``both`` share ScanCode provenance with ScanCode-labelled corpora,
    making those evals optimistic — use the SPDX-tag or crossref suite for an
    independent read.
    """
    units: list[tuple[str, str]] = []
    if layer in ("fulltext", "both"):
        units += [(k, refs[k]) for k in sorted(refs)]
    if layer in ("notices", "both"):
        units += [(k, t) for k, t in load_notice_refs() if k in refs]
    return units


def main(argv=None) -> None:
    """Summarize the notice layer — building it from scancode-toolkit if uncached."""
    from collections import Counter

    refs = load_references()
    units = load_notice_refs()
    keys = Counter(k for k, _ in units)
    matched = sum(1 for k, _ in units if k in refs)
    print(f"notice reference units: {len(units)}  distinct licenses: {len(keys)}")
    print(f"units whose license has a full-text reference: {matched}")
    print(f"full-text references: {len(refs)} licenses")
    print("top 10 by unit count:", keys.most_common(10))


if __name__ == "__main__":
    main()
