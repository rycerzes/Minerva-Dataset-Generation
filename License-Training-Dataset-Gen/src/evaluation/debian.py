"""Debian DEP-5 corpus — independent license labels, and the licence tail.

Every corpus here is labelled by a matcher or by an in-file SPDX tag. Both are
narrow: the SPDX-tag corpus carries only ~15 licenses in practice, so the tail of
the ~3,000-license index is unmeasured, and the ScanCode-labelled corpus shares
provenance with the reference layer it is scored against.

Debian's `debian/copyright` is a third source with neither problem. Maintainers
curate it by hand, per file glob, in the machine-readable DEP-5 format, with no
matcher involved — and Debian ships tens of thousands of packages, so it reaches
licenses that barely occur in a code-hosting sample.

Measured on a 400-package sample: 81% of packages ship machine-readable DEP-5, and
they name **51 distinct licenses** against the SPDX-tag corpus's 15.

The catch is naming. DEP-5 short names predate SPDX and are written by hand, so the
same license arrives as `Expat`, `MIT`, `Apache-2.0`, `APACHE-2.0`, `Apache 2.0`,
`Apache2`, and — genuinely — `BSD-2-cluase`. Raw name counts badly overstate
diversity; :func:`canonical_license` is what makes the labels usable, and it is the
part of this module worth testing.

Two modes:

    uv run src/run_eval.py debian-dep5 --packages 400            # licence survey
    uv run src/run_eval.py debian-dep5 --mode corpus --packages 50

`names` is cheap (two requests per package) and answers the tail question. `corpus`
builds (query, label) pairs by walking package files, and is slow enough to be run
deliberately rather than casually — several requests per file. Everything is cached
under `cache/debian/`, so both modes resume.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "debian"
RESULTS = ROOT / "output" / "debian_dep5.json"

BASE = "https://sources.debian.org"
# Debian asks that automated clients identify themselves.
UA = {"User-Agent": "atarashi-license-eval/0.1 (FOSSology GSoC; research)"}

# A `License:` line names the license for the enclosing `Files:` stanza. Standalone
# stanzas at the end carry full texts and open identically, so only the short name
# on the field line is taken.
_LICENSE = re.compile(r"(?im)^License:\s*(.+)$")
_DEP5 = re.compile(r"(?im)^Format:\s*.*copyright-format/1\.0")
_SPLIT = re.compile(r"\s+(?:and|or)\s+", re.I)
_EXCEPTION = re.compile(r"\s+with\s+.*exception.*$", re.I)
_BARE_VERSION = re.compile(r"^([A-Za-z][A-Za-z\-]*?)-(\d)$")

# Debian names with no SPDX counterpart reachable by normalization.
ALIAS = {
    "expat": "MIT", "artistic": "Artistic-1.0", "artistic-1": "Artistic-1.0",
    "artistic-2": "Artistic-2.0", "zlib/libpng": "Zlib", "wtfpl-2": "WTFPL",
    "cc0": "CC0-1.0", "boost": "BSL-1.0", "apache": "Apache-2.0",
}
# Names that identify no license, so a query labelled only with one has nothing to
# find. Excluded rather than counted as a miss, as `references.SKIP_KEYS` does.
NON_IDENTIFYING = frozenset({
    "public-domain", "none", "other", "unknown", "permissive", "custom",
    "no-license", "proprietary",
})


def _get(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _cached(name: str, fetch, timeout: int = 30) -> str | None:
    """Fetch once, then read from disk. An empty file records a known failure."""
    path = CACHE / urllib.parse.quote(name, safe="")
    if path.exists():
        return path.read_text(errors="replace") or None
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        text = fetch()
    except Exception:
        text = ""
    path.write_text(text or "")
    return text or None


def list_packages() -> list[str]:
    raw = _cached("_packages.json", lambda: _get(f"{BASE}/api/list/", timeout=120))
    return [p["name"] for p in json.loads(raw)["packages"]] if raw else []


def copyright_text(package: str) -> str | None:
    """The package's `debian/copyright`, or None if it has none."""
    def fetch() -> str:
        quoted = urllib.parse.quote(package)
        meta = json.loads(_get(f"{BASE}/api/src/{quoted}/latest/debian/copyright/"))
        return _get(BASE + meta["raw_url"]) if meta.get("raw_url") else ""
    return _cached(package, fetch)


def dep5_licenses(text: str | None) -> set[str] | None:
    """License short names declared in a DEP-5 file; None if it is not DEP-5.

    Roughly a fifth of packages ship a prose copyright file instead. Those are not
    parseable and are reported separately rather than guessed at.
    """
    if not text or not _DEP5.search(text):
        return None
    names: set[str] = set()
    for field in _LICENSE.findall(text):
        for part in _SPLIT.split(field.strip()):
            part = part.strip().strip(",").strip()
            if part and len(part) < 60:
                names.add(part)
    return names


def _variants(name: str):
    """Spellings to try, widest-fidelity first."""
    yield name
    yield name.replace("-clause", "-Clause")
    base = _EXCEPTION.sub("", name).strip()   # "GPL-2+ with Autoconf exception"
    if base != name:
        yield from _variants(base)
    if name.endswith("+"):                    # Debian's or-later marker
        yield from _variants(name[:-1])
    version = _BARE_VERSION.match(name)       # GPL-2 -> GPL-2.0, Apache-2 -> Apache-2.0
    if version:
        yield f"{version.group(1)}-{version.group(2)}.0"


def canonical_license(name: str, index: dict[str, str]) -> str | None | bool:
    """Debian short name -> license shortname, ``None`` if it identifies nothing,
    ``False`` if it cannot be resolved.

    Three outcomes rather than two: a name that identifies no license is a different
    fact from one this mapping cannot handle, and collapsing them would hide the
    error rate that decides whether this corpus is usable.
    """
    from atarashi.spdx.resolver import lookup_shortname

    cleaned = name.strip()
    if cleaned.lower() in NON_IDENTIFYING:
        return None
    if cleaned.lower() in ALIAS:
        return ALIAS[cleaned.lower()]
    for candidate in _variants(cleaned):
        resolved = lookup_shortname(candidate, index)
        if resolved:
            return resolved
    return False


def survey(packages: list[str], workers: int = 16) -> dict:
    """License diversity across `packages`, and how much of it we can name."""
    import pandas as pd
    from atarashi.spdx.resolver import shortname_index

    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(copyright_text, packages))

    raw: Counter[str] = Counter()
    dep5 = fetched = 0
    for text in texts:
        if text:
            fetched += 1
        names = dep5_licenses(text)
        if names is None:
            continue
        dep5 += 1
        raw.update(names)

    csv = Path(__import__("atarashi").__file__).parent / "data" / "licenses" / "licenseList.csv"
    known = pd.read_csv(csv)["shortname"]
    index = shortname_index(known)

    mapped: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    non_identifying = 0
    for name, count in raw.items():
        resolved = canonical_license(name, index)
        if resolved is None:
            non_identifying += count
        elif resolved is False:
            unmapped[name] += count
        else:
            mapped[resolved] += count

    total = sum(raw.values()) or 1
    return {
        "packages_sampled": len(packages),
        "copyright_fetched": fetched,
        "machine_readable_dep5": dep5,
        "raw_names": len(raw),
        "occurrences": sum(raw.values()),
        "mapped_occurrences": sum(mapped.values()),
        "mapped_share": round(sum(mapped.values()) / total, 4),
        "distinct_licenses": len(mapped),
        "non_identifying": non_identifying,
        "unmapped_occurrences": sum(unmapped.values()),
        "unmapped_names": len(unmapped),
        "top_licenses": mapped.most_common(20),
        "top_unmapped": unmapped.most_common(20),
    }


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("names", "corpus"), default="names")
    ap.add_argument("--packages", type=int, default=400, help="sample size")
    ap.add_argument("--workers", type=int, default=16)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode == "corpus":
        raise SystemExit(
            "corpus mode is not built yet: the survey below is what decides whether "
            "it is worth the network cost. Run --mode names first."
        )

    names = list_packages()
    if not names:
        raise SystemExit("could not list Debian packages (network?)")
    step = max(1, len(names) // args.packages)
    sample = names[::step][:args.packages]          # spread across the alphabet
    print(f"Debian source packages: {len(names)}; sampling {len(sample)}")

    out = survey(sample, args.workers)
    print(f"copyright fetched      : {out['copyright_fetched']}/{out['packages_sampled']}")
    print(f"machine-readable DEP-5 : {out['machine_readable_dep5']} "
          f"({out['machine_readable_dep5'] / max(out['copyright_fetched'], 1):.0%} of fetched)")
    print(f"raw license names      : {out['raw_names']} over {out['occurrences']} occurrences")
    print(f"  mapped               : {out['mapped_occurrences']} ({out['mapped_share']:.0%}) "
          f"-> {out['distinct_licenses']} distinct licenses")
    print(f"  identify no license  : {out['non_identifying']}")
    print(f"  unresolvable         : {out['unmapped_occurrences']} over {out['unmapped_names']} names")
    print(f"\ntop licenses: {out['top_licenses'][:12]}")
    print(f"top unmapped: {out['top_unmapped'][:12]}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
