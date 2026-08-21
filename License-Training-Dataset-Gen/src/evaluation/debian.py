"""Debian DEP-5 corpus — independent license labels, and the licence tail.

Every corpus here is labelled by a matcher or by an in-file SPDX tag. Both are
narrow: the SPDX-tag corpus carries only ~15 licenses in practice, so the tail of
the ~3,000-license index is unmeasured, and the ScanCode-labelled corpus shares
provenance with the reference layer it is scored against.

Debian's `debian/copyright` is a third source with neither problem. Maintainers
curate it by hand, per file glob, in the machine-readable DEP-5 format, with no
matcher involved — and Debian ships tens of thousands of packages, so it reaches
licenses that barely occur in a code-hosting sample.

Measured on a 1,600-package sample: 79% of packages ship machine-readable DEP-5, and
they name **59 distinct licenses** against the SPDX-tag corpus's 15.

**Label noise is real and must be quoted with any result from this corpus.** Of 359
corpus queries that both Atarashi and ScanCode answered, 48 contradict the DEP-5
label and 26 of those have the two independent engines agreeing with *each other* —
a ~7% floor on label error. It comes from DEP-5 being loose about only-vs-or-later
(maintainers write `GPL-2` for a file that says "version 2 or any later version") and
from globs attributing a package-level license to a file whose own header differs.
Good enough to compare engines against each other on identical labels; not good
enough to read as absolute accuracy.

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
    "expat": "MIT", "artistic-1": "Artistic-1.0", "artistic-2": "Artistic-2.0",
    "zlib/libpng": "Zlib", "wtfpl-2": "WTFPL", "cc0": "CC0-1.0",
    "boost": "BSL-1.0", "apache": "Apache-2.0",
}
# Names that name a family rather than a license. Resolving them to a specific
# version is a guess, and a guess in the ground truth is worse than a gap: bare
# `Artistic` was mapped to Artistic-1.0 and produced four "errors" where both the
# agent and ScanCode independently read Artistic-2.0 from the file.
AMBIGUOUS = frozenset({"artistic", "gpl", "lgpl", "agpl", "bsd", "cc-by", "zlib/png"})
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
        stem = name[:-1]
        # Try the or-later form *first*: the list carries `GPL-3.0+` alongside
        # `GPL-3.0`, and collapsing to the bare form turns an or-later grant into an
        # only grant. That is the same distinction the SPDX resolver preserves, and
        # dropping it here would score a correct or-later answer as wrong.
        bumped = _BARE_VERSION.match(stem)
        if bumped:
            yield f"{bumped.group(1)}-{bumped.group(2)}.0+"
        yield from _variants(stem)
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
    if cleaned.lower() in AMBIGUOUS:
        return False
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


# Files worth asking about. A DEP-5 glob labels every path it covers, including
# build files and data, so restricting to source keeps the query set to things that
# could carry a license header at all.
SOURCE_SUFFIXES = (".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".java", ".py", ".rb",
                   ".pl", ".pm", ".go", ".rs", ".js", ".ts", ".php", ".sh", ".m",
                   ".swift", ".kt", ".scala", ".lua", ".el")
# Directories that carry packaging or vendored copies rather than the project's own
# source; their licenses are real but attributing them to this package is noise.
SKIP_DIRS = frozenset({".pc", "debian", ".git", ".github", "test", "tests", "po"})
HEADER_LINES = 40


def _api(path: str):
    raw = _cached("api:" + path, lambda: _get(f"{BASE}/api/src/{path}"))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def source_files(package: str, limit: int = 6) -> tuple[str, list[str]]:
    """Up to `limit` source paths in `package`, with the version they came from.

    Walks the top level and one directory down. Deeper walks cost a request each and
    buy little: projects that put source in `src/` or `lib/` are covered, and those
    that nest further contribute their top-level files instead.
    """
    root = _api(f"{urllib.parse.quote(package)}/latest/")
    if not root or not root.get("version"):
        return "", []
    version = root["version"]
    quoted = f"{urllib.parse.quote(package)}/{urllib.parse.quote(version)}"

    def sources(entries, prefix=""):
        return [prefix + e["name"] for e in entries
                if e.get("type") == "file" and e["name"].endswith(SOURCE_SUFFIXES)]

    entries = root.get("content", [])
    found = sources(entries)
    for entry in entries:
        if len(found) >= limit:
            break
        if entry.get("type") != "directory" or entry["name"] in SKIP_DIRS:
            continue
        child = _api(f"{quoted}/{urllib.parse.quote(entry['name'])}/")
        if child:
            found += sources(child.get("content", []), prefix=entry["name"] + "/")
    return version, found[:limit]


def file_license(package: str, version: str, path: str) -> str | None:
    """The DEP-5 license Debian records for one file, resolved by their own globs."""
    url = (f"{BASE}/copyright/api/file/{urllib.parse.quote(package)}/"
           f"{urllib.parse.quote(version)}/{urllib.parse.quote(path)}/")
    raw = _cached(f"lic:{package}:{version}:{path}", lambda: _get(url))
    if not raw:
        return None
    try:
        result = json.loads(raw).get("result") or []
    except ValueError:
        return None
    return result[0]["copyright"]["license"] if result else None


def file_header(package: str, version: str, path: str) -> str | None:
    meta = _api(f"{urllib.parse.quote(package)}/{urllib.parse.quote(version)}/"
                f"{urllib.parse.quote(path)}/")
    if not meta or not meta.get("raw_url"):
        return None
    body = _cached(f"raw:{package}:{version}:{path}", lambda: _get(BASE + meta["raw_url"]))
    return body


def build_corpus(packages: list[str], per_package: int = 6, workers: int = 12) -> list[dict]:
    """(query, label) pairs: a file header, labelled by Debian's own DEP-5 record.

    The SPDX tag is stripped from the query, as in the spdx-tag suite, so a file
    that carries both is still scored on its prose rather than on the tag path.
    """
    import pandas as pd
    from atarashi.spdx.resolver import shortname_index
    from evaluation.spdx_tag import TAGLINE, classify_regime

    csv = Path(__import__("atarashi").__file__).parent / "data" / "licenses" / "licenseList.csv"
    index = shortname_index(pd.read_csv(csv)["shortname"])

    def one(package: str) -> list[dict]:
        version, paths = source_files(package, per_package)
        rows = []
        for path in paths:
            raw_name = file_license(package, version, path)
            if not raw_name:
                continue
            label = canonical_license(raw_name, index)
            if label is None or label is False:
                continue
            body = file_header(package, version, path)
            if not body:
                continue
            head = " ".join(TAGLINE.sub(" ", "\n".join(
                body.splitlines()[:HEADER_LINES])).split())
            if len(head) < 40:
                continue
            rows.append({"text": head, "gt": label, "dep5_name": raw_name,
                         "regime": classify_regime(head),
                         "package": package, "path": path, "version": version})
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [row for rows in pool.map(one, packages) for row in rows]


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("names", "corpus"), default="names")
    ap.add_argument("--packages", type=int, default=400, help="sample size")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-package", type=int, default=6, help="corpus mode: files per package")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    names = list_packages()
    if not names:
        raise SystemExit("could not list Debian packages (network?)")
    step = max(1, len(names) // args.packages)
    sample = names[::step][:args.packages]          # spread across the alphabet
    print(f"Debian source packages: {len(names)}; sampling {len(sample)}")

    if args.mode == "corpus":
        rows = build_corpus(sample, args.per_package, min(args.workers, 12))
        by_regime = Counter(r["regime"] for r in rows)
        licenses = Counter(r["gt"] for r in rows)
        print(f"labelled files: {len(rows)}  distinct licenses: {len(licenses)}")
        print(f"  notice regime   : {by_regime['notice']}  (answerable)")
        print(f"  no-signal       : {by_regime['no-signal']}  (correct answer is UNKNOWN)")
        print(f"top licenses: {licenses.most_common(12)}")
        out_path = ROOT / "cache" / "debian_corpus.json"
        out_path.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {out_path}")
        return

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
