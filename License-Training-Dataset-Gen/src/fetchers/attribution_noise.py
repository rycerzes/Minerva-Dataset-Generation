"""Synthesize hard-negative code comments in the attribution/credit register.

Error-mining potion-32M's `generic_code_comment` false positives showed the
model conflates a specific register with license/copyright notices: author tags
(``@author X``), credit lines (``Created by X on DATE``), contributor lines
(``# Name <email>``), short descriptive docstrings, and URL "see also" comments.
None contain license keywords, so the real-comment fetcher never drops them —
there just aren't enough in the corpus for the model to learn the boundary.

These deterministic, entity-diversified templates flood that boundary (same
playbook that broke the srn ceiling). No LLM, no network. Diversity matters:
low-entropy templates collapse under near-dedup, so entities are randomized
widely. Tagged ``generic_code_comment`` so they land in the negative class.
"""

from __future__ import annotations

import random

try:
    from .code_comments import CodeCommentSample, LICENSE_KEYWORDS
except ImportError:  # pragma: no cover — direct-script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from fetchers.code_comments import CodeCommentSample, LICENSE_KEYWORDS

_FIRST = [
    "Eugene", "Boris", "Dan", "Gavri", "Mei", "Priya", "Lars", "Kenji", "Ana",
    "Omar", "Freya", "Tariq", "Ingrid", "Diego", "Yuki", "Kwame", "Sofia",
    "Nikolai", "Aisha", "Mateo", "Hana", "Ravi", "Elena", "Sven", "Chidi",
    "Leila", "Marco", "Wei", "Fatima", "Johan", "Isabela", "Arjun",
]
_LAST = [
    "Zhuravlev", "Heithecker", "Schult", "Okafor", "Nakamura", "Patel", "Larsson",
    "Rossi", "Kim", "Haddad", "Andersen", "Silva", "Novak", "Mensah", "Vidal",
    "Fischer", "Costa", "Ivanov", "Khan", "Weber", "Tanaka", "Mbeki", "Dubois",
    "Sato", "Nguyen", "Reyes", "Bauer", "Moreau", "Singh", "Lindqvist",
]
_ORGS = [
    "Northwind Labs", "Contoso", "Initech", "Umbrella Systems", "Globex",
    "Hooli", "Wayne Robotics", "Stark Devtools", "Pied Piper", "Vandelay",
    "Soylent Analytics", "Cyberdyne", "Massive Dynamic", "Wonka Software",
    "Acme", "Tyrell", "Aperture", "Bluth Group", "Gekko Capital", "Nakatomi",
]
_TLDS = ["com", "io", "org", "dev", "net", "co", "app"]
_LANGS = ["C", "Perl", "Java", "Lisp", "Fortran", "Ruby", "Scala", "Haskell", "Go"]
_VERS = ["1.0", "2.3", "0.9", "3.1.4", "1.12", "4.0-beta", "2.0.0", "0.14"]
_VERB = ["Provides", "Handles", "Manages", "Represents", "Encapsulates",
         "Implements", "Coordinates", "Wraps", "Exposes", "Tracks"]
_OBJ = ["access to blog entries", "the connection pool", "billing information",
        "user session state", "the render pipeline", "cache invalidation",
        "the event queue", "document references", "the simulated world",
        "retry and backoff logic", "the plugin registry", "audio playback",
        "geometry buffers", "the scheduler", "request throttling",
        "tick styling", "the widget tree", "telemetry counters"]


def _name(rng): return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def _email(rng, name):
    user = name.lower().replace(" ", ".")
    return f"{user}@{rng.choice(_ORGS).split()[0].lower()}.{rng.choice(_TLDS)}"


def _url(rng):
    return f"https://{rng.choice(_ORGS).split()[0].lower()}.{rng.choice(_TLDS)}/{rng.choice(['docs','wiki','ref','api','blog'])}"


def _date(rng):
    return f"{rng.randint(2005, 2024)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"


# Each returns a bare comment body (no comment prefix yet).
_BUILDERS = [
    lambda rng: f"@author {_name(rng)}",
    lambda rng: (lambda n: f"@author {n} <{_email(rng, n)}>")(_name(rng)),
    lambda rng: f"@since {rng.choice(_VERS)}",
    lambda rng: f"@version {rng.choice(_VERS)}",
    lambda rng: f"@contributor {_name(rng)}",
    lambda rng: f"Created by {_name(rng)} on {_date(rng)}.",
    lambda rng: f"Written by {_name(rng)}, {rng.randint(2005,2024)}.",
    lambda rng: (lambda n: f"Authored by {n} ({_email(rng, n)}).")(_name(rng)),
    lambda rng: f"Maintained by {_name(rng)}.",
    lambda rng: (lambda n: f"{n} <{_email(rng, n)}>")(_name(rng)),
    lambda rng: f"Ported from {rng.choice(_LANGS)} by {_name(rng)}.",
    lambda rng: f"Based on {rng.choice(_ORGS)}'s implementation.",
    lambda rng: f"Adapted from {_url(rng)}.",
    lambda rng: f"Originally by {_name(rng)}, refactored {rng.randint(2015,2024)}.",
    lambda rng: f"{rng.choice(_VERB)} {rng.choice(_OBJ)}.",
    lambda rng: f"{rng.choice(_VERB)} {rng.choice(_OBJ)} for the {rng.choice(_LANGS)} runtime.",
    lambda rng: f"See {_url(rng)} for details.",
    lambda rng: f"More info: {_url(rng)}",
    lambda rng: f"Reference implementation: {_url(rng)}",
]

_PREFIXES = ["//", "#", " *", "/**", '"""', "--", ";;", ""]


def generate_attribution_negatives(n: int = 4000, seed: int = 0) -> list[CodeCommentSample]:
    """Return up to *n* deduplicated attribution-register comments as negatives."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[CodeCommentSample] = []
    # Cap attempts so we never spin forever if the template space is exhausted.
    for _ in range(n * 20):
        if len(out) >= n:
            break
        body = rng.choice(_BUILDERS)(rng)
        low = body.lower()
        if any(kw in low for kw in LICENSE_KEYWORDS):  # never a license notice
            continue
        pre = rng.choice(_PREFIXES)
        text = f"{pre} {body}".strip() if pre else body
        if pre == "/**":
            text = f"/** {body} */"
        elif pre == '"""':
            text = f'"""{body}"""'
        if text in seen:
            continue
        seen.add(text)
        out.append(CodeCommentSample(
            text=text, language="synthetic", source="attribution_template",
            comment_type="attribution_noise",
        ))
    return out


if __name__ == "__main__":  # self-check
    s = generate_attribution_negatives(3000)
    assert len(s) > 2500, f"only {len(s)} unique — template space too small"
    assert not any(
        kw in x.text.lower() for x in s for kw in LICENSE_KEYWORDS
    ), "a template leaked a license keyword"
    uniq = len({x.text for x in s})
    assert uniq == len(s), "duplicates present"
    print(f"OK: {len(s)} unique attribution negatives, e.g.:")
    for x in s[:8]:
        print("  ", x.text)
