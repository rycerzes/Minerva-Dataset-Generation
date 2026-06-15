"""Code comment fetcher for generic developer noise (negative class).

Streams real code comments from ``bigcode/the-stack-smol`` via the
HuggingFace datasets library and returns short, non-license text spans
suitable for use as ``not_license_related`` training samples.

The HuggingFace API token is read from the ``HF_API_TOKEN`` key in the
project ``.env`` file (same directory as the repository root).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Regex patterns for extracting comment-length spans

COMMENT_PATTERNS: dict[str, list[str]] = {
    "single_line": [
        r"(?m)^[ \t]*(#[^!\n][^\n]*)",  # Python/Shell (exclude shebangs)
        r"(?m)^[ \t]*(//(?!/).+)",  # C-style single line
        r"(?m)^[ \t]*(--\s+.+)",  # SQL
    ],
    "docstring": [
        r'"""(.+?)"""',  # Python triple-quoted (non-greedy)
        r"'''(.+?)'''",
        r"/\*\*(.+?)\*/",  # JSDoc
    ],
    "inline": [
        r"[^:\"'\n]\s(#\s[A-Z].{10,80})",  # Capitalised inline comment
    ],
}

# Any extracted text containing these tokens is dropped so license headers
# that happen to live inside comments are never labelled not_license_related.
LICENSE_KEYWORDS: frozenset[str] = frozenset(
    {
        "license",
        "copyright",
        "spdx",
        "permission",
        "warranty",
        "redistribution",
        "sublicense",
        "licensor",
        "gpl",
        "mit license",
        "apache",
        "bsd",
        "lgpl",
        "mpl",
        "gnu",
        "all rights reserved",
        "proprietary",
        "confidential",
    }
)

# Compile all patterns once at import time
_COMPILED: dict[str, list[re.Pattern[str]]] = {
    comment_type: [re.compile(p, re.DOTALL | re.IGNORECASE) for p in patterns]
    for comment_type, patterns in COMMENT_PATTERNS.items()
}

# Default languages sampled from the-stack-smol
_DEFAULT_LANGUAGES = [
    "python",
    "javascript",
    "java",
    "go",
    "c",
    "cpp",
    "rust",
    "shell",
    "sql",
    "typescript",
    "ruby",
]


# Data model


class CodeCommentSample(BaseModel):
    """A single non-license code comment extracted from a real source file."""

    text: str
    language: str
    source: str = "the-stack-smol"
    comment_type: str  # "single_line" | "docstring" | "inline"


# Helper — .env reader (mirrors config._parse_env_file without importing it)


def _read_hf_token(env_path: Optional[Path] = None) -> Optional[str]:
    """Return ``HF_API_TOKEN`` from the project ``.env`` file, or *None*."""
    if env_path is None:
        # Walk up from this file to find the repo root .env
        env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        logger.debug(".env not found at %s — no HF token available", env_path)
        return None
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("HF_API_TOKEN="):
                token = line.split("=", 1)[1].strip()
                return token or None
    return None


# Comment extraction


def _extract_comments(
    content: str,
    min_len: int,
    max_len: int,
) -> Iterator[tuple[str, str]]:
    """Yield ``(comment_text, comment_type)`` spans from *content*.

    Only spans whose character length is within [*min_len*, *max_len*] are
    returned.  Strips leading/trailing whitespace and collapses internal
    newlines to a single space for single-line & inline comment types.
    """
    for comment_type, patterns in _COMPILED.items():
        for pattern in patterns:
            for match in pattern.finditer(content):
                raw = match.group(1)
                if comment_type in ("single_line", "inline"):
                    text = re.sub(r"\s+", " ", raw).strip()
                else:
                    # Docstrings: collapse excessive whitespace but keep shape
                    text = re.sub(r"\n\s*\n", "\n", raw).strip()
                    text = re.sub(r"[ \t]+", " ", text)
                if min_len <= len(text) <= max_len:
                    yield text, comment_type


def _is_license_adjacent(text: str) -> bool:
    """Return True if *text* contains any license-related keyword."""
    lower = text.lower()
    return any(kw in lower for kw in LICENSE_KEYWORDS)


# Fetcher


class CodeCommentFetcher:
    """Stream code comments from ``bigcode/the-stack-smol`` via HuggingFace.

    Parameters
    ----------
    languages:
        Programming languages to sample from.  Defaults to the most common
        languages in real OSS repositories.
    min_len:
        Minimum character length for an extracted comment.
    max_len:
        Maximum character length for an extracted comment.
    cache_dir:
        Directory for caching the fetched samples as JSON.  Subsequent runs
        load from cache and skip the download entirely.
    env_path:
        Optional explicit path to the ``.env`` file containing
        ``HF_API_TOKEN``.  Defaults to the repository root ``.env``.
    """

    CACHE_FILENAME = "code_comments.json"
    DATASET_REPO = "bigcode/the-stack-smol"

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        min_len: int = 20,
        max_len: int = 400,
        cache_dir: Optional[str | Path] = None,
        env_path: Optional[Path] = None,
    ) -> None:
        self.languages = languages or _DEFAULT_LANGUAGES
        self.min_len = min_len
        self.max_len = max_len
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._token = _read_hf_token(env_path)
        if self._token:
            logger.debug("HF_API_TOKEN loaded from .env")
        else:
            logger.warning("HF_API_TOKEN not found — dataset access may be restricted")

    # Cache helpers

    def _cache_path(self) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / self.CACHE_FILENAME

    def _load_cache(self) -> Optional[list[CodeCommentSample]]:
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        logger.info("Loading code comments from cache: %s", path)
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return [CodeCommentSample(**item) for item in raw]

    def _save_cache(self, samples: list[CodeCommentSample]) -> None:
        path = self._cache_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([s.model_dump() for s in samples], fh, indent=2)
        logger.info("Cached %d code comments → %s", len(samples), path)

    # Main public API

    def fetch(self, max_samples: int = 25_000) -> list[CodeCommentSample]:
        """Return up to *max_samples* clean non-license code comments.

        Results are cached to ``cache_dir/code_comments.json``; subsequent
        calls with the same ``cache_dir`` skip the HuggingFace download.

        Parameters
        ----------
        max_samples:
            Total number of comments to return across all languages.
        """
        cached = self._load_cache()
        if cached is not None:
            logger.info("Using %d cached code comments", len(cached))
            return cached[:max_samples]

        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for code comment fetching. "
                "Install it with: pip install datasets"
            )

        samples: list[CodeCommentSample] = []
        seen: set[str] = set()

        per_language = max(1, max_samples // len(self.languages))
        logger.info(
            "Streaming %s (%d languages, ~%d comments each)",
            self.DATASET_REPO,
            len(self.languages),
            per_language,
        )

        for lang in self.languages:
            if len(samples) >= max_samples:
                break
            lang_count = 0
            lang_target = min(per_language, max_samples - len(samples))
            logger.debug("Streaming language='%s', target=%d", lang, lang_target)

            try:
                ds = load_dataset(
                    self.DATASET_REPO,
                    data_dir=f"data/{lang}",
                    split="train",
                    streaming=True,
                    token=self._token,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to stream '%s' for language '%s': %s — skipping",
                    self.DATASET_REPO,
                    lang,
                    exc,
                )
                continue

            for row in ds:
                if lang_count >= lang_target:
                    break

                content: str = row.get("content", "") or ""
                if not content:
                    continue

                for text, comment_type in _extract_comments(
                    content, self.min_len, self.max_len
                ):
                    if lang_count >= lang_target:
                        break
                    if text in seen:
                        continue
                    if _is_license_adjacent(text):
                        continue
                    seen.add(text)
                    samples.append(
                        CodeCommentSample(
                            text=text,
                            language=lang,
                            comment_type=comment_type,
                        )
                    )
                    lang_count += 1

            logger.info("  %-12s → %d comments", lang, lang_count)

        logger.info("Total code comments collected: %d", len(samples))
        self._save_cache(samples)
        return samples
