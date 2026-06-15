"""Rare-license paraphrase augmenter for Atarashi.

Identifies licenses with fewer than *threshold* sliding-window fragments
in the current fragment pool and uses an LLM to generate additional
reformatted variations of the original license text.

Filtering
---------
Each generated variant is checked against the original license text using
character 3-gram Jaccard similarity:

  * similarity < ``min_similarity``: rejected — LLM drifted too far from
    the original legal language (potential class poisoning).
  * similarity > ``max_similarity``: rejected — essentially a duplicate,
    adds no useful diversity.
  * otherwise: accepted as a synthetic ``SplitFragment`` with
    ``source="synthetic"``.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from ..builder.hybrid_merge import DatasetEntry
    from ..config import LLMConfig, RateLimiter
    from .legal_structure_splitter import SplitFragment
    from .llm_cache import LLMCache
except ImportError:  # pragma: no cover — direct-script execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from builder.hybrid_merge import DatasetEntry
    from config import LLMConfig, RateLimiter
    from augmentation.legal_structure_splitter import SplitFragment
    from augmentation.llm_cache import LLMCache


# Configuration


class RareLicenseAugmenterConfig(BaseModel):
    """Knobs for the rare-license paraphrase augmenter."""

    model_config = {"protected_namespaces": ()}

    threshold: int = Field(
        default=5,
        ge=1,
        description=(
            "Augment licenses whose sliding-window fragment count is below this value."
        ),
    )
    augment_count: int = Field(
        default=5,
        ge=1,
        description="Number of paraphrased variants to request per rare license.",
    )
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum character 3-gram Jaccard similarity between a generated "
            "variant and the original text.  Variants below this threshold have "
            "drifted too far from the original legal language and are discarded."
        ),
    )
    max_similarity: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum Jaccard similarity allowed.  Variants above this threshold "
            "are essentially duplicates and are discarded."
        ),
    )
    max_text_chars: int = Field(
        default=2000,
        ge=100,
        description=(
            "Maximum characters of the original license text included in the "
            "prompt.  Very long licenses are truncated to this limit."
        ),
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        description=(
            "Max tokens for the LLM response.  Each variant can be up to "
            "~800 tokens, so the default of 4096 comfortably fits 5 variants. "
            "Overrides the global LLMConfig.max_tokens for this stage only."
        ),
    )


# Similarity helper

_VARIANT_SEP = "---VARIANT---"

# Accept the exact requested separator plus common Markdown-decorated variants
# observed in cached LLM responses, e.g. **VARIANT 1**, --- variant ---,
# **VARIANT---**.  The pattern is intentionally line-anchored so occurrences
# of the word "variant" in legal prose do not split the text.
_VARIANT_SEPARATOR_RE = re.compile(
    r"(?im)^\s*(?:[-*_`\s]*)?(?:variant)\b(?:\s*[-#:]*\s*\d+\s*:?)?(?:\s*[-*_`]*)?\s*$"
)
_VARIANT_PREAMBLE_RE = re.compile(r"(?im)^\s*here (?:are|is) .*variants?.*$")
_LICENSE_LABEL_RE = re.compile(r"(?im)^\s*\*{0,2}\s*\[?\s*license\s*:[^\n]*\]?\s*\*{0,2}\s*$")


def _jaccard_ngram(a: str, b: str, n: int = 3) -> float:
    """Return character *n*-gram Jaccard similarity between *a* and *b*."""

    def _ngrams(text: str) -> set[str]:
        t = re.sub(r"\s+", " ", text.lower().strip())
        return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}

    sa, sb = _ngrams(a), _ngrams(b)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0


# Augmenter


class RareLicenseAugmenter:
    """Generate paraphrased variants for rare-class licenses in Atarashi.

    Parameters
    ----------
    config:
        Augmenter configuration.
    llm_config:
        LLM connection settings (model, API key, RPM limit).
    cache:
        Optional shared :class:`LLMCache`.  Results are persisted so
        re-runs skip already-completed licenses.
    """

    CACHE_NAMESPACE = "rare_license_augmentation"
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0

    def __init__(
        self,
        config: Optional[RareLicenseAugmenterConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        cache: Optional[LLMCache] = None,
    ) -> None:
        self.config = config or RareLicenseAugmenterConfig()
        self._llm_config = llm_config or LLMConfig()
        self._rate_limiter = RateLimiter(self._llm_config.rpm)
        self._llm_client = None
        self._cache = cache

    # LLM helpers

    def _get_llm_client(self):
        if self._llm_client is None:
            import litellm

            litellm.drop_params = True
            self._llm_client = litellm
        return self._llm_client

    def _build_prompt(self, license_key: str, license_text: str) -> str:
        n = self.config.augment_count
        snippet = license_text[: self.config.max_text_chars].strip()
        return (
            f"You are a legal text formatter. "
            f"Produce exactly {n} reformatted variants of the following software "
            f"license text.\n\n"
            "STRICT RULES:\n"
            "- Preserve ALL legal terms, conditions, obligations, permissions, "
            "and defined terms WORD FOR WORD.\n"
            "- You may ONLY change: line wrapping, whitespace, paragraph spacing, "
            "list/bullet formatting, or heading capitalisation.\n"
            "- Do NOT add, remove, summarise, paraphrase, or alter any legal "
            "language whatsoever.\n"
            "- Do NOT change any URLs, version numbers, copyright notices, or "
            "SPDX identifiers.\n"
            f'- Separate each variant with a line containing only "{_VARIANT_SEP}".\n\n'
            f"LICENSE: {license_key}\n\n"
            f"ORIGINAL TEXT:\n{snippet}\n\n"
            f"Generate {n} formatted variants now:"
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        llm = self._get_llm_client()
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limiter.acquire()
                resp = llm.completion(
                    model=self._llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=0.7,
                    api_base=self._llm_config.api_base_url,
                    api_key=self._llm_config.api_key,
                    custom_llm_provider="openai",
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                if attempt < self.MAX_RETRIES - 1:
                    backoff = self.RETRY_BACKOFF_BASE**attempt
                    logger.warning(
                        "LLM call failed (attempt %d): %s — retrying in %.1fs",
                        attempt + 1,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "LLM call failed after %d retries: %s",
                        self.MAX_RETRIES,
                        exc,
                    )
        return None

    def _parse_variants(self, raw: str) -> list[str]:
        """Split a raw LLM response into individual variant texts.

        The prompt asks for a line containing exactly ``---VARIANT---``, but
        models sometimes return Markdown headings such as ``**VARIANT 1**`` or
        ``---`` followed by ``**VARIANT---**``.  Normalize those separators so a
        single response containing 10 variants does not become one giant sample.
        """
        normalized = raw.replace(_VARIANT_SEP, "\nVARIANT\n")
        normalized = _VARIANT_PREAMBLE_RE.sub("", normalized)
        normalized = _LICENSE_LABEL_RE.sub("", normalized)
        normalized = re.sub(r"```(?:[A-Za-z0-9_+-]+)?", "", normalized)

        parts = _VARIANT_SEPARATOR_RE.split(normalized)
        if len(parts) == 1:
            # Fallback for cases like "---\n**VARIANT 1**" where a horizontal
            # rule was emitted separately from the variant heading.
            parts = re.split(
                r"(?im)^\s*-{3,}\s*\n\s*\*{0,2}\s*variant\s*\d*\s*\*{0,2}\s*$",
                normalized,
            )

        variants: list[str] = []
        for part in parts:
            text = part.strip()
            text = _VARIANT_PREAMBLE_RE.sub("", text)
            text = _LICENSE_LABEL_RE.sub("", text)
            text = re.sub(r"(?m)^\s*[-=_]{3,}\s*$", "", text).strip()
            if len(text) >= 50:
                variants.append(text)
        return variants

    def _build_fragment(
        self,
        license_key: str,
        variant: str,
    ) -> SplitFragment:
        """Create a synthetic SplitFragment from an accepted variant."""
        return SplitFragment(
            license_key=license_key,
            fragment_text=variant,
            start_position=0,
            end_position=len(variant),
            fragment_index=0,
            total_fragments=1,
            source="synthetic",
            placeholders=[],
            is_first=True,
            is_last=True,
        )

    def _postprocess_variants(
        self,
        license_key: str,
        license_text: str,
        variants: list[str],
    ) -> list[SplitFragment]:
        """Apply similarity filters and convert accepted variants to fragments."""
        cfg = self.config
        result: list[SplitFragment] = []
        for variant in variants:
            sim = _jaccard_ngram(license_text, variant)
            if sim < cfg.min_similarity:
                logger.debug(
                    "Rejected variant for %s: sim=%.2f < min=%.2f",
                    license_key,
                    sim,
                    cfg.min_similarity,
                )
                continue
            if sim > cfg.max_similarity:
                logger.debug(
                    "Skipped near-duplicate for %s: sim=%.2f > max=%.2f",
                    license_key,
                    sim,
                    cfg.max_similarity,
                )
                continue
            result.append(self._build_fragment(license_key, variant))
        return result

    # Per-license augmentation

    def _augment_one(
        self,
        license_key: str,
        license_text: str,
        source: str,
    ) -> tuple[list[SplitFragment], bool]:
        """Return ``(fragments, was_cache_hit)`` for a single license."""
        if self._cache is not None:
            cached = self._cache.get(self.CACHE_NAMESPACE, license_key)
            if cached is not None:
                # Older cache entries may contain a whole multi-variant LLM
                # response as one fragment.  Re-parse cached text in memory so
                # expensive LLM calls do not need to be repeated.
                cached_variants: list[str] = []
                for item in cached:
                    cached_variants.extend(self._parse_variants(item["fragment_text"]))
                return self._postprocess_variants(
                    license_key, license_text, cached_variants
                ), True

        prompt = self._build_prompt(license_key, license_text)
        raw = self._call_llm(prompt)
        if not raw:
            return [], False

        variants = self._parse_variants(raw)
        result = self._postprocess_variants(license_key, license_text, variants)

        if self._cache is not None:
            self._cache.set(
                self.CACHE_NAMESPACE,
                license_key,
                [f.model_dump() for f in result],
            )
        return result, False

    # Public API

    def augment(
        self,
        base_dataset: list[DatasetEntry],
        existing_fragments: list[SplitFragment],
    ) -> list[SplitFragment]:
        """Return synthetic SplitFragment objects for rare-class licenses.

        Parameters
        ----------
        base_dataset:
            Full hybrid dataset (source of original license texts).
        existing_fragments:
            Already-computed sliding-window fragments used to measure how
            many fragments each license already has.

        Returns
        -------
        list[SplitFragment]
            Synthetic fragments (``source="synthetic"``) for licenses
            below the configured threshold.  Append to the existing
            fragment list before passing to downstream pipeline stages.
        """
        threshold = self.config.threshold

        frag_counts: Counter[str] = Counter(f.license_key for f in existing_fragments)
        by_key: dict[str, DatasetEntry] = {
            e.license_key: e for e in base_dataset if e.license_text
        }
        rare_keys = [key for key in by_key if frag_counts.get(key, 0) < threshold]

        if not rare_keys:
            logger.info(
                "No licenses below fragment threshold %d — skipping rare-license augmentation",
                threshold,
            )
            return []

        logger.info(
            "Rare-license augmenter: %d/%d licenses below threshold %d",
            len(rare_keys),
            len(by_key),
            threshold,
        )

        results: list[SplitFragment] = []
        cache_hits = cache_misses = accepted = rejected = 0

        for i, key in enumerate(rare_keys, 1):
            entry = by_key[key]
            frags, was_hit = self._augment_one(
                key, entry.license_text, entry.source.value
            )
            if was_hit:
                cache_hits += 1
            else:
                cache_misses += 1

            accepted += len(frags)
            rejected += max(0, self.config.augment_count - len(frags))
            results.extend(frags)

            if i % 50 == 1 or i == len(rare_keys):
                logger.info("  [%d/%d] %s", i, len(rare_keys), key)

        logger.info(
            "Rare-license augmentation complete: %d synthetic fragments "
            "(%d accepted / %d rejected by similarity filter) | "
            "Cache hits/misses: %d/%d",
            len(results),
            accepted,
            rejected,
            cache_hits,
            cache_misses,
        )
        return results

    def get_statistics(self, fragments: list[SplitFragment]) -> dict:
        """Return a summary dict for the generated fragments."""
        if not fragments:
            return {"total": 0, "unique_licenses": 0}
        return {
            "total": len(fragments),
            "unique_licenses": len({f.license_key for f in fragments}),
        }
