from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from ..config import LLMConfig, RateLimiter
    from .llm_cache import LLMCache
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import LLMConfig, RateLimiter
    from augmentation.llm_cache import LLMCache

# Section markers used in the combined prompt / response
_SECTION_MARKERS = {
    "license_discussion": "=== LICENSE_DISCUSSION ===",
    "todo_fixme": "=== TODO_FIXME ===",
    "commented_code": "=== COMMENTED_CODE ===",
    "copyright_discussion": "=== COPYRIGHT_DISCUSSION ===",
}

_ORDERED_TYPES = list(_SECTION_MARKERS.keys())


class HardNegativeSample(BaseModel):
    model_config = {"protected_namespaces": ()}

    text: str
    negative_type: str
    generation_method: Literal["llm_generated"]
    source_license: Optional[str] = None
    llm_model_used: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class HardNegativeGeneratorError(Exception):
    pass


class HardNegativeGenerator:
    """Generate hard-negative samples for license classification training.

    Optimisations over the original implementation
    -----------------------------------------------
    * **Combined prompt** — all 4 negative categories are requested in a
      single LLM call per license (was 4 calls).
    * **Disk cache** — results are persisted so re-runs skip already-
      completed licenses.
    * **Configurable subset** — callers can limit how many licenses are
      processed via ``max_licenses`` in ``generate_batch``.
    """

    # Category weights for distribution across negative types
    LICENSE_DISCUSSION_WEIGHT: float = 0.3
    TODO_FIXME_WEIGHT: float = 0.3
    COMMENTED_CODE_WEIGHT: float = 0.2
    COPYRIGHT_DISCUSSION_WEIGHT: float = 0.2

    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 2.0

    CACHE_NAMESPACE: str = "hard_negatives"

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        samples_per_category: int = 5,
        cache: Optional[LLMCache] = None,
    ):
        self.config = config or LLMConfig()
        self.samples_per_category = samples_per_category
        self._rate_limiter = RateLimiter(self.config.rpm)
        self._llm_client = None
        self._cache = cache

    # LLM client

    def _get_llm_client(self):
        import litellm

        litellm.drop_params = True
        self._llm_client = litellm
        return self._llm_client

    # Combined prompt (Option 1: 4 calls → 1)

    def _build_combined_prompt(
        self, license_key: str, context: Optional[str] = None
    ) -> str:
        n = self.samples_per_category
        ctx = f"\n(Optional context: {context})" if context else ""

        prompt = f"""You are a developer and code-comment generator. Generate realistic hard-negative examples for the "{license_key}" license across 4 categories.{ctx}

Output EXACTLY {n} items per category. Separate categories using the markers shown below.
Each item should be on its own line. Do not number them or add extra formatting.

{_SECTION_MARKERS["license_discussion"]}
Generate {n} informal developer discussions about the "{license_key}" license in code comments.
Style: casual, opinionated, uncertain. Mix comment styles (//, #, /* */). 1-3 sentences each.
Do NOT include actual license text — only discussions, questions, opinions.

{_SECTION_MARKERS["todo_fixme"]}
Generate {n} realistic TODO/FIXME comments related to "{license_key}" licensing tasks.
Include practical tasks: updating headers, checking compliance, adding licenses.
Mix TODO and FIXME prefixes. Include realistic file paths or function names. 1-2 lines each.

{_SECTION_MARKERS["commented_code"]}
Generate {n} examples of code commented out or disabled due to "{license_key}" licensing issues.
Show commented-out code blocks with license-related explanations.
Mix comment styles (#, //, /* */). Include realistic function names and variables.

{_SECTION_MARKERS["copyright_discussion"]}
Generate {n} informal developer comments discussing copyright issues for "{license_key}".
Questions about ownership, updating years, adding contributors.
Do NOT write actual license text — only discussions. Mix comment styles."""
        return prompt

    # Response parsing

    @staticmethod
    def _parse_combined_response(
        text: str, license_key: str, model: str
    ) -> list[HardNegativeSample]:
        """Parse a combined LLM response into individual ``HardNegativeSample``s."""
        results: list[HardNegativeSample] = []

        # Locate each section marker
        marker_positions: list[tuple[int, str]] = []
        for neg_type, marker in _SECTION_MARKERS.items():
            pos = text.find(marker)
            if pos != -1:
                marker_positions.append((pos, neg_type))

        marker_positions.sort()

        # Extract section text between markers
        sections: dict[str, str] = {}
        for i, (pos, neg_type) in enumerate(marker_positions):
            start = pos + len(_SECTION_MARKERS[neg_type])
            end = (
                marker_positions[i + 1][0]
                if i + 1 < len(marker_positions)
                else len(text)
            )
            sections[neg_type] = text[start:end].strip()

        # Parse each section into individual lines → samples
        for neg_type in _ORDERED_TYPES:
            section_text = sections.get(neg_type, "")
            if not section_text:
                continue

            lines = [line.strip() for line in section_text.split("\n") if line.strip()]
            # Strip leading numbering (e.g. "1. " or "2) ")
            lines = [re.sub(r"^\d+[\.\)]\s*", "", ln) for ln in lines if len(ln) > 5]
            lines = [ln for ln in lines if len(ln) > 5]

            for line in lines:
                results.append(
                    HardNegativeSample(
                        text=line,
                        negative_type=neg_type,
                        generation_method="llm_generated",
                        source_license=license_key,
                        llm_model_used=model,
                    )
                )

        return results

    # LLM call with retries

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the raw response text."""
        client = self._get_llm_client()
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            self._rate_limiter.acquire()
            try:
                response = client.completion(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    api_base=self.config.api_base_url,
                    api_key=self.config.api_key,
                    custom_llm_provider="openai",
                )
                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise HardNegativeGeneratorError("LLM returned None content")
                text = content.strip()
                if not text:
                    raise HardNegativeGeneratorError("LLM returned empty response")
                return text
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait = self.RETRY_BACKOFF_BASE**attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt,
                        self.MAX_RETRIES,
                        wait,
                        e,
                    )
                    time.sleep(wait)

        raise HardNegativeGeneratorError(
            f"LLM API call failed after {self.MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    # Per-license generation (combined prompt + cache)

    def generate_for_license(
        self, license_key: str, context: Optional[str] = None
    ) -> list[HardNegativeSample]:
        """Generate hard negatives for a single license (1 LLM call).

        Results are read from / written to the disk cache when available.
        """
        # --- Cache check ---
        if self._cache is not None:
            cached = self._cache.get(self.CACHE_NAMESPACE, license_key)
            if cached is not None:
                return [HardNegativeSample(**s) for s in cached]

        # --- LLM call (single combined prompt) ---
        prompt = self._build_combined_prompt(license_key, context)
        raw_response = self._call_llm(prompt)
        samples = self._parse_combined_response(
            raw_response, license_key, self.config.model
        )

        # --- Trim to requested counts per category ---
        weight_map = {
            "license_discussion": self.LICENSE_DISCUSSION_WEIGHT,
            "todo_fixme": self.TODO_FIXME_WEIGHT,
            "commented_code": self.COMMENTED_CODE_WEIGHT,
            "copyright_discussion": self.COPYRIGHT_DISCUSSION_WEIGHT,
        }

        trimmed: list[HardNegativeSample] = []
        for neg_type, weight in weight_map.items():
            count = max(1, int(self.samples_per_category * weight))
            typed = [s for s in samples if s.negative_type == neg_type]
            trimmed.extend(typed[:count])

        # --- Persist to cache ---
        if self._cache is not None:
            self._cache.set(
                self.CACHE_NAMESPACE,
                license_key,
                [s.model_dump() for s in trimmed],
            )

        return trimmed

    # Legacy single-category helpers (backward compat for tests)

    def generate_license_discussion(
        self, license_key: str, context: Optional[str] = None
    ) -> list[HardNegativeSample]:
        prompt = self._build_combined_prompt(license_key, context)
        raw = self._call_llm(prompt)
        all_samples = self._parse_combined_response(raw, license_key, self.config.model)
        return [s for s in all_samples if s.negative_type == "license_discussion"]

    def generate_todo_fixme(self, license_key: str) -> list[HardNegativeSample]:
        prompt = self._build_combined_prompt(license_key)
        raw = self._call_llm(prompt)
        all_samples = self._parse_combined_response(raw, license_key, self.config.model)
        return [s for s in all_samples if s.negative_type == "todo_fixme"]

    def generate_commented_code(self, license_key: str) -> list[HardNegativeSample]:
        prompt = self._build_combined_prompt(license_key)
        raw = self._call_llm(prompt)
        all_samples = self._parse_combined_response(raw, license_key, self.config.model)
        return [s for s in all_samples if s.negative_type == "commented_code"]

    def generate_copyright_discussion(
        self,
        license_key: str,
    ) -> list[HardNegativeSample]:
        prompt = self._build_combined_prompt(license_key)
        raw = self._call_llm(prompt)
        all_samples = self._parse_combined_response(raw, license_key, self.config.model)
        return [s for s in all_samples if s.negative_type == "copyright_discussion"]

    # Batch generation

    def generate_batch(
        self,
        license_keys: list[str],
        max_licenses: Optional[int] = None,
    ) -> list[HardNegativeSample]:
        """Generate hard negatives for a list of licenses.

        Parameters
        ----------
        license_keys:
            All license keys to process.
        max_licenses:
            If set, only process the first *max_licenses* keys
            (Option 3 — subset selection).
        """
        keys = license_keys[:max_licenses] if max_licenses else license_keys
        total = len(keys)
        results: list[HardNegativeSample] = []

        for i, license_key in enumerate(keys, 1):
            is_cached = self._cache is not None and self._cache.has(
                self.CACHE_NAMESPACE, license_key
            )
            if i % 50 == 1 or i == total:
                tag = " (cached)" if is_cached else ""
                print(f"  [{i}/{total}] {license_key}{tag}")
            results.extend(self.generate_for_license(license_key))

        return results

    # Statistics

    def get_statistics(self, samples: list[HardNegativeSample]) -> dict:
        if not samples:
            return {"total": 0, "by_type": {}, "by_method": {}}

        by_type: dict[str, int] = {}
        by_method: dict[str, int] = {}

        for sample in samples:
            by_type[sample.negative_type] = by_type.get(sample.negative_type, 0) + 1
            by_method[sample.generation_method] = (
                by_method.get(sample.generation_method, 0) + 1
            )

        stats: dict = {
            "total": len(samples),
            "by_type": by_type,
            "by_method": by_method,
        }

        if self._cache:
            stats["cache"] = self._cache.stats

        return stats


# Module-level convenience function


def generate_hard_negatives(
    license_keys: list[str],
    config: Optional[LLMConfig] = None,
    samples_per_category: int = 5,
    cache: Optional[LLMCache] = None,
    max_licenses: Optional[int] = None,
) -> list[HardNegativeSample]:
    generator = HardNegativeGenerator(
        config=config,
        samples_per_category=samples_per_category,
        cache=cache,
    )
    return generator.generate_batch(license_keys, max_licenses=max_licenses)
