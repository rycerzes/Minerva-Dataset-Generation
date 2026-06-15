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
    from .legal_structure_splitter import SplitFragment, LEGAL_PLACEHOLDER_PATTERNS
    from ..config import LLMConfig, RateLimiter
    from .llm_cache import LLMCache
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from augmentation.legal_structure_splitter import (
        SplitFragment,
        LEGAL_PLACEHOLDER_PATTERNS,
    )
    from config import LLMConfig, RateLimiter
    from augmentation.llm_cache import LLMCache


class PlaceholderContext(BaseModel):
    model_config = {"protected_namespaces": ()}

    placeholder: str
    placeholder_type: str
    suggested_values: list[str] = Field(default_factory=list)


class AugmentedFragment(BaseModel):
    model_config = {"protected_namespaces": ()}

    original_fragment: SplitFragment
    augmented_text: str
    filled_placeholders: dict[str, str]
    augmentation_method: Literal["llm_injected"]
    llm_model_used: str


class LLMSurgicalInjectionError(Exception):
    pass


class SurgicalLLMInjector:
    PLACEHOLDER_TYPE_MAPPING = {
        "owner": ["owner", "copyright holder", "copyright_owner", "author"],
        "year": ["year", "date", "copyright_year"],
        "name": ["name", "full_name", "author_name"],
        "organization": ["organization", "org", "company"],
        "project": ["project", "software", "program"],
        "version": ["version", "ver"],
        "license": ["license", "license_name"],
    }

    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 2.0

    CACHE_NAMESPACE: str = "surgical_injection"

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache: Optional[LLMCache] = None,
    ):
        self.config = config or LLMConfig()
        self._rate_limiter = RateLimiter(self.config.rpm)
        self._llm_client = None
        self._cache = cache

    def _get_llm_client(self):
        if self._llm_client is None:
            try:
                import litellm

                litellm.drop_params = True
                self._llm_client = litellm
            except ImportError as e:
                raise ImportError(
                    "litellm is required for LLM-based injection. "
                    "Install with: pip install litellm"
                ) from e

        return self._llm_client

    def _classify_placeholder(self, placeholder: str) -> str:
        placeholder_lower = placeholder.lower()
        for ptype, keywords in self.PLACEHOLDER_TYPE_MAPPING.items():
            if any(kw in placeholder_lower for kw in keywords):
                return ptype

        for pattern in LEGAL_PLACEHOLDER_PATTERNS:
            if re.search(pattern, placeholder, re.IGNORECASE):
                if "owner" in placeholder_lower or "copyright" in placeholder_lower:
                    return "owner"
                elif "year" in placeholder_lower or "date" in placeholder_lower:
                    return "year"
                elif "name" in placeholder_lower:
                    return "name"
                elif "org" in placeholder_lower:
                    return "organization"
                elif "project" in placeholder_lower or "software" in placeholder_lower:
                    return "project"
                elif "version" in placeholder_lower:
                    return "version"
                elif "license" in placeholder_lower:
                    return "license"

        return "generic"

    def _extract_placeholders_from_text(self, text: str) -> list[PlaceholderContext]:
        placeholders = []
        seen = set()

        for pattern in LEGAL_PLACEHOLDER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                placeholder = match.group(0)
                if placeholder not in seen:
                    seen.add(placeholder)
                    ptype = self._classify_placeholder(placeholder)
                    placeholders.append(
                        PlaceholderContext(
                            placeholder=placeholder,
                            placeholder_type=ptype,
                        )
                    )

        return placeholders

    def _build_injection_prompt(
        self, fragment: SplitFragment, placeholders: list[PlaceholderContext]
    ) -> str:
        placeholder_list = "\n".join(
            f"- {p.placeholder} (type: {p.placeholder_type})" for p in placeholders
        )

        prompt = f"""You are a legal text augmentation system. Given a license text fragment with placeholder variables, replace them with realistic, contextually appropriate values.

Original license: {fragment.license_key}
Source: {fragment.source}

Text fragment:
```
{fragment.fragment_text}
```

Placeholders to replace:
{placeholder_list}

Requirements:
1. Replace each placeholder with a realistic value appropriate for a software license
2. Use varied but realistic names (companies, authors, years, project names)
3. Maintain the legal tone and structure of the text
4. Ensure replaced values look natural and not obviously synthetic
5. Keep the same format as the original placeholder

Output ONLY the augmented text, nothing else. Do not add explanations or comments."""
        return prompt

    def _call_llm(self, prompt: str) -> str:
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
                augmented_text = response["choices"][0]["message"]["content"].strip()
                if not augmented_text or len(augmented_text) < 10:
                    raise LLMSurgicalInjectionError(
                        f"LLM returned invalid response: {augmented_text!r}"
                    )
                return augmented_text
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

        raise LLMSurgicalInjectionError(
            f"LLM API call failed after {self.MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _cache_key(fragment: SplitFragment) -> str:
        """Deterministic cache key for a fragment."""
        import hashlib

        blob = (
            f"{fragment.license_key}|{fragment.fragment_index}|{fragment.fragment_text}"
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def augment_fragment(self, fragment: SplitFragment) -> AugmentedFragment:
        if not fragment.placeholders:
            return AugmentedFragment(
                original_fragment=fragment,
                augmented_text=fragment.fragment_text,
                filled_placeholders={},
                augmentation_method="llm_injected",
                llm_model_used=self.config.model,
            )

        placeholders = self._extract_placeholders_from_text(fragment.fragment_text)

        if not placeholders:
            return AugmentedFragment(
                original_fragment=fragment,
                augmented_text=fragment.fragment_text,
                filled_placeholders={},
                augmentation_method="llm_injected",
                llm_model_used=self.config.model,
            )

        # --- Cache check ---
        cache_key = self._cache_key(fragment)
        if self._cache is not None:
            cached = self._cache.get(self.CACHE_NAMESPACE, cache_key)
            if cached is not None:
                return AugmentedFragment(**cached)

        prompt = self._build_injection_prompt(fragment, placeholders)
        augmented_text = self._call_llm(prompt)

        filled = {p.placeholder: "llm_generated" for p in placeholders}
        result = AugmentedFragment(
            original_fragment=fragment,
            augmented_text=augmented_text,
            filled_placeholders=filled,
            augmentation_method="llm_injected",
            llm_model_used=self.config.model,
        )

        # --- Persist to cache ---
        if self._cache is not None:
            self._cache.set(self.CACHE_NAMESPACE, cache_key, result.model_dump())

        return result

    def augment_batch(self, fragments: list[SplitFragment]) -> list[AugmentedFragment]:
        results = []
        for fragment in fragments:
            result = self.augment_fragment(fragment)
            results.append(result)
        return results

    def augment_dataset(
        self, fragments: list[SplitFragment]
    ) -> list[AugmentedFragment]:
        return self.augment_batch(fragments)

    def get_augmentation_statistics(
        self, augmented_fragments: list[AugmentedFragment]
    ) -> dict:
        if not augmented_fragments:
            return {
                "total": 0,
                "llm_injected": 0,
            }

        return {
            "total": len(augmented_fragments),
            "llm_injected": len(augmented_fragments),
        }


def surgical_llm_injection(
    fragments: list[SplitFragment],
    config: Optional[LLMConfig] = None,
    cache: Optional[LLMCache] = None,
) -> list[AugmentedFragment]:
    injector = SurgicalLLMInjector(config, cache=cache)
    return injector.augment_dataset(fragments)
