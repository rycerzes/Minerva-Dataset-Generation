"""Class Balancing for Nirjas comment classification.

Produces a balanced 2-class dataset for training Nirjas:
  Class 1 (``license_related``):
      sliding-window fragments, LLM-injected edge cases,
      truncated / corrupted license headers.
  Class 2 (``not_license_related``):
      hard-negative developer comments from the LLM generator.

The balancer accepts pre-computed artefacts from earlier pipeline stages
and assembles them into a single labelled collection with a configurable
target ratio (default 50 / 50).
"""

from __future__ import annotations

import logging
import random
import re
from typing import Literal, Optional

from pathlib import Path
from pydantic import BaseModel, Field

from .legal_structure_splitter import SplitFragment
from .llm_synthetic import AugmentedFragment
from .hard_negative_generator import HardNegativeSample

try:
    from ..fetchers.code_comments import CodeCommentSample
except ImportError:
    # Allow direct / standalone execution
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from fetchers.code_comments import CodeCommentSample  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class BalancedSample(BaseModel):
    """A single labelled sample in the Nirjas 2-class dataset."""

    model_config = {"protected_namespaces": ()}

    text: str
    label: Literal["license_related", "not_license_related"]
    source: str  # scancode / fossology / augmented / generated
    license_key: Optional[str] = None
    negative_type: Optional[str] = None  # only for not_license_related
    generation_method: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class BalancingConfig(BaseModel):
    """Knobs for the class-balancing step."""

    model_config = {"protected_namespaces": ()}

    target_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Desired fraction of license_related samples in the final dataset. "
            "0.5 means an even 50/50 split."
        ),
    )
    max_total_samples: Optional[int] = Field(
        default=None,
        description="Hard cap on total dataset size.  None = no cap.",
    )
    random_seed: int = 42
    include_truncated_headers: bool = Field(
        default=True,
        description="Generate truncated / corrupted license headers as positive samples.",
    )
    truncation_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of license_related pool to derive from truncated headers. "
            "Applied relative to the number of sliding-window fragments."
        ),
    )


def _truncate_fragment(text: str, rng: random.Random) -> str:
    """Return a truncated or corrupted version of a license fragment.

    Strategies (chosen at random):
      1. Keep only the first N% of characters.
      2. Keep only the last N% of characters.
      3. Remove a random contiguous span from the middle.
      4. Strip all whitespace normalisation (collapse to single line).
    """
    if len(text) < 20:
        return text

    strategy = rng.choice(["head", "tail", "middle_gap", "collapse_ws"])

    if strategy == "head":
        keep = rng.randint(30, 70)
        cutoff = max(10, len(text) * keep // 100)
        return text[:cutoff].rstrip()

    if strategy == "tail":
        keep = rng.randint(30, 70)
        cutoff = max(10, len(text) * keep // 100)
        return text[-cutoff:].lstrip()

    if strategy == "middle_gap":
        gap_size = rng.randint(len(text) // 5, len(text) // 2)
        gap_start = rng.randint(0, len(text) - gap_size)
        return (text[:gap_start] + " [...] " + text[gap_start + gap_size :]).strip()

    # collapse_ws
    return re.sub(r"\s+", " ", text).strip()


class NirjasClassBalancer:
    """Assemble and balance a 2-class dataset for Nirjas training.

    Inputs (all optional — supply whatever the pipeline has produced so far):
      * ``fragments``  — ``SplitFragment`` list from the sliding-window splitter
      * ``augmented``  — ``AugmentedFragment`` list from surgical LLM injection
      * ``hard_negatives`` — ``HardNegativeSample`` list from the hard-neg generator

    The balancer:
      1. Builds the **license_related** pool from fragments + augmented texts
         (+ optionally truncated / corrupted headers).
      2. Builds the **not_license_related** pool from LLM-generated
         hard negatives.
      3. Down-samples the larger pool (or up-samples via duplication) so that
         the final class ratio matches ``config.target_ratio``.
    """

    def __init__(self, config: Optional[BalancingConfig] = None):
        self.config = config or BalancingConfig()
        self._rng = random.Random(self.config.random_seed)

    def _build_license_related_pool(
        self,
        fragments: list[SplitFragment],
        augmented: list[AugmentedFragment],
    ) -> list[BalancedSample]:
        """Class 1 — license-related samples."""
        pool: list[BalancedSample] = []
        seen_texts: set[str] = set()

        # 1a. Raw sliding-window fragments
        for frag in fragments:
            text = frag.fragment_text.strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            pool.append(
                BalancedSample(
                    text=text,
                    label="license_related",
                    source=frag.source,
                    license_key=frag.license_key,
                    generation_method="sliding_window",
                )
            )

        # 1b. LLM-injected augmented fragments (edge cases with realistic entities)
        for aug in augmented:
            text = aug.augmented_text.strip()
            if not text or text in seen_texts:
                continue
            # Skip pass-through (text identical to original fragment)
            if text == aug.original_fragment.fragment_text.strip():
                continue
            seen_texts.add(text)
            pool.append(
                BalancedSample(
                    text=text,
                    label="license_related",
                    source="augmented",
                    license_key=aug.original_fragment.license_key,
                    generation_method="llm_injected",
                )
            )

        # 1c. Truncated / corrupted license headers
        if self.config.include_truncated_headers and fragments:
            n_truncated = max(1, int(len(fragments) * self.config.truncation_ratio))
            source_frags = self._rng.sample(fragments, min(n_truncated, len(fragments)))
            for frag in source_frags:
                truncated = _truncate_fragment(frag.fragment_text, self._rng)
                truncated = truncated.strip()
                if not truncated or truncated in seen_texts:
                    continue
                seen_texts.add(truncated)
                pool.append(
                    BalancedSample(
                        text=truncated,
                        label="license_related",
                        source="synthetic",
                        license_key=frag.license_key,
                        generation_method="truncated",
                        metadata={"corruption": "truncated_header"},
                    ),
                )

        return pool

    def _build_not_license_related_pool(
        self,
        hard_negatives: list[HardNegativeSample],
        code_comments: list[CodeCommentSample] | None = None,
    ) -> list[BalancedSample]:
        """Class 2 — not-license-related samples."""
        pool: list[BalancedSample] = []
        seen_texts: set[str] = set()

        # 2a. LLM-generated hard negatives
        for neg in hard_negatives:
            text = neg.text.strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            pool.append(
                BalancedSample(
                    text=text,
                    label="not_license_related",
                    source="generated",
                    license_key=neg.source_license,
                    negative_type=neg.negative_type,
                    generation_method=neg.generation_method,
                )
            )

        # 2b. Real code comments extracted from public source repositories
        for comment in code_comments or []:
            text = comment.text.strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            pool.append(
                BalancedSample(
                    text=text,
                    label="not_license_related",
                    source="code_corpus",
                    negative_type="generic_code_comment",
                    generation_method="extracted",
                    metadata={
                        "language": comment.language,
                        "comment_type": comment.comment_type,
                    },
                )
            )

        return pool

    @staticmethod
    def _downsample(
        pool: list[BalancedSample],
        target: int,
        rng: random.Random,
    ) -> list[BalancedSample]:
        """Randomly down-sample *pool* to *target* items."""
        if len(pool) <= target:
            return list(pool)
        return rng.sample(pool, target)

    @staticmethod
    def _upsample(
        pool: list[BalancedSample],
        target: int,
        rng: random.Random,
    ) -> list[BalancedSample]:
        """Duplicate random items from *pool* until it reaches *target* items."""
        if not pool:
            return []
        result = list(pool)
        while len(result) < target:
            result.append(rng.choice(pool))
        return result[:target]

    def _apply_balance(
        self,
        license_pool: list[BalancedSample],
        non_license_pool: list[BalancedSample],
    ) -> list[BalancedSample]:
        """Adjust pool sizes to achieve the configured target ratio."""
        ratio = self.config.target_ratio  # fraction that should be license_related
        n_lic = len(license_pool)
        n_neg = len(non_license_pool)
        total_available = n_lic + n_neg

        if total_available == 0:
            return []

        # Determine desired counts
        if self.config.max_total_samples is not None:
            total = min(self.config.max_total_samples, total_available)
        else:
            total = total_available

        target_lic = max(1, round(total * ratio))
        target_neg = max(1, total - target_lic)

        # Adjust pools
        if n_lic >= target_lic:
            license_final = self._downsample(license_pool, target_lic, self._rng)
        else:
            license_final = self._upsample(license_pool, target_lic, self._rng)

        if n_neg >= target_neg:
            non_license_final = self._downsample(
                non_license_pool, target_neg, self._rng
            )
        else:
            non_license_final = self._upsample(non_license_pool, target_neg, self._rng)

        combined = license_final + non_license_final
        self._rng.shuffle(combined)
        return combined

    def balance(
        self,
        fragments: Optional[list[SplitFragment]] = None,
        augmented: Optional[list[AugmentedFragment]] = None,
        hard_negatives: Optional[list[HardNegativeSample]] = None,
        code_comments: Optional[list[CodeCommentSample]] = None,
    ) -> list[BalancedSample]:
        """Build and balance the 2-class Nirjas dataset.

        Parameters
        ----------
        fragments:
            Sliding-window ``SplitFragment`` objects (Class 1 source).
        augmented:
            LLM-augmented ``AugmentedFragment`` objects (Class 1 source).
        hard_negatives:
            LLM-generated ``HardNegativeSample`` objects (Class 2 source).
        code_comments:
            Real code comments from a public corpus (Class 2 source).

        Returns
        -------
        list[BalancedSample]
            Shuffled, balanced dataset ready for export.
        """
        fragments = fragments or []
        augmented = augmented or []
        hard_negatives = hard_negatives or []

        license_pool = self._build_license_related_pool(fragments, augmented)
        non_license_pool = self._build_not_license_related_pool(
            hard_negatives, code_comments
        )

        logger.info(
            "Pre-balance pool sizes — license_related: %d, not_license_related: %d",
            len(license_pool),
            len(non_license_pool),
        )

        balanced = self._apply_balance(license_pool, non_license_pool)

        logger.info(
            "Post-balance dataset size: %d (ratio target: %.2f)",
            len(balanced),
            self.config.target_ratio,
        )

        return balanced

    def get_statistics(self, samples: list[BalancedSample]) -> dict:
        """Return summary statistics about a balanced dataset."""
        if not samples:
            return {
                "total": 0,
                "license_related": 0,
                "not_license_related": 0,
                "actual_ratio": 0.0,
                "by_source": {},
                "by_generation_method": {},
                "by_negative_type": {},
            }

        n_lic = sum(1 for s in samples if s.label == "license_related")
        n_neg = sum(1 for s in samples if s.label == "not_license_related")

        by_source: dict[str, int] = {}
        by_method: dict[str, int] = {}
        by_neg_type: dict[str, int] = {}

        for s in samples:
            by_source[s.source] = by_source.get(s.source, 0) + 1
            if s.generation_method:
                by_method[s.generation_method] = (
                    by_method.get(s.generation_method, 0) + 1
                )
            if s.negative_type:
                by_neg_type[s.negative_type] = by_neg_type.get(s.negative_type, 0) + 1

        return {
            "total": len(samples),
            "license_related": n_lic,
            "not_license_related": n_neg,
            "actual_ratio": n_lic / len(samples) if samples else 0.0,
            "by_source": by_source,
            "by_generation_method": by_method,
            "by_negative_type": by_neg_type,
        }

    def print_statistics(self, samples: list[BalancedSample]) -> None:
        """Pretty-print dataset statistics to stdout."""
        stats = self.get_statistics(samples)
        print("=" * 60)
        print("NIRJAS CLASS BALANCING REPORT")
        print("=" * 60)
        print(f"Total samples:          {stats['total']:,}")
        print(f"  license_related:      {stats['license_related']:,}")
        print(f"  not_license_related:  {stats['not_license_related']:,}")
        print(f"  actual ratio:         {stats['actual_ratio']:.4f}")
        print()
        print("By source:")
        for src, count in sorted(stats["by_source"].items()):
            print(f"  {src:25s} {count:,}")
        print()
        print("By generation method:")
        for method, count in sorted(stats["by_generation_method"].items()):
            print(f"  {method:25s} {count:,}")
        if stats["by_negative_type"]:
            print()
            print("Negative types:")
            for ntype, count in sorted(stats["by_negative_type"].items()):
                print(f"  {ntype:25s} {count:,}")
        print("=" * 60)


def balance_nirjas_dataset(
    fragments: Optional[list[SplitFragment]] = None,
    augmented: Optional[list[AugmentedFragment]] = None,
    hard_negatives: Optional[list[HardNegativeSample]] = None,
    code_comments: Optional[list[CodeCommentSample]] = None,
    target_ratio: float = 0.5,
    max_total_samples: Optional[int] = None,
    random_seed: int = 42,
    include_truncated_headers: bool = True,
    truncation_ratio: float = 0.1,
) -> list[BalancedSample]:
    """One-shot convenience function to build a balanced Nirjas dataset.

    See :class:`NirjasClassBalancer` for parameter details.
    """
    config = BalancingConfig(
        target_ratio=target_ratio,
        max_total_samples=max_total_samples,
        random_seed=random_seed,
        include_truncated_headers=include_truncated_headers,
        truncation_ratio=truncation_ratio,
    )
    balancer = NirjasClassBalancer(config)
    return balancer.balance(
        fragments=fragments,
        augmented=augmented,
        hard_negatives=hard_negatives,
        code_comments=code_comments,
    )
