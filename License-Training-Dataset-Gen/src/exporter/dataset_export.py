"""Dataset Exporter — Phase 3.1.

Exports the compiled Atarashi and Nirjas datasets to Parquet-backed
Hugging Face ``datasets.Dataset`` format.  Each exported dataset is a
directory containing:

* ``data-00000-of-00001.parquet`` — the data shard(s)
* ``dataset_info.json`` — HF metadata (features, description, …)
* ``state.json`` — split info

The datasets can be loaded back with:

    >>> from datasets import load_from_disk
    >>> ds = load_from_disk("output/atarashi")

Exports
-------
* **Atarashi dataset** — licence texts with labels for similarity-based
  training.  Columns: ``license_key``, ``text``, ``source``.
* **Nirjas dataset** — comment texts with 2-class labels.  Columns:
  ``text``, ``label``, ``source``, ``negative_type``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from pydantic import BaseModel, Field

try:
    from ..builder.augmented_merge import AtarashiSample, NirjasSample
except ImportError:  # pragma: no cover — direct-script execution
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from builder.augmented_merge import AtarashiSample, NirjasSample

logger = logging.getLogger(__name__)


def _split_of(text: str, val_ratio: float, test_ratio: float) -> str:
    """Assign a split from the text's content hash — deterministic and frozen.

    A sample's split is a pure function of its text, so the same text always
    lands in the same split no matter what else is in the pool. This keeps the
    test set stable across pipeline re-runs (benchmark numbers stay comparable)
    and prevents identical/upsampled duplicate texts from leaking across the
    train/test boundary. Stratification is preserved statistically: the hash is
    uniform and independent of label/license, so each group splits at the same
    ratios over any reasonably sized pool.
    """
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) / (1 << 128)
    if h < test_ratio:
        return "test"
    if h < test_ratio + val_ratio:
        return "validation"
    return "train"


# Configuration


class ExportConfig(BaseModel):
    """Configuration knobs for the dataset exporter."""

    model_config = {"protected_namespaces": ()}

    output_dir: str = Field(
        default="output",
        description="Root directory for all exported datasets.",
    )
    atarashi_dir_name: str = Field(
        default="atarashi",
        description="Sub-directory name for the Atarashi dataset.",
    )
    nirjas_dir_name: str = Field(
        default="nirjas",
        description="Sub-directory name for the Nirjas dataset.",
    )
    train_split_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of samples assigned to the 'train' split.",
    )
    validation_split_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of samples assigned to the 'validation' split. The "
            "remaining fraction after train+validation goes to 'test'. Set to "
            "0.0 for a train/test export."
        ),
    )
    random_seed: int = 42
    write_statistics: bool = Field(
        default=True,
        description="Write a ``statistics.json`` alongside each exported dataset.",
    )
    clean_splits: bool = Field(
        default=True,
        description=(
            "Quarantine junk fragments and remove cross-split leakage before "
            "writing. The sliding-window splitter emits overlapping fragments of "
            "one license which the content-hash split scatters across "
            "train/validation/test; without this the export leaks and evaluation "
            "measures memorisation. Disable only to reproduce a pre-cleaning build."
        ),
    )
    clean_min_chars: int = Field(
        default=60,
        ge=0,
        description=(
            "Drop fragments shorter than this many normalized characters. These "
            "are generic boilerplate windows labelled to one license though the "
            "phrasing recurs across many, so the label is not inferable. 0 disables."
        ),
    )
    clean_near_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Jaccard threshold for MinHash cross-split near-dedup.",
    )


# Export result


class ExportResult(BaseModel):
    """Summary returned after an export run."""

    model_config = {"protected_namespaces": ()}

    atarashi_path: Optional[str] = None
    nirjas_path: Optional[str] = None
    atarashi_total: int = 0
    atarashi_train: int = 0
    atarashi_validation: int = 0
    atarashi_test: int = 0
    nirjas_total: int = 0
    nirjas_train: int = 0
    nirjas_validation: int = 0
    nirjas_test: int = 0


# HF Features schemas

ATARASHI_FEATURES = Features(
    {
        "license_key": Value("string"),
        "text": Value("string"),
        "source": Value("string"),
    }
)

NIRJAS_FEATURES = Features(
    {
        "text": Value("string"),
        "label": ClassLabel(names=["license_related", "not_license_related"]),
        "source": Value("string"),
        "negative_type": Value("string"),
    }
)


# Core exporter


class DatasetExporter:
    """Export Atarashi and Nirjas datasets as HF ``Dataset`` on disk.

    Parameters
    ----------
    config : ExportConfig, optional
        Export configuration.  Uses defaults if omitted.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        if self.config.train_split_ratio + self.config.validation_split_ratio > 1.0:
            raise ValueError(
                "train_split_ratio + validation_split_ratio must be <= 1.0"
            )

    @property
    def _test_split_ratio(self) -> float:
        """Fraction assigned to test after train and validation."""
        return max(
            0.0,
            1.0 - self.config.train_split_ratio - self.config.validation_split_ratio,
        )

    # -- Atarashi -----------------------------------------------------------

    @staticmethod
    def _atarashi_to_dict(samples: list[AtarashiSample]) -> dict[str, list]:
        """Convert ``AtarashiSample`` list into a column-oriented dict."""
        return {
            "license_key": [s.license_key for s in samples],
            "text": [s.text for s in samples],
            "source": [s.source for s in samples],
        }

    def _build_atarashi_dataset(
        self,
        samples: list[AtarashiSample],
    ) -> DatasetDict | Dataset:
        """Build an HF Atarashi DatasetDict with per-license splitting.

        A global random split can place all samples for a rare ``license_key`` in
        validation/test. For multi-class license identification that creates
        impossible evaluation labels. Split each license independently and keep
        singleton classes in train only, ensuring every validation/test label is
        seen in train.
        """
        if self.config.train_split_ratio >= 1.0:
            ds = Dataset.from_dict(
                self._atarashi_to_dict(samples),
                features=ATARASHI_FEATURES,
            )
            return DatasetDict({"train": ds})

        by_key: dict[str, list[AtarashiSample]] = defaultdict(list)
        for sample in samples:
            by_key[sample.license_key].append(sample)

        train: list[AtarashiSample] = []
        validation: list[AtarashiSample] = []
        test: list[AtarashiSample] = []
        val_ratio = self.config.validation_split_ratio
        test_ratio = self._test_split_ratio

        for group in by_key.values():
            # Content-hash split (frozen across re-runs), applied per license so
            # the learnability guard below can run on each class independently.
            g_train, g_val, g_test = [], [], []
            g_buckets = {"train": g_train, "validation": g_val, "test": g_test}
            for sample in group:
                g_buckets[_split_of(sample.text, val_ratio, test_ratio)].append(sample)

            # Every validation/test label must be seen in train, else its
            # metrics are impossible. If the hash left this license with no
            # train sample, fold everything back into train.
            if not g_train:
                train.extend(group)
                continue

            train.extend(g_train)
            validation.extend(g_val)
            test.extend(g_test)

        result = DatasetDict(
            {
                "train": Dataset.from_dict(
                    self._atarashi_to_dict(train),
                    features=ATARASHI_FEATURES,
                )
            }
        )
        if validation:
            result["validation"] = Dataset.from_dict(
                self._atarashi_to_dict(validation),
                features=ATARASHI_FEATURES,
            )
        if test:
            result["test"] = Dataset.from_dict(
                self._atarashi_to_dict(test),
                features=ATARASHI_FEATURES,
            )
        return result

    # -- Nirjas -------------------------------------------------------------

    @staticmethod
    def _nirjas_to_dict(samples: list[NirjasSample]) -> dict[str, list]:
        """Convert ``NirjasSample`` list into a column-oriented dict."""
        return {
            "text": [s.text for s in samples],
            "label": [s.label for s in samples],
            "source": [s.source for s in samples],
            "negative_type": [s.negative_type or "" for s in samples],
        }

    def _build_nirjas_dataset(
        self,
        samples: list[NirjasSample],
    ) -> DatasetDict | Dataset:
        """Build a stratified HF Nirjas DatasetDict.

        Splits are stratified by binary label so validation/test preserve the
        license-related vs non-license-related ratio.
        """
        if self.config.train_split_ratio >= 1.0:
            ds = Dataset.from_dict(
                self._nirjas_to_dict(samples),
                features=NIRJAS_FEATURES,
            )
            return DatasetDict({"train": ds})

        val_ratio = self.config.validation_split_ratio
        test_ratio = self._test_split_ratio

        train: list[NirjasSample] = []
        validation: list[NirjasSample] = []
        test: list[NirjasSample] = []
        buckets = {"train": train, "validation": validation, "test": test}
        for sample in samples:
            buckets[_split_of(sample.text, val_ratio, test_ratio)].append(sample)

        result = DatasetDict(
            {
                "train": Dataset.from_dict(
                    self._nirjas_to_dict(train),
                    features=NIRJAS_FEATURES,
                )
            }
        )
        if validation:
            result["validation"] = Dataset.from_dict(
                self._nirjas_to_dict(validation),
                features=NIRJAS_FEATURES,
            )
        if test:
            result["test"] = Dataset.from_dict(
                self._nirjas_to_dict(test),
                features=NIRJAS_FEATURES,
            )
        return result

    # -- Statistics ---------------------------------------------------------

    def _atarashi_statistics(self, ds: DatasetDict) -> dict:
        """Compute summary statistics for the Atarashi export."""
        train_keys = set(ds["train"]["license_key"]) if "train" in ds else set()
        stats: dict = {"splits": {}}
        total = 0
        all_keys: set[str] = set()
        source_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}

        for split_name, split_ds in ds.items():
            n = len(split_ds)
            total += n
            keys = set(split_ds["license_key"])
            all_keys.update(keys)
            for key in split_ds["license_key"]:
                class_counts[key] = class_counts.get(key, 0) + 1
            split_sources: dict[str, int] = {}
            for src in split_ds["source"]:
                split_sources[src] = split_sources.get(src, 0) + 1
                source_counts[src] = source_counts.get(src, 0) + 1
            stats["splits"][split_name] = {
                "samples": n,
                "unique_licenses": len(keys),
                "by_source": split_sources,
                "labels_not_in_train": len(keys - train_keys)
                if split_name != "train"
                else 0,
            }

        stats["split_policy"] = {
            "train_ratio": self.config.train_split_ratio,
            "validation_ratio": self.config.validation_split_ratio,
            "test_ratio": self._test_split_ratio,
            "per_license": True,
            "assignment": "content_hash_frozen",
            "learnability_guard": "license_kept_in_train_if_no_train_sample",
        }
        stats["total_samples"] = total
        stats["total_unique_licenses"] = len(all_keys)
        stats["by_source"] = source_counts
        stats["classes_with_1_sample"] = sum(1 for v in class_counts.values() if v == 1)
        stats["classes_with_le_3_samples"] = sum(
            1 for v in class_counts.values() if v <= 3
        )
        stats["classes_with_le_5_samples"] = sum(
            1 for v in class_counts.values() if v <= 5
        )
        return stats

    def _nirjas_statistics(self, ds: DatasetDict) -> dict:
        """Compute summary statistics for the Nirjas export."""
        stats: dict = {"splits": {}}
        total = 0
        label_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        neg_type_counts: dict[str, int] = {}

        label_names = ds[list(ds.keys())[0]].features["label"].names

        for split_name, split_ds in ds.items():
            n = len(split_ds)
            total += n
            split_labels: dict[str, int] = {}
            split_sources: dict[str, int] = {}
            split_neg: dict[str, int] = {}
            for i in range(n):
                row = split_ds[i]
                label_str = (
                    label_names[row["label"]]
                    if isinstance(row["label"], int)
                    else row["label"]
                )
                split_labels[label_str] = split_labels.get(label_str, 0) + 1
                label_counts[label_str] = label_counts.get(label_str, 0) + 1
                split_sources[row["source"]] = split_sources.get(row["source"], 0) + 1
                source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
                neg = row.get("negative_type", "")
                if neg:
                    split_neg[neg] = split_neg.get(neg, 0) + 1
                    neg_type_counts[neg] = neg_type_counts.get(neg, 0) + 1
            stats["splits"][split_name] = {
                "samples": n,
                "by_label": split_labels,
                "by_source": split_sources,
                "by_negative_type": split_neg,
            }

        stats["split_policy"] = {
            "train_ratio": self.config.train_split_ratio,
            "validation_ratio": self.config.validation_split_ratio,
            "test_ratio": self._test_split_ratio,
            "stratify_by": "label",
            "assignment": "content_hash_frozen",
        }
        stats["total_samples"] = total
        stats["by_label"] = label_counts
        stats["by_source"] = source_counts
        stats["by_negative_type"] = neg_type_counts
        return stats

    # -- Split hygiene ------------------------------------------------------

    def _clean(self, ds_dict, label_column: str | None):
        """Quarantine junk and de-leak splits before writing. See ``cleaning``."""
        if not self.config.clean_splits or len(ds_dict) < 2:
            return ds_dict
        try:
            from exporter.cleaning import clean
        except ImportError:  # running as a script, package root not on sys.path
            from src.exporter.cleaning import clean  # type: ignore[no-redef]
        cleaned, report = clean(
            ds_dict,
            min_chars=self.config.clean_min_chars,
            near=True,
            threshold=self.config.clean_near_threshold,
            label_column=label_column,
        )
        logger.info(
            "split hygiene: quarantined %s, deduped %s, %s -> %s",
            report["quarantined"], report["deduped"], report["before"], report["after"],
        )
        return cleaned

    # -- Main export --------------------------------------------------------

    def export(
        self,
        atarashi_samples: Optional[list[AtarashiSample]] = None,
        nirjas_samples: Optional[list[NirjasSample]] = None,
    ) -> ExportResult:
        """Export one or both datasets to disk.

        Parameters
        ----------
        atarashi_samples :
            Atarashi licence-similarity samples from ``AugmentedMerger``.
        nirjas_samples :
            Nirjas 2-class comment-classification samples.

        Returns
        -------
        ExportResult
            Paths and counts for each exported dataset.
        """
        root = Path(self.config.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        result = ExportResult()

        # --- Atarashi -------------------------------------------------------
        if atarashi_samples:
            atarashi_path = root / self.config.atarashi_dir_name
            ds_dict = self._build_atarashi_dataset(atarashi_samples)
            if isinstance(ds_dict, Dataset):
                ds_dict = DatasetDict({"train": ds_dict})
            ds_dict = self._clean(ds_dict, label_column="license_key")
            ds_dict.save_to_disk(str(atarashi_path))
            result.atarashi_path = str(atarashi_path)
            result.atarashi_total = len(atarashi_samples)
            result.atarashi_train = len(ds_dict.get("train", []))
            result.atarashi_validation = len(ds_dict.get("validation", []))
            result.atarashi_test = len(ds_dict.get("test", []))

            if self.config.write_statistics:
                stats = self._atarashi_statistics(ds_dict)
                stats_path = atarashi_path / "statistics.json"
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                logger.info("Atarashi statistics written to %s", stats_path)

            logger.info(
                "Atarashi dataset exported: %d total (%d train / %d validation / %d test) → %s",
                result.atarashi_total,
                result.atarashi_train,
                result.atarashi_validation,
                result.atarashi_test,
                atarashi_path,
            )

        # --- Nirjas ---------------------------------------------------------
        if nirjas_samples:
            nirjas_path = root / self.config.nirjas_dir_name
            ds_dict = self._build_nirjas_dataset(nirjas_samples)
            if isinstance(ds_dict, Dataset):
                ds_dict = DatasetDict({"train": ds_dict})
            # No per-class guard: the binary classes both have ample samples.
            ds_dict = self._clean(ds_dict, label_column=None)
            ds_dict.save_to_disk(str(nirjas_path))
            result.nirjas_path = str(nirjas_path)
            result.nirjas_total = len(nirjas_samples)
            result.nirjas_train = len(ds_dict.get("train", []))
            result.nirjas_validation = len(ds_dict.get("validation", []))
            result.nirjas_test = len(ds_dict.get("test", []))

            if self.config.write_statistics:
                stats = self._nirjas_statistics(ds_dict)
                stats_path = nirjas_path / "statistics.json"
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
                logger.info("Nirjas statistics written to %s", stats_path)

            logger.info(
                "Nirjas dataset exported: %d total (%d train / %d validation / %d test) → %s",
                result.nirjas_total,
                result.nirjas_train,
                result.nirjas_validation,
                result.nirjas_test,
                nirjas_path,
            )

        return result

    def print_summary(self, result: ExportResult) -> None:
        """Pretty-print the export results to stdout."""
        print("=" * 60)
        print("DATASET EXPORT SUMMARY")
        print("=" * 60)

        if result.atarashi_path:
            print("\n--- Atarashi (licence similarity) ---")
            print(f"  Path:           {result.atarashi_path}")
            print(f"  Total samples:  {result.atarashi_total:,}")
            print(f"  Train split:      {result.atarashi_train:,}")
            print(f"  Validation split: {result.atarashi_validation:,}")
            print(f"  Test split:       {result.atarashi_test:,}")
        else:
            print("\n  Atarashi: (not exported)")

        if result.nirjas_path:
            print("\n--- Nirjas (2-class classification) ---")
            print(f"  Path:           {result.nirjas_path}")
            print(f"  Total samples:  {result.nirjas_total:,}")
            print(f"  Train split:      {result.nirjas_train:,}")
            print(f"  Validation split: {result.nirjas_validation:,}")
            print(f"  Test split:       {result.nirjas_test:,}")
        else:
            print("\n  Nirjas: (not exported)")

        print("=" * 60)


# Convenience function


def export_datasets(
    atarashi_samples: Optional[list[AtarashiSample]] = None,
    nirjas_samples: Optional[list[NirjasSample]] = None,
    output_dir: str = "output",
    train_split_ratio: float = 0.8,
    validation_split_ratio: float = 0.1,
    random_seed: int = 42,
    write_statistics: bool = True,
) -> ExportResult:
    """One-shot convenience function for exporting both datasets.

    See :class:`DatasetExporter` for full parameter documentation.
    """
    config = ExportConfig(
        output_dir=output_dir,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
        random_seed=random_seed,
        write_statistics=write_statistics,
    )
    exporter = DatasetExporter(config)
    return exporter.export(
        atarashi_samples=atarashi_samples,
        nirjas_samples=nirjas_samples,
    )
