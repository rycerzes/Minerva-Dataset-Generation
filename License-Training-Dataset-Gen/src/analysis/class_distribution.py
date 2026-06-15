from __future__ import annotations

from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path
from collections import Counter
import json

try:
    from ..builder.hybrid_merge import DatasetEntry
    from ..utils import read_parquet
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from builder.hybrid_merge import DatasetEntry
    from utils import read_parquet


class ClassDistributionStats(BaseModel):
    model_config = {"protected_namespaces": ()}

    license_key: str
    sample_count: int
    percentage: float
    cumulative_percentage: float
    needs_augmentation: bool
    target_samples: Optional[int] = None


class ImbalanceReport(BaseModel):
    model_config = {"protected_namespaces": ()}

    total_samples: int
    total_classes: int
    max_samples_per_class: int
    min_samples_per_class: int
    mean_samples_per_class: float
    median_samples_per_class: float
    imbalance_ratio: float
    classes_needing_augmentation: list[str]
    class_distribution: list[ClassDistributionStats]


class ClassImbalanceAnalyzer:
    def __init__(
        self,
        dataset: list[DatasetEntry],
        min_samples_per_class: int = 10,
        max_imbalance_ratio: float = 10.0,
    ):
        self.dataset = dataset
        self.min_samples_per_class = min_samples_per_class
        self.max_imbalance_ratio = max_imbalance_ratio

    def analyze(self) -> ImbalanceReport:
        class_counts = Counter(entry.license_key for entry in self.dataset)
        total_samples = len(self.dataset)
        total_classes = len(class_counts)

        sorted_counts = sorted(class_counts.values())
        max_samples = max(sorted_counts)
        min_samples = min(sorted_counts)
        mean_samples = total_samples / total_classes if total_classes > 0 else 0

        median_samples = sorted_counts[total_classes // 2] if total_classes > 0 else 0

        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float("inf")

        cumulative = 0
        class_distribution = []
        classes_needing_augmentation = []

        for license_key, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            cumulative += count
            percentage = (count / total_samples) * 100
            cumulative_percentage = (cumulative / total_samples) * 100

            needs_aug = (
                count < self.min_samples_per_class
                or imbalance_ratio > self.max_imbalance_ratio
            )

            if needs_aug:
                target = max(self.min_samples_per_class, int(mean_samples))
                classes_needing_augmentation.append(license_key)
            else:
                target = None

            class_distribution.append(
                ClassDistributionStats(
                    license_key=license_key,
                    sample_count=count,
                    percentage=percentage,
                    cumulative_percentage=cumulative_percentage,
                    needs_augmentation=needs_aug,
                    target_samples=target,
                )
            )

        return ImbalanceReport(
            total_samples=total_samples,
            total_classes=total_classes,
            max_samples_per_class=max_samples,
            min_samples_per_class=min_samples,
            mean_samples_per_class=mean_samples,
            median_samples_per_class=median_samples,
            imbalance_ratio=imbalance_ratio,
            classes_needing_augmentation=classes_needing_augmentation,
            class_distribution=class_distribution,
        )

    def print_report(self, report: ImbalanceReport) -> None:
        print("=" * 60)
        print("CLASS IMBALANCE ANALYSIS REPORT")
        print("=" * 60)
        print(f"Total Samples:        {report.total_samples:,}")
        print(f"Total Classes:        {report.total_classes:,}")
        print(f"Max Samples/Class:     {report.max_samples_per_class:,}")
        print(f"Min Samples/Class:    {report.min_samples_per_class:,}")
        print(f"Mean Samples/Class:   {report.mean_samples_per_class:.2f}")
        print(f"Median Samples/Class: {report.median_samples_per_class:.2f}")
        print(f"Imbalance Ratio:       {report.imbalance_ratio:.2f}x")
        print()
        print(
            f"Classes needing augmentation: {len(report.classes_needing_augmentation)}"
        )
        if report.classes_needing_augmentation:
            print("  " + ", ".join(report.classes_needing_augmentation[:10]))
            if len(report.classes_needing_augmentation) > 10:
                print(f"  ... and {len(report.classes_needing_augmentation) - 10} more")
        print()
        print("Top 10 Classes by Sample Count:")
        print("-" * 60)
        for stat in report.class_distribution[:10]:
            aug_marker = " [NEEDS AUGMENTATION]" if stat.needs_augmentation else ""
            print(
                f"  {stat.license_key:30s} {stat.sample_count:5d} ({stat.percentage:5.2f}%){aug_marker}"
            )
        print()
        print("Bottom 10 Classes by Sample Count:")
        print("-" * 60)
        for stat in report.class_distribution[-10:]:
            aug_marker = " [NEEDS AUGMENTATION]" if stat.needs_augmentation else ""
            print(
                f"  {stat.license_key:30s} {stat.sample_count:5d} ({stat.percentage:5.2f}%){aug_marker}"
            )
        print("=" * 60)

    def save_report(self, report: ImbalanceReport, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)


def analyze_dataset(
    dataset_path: str,
    output_path: Optional[str] = None,
    min_samples_per_class: int = 10,
    max_imbalance_ratio: float = 10.0,
) -> ImbalanceReport:
    dataset = read_parquet(dataset_path, DatasetEntry)
    analyzer = ClassImbalanceAnalyzer(
        dataset,
        min_samples_per_class=min_samples_per_class,
        max_imbalance_ratio=max_imbalance_ratio,
    )
    report = analyzer.analyze()
    analyzer.print_report(report)
    if output_path:
        analyzer.save_report(report, output_path)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze class distribution in merged dataset"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="hybrid_dataset.parquet",
        help="Path to the merged dataset parquet file",
    )
    parser.add_argument("-o", "--output", help="Output JSON report path", default=None)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per class threshold",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=10.0,
        help="Maximum imbalance ratio threshold",
    )

    args = parser.parse_args()

    analyze_dataset(
        args.dataset,
        output_path=args.output,
        min_samples_per_class=args.min_samples,
        max_imbalance_ratio=args.max_ratio,
    )
