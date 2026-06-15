from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fetchers.scancode import ScanCodeFetcher
from fetchers.fossology import FossologyFetcher
from builder.hybrid_merge import (
    HybridMerger,
    save_hybrid_dataset,
)
from builder.augmented_merge import AugmentedMerger, AugmentedMergerConfig
from exporter.dataset_export import DatasetExporter, ExportConfig
from augmentation.legal_structure_splitter import LegalStructureSplitter
from augmentation.llm_synthetic import SurgicalLLMInjector
from augmentation.hard_negative_generator import HardNegativeGenerator
from augmentation.llm_cache import LLMCache
from augmentation.class_balancing import NirjasClassBalancer, BalancingConfig
from augmentation.rare_license_augmenter import (
    RareLicenseAugmenter,
    RareLicenseAugmenterConfig,
)
from fetchers.code_comments import CodeCommentFetcher

logger = logging.getLogger(__name__)


def _llm_available() -> bool:
    """Return True if a .env with LLM config exists and is parseable."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return False
    try:
        from config import LLMConfig

        LLMConfig(env_path=env_path)
        return True
    except Exception:
        return False


def run_pipeline(
    output_path: str = "hybrid_dataset.parquet",
    scancode_base_url: str = "https://scancode-licensedb.aboutcode.org",
    fossology_base_url: str = "https://raw.githubusercontent.com/fossology/fossology/master/install/db/licenseRef.json",
    include_exceptions: bool = True,
    include_deprecated: bool = True,
    export_dir: str = "output",
    train_split_ratio: float = 0.8,
    validation_split_ratio: float = 0.1,
    enable_llm: bool = True,
    verbose: bool = False,
    cache_dir: str = "cache",
    samples_per_category: int = 5,
    hard_negative_limit: int | None = None,
    code_comments_limit: int = 25_000,
    max_nirjas_samples: int | None = None,
    rare_license_threshold: int = 5,
    rare_license_augment_count: int = 5,
) -> dict:
    total_steps = 10 if enable_llm else 7

    print("=" * 60)
    print("Minerva Dataset Pipeline")
    print("=" * 60)

    print(f"\n[1/{total_steps}] Fetching ScanCode LicenseDB...")
    scancode_fetcher = ScanCodeFetcher(base_url=scancode_base_url, cache_dir=cache_dir)
    scancode_licenses = scancode_fetcher.fetch_all(
        include_exceptions=include_exceptions,
        include_deprecated=include_deprecated,
    )
    scancode_count = len(scancode_licenses)
    scancode_with_text = sum(1 for lic in scancode_licenses if lic.license_text)
    print(f"  Fetched {scancode_count} licenses ({scancode_with_text} with text)")

    print(f"\n[2/{total_steps}] Fetching FOSSology licenseRef.json...")
    fossology_fetcher = FossologyFetcher(
        base_url=fossology_base_url, cache_dir=cache_dir
    )
    fossology_licenses = fossology_fetcher.fetch_all()
    fossology_count = len(fossology_licenses)
    fossology_with_text = sum(1 for lic in fossology_licenses if lic.rf_text)
    print(f"  Fetched {fossology_count} licenses ({fossology_with_text} with text)")

    print(f"\n[3/{total_steps}] Merging datasets...")
    merger = HybridMerger(scancode_licenses, fossology_licenses)
    dataset = merger.merge()
    stats = merger.get_statistics(dataset)

    print(f"  Total entries: {stats['total']}")
    print(f"  ScanCode: {stats['scancode']}")
    print(f"  FOSSology legacy: {stats['fossology_legacy']}")
    print(f"  Exceptions: {stats['exceptions']}")

    print(f"\n  Saving hybrid dataset to {output_path}...")
    count = save_hybrid_dataset(dataset, output_path)
    print(f"  Saved {count} entries")

    step = 4
    print(f"\n[{step}/{total_steps}] Splitting license texts (sliding window)...")
    splitter = LegalStructureSplitter()
    fragments = splitter.split_dataset(dataset)
    frag_stats = splitter.get_fragment_statistics(fragments)
    print(f"  Total fragments:             {frag_stats['total_fragments']:,}")
    print(f"  Licenses with fragments:     {frag_stats['licenses_with_fragments']:,}")
    print(
        f"  Avg fragments/license:       {frag_stats['avg_fragments_per_license']:.1f}"
    )
    print(
        f"  Fragments with placeholders: {frag_stats['fragments_with_placeholders']:,}"
    )

    augmented_fragments = []
    hard_negatives = []
    code_comments = []

    if enable_llm:
        if not _llm_available():
            print(
                "\n  WARNING: --enable-llm is set but no valid .env found. "
                "Skipping LLM augmentation stages."
            )
            enable_llm = False

    if enable_llm:
        from config import LLMConfig

        llm_config = LLMConfig()
        llm_cache = LLMCache(cache_dir)

        step = 5
        fragments_with_ph = [f for f in fragments if f.placeholders]
        print(
            f"\n[{step}/{total_steps}] Surgical LLM injection "
            f"({len(fragments_with_ph)} fragments with placeholders)..."
        )
        injector = SurgicalLLMInjector(config=llm_config, cache=llm_cache)
        augmented_fragments = injector.augment_dataset(fragments)
        n_actually_augmented = sum(
            1
            for a in augmented_fragments
            if a.augmented_text.strip() != a.original_fragment.fragment_text.strip()
        )
        print(f"  Augmented fragments:  {n_actually_augmented:,}")
        print(
            f"  Pass-through (no ph): {len(augmented_fragments) - n_actually_augmented:,}"
        )
        inj_cache = llm_cache.stats
        print(f"  Cache hits/misses:    {inj_cache['hits']}/{inj_cache['misses']}")

        step = 6
        print(
            f"\n[{step}/{total_steps}] Augmenting rare-class licenses "
            f"(threshold < {rare_license_threshold} fragments)..."
        )
        rare_augmenter = RareLicenseAugmenter(
            config=RareLicenseAugmenterConfig(
                threshold=rare_license_threshold,
                augment_count=rare_license_augment_count,
            ),
            llm_config=llm_config,
            cache=llm_cache,
        )
        rare_fragments = rare_augmenter.augment(
            base_dataset=dataset,
            existing_fragments=fragments,
        )
        rare_stats = rare_augmenter.get_statistics(rare_fragments)
        print(f"  Synthetic fragments generated: {rare_stats['total']:,}")
        print(f"  Rare licenses augmented:       {rare_stats['unique_licenses']:,}")
        # Append to fragment pool so downstream stages (Atarashi + Nirjas) see them
        fragments = fragments + rare_fragments

        step = 7
        license_keys = [e.license_key for e in dataset if e.license_text]
        limit_tag = (
            f" (limited to {hard_negative_limit})" if hard_negative_limit else ""
        )
        print(
            f"\n[{step}/{total_steps}] Generating hard negatives "
            f"for {len(license_keys)} licenses{limit_tag}..."
        )
        neg_generator = HardNegativeGenerator(
            config=llm_config,
            samples_per_category=samples_per_category,
            cache=llm_cache,
        )
        hard_negatives = neg_generator.generate_batch(
            license_keys, max_licenses=hard_negative_limit
        )
        neg_stats = neg_generator.get_statistics(hard_negatives)
        print(f"  Total hard negatives: {neg_stats['total']:,}")
        for ntype, ncount in sorted(neg_stats.get("by_type", {}).items()):
            print(f"    {ntype:25s} {ncount:,}")
        if "cache" in neg_stats:
            cs = neg_stats["cache"]
            print(f"  Cache hits/misses:    {cs['hits']}/{cs['misses']}")

    step = total_steps - 2
    if code_comments_limit > 0:
        print(
            f"\n[{step}/{total_steps}] Fetching generic code comments (negative class)..."
        )
        comment_fetcher = CodeCommentFetcher(cache_dir=cache_dir)
        code_comments = comment_fetcher.fetch(max_samples=code_comments_limit)
        print(f"  Fetched {len(code_comments):,} clean non-license code comments")
    else:
        print(
            f"\n[{step}/{total_steps}] Skipping code comments (--code-comments-limit 0)"
        )

    step = total_steps - 1
    print(f"\n[{step}/{total_steps}] Balancing Nirjas classes & augmented merge...")

    # Nirjas class balancing
    balancer = NirjasClassBalancer(
        BalancingConfig(max_total_samples=max_nirjas_samples)
        if max_nirjas_samples is not None
        else None
    )
    nirjas_balanced = balancer.balance(
        fragments=fragments,
        augmented=augmented_fragments,
        hard_negatives=hard_negatives,
        code_comments=code_comments if code_comments_limit > 0 else None,
    )
    balancer.print_statistics(nirjas_balanced)

    # Augmented merge (Atarashi stratification + Nirjas conversion)
    aug_merger = AugmentedMerger(AugmentedMergerConfig())
    atarashi_samples, nirjas_samples = aug_merger.merge(
        base_dataset=dataset,
        fragments=fragments,
        augmented_fragments=augmented_fragments,
        hard_negatives=hard_negatives,
        nirjas_balanced=nirjas_balanced,
    )
    aug_merger.print_statistics(atarashi_samples, nirjas_samples)

    step = total_steps
    print(f"\n[{step}/{total_steps}] Exporting HF datasets...")
    export_config = ExportConfig(
        output_dir=export_dir,
        train_split_ratio=train_split_ratio,
        validation_split_ratio=validation_split_ratio,
    )
    exporter = DatasetExporter(export_config)
    export_result = exporter.export(
        atarashi_samples=atarashi_samples,
        nirjas_samples=nirjas_samples,
    )
    exporter.print_summary(export_result)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    return {
        "scancode_fetched": scancode_count,
        "scancode_with_text": scancode_with_text,
        "fossology_fetched": fossology_count,
        "fossology_with_text": fossology_with_text,
        "output_path": output_path,
        "entries_written": count,
        "statistics": stats,
        "fragments_total": frag_stats["total_fragments"],
        "fragments_with_placeholders": frag_stats["fragments_with_placeholders"],
        "augmented_fragments": len(augmented_fragments),
        "rare_augmented_fragments": rare_stats["total"] if enable_llm else 0,
        "hard_negatives": len(hard_negatives),
        "code_comments": len(code_comments),
        "nirjas_balanced": len(nirjas_balanced),
        "atarashi_samples": len(atarashi_samples),
        "nirjas_samples": len(nirjas_samples),
        "export_result": export_result.model_dump(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Minerva Dataset Pipeline - Generate training dataset for Atarashi"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="hybrid_dataset.parquet",
        help="Output path for the merged dataset (default: hybrid_dataset.parquet)",
    )
    parser.add_argument(
        "--scancode-url",
        default="https://scancode-licensedb.aboutcode.org",
        help="ScanCode LicenseDB base URL",
    )
    parser.add_argument(
        "--fossology-url",
        default="https://raw.githubusercontent.com/fossology/fossology/master/install/db/licenseRef.json",
        help="FOSSology licenseRef.json URL",
    )
    parser.add_argument(
        "--include-exceptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include license exceptions in the dataset",
    )
    parser.add_argument(
        "--include-deprecated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include deprecated licenses in the dataset",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--export-dir",
        default="output",
        help="Root directory for exported HF datasets (default: output)",
    )
    parser.add_argument(
        "--train-split-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for the train split (default: 0.8)",
    )
    parser.add_argument(
        "--validation-split-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of data for the validation split (default: 0.1). "
            "The remaining fraction goes to test."
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help=(
            "Skip LLM-dependent stages (surgical injection & hard negatives). "
            "Useful for offline/CI runs.  The pipeline will still produce "
            "sliding-window fragments and export datasets, but without "
            "synthetic augmentation or Nirjas hard negatives."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help=(
            "Directory for caching LLM responses so re-runs skip "
            "already-completed work (default: cache)"
        ),
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help=(
            "Number of hard-negative samples to request per category "
            "per license (default: 5). Lower values reduce token usage."
        ),
    )
    parser.add_argument(
        "--hard-negative-limit",
        type=int,
        default=None,
        help=(
            "Maximum number of licenses to generate hard negatives for. "
            "If omitted, all licenses with text are processed. "
            "Use e.g. --hard-negative-limit 200 to save tokens."
        ),
    )
    parser.add_argument(
        "--code-comments-limit",
        type=int,
        default=25_000,
        help=(
            "Number of generic code comments to fetch from bigcode/the-stack-smol "
            "and add to the not_license_related class (default: 25000). "
            "Set to 0 to disable and reproduce the previous behaviour."
        ),
    )
    parser.add_argument(
        "--max-nirjas-samples",
        type=int,
        default=None,
        help=(
            "Hard cap on the total Nirjas dataset size before train/test split. "
            "Prevents excessive upsampling of the positive class when the negative "
            "pool (hard negatives + code comments) is much larger than the positive "
            "pool (license fragments). The cap is split 50/50 between classes. "
            "Example: --max-nirjas-samples 200000 caps at 100k per class. "
            "If omitted, the dataset size equals the sum of both pools."
        ),
    )
    parser.add_argument(
        "--rare-license-threshold",
        type=int,
        default=5,
        help=(
            "Licenses with fewer than this many sliding-window fragments will be "
            "augmented with LLM-paraphrased variants (default: 5). "
            "Set higher to augment more licenses."
        ),
    )
    parser.add_argument(
        "--rare-license-augment-count",
        type=int,
        default=5,
        help=(
            "Number of paraphrased variants to generate per rare license "
            "(default: 5). Results are cached so re-runs are free."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        output_path=args.output,
        scancode_base_url=args.scancode_url,
        fossology_base_url=args.fossology_url,
        include_exceptions=args.include_exceptions,
        include_deprecated=args.include_deprecated,
        export_dir=args.export_dir,
        train_split_ratio=args.train_split_ratio,
        validation_split_ratio=args.validation_split_ratio,
        enable_llm=not args.no_llm,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
        samples_per_category=args.samples_per_category,
        hard_negative_limit=args.hard_negative_limit,
        code_comments_limit=args.code_comments_limit,
        max_nirjas_samples=args.max_nirjas_samples,
        rare_license_threshold=args.rare_license_threshold,
        rare_license_augment_count=args.rare_license_augment_count,
    )


if __name__ == "__main__":
    main()
