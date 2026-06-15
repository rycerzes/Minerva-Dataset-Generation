from .hybrid_merge import (
    HybridMerger,
    DatasetEntry,
    DataSource,
    create_hybrid_dataset,
    save_hybrid_dataset,
)

# NOTE: augmented_merge imports from augmentation.*, which in turn imports
# from builder.hybrid_merge → circular at package-init time.  We defer
# those imports via __getattr__ so that ``from builder.augmented_merge
# import ...`` (the pattern used everywhere) keeps working while
# ``from builder import AugmentedMerger`` also works lazily.

_AUGMENTED_MERGE_NAMES = {
    "AugmentedMerger",
    "AugmentedMergerConfig",
    "AtarashiSample",
    "NirjasSample",
    "augmented_merge",
}


def __getattr__(name: str):
    if name in _AUGMENTED_MERGE_NAMES:
        from . import augmented_merge as _am

        return getattr(_am, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HybridMerger",
    "DatasetEntry",
    "DataSource",
    "create_hybrid_dataset",
    "save_hybrid_dataset",
    "AugmentedMerger",
    "AugmentedMergerConfig",
    "AtarashiSample",
    "NirjasSample",
    "augmented_merge",
]
