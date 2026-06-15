from __future__ import annotations

from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path
from enum import Enum

try:
    from ..fetchers.scancode import LicenseData as ScanCodeLicense
    from ..fetchers.fossology import FossologyLicense
    from ..utils import model_to_parquet, iter_to_parquet
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from fetchers.scancode import LicenseData as ScanCodeLicense
    from fetchers.fossology import FossologyLicense
    from utils import model_to_parquet, iter_to_parquet  # noqa: F401


class DataSource(str, Enum):
    SCANCODE = "scancode"
    FOSSOLOGY = "fossology"


class DatasetEntry(BaseModel):
    model_config = {"protected_namespaces": ()}

    license_key: str
    short_name: str
    full_name: str
    category: str
    license_text: Optional[str] = None
    source: DataSource
    source_url: Optional[str] = None
    spdx_license_key: Optional[str] = None
    is_exception: bool = False
    is_fossology_legacy: bool = False


class HybridMerger:
    def __init__(
        self,
        scancode_licenses: list[ScanCodeLicense],
        fossology_licenses: list[FossologyLicense],
    ):
        self.scancode_licenses = {lic.license_key: lic for lic in scancode_licenses}
        self.fossology_licenses = {
            lic.rf_shortname: lic for lic in fossology_licenses if lic.rf_shortname
        }

    def _normalize_key(self, key: str) -> str:
        return key.lower().replace("_", "-").replace(" ", "-")

    def merge(self) -> list[DatasetEntry]:
        merged: list[DatasetEntry] = []
        added_keys: set[str] = set()

        for key, sc in self.scancode_licenses.items():
            normalized = self._normalize_key(key)
            entry = DatasetEntry(
                license_key=key,
                short_name=sc.short_name or key,
                full_name=sc.name or sc.short_name or key,
                category=sc.category or "unknown",
                license_text=sc.license_text,
                source=DataSource.SCANCODE,
                source_url=sc.source_url,
                spdx_license_key=sc.spdx_license_key,
                is_exception=sc.is_exception,
                is_fossology_legacy=False,
            )
            merged.append(entry)
            added_keys.add(normalized)

        for key, fo in self.fossology_licenses.items():
            normalized = self._normalize_key(key)
            if normalized in added_keys:
                continue

            entry = DatasetEntry(
                license_key=key,
                short_name=fo.rf_shortname or key,
                full_name=fo.rf_fullname or fo.rf_shortname or key,
                category=fo.rf_copyleft or "unknown",
                license_text=fo.rf_text,
                source=DataSource.FOSSOLOGY,
                source_url=fo.rf_url,
                spdx_license_key=None,
                is_exception=False,
                is_fossology_legacy=True,
            )
            merged.append(entry)
            added_keys.add(normalized)

        return merged

    def get_statistics(self, merged: list[DatasetEntry]) -> dict:
        scancode_count = sum(1 for e in merged if e.source == DataSource.SCANCODE)
        fossology_legacy_count = sum(
            1 for e in merged if e.source == DataSource.FOSSOLOGY
        )
        exception_count = sum(1 for e in merged if e.is_exception)
        return {
            "total": len(merged),
            "scancode": scancode_count,
            "fossology_legacy": fossology_legacy_count,
            "exceptions": exception_count,
        }


def create_hybrid_dataset(
    scancode_licenses: list[ScanCodeLicense],
    fossology_licenses: list[FossologyLicense],
) -> list[DatasetEntry]:
    merger = HybridMerger(scancode_licenses, fossology_licenses)
    return merger.merge()


def save_hybrid_dataset(
    dataset: list[DatasetEntry],
    output_path: str,
) -> int:
    return iter_to_parquet(dataset, output_path)


if __name__ == "__main__":
    from fetchers.scancode import fetch_scancode_licenses
    from fetchers.fossology import fetch_fossology_licenses

    print("Fetching ScanCode licenses...")
    scancode = fetch_scancode_licenses(include_exceptions=False)
    print(f"  Fetched {len(scancode)} ScanCode licenses")

    print("Fetching FOSSology licenses...")
    fossology = fetch_fossology_licenses()
    print(f"  Fetched {len(fossology)} FOSSology licenses")

    print("Merging datasets...")
    merger = HybridMerger(scancode, fossology)
    dataset = merger.merge()

    stats = merger.get_statistics(dataset)
    print(f"  Total entries: {stats['total']}")
    print(f"  ScanCode: {stats['scancode']}")
    print(f"  FOSSology legacy: {stats['fossology_legacy']}")
    print(f"  Exceptions: {stats['exceptions']}")

    output_path = "hybrid_dataset.parquet"
    count = save_hybrid_dataset(dataset, output_path)
    print(f"Saved {count} entries to {output_path}")
