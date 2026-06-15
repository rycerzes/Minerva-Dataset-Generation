from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, Generator
import json
import logging
import sys
from pathlib import Path


LICENSEREF_DEFAULT_URL = "https://raw.githubusercontent.com/fossology/fossology/master/install/db/licenseRef.json"

try:
    from ..utils import download, iter_to_parquet
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import download, iter_to_parquet

logger = logging.getLogger(__name__)


class FossologyLicense(BaseModel):
    model_config = {"extra": "allow"}

    rf_shortname: Optional[str] = None
    rf_text: Optional[str] = None
    rf_url: Optional[str] = None
    rf_add_date: Optional[str] = None
    rf_copyleft: Optional[str] = None
    rf_OSIapproved: Optional[str] = None
    rf_fullname: Optional[str] = None
    rf_FSFfree: Optional[str] = None
    rf_GPLv2compatible: Optional[str] = None
    rf_GPLv3compatible: Optional[str] = None
    rf_notes: Optional[str] = None
    rf_Fedora: Optional[str] = None
    marydone: Optional[str] = None
    rf_active: Optional[str] = None
    rf_text_updatable: Optional[str] = None
    rf_detector_type: Optional[int] = None
    rf_source: Optional[str] = None
    rf_risk: Optional[str] = None
    rf_spdx_compatible: Optional[str] = None
    rf_flag: Optional[str] = None


class FossologyFetcher:
    CACHE_FILENAME = "fossology_licenses.json"

    def __init__(
        self,
        base_url: str = LICENSEREF_DEFAULT_URL,
        cache_dir: Optional[str | Path] = None,
    ):
        self.base_url = base_url
        self._licenses: Optional[list[FossologyLicense]] = None
        self._cache_dir = Path(cache_dir) if cache_dir else None

    # Cache helpers

    def _cache_path(self) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / self.CACHE_FILENAME

    def _load_cache(self) -> Optional[list[FossologyLicense]]:
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            licenses = [FossologyLicense.model_validate(entry) for entry in data]
            logger.info("Loaded %d FOSSology licenses from cache", len(licenses))
            return licenses
        except Exception as exc:
            logger.warning("Corrupt FOSSology cache, re-fetching: %s", exc)
            return None

    def _save_cache(self, licenses: list[FossologyLicense]) -> None:
        path = self._cache_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                [lic.model_dump() for lic in licenses],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved %d FOSSology licenses to cache", len(licenses))

    def fetch_all(self) -> list[FossologyLicense]:
        if self._licenses is not None:
            return self._licenses

        # --- Try disk cache first ---
        cached = self._load_cache()
        if cached is not None:
            self._licenses = cached
            return self._licenses

        # --- Fetch from network ---
        response = download(self.base_url)
        data = response.json()
        self._licenses = [FossologyLicense.model_validate(entry) for entry in data]

        self._save_cache(self._licenses)
        return self._licenses

    def iter_all(self) -> Generator[FossologyLicense, None, None]:
        if self._licenses is not None:
            for license in self._licenses:
                yield license
            return

        response = download(self.base_url)
        data = response.json()
        for entry in data:
            try:
                yield FossologyLicense.model_validate(entry)
            except Exception:
                continue


def fetch_fossology_licenses(
    base_url: str = LICENSEREF_DEFAULT_URL,
) -> list[FossologyLicense]:
    fetcher = FossologyFetcher(base_url)
    return fetcher.fetch_all()


def fetch_and_save_parquet(
    output_path: str = "fossology_licenses.parquet",
    base_url: str = LICENSEREF_DEFAULT_URL,
) -> int:
    fetcher = FossologyFetcher(base_url)
    return iter_to_parquet(fetcher.iter_all(), output_path)


if __name__ == "__main__":
    fetcher = FossologyFetcher()
    licenses = fetcher.fetch_all()
    print(f"Fetched {len(licenses)} licenses from FOSSology licenseRef.json")

    if licenses:
        sample = licenses[0]
        print(f"Sample license: {sample.rf_shortname}")
        print(f"  Full name: {sample.rf_fullname}")
