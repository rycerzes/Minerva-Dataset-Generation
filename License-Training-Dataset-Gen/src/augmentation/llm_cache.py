"""Disk-based LLM response cache for the Minerva Dataset Pipeline.

Persists LLM responses as JSON files so that re-runs or pipeline resumptions
skip previously-completed work.  Each cache entry is keyed by a namespace
(e.g. ``"hard_negatives"``, ``"surgical_injection"``) and a caller-defined
string key (typically a license key or content hash).

Usage::

    cache = LLMCache(Path("cache"))
    cached = cache.get("hard_negatives", "MIT")
    if cached is None:
        result = call_llm(...)
        cache.set("hard_negatives", "MIT", result)
    else:
        result = cached
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _safe_filename(key: str) -> str:
    """Convert an arbitrary string key into a filesystem-safe filename.

    Short, printable keys are kept mostly human-readable.  Longer or
    problematic keys are SHA-256 hashed.
    """
    # Replace non-alphanumeric characters with underscores
    safe = re.sub(r"[^\w\-.]", "_", key)
    if len(safe) > 200:
        safe = hashlib.sha256(key.encode()).hexdigest()
    return safe


class LLMCache:
    """Simple file-system cache for LLM responses."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    # Internal helpers

    def _key_path(self, namespace: str, key: str) -> Path:
        return self.cache_dir / namespace / f"{_safe_filename(key)}.json"

    # Public API

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Return cached value or ``None`` on miss."""
        path = self._key_path(namespace, key)
        if path.exists():
            self._hits += 1
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Corrupt cache entry %s – ignoring: %s", path, exc)
                return None
        self._misses += 1
        return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Persist *value* (must be JSON-serialisable)."""
        path = self._key_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def has(self, namespace: str, key: str) -> bool:
        return self._key_path(namespace, key).exists()

    def clear(self, namespace: Optional[str] = None) -> int:
        """Delete cached entries.  Returns number of files removed.

        If *namespace* is given, only that namespace is cleared.
        Otherwise the entire cache directory is wiped.
        """
        import shutil

        target = self.cache_dir / namespace if namespace else self.cache_dir
        if not target.exists():
            return 0

        count = sum(1 for _ in target.rglob("*.json"))
        if namespace:
            shutil.rmtree(target, ignore_errors=True)
        else:
            shutil.rmtree(target, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        return count

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}
