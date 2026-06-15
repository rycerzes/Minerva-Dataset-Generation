from __future__ import annotations

import logging
import re
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from typing import TypeVar, Iterable, Any, Optional

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


def download(
    url: str, timeout: float = 60.0, client: Optional[httpx.Client] = None
) -> httpx.Response:
    if client is not None:
        response = client.get(url, timeout=timeout)
        response.raise_for_status()
        return response

    local_client = httpx.Client(timeout=timeout)
    try:
        response = local_client.get(url)
        response.raise_for_status()
        return response
    finally:
        local_client.close()


def save(content: Any, output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write(str(content))


def to_parquet(records: list[dict], output_path: str) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)


def model_to_parquet(models: list[T], output_path: str) -> None:
    records = [model.model_dump() for model in models]
    to_parquet(records, output_path)


def iter_to_parquet(models: Iterable[T], output_path: str) -> int:
    records = [model.model_dump() for model in models]
    to_parquet(records, output_path)
    return len(records)


def read_parquet(path: str, model_class: type[T]) -> list[T]:
    table = pq.read_table(path)
    return [model_class.model_validate(row) for row in table.to_pylist()]


def dedup_near_duplicates(
    texts: list[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    n: int = 3,
) -> list[int]:
    """Return the indices of *texts* to keep after MinHash LSH near-dedup.

    Uses ``datasketch.MinHashLSH`` with character *n*-gram shingles.  The
    first occurrence in each near-duplicate cluster is retained.  Query and
    insertion are both O(1) amortised per document, making this suitable for
    datasets with hundreds of thousands of samples.

    Parameters
    ----------
    texts:
        Input text strings.
    threshold:
        Jaccard similarity threshold above which two texts are treated as
        near-duplicates.  Default 0.8.
    num_perm:
        Number of MinHash permutations.  Higher values give a more accurate
        Jaccard estimate at the cost of memory.  Default 128.
    n:
        Character n-gram size for shingling.  Default 3.

    Returns
    -------
    list[int]
        Sorted list of indices into *texts* that should be kept.
    """
    if len(texts) < 2:
        return list(range(len(texts)))

    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "datasketch not installed — skipping near-dedup. "
            "Install with: pip install datasketch"
        )
        return list(range(len(texts)))

    lsh: MinHashLSH = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep: list[int] = []

    for i, text in enumerate(texts):
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        m: MinHash = MinHash(num_perm=num_perm)
        for j in range(max(1, len(normalized) - n + 1)):
            m.update(normalized[j : j + n].encode("utf-8"))

        if not lsh.query(m):
            lsh.insert(str(i), m)
            keep.append(i)

    return keep
