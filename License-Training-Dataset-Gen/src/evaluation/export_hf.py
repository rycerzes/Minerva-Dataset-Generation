"""Export the DEP-5 corpus in a form that is safe to redistribute.

The corpus is two-thirds no-signal rows — files carrying no license notice, whose
text is therefore just third-party source code. Republishing ~1,400 code excerpts
from 544 packages under mixed licenses is a redistribution question a
license-compliance project should not get wrong.

So: notice rows ship their text, because a license notice exists to be reproduced.
No-signal rows ship a pointer — package, version, path, and the SHA-256 of the text
we derived — and `rehydrate.py` reconstructs them from Debian on demand. Nothing
beyond license notices is redistributed, the negatives stay usable, and the
reconstruction is verifiable rather than merely described.

    uv run src/evaluation/export_hf.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets" / "debian-dep5" / "corpus.json"
OUT = ROOT / "datasets" / "debian-dep5" / "hf"

HEADER_LINES = 40  # must match evaluation.debian.HEADER_LINES


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_record(row: dict) -> dict:
    """One row; text only where the text is a license notice."""
    keep = row["regime"] == "notice"
    return {
        "regime": row["regime"],
        "license": row["gt"],
        "dep5_name": row["dep5_name"],
        "text": row["text"] if keep else None,
        "text_sha256": text_sha(row["text"]),
        "package": row["package"],
        "version": row["version"],
        "path": row["path"],
    }


def main() -> None:
    rows = json.loads(SOURCE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    records = [to_record(r) for r in rows]
    with (OUT / "corpus.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    inline = sum(1 for r in records if r["text"] is not None)
    print(f"wrote {len(records)} rows -> {OUT / 'corpus.jsonl'}")
    print(f"  text inline (notice regime) : {inline}")
    print(f"  pointer only (no-signal)    : {len(records) - inline}")


if __name__ == "__main__":
    main()
