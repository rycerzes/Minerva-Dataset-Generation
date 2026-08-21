#!/usr/bin/env python3
"""Reconstruct the no-signal rows of the DEP-5 corpus from Debian.

Those rows ship as pointers rather than text: they carry no license notice, so their
content is plain third-party source and redistributing it is not this dataset's to do.
Everything needed to rebuild them byte-for-byte is here, and each result is checked
against the recorded SHA-256 — so the corpus is verifiable, not merely described.

    python hf_rehydrate.py corpus.jsonl --out corpus.full.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://sources.debian.org"
UA = {"User-Agent": "atarashi-dep5-rehydrate/1.0"}
TAGLINE = re.compile(r"(?im)^.*SPDX-License-Identifier:.*$")
HEADER_LINES = 40


def fetch(record: dict) -> dict:
    if record["text"] is not None:
        return record
    quoted = "/".join(urllib.parse.quote(p) for p in
                      (record["package"], record["version"], record["path"]))
    try:
        request = urllib.request.Request(f"{BASE}/api/src/{quoted}/", headers=UA)
        with urllib.request.urlopen(request, timeout=30) as response:
            meta = json.loads(response.read().decode("utf-8", "replace"))
        request = urllib.request.Request(BASE + meta["raw_url"], headers=UA)
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception as error:
        return {**record, "error": str(error)}
    head = "\n".join(body.splitlines()[:HEADER_LINES])
    text = " ".join(TAGLINE.sub(" ", head).split())
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != record["text_sha256"]:
        # Upstream moved: report rather than silently substitute different content.
        return {**record, "error": f"sha mismatch (upstream changed): {got}"}
    return {**record, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, default=Path("corpus.full.jsonl"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    records = [json.loads(line) for line in args.corpus.read_text().splitlines() if line]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        filled = list(pool.map(fetch, records))
    with args.out.open("w") as handle:
        for record in filled:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    failed = sum(1 for r in filled if r.get("error"))
    print(f"wrote {len(filled)} rows -> {args.out}  ({failed} could not be rebuilt)")


if __name__ == "__main__":
    main()
