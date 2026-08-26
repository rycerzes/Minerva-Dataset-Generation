#!/usr/bin/env python3
"""Attach blob texts to `prevalence.jsonl` / `training.jsonl` from the SH annex.

Those two ship as pointers because the blobs bulk-download in one command and the
archive is 335 MB. `tail.jsonl` already carries its text and needs nothing.

    python rehydrate.py --data-dir ./sh-data            # downloads if absent
    python rehydrate.py --data-dir ./sh-data --split training

Verifies every blob against its sha1 before writing, so a truncated download fails
loudly instead of quietly producing wrong text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = "https://annex.softwareheritage.org/public/dataset/license-blobs/2022-04-25"
HERE = Path(__file__).resolve().parent


def ensure(data_dir: Path) -> Path:
    blobs = data_dir / "blobs20k"
    if blobs.is_dir() and any(blobs.rglob("*")):
        return blobs
    data_dir.mkdir(parents=True, exist_ok=True)
    tar = data_dir / "blobs-sample20k.tar.zst"
    if not tar.exists():
        print(f"downloading {tar.name} (27 MB) …", file=sys.stderr)
        subprocess.run(["curl", "-sSLo", str(tar), f"{BASE}/{tar.name}"], check=True)
    blobs.mkdir(exist_ok=True)
    print("extracting …", file=sys.stderr)
    subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(tar),
                    "-C", str(blobs)], check=True)
    return blobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("sh-data"))
    ap.add_argument("--split", choices=("prevalence", "training"), default="prevalence")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    blobs = ensure(args.data_dir)
    index = {p.name: p for p in blobs.rglob("*") if p.is_file()}
    rows = [json.loads(l) for l in (HERE / f"{args.split}.jsonl").open()]
    out = args.out or HERE / f"{args.split}-rehydrated.jsonl"

    done = missing = bad = 0
    with out.open("w") as fh:
        for r in rows:
            path = index.get(r["sha1"])
            if path is None:
                missing += 1
                continue
            raw = path.read_bytes()
            if hashlib.sha1(raw).hexdigest() != r["sha1"]:
                bad += 1
                continue
            r["text"] = raw.decode("utf-8", errors="replace")
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            done += 1
    print(f"{done} rehydrated -> {out}")
    if missing:
        print(f"  {missing} not in the 20k sample (expected 0)", file=sys.stderr)
    if bad:
        print(f"  {bad} FAILED sha1 verification", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
