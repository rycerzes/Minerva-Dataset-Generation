#!/usr/bin/env python3
"""Software Heritage license blobs — the licence-granularity training corpus.

The ranker's remaining failure is *within* a family: `MIT-advertising` loses to `MIT`,
three `CC-BY-*-3.0` variants lose to `CC-BY-3.0`. No corpus we had carries that
distinction. Debian DEP-5 is written by maintainers who put bare `MIT` on every MIT
variant, and only 18 licences carry SPDX tags in the-stack. The Software Heritage
License Dataset does: 20,000 sampled real licence blobs with per-blob ScanCode labels
at full granularity, plus 8,102 human-annotated rows used as the tail benchmark.

    https://annex.softwareheritage.org/public/dataset/license-blobs/2022-04-25/
      blobs-sample20k.tar.zst    27M   the blobs
      blobs-scancode.csv.zst    116M   sha1,license,score
      licenses-annotated-sample.tar.gz  808K  human ground truth (benchmark, not this)

**Labels here are ScanCode's, so this corpus distils ScanCode's variant knowledge
rather than being independent of it.** That is why it is training data only: the
human-annotated sample and the DEP-5 / SPDX-tag corpora stay as the evaluation sets,
and they are what says whether the distillation transferred.

Measured effect of adding it (cap 4/licence, 523 records, 37 -> 152 trained licences),
at matched coverage:

    SH tail (n=66)   R@1 0.697 -> 0.818     exact-set 0.667 -> 0.788
    DEP-5 (n=671)    R@1 0.833 -> 0.824     exact-set 0.826 -> 0.808
    SPDX-tag (n=266) R@1 0.947 -> 0.944     exact-set unchanged at 0.884

This is the axis the family-count curve does not cover. Widening *families* 18 -> 34
plateaued; widening *licences within* them is what moved the tail.

    python -m evaluation.sh_blobs --blobs DIR --labels blobs-scancode.csv.zst --cap 4
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import subprocess
from pathlib import Path

MIN_SCORE = 95.0     # ScanCode's own confidence; below this the label is a guess


def build(blob_dir: Path, labels: Path, cap: int = 4) -> list[dict]:
    """(text, licence) pairs, rarest licence first and capped.

    The cap is the whole point: 8,000 more MIT files buy nothing, and spending the
    corpus on licences the model has never seen is what the exercise is for. Cap 4
    keeps the full tail gain of cap 12 at half the head cost — the extra blobs add
    mass, not breadth, and mass on this corpus outweighs the head corpora.
    """
    have = {f: os.path.join(root, f)
            for root, _, files in os.walk(blob_dir) for f in files}
    proc = subprocess.Popen(["zstd", "-dc", str(labels)], stdout=subprocess.PIPE)
    by_lic: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    reader = csv.DictReader(io.TextIOWrapper(proc.stdout, encoding="utf-8",
                                             errors="replace"))
    for row in reader:
        path = have.get(row["sha1"])
        if not path or not row["license"] or " " in row["license"]:
            continue                     # compound expressions are the notice layer's job
        try:
            if float(row["score"]) < MIN_SCORE:
                continue
        except ValueError:
            continue
        by_lic[row["license"]].append((row["sha1"], path))
    proc.stdout.close()
    proc.wait()

    out = []
    for lic in sorted(by_lic, key=lambda k: len(by_lic[k])):
        for sha, path in by_lic[lic][:cap]:
            try:
                text = open(path, errors="replace").read()
            except OSError:
                continue
            if len(text.split()) >= 15:
                out.append({"sha1": sha, "gt": lic, "text": text, "regime": "notice"})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blobs", type=Path, required=True, help="extracted blob tree")
    ap.add_argument("--labels", type=Path, required=True, help="blobs-scancode.csv.zst")
    ap.add_argument("--cap", type=int, default=4, help="blobs per licence")
    ap.add_argument("--out", type=Path, default=Path("cache/sh_blobs.json"))
    ap.add_argument("--exclude", type=Path, default=None,
                    help="corpus whose sha1s must not be trained on (the benchmark)")
    args = ap.parse_args(argv)

    rows = build(args.blobs, args.labels, args.cap)
    if args.exclude and args.exclude.exists():
        held = {r["sha1"] for r in json.loads(args.exclude.read_text())}
        before = len(rows)
        rows = [r for r in rows if r["sha1"] not in held]
        print(f"excluded {before - len(rows)} blobs the benchmark scores on")
    lics = collections.Counter(r["gt"] for r in rows)
    fams = collections.Counter(k.split("-")[0].upper() for k in lics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows))
    print(f"{len(rows)} blobs across {len(lics)} licences (cap {args.cap})")
    print(f"  families holding more than one licence: "
          f"{sum(1 for n in fams.values() if n > 1)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
