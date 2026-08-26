"""Export the Software Heritage corpora in a form that is safe to redistribute.

Same split as `export_hf.py` makes for DEP-5, for the same reason and with the
opposite default. There, most rows are third-party *source code* and only the licence
notices ship inline. Here every row is a licence file, so redistribution is not the
constraint — size is. The blobs bulk-download from the Software Heritage annex in one
command, so the two large corpora ship as pointers and `rehydrate.py` joins them.

The tail set is the exception and ships its text: 81 blobs that came one at a time
through a content API allowing 120 anonymous requests an hour. Shipping hashes there
would put a day's wait between a reader and a number the engine report quotes.

    uv run src/evaluation/export_hf_sh.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets" / "sh-blobs"
OUT = SRC / "hf"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = {}

    prevalence = json.loads((SRC / "prevalence_manifest.json").read_text())
    with (OUT / "prevalence.jsonl").open("w") as fh:
        for r in prevalence:
            fh.write(json.dumps({"sha1": r["sha1"], "license": r["gt"],
                                 "text": None}, ensure_ascii=False) + "\n")
    written["prevalence.jsonl"] = len(prevalence)

    training = json.loads((SRC / "training_manifest.json").read_text())
    with (OUT / "training.jsonl").open("w") as fh:
        for r in training:
            fh.write(json.dumps({"sha1": r["sha1"], "license": r["gt"],
                                 "chars": r["chars"], "text": None},
                                ensure_ascii=False) + "\n")
    written["training.jsonl"] = len(training)

    with gzip.open(SRC / "tail_corpus.json.gz", "rt") as fh:
        tail = json.load(fh)
    with (OUT / "tail.jsonl").open("w") as fh:
        for r in tail:
            fh.write(json.dumps({"sha1": r["sha1"], "license": r["gt"],
                                 "text": r["text"],
                                 "blobs_for_license": r.get("n_blobs_for_license")},
                                ensure_ascii=False) + "\n")
    written["tail.jsonl"] = len(tail)

    with gzip.open(SRC / "prevalence_baseline.jsonl.gz", "rt") as fh:
        base = [json.loads(l) for l in fh]
    with (OUT / "baseline.jsonl").open("w") as fh:
        for r in base:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    written["baseline.jsonl"] = len(base)

    for name, n in written.items():
        size = (OUT / name).stat().st_size / 1e6
        print(f"  {name:22s} {n:6d} rows  {size:5.1f} MB")
    print(f"\nwrote {OUT}")
    print("text inline: tail.jsonl only; the rest rehydrate from the SH annex")


if __name__ == "__main__":
    main()
