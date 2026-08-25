# Debian DEP-5 license corpus

Independent, matcher-free license labels for source-file headers, drawn from Debian
maintainers' hand-curated `debian/copyright` records.

**This is one of the three corpora the ranker in `atarashi/data/ranker.joblib` is
trained on**, alongside SPDX-tag queries from the-stack and Software Heritage licence
blobs (`evaluation.sh_blobs`), which together take the trained set to 152 licences.
This one supplies the independent, matcher-free labels; the other two supply
tag-declared labels and licence-granularity breadth respectively.

It is **not** `rycerzes/atarashi-dataset` on HuggingFace — that is an earlier,
unrelated set of 28,491 text fragments built for a classifier approach that was
abandoned, and ~86% of its fragments are verbatim substrings of the references they
would be scored against. Nothing here derives from it.

## What it is

| | |
|---|---|
| rows | 2,050 files from 537 source packages |
| notice regime | 671 (license prose present — answerable) |
| no-signal regime | 1,379 (no prose; the correct answer is UNKNOWN) |
| distinct licenses (notice) | 33 |

*(An earlier edition of this table read 2,077 / 544 / 678 / 1,399 / 34. Those are the
counts from before the licence list was merged with SPDX and Debian's `GPL-2` style
names were re-mapped to `GPL-2.0-only` / `GPL-2.0-or-later`; a handful of rows became
unmappable and were dropped. `corpus.json` was refreshed then, this table was not. The
figures above are read from the shipped file and match what the eval reports.)*

Fields: `text` (the file header, SPDX tag stripped), `gt` (license shortname),
`dep5_name` (Debian's own spelling, before normalization), `regime`, `package`,
`version`, `path`.

## Why it exists

Every other corpus in this repo is labelled either by a matcher — which makes any
evaluation against a matcher circular — or by an in-file `SPDX-License-Identifier`,
which in practice reaches only ~18 licenses. Debian's records are curated by hand, per
file glob, with no matcher involved, across 58,471 source packages.

Packages were selected **rarest-license-first**: surveying a package costs two HTTP
requests and building rows from it costs about thirteen, so a broad survey picks the
targets and the expensive pass is spent only where it buys a license not already
covered. Random sampling saturates on MIT, GPL and Apache and stalls around 18
licenses; this reaches 33.

## Label noise — quote this with any result

**About 10%.** Of 651 queries that both Atarashi and ScanCode answered, 73 contradict
the DEP-5 label and **63 of those have the two independent engines agreeing with *each
other* against it** — 9.7%.

*(Earlier editions of this datasheet said ~7%, from 26 of 359. That was measured when
both engines answered far fewer queries; the denominator has since nearly doubled and
the estimate is better resolved, not worse. Quote 9.7%.)*

Two causes:

- DEP-5 is loose about only-vs-or-later — maintainers write `GPL-2` for a file whose
  header says "version 2 or any later version".
- `Files:` globs attribute a package-level license to files whose own header differs.

Good enough to compare engines against identical labels. Not readable as absolute
accuracy, and not clean enough to treat a disagreement as an engine error without
looking at the file.

**This is bias, not variance: it does not shrink with a bigger crawl.** Any effect
smaller than ~10% on this corpus is below the label floor, which is why the
prevalence-weighted Software Heritage benchmark (n=11,680, ±0.0028) carries the
small-effect measurements and this corpus carries the independent-label ones.

## Published

`hf/` holds the redistributable form, mirrored to
**https://huggingface.co/datasets/rycerzes/atarashi-dep5** (private until made
public). Notice rows carry their text; no-signal rows carry a pointer plus the
SHA-256 of the derived text, because those files hold no license notice and their
content is plain third-party source. `hf/rehydrate.py` rebuilds them from Debian and
verifies each against its hash — spot-checked at 8/8 exact.

Regenerate with `uv run src/evaluation/export_hf.py`.

## Reproducing

```
uv run src/run_eval.py debian-dep5 --packages 6000                      # survey
uv run src/run_eval.py debian-dep5 --mode corpus --packages 6000 --tail \
    --per-license 15 --per-package 5 --out cache/debian_corpus.json     # build
```

Both cache under `cache/debian/` and resume. The build is roughly 20,000 requests
against `sources.debian.org`, so it is committed here rather than regenerated
casually.

## Provenance and licensing

Content is upstream source-file headers redistributed from Debian, each under its own
license as recorded in `gt`. The selection, labelling and normalization are part of
this repository. Debian short names are mapped to SPDX-style shortnames by
`evaluation.debian.canonical_license`; `dep5_name` preserves the original so the
mapping can be audited or redone.
