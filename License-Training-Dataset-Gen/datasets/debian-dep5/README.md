# Debian DEP-5 license corpus

Independent, matcher-free license labels for source-file headers, drawn from Debian
maintainers' hand-curated `debian/copyright` records.

**This is the corpus the ranker in `atarashi/data/ranker.joblib` is trained on.** It is
not `rycerzes/atarashi-dataset` on HuggingFace — that is an earlier, unrelated set of
28,491 text fragments built for a classifier approach that was abandoned, and ~86% of
its fragments are verbatim substrings of the references they would be scored against.
Nothing here derives from it.

## What it is

| | |
|---|---|
| rows | 2,077 files from 544 source packages |
| notice regime | 678 (license prose present — answerable) |
| no-signal regime | 1,399 (no prose; the correct answer is UNKNOWN) |
| distinct licenses (notice) | 34 |

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
licenses; this reaches 34.

## Label noise — quote this with any result

**About 7%.** Of 359 queries that both Atarashi and ScanCode answered, 48 contradict
the DEP-5 label and 26 of those have the two independent engines agreeing with *each
other* against it. Two causes:

- DEP-5 is loose about only-vs-or-later — maintainers write `GPL-2` for a file whose
  header says "version 2 or any later version".
- `Files:` globs attribute a package-level license to files whose own header differs.

Good enough to compare engines against identical labels. Not readable as absolute
accuracy, and not clean enough to treat a disagreement as an engine error without
looking at the file.

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
