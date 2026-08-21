---
license: other
license_name: mixed-upstream
license_link: https://sources.debian.org/
task_categories:
  - text-classification
language:
  - en
tags:
  - license-detection
  - software-licenses
  - spdx
  - compliance
size_categories:
  - 1K<n<10K
---

# Debian DEP-5 license corpus

Source-file headers labelled with the license Debian maintainers recorded for them,
in the machine-readable DEP-5 `debian/copyright` format. **No matcher was involved in
producing these labels**, which is what makes the corpus usable for evaluating license
detectors without circularity.

Built for [FOSSology/Atarashi](https://github.com/fossology/atarashi); it is the
training and evaluation set for that project's candidate-ranking model.

## Not to be confused with `rycerzes/atarashi-dataset`

That is an earlier, unrelated set of 28,491 text fragments built for a classifier
approach that was abandoned. Roughly 86% of its fragments are verbatim substrings of
the reference texts they would be scored against, so recall measured on it reflects
copy-detection rather than identification. Nothing here derives from it.

## Contents

| | |
|---|---|
| rows | 2,077 files from 544 source packages |
| `notice` regime | 678 — license prose present, answerable |
| `no-signal` regime | 1,399 — no prose; the correct answer is UNKNOWN |
| distinct licenses (notice) | 34 |

Fields: `regime`, `license` (SPDX-style shortname), `dep5_name` (Debian's own
spelling, pre-normalization), `text`, `text_sha256`, `package`, `version`, `path`.

### Why `text` is null on no-signal rows

Those files carry no license notice, so their content is plain third-party source
code. Redistributing ~1,400 code excerpts from 544 packages under mixed licenses is
not this dataset's to do. They ship as pointers instead — package, version, path, and
the SHA-256 of the derived text — and `rehydrate.py` rebuilds them from Debian:

```bash
python rehydrate.py corpus.jsonl --out corpus.full.jsonl
```

Each rebuilt row is checked against its recorded hash, so reconstruction is verifiable
and a row whose upstream has changed is reported rather than silently substituted.
Notice rows carry their text inline: a license notice exists to be reproduced.

Derivation, if you would rather redo it: take the first 40 lines of the file, remove
every line containing `SPDX-License-Identifier:`, collapse whitespace.

## Label noise is ~7% — quote it with any result

Of 359 queries that two independent detectors (Atarashi and ScanCode) both answered,
48 contradict the DEP-5 label, and **26 of those have the two detectors agreeing with
each other** against it. Two causes:

- DEP-5 is loose about only-vs-or-later: maintainers write `GPL-2` for a file whose
  header says "version 2 or any later version".
- `Files:` globs attribute a package-level license to files whose own header differs.

Good enough to compare engines against identical labels. **Not** an absolute accuracy
ceiling, and a disagreement is not evidence of an engine error without reading the file.

## How packages were chosen

Rarest-license-first. Surveying a package costs two HTTP requests; building rows from
it costs about thirteen. So a broad survey picks the targets and the expensive pass is
spent only where it buys a license not already covered. Random sampling saturates on
MIT, GPL and Apache around 18 licenses; this reaches 34 in the notice regime and 55
overall.

The 6,000-package survey behind the selection saw 97 distinct licenses. Those 55 cover
**≥92% of all license occurrences** in it — license usage is extremely long-tailed, so
broad coverage of the catalogue and broad coverage of reality are very different
targets.

## Licensing

The selection, labelling, normalization and tooling are Apache-2.0. The `text` of each
notice row is a license notice reproduced from Debian source, under the license named
in that row's `license` field, with full provenance in `package`/`version`/`path`. No
other file content is redistributed.
