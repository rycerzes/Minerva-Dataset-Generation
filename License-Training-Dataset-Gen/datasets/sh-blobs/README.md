# Software Heritage licence-blob corpora

Two corpora and one benchmark record, all derived from the
[Software Heritage License Dataset (2022 edition)](https://annex.softwareheritage.org/public/dataset/license-blobs/2022-04-25/).
They answer two questions no other corpus here can.

**Licence granularity.** Debian maintainers write bare `MIT` on every MIT variant and
only 18 licences carry SPDX tags in the-stack, so nothing else could teach the ranker
`MIT-advertising` against `MIT` or `CC-BY-SA-3.0` against `CC-BY-3.0`. This can:
ScanCode labels every blob at full granularity.

**Prevalence.** The 20k sample is a *random* draw from 6.9M real licence blobs, so its
label distribution is the real one. The other tail benchmark here takes one blob per
licence and is macro by construction — the right instrument for "which licences can we
identify at all" and the wrong one for "how often are we right on real files".

## What is here

| file | rows | content |
|---|---|---|
| `prevalence_manifest.json` | 15,317 label rows / 12,881 blobs | sha1 + ScanCode label, **no text** |
| `training_manifest.json` | 1,997 | sha1 + label + size, **no text** |
| `tail_corpus.json.gz` | 81 licences | one human-confirmed blob each, **with text** |
| `prevalence_baseline.jsonl.gz` | 12,881 | per-blob predictions of the shipped engine |

Manifests carry no text because the blobs are re-fetchable in bulk and the archive is
335 MB; the tail corpus carries text because it is not. Its blobs came one at a time
from the Software Heritage content API, which allows **120 requests per hour** to
anonymous callers, so refetching it costs the better part of a day.

## Rehydrating

```bash
cd $SH_DATA_DIR                      # anywhere; export SH_DATA_DIR to point at it
base=https://annex.softwareheritage.org/public/dataset/license-blobs/2022-04-25
curl -sSLO $base/blobs-sample20k.tar.zst        #  27 MB -> 219 MB extracted
curl -sSLO $base/blobs-scancode.csv.zst         # 116 MB, sha1,license,score
curl -sSLO $base/licenses-annotated-sample.tar.gz   # 808 KB, the human annotations
mkdir -p blobs20k && tar --use-compress-program=unzstd -xf blobs-sample20k.tar.zst -C blobs20k
```

Then `python -m evaluation.sh_blobs` builds the training corpus and
`python -m evaluation.sh_prevalence` scores the benchmark.

## Labels, and what they are not

**ScanCode's, at score >= 95.** This corpus therefore *distils* ScanCode's variant
knowledge rather than being independent of it, which is why it is training data and a
self-comparison benchmark, never a head-to-head. Independent labels live in
`datasets/debian-dep5` (Debian maintainers) and the SPDX-tag corpus (file authors).

The `tail_corpus` blobs are the exception: they come from the 8,102 rows a human
annotator reviewed, restricted to those where the annotator did **not** dispute
ScanCode's answer. That makes the label human-validated — and makes the set an easy
subset *for ScanCode*, so it measures our coverage and not a win over it.

## Two things that will bite you

**A blob can carry several label rows.** The CSV records one row per detected licence,
so 15,317 rows cover 12,881 blobs. Group by sha1 and compare *sets*: scoring the rows
independently grades a multi-licence file as several single-licence failures, and an
early reading of this benchmark reported R@1 0.7962 for exactly that reason against a
true 0.9752.

**Rare licences hide in giant aggregate files, and capping them out makes things
worse.** `NOTICE` and `THIRD-PARTY-NOTICES` files bundle hundreds of licences and get
one label at high confidence. They are 22.5% of the training rows and 148 MB of its
183 MB, and for licences like `Unicode-TOU`, `FTL` and `NAIST-2003` *every* training
example is one of them. That looks like data worth throwing away. It is not:

| | capped at 50 KB | uncapped (shipped) |
|---|---|---|
| prevalence R@1 | 0.9600 | **0.9752** |
| coverage | 0.9754 | **0.9907** |
| precision | 0.9842 | 0.9844 |
| tail R@1 | 0.833 | **0.848** |

Precision is identical; the cap costs only coverage, because it drops 452 licences to
297 and the model then declines more often. The features are match-evidence, not text,
so a blob where the licence is one of hundreds simply produces weak rows rather than
misleading ones. **There is no size cap, and adding one is a measured regression.**

The benchmark is unaffected either way — such files carry several label rows and are
excluded as multi-licence, leaving 99.9% of scored blobs under 50 KB.

## Measured on this data

| | |
|---|---|
| prevalence R@1 | **0.9752** ±0.0028 (n=11,680, 52 licences) |
| prevalence precision / exact-set | 0.9844 / 0.9742 |
| tail R@1 (macro, one blob per licence) | 0.848 (n=66 nameable of 81) |
| what it caught | 286 of 639 verbatim Apache-2.0 files reported as `ImageMagick` |

The last row is why this corpus exists. Both head corpora are notice-regime and the
tail benchmark holds one Apache-2.0 file, so a defect costing 45% of the most common
permissive licence in the verbatim regime was invisible until the pool was scored whole.

## Published

`hf/` holds the redistributable form, built by
`uv run src/evaluation/export_hf_sh.py` and intended for
**`rycerzes/atarashi-sh-blobs`** on HuggingFace, alongside `rycerzes/atarashi-dep5`.
Four configs: `prevalence`, `training`, `tail`, `baseline`. The two large ones ship as
pointers with `hf/rehydrate.py` to attach text from the SH annex and verify each blob
against its sha1 — spot-checked at 1,997/1,997 exact. `tail` ships text inline.

The card leads with the label caveat rather than burying it: on HuggingFace this will
be found by people looking for a licence-detection benchmark, and it is a distillation
of one scanner's output.

## Provenance, licensing and attribution

Derived from the **Software Heritage License Dataset (2022 edition)**, published on
Zenodo under **CC-BY-4.0**. Attribution is the condition of that licence and is given
here and in `evaluation/sh_blobs.py`:

> Gonzalez-Barahona, J.M., Montes-Leon, S., Robles, G., Zacchiroli, S.
> *The Software Heritage License Dataset (2022 Edition).*
> Empirical Software Engineering, Springer.
> Dataset: https://zenodo.org/records/8200352 · https://annex.softwareheritage.org/public/dataset/license-blobs/

> Zacchiroli, S. *A Large-scale Dataset of (Open Source) License Text Variants.*
> MSR 2022.

**Why `tail_corpus.json.gz` carries text and the manifests do not** is a practical
split, not a legal one. The manifests are large and the blobs bulk-download in one
command, so shipping hashes costs a reproducer nothing. The tail set cannot be
rebuilt that way — 120 anonymous requests per hour against the content API — so
withholding its 81 files would put a day's wait between a reader and a number this
repository quotes.

The blobs themselves are **LICENSE files**: the texts of open source licences, each
under its own terms, filtered to rows a human annotator confirmed *are* licence files.
Redistributing licence texts is ordinary and already established here — `atarashi`
ships 774 full licence texts in `processedLicenses.csv` and ~28k ScanCode rule texts
in `notice_rules.json`. 81 more, credited, is the same act at a smaller scale.

The selection, the grouping by blob, the manifests and the measured results are part of
this repository.
