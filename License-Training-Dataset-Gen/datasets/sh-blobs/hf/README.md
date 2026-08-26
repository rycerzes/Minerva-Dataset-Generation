---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
tags:
  - license-detection
  - software-licenses
  - spdx
  - compliance
  - scancode
size_categories:
  - 10K<n<100K
configs:
  - config_name: prevalence
    data_files: prevalence.jsonl
  - config_name: training
    data_files: training.jsonl
  - config_name: tail
    data_files: tail.jsonl
  - config_name: baseline
    data_files: baseline.jsonl
---

# Software Heritage licence blobs — prevalence and tail corpora

> **The labels here are ScanCode's, not ground truth.** This is a distillation of one
> scanner's output at score ≥ 95, useful as *training data* and as a *self-comparison
> regression benchmark*. It cannot be used to show that anything beats ScanCode, and a
> result quoted from it without that caveat is misleading. The one exception is the
> `tail` config, where a human annotator reviewed and did not dispute the label — which
> also makes that split an easy subset **for** ScanCode.

Derived from the [Software Heritage License Dataset (2022 edition)](https://zenodo.org/records/8200352).
Built for [Atarashi](https://github.com/fossology/atarashi), FOSSology's licence
identifier. Companion to [`rycerzes/atarashi-dep5`](https://huggingface.co/datasets/rycerzes/atarashi-dep5),
which carries *independent* labels from Debian maintainers.

## Why it exists

**Licence granularity.** Debian maintainers write bare `MIT` on every MIT variant, and
only 18 licences carry SPDX tags in the-stack, so no other corpus could teach a ranker
`MIT-advertising` against `MIT`, or `CC-BY-SA-3.0` against `CC-BY-3.0`. ScanCode labels
every blob at full granularity, and this corpus took a model's trained licence set from
37 to 152.

**Prevalence.** The Software Heritage 20k sample is a *random* draw from 6.9M real
licence blobs, so its label distribution is the real one. Most licence benchmarks take
one file per licence and are macro by construction — the right instrument for "which
licences can we identify at all", the wrong one for "how often are we right on real
files". The two differ by a factor of two on the same engine.

## Configs

| config | rows | text | what it is |
|---|---|---|---|
| `prevalence` | 15,317 label rows over 12,881 blobs | pointer | prevalence-weighted benchmark |
| `training` | 1,997 | pointer | licence-granularity training corpus |
| `tail` | 81 | **inline** | one human-confirmed blob per licence |
| `baseline` | 12,881 | — | reference predictions from a scanner run |

`python rehydrate.py --split prevalence` attaches text to the pointer configs from the
SH annex (27 MB download) and verifies every blob against its sha1. `tail` ships text
because it was fetched one blob at a time through an API allowing 120 anonymous
requests an hour.

## Two traps, both of which cost real time

**A blob can carry several label rows.** The source CSV records one row per detected
licence, so 15,317 rows cover 12,881 blobs. **Group by `sha1` and compare sets.**
Scoring rows independently grades a multi-licence file as several single-licence
failures — doing that reported R@1 0.7962 for an engine whose true score was 0.9752.

**Do not filter out the large blobs.** Rare licences hide in multi-megabyte `NOTICE`
and `THIRD-PARTY-NOTICES` aggregates; they are 22.5% of `training` and for licences
like `Unicode-TOU`, `FTL` and `NAIST-2003` they are *every* example. Dropping them
looks obviously right and is a measured regression — capping at 50 KB moved R@1
0.9752 → 0.9600 at identical precision, because it drops 452 licences to 297.

## Reference results

Measured on Atarashi's cascade, `prevalence` config, n=11,680 single-licence nameable
blobs across 52 licences:

| | |
|---|---|
| R@1 | 0.9752 ±0.0028 |
| precision | 0.9844 |
| exact-set | 0.9742 |
| macro over 52 licences | 0.7387 |

Scoring this pool whole found a defect four other benchmarks missed: **286 of 639
verbatim Apache-2.0 files reported as `ImageMagick`**, whose licence is Apache-2.0 plus
extra text and so wins a longest-run comparison while covering less of its own
reference.

## Licence and citation

CC-BY-4.0, inherited from the source dataset. The blob texts are licence files, each
under its own terms. Please cite:

```bibtex
@article{shlicensedataset2023,
  title   = {The Software Heritage License Dataset (2022 Edition)},
  author  = {Gonzalez-Barahona, Jesus M. and Montes-Leon, Sergio and
             Robles, Gregorio and Zacchiroli, Stefano},
  journal = {Empirical Software Engineering},
  year    = {2023}, publisher = {Springer}
}
@inproceedings{zacchiroli2022licensevariants,
  title     = {A Large-scale Dataset of (Open Source) License Text Variants},
  author    = {Zacchiroli, Stefano},
  booktitle = {MSR}, year = {2022}
}
```

The selection, grouping, manifests and measured results are part of the Atarashi
dataset pipeline.
