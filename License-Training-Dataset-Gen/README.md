# Minerva Dataset Pipeline

Generates training datasets for two FOSSology ML components:

- **Atarashi** - multi-class license identification (which license is this text?)
- **Nirjas** - binary license detection (is this a license-related comment?)

## Overview

The pipeline pulls license texts from two upstream sources, augments them into
training fragments, generates hard negatives, and exports the results as
Hugging Face `DatasetDict` objects ready for model training.

```
ScanCode LicenseDB   ─┐
                      ├─ merge ─ split (sliding window) ─ augment ─ export
FOSSology licenseRef ─┘
```

### Data sources

- [ScanCode LicenseDB](https://scancode-licensedb.aboutcode.org) - SPDX and
  non-SPDX licenses with full texts
- [FOSSology licenseRef.json](https://github.com/fossology/fossology) -
  FOSSology's internal license reference database
- [The Stack Smol](https://huggingface.co/datasets/bigcode/the-stack-smol) - generic code comments for the Nirjas negative class

### Augmentation stages

1. **Sliding-window splitting** (`LegalStructureSplitter`) - each license text
   is split into overlapping fixed-size fragments to increase sample diversity.
2. **Rare-license LLM augmentation** (`RareLicenseAugmenter`) - licenses with
   fewer than `--rare-license-threshold` fragments receive LLM-paraphrased
   variants. Variants are filtered by character 3-gram Jaccard similarity
   (`min_similarity`, `max_similarity`) to avoid drift and near-duplicates.
3. **Surgical LLM injection** (`SurgicalLLMInjector`) - targeted synthetic
   samples for under-represented categories.
4. **Hard negative generation** (`HardNegativeGenerator`) - LLM-generated
   samples that look license-like but are not, used as Nirjas negatives.
5. **Near-dedup** (`datasketch` MinHashLSH, threshold 0.8) - removes near-
   duplicate fragments within and across sources before export.
6. **Class balancing** (`NirjasClassBalancer`) - balances the Nirjas binary
   classes; use `--max-nirjas-samples` to cap total size.
7. **Split hygiene** (`exporter/cleaning.py`) - runs inside the export, so what
   the pipeline writes is what ships. Quarantines sub-60-character fragments
   (generic boilerplate whose label is not inferable from the text), then removes
   cross-split leakage with MinHash near-dedup. Train is never dropped from, so
   every validation/test label stays present in train. Previously this was a
   manual post-pass; the documented pipeline emitted a leaky dataset and only an
   undocumented script produced the usable one.

### Output

| Path | Task | Format |
|---|---|---|
| `output/atarashi` | Multi-class license identification | HF DatasetDict |
| `output/nirjas` | Binary license detection | HF DatasetDict |

Each DatasetDict has `train` and `test` splits (default 80/20).

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```
uv sync
```

For LLM-dependent stages, copy `.env.example` to `.env` and fill in.

### Required — LLM provider

The model prefix follows LiteLLM convention: `provider/model-name`
(e.g. `openai/gpt-4o`, `anthropic/claude-3-haiku`, `groq/llama-3.1-70b`).

| Variable | Description |
|---|---|
| `LITELLM_API_BASE_URL` | LLM provider base URL |
| `LITELLM_API_KEY` | API key or token |
| `LITELLM_MODEL` | Model identifier with provider prefix |
| `LITELLM_TEMPERATURE` | Sampling temperature (default: 0.7) |
| `LITELLM_MAX_TOKENS` | Max tokens per response (default: 4096) |
| `LITELLM_RPM` | Rate limit, requests per minute (default: 60) |

### Optional — HuggingFace code comments

`HF_API_TOKEN` — needed only for fetching real code comments from
[bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol).
The dataset is public, so small fetches usually work without a token.
Generate one at https://huggingface.co/settings/tokens.

These LLM stages use the provider config:

| Stage | What it does |
|---|---|
| **Surgical LLM injection** | Fills placeholder variables in fragments with realistic entities |
| **Rare-license augmentation** | Paraphrases licenses with too few fragments |
| **Hard negative generation** | Generates license-like text for the Nirjas negative class |

Run with `--no-llm` to skip all LLM-dependent stages.

## Usage

### Production

```
uv run src/main.py \
  --cache-dir cache \
  --samples-per-category 20 \
  --code-comments-limit 100000 \
  --max-nirjas-samples 80000 \
  --export-dir output
```

Generates ~37k Atarashi samples. For Nirjas, `--max-nirjas-samples 80000` caps
the total at 80k (40k per class), preventing the upsampling pathology that arises
when `--samples-per-category 20` and `--code-comments-limit 100000` together make
the negative pool orders of magnitude larger than the positive license-fragment pool.
LLM responses are cached under `cache/` so re-runs only call the API for new licenses.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--cache-dir` | `cache` | Directory for LLM response cache |
| `--export-dir` | `output` | Root output directory |
| `--samples-per-category` | `5` | Hard-negative samples per category per license |
| `--code-comments-limit` | `25000` | Generic code comments added to Nirjas negatives |
| `--max-nirjas-samples` | _(none)_ | Hard cap on total Nirjas dataset size |
| `--rare-license-threshold` | `5` | Fragment count below which a license is augmented |
| `--rare-license-augment-count` | `5` | LLM variants to generate per rare license |
| `--hard-negative-limit` | _(none)_ | Max licenses to generate hard negatives for |
| `--train-split-ratio` | `0.8` | Train fraction |
| `--validation-split-ratio` | `0.1` | Validation fraction; remaining fraction is test |
| `--no-llm` | `false` | Skip all LLM stages |

## Project structure

```
src/
  config.py                  LLM config and rate limiter
  main.py                    Pipeline entry point
  fetchers/
    scancode.py              ScanCode LicenseDB fetcher
    fossology.py             FOSSology licenseRef.json fetcher
    code_comments.py         Code comment fetcher (the-stack-smol)
  builder/
    hybrid_merge.py          Merges ScanCode + FOSSology entries
    augmented_merge.py       Builds the final fragment pool with dedup
  augmentation/
    legal_structure_splitter.py  Sliding-window text splitter
    rare_license_augmenter.py    LLM augmentation for rare licenses
    llm_synthetic.py             Surgical LLM injection
    hard_negative_generator.py   Hard negative generation
    class_balancing.py           Nirjas class balancer
    llm_cache.py                 Persistent LLM response cache
  exporter/
    dataset_export.py        Exports HF DatasetDicts
    cleaning.py              Split hygiene: junk quarantine + cross-split dedup
  utils.py                   Near-dedup (MinHashLSH)
  run_eval.py                Evaluation entry point (counterpart to main.py)
  evaluation/
    retrieval.py             Encoders, top-k, metrics (shared machinery)
    references.py            Reference index + license-key mapping
    corpus.py                Query corpora from real source files
    crossref.py              Cross-rendering eval (non-circular)
    spdx_tag.py              SPDX-tag eval, regime-split (non-circular)
    cascade.py               precision@coverage with abstention
    real_corpus.py           ScanCode-labelled corpus eval (optimistic)
    nirjas_gate.py           Binary gate vs real source files
    nomos.py                 Binary gate vs FOSSology nomos testdata
tests/                       Split-hygiene and evaluation invariants
```

## Evaluation

Evaluation is a separate entry point, not a pipeline stage: it reads build outputs
plus a real-code corpus, runs on a different cadence, and needs heavier optional
dependencies that a dataset regeneration should not have to install.

```
uv sync --extra eval
uv run src/run_eval.py --list
```

| suite | measures | trust |
|---|---|---|
| `crossref` | ScanCode index vs FOSSology's independent text for the same license | **non-circular** |
| `spdx-tag` | real headers, author-declared SPDX labels, split by regime | **non-circular** |
| `cascade` | precision@coverage with abstention | non-circular corpus |
| `real-corpus` | broad ScanCode-labelled corpus | *optimistic* — shares label provenance with its own notice layer |
| `nirjas-gate`, `nomos` | the binary gate against real code / nomos testdata | — |

Two rules for reading the output:

- **Prefer macro over micro.** apache-2.0 is ~64% of real-world license
  occurrences, so micro tracks the class mix rather than quality — it swings
  0.605 → 0.782 on the SPDX-tag suite purely with the stratification cap, while
  macro holds at ~0.65.
- **Watch the regime split.** ~72% of real SPDX-tagged files carry no license
  prose once the tag is stripped, and the correct answer for those is UNKNOWN.
  Scoring them as misses understates the engine; the suite reports them
  separately and prints the similarity distribution used to calibrate an
  abstention threshold.

`crossref` runs from the cached license JSONs alone. The others need the
the-stack-smol snapshot; set `STACK_SNAPSHOT` or let it find the HuggingFace cache.

### Datasets on the Hub

- [`rycerzes/atarashi-dataset`](https://huggingface.co/datasets/rycerzes/atarashi-dataset)
- [`rycerzes/nirjas-dataset`](https://huggingface.co/datasets/rycerzes/nirjas-dataset)

The Atarashi split is **training data, not a benchmark** — ~86% of its fragments
are verbatim substrings of the reference texts they would be scored against, so
Recall@k there measures substring lookup. Use `crossref` or `spdx-tag` instead.
