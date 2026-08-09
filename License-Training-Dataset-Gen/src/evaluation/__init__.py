"""Evaluation suites for the exported datasets and the license matchers.

Evaluation is a *consumer* of build outputs, not a build stage: it reads
``output/`` plus a real-code corpus, runs on a different cadence from dataset
generation, and pulls heavier optional dependencies. So it lives beside the
pipeline rather than inside ``run_pipeline`` — ``uv run src/run_eval.py <suite>``.

Layers:

* ``retrieval``  — encoders, top-k, metrics (shared machinery)
* ``references`` — reference index construction and license-key mapping
* ``corpus``     — query corpora drawn from real source files

Suites, ordered by how much you should trust them:

* ``crossref``    non-circular: two independent renderings of the same license
* ``spdx_tag``    non-circular: author-declared labels, regime-split
* ``cascade``     precision@coverage with abstention, on the SPDX-tag corpus
* ``real_corpus`` broad but *optimistic* — ScanCode labels, ScanCode-derived rules
* ``nirjas_gate`` / ``nomos``  the binary gate, against real code and nomos testdata
"""
