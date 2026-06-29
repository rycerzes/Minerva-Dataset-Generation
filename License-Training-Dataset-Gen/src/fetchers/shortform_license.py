"""Synthesize short-form license positives (SPDX tags, one-line references).

Real license-related comments are frequently terse — `// SPDX-License-Identifier: MIT`,
`# Licensed under the Apache-2.0 license.` — but the corpus positives are all long
license prose. Without these forms the Nirjas classifier learns "license = prose" and
silently drops SPDX-tagged files before they reach Atarashi. Deterministic templates over
known license IDs; no LLM, no network.
"""

import random

PREFIXES = ["//", "#", "/*", " *", "--", ";;", "<!--", "%", "'", "rem"]
TEMPLATES = [
    "{p} SPDX-License-Identifier: {k}",
    "{p} Licensed under the {k} license.",
    "{p} Distributed under the terms of the {k} license. See LICENSE.",
    "{p} This file is licensed under {k}.",
    "{p} Released under {k}, see the LICENSE file in the project root.",
    "{p} Copyright (c) 2021 Acme Corp. SPDX-License-Identifier: {k}",
    "{p} This software is provided under the {k} license.",
    "{p} Use of this source code is governed by the {k} license.",
]


def generate_shortform_positives(license_keys, n_per: int = 2, seed: int = 0) -> list[str]:
    """Return deduplicated short-form license comment strings for the given license IDs."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    for k in license_keys:
        for _ in range(n_per):
            t = rng.choice(TEMPLATES).format(p=rng.choice(PREFIXES), k=k)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
