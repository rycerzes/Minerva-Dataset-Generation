"""Fix the SPDX/short-reference recall hole by adding short-form license positives.

The model misses `// SPDX-License-Identifier: MIT` etc. because every training positive
is long license prose. Synthesize short-form positives from the known license IDs, add to
train, retrain word+char TF-IDF+LR, and re-probe. Reports test F1 (regression check) and
held-out short-form recall (the thing we're fixing).
"""

import random
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import hstack

rng = random.Random(0)
PREFIX = ["//", "#", "/*", " *", "--", ";;", "<!--"]
TEMPLATES = [
    "{p} SPDX-License-Identifier: {k}",
    "{p} Licensed under the {k} license.",
    "{p} Distributed under the terms of the {k} license. See LICENSE.",
    "{p} This file is licensed under {k}.",
    "{p} Released under {k}, see the LICENSE file in the project root.",
    "{p} Copyright (c) 2021 Acme Corp. SPDX-License-Identifier: {k}",
]


def gen(keys, n_per=2):
    out = []
    for k in keys:
        for _ in range(n_per):
            out.append(rng.choice(TEMPLATES).format(p=rng.choice(PREFIX), k=k))
    return out


class Model:
    """word+char TF-IDF + LR with its own vectorizers."""
    def __init__(self):
        self.w = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 max_features=100000, sublinear_tf=True)

    def _f(self, texts, fit=False):
        if fit:
            return hstack([self.w.fit_transform(texts), self.c.fit_transform(texts)]).tocsr()
        return hstack([self.w.transform(texts), self.c.transform(texts)]).tocsr()

    def fit(self, X, y):
        self.clf = LogisticRegression(max_iter=1000, C=10.0).fit(self._f(X, fit=True), y)
        return self

    def proba(self, texts):
        return self.clf.predict_proba(self._f(texts))[:, 1]


keys = sorted(set(load_from_disk("output/atarashi")["train"]["license_key"]))
short = gen(keys, n_per=2)
rng.shuffle(short)
cut = int(len(short) * 0.85)
short_tr, short_te = short[:cut], short[cut:]  # held-out short-forms = the fix metric

d = load_from_disk("output/nirjas")
tr, te = d["train"], d["test"]
base_text = list(tr["text"])
base_y = [1 if l == 0 else 0 for l in tr["label"]]
te_text, te_y = list(te["text"]), [1 if l == 0 else 0 for l in te["label"]]

PROBE_LIC = [
    "// SPDX-License-Identifier: Apache-2.0",
    "# SPDX-License-Identifier: MIT",
    "# Licensed under the MIT License. See LICENSE file in the project root.",
    "/* This program is free software: redistribute under the GNU GPL v3. */",
]
PROBE_NON = ["// TODO: fix this race condition", "# returns the number of active users",
             "/* initialize the buffer */"]


def report(tag, model):
    f1 = f1_score(te_y, (model.proba(te_text) > 0.5).astype(int))
    sr = (model.proba(short_te) > 0.5).mean()
    pl = [round(x, 3) for x in model.proba(PROBE_LIC)]
    pn = [round(x, 3) for x in model.proba(PROBE_NON)]
    print(f"{tag:16s} test_F1={f1:.4f} shortform_recall={sr:.4f}\n"
          f"                 probe_lic={pl}  probe_non={pn}")


report("BASELINE", Model().fit(base_text, base_y))
report("WITH SHORTFORM", Model().fit(base_text + short_tr, base_y + [1] * len(short_tr)))
