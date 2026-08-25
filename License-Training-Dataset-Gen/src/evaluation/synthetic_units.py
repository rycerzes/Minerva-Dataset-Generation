"""Synthesise a short-form unit for licences that have none — built, NOT shipped.

**Read this before proposing it again.** The construction works and the output is not
used: its benefit cannot be measured and its risk is all that can be, which does not
clear the bar this project applies to everything else. See the engine report §2k.

288 licences the agent can name have no notice/reference rule, so they are matchable
only against a full body — the configuration measured at R@1 ~0.004 on real headers.
Mining does not reach them: SPDX ships a standardLicenseHeader for 0 of them, the
Software Heritage blobs give whole bodies rather than notices, and only 9 have any
notice-register span in the-stack. They are rare in public code, not missing from our
tooling.

What we do have is every one of their bodies. So pick the span of the body that is
most *specific to it* — the window whose k-grams appear in the fewest other licence
bodies — and index that. Deterministic, needs no corpus, and applies to all 320.

The risk is that a synthetic span is generic enough to fire on unrelated text. Measured
on 1,000 DEP-5 no-signal rows and the 66-blob tail, adding all 314 units changed
**nothing**: false answers 52/1000 either way, tail R@1 0.8182 either way. Inert.

The benefit is structurally unmeasurable. The span is extracted *from* the body, so a
blob that *is* that body matches it tautologically; the real question is whether it
matches files that merely *reference* the licence, and those barely exist for these
licences — which is why they have no rules in the first place. Of the 132 Software
Heritage blobs carrying an evidence-less licence, 112 went into the training pool and
only 20 across 4 licences remain unused.

What this needs is a corpus where these licences are *referenced*. That corpus is what
does not exist, and finding one is the actual prerequisite — not more synthesis.

    python -m evaluation.synthetic_units      # writes synthetic_units.json
"""
import collections, json, os, re, sys
import pandas as pd
from atarashi.libs.normalize import tokens
from atarashi.libs.references import load_notice_units

S = os.path.dirname(os.path.abspath(__file__))
K = 6            # k-gram used to measure specificity
# A copyright line is unique to the body and useless as a reference: a file adopting
# the licence carries its own holder and year, so the span would never match there and
# could match an unrelated file that happens to quote the same notice. Windows are
# scored on the operative text only.
_NOISE = re.compile(r"\bcopyright\b|\b(19|20)\d\d\b|\ball rights reserved\b")
WINDOW = 18      # tokens emitted; comfortably above the indexing floor of 5

df = pd.read_csv("/Users/swapnil.dutta/Projects/oss/fossology-gsoc/atarashi/"
                 "atarashi/data/licenses/processedLicenses.csv")
indexed = collections.Counter(u[0] for u in load_notice_units(df["shortname"]))
bodies = {str(r["shortname"]): tokens(str(r["processed_text"]))
          for _, r in df.iterrows() if isinstance(r["processed_text"], str)}
need = [n for n in bodies if not indexed.get(n) and len(bodies[n]) >= WINDOW]
print(f"licences needing a unit: {len(need)}")

# document frequency of each k-gram across licence bodies
dfreq = collections.Counter()
for name, toks in bodies.items():
    seen = {tuple(toks[i:i + K]) for i in range(len(toks) - K + 1)}
    dfreq.update(seen)
print(f"k-gram vocabulary: {len(dfreq)}")

units, skipped = [], 0
for name in need:
    toks = bodies[name]
    grams = [tuple(toks[i:i + K]) for i in range(len(toks) - K + 1)]
    scores = [dfreq[g] for g in grams]
    span = WINDOW - K + 1
    if len(scores) < span:
        skipped += 1
        continue
    # cheapest window by total document frequency: the least shared stretch of text
    def clean(i):
        return not _NOISE.search(" ".join(toks[i:i + WINDOW]))

    run = sum(scores[:span])
    best = (run, 0) if clean(0) else (float("inf"), -1)
    for i in range(1, len(scores) - span + 1):
        run += scores[i + span - 1] - scores[i - 1]
        if run < best[0] and clean(i):
            best = (run, i)
    total, start = best
    if start < 0:          # every window mentions a holder or a year
        skipped += 1
        continue
    text = " ".join(toks[start:start + WINDOW])
    # A window whose every k-gram is unique to this licence scores exactly `span`.
    units.append([name, text, [], round(total / span, 2)])

uniq = sum(1 for u in units if u[3] <= 1.0)
print(f"synthesised {len(units)} units ({skipped} bodies too short)")
print(f"  perfectly specific (every k-gram unique to the licence): {uniq}")
print(f"  median shared-licence count per k-gram: "
      f"{sorted(u[3] for u in units)[len(units)//2]:.2f}")
json.dump([u[:3] for u in units], open(f"{S}/synthetic_units.json", "w"))
print("examples:")
for u in units[:4]:
    print(f"   [{u[3]:5.2f}] {u[0]:34s} {u[1][:70]!r}")
