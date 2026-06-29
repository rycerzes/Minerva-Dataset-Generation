"""Length-match the Nirjas positives to the negatives.

Root cause (see memory nirjas-provenance-shortcut): positives were windowed at
max_window_size=500 while LLM hard negatives are ~110 chars, so label correlates
with length and any classifier exploits that instead of license semantics.

Fix: re-window each positive into ~negative-length fragments using the pipeline's
own LegalStructureSplitter. Negatives untouched. Done per-split so no cross-split
leakage is introduced. Writes output/nirjas_lenmatched for re-eval.
"""

import sys
from pathlib import Path
import statistics as st

from datasets import load_from_disk, Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).parent / "src"))
from augmentation.legal_structure_splitter import LegalStructureSplitter, SplitterConfig

# target the negatives' length: median ~110. window 120 / overlap 20.
splitter = LegalStructureSplitter(SplitterConfig(max_window_size=120, overlap_tokens=20))

d = load_from_disk("output/nirjas")
out = {}
for name, split in d.items():
    rows = {"text": [], "label": [], "source": [], "negative_type": []}
    for r in split:
        if r["label"] != 0:  # negative -> keep as-is
            for k in rows:
                rows[k].append(r[k])
            continue
        # positive -> re-window
        frags = splitter._sliding_window_split(r["text"], "lic", r["source"])
        for f in frags:
            if len(f.fragment_text.strip()) < 20:  # drop tiny scraps
                continue
            rows["text"].append(f.fragment_text)
            rows["label"].append(0)
            rows["source"].append(r["source"])
            rows["negative_type"].append("")
    out[name] = Dataset.from_dict(rows)
    pos = [len(t) for t, l in zip(rows["text"], rows["label"]) if l == 0]
    neg = [len(t) for t, l in zip(rows["text"], rows["label"]) if l == 1]
    print(f"{name:11s} n={len(rows['text']):6d} "
          f"pos={rows['label'].count(0)} neg={rows['label'].count(1)} "
          f"| median_len pos={st.median(pos):.0f} neg={st.median(neg):.0f}")

DatasetDict(out).save_to_disk("output/nirjas_lenmatched")
print("wrote output/nirjas_lenmatched")
