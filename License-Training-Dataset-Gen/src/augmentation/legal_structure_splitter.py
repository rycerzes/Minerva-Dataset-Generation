from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
import re
import sys
from pathlib import Path

try:
    from ..builder.hybrid_merge import DatasetEntry
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from builder.hybrid_merge import DatasetEntry


class SplitterConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    max_window_size: int = 500
    overlap_tokens: int = 50
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    extract_placeholders: bool = True


class SplitFragment(BaseModel):
    model_config = {"protected_namespaces": ()}

    license_key: str
    fragment_text: str
    start_position: int
    end_position: int
    fragment_index: int
    total_fragments: int
    source: str
    placeholders: list[str] = Field(default_factory=list)
    is_first: bool = False
    is_last: bool = False


LEGAL_PLACEHOLDER_PATTERNS = [
    r"<[^>]+>",
    r"\{[^}]+\}",
    r"\[OWNER\]",
    r"\[YEAR\]",
    r"\[COPYRIGHT_HOLDER\]",
    r"\[DATE\]",
    r"\[VERSION\]",
    r"\[LICENSE_NAME\]",
    r"\[ORGANIZATION\]",
    r"\[PROJECT\]",
    r"\b(?:Copyright\s+\(?c?\)?\s*\d{4}(?:\s*-\s*\d{4})?\s+)[A-Z][^\n]+",
    r"\[NAME\s+OF\s+AUTHOR\]",
    r"\[NAME\s+OF\s+COPYRIGHT\s+OWNER\]",
    r"\[FULL\s+NAME\]",
    r"\[YEAR\s+OF\s+WORK\]",
]


class LegalStructureSplitter:
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()

    @staticmethod
    def _is_junk_fragment(text: str) -> bool:
        """True for sliding-window artifacts that shouldn't be labelled license text.

        Error-mining Nirjas false negatives showed the recall gap is mostly the
        splitter emitting garbage positives: ruler lines (``----*/``), form-field
        stubs (``Date:``, ``Email Address: Phone Number:``), and tiny mid-word
        cuts. The model is right to reject those; they're label noise. ponytail:
        conservative heuristics — anything with real prose passes.
        """
        t = text.strip()
        if not t:
            return True
        letters = sum(c.isalpha() for c in t)
        if letters < 20:
            return True  # tiny cuts, ruler lines, "Date:"
        if letters / len(t) < 0.5:
            return True  # punctuation/ruler heavy
        if len(re.findall(r"[A-Za-z]{2,}", t)) < 4:
            return True  # stubs
        # colon-heavy label:value forms with no sentence structure (form fields)
        if len(t) < 120 and t.count(":") >= 2 and not any(p in t for p in ".!?"):
            return True
        return False

    def _extract_placeholders(self, text: str) -> list[str]:
        placeholders: list[str] = []
        for pattern in LEGAL_PLACEHOLDER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            placeholders.extend(matches)
        return list(set(placeholders))

    def _split_into_sentences(self, text: str) -> list[str]:
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _find_best_split_point(self, text: str, target_position: int) -> int:
        if self.config.preserve_paragraphs:
            paragraphs = self._split_into_paragraphs(text)
            current_pos = 0
            for para in paragraphs:
                para_end = current_pos + len(para)
                if para_end >= target_position:
                    if abs(current_pos - target_position) < abs(
                        para_end - target_position
                    ):
                        return current_pos
                    else:
                        return para_end
                current_pos = para_end + 2

        if self.config.preserve_sentences:
            sentences = self._split_into_sentences(text)
            current_pos = 0
            for sentence in sentences:
                sentence_end = current_pos + len(sentence)
                if sentence_end >= target_position:
                    if abs(current_pos - target_position) < abs(
                        sentence_end - target_position
                    ):
                        return current_pos
                    else:
                        return sentence_end
                current_pos = sentence_end + 1
        return target_position

    def _sliding_window_split(
        self, text: str, license_key: str, source: str
    ) -> list[SplitFragment]:
        if not text or not text.strip():
            return []

        text = text.strip()
        text_length = len(text)

        max_size = self.config.max_window_size
        overlap = self.config.overlap_tokens

        if text_length <= max_size:
            if self._is_junk_fragment(text):
                return []
            placeholders = []
            if self.config.extract_placeholders:
                placeholders = self._extract_placeholders(text)
            return [
                SplitFragment(
                    license_key=license_key,
                    fragment_text=text,
                    start_position=0,
                    end_position=text_length,
                    fragment_index=0,
                    total_fragments=1,
                    source=source,
                    placeholders=placeholders,
                    is_first=True,
                    is_last=True,
                )
            ]

        fragments: list[SplitFragment] = []
        current_start = 0
        fragment_index = 0

        while current_start < text_length:
            hard_end = min(current_start + max_size, text_length)

            if hard_end < text_length:
                best = self._find_best_split_point(text, hard_end)
                # Only use the boundary split if it actually advances past the
                # current position; otherwise fall back to the hard limit so
                # we always make meaningful forward progress.
                current_end = best if best > current_start else hard_end
            else:
                current_end = hard_end

            # Snap the display slice off mid-word cuts (the overlap start and
            # the hard-limit end often land inside a word → "h Labs...", "ution.").
            # Advancing still uses current_end, so overlap logic is untouched.
            frag_start, frag_end = current_start, current_end
            if frag_start > 0 and text[frag_start - 1].isalnum() and text[frag_start].isalnum():
                sp = text.find(" ", frag_start, frag_end)
                if sp != -1:
                    frag_start = sp + 1
            if frag_end < text_length and text[frag_end - 1].isalnum() and text[frag_end].isalnum():
                sp = text.rfind(" ", frag_start, frag_end)
                if sp != -1:
                    frag_end = sp

            fragment_text = text[frag_start:frag_end].strip()

            if not self._is_junk_fragment(fragment_text):
                placeholders = []
                if self.config.extract_placeholders:
                    placeholders = self._extract_placeholders(fragment_text)

                fragments.append(
                    SplitFragment(
                        license_key=license_key,
                        fragment_text=fragment_text,
                        start_position=frag_start,
                        end_position=frag_end,
                        fragment_index=fragment_index,
                        total_fragments=0,  # renumbered after filtering
                        source=source,
                        placeholders=placeholders,
                        is_first=False,
                        is_last=False,
                    )
                )
                fragment_index += 1

            # Advance by (window - overlap), but guarantee at least 1 char of
            # progress so the loop can never stall or regress.
            next_start = current_end - overlap
            if next_start <= current_start:
                next_start = current_end  # no overlap when window is tiny
            current_start = next_start

            if current_start >= text_length:
                break

        total = len(fragments)
        for i, frag in enumerate(fragments):
            frag.total_fragments = total
            frag.is_first = i == 0
            frag.is_last = i == total - 1

        return fragments

    def split_license(self, entry: DatasetEntry) -> list[SplitFragment]:
        if not entry.license_text:
            return []

        return self._sliding_window_split(
            entry.license_text,
            entry.license_key,
            entry.source.value,
        )

    def split_dataset(self, dataset: list[DatasetEntry]) -> list[SplitFragment]:
        all_fragments: list[SplitFragment] = []
        for entry in dataset:
            fragments = self.split_license(entry)
            all_fragments.extend(fragments)
        return all_fragments

    def get_fragment_statistics(self, fragments: list[SplitFragment]) -> dict:
        if not fragments:
            return {
                "total_fragments": 0,
                "licenses_with_fragments": 0,
                "avg_fragments_per_license": 0,
                "fragments_with_placeholders": 0,
                "unique_licenses": 0,
            }

        license_counts: dict[str, int] = {}
        for frag in fragments:
            license_counts[frag.license_key] = (
                license_counts.get(frag.license_key, 0) + 1
            )

        fragments_with_placeholders = sum(1 for f in fragments if f.placeholders)

        return {
            "total_fragments": len(fragments),
            "licenses_with_fragments": len(license_counts),
            "avg_fragments_per_license": len(fragments) / len(license_counts)
            if license_counts
            else 0,
            "fragments_with_placeholders": fragments_with_placeholders,
            "unique_licenses": len(set(f.license_key for f in fragments)),
        }


def split_license_texts(
    dataset: list[DatasetEntry],
    max_window_size: int = 500,
    overlap_tokens: int = 50,
    preserve_sentences: bool = True,
    preserve_paragraphs: bool = True,
    extract_placeholders: bool = True,
) -> list[SplitFragment]:
    config = SplitterConfig(
        max_window_size=max_window_size,
        overlap_tokens=overlap_tokens,
        preserve_sentences=preserve_sentences,
        preserve_paragraphs=preserve_paragraphs,
        extract_placeholders=extract_placeholders,
    )
    splitter = LegalStructureSplitter(config)
    return splitter.split_dataset(dataset)


if __name__ == "__main__":  # self-check for junk filtering + boundary snapping
    s = LegalStructureSplitter()
    junk = [
        "Date:", "----------------------------------------------*/",
        "SDLC Personal ID:   Email Address:   Phone Number:", "*****",
    ]
    real = [
        "without express or implied warranty of any kind.",
        "Permission to copy, use, modify, sell and distribute this software is granted "
        "provided this copyright notice appears in all copies.",
    ]
    for t in junk:
        assert s._is_junk_fragment(t), f"should be junk: {t!r}"
    for t in real:
        assert not s._is_junk_fragment(t), f"should be kept: {t!r}"
    # Boundary snapping: a long text windows without mid-word starts/ends.
    long = ("This license permits redistribution and modification of the covered work "
            "under the following conditions. " * 20)
    frags = s._sliding_window_split(long, "test-1.0", "scancode")
    assert len(frags) > 1, "expected multiple windows"
    for f in frags:
        assert not s._is_junk_fragment(f.fragment_text), f"junk leaked: {f.fragment_text[:40]!r}"
    print(f"OK: junk filter + snapping work; {len(frags)} clean fragments from demo license")
