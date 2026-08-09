"""Conservative text cleaning for Vietnamese legal documents."""

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from udsc2026.ingestion.cleaners.abbreviations import build_abbreviation_dictionary
from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.cleaners.patterns import (
    HORIZONTAL_WHITESPACE,
    LEADING_BULLET,
    LEGAL_STRUCTURE_LINE,
    LIGHT_OCR_WORD_REPLACEMENTS,
    MULTIPLE_BLANK_LINES,
    ONLY_SYMBOLS,
    PAGE_NUMBER_LINE,
    TABLE_OF_CONTENTS,
    TOC_ENTRY,
    WATERMARK_LINE,
)
from udsc2026.ingestion.readers.models import RawDocument

OCR_REPLACEMENTS = {
    "\u00a0": " ",
    "\u00ad": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufffd": "",
    # Latin eth is a recurring PDF/OCR confusable for Vietnamese D with
    # stroke. Normalize it before structural-line detection so ``Ðiều`` is
    # neither missed nor merged into the preceding chapter heading.
    "Ð": "Đ",
    "ð": "đ",
}

_BARE_NUMERIC_LINE = re.compile(r"^\d{1,4}$")
_COMPACT_NUMERIC_DATA_LINE = re.compile(
    r"^[+-]?\d+(?:[./:-]\d+)+(?:\s*(?:%|‰))?$", re.UNICODE
)

# These are whole-word substitutions only. Replacing the bare sequence ``uý``
# would incorrectly turn valid words such as ``quý`` into ``qúy``.
VIETNAMESE_VOWEL_MAP = {
    "hoà": "hòa",
    "hoả": "hỏa",
    "hoã": "hõa",
    "hoè": "hòe",
    "hoé": "hóe",
    "hoẻ": "hỏe",
    "hoẽ": "hõe",
    "hoẹ": "họe",
    "toà": "tòa",
    "toả": "tỏa",
    "toã": "tõa",
    "thuý": "thúy",
    "thuỳ": "thùy",
    "thuỷ": "thủy",
    "thuỹ": "thũy",
    "thuỵ": "thụy",
}


def _match_case(source: str, replacement: str) -> str:
    """Keep all-caps and title-case legal text readable after replacement."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def normalize_vietnamese_vowels(text: str) -> str:
    """Normalize selected old-style Vietnamese tone placements to modern style."""
    normalized = text
    for old_style, new_style in VIETNAMESE_VOWEL_MAP.items():
        pattern = re.compile(r"\b{0}\b".format(re.escape(old_style)), re.IGNORECASE)
        normalized = pattern.sub(
            lambda match: _match_case(match.group(0), new_style), normalized
        )
    return normalized


def normalize_unicode_and_ocr(text: str) -> str:
    """Normalize Vietnamese Unicode and remove non-semantic OCR artifacts."""
    normalized = unicodedata.normalize("NFC", text)
    for source, replacement in OCR_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    for source, replacement in LIGHT_OCR_WORD_REPLACEMENTS.items():
        normalized = re.sub(
            (
                rf"^(?P<prefix>[ \t]*(?:(?:[-–—•*]|\d+[.)])[ \t]*)?)"
                rf"(?P<marker>{re.escape(source)})(?=[ \t]+\d)"
            ),
            lambda match: (
                match.group("prefix") + _match_case(match.group("marker"), replacement)
            ),
            normalized,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalize_vietnamese_vowels(normalized)


def normalize_line(line: str) -> str:
    """Normalize horizontal whitespace and visual bullets in one source line."""
    without_bullet = LEADING_BULLET.sub(r"\1- ", line)
    return HORIZONTAL_WHITESPACE.sub(" ", without_bullet).strip()


def _page_edge_candidates(pages: Sequence[Sequence[str]]) -> Counter[str]:
    """Count short first/last lines of each page as header/footer candidates."""
    candidates: Counter[str] = Counter()
    for lines in pages:
        nonempty = [line for line in lines if line]
        edge_lines = {line.casefold(): line for line in nonempty[:3] + nonempty[-3:]}
        for line in edge_lines.values():
            if (
                len(line) <= 160
                and not LEGAL_STRUCTURE_LINE.match(line)
                and not _is_atomic_data_line(line)
                and not WATERMARK_LINE.match(line)
            ):
                candidates[line.casefold()] += 1
    return candidates


def repeated_page_edges(pages: Sequence[Sequence[str]]) -> Set[str]:
    """Return repeated edge lines present on at least 60% of multi-page input."""
    if len(pages) < 2:
        return set()
    threshold = max(2, math.ceil(len(pages) * 0.6))
    candidates = _page_edge_candidates(pages)
    return {line for line, count in candidates.items() if count >= threshold}


def is_garbage_line(line: str, in_table_of_contents: bool) -> bool:
    """Identify non-legal noise without discarding legal structure or punctuation."""
    if not line or PAGE_NUMBER_LINE.match(line):
        return True
    if in_table_of_contents and TOC_ENTRY.match(line):
        return True
    return False


def _is_atomic_data_line(line: str) -> bool:
    """Return whether a standalone line can be a meaningful table/formula cell."""

    return bool(
        _BARE_NUMERIC_LINE.fullmatch(line)
        or _COMPACT_NUMERIC_DATA_LINE.fullmatch(line)
        or ONLY_SYMBOLS.fullmatch(line)
    )


def _remove_repeated_noise(text: str) -> Tuple[str, List[str]]:
    """Remove headers, footers, page markers, watermarks, and TOC entries."""
    pages = [
        [normalize_line(line) for line in page.split("\n")] for page in text.split("\f")
    ]
    repeated_edges = repeated_page_edges(pages)
    cleaned_pages: List[str] = []
    removed_lines: List[str] = []
    in_table_of_contents = False
    seen_repeated_edges: Set[str] = set()

    for page in pages:
        kept_lines: List[str] = []
        for line in page:
            if TABLE_OF_CONTENTS.match(line):
                in_table_of_contents = True
                removed_lines.append(line)
                continue
            # A TOC line may itself start with "Chương" or "Điều". It only
            # ends the TOC block when it is a real structure line, not an entry.
            if (
                in_table_of_contents
                and LEGAL_STRUCTURE_LINE.match(line)
                and not TOC_ENTRY.match(line)
            ):
                in_table_of_contents = False
            folded_line = line.casefold()
            if is_garbage_line(line, in_table_of_contents):
                if line:
                    removed_lines.append(line)
                continue
            if folded_line in repeated_edges:
                # Repeated page headers can still contain the document name or
                # another useful retrieval term. Keep one canonical occurrence
                # and remove only subsequent copies.
                if folded_line not in seen_repeated_edges:
                    seen_repeated_edges.add(folded_line)
                    kept_lines.append(line)
                    continue
                if line:
                    removed_lines.append(line)
                continue
            kept_lines.append(line)
        cleaned_pages.append("\n".join(kept_lines).strip())
    return "\f".join(cleaned_pages), removed_lines


def _should_merge(previous: str, current: str) -> bool:
    """Decide whether a PDF hard line break belongs inside one legal sentence."""
    if not previous or not current:
        return False
    if _is_atomic_data_line(previous) or _is_atomic_data_line(current):
        return False
    # Preserve a line that *starts* a new Điều/Khoản/Điểm. A structural line
    # may still have a hard-wrapped continuation on its following line.
    if LEGAL_STRUCTURE_LINE.match(current):
        return False
    return previous[-1] not in ".;:?!"


def _merge_page_lines(lines: Iterable[str]) -> str:
    """Join continuation lines inside one page only."""
    merged: List[str] = []
    for line in lines:
        if not line:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if merged and merged[-1] and _should_merge(merged[-1], line):
            separator = "" if merged[-1].endswith("-") and line[:1].islower() else " "
            merged[-1] = merged[-1].rstrip("-") + separator + line
        else:
            merged.append(line)
    return MULTIPLE_BLANK_LINES.sub("\n\n", "\n".join(merged)).strip()


def merge_hard_wrapped_lines(text_with_pages: str) -> str:
    """Join hard wraps without ever combining text from different PDF pages."""
    cleaned_pages = [
        _merge_page_lines(page.splitlines()) for page in text_with_pages.split("\f")
    ]
    return "\f".join(page for page in cleaned_pages if page)


def clean_text(text: str) -> Tuple[str, List[str]]:
    """Apply all text-only cleaning rules and report removed noise lines."""
    normalized_text = normalize_unicode_and_ocr(text)
    text_without_noise, removed_lines = _remove_repeated_noise(normalized_text)
    return merge_hard_wrapped_lines(text_without_noise), removed_lines


def clean_document(document: RawDocument) -> CleanDocument:
    """Clean one RawDocument without altering its identity or source metadata."""
    cleaned_text, removed_lines = clean_text(document.raw_text)
    abbreviations = build_abbreviation_dictionary(cleaned_text)
    metadata: Dict[str, object] = dict(document.metadata)
    metadata.update(
        {
            "cleaning": {
                "unicode_normalization": "NFC",
                "removed_line_count": len(removed_lines),
                "removed_line_reason_counts": _removed_line_reason_counts(
                    removed_lines
                ),
                "page_delimiter": "form_feed",
            }
        }
    )
    return CleanDocument(
        doc_id=document.doc_id,
        source_path=document.source_path,
        title=document.title,
        cleaned_text=cleaned_text,
        file_format=document.file_format,
        metadata=metadata,
        abbreviations=abbreviations,
        removed_lines=removed_lines,
    )


def _removed_line_reason_counts(lines: Sequence[str]) -> Dict[str, int]:
    """Classify every removed line so corpus audits can reject new loss classes."""

    counts: Counter[str] = Counter()
    for line in lines:
        if PAGE_NUMBER_LINE.fullmatch(line):
            reason = "explicit_page_marker"
        elif TABLE_OF_CONTENTS.fullmatch(line):
            reason = "table_of_contents_header"
        elif TOC_ENTRY.fullmatch(line):
            reason = "table_of_contents_entry"
        else:
            reason = "repeated_page_edge_duplicate"
        counts[reason] += 1
    return dict(sorted(counts.items()))
