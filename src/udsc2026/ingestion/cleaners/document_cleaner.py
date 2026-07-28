"""Conservative text cleaning for Vietnamese legal documents."""

import math
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from udsc2026.ingestion.cleaners.abbreviations import build_abbreviation_dictionary
from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.cleaners.patterns import (
    HORIZONTAL_WHITESPACE,
    LEADING_BULLET,
    LEGAL_STRUCTURE_LINE,
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
}


def normalize_unicode_and_ocr(text: str) -> str:
    """Normalize Vietnamese Unicode and remove non-semantic OCR artifacts."""
    normalized = unicodedata.normalize("NFC", text)
    for source, replacement in OCR_REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def normalize_line(line: str) -> str:
    """Normalize horizontal whitespace and visual bullets in one source line."""
    without_bullet = LEADING_BULLET.sub(r"\1- ", line)
    return HORIZONTAL_WHITESPACE.sub(" ", without_bullet).strip()


def _page_edge_candidates(pages: Sequence[Sequence[str]]) -> Counter[str]:
    """Count short first/last lines of each page as header/footer candidates."""
    candidates: Counter[str] = Counter()
    for lines in pages:
        nonempty = [line for line in lines if line]
        for line in nonempty[:3] + nonempty[-3:]:
            if len(line) <= 160 and not LEGAL_STRUCTURE_LINE.match(line):
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
    if not line or PAGE_NUMBER_LINE.match(line) or WATERMARK_LINE.match(line):
        return True
    if in_table_of_contents and TOC_ENTRY.match(line):
        return True
    if len(line) <= 2 and ONLY_SYMBOLS.match(line) and not LEGAL_STRUCTURE_LINE.match(line):
        return True
    return False


def _remove_repeated_noise(text: str) -> Tuple[List[str], List[str]]:
    """Remove headers, footers, page markers, watermarks, and TOC entries."""
    pages = [[normalize_line(line) for line in page.split("\n")] for page in text.split("\f")]
    repeated_edges = repeated_page_edges(pages)
    kept_lines: List[str] = []
    removed_lines: List[str] = []
    in_table_of_contents = False

    for page in pages:
        for line in page:
            if TABLE_OF_CONTENTS.match(line):
                in_table_of_contents = True
                removed_lines.append(line)
                continue
            if in_table_of_contents and LEGAL_STRUCTURE_LINE.match(line):
                in_table_of_contents = False
            if line.casefold() in repeated_edges or is_garbage_line(
                line, in_table_of_contents
            ):
                if line:
                    removed_lines.append(line)
                continue
            kept_lines.append(line)
        if kept_lines and kept_lines[-1]:
            kept_lines.append("")
    return kept_lines, removed_lines


def _should_merge(previous: str, current: str) -> bool:
    """Decide whether a PDF hard line break belongs inside one legal sentence."""
    if not previous or not current:
        return False
    if LEGAL_STRUCTURE_LINE.match(previous) or LEGAL_STRUCTURE_LINE.match(current):
        return False
    return previous[-1] not in ".;:?!"


def merge_hard_wrapped_lines(lines: Iterable[str]) -> str:
    """Join continuation lines while retaining legal structural line boundaries."""
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


def clean_text(text: str) -> Tuple[str, List[str]]:
    """Apply all text-only cleaning rules and report removed noise lines."""
    normalized_text = normalize_unicode_and_ocr(text)
    kept_lines, removed_lines = _remove_repeated_noise(normalized_text)
    return merge_hard_wrapped_lines(kept_lines), removed_lines


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
