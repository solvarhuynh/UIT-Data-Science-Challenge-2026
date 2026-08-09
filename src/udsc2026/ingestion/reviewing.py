"""Manual-review classification and reporting for BTC ingestion."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from udsc2026.ingestion.cleaners.models import CleanDocument

_PRACTICAL_FALLBACK_MARKERS = re.compile(
    r"(?im)\b(?:Lời nói đầu|Phạm vi điều chỉnh|Đối tượng áp dụng|"
    r"Giải thích từ ngữ|Quy định chung|Điều khoản thi hành)\b"
)
_ARTICLE_LIKE_MARKERS = re.compile(
    r"(?im)(?:^|\s)(?:Điều|Dieu|DIEU)\s+\d+|(?:^|\s)(?:Khoản|Khoan|KHOAN)\s+\d+|(?:^|\s)(?:Điểm|Diem|DIEM)\s+[a-zđ]"
)
_LEGAL_SECTION_MARKERS = re.compile(
    r"(?im)^\s*(?:Chương|Chuong|CHUONG|Mục|Muc|MUC|Điều|Dieu|DIEU|Khoản|Khoan|KHOAN|Điểm|Diem|DIEM)\b"
)


class ManualReviewItem(BaseModel):
    """One document that needs review or fallback classification."""

    doc_id: str
    reason: str
    title: Optional[str] = None
    source_path: Optional[str] = None
    char_count: int
    token_count: int
    removed_line_count: int = 0
    note: Optional[str] = None


class ManualReviewBucket(BaseModel):
    """Grouped manual-review documents with guidance for the next action."""

    reason: str
    count: int
    document_ids: List[str] = Field(default_factory=list)
    sample_document_ids: List[str] = Field(default_factory=list)
    recommendation: str


class ManualReviewBreakdown(BaseModel):
    """Partition of the manual-review corpus into actionable reasons."""

    schema_version: str = "manual-review-breakdown-v1"
    total_documents: int
    buckets: List[ManualReviewBucket] = Field(default_factory=list)


class ManualReviewDocumentDetail(BaseModel):
    """Per-document metadata used to inspect a manual-review bucket."""

    doc_id: str
    title: Optional[str] = None
    source_path: Optional[str] = None
    char_count: int
    token_count: int
    removed_line_count: int = 0
    note: Optional[str] = None


class ManualReviewDocumentReport(BaseModel):
    """Detailed list of manual-review documents for a single reason."""

    schema_version: str = "manual-review-doc-report-v1"
    reason: str
    count: int
    documents: List[ManualReviewDocumentDetail] = Field(default_factory=list)


def build_manual_review_breakdown(
    documents: Sequence[CleanDocument],
    manual_review_ids: Sequence[str],
    sample_size: int = 10,
) -> ManualReviewBreakdown:
    """Classify manual-review docs into actionable buckets."""
    by_id = {document.doc_id: document for document in documents}
    grouped: Dict[str, List[str]] = defaultdict(list)
    for doc_id in manual_review_ids:
        document = by_id.get(doc_id)
        if document is None:
            continue
        reason, _ = classify_manual_review_document(document)
        grouped[reason].append(doc_id)

    buckets = [
        ManualReviewBucket(
            reason=reason,
            count=len(doc_ids),
            document_ids=doc_ids,
            sample_document_ids=doc_ids[:sample_size],
            recommendation=_recommendation_for(reason),
        )
        for reason, doc_ids in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]
    return ManualReviewBreakdown(
        total_documents=len(manual_review_ids),
        buckets=buckets,
    )


def build_manual_review_document_report(
    documents: Sequence[CleanDocument],
    manual_review_ids: Sequence[str],
    reason: str,
) -> ManualReviewDocumentReport:
    """Build a detailed document list for one manual-review reason."""
    by_id = {document.doc_id: document for document in documents}
    items: List[ManualReviewDocumentDetail] = []
    for doc_id in manual_review_ids:
        document = by_id.get(doc_id)
        if document is None:
            continue
        document_reason, note = classify_manual_review_document(document)
        if document_reason != reason:
            continue
        items.append(
            ManualReviewDocumentDetail(
                doc_id=document.doc_id,
                title=document.title,
                source_path=document.source_path,
                char_count=len(document.cleaned_text.strip()),
                token_count=_token_count(document.cleaned_text),
                removed_line_count=len(document.removed_lines),
                note=note,
            )
        )
    return ManualReviewDocumentReport(reason=reason, count=len(items), documents=items)


def write_manual_review_breakdown(
    breakdown: ManualReviewBreakdown, output_path: str | Path
) -> Path:
    """Persist the breakdown as pretty JSON for TV4 audit use."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(breakdown.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def write_manual_review_document_report(
    report: ManualReviewDocumentReport, output_path: str | Path
) -> Path:
    """Persist a single-reason manual-review report as pretty JSON."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def classify_manual_review_document(
    document: CleanDocument,
) -> tuple[str, Optional[str]]:
    """Assign one primary reason for a manual-review document."""
    text = document.cleaned_text.strip()
    char_count = len(text)
    token_count = _token_count(text)
    removed_line_count = len(document.removed_lines)

    if char_count == 0:
        return "cleaner_cleared_text", "cleaned text is empty"

    if _looks_like_fallback_candidate(document):
        return (
            "fallback_chunkable_no_article",
            "legal preamble or glossary section can be fallback-chunked safely",
        )

    if _ARTICLE_LIKE_MARKERS.search(text) or _LEGAL_SECTION_MARKERS.search(text):
        return (
            "regex_recoverable",
            "article-like markers exist but parser likely missed them due layout/OCR",
        )

    if _looks_like_ocr_noise(text, removed_line_count):
        return "ocr_noise", "high noise density or table-like OCR artefacts"

    if token_count < 20:
        return "cleaner_cleared_text", "cleaned text is too small to recover safely"

    return "no_article_structure", "content is real but lacks article-level structure"


def _looks_like_fallback_candidate(document: CleanDocument) -> bool:
    text = document.cleaned_text
    if _PRACTICAL_FALLBACK_MARKERS.search(text):
        return True
    if document.title:
        title = document.title.casefold()
        if any(
            keyword in title
            for keyword in (
                "luat",
                "nghi dinh",
                "nghi-quyet",
                "thong tu",
                "quyet dinh",
                "huong dan",
            )
        ):
            return bool(
                re.search(
                    r"(?i)\b(?:phạm vi điều chỉnh|đối tượng áp dụng|"
                    r"giải thích từ ngữ)\b",
                    text,
                )
            )
    return False


def _looks_like_ocr_noise(text: str, removed_line_count: int) -> bool:
    total = max(len(text), 1)
    digit_ratio = sum(char.isdigit() for char in text) / total
    symbol_ratio = (
        sum((not char.isalnum()) and not char.isspace() for char in text) / total
    )
    short_line_ratio = 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        short_line_ratio = sum(len(line) <= 4 for line in lines) / len(lines)
    return (
        digit_ratio > 0.12
        or symbol_ratio > 0.18
        or short_line_ratio > 0.4
        or removed_line_count > 40
    )


def _recommendation_for(reason: str) -> str:
    if reason == "cleaner_cleared_text":
        return (
            "Inspect cleaner rules for over-removal; keep legal headings and short "
            "preambles."
        )
    if reason == "ocr_noise":
        return (
            "Expand OCR/format heuristics and preserve legal heading line breaks "
            "before parsing."
        )
    if reason == "regex_recoverable":
        return (
            "Relax structure regex for accentless/OCR variants and pre-heading "
            "separators."
        )
    if reason == "fallback_chunkable_no_article":
        return (
            "Apply fallback chunking at paragraph/sentence level with a synthetic "
            "preamble article label."
        )
    return (
        "Keep in manual review unless a new rule is explicitly justified by sample "
        "evidence."
    )


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
