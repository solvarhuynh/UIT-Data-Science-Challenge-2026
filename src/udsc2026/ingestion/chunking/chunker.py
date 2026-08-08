"""Parent-child chunking that respects article, clause, and point boundaries."""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from udsc2026.contracts import LegalChunk, LegalParent
from udsc2026.ingestion.chunking.models import ChunkingResult
from udsc2026.ingestion.cleaners.abbreviations import expanded_terms_in_text
from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.legal_structure import parse_legal_document
from udsc2026.ingestion.legal_structure.models import (
    Article,
    Clause,
    LegalStructureDocument,
    Point,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+")
_ID_UNSAFE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class _ArticleContext:
    chapter: Optional[str]
    section: Optional[str]


def token_len(text: str) -> int:
    """Return a deterministic whitespace-token approximation for chunk sizing."""
    return len(re.findall(r"\S+", text))


def split_by_sentence_with_overlap(
    text: str, chunk_size: int = 192, chunk_overlap: int = 32
) -> List[str]:
    """Split only at sentence boundaries and retain a small preceding context.

    A sentence that alone exceeds ``chunk_size`` falls back to word-token
    boundaries within that same sentence. This avoids arbitrary character
    cuts while ensuring search children stay within the requested size.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    normalized = " ".join(text.split())
    if not normalized or token_len(normalized) <= chunk_size:
        return [normalized] if normalized else []

    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(normalized) if part]
    if len(sentences) == 1:
        return _split_long_sentence(normalized, chunk_size, chunk_overlap)

    parts: List[str] = []
    current: List[str] = []
    current_count = 0
    for sentence in sentences:
        sentence_count = token_len(sentence)
        if sentence_count > chunk_size:
            if current:
                parts.append(" ".join(current))
                current = []
                current_count = 0
            parts.extend(_split_long_sentence(sentence, chunk_size, chunk_overlap))
            continue
        if current and current_count + sentence_count > chunk_size:
            completed = " ".join(current)
            parts.append(completed)
            available_overlap = max(chunk_size - sentence_count, 0)
            overlap = _tail_tokens(completed, min(chunk_overlap, available_overlap))
            current = [overlap] if overlap else []
            current_count = token_len(overlap)
        current.append(sentence)
        current_count += sentence_count
    if current:
        parts.append(" ".join(current))
    return parts


def _split_long_sentence(
    sentence: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """Split one unusually long sentence at word boundaries only."""
    tokens = re.findall(r"\S+", sentence)
    overlap = min(chunk_overlap, max(chunk_size - 1, 0))
    parts: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        parts.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return parts


def _tail_tokens(text: str, count: int) -> str:
    if count == 0:
        return ""
    tokens = re.findall(r"\S+", text)
    return " ".join(tokens[-count:])


def chunk_clean_document(
    document: CleanDocument,
    chunk_size: int = 192,
    chunk_overlap: int = 32,
) -> ChunkingResult:
    """Parse and chunk one cleaned document without silently dropping fallbacks."""
    parsed = parse_legal_document(document)
    if parsed.law_name is None:
        parsed.law_name = _law_name_from_source_link(document.metadata)
    effective_date = document.metadata.get("effective_date")
    normalized_effective_date = str(effective_date) if effective_date else None
    if parsed.requires_manual_review and document.cleaned_text.strip():
        return _chunk_unstructured_document(
            document,
            parsed,
            effective_date=normalized_effective_date,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return chunk_legal_structure(
        parsed,
        abbreviations=document.abbreviations,
        effective_date=normalized_effective_date,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_unstructured_document(
    document: CleanDocument,
    chunk_size: int = 192,
    chunk_overlap: int = 32,
    *,
    law_name: Optional[str] = None,
    review_reasons: Optional[List[str]] = None,
) -> ChunkingResult:
    """Chunk a known parser miss directly, without reparsing a large document."""

    effective_date = document.metadata.get("effective_date")
    structure = LegalStructureDocument(
        doc_id=document.doc_id,
        source_path=document.source_path,
        law_name=law_name or _law_name_from_source_link(document.metadata),
        requires_manual_review=True,
        review_reasons=list(review_reasons or ["recovered_from_empty_chunk_file"]),
    )
    return _chunk_unstructured_document(
        document,
        structure,
        effective_date=str(effective_date) if effective_date else None,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_legal_structure(
    structure: LegalStructureDocument,
    *,
    abbreviations: Optional[Dict[str, str]] = None,
    effective_date: Optional[str] = None,
    chunk_size: int = 192,
    chunk_overlap: int = 32,
) -> ChunkingResult:
    """Create article parents and legal-unit children from a parsed document."""
    if not structure.doc_id:
        raise ValueError("LegalStructureDocument.doc_id is required for chunking")
    result = ChunkingResult(
        doc_id=structure.doc_id,
        requires_manual_review=structure.requires_manual_review,
        review_reasons=list(structure.review_reasons),
    )
    if structure.requires_manual_review:
        return result

    seen_parent_ids: Dict[str, int] = {}
    seen_chunk_ids: Dict[str, int] = {}
    abbreviation_map = abbreviations or {}
    for article, context in _iter_articles(structure):
        result.article_count += 1
        parent_id = _unique_id(
            _article_parent_id(structure.doc_id, article.identifier), seen_parent_ids
        )
        parent = _build_parent(
            parent_id,
            structure,
            article,
            context,
            abbreviation_map,
            effective_date,
        )
        result.parents.append(parent)
        result.chunks.extend(
            chunk_legal_article(
                article,
                doc_id=structure.doc_id,
                parent=parent,
                context=context,
                law_name=structure.law_name,
                source=structure.source_path,
                abbreviations=abbreviation_map,
                effective_date=effective_date,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                seen_chunk_ids=seen_chunk_ids,
            )
        )
    return result


def chunk_legal_article(
    article: Article,
    *,
    doc_id: str,
    parent: LegalParent,
    context: _ArticleContext,
    law_name: Optional[str],
    source: Optional[str],
    abbreviations: Dict[str, str],
    effective_date: Optional[str] = None,
    chunk_size: int = 192,
    chunk_overlap: int = 32,
    seen_chunk_ids: Optional[Dict[str, int]] = None,
) -> List[LegalChunk]:
    """Create search children from one article without crossing clause borders."""
    seen = seen_chunk_ids if seen_chunk_ids is not None else {}
    units = _article_units(article, parent.parent_id)
    chunks: List[LegalChunk] = []
    for text, clause, point, base_id in units:
        parts = split_by_sentence_with_overlap(text, chunk_size, chunk_overlap)
        for index, part in enumerate(parts, start=1):
            identifier = base_id
            if len(parts) > 1:
                identifier = "{0}_part_{1}".format(base_id, index)
            chunks.append(
                _build_chunk(
                    _unique_id(identifier, seen),
                    doc_id,
                    parent,
                    part,
                    context,
                    law_name,
                    source,
                    clause,
                    point,
                    abbreviations,
                    effective_date,
                )
            )
    return chunks


def _article_units(
    article: Article, parent_id: str
) -> Iterable[Tuple[str, Optional[Clause], Optional[Point], str]]:
    if not article.clauses:
        yield article.text, None, None, "{0}_body".format(parent_id)
        return
    for clause in article.clauses:
        if not clause.points or _has_direct_content(clause):
            yield (
                clause.content,
                clause,
                None,
                "{0}_clause_{1}".format(parent_id, _safe_id(clause.identifier)),
            )
        for point in clause.points:
            yield (
                point.text,
                clause,
                point,
                "{0}_clause_{1}_point_{2}".format(
                    parent_id, _safe_id(clause.identifier), _safe_id(point.identifier)
                ),
            )


def _has_direct_content(clause: Clause) -> bool:
    return clause.title is not None or "\n" in clause.content


def _build_parent(
    parent_id: str,
    structure: LegalStructureDocument,
    article: Article,
    context: _ArticleContext,
    abbreviations: Dict[str, str],
    effective_date: Optional[str],
) -> LegalParent:
    article_label = _label("Điều", article.identifier)
    metadata = _metadata(
        article.text,
        structure.law_name,
        context,
        article_label,
        None,
        None,
        structure.source_path,
        abbreviations,
        effective_date,
    )
    return LegalParent(
        parent_id=parent_id,
        doc_id=structure.doc_id or "",
        text=article.text.strip(),
        law_name=structure.law_name,
        chapter=context.chapter,
        section=context.section,
        article=article_label,
        source=structure.source_path,
        metadata=metadata,
    )


def _build_chunk(
    chunk_id: str,
    doc_id: str,
    parent: LegalParent,
    text: str,
    context: _ArticleContext,
    law_name: Optional[str],
    source: Optional[str],
    clause: Optional[Clause],
    point: Optional[Point],
    abbreviations: Dict[str, str],
    effective_date: Optional[str],
) -> LegalChunk:
    article_label = parent.article
    clause_label = _label("Khoản", clause.identifier) if clause else None
    point_label = _label("Điểm", point.identifier) if point else None
    metadata = _metadata(
        text,
        law_name,
        context,
        article_label,
        clause_label,
        point_label,
        source,
        abbreviations,
        effective_date,
    )
    return LegalChunk(
        chunk_id=chunk_id,
        parent_id=parent.parent_id,
        doc_id=doc_id,
        text=text.strip(),
        # Parent content is written once to data/processed/parents. Repeating it
        # in every child made the BTC processed corpus grow to tens of GB.
        parent_text=None,
        law_name=law_name,
        chapter=context.chapter,
        section=context.section,
        article=article_label,
        clause=clause_label,
        point=point_label,
        effective_date=effective_date,
        source=source,
        metadata=metadata,
    )


def _metadata(
    text: str,
    law_name: Optional[str],
    context: _ArticleContext,
    article: Optional[str],
    clause: Optional[str],
    point: Optional[str],
    source: Optional[str],
    abbreviations: Dict[str, str],
    effective_date: Optional[str],
) -> Dict[str, object]:
    return {
        "law_name": law_name,
        "chapter": context.chapter,
        "section": context.section,
        "article": article,
        "clause": clause,
        "point": point,
        "effective_date": effective_date,
        "source": source,
        "expanded_terms": expanded_terms_in_text(text, abbreviations),
        "char_count": len(text),
        "token_count": token_len(text),
    }


def _iter_articles(
    structure: LegalStructureDocument,
) -> Iterable[Tuple[Article, _ArticleContext]]:
    for article in structure.articles:
        yield article, _ArticleContext(None, None)
    for section in structure.sections:
        context = _ArticleContext(None, _label("Mục", section.identifier))
        for article in section.articles:
            yield article, context
    for chapter in structure.chapters:
        chapter_label = _label("Chương", chapter.identifier)
        for article in chapter.articles:
            yield article, _ArticleContext(chapter_label, None)
        for section in chapter.sections:
            context = _ArticleContext(chapter_label, _label("Mục", section.identifier))
            for article in section.articles:
                yield article, context


def _chunk_unstructured_document(
    document: CleanDocument,
    structure: LegalStructureDocument,
    *,
    effective_date: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
) -> ChunkingResult:
    """Create searchable fallback chunks for appendices and non-article texts.

    These documents stay flagged for manual review, but retrieval no longer
    loses their complete text. IDs retain the source ``doc_id`` prefix so an
    official LegalIR context ID can still be mapped back to every fallback hit.
    """

    context = _ArticleContext(None, None)
    parent_id = "{0}_document".format(_safe_id(document.doc_id))
    parent_metadata = _metadata(
        document.cleaned_text,
        structure.law_name,
        context,
        None,
        None,
        None,
        structure.source_path,
        document.abbreviations,
        effective_date,
    )
    parent_metadata["fallback_chunking"] = True
    parent_metadata["review_reasons"] = list(structure.review_reasons)
    parent = LegalParent(
        parent_id=parent_id,
        doc_id=document.doc_id,
        text=document.cleaned_text.strip(),
        law_name=structure.law_name,
        source=structure.source_path,
        metadata=parent_metadata,
    )
    parts = split_by_sentence_with_overlap(
        document.cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: List[LegalChunk] = []
    for index, part in enumerate(parts, start=1):
        metadata = _metadata(
            part,
            structure.law_name,
            context,
            None,
            None,
            None,
            structure.source_path,
            document.abbreviations,
            effective_date,
        )
        metadata["fallback_chunking"] = True
        metadata["review_reasons"] = list(structure.review_reasons)
        chunks.append(
            LegalChunk(
                chunk_id="{0}_part_{1}".format(parent_id, index),
                parent_id=parent_id,
                doc_id=document.doc_id,
                text=part,
                law_name=structure.law_name,
                effective_date=effective_date,
                source=structure.source_path,
                metadata=metadata,
            )
        )
    return ChunkingResult(
        doc_id=document.doc_id,
        chunks=chunks,
        parents=[parent],
        requires_manual_review=True,
        review_reasons=list(structure.review_reasons),
    )


def _law_name_from_source_link(metadata: Dict[str, object]) -> Optional[str]:
    """Use a public legal-document URL slug when the source omitted a title."""

    raw_link = metadata.get("source_link")
    if not isinstance(raw_link, str) or not raw_link.strip():
        return None
    path_name = unquote(urlparse(raw_link).path.rsplit("/", 1)[-1])
    stem = re.sub(r"\.(?:aspx?|html?)$", "", path_name, flags=re.IGNORECASE)
    stem = re.sub(r"-\d{5,}$", "", stem)
    inferred = " ".join(stem.replace("_", "-").split("-")).strip()
    return inferred or None


def _article_parent_id(doc_id: str, article_id: str) -> str:
    return "{0}_article_{1}".format(_safe_id(doc_id), _safe_id(article_id))


def _safe_id(value: str) -> str:
    normalized = _ID_UNSAFE.sub("_", value.casefold()).strip("_")
    return normalized or "unknown"


def _unique_id(candidate: str, occurrences: Dict[str, int]) -> str:
    count = occurrences.get(candidate, 0) + 1
    occurrences[candidate] = count
    return candidate if count == 1 else "{0}_occurrence_{1}".format(candidate, count)


def _label(prefix: str, identifier: str) -> str:
    return "{0} {1}".format(prefix, identifier)
