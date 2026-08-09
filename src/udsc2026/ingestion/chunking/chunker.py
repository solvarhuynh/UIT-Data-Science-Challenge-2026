"""Parent-child chunking that respects article, clause, and point boundaries."""

import re
import unicodedata
from collections import Counter
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
    chapter_heading: Optional[str] = None
    section_heading: Optional[str] = None


@dataclass(frozen=True)
class _StructureAssessment:
    status: str
    source_family: str
    warnings: Tuple[str, ...] = ()
    force_fallback: bool = False


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
    carry_overlap = ""
    for sentence in sentences:
        sentence_count = token_len(sentence)
        # An exactly-full sentence also needs the long-sentence path when a
        # previous sentence is pending.  The regular overflow branch has no
        # room for even one overlap token in that case, which silently drops
        # the source bigram spanning the sentence boundary.
        if sentence_count >= chunk_size:
            prefix = carry_overlap
            if current:
                completed = " ".join(current)
                parts.append(completed)
                prefix = _tail_tokens(completed, min(chunk_overlap, chunk_size - 1))
                current = []
                current_count = 0
            long_input = "{0} {1}".format(prefix, sentence).strip()
            long_parts = _split_long_sentence(long_input, chunk_size, chunk_overlap)
            parts.extend(long_parts)
            carry_overlap = _tail_tokens(
                long_parts[-1], min(chunk_overlap, chunk_size - 1)
            )
            continue
        if not current and carry_overlap:
            available_overlap = max(chunk_size - sentence_count, 0)
            overlap = _tail_tokens(carry_overlap, min(chunk_overlap, available_overlap))
            current = [overlap] if overlap else []
            current_count = token_len(overlap)
            carry_overlap = ""
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
    assessment = _assess_structure(document, parsed)
    if not document.cleaned_text.strip():
        empty_result = _chunk_empty_source_document(
            document,
            parsed,
            effective_date=normalized_effective_date,
        )
        _set_searchable_content_coverage(empty_result, document.cleaned_text)
        _apply_structure_assessment(empty_result, assessment)
        return empty_result
    if assessment.force_fallback:
        parsed.requires_manual_review = True
        for warning in assessment.warnings:
            if warning not in parsed.review_reasons:
                parsed.review_reasons.append(warning)
    if parsed.requires_manual_review and document.cleaned_text.strip():
        # Keep origin/main's useful recovery for recognizable legal preambles,
        # while retaining the V3 full-text fallback for every other parser miss.
        # Source-family fallbacks stay reviewable because they can contain
        # article-like citations that must not be promoted to legal structure.
        if not assessment.force_fallback:
            recovered = _chunk_manual_review_document(
                document,
                parsed,
                effective_date=normalized_effective_date,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if recovered is not None:
                _set_source_assignment_metrics(recovered, document.cleaned_text)
                recovered.repaired_split_article_count = (
                    parsed.repaired_split_article_count
                )
                recovered.suspected_split_article_count = (
                    parsed.suspected_split_article_count
                )
                _set_searchable_content_coverage(recovered, document.cleaned_text)
                _apply_structure_assessment(recovered, assessment)
                return recovered
        fallback = _chunk_unstructured_document(
            document,
            parsed,
            effective_date=normalized_effective_date,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        _set_source_assignment_metrics(fallback, document.cleaned_text)
        fallback.repaired_split_article_count = parsed.repaired_split_article_count
        fallback.suspected_split_article_count = parsed.suspected_split_article_count
        _set_searchable_content_coverage(fallback, document.cleaned_text)
        _apply_structure_assessment(fallback, assessment)
        return fallback
    result = chunk_legal_structure(
        parsed,
        abbreviations=document.abbreviations,
        effective_date=normalized_effective_date,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    result.repaired_split_article_count = parsed.repaired_split_article_count
    result.suspected_split_article_count = parsed.suspected_split_article_count
    _append_supplemental_contexts(
        result,
        document,
        parsed,
        effective_date=normalized_effective_date,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    _set_searchable_content_coverage(result, document.cleaned_text)
    _apply_structure_assessment(result, assessment)
    return result


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
    result.structured_chunk_count = len(result.chunks)
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
        if clause is None and point is None:
            text = _with_hierarchy_context(text, context)
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
    if article.content.strip():
        yield article.content, None, None, "{0}_intro".format(parent_id)
    for clause in article.clauses:
        has_direct_content = _has_direct_content(clause)
        if not clause.points or has_direct_content:
            yield (
                clause.content,
                clause,
                None,
                "{0}_clause_{1}".format(parent_id, _safe_id(clause.identifier)),
            )
        for point_index, point in enumerate(clause.points):
            point_text = point.text
            if point_index == 0 and not has_direct_content and clause.content.strip():
                # A label-only clause (for example ``Khoản 228.``) used to
                # disappear from every searchable child when points followed
                # it.  Prefix it once to the first point: this retains source
                # text and legal context without creating a one-token chunk.
                point_text = "{0}\n{1}".format(clause.content.strip(), point.text)
            yield (
                point_text,
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
    parent_text = _with_hierarchy_context(article.text, context)
    metadata = _metadata(
        parent_text,
        structure.law_name,
        context,
        article_label,
        None,
        None,
        structure.source_path,
        abbreviations,
        effective_date,
    )
    metadata["structure_type"] = "article"
    return LegalParent(
        parent_id=parent_id,
        doc_id=structure.doc_id or "",
        text=parent_text.strip(),
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
    metadata["structure_type"] = "article"
    return LegalChunk(
        chunk_id=chunk_id,
        parent_id=parent.parent_id,
        doc_id=doc_id,
        text=text.strip(),
        # Parent content is written once to the configured parent store. Repeating it
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
        "chapter_heading": context.chapter_heading,
        "section_heading": context.section_heading,
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
        context = _ArticleContext(
            None,
            _label("Mục", section.identifier),
            section_heading=section.heading,
        )
        for article in section.articles:
            yield article, context
    for chapter in structure.chapters:
        chapter_label = _label("Chương", chapter.identifier)
        for article in chapter.articles:
            yield (
                article,
                _ArticleContext(chapter_label, None, chapter_heading=chapter.heading),
            )
        for section in chapter.sections:
            context = _ArticleContext(
                chapter_label,
                _label("Mục", section.identifier),
                chapter_heading=chapter.heading,
                section_heading=section.heading,
            )
            for article in section.articles:
                yield article, context


def _assess_structure(
    document: CleanDocument, structure: LegalStructureDocument
) -> _StructureAssessment:
    """Classify parsed hierarchy confidence without discarding source text."""

    source_family = _source_family(document.metadata)
    articles = [article for article, _ in _iter_articles(structure)]
    if not articles:
        return _StructureAssessment("unstructured", source_family)

    warnings: List[str] = []
    force_fallback = False
    if source_family in {"cong_van", "tcvn"}:
        warnings.append("unstructured_source_family")
        force_fallback = True

    identifiers = [article.identifier.casefold() for article in articles]
    if len(set(identifiers)) != len(identifiers):
        warnings.append("duplicate_article_identifiers")
    numeric_ids = [
        int(match.group(1))
        for identifier in identifiers
        if (match := re.match(r"^(\d+)", identifier)) is not None
    ]
    if any(
        current < previous for previous, current in zip(numeric_ids, numeric_ids[1:])
    ):
        warnings.append("nonmonotonic_article_sequence")

    physical_line_count = max(len(document.cleaned_text.splitlines()), 1)
    first_article_ratio = (articles[0].start_line - 1) / physical_line_count
    if first_article_ratio > 0.5:
        warnings.append("late_first_article")
    if len(articles) <= 3 and first_article_ratio > 0.5:
        warnings.append("low_structure_confidence")
        force_fallback = True
    if structure.suspected_split_article_count:
        warnings.append("ambiguous_split_article_candidates")
    if structure.repaired_split_article_count:
        warnings.append("repaired_split_article_headings")
        if len(articles) <= 3:
            warnings.append("sparse_repaired_article_sequence")
    if structure.appendix_boundary_count:
        warnings.append("appendix_segments_detected")

    status = (
        "partial"
        if any(
            warning
            in {
                "duplicate_article_identifiers",
                "nonmonotonic_article_sequence",
                "late_first_article",
                "sparse_repaired_article_sequence",
            }
            for warning in warnings
        )
        else "structured"
    )
    if force_fallback:
        status = "unstructured_fallback"
    return _StructureAssessment(
        status=status,
        source_family=source_family,
        warnings=tuple(warnings),
        force_fallback=force_fallback,
    )


def _source_family(metadata: Dict[str, object]) -> str:
    raw_link = metadata.get("source_link")
    if not isinstance(raw_link, str):
        return "unknown"
    path = urlparse(raw_link).path.casefold()
    if "/tcvn/" in path:
        return "tcvn"
    if "/cong-van/" in path:
        return "cong_van"
    if "/van-ban/" in path:
        return "van_ban"
    return "unknown"


def _apply_structure_assessment(
    result: ChunkingResult, assessment: _StructureAssessment
) -> None:
    if result.source_content_empty:
        status = "empty_source"
    elif result.fallback_chunk_count:
        status = "unstructured_fallback"
    else:
        status = assessment.status
    result.structure_status = status
    result.source_family = assessment.source_family
    result.structure_warnings = list(assessment.warnings)
    for parent in result.parents:
        parent.metadata["structure_status"] = status
        parent.metadata["source_family"] = assessment.source_family
        parent.metadata["structure_warnings"] = list(assessment.warnings)
    for chunk in result.chunks:
        chunk.metadata["structure_status"] = status
        chunk.metadata["source_family"] = assessment.source_family
        chunk.metadata["structure_warnings"] = list(assessment.warnings)


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
    parent_metadata["structure_type"] = "unstructured_fallback"
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
        metadata["structure_type"] = "unstructured_fallback"
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
        fallback_chunk_count=len(chunks),
    )


def _append_supplemental_contexts(
    result: ChunkingResult,
    document: CleanDocument,
    structure: LegalStructureDocument,
    *,
    effective_date: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Assign source regions outside parsed Articles to auditable chunks."""

    physical_lines = document.cleaned_text.splitlines()
    source_lines = {
        index: line.strip()
        for index, line in enumerate(physical_lines, start=1)
        if line.strip()
    }
    covered_physical_lines = _article_source_lines(structure, len(physical_lines))
    covered_nonempty_lines = set(source_lines) & covered_physical_lines
    supplemented_nonempty_lines: set[int] = set()
    context = _ArticleContext(None, None)

    for sequence, (start_line, end_line) in enumerate(
        _unassigned_source_ranges(physical_lines, covered_physical_lines), start=1
    ):
        text = "\n".join(physical_lines[start_line - 1 : end_line]).strip()
        if not text:
            continue
        supplemented_nonempty_lines.update(
            line_number
            for line_number in range(start_line, end_line + 1)
            if line_number in source_lines
        )
        parent_id = "{0}_document_context_{1}".format(
            _safe_id(document.doc_id), sequence
        )
        parent_metadata = _metadata(
            text,
            structure.law_name,
            context,
            None,
            None,
            None,
            structure.source_path,
            document.abbreviations,
            effective_date,
        )
        parent_metadata.update(
            {
                "structure_type": "document_context",
                "supplemental_context": True,
                "source_line_start": start_line,
                "source_line_end": end_line,
            }
        )
        result.parents.append(
            LegalParent(
                parent_id=parent_id,
                doc_id=document.doc_id,
                text=text,
                law_name=structure.law_name,
                source=structure.source_path,
                metadata=parent_metadata,
            )
        )
        parts = split_by_sentence_with_overlap(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for part_number, part in enumerate(parts, start=1):
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
            metadata.update(parent_metadata)
            metadata["char_count"] = len(part)
            metadata["token_count"] = token_len(part)
            metadata["expanded_terms"] = expanded_terms_in_text(
                part, document.abbreviations
            )
            result.chunks.append(
                LegalChunk(
                    chunk_id="{0}_part_{1}".format(parent_id, part_number),
                    parent_id=parent_id,
                    doc_id=document.doc_id,
                    text=part,
                    parent_text=None,
                    law_name=structure.law_name,
                    effective_date=effective_date,
                    source=structure.source_path,
                    metadata=metadata,
                )
            )
            result.supplemental_chunk_count += 1

    assigned_lines = covered_nonempty_lines | supplemented_nonempty_lines
    result.source_nonempty_line_count = len(source_lines)
    result.assigned_source_line_count = len(assigned_lines)
    result.supplemented_source_line_count = len(supplemented_nonempty_lines)
    result.unassigned_source_line_count = len(set(source_lines) - assigned_lines)
    result.source_character_count = sum(len(line) for line in source_lines.values())
    result.assigned_source_character_count = sum(
        len(source_lines[index]) for index in assigned_lines
    )
    result.supplemented_source_character_count = sum(
        len(source_lines[index]) for index in supplemented_nonempty_lines
    )


def _article_source_lines(
    structure: LegalStructureDocument, physical_line_count: int
) -> set[int]:
    covered: set[int] = set()
    for article, _ in _iter_articles(structure):
        start = max(article.start_line, 1)
        end = min(article.end_line, physical_line_count)
        covered.update(range(start, end + 1))
    return covered


def _unassigned_source_ranges(
    physical_lines: List[str], covered_lines: set[int]
) -> Iterable[Tuple[int, int]]:
    start: Optional[int] = None
    for line_number in range(1, len(physical_lines) + 1):
        if line_number in covered_lines:
            if start is not None:
                yield start, line_number - 1
                start = None
            continue
        if start is None:
            start = line_number
    if start is not None:
        yield start, len(physical_lines)


def _set_source_assignment_metrics(result: ChunkingResult, text: str) -> None:
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    character_count = sum(len(line) for line in nonempty_lines)
    result.source_nonempty_line_count = len(nonempty_lines)
    result.assigned_source_line_count = len(nonempty_lines)
    result.unassigned_source_line_count = 0
    result.source_character_count = character_count
    result.assigned_source_character_count = character_count


def _set_searchable_content_coverage(result: ChunkingResult, text: str) -> None:
    """Verify occurrence-aware source coverage in parents and children.

    Chunk overlap and repeated legal phrases make set membership insufficient:
    one surviving copy could hide a dropped occurrence elsewhere.  Counter
    subtraction preserves multiplicity.  Both storage layers are checked so a
    complete parent cannot mask a missing searchable child (or vice versa).
    """

    source_tokens = Counter(re.findall(r"\S+", text))
    child_tokens: Counter[str] = Counter()
    child_bigrams: Counter[Tuple[str, str]] = Counter()
    for chunk in result.chunks:
        tokens = re.findall(r"\S+", chunk.text)
        child_tokens.update(tokens)
        child_bigrams.update(zip(tokens, tokens[1:]))

    parent_tokens: Counter[str] = Counter()
    parent_bigrams: Counter[Tuple[str, str]] = Counter()
    for parent in result.parents:
        tokens = re.findall(r"\S+", parent.text)
        parent_tokens.update(tokens)
        parent_bigrams.update(zip(tokens, tokens[1:]))

    source_bigrams: Counter[Tuple[str, str]] = Counter()
    for line in text.splitlines():
        tokens = re.findall(r"\S+", line)
        source_bigrams.update(zip(tokens, tokens[1:]))

    missing_tokens = (source_tokens - child_tokens) | (source_tokens - parent_tokens)
    missing_bigrams = (source_bigrams - child_bigrams) | (
        source_bigrams - parent_bigrams
    )
    result.source_unique_token_count = len(source_tokens)
    result.missing_source_token_count = sum(missing_tokens.values())
    result.source_intraline_bigram_count = sum(source_bigrams.values())
    result.missing_source_bigram_count = sum(missing_bigrams.values())
    result.missing_source_token_examples = sorted(missing_tokens)[:20]
    result.missing_source_bigram_examples = [
        "{0} {1}".format(first, second)
        for first, second in sorted(missing_bigrams)[:20]
    ]


def _chunk_empty_source_document(
    document: CleanDocument,
    structure: LegalStructureDocument,
    *,
    effective_date: Optional[str],
) -> ChunkingResult:
    """Represent an official empty passage without indexing its JSON or URL."""

    display_name = _empty_source_display_name(document, structure)
    placeholder_text = "Tên văn bản: {0}".format(display_name)
    parent_id = "{0}_empty_source".format(_safe_id(document.doc_id))
    context = _ArticleContext(None, None)
    metadata = _metadata(
        placeholder_text,
        structure.law_name or display_name,
        context,
        None,
        None,
        None,
        structure.source_path,
        document.abbreviations,
        effective_date,
    )
    metadata.update(
        {
            "structure_type": "empty_source_placeholder",
            "source_content_empty": True,
            "synthetic_placeholder": True,
            "review_reasons": ["empty_source_content"],
        }
    )
    parent = LegalParent(
        parent_id=parent_id,
        doc_id=document.doc_id,
        text=placeholder_text,
        law_name=structure.law_name or display_name,
        source=structure.source_path,
        metadata=dict(metadata),
    )
    chunk = LegalChunk(
        chunk_id="{0}_placeholder".format(parent_id),
        parent_id=parent_id,
        doc_id=document.doc_id,
        text=placeholder_text,
        parent_text=None,
        law_name=structure.law_name or display_name,
        effective_date=effective_date,
        source=structure.source_path,
        metadata=dict(metadata),
    )
    return ChunkingResult(
        doc_id=document.doc_id,
        chunks=[chunk],
        parents=[parent],
        requires_manual_review=True,
        review_reasons=["empty_source_content"],
        empty_source_placeholder_chunk_count=1,
        source_content_empty=True,
        repaired_split_article_count=structure.repaired_split_article_count,
        suspected_split_article_count=structure.suspected_split_article_count,
    )


def _empty_source_display_name(
    document: CleanDocument, structure: LegalStructureDocument
) -> str:
    value = (
        document.title
        or structure.law_name
        or document.metadata.get("source_name")
        or "context {0}".format(document.doc_id)
    )
    return " ".join(str(value).replace("_", "-").split("-")).strip()


def _with_hierarchy_context(text: str, context: _ArticleContext) -> str:
    headings = [
        heading.strip()
        for heading in (context.chapter_heading, context.section_heading)
        if heading and heading.strip()
    ]
    body = text.strip()
    if not headings:
        return body
    return "\n".join(headings + [body])


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
    ascii_source = unicodedata.normalize("NFKD", value.casefold())
    ascii_source = ascii_source.replace("đ", "d_vn")
    ascii_source = ascii_source.encode("ascii", "ignore").decode("ascii")
    normalized = _ID_UNSAFE.sub("_", ascii_source).strip("_")
    return normalized or "unknown"


def _unique_id(candidate: str, occurrences: Dict[str, int]) -> str:
    count = occurrences.get(candidate, 0) + 1
    occurrences[candidate] = count
    return candidate if count == 1 else "{0}_occurrence_{1}".format(candidate, count)


def _label(prefix: str, identifier: str) -> str:
    return "{0} {1}".format(prefix, identifier)


def _chunk_manual_review_document(
    document: CleanDocument,
    structure: LegalStructureDocument,
    *,
    effective_date: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
) -> Optional[ChunkingResult]:
    """Recover a recognizable legal preamble without inventing article nodes."""
    text = document.cleaned_text.strip()
    if not text or not _looks_like_fallback_source(document):
        return None

    article_label = _fallback_article_label(document)
    parent_id = "{0}_fallback".format(_safe_id(document.doc_id))
    context = _ArticleContext(None, None)
    parent_metadata = _metadata(
        text,
        structure.law_name,
        context,
        article_label,
        None,
        None,
        structure.source_path,
        document.abbreviations,
        effective_date,
    )
    parent_metadata.update(
        {
            "fallback_chunking": True,
            "fallback_mode": "manual_review_no_article",
            "structure_type": "unstructured_fallback",
            "review_reasons": list(structure.review_reasons),
        }
    )
    parent = LegalParent(
        parent_id=parent_id,
        doc_id=document.doc_id,
        text=text,
        law_name=structure.law_name,
        chapter=None,
        section=None,
        article=article_label,
        source=structure.source_path,
        metadata=parent_metadata,
    )
    result = ChunkingResult(
        doc_id=document.doc_id,
        # The searchable label is synthetic; do not count it as a parsed
        # legal article or the document appears in both structured/fallback
        # partitions.
        article_count=0,
        chunks=[],
        parents=[parent],
        # Recovery makes the document searchable, but it does not prove that
        # its legal hierarchy was parsed correctly. Keep it in the review
        # queue so the fallback bucket remains visible in corpus diagnostics.
        requires_manual_review=True,
        review_reasons=["fallback_chunked_no_article"],
    )

    for index, part in enumerate(
        _fallback_text_parts(text, chunk_size, chunk_overlap), start=1
    ):
        chunk_id = "{0}_part_{1}".format(parent_id, index)
        metadata = _metadata(
            part,
            structure.law_name,
            context,
            article_label,
            None,
            None,
            structure.source_path,
            document.abbreviations,
            effective_date,
        )
        metadata.update(
            {
                "fallback_chunking": True,
                "fallback_mode": "manual_review_no_article",
                "structure_type": "unstructured_fallback",
                "review_reasons": list(structure.review_reasons),
            }
        )
        result.chunks.append(
            LegalChunk(
                chunk_id=chunk_id,
                parent_id=parent_id,
                doc_id=document.doc_id,
                text=part,
                parent_text=None,
                law_name=parent.law_name,
                chapter=None,
                section=None,
                article=article_label,
                clause=None,
                point=None,
                effective_date=effective_date,
                source=structure.source_path,
                metadata=metadata,
            )
        )
    result.fallback_chunk_count = len(result.chunks)
    return result


def _looks_like_fallback_source(document: CleanDocument) -> bool:
    text = document.cleaned_text
    fallback_markers = (
        "lời nói đầu",
        "phạm vi điều chỉnh",
        "đối tượng áp dụng",
        "giải thích từ ngữ",
        "quy định chung",
        "điều khoản thi hành",
    )
    lowered = text.casefold()
    if any(marker in lowered for marker in fallback_markers):
        return True
    if document.title:
        title = document.title.casefold()
        legal_title_keywords = (
            "luat",
            "nghi dinh",
            "nghi-quyet",
            "thong tu",
            "quyet dinh",
            "huong dan",
        )
        if any(keyword in title for keyword in legal_title_keywords):
            return any(marker in lowered for marker in fallback_markers)
    return False


def _fallback_article_label(document: CleanDocument) -> str:
    lowered = document.cleaned_text.casefold()
    for candidate in (
        "Lời nói đầu",
        "Phạm vi điều chỉnh",
        "Đối tượng áp dụng",
        "Giải thích từ ngữ",
        "Quy định chung",
        "Điều khoản thi hành",
    ):
        if candidate.casefold() in lowered:
            return candidate
    return "Phần mở đầu"


def _fallback_text_parts(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    source_parts = paragraphs if paragraphs else [text]
    chunks: List[str] = []
    for part in source_parts:
        if token_len(part) <= chunk_size:
            chunks.append(part)
        else:
            chunks.extend(
                split_by_sentence_with_overlap(part, chunk_size, chunk_overlap)
            )
    return chunks
