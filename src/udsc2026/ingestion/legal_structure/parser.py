"""State-machine parser for the hierarchy of Vietnamese legal documents."""

import re
from typing import List, Optional, Tuple, Union

from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.legal_structure.models import (
    Article,
    Chapter,
    Clause,
    LegalStructureDocument,
    Point,
    Section,
    StructureEntry,
)
from udsc2026.ingestion.legal_structure.patterns import (
    LABELED_CLAUSE,
    LABELED_POINT,
    PATTERNS,
)
from udsc2026.ingestion.readers.models import RawDocument

_UPPERCASE_LETTER = re.compile(r"[A-ZÀ-ỴĐ]")
_DocumentInput = Union[str, RawDocument, CleanDocument]
_StructureNode = Union[Chapter, Section, Article, Clause, Point]


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def infer_law_name(text: str, title: Optional[str] = None) -> Optional[str]:
    """Use reader metadata first, then the first all-caps line as a fallback."""
    if title and title.strip():
        return _normalize_space(title)
    for line in text.splitlines():
        candidate = _normalize_space(line)
        if candidate and _UPPERCASE_LETTER.search(candidate) and candidate.isupper():
            return candidate
    return None


def _heading_parts(match: re.Match[str]) -> Tuple[str, Optional[str]]:
    identifier = match.group(1)
    title = _normalize_space(match.group(2)) if match.group(2).strip() else None
    return identifier, title


def _line_for_text(line: str) -> str:
    """Keep source content readable while removing only line-edge whitespace."""
    return line.strip()


class LegalStructureParser:
    """Parse headings in one pass while keeping the active legal context."""

    def parse(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> LegalStructureDocument:
        """Return a hierarchy and an ordered event stream for *text*.

        A numeric clause or alphabetic point is considered only after an
        ``Điều`` has been observed.  This is the context guard that prevents
        list markers in a preamble from becoming fake legal nodes.
        """
        result = LegalStructureDocument(
            doc_id=doc_id,
            source_path=source_path,
            law_name=infer_law_name(text, title),
        )
        chapter: Optional[Chapter] = None
        section: Optional[Section] = None
        article: Optional[Article] = None
        clause: Optional[Clause] = None
        point: Optional[Point] = None

        def active_nodes() -> List[_StructureNode]:
            return [
                node
                for node in (chapter, section, article, clause, point)
                if node is not None
            ]

        def append_to_span(line: str, line_number: int) -> None:
            for node in active_nodes():
                node.text = _append(node.text, line)
                node.end_line = line_number

        def append_to_deepest_content(line: str, line_number: int) -> None:
            nodes = active_nodes()
            if nodes:
                deepest = nodes[-1]
                deepest.content = _append(deepest.content, line)
                deepest.end_line = line_number

        def add_entry(
            level: str, identifier: str, heading: str, line_number: int
        ) -> None:
            path = {}
            if chapter is not None:
                path["chapter"] = "Chương {0}".format(chapter.identifier)
            if section is not None:
                path["section"] = "Mục {0}".format(section.identifier)
            if article is not None:
                path["article"] = "Điều {0}".format(article.identifier)
            if clause is not None:
                path["clause"] = "Khoản {0}".format(clause.identifier)
            if point is not None:
                path["point"] = "Điểm {0}".format(point.identifier)
            labels = {
                "chapter": "Chương",
                "section": "Mục",
                "article": "Điều",
                "clause": "Khoản",
                "point": "Điểm",
            }
            result.entries.append(
                StructureEntry(
                    level=level,
                    identifier=identifier,
                    label="{0} {1}".format(labels[level], identifier),
                    heading=heading,
                    line_number=line_number,
                    path=path,
                )
            )

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = _line_for_text(raw_line)
            if not line:
                continue

            match = PATTERNS["chapter"].search(line)
            if match:
                # A same-level heading closes the prior chapter before this
                # line is included in any source span.
                chapter = section = article = clause = point = None
                append_to_span(line, line_number)
                identifier, node_title = _heading_parts(match)
                chapter = Chapter(
                    identifier=identifier.upper(),
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_number,
                )
                result.chapters.append(chapter)
                add_entry("chapter", chapter.identifier, line, line_number)
                continue

            match = PATTERNS["section"].search(line)
            if match:
                section = article = clause = point = None
                append_to_span(line, line_number)
                identifier, node_title = _heading_parts(match)
                section = Section(
                    identifier=identifier,
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_number,
                )
                if chapter is None:
                    result.sections.append(section)
                else:
                    chapter.sections.append(section)
                add_entry("section", section.identifier, line, line_number)
                continue

            match = PATTERNS["article"].search(line)
            if match:
                article = clause = point = None
                append_to_span(line, line_number)
                identifier, node_title = _heading_parts(match)
                article = Article(
                    identifier=identifier,
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_number,
                )
                if section is not None:
                    section.articles.append(article)
                elif chapter is not None:
                    chapter.articles.append(article)
                else:
                    result.articles.append(article)
                add_entry("article", article.identifier, line, line_number)
                continue

            # Clause and point rules run only in the active Article state.
            # Check points first: otherwise ``a)`` is not a numeric clause,
            # but this ordering documents the intended hierarchy explicitly.
            point_match = LABELED_POINT.search(line) or PATTERNS["point"].match(line)
            if article is not None and point_match:
                point = None
                append_to_span(line, line_number)
                identifier, node_title = _heading_parts(point_match)
                point = Point(
                    identifier=identifier.casefold(),
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_number,
                )
                if clause is not None:
                    clause.points.append(point)
                else:
                    # A point without a numbered clause is still legal source
                    # material.  Keep it in the article span but do not invent
                    # a missing parent clause.
                    article.content = _append(article.content, line)
                add_entry("point", point.identifier, line, line_number)
                continue

            clause_match = LABELED_CLAUSE.search(line) or PATTERNS["clause"].match(line)
            if article is not None and clause_match:
                clause = point = None
                append_to_span(line, line_number)
                identifier, node_title = _heading_parts(clause_match)
                clause = Clause(
                    identifier=identifier,
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_number,
                )
                article.clauses.append(clause)
                add_entry("clause", clause.identifier, line, line_number)
                continue

            append_to_span(line, line_number)
            append_to_deepest_content(line, line_number)

        if not _has_articles(result):
            result.requires_manual_review = True
            result.review_reasons.append("no_article_detected")
        return result

    def parse_document(
        self, document: Union[RawDocument, CleanDocument]
    ) -> LegalStructureDocument:
        """Parse a reader or cleaner model without losing document provenance."""
        text = (
            document.cleaned_text
            if isinstance(document, CleanDocument)
            else document.raw_text
        )
        return self.parse(
            text,
            title=document.title,
            doc_id=document.doc_id,
            source_path=document.source_path,
        )


def _append(existing: str, line: str) -> str:
    return "{0}\n{1}".format(existing, line) if existing else line


def _has_articles(result: LegalStructureDocument) -> bool:
    if result.articles:
        return True
    for section in result.sections:
        if section.articles:
            return True
    for chapter in result.chapters:
        if chapter.articles or any(section.articles for section in chapter.sections):
            return True
    return False


def parse_legal_text(text: str, title: Optional[str] = None) -> LegalStructureDocument:
    """Convenience function for parsing text already extracted and cleaned."""
    return LegalStructureParser().parse(text, title=title)


def parse_legal_document(
    document: Union[RawDocument, CleanDocument],
) -> LegalStructureDocument:
    """Convenience function for parsing a pipeline document model."""
    return LegalStructureParser().parse_document(document)


def parse_legal_structure(
    source: _DocumentInput, title: Optional[str] = None
) -> LegalStructureDocument:
    """Parse text or a RawDocument/CleanDocument through one public entry point."""
    if isinstance(source, (RawDocument, CleanDocument)):
        return parse_legal_document(source)
    return parse_legal_text(source, title=title)
