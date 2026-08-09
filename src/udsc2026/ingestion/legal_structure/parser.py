"""State-machine parser for the hierarchy of Vietnamese legal documents."""

import re
import unicodedata
from dataclasses import dataclass
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
    STRUCTURE_PREFIX,
)
from udsc2026.ingestion.readers.models import RawDocument

_UPPERCASE_LETTER = re.compile(r"[A-ZÀ-ỴĐ]")
_STANDALONE_ARTICLE_LABEL = re.compile(
    r"^\s*(?:Điều|Dieu)\s*$", re.IGNORECASE | re.UNICODE
)
_ARTICLE_NUMBER_CONTINUATION = re.compile(
    r"^\s*\d+[a-zđ]?(?![\w,;/])(?:"
    r"[ \t]*[.:\-)](?:[ \t]*.*)?|"
    r"(?=[ \t]+[^ \t,;/])[ \t]+.+)$",
    re.IGNORECASE | re.UNICODE,
)
_SPLIT_ARTICLE_NUMBER_CANDIDATE = re.compile(
    r"^\s*\d+[a-zđ]?(?![\w,;/])(?:[.:\-)]?\s*.*)?$",
    re.IGNORECASE | re.UNICODE,
)
_CONTAINER_WITH_TRAILING_ARTICLE = re.compile(
    r"^\s*((?:(?:Chương|Chuong)\s+(?:[IVXLCDM]+|\d+)|"
    r"(?:Mục|Muc)\s+(?:[IVXLCDM]+|\d+)).*?)"
    r"\s+(Điều|Dieu)\s*$",
    re.IGNORECASE | re.UNICODE,
)
_ANY_TRAILING_ARTICLE = re.compile(r"\b(?:Điều|Dieu)\s*$", re.IGNORECASE | re.UNICODE)
_ARTICLE_EXPLICIT_DELIMITER = re.compile(
    rf"{STRUCTURE_PREFIX}[\"“”«]?(?:Điều|Dieu)[ \t]+"
    r"\d+[a-zđ]?[ \t]*[.:\-)]",
    re.IGNORECASE | re.UNICODE,
)
_ARTICLE_CITATION_TITLE = re.compile(
    r"^(?:"
    r"nhu\s+sau\s*:?\s*$|"
    r"(?:cua\s+)?(?:bo\s+luat|luat|nghi\s+dinh|thong\s+tu|"
    r"quyet\s+dinh|phap\s+lenh|hien\s+phap|nghi\s+quyet|quy\s+che|"
    r"van\s+ban|dieu\s+uoc)\b|"
    r"(?:va|hoac|thi)\b|neu\s+tren\b|"
    r"duoc\s+(?:sua\s+doi|bo\s+sung|thay\s+the|neu)\b|"
    r"quy\s+dinh(?:\s+(?:nay|tai|kem\s+theo)\b|\s*:)"
    r")",
    re.IGNORECASE,
)
_APPENDIX_HEADING = re.compile(
    rf"{STRUCTURE_PREFIX}(?:PHỤ\s+LỤC|PHU\s+LUC)(?:"
    r"\s+(?:(?:SỐ|SO)\s+)?(?:[IVXLCDM]+|\d+)(?=$|[\s.:(\-])|\s*\(|\s*$)",
    re.IGNORECASE | re.UNICODE,
)
_DocumentInput = Union[str, RawDocument, CleanDocument]
_StructureNode = Union[Chapter, Section, Article, Clause, Point]


@dataclass(frozen=True)
class _LogicalLine:
    """One parser line mapped back to its inclusive physical source span."""

    start_line: int
    end_line: int
    text: str
    repaired_split_article: bool = False


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _fold_vietnamese(value: str) -> str:
    """Return an accentless form for conservative citation-title checks."""

    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(
        char for char in decomposed if not unicodedata.combining(char)
    ).replace("đ", "d")


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
        in_appendix = False

        def active_nodes() -> List[_StructureNode]:
            return [
                node
                for node in (chapter, section, article, clause, point)
                if node is not None
            ]

        def append_to_span(line: str, end_line: int) -> None:
            for node in active_nodes():
                node.text = _append(node.text, line)
                node.end_line = end_line

        def append_to_deepest_content(line: str, end_line: int) -> None:
            nodes = active_nodes()
            if nodes:
                deepest = nodes[-1]
                deepest.content = _append(deepest.content, line)
                deepest.end_line = end_line

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

        logical_lines, suspected_split_count = _logical_lines(text)
        result.suspected_split_article_count = suspected_split_count
        for logical_line in logical_lines:
            line_number = logical_line.start_line
            line_end = logical_line.end_line
            line = _line_for_text(logical_line.text)
            if not line:
                continue
            # An appendix is a separate retrieval segment.  Do not allow
            # article-looking examples or references inside it to reopen the
            # main document hierarchy.
            if in_appendix:
                continue
            if logical_line.repaired_split_article:
                result.repaired_split_article_count += 1

            if _APPENDIX_HEADING.match(line):
                chapter = section = article = clause = point = None
                in_appendix = True
                result.appendix_boundary_count += 1
                continue

            match = PATTERNS["chapter"].match(line)
            if match:
                # A same-level heading closes the prior chapter before this
                # line is included in any source span.
                chapter = section = article = clause = point = None
                append_to_span(line, line_end)
                identifier, node_title = _heading_parts(match)
                chapter = Chapter(
                    identifier=identifier.upper(),
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_end,
                )
                result.chapters.append(chapter)
                add_entry("chapter", chapter.identifier, line, line_number)
                continue

            match = PATTERNS["section"].match(line)
            if match:
                section = article = clause = point = None
                append_to_span(line, line_end)
                identifier, node_title = _heading_parts(match)
                section = Section(
                    identifier=identifier,
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_end,
                )
                if chapter is None:
                    result.sections.append(section)
                else:
                    chapter.sections.append(section)
                add_entry("section", section.identifier, line, line_number)
                continue

            match = PATTERNS["article"].match(line)
            if (
                match
                and not _ARTICLE_EXPLICIT_DELIMITER.match(line)
                and _ARTICLE_CITATION_TITLE.match(_fold_vietnamese(match.group(2)))
            ):
                # Amendment prose is often hard-wrapped as
                # ``Bổ sung Điều 98a vào sau / Điều 98 như sau: / “Điều
                # 98a. ...``.  The middle line is a cross-reference, not a
                # new article. Keep it in the active source span and let the
                # quoted heading on the following line open the real node.
                append_to_span(line, line_end)
                append_to_deepest_content(line, line_end)
                continue
            if match:
                article = clause = point = None
                append_to_span(line, line_end)
                identifier, node_title = _heading_parts(match)
                article = Article(
                    identifier=identifier,
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_end,
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
            point_match = LABELED_POINT.match(line) or PATTERNS["point"].match(line)
            if article is not None and point_match:
                point = None
                append_to_span(line, line_end)
                identifier, node_title = _heading_parts(point_match)
                point = Point(
                    identifier=identifier.casefold(),
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_end,
                )
                if clause is not None:
                    clause.points.append(point)
                else:
                    # A point without a numbered clause is still legal source
                    # material.  Keep it in the article span but do not invent
                    # a missing parent clause.
                    article.content = _append(article.content, line)
                add_entry("point", point.identifier, line, line_number)
                if clause is None:
                    # The point object has no legal parent and is intentionally
                    # not retained in the hierarchy. Keep subsequent physical
                    # lines in article.content instead of an unreachable object.
                    point = None
                continue

            clause_match = LABELED_CLAUSE.match(line) or PATTERNS["clause"].match(line)
            if article is not None and clause_match:
                clause = point = None
                append_to_span(line, line_end)
                identifier, node_title = _heading_parts(clause_match)
                clause = Clause(
                    identifier=identifier,
                    heading=line,
                    title=node_title,
                    content=line,
                    text=line,
                    start_line=line_number,
                    end_line=line_end,
                )
                article.clauses.append(clause)
                add_entry("clause", clause.identifier, line, line_number)
                continue

            append_to_span(line, line_end)
            append_to_deepest_content(line, line_end)

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


def _logical_lines(text: str) -> Tuple[List[_LogicalLine], int]:
    """Repair only high-confidence split ``Điều`` headings.

    BTC pages sometimes hard-wrap ``Điều`` away from its number. Joining an
    exact standalone label is safe. A merged ``Chương ... Điều`` or
    ``Mục ... Điều`` line is split into two logical headings while retaining
    the original physical line range. Arbitrary prose ending in ``Điều`` is
    never rewritten; it is counted for audit instead.
    """

    physical_lines = text.splitlines()
    logical: List[_LogicalLine] = []
    suspected = 0
    index = 0
    while index < len(physical_lines):
        raw_line = physical_lines[index]
        line_number = index + 1
        next_line = physical_lines[index + 1] if index + 1 < len(physical_lines) else ""
        continuation = _ARTICLE_NUMBER_CONTINUATION.match(next_line)
        split_candidate = _SPLIT_ARTICLE_NUMBER_CANDIDATE.match(next_line)
        if continuation is not None and _STANDALONE_ARTICLE_LABEL.match(raw_line):
            logical.append(
                _LogicalLine(
                    start_line=line_number,
                    end_line=line_number + 1,
                    text="{0} {1}".format(raw_line.strip(), next_line.strip()),
                    repaired_split_article=True,
                )
            )
            index += 2
            continue

        container = (
            _CONTAINER_WITH_TRAILING_ARTICLE.match(raw_line)
            if continuation is not None
            else None
        )
        if container is not None:
            logical.append(
                _LogicalLine(
                    start_line=line_number,
                    end_line=line_number,
                    text=container.group(1).strip(),
                )
            )
            logical.append(
                _LogicalLine(
                    start_line=line_number,
                    end_line=line_number + 1,
                    text="{0} {1}".format(container.group(2), next_line.strip()),
                    repaired_split_article=True,
                )
            )
            index += 2
            continue

        if split_candidate is not None and _ANY_TRAILING_ARTICLE.search(raw_line):
            suspected += 1
        logical.append(
            _LogicalLine(
                start_line=line_number,
                end_line=line_number,
                text=raw_line,
            )
        )
        index += 1
    return logical, suspected


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
