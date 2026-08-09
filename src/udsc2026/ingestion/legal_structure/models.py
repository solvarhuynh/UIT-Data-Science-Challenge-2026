"""Pydantic output contracts for the legal-structure parser."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Point(BaseModel):
    """A point (``Điểm``) nested in a clause."""

    identifier: str
    heading: str
    title: Optional[str] = None
    content: str = ""
    text: str = ""
    start_line: int
    end_line: int


class Clause(BaseModel):
    """A clause (``Khoản``) nested in an article."""

    identifier: str
    heading: str
    title: Optional[str] = None
    content: str = ""
    text: str = ""
    start_line: int
    end_line: int
    points: List[Point] = Field(default_factory=list)


class Article(BaseModel):
    """An article (``Điều``), the minimum legal unit accepted for chunking."""

    identifier: str
    heading: str
    title: Optional[str] = None
    content: str = ""
    text: str = ""
    start_line: int
    end_line: int
    clauses: List[Clause] = Field(default_factory=list)


class Section(BaseModel):
    """A section (``Mục``), optionally nested in a chapter."""

    identifier: str
    heading: str
    title: Optional[str] = None
    content: str = ""
    text: str = ""
    start_line: int
    end_line: int
    articles: List[Article] = Field(default_factory=list)


class Chapter(BaseModel):
    """A chapter (``Chương``) containing sections and/or articles."""

    identifier: str
    heading: str
    title: Optional[str] = None
    content: str = ""
    text: str = ""
    start_line: int
    end_line: int
    sections: List[Section] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)


class StructureEntry(BaseModel):
    """One recognised heading, retained in source order for auditing."""

    level: str
    identifier: str
    label: str
    heading: str
    line_number: int
    path: Dict[str, str] = Field(default_factory=dict)


class LegalStructureDocument(BaseModel):
    """Parsed hierarchy plus root-level nodes for legal text without chapters."""

    doc_id: Optional[str] = None
    source_path: Optional[str] = None
    law_name: Optional[str] = None
    chapters: List[Chapter] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)
    entries: List[StructureEntry] = Field(default_factory=list)
    requires_manual_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    repaired_split_article_count: int = 0
    suspected_split_article_count: int = 0
    appendix_boundary_count: int = 0
