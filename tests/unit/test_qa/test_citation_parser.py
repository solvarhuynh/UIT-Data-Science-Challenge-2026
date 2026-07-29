"""Unit tests for CitationParser – no GPU or model weights required."""

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.qa.citation_parser import CitationParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parser() -> CitationParser:
    """Return a fresh CitationParser for each test."""
    return CitationParser()


@pytest.fixture
def sample_hit() -> RetrievalHit:
    """A representative retrieval hit matching the standard citation format."""
    return RetrievalHit(
        chunk_id="doc001_article_10_clause_1",
        doc_id="doc001",
        text="Người lao động có quyền làm việc, tự do lựa chọn việc làm.",
        score=0.92,
        law_name="Bộ luật Lao động 2019",
        article="Điều 10",
        clause="Khoản 1",
        source="data/raw/bllđ2019.txt",
    )


# ---------------------------------------------------------------------------
# parse_citations tests
# ---------------------------------------------------------------------------


class TestParseCitations:
    """Tests for CitationParser.parse_citations."""

    @pytest.mark.unit
    def test_parses_full_citation(self, parser: CitationParser) -> None:
        """Should extract law_name, article, clause, and point from a full citation."""
        answer = (
            "Người lao động có quyền làm việc. "
            "[Bộ luật Lao động 2019, Điều 10, Khoản 1, Điểm a]"
        )
        citations = parser.parse_citations(answer)

        assert len(citations) == 1
        c = citations[0]
        assert c.law_name == "Bộ luật Lao động 2019"
        assert c.article == "Điều 10"
        assert c.clause == "Khoản 1"
        assert c.point == "Điểm a"
        assert c.is_verified is False  # unverified until validate step

    @pytest.mark.unit
    def test_parses_citation_without_clause(self, parser: CitationParser) -> None:
        """Should handle citations that only specify article, no clause."""
        answer = "Quy định xử phạt tại [Bộ luật Hình sự 2015, Điều 134]."
        citations = parser.parse_citations(answer)

        assert len(citations) == 1
        assert citations[0].article == "Điều 134"
        assert citations[0].clause is None

    @pytest.mark.unit
    def test_parses_multiple_citations(self, parser: CitationParser) -> None:
        """Should extract all citations when multiple are present in the answer."""
        answer = (
            "Quy định A [Luật Doanh nghiệp 2020, Điều 15] và "
            "quy định B [Bộ luật Lao động 2019, Điều 10, Khoản 1]."
        )
        citations = parser.parse_citations(answer)
        assert len(citations) == 2

    @pytest.mark.unit
    def test_returns_empty_when_no_citations(self, parser: CitationParser) -> None:
        """Should return an empty list when the answer contains no bracket citations."""
        citations = parser.parse_citations("Câu trả lời không có trích dẫn.")
        assert citations == []

    @pytest.mark.unit
    def test_ignores_non_legal_brackets(self, parser: CitationParser) -> None:
        """Should ignore brackets that do not contain a Điều reference."""
        answer = "Xem thêm tại [trang 5] và [Phụ lục 1]."
        citations = parser.parse_citations(answer)
        assert citations == []


# ---------------------------------------------------------------------------
# validate_citations tests
# ---------------------------------------------------------------------------


class TestValidateCitations:
    """Tests for CitationParser.validate_citations."""

    @pytest.mark.unit
    def test_verifies_matching_citation(
        self, parser: CitationParser, sample_hit: RetrievalHit
    ) -> None:
        """A citation matching the context hit should be marked is_verified=True."""
        answer = "Người lao động có quyền làm việc [Bộ luật Lao động 2019, Điều 10, Khoản 1]."
        citations = parser.parse_citations(answer)
        verified = parser.validate_citations(citations, [sample_hit])

        assert len(verified) == 1
        assert verified[0].is_verified is True
        assert verified[0].warning is None

    @pytest.mark.unit
    def test_flags_hallucinated_citation(
        self, parser: CitationParser, sample_hit: RetrievalHit
    ) -> None:
        """A citation with no matching hit should be is_verified=False with a warning."""
        answer = "Quy định tại [Luật Nhà ở 2023, Điều 99, Khoản 5]."
        citations = parser.parse_citations(answer)
        verified = parser.validate_citations(citations, [sample_hit])

        assert len(verified) == 1
        assert verified[0].is_verified is False
        assert verified[0].warning is not None
        assert "hallucination" in verified[0].warning.lower() or "nghi ngờ" in verified[0].warning

    @pytest.mark.unit
    def test_empty_citations_returns_empty(
        self, parser: CitationParser, sample_hit: RetrievalHit
    ) -> None:
        """Validating an empty citation list should return an empty list."""
        result = parser.validate_citations([], [sample_hit])
        assert result == []

    @pytest.mark.unit
    def test_empty_contexts_flags_all_as_unverified(
        self, parser: CitationParser
    ) -> None:
        """When contexts list is empty, all citations should be flagged."""
        answer = "Xem [Bộ luật Lao động 2019, Điều 10]."
        citations = parser.parse_citations(answer)
        verified = parser.validate_citations(citations, [])

        assert all(not c.is_verified for c in verified)

    @pytest.mark.unit
    def test_parse_and_validate_convenience(
        self, parser: CitationParser, sample_hit: RetrievalHit
    ) -> None:
        """parse_and_validate should return citations and separate warnings list."""
        answer = (
            "Có quyền [Bộ luật Lao động 2019, Điều 10, Khoản 1]. "
            "Nhưng cũng xem [Luật Bịa 9999, Điều 999]."
        )
        verified, warnings = parser.parse_and_validate(answer, [sample_hit])

        assert len(verified) == 2
        # First citation should be verified, second should not.
        assert verified[0].is_verified is True
        assert verified[1].is_verified is False
        # Warnings list should have exactly one entry for the hallucinated one.
        assert len(warnings) == 1
