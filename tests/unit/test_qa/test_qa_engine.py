"""Unit tests for QAEngine using MockLLMClient – no GPU required."""

from pathlib import Path
from typing import Optional

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.llm.client import MockLLMClient
from udsc2026.infrastructure.llm.config import LLMConfig
from udsc2026.qa.citation_parser import CitationParser
from udsc2026.qa.prompt_builder import PromptBuilder
from udsc2026.qa.qa_engine import QAEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_prompts(tmp_path: Path) -> Path:
    """Minimal prompts directory tree for tests."""
    system_dir = tmp_path / "system"
    rag_dir = tmp_path / "rag_templates"
    system_dir.mkdir()
    rag_dir.mkdir()
    (system_dir / "legal_qa_v1.md").write_text(
        "Chỉ dùng CONTEXT để trả lời.", encoding="utf-8"
    )
    (rag_dir / "default_rag_v1.md").write_text(
        "CONTEXT:\n{context_block}\n\nCÂU HỎI:\n{question}\n\nTRẢ LỜI:",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """MockLLMClient with a realistic legal response containing a valid citation."""
    return MockLLMClient(
        fixed_response=(
            "Người lao động có quyền làm việc và tự do lựa chọn việc làm. "
            "[Bộ luật Lao động 2019, Điều 10, Khoản 1]"
        )
    )


@pytest.fixture
def good_context() -> list[RetrievalHit]:
    """A context hit that matches the citation in mock_llm's response."""
    return [
        RetrievalHit(
            chunk_id="doc001_article_10_clause_1",
            doc_id="doc001",
            text="Người lao động có quyền làm việc, tự do lựa chọn việc làm.",
            score=0.92,
            law_name="Bộ luật Lao động 2019",
            article="Điều 10",
            clause="Khoản 1",
        )
    ]


@pytest.fixture
def engine(
    mock_llm: MockLLMClient, tmp_prompts: Path
) -> QAEngine:
    """QAEngine wired with MockLLMClient, real PromptBuilder and CitationParser."""
    return QAEngine(
        llm_client=mock_llm,  # type: ignore[arg-type]
        prompt_builder=PromptBuilder(prompts_root=tmp_prompts),
        citation_parser=CitationParser(),
    )


# ---------------------------------------------------------------------------
# generate_answer tests
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    """Tests for QAEngine.generate_answer."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_qa_response(
        self, engine: QAEngine, good_context: list[RetrievalHit]
    ) -> None:
        """Should return a QAResponse with non-empty answer."""
        response = await engine.generate_answer(
            question="Quyền của người lao động?",
            contexts=good_context,
        )
        assert response.answer != ""
        assert response.used_prompt_version == "legal_qa_v1"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_verified_citation_when_context_matches(
        self, engine: QAEngine, good_context: list[RetrievalHit]
    ) -> None:
        """Citation present in context should be is_verified=True."""
        response = await engine.generate_answer(
            question="Quyền của người lao động?",
            contexts=good_context,
        )
        verified = [c for c in response.citations if c.is_verified]
        assert len(verified) >= 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_context_returns_refusal(
        self, engine: QAEngine
    ) -> None:
        """Empty context should trigger the safety guard and return refusal answer."""
        response = await engine.generate_answer(
            question="Quyền của người lao động?",
            contexts=[],
        )
        assert "không có đủ căn cứ" in response.answer.lower() or "không có văn bản" in response.answer.lower()
        assert response.confidence == 0.0
        assert len(response.warnings) > 0
        # LLM must NOT have been called – citations should be empty.
        assert response.citations == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_id_propagated(
        self, engine: QAEngine, good_context: list[RetrievalHit]
    ) -> None:
        """Supplied trace_id should appear in QAResponse."""
        response = await engine.generate_answer(
            question="test",
            contexts=good_context,
            trace_id="test-trace-001",
        )
        assert response.trace_id == "test-trace-001"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_trace_id_generated(
        self, engine: QAEngine, good_context: list[RetrievalHit]
    ) -> None:
        """When trace_id is not supplied, one should be auto-generated."""
        response = await engine.generate_answer(
            question="test",
            contexts=good_context,
        )
        assert response.trace_id is not None
        assert len(response.trace_id) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_hits_included_in_response(
        self, engine: QAEngine, good_context: list[RetrievalHit]
    ) -> None:
        """QAResponse should carry the retrieval hits for downstream TV5 evaluation."""
        response = await engine.generate_answer(
            question="test",
            contexts=good_context,
        )
        assert response.retrieval_hits == good_context


# ---------------------------------------------------------------------------
# MockLLMClient tests
# ---------------------------------------------------------------------------


class TestMockLLMClient:
    """Ensure MockLLMClient behaves as expected in isolation."""

    @pytest.mark.unit
    def test_generate_returns_fixed_response(self) -> None:
        """generate() should always return the configured response."""
        mock = MockLLMClient(fixed_response="fixed answer")
        assert mock.generate("any prompt") == "fixed answer"

    @pytest.mark.unit
    def test_generate_stream_yields_fixed_response(self) -> None:
        """generate_stream() should yield the fixed response as one chunk."""
        mock = MockLLMClient(fixed_response="streamed answer")
        chunks = list(mock.generate_stream("any prompt"))
        assert "".join(chunks) == "streamed answer"

    @pytest.mark.unit
    def test_default_config_is_valid(self) -> None:
        """MockLLMClient should instantiate without a config argument."""
        mock = MockLLMClient()
        assert isinstance(mock.config, LLMConfig)
