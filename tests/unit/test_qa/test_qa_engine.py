"""Unit tests for QAEngine using MockLLMClient – no GPU required."""

from pathlib import Path
from typing import Any

import pytest

import udsc2026.qa.qa_engine as qa_engine_module
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
def engine(mock_llm: MockLLMClient, tmp_prompts: Path) -> QAEngine:
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
    async def test_offloads_blocking_llm_call(
        self,
        engine: QAEngine,
        good_context: list[RetrievalHit],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        async def fake_to_thread(function: object, prompt: str) -> str:
            calls.append(prompt)
            return function(prompt)  # type: ignore[operator]

        monkeypatch.setattr(
            qa_engine_module.asyncio,
            "to_thread",
            fake_to_thread,
        )

        await engine.generate_answer("Quyền lao động?", good_context)

        assert len(calls) == 1

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
    async def test_empty_context_returns_refusal(self, engine: QAEngine) -> None:
        """Empty context should trigger the safety guard and return refusal answer."""
        response = await engine.generate_answer(
            question="Quyền của người lao động?",
            contexts=[],
        )
        assert (
            "không có đủ căn cứ" in response.answer.lower()
            or "không có văn bản" in response.answer.lower()
        )
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

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parent_expansion_changes_prompt_but_not_response_provenance(
        self,
        tmp_prompts: Path,
        good_context: list[RetrievalHit],
    ) -> None:
        class CapturingLLM:
            def __init__(self) -> None:
                self.prompt = ""

            def generate(self, prompt: str) -> str:
                self.prompt = prompt
                return "Trả lời [Bộ luật Lao động 2019, Điều 10, Khoản 1]"

        class StubExpander:
            def expand(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
                return [
                    hits[0].model_copy(
                        update={"text": "TOÀN VĂN PARENT ĐÃ ĐƯỢC HYDRATE"}
                    )
                ]

        llm = CapturingLLM()
        parent_engine = QAEngine(
            llm_client=llm,  # type: ignore[arg-type]
            prompt_builder=PromptBuilder(prompts_root=tmp_prompts),
            citation_parser=CitationParser(),
            context_expander=StubExpander(),
        )

        response = await parent_engine.generate_answer("Câu hỏi?", good_context)

        assert "TOÀN VĂN PARENT ĐÃ ĐƯỢC HYDRATE" in llm.prompt
        assert response.retrieval_hits == good_context

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_accepts_high_final_score_from_reranked_pipeline(
        self,
        mock_llm: MockLLMClient,
        tmp_prompts: Path,
        good_context: list[RetrievalHit],
    ) -> None:
        """The final reranker score must drive the context-quality guard."""

        reranked_context = good_context[0].model_copy(
            update={
                "score": None,
                "hybrid_score": 0.7,
                "rerank_score": 0.9,
                "final_score": 0.9,
            }
        )
        threshold_engine = QAEngine(
            llm_client=mock_llm,  # type: ignore[arg-type]
            prompt_builder=PromptBuilder(prompts_root=tmp_prompts),
            citation_parser=CitationParser(),
            min_context_score=0.5,
        )

        response = await threshold_engine.generate_answer(
            "Quyền của người lao động?",
            [reranked_context],
        )

        assert response.confidence == pytest.approx(1.0)
        assert any(citation.is_verified for citation in response.citations)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_final_score_takes_precedence_over_legacy_score(
        self,
        mock_llm: MockLLMClient,
        tmp_prompts: Path,
        good_context: list[RetrievalHit],
    ) -> None:
        """A stale vector score cannot hide a low final pipeline score."""

        context = good_context[0].model_copy(update={"score": 0.99, "final_score": 0.1})
        threshold_engine = QAEngine(
            llm_client=mock_llm,  # type: ignore[arg-type]
            prompt_builder=PromptBuilder(prompts_root=tmp_prompts),
            min_context_score=0.5,
        )

        response = await threshold_engine.generate_answer("question", [context])

        assert response.confidence == 0.0
        assert response.citations == []


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf"), True, "0.5"])
def test_rejects_invalid_min_context_score(
    threshold: Any,
    mock_llm: MockLLMClient,
    tmp_prompts: Path,
) -> None:
    with pytest.raises(ValueError, match="min_context_score"):
        QAEngine(
            llm_client=mock_llm,  # type: ignore[arg-type]
            prompt_builder=PromptBuilder(prompts_root=tmp_prompts),
            min_context_score=threshold,
        )


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
