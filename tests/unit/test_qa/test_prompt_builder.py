"""Unit tests for PromptBuilder – no GPU or model weights required."""

import tempfile
from pathlib import Path

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.qa.prompt_builder import PromptBuilder, _build_context_block


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_prompts(tmp_path: Path) -> Path:
    """Create a minimal prompts directory tree in a temp folder."""
    system_dir = tmp_path / "system"
    rag_dir = tmp_path / "rag_templates"
    system_dir.mkdir()
    rag_dir.mkdir()

    (system_dir / "legal_qa_v1.md").write_text(
        "Bạn là trợ lý pháp luật. Chỉ dùng CONTEXT.",
        encoding="utf-8",
    )
    (rag_dir / "default_rag_v1.md").write_text(
        "CONTEXT:\n{context_block}\n\nCÂU HỎI:\n{question}\n\nTRẢ LỜI:",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def builder(tmp_prompts: Path) -> PromptBuilder:
    """Return a PromptBuilder pointing at the temp prompts directory."""
    return PromptBuilder(prompts_root=tmp_prompts)


@pytest.fixture
def sample_hits() -> list[RetrievalHit]:
    """Two retrieval hits for prompt rendering tests."""
    return [
        RetrievalHit(
            chunk_id="doc001_article_10_clause_1",
            doc_id="doc001",
            text="Người lao động có quyền làm việc.",
            score=0.92,
            law_name="Bộ luật Lao động 2019",
            article="Điều 10",
            clause="Khoản 1",
        ),
        RetrievalHit(
            chunk_id="doc002_article_5",
            doc_id="doc002",
            text="Doanh nghiệp có trách nhiệm trả lương đúng hạn.",
            score=0.80,
            law_name="Bộ luật Lao động 2019",
            article="Điều 5",
        ),
    ]


# ---------------------------------------------------------------------------
# build_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for PromptBuilder.build_prompt."""

    @pytest.mark.unit
    def test_prompt_contains_question(
        self, builder: PromptBuilder, sample_hits: list[RetrievalHit]
    ) -> None:
        """The rendered prompt must include the original question."""
        question = "Quyền của người lao động là gì?"
        prompt = builder.build_prompt(question=question, contexts=sample_hits)
        assert question in prompt

    @pytest.mark.unit
    def test_prompt_contains_context_text(
        self, builder: PromptBuilder, sample_hits: list[RetrievalHit]
    ) -> None:
        """Each hit's text must appear in the rendered prompt."""
        prompt = builder.build_prompt(
            question="Câu hỏi?",
            contexts=sample_hits,
        )
        for hit in sample_hits:
            assert hit.text in prompt

    @pytest.mark.unit
    def test_prompt_contains_system_instruction(
        self, builder: PromptBuilder, sample_hits: list[RetrievalHit]
    ) -> None:
        """The system instruction from the prompt file must appear in output."""
        prompt = builder.build_prompt(
            question="Câu hỏi?",
            contexts=sample_hits,
        )
        assert "Chỉ dùng CONTEXT" in prompt

    @pytest.mark.unit
    def test_missing_template_raises_file_not_found(
        self, builder: PromptBuilder, sample_hits: list[RetrievalHit]
    ) -> None:
        """Requesting a non-existent prompt version should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            builder.build_prompt(
                question="test",
                contexts=sample_hits,
                prompt_version="nonexistent_v99",
            )

    @pytest.mark.unit
    def test_empty_contexts_uses_sentinel(
        self, builder: PromptBuilder
    ) -> None:
        """Empty context list should produce a sentinel string in the prompt."""
        prompt = builder.build_prompt(question="Câu hỏi?", contexts=[])
        assert "Không có văn bản pháp luật" in prompt

    @pytest.mark.unit
    def test_template_caching(
        self, builder: PromptBuilder, sample_hits: list[RetrievalHit]
    ) -> None:
        """Second call should use cached template (same result, no FileNotFoundError)."""
        p1 = builder.build_prompt(question="Q1?", contexts=sample_hits)
        p2 = builder.build_prompt(question="Q1?", contexts=sample_hits)
        assert p1 == p2


# ---------------------------------------------------------------------------
# _build_context_block tests (module-level helper)
# ---------------------------------------------------------------------------


class TestBuildContextBlock:
    """Tests for the _build_context_block helper function."""

    @pytest.mark.unit
    def test_ranks_are_sequential(self, sample_hits: list[RetrievalHit]) -> None:
        """Each hit should be numbered starting from 1."""
        block = _build_context_block(sample_hits)
        assert "[1]" in block
        assert "[2]" in block

    @pytest.mark.unit
    def test_metadata_present_in_block(self, sample_hits: list[RetrievalHit]) -> None:
        """law_name and article fields must appear in the context block."""
        block = _build_context_block(sample_hits)
        assert "Bộ luật Lao động 2019" in block
        assert "Điều 10" in block

    @pytest.mark.unit
    def test_empty_list_returns_sentinel(self) -> None:
        """Empty context should return the sentinel string."""
        block = _build_context_block([])
        assert "Không có văn bản pháp luật" in block
