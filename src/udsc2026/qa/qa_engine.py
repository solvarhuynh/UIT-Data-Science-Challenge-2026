"""Orchestrate LLM generation, prompt management, and citation validation."""

import asyncio
import logging
import math
import re
import uuid
from typing import Optional, Protocol

from udsc2026.contracts.qa import QAResponse
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.llm.client import LLMClient
from udsc2026.qa.citation_parser import CitationParser
from udsc2026.qa.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# Threshold below which all context hits are considered too low-quality to
# justify calling the LLM.  TV1 can override this via LLMConfig / configs/.
_MIN_CONTEXT_SCORE = 0.0

# Message returned when the engine refuses to answer due to insufficient context.
_NO_CONTEXT_ANSWER = (
    "Dựa trên dữ liệu pháp lý được cung cấp, không có đủ căn cứ để trả lời câu hỏi này."
)

# Pattern that signals the LLM has started generating an additional
# question/context block beyond the intended answer – a "continuation" artifact
# observed when the model treats the few-shot RAG template as a repeating
# sequence.  Any text from these markers onward is trimmed.
_CONTINUATION_PATTERN = re.compile(
    r"(?:\n\s*\[\d+\]\s*\(|\nCÂU HỎi:|\nTRẢ LỜI:|\n\s*\[\d{1,3}\]\s*law_name)",
    re.IGNORECASE,
)


class ContextExpander(Protocol):
    """Post-rerank context expansion interface used only for prompt input."""

    def expand(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Return bounded prompt contexts without mutating original hits."""


class QAEngine:
    """Coordinate prompt building, LLM inference and citation verification.

    TV1 constructs this class via Dependency Injection and calls
    ``generate_answer`` with the question and the ranked retrieval hits
    produced by TV2/TV3 sparse/hybrid retrieval and TV5 reranking.

    The engine never performs retrieval itself; it is a pure *generation*
    component that expects a ready-made ``list[RetrievalHit]`` as input.

    Example::

        engine = QAEngine(
            llm_client=LLMClient(LLMConfig()),
            prompt_builder=PromptBuilder(),
            citation_parser=CitationParser(),
        )
        response = await engine.generate_answer(
            question="Quyền lợi của người lao động?",
            contexts=reranked_hits,
        )
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: Optional[PromptBuilder] = None,
        citation_parser: Optional[CitationParser] = None,
        min_context_score: float = _MIN_CONTEXT_SCORE,
        context_expander: Optional[ContextExpander] = None,
    ) -> None:
        """Initialise the QA Engine with its dependencies.

        Args:
            llm_client: A loaded ``LLMClient`` (or ``MockLLMClient`` for
                tests).  TV1 injects this via DI at application startup.
            prompt_builder: Instance of ``PromptBuilder``.  Defaults to a
                new instance with the standard prompts directory.
            citation_parser: Instance of ``CitationParser``.  Defaults to a
                new instance.
            min_context_score: Minimum effective retrieval score required for
                at least one context hit. The effective score prefers output
                from the latest retrieval stage over earlier component scores.
        """
        raw_min_context_score: object = min_context_score
        if (
            isinstance(raw_min_context_score, bool)
            or not isinstance(raw_min_context_score, (int, float))
            or not math.isfinite(float(raw_min_context_score))
            or raw_min_context_score < 0
        ):
            raise ValueError("min_context_score must be a finite non-negative number")
        self._llm = llm_client
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._citation_parser = citation_parser or CitationParser()
        self._min_score = float(raw_min_context_score)
        self._context_expander = context_expander

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_answer(
        self,
        question: str,
        contexts: list[RetrievalHit],
        prompt_version: str = "legal_qa_v1",
        rag_template: str = "default_rag_v1",
        trace_id: Optional[str] = None,
    ) -> QAResponse:
        """Generate a grounded legal answer with citation verification.

        Implements the full TV3 pipeline:
        1. Safety guard – reject if contexts are empty or all below threshold.
        2. Build prompt from versioned template.
        3. Call the local LLM.
        4. Parse and validate citations against the supplied context.
        5. Return a ``QAResponse`` with all metadata.

        Args:
            question: The user's legal question string (pre-validated by TV1).
            contexts: Ordered ``RetrievalHit`` list from retrieval/rerank.
                Must not be empty for a meaningful answer.
            prompt_version: Identifier of the system prompt to use
                (file in ``prompts/system/``).
            trace_id: Optional identifier propagated from TV1 for log
                correlation.  A UUID is generated when not supplied.

        Returns:
            ``QAResponse`` containing the answer, verified citations, and
            full pipeline metadata.
        """
        tid = trace_id or str(uuid.uuid4())
        logger.info(
            "QAEngine.generate_answer | trace_id=%s | question_len=%d | n_contexts=%d",
            tid,
            len(question),
            len(contexts),
        )

        #  1. Safety guard
        refusal = self._check_context_quality(contexts)
        if refusal is not None:
            logger.warning("trace_id=%s | Refusing to call LLM: %s", tid, refusal)
            return QAResponse(
                answer=_NO_CONTEXT_ANSWER,
                citations=[],
                used_prompt_version=prompt_version,
                retrieval_hits=contexts,
                confidence=0.0,
                warnings=[refusal],
                trace_id=tid,
            )

        prompt_contexts = (
            self._context_expander.expand(contexts)
            if self._context_expander is not None
            else contexts
        )
        if not prompt_contexts:
            prompt_contexts = contexts

        #  2. Build both compatibility text and native chat turns. The real
        # Qwen client consumes messages; lightweight mocks keep using text.
        prompt = self._prompt_builder.build_prompt(
            question=question,
            contexts=prompt_contexts,
            prompt_version=prompt_version,
            rag_template=rag_template,
        )
        messages = self._prompt_builder.build_messages(
            question=question,
            contexts=prompt_contexts,
            prompt_version=prompt_version,
            rag_template=rag_template,
        )
        logger.debug("trace_id=%s | Prompt length: %d chars.", tid, len(prompt))

        #  3. Call LLM
        generate_messages = getattr(self._llm, "generate_messages", None)
        if callable(generate_messages):
            raw_answer = await asyncio.to_thread(generate_messages, messages)
        else:
            raw_answer = await asyncio.to_thread(self._llm.generate, prompt)
        logger.debug("trace_id=%s | Raw answer length: %d chars.", tid, len(raw_answer))

        #  3.5 Post-process: trim LLM continuation artifacts
        # Some small models repeat the RAG template and self-generate extra
        # questions after answering.  Trim everything from the first
        # continuation marker onwards and log a warning for monitoring.
        clean_answer = _trim_continuation(raw_answer)
        if clean_answer != raw_answer:
            trimmed_chars = len(raw_answer) - len(clean_answer)
            logger.warning(
                "trace_id=%s | Trimmed %d chars of LLM continuation artifact.",
                tid,
                trimmed_chars,
            )

        #  4. Parse and validate citations
        verified_citations, citation_warnings = (
            self._citation_parser.parse_and_validate(clean_answer, contexts)
        )

        #  5. Compute confidence proxy
        verified_count = sum(1 for c in verified_citations if c.is_verified)
        total_count = len(verified_citations)
        confidence = float(verified_count) / total_count if total_count > 0 else None

        all_warnings = list(citation_warnings)
        if clean_answer != raw_answer:
            all_warnings.append(
                "Phần thừa do mô hình tự sinh sau câu trả lời đã được cắt bỏ tự động."
            )
        if not verified_citations:
            all_warnings.append(
                "Câu trả lời không chứa trích dẫn pháp lý nào được nhận diện."
            )

        return QAResponse(
            answer=clean_answer,
            citations=verified_citations,
            used_prompt_version=self._prompt_builder.get_prompt_version_id(
                prompt_version
            ),
            retrieval_hits=contexts,
            confidence=confidence,
            warnings=all_warnings,
            trace_id=tid,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_context_quality(self, contexts: list[RetrievalHit]) -> Optional[str]:
        """Return a refusal reason string if context quality is insufficient.

        Args:
            contexts: The retrieval hits to evaluate.

        Returns:
            A human-readable refusal reason, or ``None`` if quality is
            acceptable and generation should proceed.
        """
        if not contexts:
            return "Không có văn bản pháp luật nào được truy hồi cho câu hỏi này."

        # Check if at least one hit meets the minimum score threshold.
        if self._min_score > 0.0:
            best_score = max(
                (_effective_retrieval_score(hit) for hit in contexts),
                default=0.0,
            )
            if best_score < self._min_score:
                return (
                    f"Điểm truy hồi cao nhất ({best_score:.4f}) thấp hơn ngưỡng "
                    f"tối thiểu ({self._min_score:.4f}) – không đủ căn cứ."
                )
        return None


def _effective_retrieval_score(hit: RetrievalHit) -> float:
    """Return the score produced by the latest available retrieval stage."""

    for score in (
        hit.final_score,
        hit.rerank_score,
        hit.hybrid_score,
        hit.score,
        hit.dense_score,
        hit.sparse_score,
    ):
        if score is not None:
            return score
    return 0.0


def _trim_continuation(text: str) -> str:
    """Remove LLM-generated continuation artifacts from the answer.

    Detects markers such as ``\n[2] (law_name=...)`` or ``\nCÂU HỎi:``
    that indicate the model has started repeating the RAG template beyond
    the first answer.  Everything from the first such marker is discarded.

    Args:
        text: Raw output from ``LLMClient.generate``.

    Returns:
        Cleaned answer string with continuation stripped, or the original
        string when no continuation markers are detected.
    """
    match = _CONTINUATION_PATTERN.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text
