"""Orchestrate LLM generation, prompt management, and citation validation."""

import asyncio
import logging
import math
import re
import uuid
from typing import Optional

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
        use_chat_template: bool = True,
    ) -> None:
        """Initialise the QA Engine with its dependencies."""
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
        self._use_chat_template = use_chat_template

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

        # 1.5. Lost-in-the-Middle Re-ordering (Liu et al., Stanford/Berkeley)
        contexts = _reorder_contexts_lost_in_the_middle(contexts)

        #  2. Build prompt
        if self._use_chat_template and hasattr(self._llm, "generate_chat"):
            system_text = self._prompt_builder._load_template("system", prompt_version)
            rag_text = self._prompt_builder._load_template("rag_templates", rag_template)
            from udsc2026.qa.prompt_builder import _build_context_block
            context_block = _build_context_block(contexts)
            user_content = (
                rag_text
                .replace("{context_block}", context_block)
                .replace("{question}", question)
            )
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_content},
            ]
            raw_answer = await asyncio.to_thread(self._llm.generate_chat, messages)
        else:
            prompt = self._prompt_builder.build_prompt(
                question=question,
                contexts=contexts,
                prompt_version=prompt_version,
                rag_template=rag_template,
            )
            logger.debug("trace_id=%s | Prompt length: %d chars.", tid, len(prompt))
            raw_answer = await asyncio.to_thread(self._llm.generate, prompt)

        logger.debug("trace_id=%s | Raw answer length: %d chars.", tid, len(raw_answer))

        #  3.5 Post-process: trim continuation artifacts, preambles & duplicate lines
        clean_answer = _trim_continuation(raw_answer)
        clean_answer = _clean_conversational_preamble(clean_answer)
        clean_answer = _deduplicate_repeated_lines(clean_answer)

        if _is_refused_answer(clean_answer):
            clean_answer = _NO_CONTEXT_ANSWER

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
    """Remove LLM-generated continuation artifacts from the answer."""
    match = _CONTINUATION_PATTERN.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def _reorder_contexts_lost_in_the_middle(contexts: list) -> list:
    """
    Re-orders retrieved contexts according to 'Lost in the Middle' (Liu et al., Stanford/Berkeley):
    Places Rank 1 context at index 0 (Primacy bias) and Rank 2 context at index -1 (Recency bias),
    putting lower-ranked distractors in the middle.
    """
    if len(contexts) <= 2:
        return contexts
    reordered = [None] * len(contexts)
    left = 0
    right = len(contexts) - 1
    for i, item in enumerate(contexts):
        if i % 2 == 0:
            reordered[left] = item
            left += 1
        else:
            reordered[right] = item
            right -= 1
    return reordered


_PREAMBLE_PATTERNS = re.compile(
    r"^(?:\*\*)?câu hỏi:(?:\*\*)?[^\n]*\n+|"
    r"^(?:\*\*)?trả lời:(?:\*\*)?\s*|"
    r"^(?:chắc chắn|dĩ nhiên|tất nhiên)[,!\.\s]*tôi sẽ trả lời[^\n]*\n*|"
    r"^(?:dựa trên|theo) (?:dữ liệu|ngữ cảnh|context|quy định)[^\n]*?(?:trả lời:|\n+)|"
    r"^dựa trên dữ liệu pháp lý được cung cấp(?: trong context)?[,:\s]*|"
    r"^(?:dưới đây là|sau đây là) câu trả lời[^\n]*:\s*|"
    r"^câu trả lời của bạn dựa trên context được cung cấp:\s*|"
    r"^\*\*[^\*\n]+\*\*\n+",
    re.IGNORECASE,
)


def _clean_conversational_preamble(text: str) -> str:
    """Strip greetings, title headers, and conversational preamble from answer."""
    if not text:
        return text
    text = text.strip()
    if text.startswith("Dựa trên dữ liệu pháp lý được cung cấp, ") and len(text) > 90 and "không có đủ căn cứ" not in text:
        text = text[len("Dựa trên dữ liệu pháp lý được cung cấp, "):].lstrip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
    match = _PREAMBLE_PATTERNS.match(text)
    if match:
        text = text[match.end():].lstrip()
    if text.lower().startswith("trả lời:"):
        text = text[len("trả lời:"):].lstrip()
    
    tail_patterns = [
        r"\n+lưu ý rằng context chỉ cung cấp[^\n]*$",
        r"\n+hy vọng rằng đáp án trên giúp bạn[^\n]*$",
        r"\n+chúc bạn học tốt[^\n]*$",
    ]
    for pat in tail_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).rstrip()

    return text.strip()


def _deduplicate_repeated_lines(text: str) -> str:
    """Remove consecutive duplicate lines or repeated bullet points (anti-loop filter)."""
    if not text:
        return text
    lines = text.splitlines()
    clean_lines = []
    seen_count = 0
    last_line = None

    for line in lines:
        stripped = line.strip().lower()
        if stripped == last_line and len(stripped) > 5:
            seen_count += 1
            if seen_count >= 2:
                continue
        else:
            last_line = stripped
            seen_count = 1
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def _is_refused_answer(text: str) -> bool:
    if not text.strip():
        return True
    text_lower = text.lower()
    refusal_phrases = [
        "không có đủ căn cứ",
        "không có thông tin",
        "chưa đủ dữ liệu",
        "chỉ sử dụng thông tin trong context để trả lời",
        "chỉ sử dụng thông tin trong context",
    ]
    return any(p in text_lower for p in refusal_phrases)
