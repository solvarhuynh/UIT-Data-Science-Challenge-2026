"""Prompt builder: reads versioned Markdown templates and renders full prompts."""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from udsc2026.contracts.retrieval import RetrievalHit

logger = logging.getLogger(__name__)

_PROMPTS_PATH_ENV = "UDSC2026_PROMPTS_PATH"
_TEMPLATE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_TEMPLATE_SUBFOLDERS = frozenset({"system", "rag_templates"})

# Sentinel returned by ``_build_context_block`` when context is empty.
_EMPTY_CONTEXT_SENTINEL = "[Không có văn bản pháp luật nào được truy hồi]"


class PromptBuilder:
    """Assemble a fully rendered prompt from versioned templates and context hits.

    Reads ``prompts/system/<version>.md`` for the system instruction and
    ``prompts/rag_templates/<rag_template>.md`` for the user-turn template.
    Templates are loaded once on first use and cached in memory.

    Example::

        builder = PromptBuilder()
        prompt = builder.build_prompt(
            question="Quyền của người lao động là gì?",
            contexts=retrieval_hits,
            prompt_version="legal_qa_v1",
        )
    """

    def __init__(self, prompts_root: Optional[Path] = None) -> None:
        """Initialise the builder with the prompts directory.

        Args:
            prompts_root: Override for the default ``prompts/`` directory.
                Useful in tests that supply a temporary directory.
        """
        configured_root = os.getenv(_PROMPTS_PATH_ENV, "prompts")
        self._root = prompts_root or Path(configured_root)
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        contexts: list[RetrievalHit],
        prompt_version: str = "legal_qa_v1",
        rag_template: str = "default_rag_v1",
    ) -> str:
        """Render a complete chat prompt for the LLM.

        Combines the system instruction, the numbered context block from
        ``contexts``, and the user question into a single string suitable
        for causal language model inference.

        Args:
            question: The user's legal question (already validated by TV1).
            contexts: Ordered list of ``RetrievalHit`` objects returned by the
                retrieval/rerank pipeline.  Order matters: the first hit has
                the highest relevance score.
            prompt_version: Name of the system prompt file (without ``.md``)
                stored under ``prompts/system/``.
            rag_template: Name of the RAG user-turn template (without ``.md``)
                stored under ``prompts/rag_templates/``.

        Returns:
            Fully rendered prompt string ready to be fed to ``LLMClient.generate``.
        """
        messages = self.build_messages(
            question=question,
            contexts=contexts,
            prompt_version=prompt_version,
            rag_template=rag_template,
        )
        return "\n\n".join(message["content"] for message in messages)

    def build_messages(
        self,
        question: str,
        contexts: list[RetrievalHit],
        prompt_version: str = "legal_qa_v1",
        rag_template: str = "default_rag_v1",
    ) -> list[dict[str, str]]:
        """Render separate system/user turns for tokenizer chat templates."""

        system_text = self._load_template("system", prompt_version)
        rag_text = self._load_template("rag_templates", rag_template)
        context_block = _build_context_block(contexts)
        user_turn = rag_text.replace("{context_block}", context_block).replace(
            "{question}", question
        )
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_turn},
        ]

    def get_prompt_version_id(self, prompt_version: str) -> str:
        """Return a stable identifier for the given prompt version.

        Used by ``QAEngine`` to populate ``QAResponse.used_prompt_version``
        without coupling the engine to file-system details.

        Args:
            prompt_version: The version string (e.g. ``"legal_qa_v1"``).

        Returns:
            The same version string – extended in future to include a hash.
        """
        return _validate_template_name(prompt_version)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_template(self, subfolder: str, name: str) -> str:
        """Load a Markdown template from disk with an in-memory cache.

        Args:
            subfolder: Directory under ``prompts/`` (e.g. ``"system"``).
            name: File name without the ``.md`` extension.

        Returns:
            Stripped file content as a string.

        Raises:
            FileNotFoundError: When the template file does not exist.
            ValueError: When the template name is unsafe or escapes its folder.
        """
        if subfolder not in _TEMPLATE_SUBFOLDERS:
            raise ValueError(f"Unsupported prompt template folder: {subfolder}")
        validated_name = _validate_template_name(name)
        cache_key = f"{subfolder}/{validated_name}"
        if cache_key not in self._cache:
            template_root = (self._root / subfolder).resolve()
            path = (template_root / f"{validated_name}.md").resolve()
            try:
                path.relative_to(template_root)
            except ValueError as exc:
                raise ValueError("Prompt template path escapes its folder") from exc
            if not path.is_file():
                raise FileNotFoundError(
                    f"Prompt template not found: '{path}'.  "
                    f"Create '{validated_name}.md' under 'prompts/{subfolder}/'."
                )
            self._cache[cache_key] = path.read_text(encoding="utf-8").strip()
            logger.debug("Loaded prompt template '%s'.", cache_key)
        return self._cache[cache_key]


# ---------------------------------------------------------------------------
# Module-level helper (no class state required)
# ---------------------------------------------------------------------------


def _validate_template_name(name: str) -> str:
    """Return an allowlisted file stem with no path syntax."""

    if not isinstance(name, str) or _TEMPLATE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "prompt template name must contain only ASCII letters, digits, "
            "underscores, or hyphens (maximum 64 characters)"
        )
    return name


def _build_context_block(contexts: list[RetrievalHit]) -> str:
    """Render an ordered list of retrieval hits into a human-readable block.

    Each hit is prefixed with its 1-based rank and key citation metadata so
    the LLM can construct accurate ``[law_name, article, clause]`` references.

    Args:
        contexts: Retrieval results in relevance order (highest first).

    Returns:
        Multi-line string representing all context entries, or a sentinel
        string when ``contexts`` is empty.
    """
    if not contexts:
        return _EMPTY_CONTEXT_SENTINEL

    lines: list[str] = []
    for rank, hit in enumerate(contexts, start=1):
        raw_point = hit.metadata.get("point")
        point = raw_point if isinstance(raw_point, str) else None
        meta_parts = [
            f"law_name={hit.law_name or 'N/A'}",
            f"article={hit.article or 'N/A'}",
            f"clause={hit.clause or 'N/A'}",
            f"point={point or 'N/A'}",
            f"chunk_id={hit.chunk_id}",
        ]
        meta = " | ".join(meta_parts)
        lines.append(f"[{rank}] ({meta})\n{hit.text.strip()}")

    return "\n\n".join(lines)
