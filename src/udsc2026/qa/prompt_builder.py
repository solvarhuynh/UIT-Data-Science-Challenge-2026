"""Prompt builder: reads versioned Markdown templates and renders full prompts."""

import logging
from pathlib import Path
from typing import Optional

from udsc2026.contracts.retrieval import RetrievalHit

logger = logging.getLogger(__name__)

# Root of the prompts directory relative to the package root.
# TV3 owns all files under this tree.
_PROMPTS_ROOT = Path(__file__).resolve().parents[3] / "prompts"

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
        self._root = prompts_root or _PROMPTS_ROOT
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
        system_text = self._load_template("system", prompt_version)
        rag_text = self._load_template("rag_templates", rag_template)

        context_block = _build_context_block(contexts)
        user_turn = rag_text.replace("{context_block}", context_block).replace(
            "{question}", question
        )

        return f"{system_text}\n\n{user_turn}"

    def get_prompt_version_id(self, prompt_version: str) -> str:
        """Return a stable identifier for the given prompt version.

        Used by ``QAEngine`` to populate ``QAResponse.used_prompt_version``
        without coupling the engine to file-system details.

        Args:
            prompt_version: The version string (e.g. ``"legal_qa_v1"``).

        Returns:
            The same version string – extended in future to include a hash.
        """
        return prompt_version

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
        """
        cache_key = f"{subfolder}/{name}"
        if cache_key not in self._cache:
            path = self._root / subfolder / f"{name}.md"
            if not path.exists():
                raise FileNotFoundError(
                    f"Prompt template not found: '{path}'.  "
                    f"Create '{name}.md' under 'prompts/{subfolder}/'."
                )
            self._cache[cache_key] = path.read_text(encoding="utf-8").strip()
            logger.debug("Loaded prompt template '%s'.", cache_key)
        return self._cache[cache_key]


# ---------------------------------------------------------------------------
# Module-level helper (no class state required)
# ---------------------------------------------------------------------------


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
        meta_parts = [
            f"law_name={hit.law_name or 'N/A'}",
            f"article={hit.article or 'N/A'}",
            f"clause={hit.clause or 'N/A'}",
            f"chunk_id={hit.chunk_id}",
        ]
        meta = " | ".join(meta_parts)
        lines.append(f"[{rank}] ({meta})\n{hit.text.strip()}")

    return "\n\n".join(lines)
