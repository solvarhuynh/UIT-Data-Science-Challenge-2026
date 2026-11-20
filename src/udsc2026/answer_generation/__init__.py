"""QA sub-package: prompt management, LLM generation, and citation verification."""

from udsc2026.answer_generation.citation_parser import CitationParser
from udsc2026.answer_generation.prompt_builder import PromptBuilder
from udsc2026.answer_generation.engine import QAEngine

__all__ = ["CitationParser", "PromptBuilder", "QAEngine"]
