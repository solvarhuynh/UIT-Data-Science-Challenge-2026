"""QA sub-package: prompt management, LLM generation, and citation verification."""

from udsc2026.qa.citation_parser import CitationParser
from udsc2026.qa.prompt_builder import PromptBuilder
from udsc2026.qa.qa_engine import QAEngine

__all__ = ["CitationParser", "PromptBuilder", "QAEngine"]
