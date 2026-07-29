"""Citation extraction and verification for Vietnamese legal QA answers."""

import logging
import re
from typing import Optional

from udsc2026.contracts.qa import Citation
from udsc2026.contracts.retrieval import RetrievalHit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for common Vietnamese legal citation formats
#
# Format 1 – Bracketed (preferred, instructed in system prompt):
#   [Bộ luật Lao động 2019, Điều 10, Khoản 1, Điểm a]
#   [Điều 15 Luật Doanh nghiệp 2020]
#   [Khoản 2 Điều 5 Luật Đất đai 2024]
#   [Bộ luật Hình sự 2015, Điều 134]
#
# Format 2 – Inline natural language (fallback for non-compliant LLM output):
#   "cụ thể tại Điều 5, khoản 1"
#   "theo Điều 10 Bộ luật Lao động 2019"
#   "tại Điều 7, khoản 1 Bộ luật Lao động 2019"
# ---------------------------------------------------------------------------

# Matches bracketed citations like [... Điều X ...]
_BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")

# Matches inline natural-language citations not wrapped in brackets.
# Captures an optional leading law fragment before "Điều", the article
# number, and an optional clause.
_INLINE_PATTERN = re.compile(
    r"(?:(?P<law_pre>[\w\s\-]+?)\s+)?Điều\s+(?P<article>\d+[a-z]?)"
    r"(?:[,\s]+[Kk]hoản\s+(?P<clause>\d+))?",
    re.UNICODE,
)

# Named-group pattern to extract law_name, article, clause, point from
# a single citation string (used for bracketed format).
_ARTICLE_RE = re.compile(
    r"Điều\s+(?P<article>\d+[a-z]?)",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(
    r"Khoản\s+(?P<clause>\d+)",
    re.IGNORECASE,
)
_POINT_RE = re.compile(
    r"Điểm\s+(?P<point>[a-zđ])",
    re.IGNORECASE,
)
# Law name: everything before "Điều", "Khoản" or a trailing comma.
_LAW_NAME_RE = re.compile(
    r"^(?P<law>.+?)(?:,\s*Điều|\s+Điều|,\s*Khoản|\s+Khoản)",
    re.IGNORECASE,
)

# Noise words that should not be treated as law name prefixes in inline matches.
_INLINE_NOISE_WORDS = frozenset(
    [
        "tại", "theo", "cụ thể", "cụ", "thể", "quy định", "quy", "định",
        "xem", "căn cứ", "căn", "cứ", "dựa", "vào", "của",
    ]
)


class CitationParser:
    """Extract and validate legal citations from LLM-generated answer text.

    Workflow::

        parser = CitationParser()
        citations = parser.parse_citations(answer_text)
        verified  = parser.validate_citations(citations, retrieval_hits)

    Any citation whose ``law_name`` + ``article`` pair cannot be matched to
    at least one supplied ``RetrievalHit`` is marked ``is_verified=False``
    and receives a warning string flagging a potential hallucination.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_citations(self, answer: str) -> list[Citation]:
        """Extract all legal citations from an LLM answer string.

        Performs a two-pass scan:

        1. **Bracket pass** – looks for ``[... Điều X ...]`` patterns
           (preferred format instructed in the system prompt).
        2. **Inline pass** – fallback scan for natural-language references
           such as ``"tại Điều 5, khoản 1"`` when the LLM does not use
           brackets.  Duplicates already found in the bracket pass are
           suppressed.

        Citations that cannot be parsed are silently skipped and logged at
        DEBUG level.

        Args:
            answer: Raw answer string produced by ``LLMClient.generate``.

        Returns:
            List of ``Citation`` objects (``is_verified`` defaults to
            ``False`` until ``validate_citations`` is called).
        """
        citations: list[Citation] = []

        # ── Pass 1: bracketed citations ──────────────────────────────────
        raw_matches = _BRACKET_PATTERN.findall(answer)
        for raw in raw_matches:
            citation = _parse_single_citation(raw)
            if citation is not None:
                citations.append(citation)
            else:
                logger.debug("Could not parse bracket candidate: '%s'.", raw)

        # ── Pass 2: inline natural-language citations (fallback) ──────────
        # Only run when no bracketed citations were found; avoids double-counting
        # when the LLM correctly uses brackets for every reference.
        if not citations:
            inline_citations = _extract_inline_citations(answer)
            citations.extend(inline_citations)
            if inline_citations:
                logger.debug(
                    "Extracted %d inline citation(s) (no bracket citations found).",
                    len(inline_citations),
                )

        logger.debug("Total citations extracted: %d.", len(citations))
        return citations

    def validate_citations(
        self,
        citations: list[Citation],
        contexts: list[RetrievalHit],
    ) -> list[Citation]:
        """Verify each citation against the supplied retrieval context.

        A citation is *verified* when at least one ``RetrievalHit`` in
        ``contexts`` matches on ``law_name`` AND ``article`` (case-insensitive
        partial match is accepted for law names that may be abbreviated).

        Citations that fail verification are marked ``is_verified=False`` and
        populated with a warning string.  The original citation list is not
        mutated; a new list is returned.

        Args:
            citations: List produced by ``parse_citations``.
            contexts: The ``RetrievalHit`` list that was fed to the LLM as
                context.  Validation uses the metadata fields of each hit.

        Returns:
            New list of ``Citation`` objects with ``is_verified`` and
            ``warning`` fields populated.
        """
        verified: list[Citation] = []
        for citation in citations:
            matched_hit = _find_matching_hit(citation, contexts)
            if matched_hit is not None:
                verified.append(
                    citation.model_copy(
                        update={
                            "is_verified": True,
                            "chunk_id": citation.chunk_id or matched_hit.chunk_id,
                            "source": citation.source or matched_hit.source,
                            "warning": None,
                        }
                    )
                )
            else:
                warning = (
                    f"Trích dẫn '{_citation_display(citation)}' không khớp với "
                    f"bất kỳ văn bản nào trong ngữ cảnh được cung cấp "
                    f"(nghi ngờ hallucination)."
                )
                logger.warning(warning)
                verified.append(
                    citation.model_copy(
                        update={"is_verified": False, "warning": warning}
                    )
                )
        return verified

    def parse_and_validate(
        self,
        answer: str,
        contexts: list[RetrievalHit],
    ) -> tuple[list[Citation], list[str]]:
        """Convenience wrapper: parse then validate in one call.

        Args:
            answer: LLM-generated answer string.
            contexts: Retrieval hits used as context during generation.

        Returns:
            Tuple of ``(verified_citations, warnings)`` where ``warnings`` is
            a list of warning strings for every unverified citation.
        """
        citations = self.parse_citations(answer)
        verified = self.validate_citations(citations, contexts)
        warnings = [c.warning for c in verified if c.warning is not None]
        return verified, warnings


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, not part of the public class API)
# ---------------------------------------------------------------------------


def _parse_single_citation(raw: str) -> Optional[Citation]:
    """Parse one raw bracket string into a ``Citation``.

    Returns ``None`` when the string contains no recognisable legal reference.
    """
    raw = raw.strip()

    article_match = _ARTICLE_RE.search(raw)
    if article_match is None:
        # No "Điều X" found — not a legal citation we can handle.
        return None

    article = f"Điều {article_match.group('article')}"

    clause_match = _CLAUSE_RE.search(raw)
    clause = f"Khoản {clause_match.group('clause')}" if clause_match else None

    point_match = _POINT_RE.search(raw)
    point = f"Điểm {point_match.group('point')}" if point_match else None

    law_name_match = _LAW_NAME_RE.match(raw)
    law_name: Optional[str] = None
    if law_name_match:
        law_name = law_name_match.group("law").strip().strip(",")

    return Citation(
        law_name=law_name,
        article=article,
        clause=clause,
        point=point,
        is_verified=False,
    )


def _find_matching_hit(
    citation: Citation,
    contexts: list[RetrievalHit],
) -> Optional[RetrievalHit]:
    """Return the first ``RetrievalHit`` that matches the citation, or None."""
    for hit in contexts:
        if _article_matches(citation.article, hit.article) and _law_matches(
            citation.law_name, hit.law_name
        ):
            return hit
    return None


def _article_matches(citation_article: Optional[str], hit_article: Optional[str]) -> bool:
    """Compare article numbers with normalised whitespace."""
    if citation_article is None or hit_article is None:
        return False
    return citation_article.strip().lower() == hit_article.strip().lower()


def _law_matches(citation_law: Optional[str], hit_law: Optional[str]) -> bool:
    """Fuzzy-match law names: check if one is contained in the other.

    If citation_law is None (e.g. LLM wrote "theo Điều 5, Khoản 1"), it matches
    the hit as long as the article number matches and no conflicting law is specified.
    """
    if citation_law is None or hit_law is None:
        return True
    a = citation_law.strip().lower()
    b = hit_law.strip().lower()
    return a in b or b in a


def _citation_display(citation: Citation) -> str:
    """Format a citation as a short human-readable string for logging."""
    parts = [p for p in [citation.law_name, citation.article, citation.clause] if p]
    return ", ".join(parts) if parts else "(unknown)"


def _extract_inline_citations(text: str) -> list[Citation]:
    """Scan free-form text for natural-language legal references.

    Extracts citations like ``"Theo Bộ luật Lao động 2019, Điều 5, Khoản 1"`` or
    ``"theo Điều 10 Bộ luật Lao động 2019"`` that do not use bracket
    notation.  Each unique ``(article, law_name)`` pair is returned once.

    Args:
        text: The raw LLM answer string (already confirmed to contain no
            bracketed citations by the caller).

    Returns:
        Deduplicated list of ``Citation`` objects with ``is_verified=False``.
    """
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    for match in _INLINE_PATTERN.finditer(text):
        article_num = match.group("article")
        article = f"Điều {article_num}"

        clause_num = match.group("clause")
        clause = f"Khoản {clause_num}" if clause_num else None

        law_name: Optional[str] = None

        # 1. Try to extract law name from preceding text (e.g. "Theo Bộ luật Lao động 2019, Điều 5")
        start_pos = match.start()
        preceding = text[max(0, start_pos - 80) : start_pos]
        law_lead = re.search(
            r"((?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư)[^,.\n\[\]]{3,60})",
            preceding,
            re.UNICODE,
        )
        if law_lead:
            law_name = law_lead.group(1).strip()
        else:
            # 2. Try to extract law name from trailing text (e.g. "Điều 5 Bộ luật Lao động 2019")
            end_pos = match.end()
            trailing = text[end_pos : end_pos + 80]
            law_trail = re.match(
                r"[,\s]*((?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư)[^,.\n\[\]]{3,60})",
                trailing,
                re.UNICODE,
            )
            if law_trail:
                law_name = law_trail.group(1).strip()

        # Clean up any trailing punctuation/words from law_name
        if law_name:
            law_name = re.sub(r"[,\s]+(?:tại|theo|cụ thể)?$", "", law_name, flags=re.IGNORECASE).strip()

        # Deduplicate on (article, law_name or "")
        key = (article, law_name or "")
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            Citation(
                law_name=law_name,
                article=article,
                clause=clause,
                is_verified=False,
            )
        )

    return citations
