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
# Format A: "Điều X, khoản Y" – article before clause (most common).
# Note: re.IGNORECASE does not reliably handle Vietnamese diacritical characters,
# so both lowercase and uppercase forms are listed explicitly.
_INLINE_PATTERN = re.compile(
    r"(?:(?P<law_pre>[\w\s\-]+?)\s+)?(?:[Đđ]i[ềê]u)\s+(?P<article>\d+[a-z]?)"
    r"(?:[,\s]+[Kk]ho[ảă]n\s+(?P<clause>\d+))?"
    r"(?:[,\s]+[Đđ]i[ểê]m\s+(?P<point>[a-zđ]))?",
    re.UNICODE,
)

# Format B: "khoản X Điều Y" – clause before article (common in BTC warmup answers)
# e.g. "Căn cứ khoản 2 Điều 38 Luật An toàn vệ sinh lao động năm 2015"
_INLINE_REVERSED_PATTERN = re.compile(
    r"[Kk]hoản\s+(?P<clause>\d+)\s+[Đđ]iều\s+(?P<article>\d+[a-z]?)",
    re.UNICODE,
)

# Law name patterns: matches the most common Vietnamese legal document prefixes.
_LAW_NAME_LEADING_RE = re.compile(
    r"((?:Bộ\s+luật|Luật|Nghị\s+định(?:\s+số)?|Thông\s+tư|Quyết\s+định)"
    r"[^,.\n\[\]]{3,70})",
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
# Noise words that should not be treated as law name prefixes in inline matches.
_INLINE_NOISE_WORDS = frozenset(
    [
        "tại",
        "theo",
        "cụ thể",
        "cụ",
        "thể",
        "quy định",
        "quy",
        "định",
        "xem",
        "căn cứ",
        "căn",
        "cứ",
        "dựa",
        "vào",
        "của",
    ]
)


class CitationParser:
    """Extract and validate legal citations from LLM-generated answer text.

    Workflow::

        parser = CitationParser()
        citations = parser.parse_citations(answer_text)
        verified  = parser.validate_citations(citations, retrieval_hits)

    Any citation whose explicit law/article/clause/point fields cannot be
    matched to at least one supplied ``RetrievalHit`` is marked
    ``is_verified=False`` and receives a warning flagging a potential
    hallucination.
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
        ``contexts`` matches its article and every other locator explicitly
        named by the citation. Case-insensitive partial matching is accepted
        only for law names that may be abbreviated.

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

    law_candidate = _ARTICLE_RE.sub(" ", raw)
    law_candidate = _CLAUSE_RE.sub(" ", law_candidate)
    law_candidate = _POINT_RE.sub(" ", law_candidate)
    law_candidate = re.sub(r"[\s,;:]+", " ", law_candidate).strip(" -")
    law_candidate = re.sub(
        r"^(?:theo|tại|căn\s+cứ)\s+",
        "",
        law_candidate,
        flags=re.IGNORECASE,
    ).strip()
    law_name = law_candidate or None

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
        raw_hit_point = hit.metadata.get("point")
        hit_point = raw_hit_point if isinstance(raw_hit_point, str) else None
        if (
            _locator_matches(citation.article, hit.article, required=True)
            and _law_matches(citation.law_name, hit.law_name)
            and _locator_matches(citation.clause, hit.clause)
            and _locator_matches(citation.point, hit_point)
        ):
            return hit
    return None


def _locator_matches(
    citation_value: Optional[str],
    hit_value: Optional[str],
    *,
    required: bool = False,
) -> bool:
    """Match one legal locator, treating only omitted citation fields as wildcards."""

    if citation_value is None:
        return not required
    if hit_value is None:
        return False
    return _normalize_text(citation_value) == _normalize_text(hit_value)


def _law_matches(citation_law: Optional[str], hit_law: Optional[str]) -> bool:
    """Fuzzy-match law names: check if one is contained in the other.

    If citation_law is None (e.g. LLM wrote "theo Điều 5, Khoản 1"), it matches
    the hit as long as the article number matches and no conflicting law is specified.
    """
    if citation_law is None:
        return True
    if hit_law is None:
        return False
    a = _normalize_text(citation_law)
    b = _normalize_text(hit_law)
    return a in b or b in a


def _normalize_text(value: str) -> str:
    """Case-fold and collapse whitespace for stable legal-label comparison."""

    return " ".join(value.casefold().split())


def _citation_display(citation: Citation) -> str:
    """Format a citation as a short human-readable string for logging."""
    parts = [p for p in [citation.law_name, citation.article, citation.clause] if p]
    return ", ".join(parts) if parts else "(unknown)"


def _extract_inline_citations(text: str) -> list[Citation]:
    """Scan free-form text for natural-language legal references.

    Performs two sub-passes to catch both common Vietnamese citation formats:

    * **Format A** – ``"Điều X, khoản Y ..."`` (article before clause)
    * **Format B** – ``"khoản X Điều Y ..."`` (clause before article,
      common in BTC warmup reference answers, e.g.
      ``"Căn cứ khoản 2 Điều 38 Luật An toàn vệ sinh lao động 2015"``).

    Each unique ``(article, law_name)`` pair is returned once.

    Args:
        text: The raw LLM answer string (already confirmed to contain no
            bracketed citations by the caller).

    Returns:
        Deduplicated list of ``Citation`` objects with ``is_verified=False``.
    """
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    # ── Format A: "Điều X, khoản Y" ─────────────────────────────────────
    for match in _INLINE_PATTERN.finditer(text):
        citation = _build_inline_citation(text, match)
        if citation is None:
            continue
        # Dedup on (article, clause, law_name) so that the same article with
        # different clauses (e.g. "Điều 10, Khoản 1" vs "Điều 10, Khoản 2")
        # is treated as two distinct citations.
        key = (citation.article or "", citation.clause or "", citation.law_name or "")
        if key not in seen:
            seen.add(key)
            citations.append(citation)

    # ── Format B: "khoản X Điều Y" (reversed order) ──────────────────────
    for match in _INLINE_REVERSED_PATTERN.finditer(text):
        article_num = match.group("article")
        clause_num = match.group("clause")
        article = f"Điều {article_num}"
        clause = f"Khoản {clause_num}"

        law_name = _extract_law_name_around(text, match.start(), match.end())
        key = (article, clause, law_name or "")
        if key not in seen:
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


def _build_inline_citation(text: str, match: re.Match) -> Optional[Citation]:  # type: ignore[type-arg]
    """Build a ``Citation`` from a single ``_INLINE_PATTERN`` match.

    Args:
        text: Full answer text (used to look up surrounding law name).
        match: Regex match object from ``_INLINE_PATTERN``.

    Returns:
        ``Citation`` or ``None`` if the match has no recognisable article.
    """
    article_num = match.group("article")
    if not article_num:
        return None
    article = f"Điều {article_num}"

    clause_num = match.group("clause")
    clause = f"Khoản {clause_num}" if clause_num else None

    point_match_str = match.group("point")
    point = f"Điểm {point_match_str}" if point_match_str else None

    law_name = _extract_law_name_around(text, match.start(), match.end())
    if law_name:
        law_name = re.sub(
            r"[,\s]+(?:tại|theo|cụ thể)?$", "", law_name, flags=re.IGNORECASE
        ).strip()

    return Citation(
        law_name=law_name,
        article=article,
        clause=clause,
        point=point,
        is_verified=False,
    )


def _extract_law_name_around(
    text: str, start_pos: int, end_pos: int
) -> Optional[str]:
    """Extract the law name from text immediately before or after a citation match.

    Searches up to 90 characters before the match for a law name prefix
    (e.g. ``"Bộ luật Lao động 2019"`` or ``"Nghị định số 87/2018/NĐ-CP"``),
    then falls back to searching the 90 characters after the match.

    Args:
        text: Full answer string.
        start_pos: Start index of the citation match.
        end_pos: End index of the citation match.

    Returns:
        Extracted and stripped law name string, or ``None`` if not found.
    """
    # 1. Look backwards (e.g. "Theo Bộ luật Lao động 2019, Điều 5")
    preceding = text[max(0, start_pos - 90) : start_pos]
    law_lead = _LAW_NAME_LEADING_RE.search(preceding)
    if law_lead:
        return law_lead.group(1).strip()

    # 2. Look forwards (e.g. "Điều 5 Bộ luật Lao động 2019")
    trailing = text[end_pos : end_pos + 90]
    law_trail = re.match(
        r"[,\s]*((?:Bộ\s+luật|Luật|Nghị\s+định(?:\s+số)?|Thông\s+tư|Quyết\s+định)"
        r"[^,.\n\[\]]{3,70})",
        trailing,
        re.UNICODE,
    )
    if law_trail:
        return law_trail.group(1).strip()

    return None
