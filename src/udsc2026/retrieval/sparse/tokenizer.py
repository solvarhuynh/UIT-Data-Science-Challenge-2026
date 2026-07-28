"""Vietnamese tokenizer preserving legally meaningful exact-match terms."""

import re

from pyvi import ViTokenizer

_PROTECTED = re.compile(r"\b(?:Điều|Khoản|Điểm)\s+\d+[a-zđ]?\b", re.IGNORECASE)


def tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese while preserving accents and legal citation phrases.

    BM25 relies on exact term matching.  Keeping ``Điều 10`` as one token makes
    an exact query for that article more precise than splitting it into
    ``điều`` and ``10``; the remaining terms are lowercased normally.
    """
    if text is None or not text.strip():
        return []
    tokens: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        tokens.extend(_tokenize_part(text[cursor:match.start()]))
        tokens.append(match.group(0))
        cursor = match.end()
    tokens.extend(_tokenize_part(text[cursor:]))
    return tokens


def _tokenize_part(text: str) -> list[str]:
    if not text.strip():
        return []
    segmented = ViTokenizer.tokenize(text)
    return [token.lower() for token in segmented.split() if token]
