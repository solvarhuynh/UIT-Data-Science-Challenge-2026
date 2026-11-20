"""Vietnamese tokenization that preserves exact legal references."""

import re

from pyvi import ViTokenizer

_PROTECTED = re.compile(
    r"(?:\b(?:(?:Điều|Khoản)\s+\d+[a-zđ]?|Điểm\s+[a-zđ])\b|"
    r"\b\d{1,4}/\d{4}/[a-zđ]{1,12}\d{0,4}\b)",
    re.IGNORECASE,
)


def tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese while retaining legal citation phrases as terms."""
    if text is None or not text.strip():
        return []
    tokens: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        tokens.extend(_tokenize_part(text[cursor : match.start()]))
        tokens.append(match.group(0))
        cursor = match.end()
    tokens.extend(_tokenize_part(text[cursor:]))
    return tokens


def _tokenize_part(text: str) -> list[str]:
    if not text.strip():
        return []
    segmented = ViTokenizer.tokenize(text)
    return [token.lower() for token in segmented.split() if token]
