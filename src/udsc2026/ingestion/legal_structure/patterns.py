"""Regular expressions for headings in Vietnamese legal documents.

Heading markers may follow OCR debris or an inline separator.  Every relaxed
pattern therefore requires a non-word left boundary and a valid identifier on
the right; ordinary prose such as ``điều kiện`` cannot become structure.
"""

import re
from typing import Dict, Pattern

_FLAGS = re.MULTILINE | re.UNICODE | re.IGNORECASE
_STRUCTURE_PREFIX = (
    r"(?<![\wÀ-ỹ])(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+|[a-zđ])[.):\-–—]?\s*|"
    r"[•‣▪◦●◆◇➢➤\-–—]+\s*){0,3}"
)
_HEADING_END = r"(?=\s|[.:)\-–—]|$)"

PATTERNS: Dict[str, Pattern[str]] = {
    "chapter": re.compile(
        rf"{_STRUCTURE_PREFIX}(?:Chương|Chuong)(?=\s*(?:[IVXLCDM]+|\d+))\s*"
        rf"([IVXLCDM]+|\d+){_HEADING_END}[\.:]?\s*(.*)$",
        _FLAGS,
    ),
    "section": re.compile(
        rf"{_STRUCTURE_PREFIX}(?:Mục|Muc)(?=\s*\d+)\s*(\d+){_HEADING_END}[\.:]?\s*(.*)$",
        _FLAGS,
    ),
    "article": re.compile(
        rf"{_STRUCTURE_PREFIX}(?:Điều|Dieu)(?=\s*\d+)\s*(\d+){_HEADING_END}[\.:]?\s*(.*)$",
        _FLAGS,
    ),
    "clause": re.compile(r"^\s*(\d+)[\.\)]\s+(.*)$", _FLAGS),
    "point": re.compile(r"^\s*([a-zđ])[\.\)]\s+(.*)$", _FLAGS),
}

# Some sources spell the hierarchy names out.  Keeping these separate avoids
# changing the public, compact patterns above (whose first capture group is
# consistently the identifier).
LABELED_CLAUSE = re.compile(
    rf"{_STRUCTURE_PREFIX}(?:Khoản|Khoan)(?=\s*\d+)\s*(\d+){_HEADING_END}[\.:\)]?\s*(.*)$",
    _FLAGS,
)
LABELED_POINT = re.compile(
    rf"{_STRUCTURE_PREFIX}(?:Điểm|Diem)(?=\s*[a-zđ](?=\s|[\.:)]|$))\s*"
    rf"([a-zđ])(?=\s|[\.:)]|$)[\.:\)]?\s*(.*)$",
    _FLAGS,
)
