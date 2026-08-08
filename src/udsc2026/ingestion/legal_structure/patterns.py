"""Regular expressions for headings in Vietnamese legal documents.

All patterns are anchored at the beginning of a physical line.  Lower-level
patterns are deliberately only applied by the parser while an article is
active; using them independently would confuse ordinary numbered lists with
clauses and points.
"""

import re
from typing import Dict, Pattern

_FLAGS = re.MULTILINE | re.UNICODE | re.IGNORECASE
_STRUCTURE_PREFIX = (
    r"^\s*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+|[a-zđ])[.):\-–—]?\s*|"
    r"[•‣▪◦●◆◇➢➤\-–—]+\s*){0,3}"
)

PATTERNS: Dict[str, Pattern[str]] = {
    "chapter": re.compile(
        rf"{_STRUCTURE_PREFIX}(?:Chương|Chuong|CHUONG)\s+([IVXLCDM]+|\d+)[\.:]?\s*(.*)$",
        _FLAGS,
    ),
    "section": re.compile(
        rf"{_STRUCTURE_PREFIX}(?:Mục|Muc|MUC)\s+(\d+)[\.:]?\s*(.*)$",
        _FLAGS,
    ),
    "article": re.compile(
        rf"{_STRUCTURE_PREFIX}(?:Điều|Dieu|DIEU)\s+(\d+)[\.:]?\s*(.*)$",
        _FLAGS,
    ),
    "clause": re.compile(r"^\s*(\d+)[\.\)]\s+(.*)$", _FLAGS),
    "point": re.compile(r"^\s*([a-zđ])[\.\)]\s+(.*)$", _FLAGS),
}

# Some sources spell the hierarchy names out.  Keeping these separate avoids
# changing the public, compact patterns above (whose first capture group is
# consistently the identifier).
LABELED_CLAUSE = re.compile(
    rf"{_STRUCTURE_PREFIX}(?:Khoản|Khoan|KHOAN)\s*(\d+)[\.:\)]?\s*(.*)$",
    _FLAGS,
)
LABELED_POINT = re.compile(
    rf"{_STRUCTURE_PREFIX}(?:Điểm|Diem|DIEM)\s*([a-zđ])[\.:\)]?\s*(.*)$",
    _FLAGS,
)
