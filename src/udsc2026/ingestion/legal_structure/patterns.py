"""Regular expressions for headings in Vietnamese legal documents.

All patterns are anchored at the beginning of a physical line.  Lower-level
patterns are deliberately only applied by the parser while an article is
active; using them independently would confuse ordinary numbered lists with
clauses and points.
"""

import re
from typing import Dict, Pattern

_FLAGS = re.MULTILINE | re.UNICODE | re.IGNORECASE

PATTERNS: Dict[str, Pattern[str]] = {
    "chapter": re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)[\.:]?\s*(.*)$", _FLAGS),
    "section": re.compile(r"^\s*Mục\s+(\d+)[\.:]?\s*(.*)$", _FLAGS),
    "article": re.compile(r"^\s*Điều\s+(\d+)[\.:]?\s*(.*)$", _FLAGS),
    "clause": re.compile(r"^\s*(\d+)[\.\)]\s+(.*)$", _FLAGS),
    "point": re.compile(r"^\s*([a-zđ])[\.\)]\s+(.*)$", _FLAGS),
}

# Some sources spell the hierarchy names out.  Keeping these separate avoids
# changing the public, compact patterns above (whose first capture group is
# consistently the identifier).
LABELED_CLAUSE = re.compile(r"^\s*Khoản\s*(\d+)[\.:\)]?\s*(.*)$", _FLAGS)
LABELED_POINT = re.compile(r"^\s*Điểm\s*([a-zđ])[\.:\)]?\s*(.*)$", _FLAGS)
