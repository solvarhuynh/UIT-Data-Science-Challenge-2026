"""Centralized patterns used by legal-text cleaning rules."""

import re

_STRUCTURE_PREFIX = (
    r"(?<![\wÀ-ỹ])(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+|[a-zđ])[.):\-–—]?\s*|"
    r"[•‣▪◦●◆◇➢➤\-–—]+\s*){0,3}"
)

LEGAL_STRUCTURE_LINE = re.compile(
    rf"{_STRUCTURE_PREFIX}(?:Chương|Chuong)(?=\s*(?:[IVXLCDM]+|\d+))\s*(?:[IVXLCDM]+|\d+)(?=\s|[.:)\-–—]|$)|"
    rf"{_STRUCTURE_PREFIX}(?:Mục|Muc)(?=\s*\d+)\s*\d+(?=\s|[.:)\-–—]|$)|"
    rf"{_STRUCTURE_PREFIX}(?:Điều|Dieu)(?=\s*\d+)\s*\d+(?=\s|[.:)\-–—]|$)|"
    rf"{_STRUCTURE_PREFIX}(?:Khoản|Khoan)(?=\s*\d+)\s*\d+(?=\s|[.:)\-–—]|$)|"
    rf"{_STRUCTURE_PREFIX}(?:Điểm|Diem)(?=\s*[a-zđ](?=\s|[.:)]|$))\s*[a-zđ](?=\s|[.:)]|$)|"
    r"^\s*\d+[.)]|^\s*[a-zđ][.)]",
    re.IGNORECASE | re.UNICODE,
)
PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:trang\s*)?\d{1,4}(?:\s*(?:/|trên)\s*\d{1,4})?\s*$",
    re.IGNORECASE | re.UNICODE,
)
TABLE_OF_CONTENTS = re.compile(r"^\s*mục\s+lục\s*$", re.IGNORECASE | re.UNICODE)
TOC_ENTRY = re.compile(
    r"^\s*(?:chương|mục|điều)?\s*[IVXLCDM\d].*?(?:\.{2,}|\s+)\d{1,4}\s*$",
    re.IGNORECASE | re.UNICODE,
)
WATERMARK_LINE = re.compile(
    r"^\s*(?:dự\s*thảo|draft|confidential|internal\s+use\s+only|"
    r"bản\s*(?:nháp|scan))\s*$",
    re.IGNORECASE | re.UNICODE,
)
HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]+", re.UNICODE)
MULTIPLE_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
LEADING_BULLET = re.compile(r"^(\s*)[•‣▪◦●◆◇➢➤]\s*")
ONLY_SYMBOLS = re.compile(r"^[\W_]+$", re.UNICODE)

# OCR substitutions are deliberately whole-word and limited to unambiguous
# legal markers.  They repair extraction noise without rewriting corpus prose.
LIGHT_OCR_WORD_REPLACEMENTS = {
    "s0": "số",
    "d1eu": "điều",
    "kh0an": "khoản",
}

ABBREVIATION_DEFINITION = re.compile(
    r"([A-ZĐÂÊÔƠƯ][^()]{3,80}?)\s*\(\s*"
    r"(?:sau đây (?:gọi là|gọi tắt là|viết tắt là))\s*"
    r"[\"“]?([^\)\"”]{1,20})[\"”]?\s*\)",
    re.UNICODE | re.IGNORECASE,
)
