"""Centralized patterns used by legal-text cleaning rules."""

import re

_STRUCTURE_PREFIX = (
    r"^\s*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+|[a-zđ])[.):\-–—]?\s*|"
    r"[•‣▪◦●◆◇➢➤\-–—]+\s*){0,3}"
)

LEGAL_STRUCTURE_LINE = re.compile(
    r"^\s*(?:(?:Chương|Chuong|Mục|Muc|Điều|Dieu|Khoản|Khoan|"
    r"Điểm|Diem|Phần|Phan|Phụ\s+lục|Phu\s+luc)\s*$)|"
    rf"{_STRUCTURE_PREFIX}(?:Chương|Chuong)\s+(?:[IVXLCDM]+|\d+)|"
    rf"{_STRUCTURE_PREFIX}(?:Mục|Muc)\s+(?:[IVXLCDM]+|\d+)|"
    rf"{_STRUCTURE_PREFIX}(?:Điều|Dieu)\s+\d+[a-zđ]?|"
    rf"{_STRUCTURE_PREFIX}(?:Khoản|Khoan)\s+\d+[a-zđ]?|"
    rf"{_STRUCTURE_PREFIX}(?:Điểm|Diem)\s+[a-zđ]|"
    rf"{_STRUCTURE_PREFIX}(?:Phần|Phan)\s+(?:thứ\s+)?(?:[IVXLCDM]+|\d+)|"
    rf"{_STRUCTURE_PREFIX}(?:Phụ\s+lục|Phu\s+luc)\s+"
    r"(?:số\s+)?(?:[IVXLCDM]+|\d+)|"
    r"^\s*\d+[a-zđ]?[.)]|^\s*[a-zđ][.)]",
    re.IGNORECASE | re.UNICODE,
)
PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:trang|page)\s*[:#-]?\s*\d{1,4}"
    r"(?:\s*(?:/|trên|of)\s*\d{1,4})?\s*$",
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

# Repair only legal-marker OCR errors at line-start before a numeric label.
# Inline prose, table cells, product names, and codes are deliberately excluded.
LIGHT_OCR_WORD_REPLACEMENTS = {
    "d1eu": "điều",
    "kh0an": "khoản",
}

ABBREVIATION_DEFINITION = re.compile(
    r"([A-ZĐÂÊÔƠƯ][^()]{3,80}?)\s*\(\s*"
    r"(?:sau đây (?:gọi là|gọi tắt là|viết tắt là))\s*"
    r"[\"“]?([^\)\"”]{1,20})[\"”]?\s*\)",
    re.UNICODE | re.IGNORECASE,
)
