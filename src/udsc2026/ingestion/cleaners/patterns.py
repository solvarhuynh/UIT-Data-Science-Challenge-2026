"""Centralized patterns used by legal-text cleaning rules."""

import re


LEGAL_STRUCTURE_LINE = re.compile(
    r"^\s*(?:Chương\s+(?:[IVXLCDM]+|\d+)|Mục\s+\d+|Điều\s+\d+|"
    r"Khoản\s+\d+|Điểm\s+[a-zđ]|\d+[.)]|[a-zđ][.)])(?:\s|$)",
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

ABBREVIATION_DEFINITION = re.compile(
    r"([A-ZĐÂÊÔƠƯ][^()]{3,80}?)\s*\(\s*"
    r"(?:sau đây (?:gọi là|gọi tắt là|viết tắt là))\s*"
    r"[\"“]?([^\)\"”]{1,20})[\"”]?\s*\)",
    re.UNICODE | re.IGNORECASE,
)
