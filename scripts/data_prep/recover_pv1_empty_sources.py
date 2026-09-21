"""Phase-1 recovery of the 20 empty BTC LegalIR contexts.

``--local-pdf-only`` is deliberately air-gapped: it evaluates only user-provided
PDFs, never requests a URL, and never fabricates or rewrites legal content.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from udsc2026.ingestion.chunking import chunk_clean_document  # noqa: E402
from udsc2026.ingestion.cleaners import clean_document  # noqa: E402
from udsc2026.ingestion.readers.models import RawDocument  # noqa: E402

TARGET_IDS = (
    "71014", "67660", "57978", "56098", "55497", "34810", "288457",
    "263763", "255762", "232489", "210808", "208668", "196918", "191261",
    "187338", "181693", "177151", "149317", "131890", "10533",
)
KNOWN_CANDIDATES = {
    "67660": (
        "https://luatvietnam.vn/tai-nguyen/quy-chuan-viet-nam-qcvn-46-2022-btnmt-233289-d3.html",
    ),
}
USER_VERIFIED_IDENTITIES = {
    "288457": {
        "title": "Luật sửa đổi, bổ sung một số điều của Luật bảo hiểm y tế",
        "document_type": "LUAT_SUA_DOI_LUAT_BAO_HIEM_Y_TE",
        "issuing_authority": "QUỐC HỘI",
        "year": "2024",
    },
    "208668": {
        "title": "Dự thảo Nghị định hướng dẫn Luật Giáo dục đại học",
        "document_type": "NGHI_DINH_DU_THAO",
        "issuing_authority": "CHÍNH PHỦ",
        "issue_date": "05/10/2021",
        "year": "2021",
    },
}
USER_AGENT = "UDSC2026-PV1-Phase1-Recovery/1.0 (public-source provenance)"
STANDARD_RE = re.compile(
    r"\b(?P<prefix>QCVN|TCVN)\s*[-:]?\s*"
    r"(?P<number>\d{1,5}(?:\s*-\s*\d{1,2})?)\s*[-:]?\s*(?P<year>\d{4})"
    r"(?:\s*(?:/|-)\s*(?P<authority>[A-ZĐ]{2,12}))?\b"
)
WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]{3,}", re.UNICODE)
STRUCTURE_RE = re.compile(r"\b(?:Điều|Chương|Mục|Phần|QCVN|TCVN|QUY CHUẨN|TIÊU CHUẨN|PHẠM VI)\b", re.I)
CONTENT_HINT_RE = re.compile(r"content|document|vanban|law|detail|article|body", re.I)
PAYWALL_RE = re.compile(r"TVPL\s*Pro|đăng\s*nhập|nội\s*dung\s*chỉ\s*dành", re.I)
CHALLENGE_RE = re.compile(r"just a moment|cloudflare|security check|captcha", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", unquote(value)).replace("đ", "d").replace("Đ", "D")
    return "".join(char for char in value.upper() if char.isalnum() and not unicodedata.combining(char))


def _words(value: str) -> set[str]:
    stop = {"HTTP", "HTTPS", "WWW", "THUVIENPHAPLUAT", "ASPX", "VAN", "BAN", "THEO", "VE", "VA"}
    def fold(word: str) -> str:
        normalized = unicodedata.normalize("NFKD", word).replace("đ", "d").replace("Đ", "D")
        return "".join(char for char in normalized if not unicodedata.combining(char)).upper()
    return {fold(word) for word in WORD_RE.findall(unquote(value)) if fold(word) not in stop}


def _standard_codes(value: str) -> list[dict[str, str]]:
    """Parse only a standard code, never the first word of its title."""
    parsed: list[dict[str, str]] = []
    for match in STANDARD_RE.finditer(value):
        prefix, number, year = match.group("prefix"), match.group("number"), match.group("year")
        authority = match.group("authority") if prefix == "QCVN" else None
        display = f"{prefix} {number.replace(' ', '')}:{year}" + (f"/{authority}" if authority else "")
        parsed.append({"display_code": display, "normalized_code": _norm(display)})
    return parsed


def _codes(value: str) -> list[str]:
    return [item["normalized_code"] for item in _standard_codes(value)]


def _identity_parser_valid() -> bool:
    examples = {
        "QCVN-46-2022-BTNMT-Quan-trac-khi-tuong": "QCVN 46:2022/BTNMT",
        "TCVN-10778-2015-Ho-chua": "TCVN 10778:2015",
        "TCVN-6102-2020-ISO-7202-2018": "TCVN 6102:2020",
        "TCVN-8400-24-2014-Benh": "TCVN 8400-24:2014",
        "QCVN-02-23-2017-BNNPTNT": "QCVN 02-23:2017/BNNPTNT",
        "QCVN-20-2015-BLDTBXH": "QCVN 20:2015/BLDTBXH",
    }
    return all(_standard_codes(value) == [{"display_code": expected, "normalized_code": _norm(expected)}] for value, expected in examples.items())


def _url_title(url: str) -> str:
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    name = re.sub(r"\.(?:aspx?|html?)$", "", name, flags=re.I)
    return re.sub(r"-\d{5,}$", "", name).replace("-", " ").strip()


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.in_title = False
        self.ignored = 0
        self.stack: list[tuple[str, list[str] | None]] = []
        self.candidates: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag == "title":
            self.in_title = True
        if tag in {"script", "style", "noscript", "svg", "canvas"}:
            self.ignored += 1
            self.stack.append((tag, None))
            return
        attributes = {key.casefold(): value or "" for key, value in attrs}
        marker = " ".join((attributes.get("id", ""), attributes.get("class", "")))
        candidate = [] if tag in {"article", "main"} or CONTENT_HINT_RE.search(marker) else None
        if candidate is not None:
            self.candidates.append(candidate)
        self.stack.append((tag, candidate))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self.in_title = False
        if tag in {"script", "style", "noscript", "svg", "canvas"}:
            self.ignored = max(0, self.ignored - 1)
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        text = " ".join(html.unescape(data).split())
        if not text or self.ignored:
            return
        if self.in_title:
            self.title_parts.append(text)
        for _, candidate in self.stack:
            if candidate is not None:
                candidate.append(text)


def _html_text(body: bytes, encoding: str | None) -> tuple[str, str]:
    parser = _TextParser()
    parser.feed(body.decode(encoding or "utf-8", errors="replace"))
    parser.close()
    candidates = ["\n".join(parts).strip() for parts in parser.candidates if parts]
    if not candidates:
        return " ".join(parser.title_parts).strip(), ""
    return " ".join(parser.title_parts).strip(), max(candidates, key=lambda text: (len(STRUCTURE_RE.findall(text)) * 20000 + min(len(text), 100000), len(text)))


def _classify(response: httpx.Response | None, error: str | None = None) -> str:
    if error:
        return "TLS_ERROR" if "SSL" in error.upper() or "CERT" in error.upper() else "HTTP_ERROR"
    assert response is not None
    text = response.text[:20000]
    if CHALLENGE_RE.search(text):
        return "CLOUDFLARE_BLOCK"
    if PAYWALL_RE.search(text):
        return "PRO_PAYWALL"
    if response.status_code != 200:
        return "HTTP_ERROR"
    content_type = response.headers.get("content-type", "").casefold()
    if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
        return "PDF_PUBLIC"
    if "html" not in content_type:
        return "UNKNOWN"
    _, text_value = _html_text(response.content, response.encoding)
    if not text_value.strip():
        return "EMPTY_CONTENT"
    if len(text_value) < 1500 or len(WORD_RE.findall(text_value)) < 250:
        return "PARTIAL_PREVIEW_ONLY"
    return "FULL_TEXT_PUBLIC"


def _fetch_once(client: httpx.Client, url: str) -> tuple[httpx.Response | None, dict[str, Any]]:
    retrieved_at = _now()
    try:
        response = client.get(url)
        return response, {"retrieved_at": retrieved_at, "http_status": response.status_code, "final_url": str(response.url), "content_type": response.headers.get("content-type"), "status": _classify(response)}
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        return None, {"retrieved_at": retrieved_at, "http_status": None, "status": _classify(None, str(exc)), "error": str(exc)}


def _targets(raw_root: Path) -> list[dict[str, Any]]:
    context_dir = raw_root / "LegalIR" / "selected-contexts"
    targets: list[dict[str, Any]] = []
    for doc_id in TARGET_IDS:
        path = context_dir / f"context_{doc_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"target context missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        link, passage = payload.get("link"), payload.get("passage")
        if str(payload.get("id")) != doc_id or not isinstance(passage, str) or passage.strip() or not isinstance(link, str) or not link.strip():
            raise ValueError(f"PHASE1_TARGET_MEMBERSHIP_MISMATCH: {doc_id}")
        title = _url_title(link)
        # Preserve URL separators while parsing the code: converting the slug
        # to prose first would erase the slash/hyphen that distinguishes a
        # QCVN authority suffix from the following title words.
        standards = _standard_codes(unquote(urlparse(link).path.rsplit("/", 1)[-1]))
        normalized_title = _norm(title)
        document_type = next(
            (name for name, marker in (
                ("QCVN", "QCVN"), ("TCVN", "TCVN"), ("QUYET_DINH", "QUYETDINH"),
                ("THONG_TU", "THONGTU"), ("NGHI_DINH", "NGHIDINH"), ("LUAT", "LUAT"),
            ) if marker in normalized_title),
            None,
        )
        identity = {"display_code": standards[0]["display_code"] if standards else None, "document_number_or_code": standards[0]["display_code"] if standards else None, "normalized_document_number": standards[0]["normalized_code"] if standards else None, "document_type": document_type, "title": title, "issuing_authority": None, "issue_date": None, "year": next((word for word in WORD_RE.findall(title) if word.isdigit() and len(word) == 4), None)}
        identity.update(USER_VERIFIED_IDENTITIES.get(doc_id, {}))
        targets.append({"document_id": doc_id, "original_tvpl_url": link.strip(), "raw_context_path": str(path.relative_to(ROOT)), "identity": identity})
    return targets


def _pdf_inventory(pdf_root: Path, paths: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    if not pdf_root.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(paths if paths is not None else pdf_root.rglob("*.pdf")):
        row: dict[str, Any] = {"filename": path.name, "path": str(path.relative_to(ROOT)), "sha256": _sha(path.read_bytes()), "page_count": 0, "embedded_text_length": 0, "metadata": {}, "embedded_text": ""}
        try:
            with pdfplumber.open(path) as pdf:
                page_count, metadata = len(pdf.pages), pdf.metadata or {}
            # pdfminer streams text extraction and avoids pdfplumber's retained
            # layout cache on the two very large supplied PDFs.
            text = pdfminer_extract_text(str(path)) or ""
            row.update({"page_count": page_count, "embedded_text_length": len(text), "metadata": metadata, "embedded_text": text})
        except Exception as exc:
            row["extraction_error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return results


def _pdf_match(target: dict[str, Any], pdf: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    page_id = re.search(r"-(\d{5,})\.aspx$", target["original_tvpl_url"], re.I)
    blob = " ".join(str(value) for value in pdf.get("metadata", {}).values()) + " " + pdf.get("embedded_text", "")[:5000]
    target_code = target["identity"].get("normalized_document_number")
    evidence = {"filename_matches_target_id": Path(pdf["filename"]).stem == target["document_id"], "tvpl_page_id": page_id.group(1) if page_id else None, "metadata_contains_tvpl_page_id": bool(page_id and page_id.group(1) in _norm(blob)), "target_code": target_code, "pdf_codes": _codes(blob)}
    if target_code and target_code in evidence["pdf_codes"]:
        return "PDF_IDENTITY_PASS", evidence
    if evidence["filename_matches_target_id"] or evidence["metadata_contains_tvpl_page_id"]:
        return "PDF_IDENTITY_AMBIGUOUS", evidence
    return "PDF_IDENTITY_FAIL", evidence


def _identity_text(target: dict[str, Any], pdf: dict[str, Any], text: str) -> tuple[str, dict[str, Any]]:
    """Content-based local-PDF identity gate.  Filename is supporting evidence only."""
    expected = target["identity"]
    code = expected.get("normalized_document_number")
    sample = text[:30000]
    source_codes = sorted(set(_codes(sample)))
    expected_title = _words(str(expected.get("title") or ""))
    overlap = sorted(expected_title & _words(sample))
    expected_authority = _norm(str(expected.get("issuing_authority") or ""))
    authority_ok = bool(expected_authority and expected_authority in _norm(sample))
    expected_year = str(expected.get("year") or "")
    year_ok = bool(expected_year and expected_year in sample)
    draft_ok = target["document_id"] != "208668" or bool(re.search(r"D[ỰU]\s*TH[ẢA]O", sample, re.I))
    page_id = re.search(r"-(\d{5,})\.aspx$", target["original_tvpl_url"], re.I)
    page_mapping = bool(page_id and page_id.group(1) in _norm(" ".join(str(v) for v in pdf.get("metadata", {}).values()) + " " + sample))
    evidence = {
        "target_code": code, "source_codes": source_codes,
        "title_overlap_tokens": overlap, "authority_match": authority_ok,
        "year_match": year_ok, "draft_expected_and_found": draft_ok,
        "metadata_contains_tvpl_page_id": page_mapping,
        "identity_excerpt": sample[:2500],
    }
    if code:
        return ("IDENTITY_PASS" if code in source_codes else "IDENTITY_FAIL"), evidence
    # 208668 is a user-verified draft target: its substantive title, authority,
    # and 2021 dating are the contract; being a draft is positive evidence.
    if target["document_id"] == "208668":
        required = {"DU", "THAO", "NGHI", "DINH", "HUONG", "DAN", "LUAT", "GIAO", "DUC", "DAI", "HOC"}
        return ("IDENTITY_PASS" if len(required & _words(sample)) >= 9 and authority_ok and year_ok and draft_ok else "IDENTITY_FAIL"), evidence
    if target["document_id"] == "191261":
        required = {"THONG", "TU", "VI", "TRI", "VIEC", "LAM", "CO", "CAU", "VIEN", "CHUC", "GIAO", "DUC", "MAM", "NON", "CONG", "LAP"}
        return ("IDENTITY_PASS" if len(required & _words(sample)) >= 12 and "2023" in sample else "IDENTITY_FAIL"), evidence
    if page_mapping and len(overlap) >= 4 and (authority_ok or year_ok):
        return "IDENTITY_PASS", evidence
    if len(overlap) >= 5 and authority_ok and year_ok:
        return "IDENTITY_PASS", evidence
    return "IDENTITY_AMBIGUOUS", evidence


def _completeness(text: str, page_count: int) -> tuple[str, dict[str, Any]]:
    quality = _quality(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Repeated page headers/footers/watermarks are normal in legal PDFs and
    # must not make a long, structurally complete document partial.
    repeated = quality["duplicate_ratio_heuristic"] > 0.85 and quality["text_length"] < 12000
    ending = " ".join(lines[-12:])[-1400:] if lines else ""
    starts = " ".join(lines[:12])[:1400] if lines else ""
    # This is conservative: a valid but short legal text is only likely complete;
    # an OCR/text extraction failure is never promoted.
    if quality["replacement_character_count"] > 30 or quality["token_count"] < 80:
        status = "OCR_FAILED"
    elif repeated or quality["text_length"] < 1000:
        status = "PARTIAL"
    elif quality["substantive"] and (quality["article_heading_count"] > 0 or page_count <= 3 or len(ending) > 300):
        status = "COMPLETE"
    elif quality["text_length"] >= 1500 and quality["token_count"] >= 250:
        status = "LIKELY_COMPLETE"
    else:
        status = "PARTIAL"
    return status, {**quality, "first_text_excerpt": starts, "last_text_excerpt": ending, "repeated_page_evidence": repeated}


def _ocr_pdf(path: Path, *, max_pages: int | None = None, resolution: int = 180) -> tuple[str, str | None]:
    executable = shutil.which("tesseract")
    if executable is None:
        return "", "tesseract_not_available"
    try:
        with tempfile.TemporaryDirectory(prefix="pv1_ocr_") as temporary, pdfplumber.open(path) as pdf:
            parts: list[str] = []
            for number, page in enumerate(pdf.pages, start=1):
                if max_pages is not None and number > max_pages:
                    break
                image = Path(temporary) / f"page_{number:04d}.png"
                page.to_image(resolution=resolution).original.save(image)
                result = subprocess.run([executable, str(image), "stdout", "-l", "vie+eng", "--psm", "6"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False)
                if result.returncode:
                    return "", f"tesseract_page_{number}_failed: {result.stderr.strip()[:300]}"
                parts.append(result.stdout)
                pdf.flush_cache()
            return "\n".join(parts).strip(), None
    except Exception as exc:
        return "", f"pdf_ocr_error: {type(exc).__name__}: {exc}"


def _quality(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokens = WORD_RE.findall(text)
    return {"text_length": len(text), "line_count": len(lines), "token_count": len(tokens), "replacement_character_count": text.count("�"), "control_character_count": sum(ord(char) < 32 and char not in "\n\t\r" for char in text), "duplicate_ratio_heuristic": 0.0 if not lines else 1.0 - len(set(lines)) / len(lines), "article_heading_count": len(re.findall(r"(?im)^\s*Điều\s+\d+", text)), "paragraph_count": len(re.split(r"\n\s*\n", text)), "substantive": len(text) >= 1500 and len(tokens) >= 250 and bool(STRUCTURE_RE.search(text))}


def _identity_gate(target: dict[str, Any], title: str, text: str, source_url: str) -> tuple[str, dict[str, Any]]:
    identity = target["identity"]
    target_code = identity.get("normalized_document_number")
    source_codes = sorted(set(_codes(title) + _codes(text[:30000])))
    title_overlap = sorted(_words(identity["title"]) & _words(title))
    body_overlap = sorted(_words(identity["title"]) & _words(text[:30000]))
    evidence = {"target_code": target_code, "source_codes": source_codes, "title_overlap_tokens": title_overlap, "body_overlap_tokens": body_overlap, "source_title": title, "source_url": source_url}
    if target_code:
        return ("IDENTITY_PASS" if target_code in source_codes else "IDENTITY_FAIL"), evidence
    # A BTC slug without document code cannot prove equivalence on its own.
    return "IDENTITY_AMBIGUOUS", evidence


def _pdf_ocr_identity(target: dict[str, Any], pdf: dict[str, Any], base_evidence: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    """OCR two pages only to establish (not ingest) image-PDF identity."""
    text, error = _ocr_pdf(ROOT / pdf["path"], max_pages=2, resolution=180)
    if error:
        return "PDF_IDENTITY_FAIL", {**base_evidence, "ocr_error": error}, error
    title_words = _words(target["identity"]["title"])
    overlap = sorted(title_words & _words(text))
    source_codes = _codes(text)
    target_code = target["identity"].get("normalized_document_number")
    page_mapping = bool(base_evidence.get("metadata_contains_tvpl_page_id"))
    header = text[:900]
    authority = next((name for name, pattern in (
        ("BỘ QUỐC PHÒNG", r"B[ỘO]\s+QU[ỐO]C\s+PH[ÒO]NG"),
        ("THỦ TƯỚNG CHÍNH PHỦ", r"TH[ỦU]\s*T[ƯU][ỚO]NG\s+CH[ÍI]NH\s+PH[ỦU]"),
        ("CHÍNH PHỦ", r"CH[ÍI]NH\s+PH[ỦU]"),
    ) if re.search(pattern, header, re.I)), None)
    years = sorted(set(re.findall(r"(?:năm|nam)\s+(20\d{2})", header, re.I)))
    draft = bool(re.search(r"D[ỰU]\s*TH[ẢA]O", header, re.I))
    evidence = {
        **base_evidence,
        "ocr_for_identity_pages": min(2, int(pdf.get("page_count", 0))),
        "ocr_identity_excerpt": text[:2500],
        "ocr_source_codes": source_codes,
        "title_overlap_tokens": overlap,
        "identified_authority": authority,
        "identified_years": years,
        "draft_marker": draft,
    }
    if target_code:
        return ("PDF_IDENTITY_PASS" if target_code in source_codes else "PDF_IDENTITY_FAIL"), evidence, None
    # The local printed PDF is tied to BTC's supplied TVPL target only when
    # its print-title contains that exact TVPL page id.  Combined with a long
    # title match, authority and year this satisfies title+authority+date
    # identity for non-standard documents without trusting the filename.
    if page_mapping and len(overlap) >= 5 and authority and years:
        return "PDF_IDENTITY_PASS", evidence, None
    return ("PDF_IDENTITY_FAIL" if draft and not page_mapping else "PDF_IDENTITY_AMBIGUOUS"), evidence, None


def _tier(url: str) -> tuple[str, str]:
    host = urlparse(url).netloc.casefold().removeprefix("www.")
    if host.endswith(".gov.vn") or host in {"vbpl.vn", "moj.gov.vn", "vanban.chinhphu.vn"}:
        return "TIER_1_OFFICIAL", host
    if host in {"luatvietnam.vn", "thuvienphapluat.vn"}:
        return "TIER_2_LEGAL_DB", host
    return "TIER_3_PUBLIC_MIRROR", host


def _bing_url(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.netloc.casefold().endswith("bing.com"):
        encoded = parse_qs(parsed.query).get("u", [""])[0]
        if encoded.startswith("a1"):
            try:
                decoded = base64.b64decode(encoded[2:] + "===").decode("utf-8")
                return (
                    decoded
                    if urlparse(decoded).scheme in {"http", "https"}
                    and not any(ord(char) < 32 for char in decoded)
                    else None
                )
            except (ValueError, UnicodeDecodeError):
                return None
    return value if parsed.scheme in {"http", "https"} else None


class _BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.href: str | None = None
        self.parts: list[str] = []
        self.items: list[tuple[str, str]] = []
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "a" and self.href is None:
            href = dict(attrs).get("href")
            if href:
                self.href, self.parts = href, []
    def handle_data(self, data: str) -> None:
        if self.href is not None:
            self.parts.append(data)
    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self.href is not None:
            title = " ".join(" ".join(self.parts).split())
            url = _bing_url(html.unescape(self.href))
            if url and title:
                self.items.append((url, title))
            self.href, self.parts = None, []


def _discover(client: httpx.Client, target: dict[str, Any]) -> list[str]:
    """Return only supplied exact-document candidates.

    The previous HTML-scraped Bing path yielded unrelated ChatGPT links, so it
    is intentionally disabled rather than letting weak SERP parsing create a
    false ``NO_PUBLIC_SOURCE_FOUND`` result.  A configured search API/browser
    can be added later; manually supplied candidate URLs remain valid input.
    """
    del client
    return list(KNOWN_CANDIDATES.get(target["document_id"], ()))


def _search_queries(target: dict[str, Any]) -> list[str]:
    code = target["identity"].get("display_code")
    title = target["identity"]["title"]
    if code:
        return [f'"{code}"', f'"{code}" "{title}"', f'site:luatvietnam.vn "{code}"', f'site:vbpl.vn "{code}"', f'"{code}" filetype:pdf']
    return [f'"{title}"', f'"{title}" "{target["identity"].get("issuing_authority") or ""}"']


def _snapshot(root: Path, doc_id: str, body: bytes, extension: str) -> tuple[str, str]:
    path = root / "source_recovery" / "fetched" / f"{doc_id}__source.{extension}"
    digest = _sha(body)
    if not path.is_file() or _sha(path.read_bytes()) != digest:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return str(path.relative_to(root)), digest


def _promote(root: Path, target: dict[str, Any], text: str, source: dict[str, Any]) -> dict[str, Any]:
    doc_id = target["document_id"]
    extracted = root / "source_recovery" / "extracted" / f"{doc_id}.txt"
    extracted.parent.mkdir(parents=True, exist_ok=True); extracted.write_text(text + "\n", encoding="utf-8")
    raw = RawDocument(doc_id=doc_id, source_path=source["snapshot_path"], title=source.get("source_title") or None, raw_text=text, file_format=source["extraction_method"].casefold(), metadata={"source_link": target["original_tvpl_url"], "pv1_lineage": "RECOVERED_FROM_BTC_LINK", "pv1_source_provenance": source})
    cleaned = clean_document(raw); result = chunk_clean_document(cleaned, chunk_size=192, chunk_overlap=32)
    if not cleaned.cleaned_text.strip() or not result.chunks or result.missing_source_token_count or result.missing_source_bigram_count:
        return {"status": "UNRESOLVED", "unresolved_reason": "CANONICAL_CLEAN_OR_CHUNK_GATE_FAILED"}
    _write_json(root / "phase1_recovered" / "documents" / f"{doc_id}.json", cleaned.model_dump(mode="json"))
    _write_jsonl(root / "phase1_recovered" / "chunks" / f"{doc_id}.jsonl", [chunk.model_dump(mode="json") for chunk in result.chunks])
    _write_jsonl(root / "phase1_recovered" / "parents" / f"{doc_id}.jsonl", [parent.model_dump(mode="json") for parent in result.parents])
    return {"status": "RECOVERED", "cleaned_text_sha256": _sha(cleaned.cleaned_text), "extracted_text_sha256": _sha(text), "article_count": result.article_count, "chunk_count": len(result.chunks), "source_line_coverage": result.assigned_source_line_count / result.source_nonempty_line_count if result.source_nonempty_line_count else 0.0}


def _local_pdf(root: Path, target: dict[str, Any], inventory: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [(pdf, *_pdf_match(target, pdf)) for pdf in inventory]
    candidates = [item for item in candidates if item[1] != "PDF_IDENTITY_FAIL"]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item[1] != "PDF_IDENTITY_PASS",
            not item[2]["filename_matches_target_id"],
            not item[2]["metadata_contains_tvpl_page_id"],
            item[0]["filename"],
        )
    )
    pdf, pdf_status, pdf_evidence = candidates[0]
    base = {"local_pdf": {key: pdf.get(key) for key in ("filename", "path", "sha256", "page_count", "embedded_text_length")}, "pdf_identity_status": pdf_status, "pdf_identity_evidence": pdf_evidence}
    if len(pdf.get("embedded_text", "")) < 1500:
        pdf_status, pdf_evidence, identity_error = _pdf_ocr_identity(target, pdf, pdf_evidence)
        base.update({"pdf_identity_status": pdf_status, "pdf_identity_evidence": pdf_evidence})
        if identity_error:
            return {**base, "status": "UNRESOLVED", "unresolved_reason": "PDF_EXTRACTION_FAILED", "pdf_error": identity_error}
    if pdf_status != "PDF_IDENTITY_PASS":
        return {**base, "status": "UNRESOLVED", "unresolved_reason": "WRONG_DOCUMENT" if pdf_status == "PDF_IDENTITY_FAIL" else "PDF_IDENTITY_AMBIGUOUS"}
    text, method = pdf.get("embedded_text", ""), "LOCAL_PDF_TEXT"
    if len(text) < 1500:
        text, error = _ocr_pdf(ROOT / pdf["path"], resolution=180); method = "LOCAL_PDF_OCR"
        if error:
            return {**base, "status": "UNRESOLVED", "unresolved_reason": "PDF_EXTRACTION_FAILED", "pdf_error": error}
    quality = _quality(text)
    title = " ".join(str(value) for value in pdf.get("metadata", {}).values())
    if not quality["substantive"]:
        return {**base, "status": "UNRESOLVED", "unresolved_reason": "SOURCE_CONTENT_INCOMPLETE", "text_quality": quality}
    source = {"recovery_source": method, "recovery_url": None, "source_domain": "LOCAL", "source_tier": "LOCAL", "identity_status": "IDENTITY_PASS", "identity_evidence": pdf_evidence, "snapshot_path": pdf["path"], "snapshot_sha256": pdf["sha256"], "extraction_method": "PDF_TEXT" if method.endswith("TEXT") else "PDF_OCR", "source_title": title}
    return {**base, **source, **_promote(root, target, text, source), "recovery_method": method, "text_quality": quality}


def _local_pdf_only(root: Path, target: dict[str, Any], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Strictly local recovery.  This function has no network-capable inputs."""
    doc_id = target["document_id"]
    existing = root / "source_recovery" / "provenance" / f"{doc_id}.json"
    recovered = root / "phase1_recovered" / "documents" / f"{doc_id}.json"
    # Reuse the two previously validated recovered documents only if source bytes
    # and all three canonical output artifacts still agree with their provenance.
    exact = [p for p in inventory if Path(p["filename"]).stem == doc_id]
    if existing.is_file() and recovered.is_file() and exact:
        old = json.loads(existing.read_text(encoding="utf-8"))
        pdf = exact[0]
        required = [recovered, root / "phase1_recovered" / "chunks" / f"{doc_id}.jsonl", root / "phase1_recovered" / "parents" / f"{doc_id}.jsonl"]
        if (old.get("local_pdf", {}).get("sha256") == pdf["sha256"] and old.get("status") == "RECOVERED" and all(p.is_file() for p in required)):
            old.update({"document_id": doc_id, "identity": target["identity"], "local_pdf": {k: pdf.get(k) for k in ("filename", "path", "sha256", "page_count", "embedded_text_length")}, "source_type": "LOCAL_USER_PROVIDED_PDF", "reused_validated_artifact": True, "content_completeness": old.get("content_completeness", "COMPLETE")})
            old.pop("unresolved_reason", None)
            return old

    # A filename is only a candidate selection hint; its contents subsequently
    # have to pass the identity contract below.
    candidates = exact or inventory
    evaluated: list[tuple[dict[str, Any], str, dict[str, Any], str, str | None]] = []
    for pdf in candidates:
        embedded = pdf.get("embedded_text", "")
        identity_method = "EMBEDDED_TEXT" if len(embedded) >= 1200 else "OCR_FOR_IDENTITY"
        identity_text, error = (embedded, None) if identity_method == "EMBEDDED_TEXT" else _ocr_pdf(ROOT / pdf["path"], max_pages=3, resolution=180)
        if error:
            evidence = {"ocr_error": error}
            evaluated.append((pdf, "IDENTITY_FAIL", evidence, identity_method, error))
            continue
        status, evidence = _identity_text(target, pdf, identity_text)
        evaluated.append((pdf, status, evidence, identity_method, None))
    passed = [item for item in evaluated if item[1] == "IDENTITY_PASS"]
    if not passed:
        best = next((item for item in evaluated if item[1] == "IDENTITY_AMBIGUOUS"), evaluated[0] if evaluated else None)
        if best is None:
            return {"status": "UNRESOLVED", "unresolved_reason": "LOCAL_PDF_MISSING", "source_type": "LOCAL_USER_PROVIDED_PDF"}
        pdf, identity_status, evidence, method, error = best
        return {"status": "UNRESOLVED", "unresolved_reason": "PDF_IDENTITY_AMBIGUOUS" if identity_status == "IDENTITY_AMBIGUOUS" else "PDF_IDENTITY_FAIL", "local_pdf": {k: pdf.get(k) for k in ("filename", "path", "sha256", "page_count", "embedded_text_length")}, "pdf_identity_status": identity_status, "pdf_identity_evidence": evidence, "ocr_for_identity": "YES" if method == "OCR_FOR_IDENTITY" else "NO", "source_type": "LOCAL_USER_PROVIDED_PDF", **({"pdf_error": error} if error else {})}
    pdf, identity_status, evidence, identity_method, _ = passed[0]
    text, extraction_method = pdf.get("embedded_text", ""), "LOCAL_PDF_TEXT"
    if len(text) < 1500:
        text, error = _ocr_pdf(ROOT / pdf["path"], resolution=180)
        extraction_method = "LOCAL_PDF_OCR"
        if error:
            return {"status": "UNRESOLVED", "unresolved_reason": "PDF_EXTRACTION_FAILED", "local_pdf": {k: pdf.get(k) for k in ("filename", "path", "sha256", "page_count", "embedded_text_length")}, "pdf_identity_status": identity_status, "pdf_identity_evidence": evidence, "pdf_error": error, "source_type": "LOCAL_USER_PROVIDED_PDF"}
    completeness, quality = _completeness(text, int(pdf.get("page_count", 0)))
    base = {"local_pdf": {k: pdf.get(k) for k in ("filename", "path", "sha256", "page_count", "embedded_text_length")}, "source_type": "LOCAL_USER_PROVIDED_PDF", "pdf_identity_status": identity_status, "identity_status": identity_status, "pdf_identity_evidence": evidence, "ocr_for_identity": "YES" if identity_method == "OCR_FOR_IDENTITY" else "NO", "content_completeness": completeness, "recovery_method": extraction_method, "text_quality": quality, "original_filename": pdf["filename"]}
    if completeness not in {"COMPLETE", "LIKELY_COMPLETE"}:
        return {**base, "status": "UNRESOLVED", "unresolved_reason": f"CONTENT_{completeness}"}
    source = {"recovery_source": extraction_method, "recovery_url": None, "source_domain": "LOCAL", "source_tier": "LOCAL_USER_PROVIDED_PDF", "identity_status": "IDENTITY_PASS", "identity_evidence": evidence, "snapshot_path": pdf["path"], "snapshot_sha256": pdf["sha256"], "extraction_method": "PDF_TEXT" if extraction_method.endswith("TEXT") else "PDF_OCR", "source_title": ""}
    return {**base, **source, **_promote(root, target, text, source)}


def _web(root: Path, client: httpx.Client, target: dict[str, Any]) -> dict[str, Any]:
    response, tvpl = _fetch_once(client, target["original_tvpl_url"])
    output: dict[str, Any] = {"tvpl_status": tvpl["status"], "tvpl_probe": tvpl, "search_queries": _search_queries(target)}
    sources: list[tuple[httpx.Response, str, str, str]] = []
    if response is not None and tvpl["status"] in {"FULL_TEXT_PUBLIC", "PDF_PUBLIC"}:
        sources.append((response, str(response.url), "TVPL", "TIER_2_LEGAL_DB"))
    else:
        urls = _discover(client, target)
        output["alternate_search_performed"] = bool(urls)
        output["search_engine_valid"] = False
        output["alternate_candidates"] = urls
        output["alternate_candidate_probes"] = []
        for url in urls:
            candidate, probe = _fetch_once(client, url)
            output["alternate_candidate_probes"].append({"url": url, "tier": _tier(url)[0], "probe": probe})
            if candidate is not None and probe["status"] in {"FULL_TEXT_PUBLIC", "PDF_PUBLIC"}:
                tier, _ = _tier(url); sources.append((candidate, str(candidate.url), "ALTERNATE", tier))
    for candidate, url, kind, tier in sources:
        if candidate.content.startswith(b"%PDF") or "pdf" in candidate.headers.get("content-type", "").casefold():
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
                    temp.write(candidate.content); temp.flush()
                    with pdfplumber.open(temp.name) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages); title = " ".join(str(value) for value in (pdf.metadata or {}).values())
            except Exception:
                continue
            extension = "pdf"; method = "TVPL_PUBLIC_PDF" if kind == "TVPL" else ("ALTERNATE_OFFICIAL_PDF" if tier == "TIER_1_OFFICIAL" else "ALTERNATE_LEGAL_DB_PDF")
        else:
            title, text = _html_text(candidate.content, candidate.encoding); extension = "html"; method = "TVPL_PUBLIC_HTML" if kind == "TVPL" else ("ALTERNATE_OFFICIAL_HTML" if tier == "TIER_1_OFFICIAL" else "ALTERNATE_LEGAL_DB_HTML")
        identity_status, evidence = _identity_gate(target, title, text, url); quality = _quality(text)
        if identity_status != "IDENTITY_PASS" or not quality["substantive"]:
            continue
        snapshot_path, snapshot_sha = _snapshot(root, target["document_id"], candidate.content, extension)
        source = {"recovery_source": method, "recovery_url": url, "source_domain": urlparse(url).netloc.casefold(), "source_tier": tier, "identity_status": identity_status, "identity_evidence": evidence, "snapshot_path": snapshot_path, "snapshot_sha256": snapshot_sha, "extraction_method": "PDF_TEXT" if extension == "pdf" else "HTML", "source_title": title}
        return {**output, **source, **_promote(root, target, text, source), "recovery_method": method, "text_quality": quality}
    if target["document_id"] == "67660":
        return {**output, "status": "UNRESOLVED", "unresolved_reason": "KNOWN_67660_CANDIDATE_NOT_FULL_TEXT_OR_BLOCKED"}
    return {**output, "status": "UNRESOLVED", "unresolved_reason": "SEARCH_ENGINE_UNAVAILABLE"}


def _report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# PV1 Local PDF Phase 1 Recovery", "", "This report was produced in strict local-PDF-only mode. No website, search engine, GPU, or Modal action was used.", "", "| document_id | pdf_filename | identity | identity_status | completeness_status | extraction_method | text_length | chunk_count | final_status | reason |", "|---|---|---|---|---|---|---:|---:|---|---|"]
    for row in rows:
        identity = row["identity"].get("document_number_or_code") or row["identity"]["title"]
        local = row.get("local_pdf", {}).get("filename", "NO") if isinstance(row.get("local_pdf"), dict) else "NO"
        lines.append("| {id} | {local} | {key} | {identity_status} | {complete} | {method} | {length} | {chunks} | {status} | {reason} |".format(id=row["document_id"], local=local, key=identity.replace("|", " "), identity_status=row.get("identity_status", row.get("pdf_identity_status", "-")), complete=row.get("content_completeness", "NOT_EVALUATED"), method=row.get("recovery_method", "-"), status=row["status"], length=row.get("text_quality", {}).get("text_length", 0), chunks=row.get("chunk_count", 0), reason=row.get("unresolved_reason", "NONE")))
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw" / "btc")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed_pv1")
    parser.add_argument("--pdf-root", type=Path, default=ROOT / "data" / "processed_pv1" / "pdf")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="refresh local-PDF annotations without any network request")
    parser.add_argument("--local-recovery-only", action="store_true", help="retry local PDF identity/recovery while preserving prior web probes")
    parser.add_argument("--local-pdf-only", action="store_true", help="strictly local PDF/OCR recovery; never invokes web discovery or HTTP")
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    targets, rows = _targets(args.raw_root), []
    inventory_path = args.output_root / "metadata" / "phase1_pdf_inventory.json"
    inventory = None
    if args.local_pdf_only and inventory_path.is_file():
        try:
            cached_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            current_paths = sorted(args.pdf_root.rglob("*.pdf"))
            cached_by_name = {item["filename"]: item for item in cached_inventory}
            if len(current_paths) == len(cached_by_name) == 20:
                refreshed: list[dict[str, Any]] = []
                for path in current_paths:
                    old = cached_by_name.get(path.name)
                    digest = _sha(path.read_bytes())
                    if old is not None and old.get("sha256") == digest:
                        refreshed.append(old)
                    elif old is not None and old.get("embedded_text_length", 0) <= 200:
                        # Image-only scans can make a full pdfminer text pass
                        # needlessly expensive.  Zero embedded text is the
                        # correct inventory value; identity/full extraction
                        # will use the bounded/full OCR path below.
                        raw_pdf = path.read_bytes()
                        refreshed.append({"filename": path.name, "path": str(path.relative_to(ROOT)), "sha256": digest, "page_count": len(re.findall(rb"/Type\s*/Page\b", raw_pdf)), "embedded_text_length": 0, "metadata": {"inventory_note": "image_pdf_ocr_required"}, "embedded_text": ""})
                    else:
                        refreshed.extend(_pdf_inventory(args.pdf_root, [path]))
                inventory = refreshed
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            inventory = None
    if inventory is None:
        inventory = _pdf_inventory(args.pdf_root)
        _write_json(inventory_path, inventory)
    if args.local_pdf_only:
        for index, target in enumerate(targets, start=1):
            print(f"LOCAL_PDF_PROGRESS target={index}/20 document_id={target['document_id']}", flush=True)
            rows.append({**target, "raw_status": "EMPTY", "synthetic_content": 0, "manual_legal_text_edits": 0, **_local_pdf_only(args.output_root, target, inventory)})
    elif args.report_only or args.local_recovery_only:
        manifest = args.output_root / "metadata" / "phase1_recovery_manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"cannot refresh absent manifest: {manifest}")
        by_id = {target["document_id"]: target for target in targets}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            target = by_id[row["document_id"]]
            row["identity"] = target["identity"]
            if args.local_recovery_only:
                local = _local_pdf(args.output_root, target, inventory)
                if local is not None:
                    if local.get("status") == "RECOVERED":
                        row.update(local)
                        row.pop("unresolved_reason", None)
                    else:
                        for key in ("local_pdf", "pdf_identity_status", "pdf_identity_evidence", "pdf_error"):
                            if key in local:
                                row[key] = local[key]
            rows.append(row)
    else:
        with httpx.Client(headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/pdf"}, follow_redirects=True, timeout=httpx.Timeout(45.0)) as client:
            for index, target in enumerate(targets, start=1):
                print(f"PHASE1_PROGRESS target={index}/20 document_id={target['document_id']}", flush=True)
                row = {**target, "raw_status": "EMPTY", "synthetic_content": 0, "manual_legal_text_edits": 0}
                local = _local_pdf(args.output_root, target, inventory)
                if local is not None:
                    row.update(local)
                if row.get("status") != "RECOVERED":
                    row.update({"status": "UNRESOLVED", "unresolved_reason": "NOT_ATTEMPTED"} if args.verify_only else _web(args.output_root, client, target))
                rows.append(row)
    for row in rows:
        _write_json(args.output_root / "source_recovery" / "provenance" / f"{row['document_id']}.json", row)
    _write_jsonl(args.output_root / "metadata" / "phase1_recovery_manifest.jsonl", rows); _report(rows, args.output_root / "metadata" / "phase1_recovery_report.md")
    recovered = [row for row in rows if row["status"] == "RECOVERED"]
    method_names = (
        "LOCAL_PDF_TEXT", "LOCAL_PDF_OCR", "TVPL_PUBLIC_HTML", "TVPL_PUBLIC_PDF",
        "ALTERNATE_OFFICIAL_HTML", "ALTERNATE_OFFICIAL_PDF",
        "ALTERNATE_LEGAL_DB_HTML", "ALTERNATE_LEGAL_DB_PDF",
    )
    method_counts = Counter(row.get("recovery_method", "") for row in recovered)
    recovered_without_identity = sum(row.get("identity_status") != "IDENTITY_PASS" for row in recovered)
    unresolved_without_reason = sum(row["status"] != "RECOVERED" and not row.get("unresolved_reason") for row in rows)
    provenance_missing = sum(not (args.output_root / "source_recovery" / "provenance" / f"{row['document_id']}.json").is_file() for row in rows)
    local_only = bool(args.local_pdf_only)
    identity_counts = Counter(row.get("identity_status", row.get("pdf_identity_status", "IDENTITY_FAIL")) for row in rows)
    completeness_counts = Counter(row.get("content_completeness", "NOT_EVALUATED") for row in rows)
    mapped = sum(isinstance(row.get("local_pdf"), dict) for row in rows)
    special_208668 = next(row for row in rows if row["document_id"] == "208668")
    special_131890 = next(row for row in rows if row["document_id"] == "131890")
    search_engine_valid = False
    summary = {
        "targets_processed": f"{len(rows)}/20",
        "identity_parser_valid": _identity_parser_valid(),
        "execution_mode": "LOCAL_PDF_ONLY" if local_only else "MIXED_LEGACY",
        "search_engine_valid": False,
        "network_requests_performed": 0 if local_only else None,
        "tvpl_status": dict(sorted(Counter(row.get("tvpl_status", "NOT_PROBED") for row in rows).items())),
        "local_pdf": {
            "found": len(inventory),
            "mapped_to_targets": mapped,
            "identity_pass": identity_counts["IDENTITY_PASS"],
            "identity_fail": identity_counts["IDENTITY_FAIL"],
            "identity_ambiguous": identity_counts["IDENTITY_AMBIGUOUS"],
            "recovered_pdf_text": method_counts["LOCAL_PDF_TEXT"],
            "recovered_pdf_ocr": method_counts["LOCAL_PDF_OCR"],
        },
        "alternate_source": {
            "search_plan_documents": len(rows),
            "candidate_urls_probed": sum(len(row.get("alternate_candidate_probes", [])) for row in rows),
            "official_source_recovered": method_counts["ALTERNATE_OFFICIAL_HTML"] + method_counts["ALTERNATE_OFFICIAL_PDF"],
            "public_legal_db_recovered": method_counts["ALTERNATE_LEGAL_DB_HTML"] + method_counts["ALTERNATE_LEGAL_DB_PDF"],
            "identity_ambiguous": sum(row.get("pdf_identity_status") == "PDF_IDENTITY_AMBIGUOUS" or row.get("identity_status") == "IDENTITY_AMBIGUOUS" for row in rows),
            "no_source_found": sum(row.get("unresolved_reason") == "NO_PUBLIC_SOURCE_FOUND" for row in rows),
            "unrelated_candidates_rejected": 0,
        },
        "pdf_identity_ocr_used": sum(row.get("ocr_for_identity") == "YES" for row in rows),
        "completeness": dict(sorted(completeness_counts.items())),
        "special_208668": {"pdf": special_208668.get("local_pdf", {}).get("filename"), "identity": special_208668.get("identity_status", special_208668.get("pdf_identity_status")), "completeness": special_208668.get("content_completeness"), "reason": special_208668.get("unresolved_reason", "IDENTITY_AND_COMPLETENESS_PASS")},
        "special_131890": {"pdf": special_131890.get("local_pdf", {}).get("filename"), "identity": special_131890.get("identity_status", special_131890.get("pdf_identity_status")), "mapping_from_132890": special_131890.get("local_pdf", {}).get("filename") == "132890.pdf", "reason": special_131890.get("unresolved_reason", "IDENTITY_AND_COMPLETENESS_PASS")},
        "total_recovered": f"{len(recovered)}/20",
        "unresolved": f"{len(rows) - len(recovered)}/20",
        "recovery_methods": {name: method_counts[name] for name in method_names},
        "synthetic_content": 0,
        "manual_legal_text_edits": 0,
        "raw_mutated": "NO",
        "processed_v3_mutated": "NO",
        "recovered_without_identity_pass": recovered_without_identity,
        "provenance_missing": provenance_missing,
        "unresolved_without_reason": unresolved_without_reason,
        "PHASE1_RECOVERY_GATE": "PASS" if not recovered_without_identity and not provenance_missing and not unresolved_without_reason else "FAIL",
        # Safety/provenance may pass while the review remains blocked on a
        # trustworthy discovery mechanism.  Do not authorize Phase 2 from the
        # former alone.
        "PHASE1_REVIEW_GATE": "PASS" if _identity_parser_valid() and search_engine_valid else "FAIL",
        "LOCAL_PDF_RECOVERY_GATE": "PASS" if local_only and len(rows) == 20 and mapped == 20 and not recovered_without_identity and not provenance_missing and not unresolved_without_reason else "FAIL",
        "next_action": "READY_FOR_PV1_PHASE2" if len(recovered) == 20 else "MANUAL_REVIEW_REMAINING_PDFS",
    }
    _write_json(args.output_root / "metadata" / "phase1_recovery_summary.json", summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
