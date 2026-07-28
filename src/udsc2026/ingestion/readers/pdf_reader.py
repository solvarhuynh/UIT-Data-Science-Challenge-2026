"""Reader for text-based PDF documents."""

from pathlib import Path
from typing import Any, Dict, List

from udsc2026.ingestion.readers._helpers import first_nonempty_line, make_doc_id
from udsc2026.ingestion.readers.models import RawDocument


def _require_pdfplumber():
    try:
        import pdfplumber
    except ImportError as error:  # pragma: no cover - incomplete install only
        raise ImportError(
            "Reading PDF files requires pdfplumber. "
            "Install it with `pip install pdfplumber`."
        ) from error
    return pdfplumber


def read_pdf(file_path: str) -> RawDocument:
    """Extract text from each PDF page; scanned pages are flagged for OCR."""
    pdfplumber = _require_pdfplumber()
    path = Path(file_path)

    page_texts: List[str] = []
    pages_without_text: List[int] = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Cropping prevents repeating page numbers/header text from entering
            # the legal text. layout=True retains horizontal spacing for columns.
            content_box = (0, page.height * 0.1, page.width, page.height * 0.9)
            page_text = page.crop(content_box).extract_text(layout=True) or ""
            if not page_text.strip():
                pages_without_text.append(page_number)
            page_texts.append(page_text)

        pdf_metadata = dict(pdf.metadata or {})

    # The form-feed delimiter is retained for the cleaner to identify repeated
    # headers and footers by page, then removed before structure parsing.
    raw_text = "\n\f\n".join(page_texts)
    metadata: Dict[str, Any] = {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "page_count": len(page_texts),
        "pages_without_text": pages_without_text,
        "requires_ocr": bool(pages_without_text),
        "ocr_status": (
            "required_not_performed" if pages_without_text else "not_required"
        ),
        "content_crop": {"top_percent": 10, "bottom_percent": 10},
        "text_extraction_layout": True,
    }
    if pdf_metadata:
        metadata["pdf_metadata"] = pdf_metadata

    title_value = pdf_metadata.get("Title") or pdf_metadata.get("title")
    title = str(title_value).strip() if title_value else first_nonempty_line(raw_text)
    return RawDocument(
        doc_id=make_doc_id(str(path)),
        source_path=str(path),
        title=title,
        raw_text=raw_text,
        file_format="pdf",
        metadata=metadata,
    )
