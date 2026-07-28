"""Reader for plain-text documents with encoding detection."""

from pathlib import Path
from typing import Any, Dict, Tuple

from udsc2026.ingestion.readers._helpers import first_nonempty_line, make_doc_id
from udsc2026.ingestion.readers.models import RawDocument


def _decode_text(raw_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """Decode text bytes while retaining all line breaks in the source."""
    try:
        from charset_normalizer import from_bytes
    except ImportError:  # pragma: no cover - exercised only in incomplete installs
        try:
            return raw_bytes.decode("utf-8-sig"), {"encoding": "utf-8-sig"}
        except UnicodeDecodeError as error:
            raise ImportError(
                "Reading non-UTF-8 TXT files requires charset-normalizer. "
                "Install it with `pip install charset-normalizer`."
            ) from error

    match = from_bytes(raw_bytes).best()
    if match is None or match.encoding is None:
        raise UnicodeDecodeError(
            "unknown", raw_bytes, 0, len(raw_bytes), "unknown encoding"
        )

    try:
        text = raw_bytes.decode(match.encoding)
    except (LookupError, UnicodeDecodeError):
        # charset-normalizer has already decoded the bytes successfully.
        text = str(match)

    metadata: Dict[str, Any] = {"encoding": match.encoding}
    coherence = getattr(match, "percent_coherence", None)
    if coherence is not None:
        metadata["encoding_coherence"] = coherence
    return text, metadata


def read_txt(file_path: str) -> RawDocument:
    """Read a TXT source without normalizing, cleaning, or parsing its content."""
    path = Path(file_path)
    raw_text, metadata = _decode_text(path.read_bytes())
    metadata["file_name"] = path.name
    metadata["file_size_bytes"] = path.stat().st_size

    return RawDocument(
        doc_id=make_doc_id(str(path)),
        source_path=str(path),
        title=first_nonempty_line(raw_text),
        raw_text=raw_text,
        file_format="txt",
        metadata=metadata,
    )
