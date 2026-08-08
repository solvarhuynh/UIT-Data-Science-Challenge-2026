"""BTC corpus discovery, extraction, and manifest utilities."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from udsc2026.ingestion.readers import extract_raw_document
from udsc2026.ingestion.readers.models import RawDocument

_CONTEXT_STEM = re.compile(r"^context_(.+)$", re.IGNORECASE)
_BTC_QA_FILENAMES = ("train.json", "public-official.json", "private-official.json")


class BTCQAFixtureManifest(BaseModel):
    """Audit summary for one LegalQA fixture file."""

    file_name: str
    record_count: int
    schema_hash: str
    missing_question_answer_count: int = 0


class BTCExtractionResult(BaseModel):
    """Accepted BTC context documents plus audit data for rejected files."""

    source_root: str
    discovered_context_files: List[str] = Field(default_factory=list)
    raw_documents: List[RawDocument] = Field(default_factory=list)
    orphan_context_ids: List[str] = Field(default_factory=list)
    orphan_source_paths: List[str] = Field(default_factory=list)
    error_records: List[Dict[str, str]] = Field(default_factory=list)


class BTCOrphanReport(BaseModel):
    """Rejected context files collected during BTC extraction."""

    schema_version: str = "btc-context-v1"
    source_root: str
    orphan_context_count: int
    orphan_context_ids: List[str] = Field(default_factory=list)
    orphan_source_paths: List[str] = Field(default_factory=list)
    error_count: int = 0
    errors: List[Dict[str, str]] = Field(default_factory=list)


class BTCManifest(BaseModel):
    """Hash-based corpus manifest for BTC context and QA fixture files."""

    schema_version: str = "btc-context-v1"
    source_root: str
    corpus_hash: str
    context_file_count: int
    accepted_context_count: int
    orphan_context_count: int
    document_ids: List[str] = Field(default_factory=list)
    context_files: List[str] = Field(default_factory=list)
    qa_fixtures: List[BTCQAFixtureManifest] = Field(default_factory=list)


def discover_btc_context_files(raw_directory: str | Path) -> List[Path]:
    """Find every BTC `context_*.json` file below the raw data root."""
    root = Path(raw_directory)
    return sorted(path for path in root.rglob("context_*.json") if path.is_file())


def discover_btc_qa_files(raw_directory: str | Path) -> List[Path]:
    """Find BTC LegalQA fixtures that should be validated but not chunked."""
    root = Path(raw_directory)
    discovered: List[Path] = []
    for file_name in _BTC_QA_FILENAMES:
        discovered.extend(path for path in root.rglob(file_name) if path.is_file())
    return sorted(discovered)


def extract_btc_context_documents(
    raw_directory: str | Path,
    errors_path: Optional[str | Path] = None,
) -> BTCExtractionResult:
    """Extract BTC LegalIR context files while keeping orphan diagnostics."""
    root = Path(raw_directory)
    context_files = discover_btc_context_files(root)
    documents: List[RawDocument] = []
    errors: List[Dict[str, str]] = []
    orphan_context_ids: List[str] = []
    orphan_source_paths: List[str] = []
    seen_doc_ids = set()

    for file_path in context_files:
        context_id = _context_id_from_path(file_path)
        try:
            extracted = extract_raw_document(str(file_path))
            candidates = (
                (extracted,) if isinstance(extracted, RawDocument) else extracted
            )
            for document in candidates:
                if document.doc_id in seen_doc_ids:
                    raise ValueError("Trùng context_id: {0}".format(document.doc_id))
                seen_doc_ids.add(document.doc_id)
                documents.append(document)
        except Exception as error:  # Keep extracting independent context files.
            orphan_context_ids.append(context_id)
            orphan_source_paths.append(str(file_path))
            errors.append(_error_record(file_path, error, context_id))

    _write_errors(errors, errors_path)
    return BTCExtractionResult(
        source_root=str(root),
        discovered_context_files=[str(path) for path in context_files],
        raw_documents=documents,
        orphan_context_ids=orphan_context_ids,
        orphan_source_paths=orphan_source_paths,
        error_records=errors,
    )


def build_btc_manifest(
    result: BTCExtractionResult,
    raw_directory: str | Path,
) -> BTCManifest:
    """Build a deterministic manifest for the accepted BTC corpus."""
    root = Path(raw_directory)
    qa_files = discover_btc_qa_files(root)
    qa_fixtures = [_build_qa_fixture_manifest(file_path) for file_path in qa_files]
    corpus_hash = _corpus_hash(result.raw_documents, qa_files)
    return BTCManifest(
        source_root=str(root),
        corpus_hash=corpus_hash,
        context_file_count=len(result.discovered_context_files),
        accepted_context_count=len(result.raw_documents),
        orphan_context_count=len(result.orphan_context_ids),
        document_ids=[document.doc_id for document in result.raw_documents],
        context_files=list(result.discovered_context_files),
        qa_fixtures=qa_fixtures,
    )


def build_btc_orphan_report(result: BTCExtractionResult) -> BTCOrphanReport:
    """Build the orphan report written alongside the BTC manifest."""
    return BTCOrphanReport(
        source_root=result.source_root,
        orphan_context_count=len(result.orphan_context_ids),
        orphan_context_ids=list(result.orphan_context_ids),
        orphan_source_paths=list(result.orphan_source_paths),
        error_count=len(result.error_records),
        errors=list(result.error_records),
    )


def _build_qa_fixture_manifest(file_path: Path) -> BTCQAFixtureManifest:
    payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("BTC QA fixture must be a JSON object: {0}".format(file_path))
    missing = 0
    for record in payload.values():
        if not isinstance(record, dict):
            missing += 1
            continue
        if not record.get("question") or record.get("answer") in (None, "", []):
            missing += 1
    schema_hash = _stable_json_hash(payload)
    return BTCQAFixtureManifest(
        file_name=file_path.name,
        record_count=len(payload),
        schema_hash=schema_hash,
        missing_question_answer_count=missing,
    )


def _corpus_hash(documents: Sequence[RawDocument], qa_files: Sequence[Path]) -> str:
    hasher = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.source_path):
        hasher.update(_stable_json_bytes(document.model_dump()))
    for file_path in sorted(qa_files):
        hasher.update(file_path.name.encode("utf-8"))
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _context_id_from_path(file_path: Path) -> str:
    match = _CONTEXT_STEM.match(file_path.stem)
    return match.group(1) if match else file_path.stem


def _error_record(file_path: Path, error: Exception, context_id: str) -> Dict[str, str]:
    return {
        "context_id": context_id,
        "source_path": str(file_path),
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _write_errors(
    errors: List[Dict[str, str]], errors_path: Optional[str | Path]
) -> None:
    if not errors_path:
        return
    target = Path(errors_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
