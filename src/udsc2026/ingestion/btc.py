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
    """Audit summary for one task fixture file."""

    file_name: str
    relative_path: str
    task: str
    fixture_kind: str
    record_count: int
    schema_hash: str
    answer_kind: str
    answer_required: bool
    invalid_record_count: int = 0
    missing_question_count: int = 0
    missing_answer_count: int = 0
    missing_question_answer_count: int = 0


class BTCExactDuplicate(BaseModel):
    """One physical context copy equal to an already accepted context."""

    doc_id: str
    kept_source_path: str
    duplicate_source_path: str
    content_hash: str


class BTCExtractionResult(BaseModel):
    """Accepted BTC contexts plus explicit duplicate and error audit data."""

    source_root: str
    discovered_context_files: List[str] = Field(default_factory=list)
    raw_documents: List[RawDocument] = Field(default_factory=list)
    orphan_context_ids: List[str] = Field(default_factory=list)
    orphan_source_paths: List[str] = Field(default_factory=list)
    error_records: List[Dict[str, str]] = Field(default_factory=list)
    exact_duplicates: List[BTCExactDuplicate] = Field(default_factory=list)
    conflicting_duplicate_context_ids: List[str] = Field(default_factory=list)


class BTCOrphanReport(BaseModel):
    """Rejected context files collected during BTC extraction."""

    schema_version: str = "btc-context-v2"
    source_root: str
    orphan_context_count: int
    orphan_context_ids: List[str] = Field(default_factory=list)
    orphan_source_paths: List[str] = Field(default_factory=list)
    error_count: int = 0
    errors: List[Dict[str, str]] = Field(default_factory=list)
    exact_duplicate_context_count: int = 0
    conflicting_duplicate_context_count: int = 0


class BTCManifest(BaseModel):
    """Hash-based corpus manifest for BTC context and task fixture files."""

    schema_version: str = "btc-context-v2"
    source_root: str
    corpus_hash: str
    context_file_count: int
    accepted_context_count: int
    orphan_context_count: int
    physical_context_file_count: int
    unique_context_count: int
    exact_duplicate_context_count: int
    conflicting_duplicate_context_count: int
    document_ids: List[str] = Field(default_factory=list)
    context_files: List[str] = Field(default_factory=list)
    canonical_context_files: List[str] = Field(default_factory=list)
    exact_duplicates: List[BTCExactDuplicate] = Field(default_factory=list)
    qa_fixtures: List[BTCQAFixtureManifest] = Field(default_factory=list)


def discover_btc_context_files(raw_directory: str | Path) -> List[Path]:
    """Find every BTC ``context_*.json`` file below the raw data root."""

    root = Path(raw_directory)
    return sorted(path for path in root.rglob("context_*.json") if path.is_file())


def discover_btc_qa_files(raw_directory: str | Path) -> List[Path]:
    """Find BTC task fixtures that should be validated but not chunked."""

    root = Path(raw_directory)
    discovered: List[Path] = []
    for file_name in _BTC_QA_FILENAMES:
        discovered.extend(path for path in root.rglob(file_name) if path.is_file())
    return sorted(discovered)


def extract_btc_context_documents(
    raw_directory: str | Path,
    errors_path: Optional[str | Path] = None,
) -> BTCExtractionResult:
    """Extract unique contexts and distinguish exact copies from conflicts."""

    root = Path(raw_directory)
    context_files = discover_btc_context_files(root)
    documents: List[RawDocument] = []
    errors: List[Dict[str, str]] = []
    orphan_context_ids: List[str] = []
    orphan_source_paths: List[str] = []
    exact_duplicates: List[BTCExactDuplicate] = []
    conflicting_duplicate_context_ids: List[str] = []
    seen_documents: Dict[str, RawDocument] = {}

    for file_path in context_files:
        context_id = _context_id_from_path(file_path)
        try:
            extracted = extract_raw_document(str(file_path))
            candidates = (
                (extracted,) if isinstance(extracted, RawDocument) else extracted
            )
            for document in candidates:
                existing = seen_documents.get(document.doc_id)
                if existing is not None:
                    existing_hash = _document_content_hash(existing)
                    duplicate_hash = _document_content_hash(document)
                    if existing_hash == duplicate_hash:
                        exact_duplicates.append(
                            BTCExactDuplicate(
                                doc_id=document.doc_id,
                                kept_source_path=existing.source_path,
                                duplicate_source_path=document.source_path,
                                content_hash=existing_hash,
                            )
                        )
                        continue
                    conflicting_duplicate_context_ids.append(document.doc_id)
                    raise ValueError(
                        "Conflicting duplicate context_id: {0}".format(document.doc_id)
                    )
                seen_documents[document.doc_id] = document
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
        exact_duplicates=exact_duplicates,
        conflicting_duplicate_context_ids=conflicting_duplicate_context_ids,
    )


def build_btc_manifest(
    result: BTCExtractionResult,
    raw_directory: str | Path,
) -> BTCManifest:
    """Build a deterministic manifest for unique and physical BTC contexts."""

    root = Path(raw_directory)
    qa_files = discover_btc_qa_files(root)
    qa_fixtures = [_build_qa_fixture_manifest(path, root) for path in qa_files]
    corpus_hash = _corpus_hash(result.raw_documents, qa_files, root)
    return BTCManifest(
        source_root=str(root),
        corpus_hash=corpus_hash,
        context_file_count=len(result.discovered_context_files),
        accepted_context_count=len(result.raw_documents),
        orphan_context_count=len(result.orphan_context_ids),
        physical_context_file_count=len(result.discovered_context_files),
        unique_context_count=len(result.raw_documents),
        exact_duplicate_context_count=len(result.exact_duplicates),
        conflicting_duplicate_context_count=len(
            result.conflicting_duplicate_context_ids
        ),
        document_ids=[document.doc_id for document in result.raw_documents],
        context_files=list(result.discovered_context_files),
        canonical_context_files=[
            document.source_path for document in result.raw_documents
        ],
        exact_duplicates=list(result.exact_duplicates),
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
        exact_duplicate_context_count=len(result.exact_duplicates),
        conflicting_duplicate_context_count=len(
            result.conflicting_duplicate_context_ids
        ),
    )


def _build_qa_fixture_manifest(
    file_path: Path, source_root: Path
) -> BTCQAFixtureManifest:
    payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        message = "BTC task fixture must be a JSON object: {0}".format(file_path)
        raise ValueError(message)

    invalid = 0
    missing_question = 0
    missing_answer = 0
    answer_kinds = set()
    answer_required = file_path.name.casefold() == "train.json"
    for record in payload.values():
        if not isinstance(record, dict):
            invalid += 1
            continue
        if not record.get("question"):
            missing_question += 1
        answer = record.get("answer")
        if answer in (None, "", []):
            missing_answer += 1
        else:
            answer_kinds.add(_answer_kind(answer))

    if not answer_kinds:
        answer_kind = "none"
    elif len(answer_kinds) == 1:
        answer_kind = next(iter(answer_kinds))
    else:
        answer_kind = "mixed"
    return BTCQAFixtureManifest(
        file_name=file_path.name,
        relative_path=_relative_path(file_path, source_root),
        task=_task_from_path(file_path),
        fixture_kind=file_path.stem.casefold(),
        record_count=len(payload),
        schema_hash=_stable_json_hash(payload),
        answer_kind=answer_kind,
        answer_required=answer_required,
        invalid_record_count=invalid,
        missing_question_count=missing_question,
        missing_answer_count=missing_answer,
        missing_question_answer_count=(
            invalid + missing_question + (missing_answer if answer_required else 0)
        ),
    )


def _corpus_hash(
    documents: Sequence[RawDocument], qa_files: Sequence[Path], source_root: Path
) -> str:
    hasher = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.doc_id):
        hasher.update(_stable_json_bytes(_document_content_payload(document)))
    for file_path in sorted(qa_files):
        hasher.update(_relative_path(file_path, source_root).encode("utf-8"))
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _document_content_payload(document: RawDocument) -> Dict[str, Any]:
    """Return semantic content without machine- or task-specific path data."""

    return dict(document.model_dump(exclude={"source_path"}))


def _document_content_hash(document: RawDocument) -> str:
    return _stable_json_hash(_document_content_payload(document))


def _relative_path(file_path: Path, source_root: Path) -> str:
    try:
        return file_path.relative_to(source_root).as_posix()
    except ValueError:
        return file_path.as_posix()


def _task_from_path(file_path: Path) -> str:
    for part in file_path.parts:
        normalized = part.casefold()
        if normalized == "legalir":
            return "LegalIR"
        if normalized == "legalqa":
            return "LegalQA"
    return "unknown"


def _answer_kind(answer: Any) -> str:
    if isinstance(answer, list):
        return "context_id_list"
    if isinstance(answer, str):
        return "text"
    return type(answer).__name__


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
