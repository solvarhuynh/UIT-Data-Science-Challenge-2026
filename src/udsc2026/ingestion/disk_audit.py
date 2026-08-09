"""Post-write integrity and task-coverage audit for processed BTC corpora."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from udsc2026.contracts import LegalChunk, LegalParent
from udsc2026.ingestion.cleaners.models import CleanDocument
from udsc2026.ingestion.cleaners.patterns import ONLY_SYMBOLS, WATERMARK_LINE

_EXAMPLE_LIMIT = 50
_ALLOWED_REMOVED_LINE_REASONS = {
    "explicit_page_marker",
    "repeated_page_edge_duplicate",
    "table_of_contents_entry",
    "table_of_contents_header",
}
_BARE_NUMERIC_REMOVAL = re.compile(r"^\d{1,4}$")
_FRACTION_OR_CODE_REMOVAL = re.compile(
    r"^[+-]?\d+(?:\s*[./:-]\s*\d+)+(?:\s*(?:%|‰))?$", re.UNICODE
)
_FLAT_METADATA_FIELDS = (
    "law_name",
    "chapter",
    "section",
    "article",
    "clause",
    "point",
    "effective_date",
    "source",
)


class ProcessedCorpusAuditReport(BaseModel):
    """Serializable result of reading a completed corpus back from disk."""

    schema_version: str = "processed-corpus-audit-v1"
    processed_root: str
    corpus_tree_hash: str
    document_file_count: int
    chunk_file_count: int
    parent_file_count: int
    document_count: int
    chunk_count: int
    parent_count: int
    referenced_parent_count: int
    structure_type_counts: Dict[str, int] = Field(default_factory=dict)
    invalid_document_record_count: int = 0
    invalid_chunk_record_count: int = 0
    invalid_parent_record_count: int = 0
    duplicate_chunk_id_count: int = 0
    duplicate_parent_id_count: int = 0
    empty_chunk_count: int = 0
    empty_parent_count: int = 0
    empty_chunk_file_count: int = 0
    empty_parent_file_count: int = 0
    empty_chunk_file_document_ids: List[str] = Field(default_factory=list)
    empty_parent_file_document_ids: List[str] = Field(default_factory=list)
    orphan_chunk_count: int = 0
    cross_document_parent_count: int = 0
    unreferenced_parent_count: int = 0
    non_null_parent_text_count: int = 0
    missing_required_metadata_count: int = 0
    flattened_metadata_mismatch_count: int = 0
    cleaned_removed_line_count: int = 0
    removed_line_reason_counts: Dict[str, int] = Field(default_factory=dict)
    removed_line_audit_error_count: int = 0
    semantically_risky_removed_line_count: int = 0
    manifest_document_count: int = 0
    manifest_physical_context_count: int = 0
    manifest_exact_duplicate_count: int = 0
    manifest_conflicting_duplicate_count: int = 0
    manifest_orphan_context_count: int = 0
    official_empty_source_document_count: int = 0
    official_empty_source_document_ids: List[str] = Field(default_factory=list)
    legal_ir_question_count: int = 0
    legal_ir_gold_occurrence_count: int = 0
    legal_ir_unique_gold_document_count: int = 0
    legal_ir_missing_gold_document_count: int = 0
    legal_ir_missing_gold_document_ids: List[str] = Field(default_factory=list)
    legal_ir_empty_gold_document_count: int = 0
    legal_ir_empty_gold_document_ids: List[str] = Field(default_factory=list)
    legal_ir_all_gold_empty_question_count: int = 0
    legal_qa_question_count: int = 0
    legal_qa_invalid_answer_count: int = 0
    validation_count_mismatches: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    integrity_gate_passed: bool
    semantic_completeness_gate_passed: bool
    quality_gate_passed: bool
    integrity_failures: List[str] = Field(default_factory=list)
    semantic_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error_examples: List[str] = Field(default_factory=list)


def audit_processed_corpus(
    processed_root: str | Path,
    raw_root: str | Path,
    *,
    report_path: Optional[str | Path] = None,
) -> ProcessedCorpusAuditReport:
    """Read every emitted record and verify file, ID, parent, and task coverage."""

    root = Path(processed_root)
    raw = Path(raw_root)
    documents_dir = root / "documents"
    chunks_dir = root / "chunks"
    parents_dir = root / "parents"
    metadata_dir = root / "metadata"
    document_files = sorted(documents_dir.glob("*.json"))
    chunk_files = sorted(chunks_dir.glob("*.jsonl"))
    parent_files = sorted(parents_dir.glob("*.jsonl"))
    tree_hasher = hashlib.sha256()
    error_examples: List[str] = []

    document_ids: Set[str] = set()
    invalid_documents = 0
    removed_line_count = 0
    removed_line_reason_counts: Counter[str] = Counter()
    removed_line_audit_errors = 0
    risky_removed_lines = 0
    for path in document_files:
        try:
            payload = _read_hashed_json(path, root, tree_hasher)
            document = CleanDocument.model_validate(payload)
            if document.doc_id != path.stem:
                raise ValueError(
                    "doc_id {0!r} does not match filename".format(document.doc_id)
                )
            if document.doc_id in document_ids:
                raise ValueError("duplicate document id {0}".format(document.doc_id))
            document_ids.add(document.doc_id)
            removed_line_count += len(document.removed_lines)
            cleaning = document.metadata.get("cleaning")
            reason_counts = (
                cleaning.get("removed_line_reason_counts")
                if isinstance(cleaning, dict)
                else None
            )
            if not _valid_removed_line_reason_counts(
                reason_counts, len(document.removed_lines)
            ):
                removed_line_audit_errors += 1
                _add_example(
                    error_examples,
                    "{0}: invalid removed_line_reason_counts".format(path),
                )
            else:
                assert isinstance(reason_counts, dict)
                removed_line_reason_counts.update(reason_counts)
            for line in document.removed_lines:
                if _is_semantically_risky_removed_line(line):
                    risky_removed_lines += 1
                    _add_example(
                        error_examples,
                        "{0}: risky removed line {1!r}".format(path, line),
                    )
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            invalid_documents += 1
            _add_example(error_examples, "{0}: {1}".format(path, exc))

    parent_doc_by_id: Dict[str, str] = {}
    parent_count = 0
    invalid_parents = 0
    duplicate_parent_ids: Set[str] = set()
    empty_parent_count = 0
    parent_record_stems: Set[str] = set()
    for path in parent_files:
        for line_number, payload_or_error in _iter_hashed_jsonl(
            path, root, tree_hasher
        ):
            if isinstance(payload_or_error, Exception):
                invalid_parents += 1
                _add_example(
                    error_examples,
                    "{0}:{1}: {2}".format(path, line_number, payload_or_error),
                )
                continue
            try:
                parent = LegalParent.model_validate(payload_or_error)
                parent_record_stems.add(path.stem)
                if parent.doc_id != path.stem:
                    raise ValueError("parent doc_id does not match filename")
                if not parent.text.strip():
                    empty_parent_count += 1
                if parent.parent_id in parent_doc_by_id:
                    duplicate_parent_ids.add(parent.parent_id)
                else:
                    parent_doc_by_id[parent.parent_id] = parent.doc_id
                parent_count += 1
            except (TypeError, ValueError) as exc:
                invalid_parents += 1
                _add_example(
                    error_examples, "{0}:{1}: {2}".format(path, line_number, exc)
                )

    chunk_ids: Set[str] = set()
    duplicate_chunk_ids: Set[str] = set()
    referenced_parent_ids: Set[str] = set()
    structure_type_counts: Counter[str] = Counter()
    official_empty_source_ids: Set[str] = set()
    chunk_record_stems: Set[str] = set()
    chunk_count = 0
    invalid_chunks = 0
    empty_chunks = 0
    orphan_chunks = 0
    cross_document_parents = 0
    non_null_parent_text = 0
    missing_metadata = 0
    flat_mismatches = 0
    for path in chunk_files:
        for line_number, payload_or_error in _iter_hashed_jsonl(
            path, root, tree_hasher
        ):
            if isinstance(payload_or_error, Exception):
                invalid_chunks += 1
                _add_example(
                    error_examples,
                    "{0}:{1}: {2}".format(path, line_number, payload_or_error),
                )
                continue
            try:
                chunk = LegalChunk.model_validate(payload_or_error)
                chunk_record_stems.add(path.stem)
                if chunk.doc_id != path.stem:
                    raise ValueError("chunk doc_id does not match filename")
                chunk_count += 1
                if chunk.chunk_id in chunk_ids:
                    duplicate_chunk_ids.add(chunk.chunk_id)
                else:
                    chunk_ids.add(chunk.chunk_id)
                if not chunk.text.strip():
                    empty_chunks += 1
                if chunk.parent_text is not None:
                    non_null_parent_text += 1
                structure_type = str(chunk.metadata.get("structure_type", "article"))
                structure_type_counts[structure_type] += 1
                if structure_type == "empty_source_placeholder":
                    official_empty_source_ids.add(chunk.doc_id)
                if _missing_required_metadata(chunk, structure_type):
                    missing_metadata += 1
                flat_mismatches += _flat_metadata_mismatch_count(chunk)
                if not chunk.parent_id or chunk.parent_id not in parent_doc_by_id:
                    orphan_chunks += 1
                else:
                    referenced_parent_ids.add(chunk.parent_id)
                    if parent_doc_by_id[chunk.parent_id] != chunk.doc_id:
                        cross_document_parents += 1
            except (TypeError, ValueError) as exc:
                invalid_chunks += 1
                _add_example(
                    error_examples, "{0}:{1}: {2}".format(path, line_number, exc)
                )

    manifest = _read_optional_json(metadata_dir / "manifest.json")
    manifest_ids = (
        {str(value) for value in manifest.get("document_ids", [])}
        if manifest
        else set()
    )
    validation = _read_optional_json(metadata_dir / "validation_report.json")
    validation_mismatches = _validation_count_mismatches(
        validation,
        document_count=len(document_ids),
        chunk_count=chunk_count,
        parent_count=parent_count,
    )

    legal_ir = _audit_legal_ir(raw, document_ids, official_empty_source_ids)
    legal_qa = _audit_legal_qa(raw)
    document_stems = {path.stem for path in document_files}
    chunk_stems = {path.stem for path in chunk_files}
    parent_stems = {path.stem for path in parent_files}
    empty_chunk_file_ids = sorted(document_stems - chunk_record_stems)
    empty_parent_file_ids = sorted(document_stems - parent_record_stems)
    unreferenced_parents = set(parent_doc_by_id) - referenced_parent_ids

    integrity_failures: List[str] = []
    _fail_if(
        integrity_failures,
        "document_chunk_file_set_mismatch",
        document_stems != chunk_stems,
    )
    _fail_if(
        integrity_failures,
        "document_parent_file_set_mismatch",
        document_stems != parent_stems,
    )
    _fail_if(
        integrity_failures,
        "manifest_document_set_mismatch",
        bool(manifest) and manifest_ids != document_ids,
    )
    _fail_if(
        integrity_failures,
        "manifest_context_counts_inconsistent",
        bool(manifest)
        and int(manifest.get("physical_context_file_count", 0))
        != int(manifest.get("unique_context_count", 0))
        + int(manifest.get("exact_duplicate_context_count", 0))
        + int(manifest.get("orphan_context_count", 0)),
    )
    _fail_if(integrity_failures, "invalid_document_records", invalid_documents > 0)
    _fail_if(integrity_failures, "invalid_chunk_records", invalid_chunks > 0)
    _fail_if(integrity_failures, "invalid_parent_records", invalid_parents > 0)
    _fail_if(integrity_failures, "duplicate_chunk_ids", bool(duplicate_chunk_ids))
    _fail_if(integrity_failures, "duplicate_parent_ids", bool(duplicate_parent_ids))
    _fail_if(integrity_failures, "empty_chunks", empty_chunks > 0)
    _fail_if(integrity_failures, "empty_parents", empty_parent_count > 0)
    _fail_if(
        integrity_failures,
        "documents_without_chunk_records",
        bool(empty_chunk_file_ids),
    )
    _fail_if(
        integrity_failures,
        "documents_without_parent_records",
        bool(empty_parent_file_ids),
    )
    _fail_if(integrity_failures, "orphan_chunks", orphan_chunks > 0)
    _fail_if(
        integrity_failures, "cross_document_parent_links", cross_document_parents > 0
    )
    _fail_if(integrity_failures, "unreferenced_parents", bool(unreferenced_parents))
    _fail_if(
        integrity_failures, "repeated_parent_text_in_children", non_null_parent_text > 0
    )
    _fail_if(integrity_failures, "missing_required_metadata", missing_metadata > 0)
    _fail_if(integrity_failures, "flattened_metadata_mismatch", flat_mismatches > 0)
    _fail_if(
        integrity_failures,
        "invalid_removed_line_audit",
        removed_line_audit_errors > 0,
    )
    _fail_if(
        integrity_failures,
        "semantically_risky_removed_lines",
        risky_removed_lines > 0,
    )
    _fail_if(
        integrity_failures,
        "manifest_orphans",
        int(manifest.get("orphan_context_count", 0)) > 0,
    )
    _fail_if(
        integrity_failures,
        "manifest_conflicting_duplicates",
        int(manifest.get("conflicting_duplicate_context_count", 0)) > 0,
    )
    _fail_if(
        integrity_failures, "validation_count_mismatch", bool(validation_mismatches)
    )
    _fail_if(
        integrity_failures,
        "legal_ir_missing_gold_documents",
        bool(legal_ir["missing_gold_ids"]),
    )
    _fail_if(
        integrity_failures,
        "legal_qa_invalid_train_answers",
        int(legal_qa["invalid_answer_count"]) > 0,
    )
    validation_failures = set(validation.get("quality_gate_failures", []))
    validation_failures.discard("official_empty_source_content")
    _fail_if(
        integrity_failures, "in_memory_validation_failures", bool(validation_failures)
    )

    semantic_issues: List[str] = []
    _fail_if(
        semantic_issues,
        "official_empty_source_content",
        bool(official_empty_source_ids),
    )
    _fail_if(
        semantic_issues,
        "legal_ir_gold_has_empty_source_content",
        bool(legal_ir["empty_gold_ids"]),
    )
    warnings: List[str] = []
    if validation.get("suspected_split_article_count", 0):
        warnings.append("ambiguous_split_article_candidates_require_review")
    if validation.get("partial_structure_document_count", 0):
        warnings.append("partial_structure_documents_present")

    integrity_passed = not integrity_failures
    semantic_passed = not semantic_issues
    output = ProcessedCorpusAuditReport(
        processed_root=str(root),
        corpus_tree_hash=tree_hasher.hexdigest(),
        document_file_count=len(document_files),
        chunk_file_count=len(chunk_files),
        parent_file_count=len(parent_files),
        document_count=len(document_ids),
        chunk_count=chunk_count,
        parent_count=parent_count,
        referenced_parent_count=len(referenced_parent_ids),
        structure_type_counts=dict(sorted(structure_type_counts.items())),
        invalid_document_record_count=invalid_documents,
        invalid_chunk_record_count=invalid_chunks,
        invalid_parent_record_count=invalid_parents,
        duplicate_chunk_id_count=len(duplicate_chunk_ids),
        duplicate_parent_id_count=len(duplicate_parent_ids),
        empty_chunk_count=empty_chunks,
        empty_parent_count=empty_parent_count,
        empty_chunk_file_count=len(empty_chunk_file_ids),
        empty_parent_file_count=len(empty_parent_file_ids),
        empty_chunk_file_document_ids=empty_chunk_file_ids,
        empty_parent_file_document_ids=empty_parent_file_ids,
        orphan_chunk_count=orphan_chunks,
        cross_document_parent_count=cross_document_parents,
        unreferenced_parent_count=len(unreferenced_parents),
        non_null_parent_text_count=non_null_parent_text,
        missing_required_metadata_count=missing_metadata,
        flattened_metadata_mismatch_count=flat_mismatches,
        cleaned_removed_line_count=removed_line_count,
        removed_line_reason_counts=dict(sorted(removed_line_reason_counts.items())),
        removed_line_audit_error_count=removed_line_audit_errors,
        semantically_risky_removed_line_count=risky_removed_lines,
        manifest_document_count=len(manifest_ids),
        manifest_physical_context_count=int(
            manifest.get("physical_context_file_count", 0)
        ),
        manifest_exact_duplicate_count=int(
            manifest.get("exact_duplicate_context_count", 0)
        ),
        manifest_conflicting_duplicate_count=int(
            manifest.get("conflicting_duplicate_context_count", 0)
        ),
        manifest_orphan_context_count=int(manifest.get("orphan_context_count", 0)),
        official_empty_source_document_count=len(official_empty_source_ids),
        official_empty_source_document_ids=sorted(official_empty_source_ids),
        legal_ir_question_count=int(legal_ir["question_count"]),
        legal_ir_gold_occurrence_count=int(legal_ir["gold_occurrence_count"]),
        legal_ir_unique_gold_document_count=int(legal_ir["unique_gold_count"]),
        legal_ir_missing_gold_document_count=len(legal_ir["missing_gold_ids"]),
        legal_ir_missing_gold_document_ids=legal_ir["missing_gold_ids"],
        legal_ir_empty_gold_document_count=len(legal_ir["empty_gold_ids"]),
        legal_ir_empty_gold_document_ids=legal_ir["empty_gold_ids"],
        legal_ir_all_gold_empty_question_count=int(
            legal_ir["all_gold_empty_questions"]
        ),
        legal_qa_question_count=int(legal_qa["question_count"]),
        legal_qa_invalid_answer_count=int(legal_qa["invalid_answer_count"]),
        validation_count_mismatches=validation_mismatches,
        integrity_gate_passed=integrity_passed,
        semantic_completeness_gate_passed=semantic_passed,
        quality_gate_passed=integrity_passed and semantic_passed,
        integrity_failures=integrity_failures,
        semantic_issues=semantic_issues,
        warnings=warnings,
        error_examples=error_examples,
    )
    target = (
        Path(report_path) if report_path else metadata_dir / "disk_audit_report.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _valid_removed_line_reason_counts(value: Any, expected_total: int) -> bool:
    if not isinstance(value, dict):
        return False
    if any(
        not isinstance(key, str)
        or key not in _ALLOWED_REMOVED_LINE_REASONS
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for key, count in value.items()
    ):
        return False
    return sum(value.values()) == expected_total


def _is_semantically_risky_removed_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        _BARE_NUMERIC_REMOVAL.fullmatch(stripped)
        or _FRACTION_OR_CODE_REMOVAL.fullmatch(stripped)
        or ONLY_SYMBOLS.fullmatch(stripped)
        or WATERMARK_LINE.fullmatch(stripped)
    )


def _read_hashed_json(path: Path, root: Path, hasher: Any) -> Any:
    raw = path.read_bytes()
    _start_hashed_file(hasher, path, root)
    hasher.update(raw)
    return _strict_json_loads(raw.decode("utf-8"))


def _iter_hashed_jsonl(
    path: Path, root: Path, hasher: Any
) -> Iterator[Tuple[int, Any]]:
    _start_hashed_file(hasher, path, root)
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            hasher.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                yield line_number, _strict_json_loads(raw_line.decode("utf-8"))
            except (UnicodeError, TypeError, ValueError) as exc:
                yield line_number, exc


def _start_hashed_file(hasher: Any, path: Path, root: Path) -> None:
    hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
    hasher.update(b"\0")


def _strict_json_loads(value: str) -> Any:
    def unique_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key {0!r}".format(key))
            result[key] = item
        return result

    def reject_non_finite(constant: str) -> None:
        raise ValueError("non-finite JSON number {0}".format(constant))

    return json.loads(
        value,
        object_pairs_hook=unique_pairs,
        parse_constant=reject_non_finite,
    )


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _missing_required_metadata(chunk: LegalChunk, structure_type: str) -> bool:
    if chunk.metadata.get("law_name") in (None, ""):
        return True
    if chunk.metadata.get("source") in (None, ""):
        return True
    if structure_type == "article":
        return chunk.metadata.get("article") in (None, "")
    if structure_type == "unstructured_fallback":
        return not bool(chunk.metadata.get("fallback_chunking"))
    if structure_type == "document_context":
        return not bool(chunk.metadata.get("supplemental_context"))
    if structure_type == "empty_source_placeholder":
        return not bool(chunk.metadata.get("source_content_empty"))
    return True


def _flat_metadata_mismatch_count(chunk: LegalChunk) -> int:
    return sum(
        getattr(chunk, field) != chunk.metadata.get(field)
        for field in _FLAT_METADATA_FIELDS
    )


def _validation_count_mismatches(
    validation: Dict[str, Any], **actual_counts: int
) -> Dict[str, Dict[str, int]]:
    mismatches: Dict[str, Dict[str, int]] = {}
    for key, actual in actual_counts.items():
        expected = validation.get(key)
        if isinstance(expected, int) and expected != actual:
            mismatches[key] = {"reported": expected, "disk": actual}
    return mismatches


def _audit_legal_ir(
    raw_root: Path, document_ids: Set[str], empty_source_ids: Set[str]
) -> Dict[str, Any]:
    path = raw_root / "LegalIR" / "train.json"
    if not path.is_file():
        return {
            "question_count": 0,
            "gold_occurrence_count": 0,
            "unique_gold_count": 0,
            "missing_gold_ids": [],
            "empty_gold_ids": [],
            "all_gold_empty_questions": 0,
        }
    payload = _strict_json_loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("LegalIR train root must be an object")
    gold_ids: List[str] = []
    all_gold_empty_questions = 0
    for record in payload.values():
        if not isinstance(record, dict) or not isinstance(record.get("answer"), list):
            continue
        answers = [str(value) for value in record["answer"]]
        gold_ids.extend(answers)
        if answers and all(value in empty_source_ids for value in answers):
            all_gold_empty_questions += 1
    unique_gold = set(gold_ids)
    return {
        "question_count": len(payload),
        "gold_occurrence_count": len(gold_ids),
        "unique_gold_count": len(unique_gold),
        "missing_gold_ids": sorted(unique_gold - document_ids),
        "empty_gold_ids": sorted(unique_gold & empty_source_ids),
        "all_gold_empty_questions": all_gold_empty_questions,
    }


def _audit_legal_qa(raw_root: Path) -> Dict[str, int]:
    path = raw_root / "LegalQA" / "train.json"
    if not path.is_file():
        return {"question_count": 0, "invalid_answer_count": 0}
    payload = _strict_json_loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("LegalQA train root must be an object")
    invalid = sum(
        not isinstance(record, dict)
        or not isinstance(record.get("answer"), str)
        or not record["answer"].strip()
        for record in payload.values()
    )
    return {"question_count": len(payload), "invalid_answer_count": invalid}


def _add_example(examples: List[str], value: str) -> None:
    if len(examples) < _EXAMPLE_LIMIT:
        examples.append(value)


def _fail_if(failures: List[str], name: str, condition: bool) -> None:
    if condition:
        failures.append(name)
