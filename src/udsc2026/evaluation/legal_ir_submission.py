"""Strict, deterministic JSON and ZIP utilities for LegalIR submissions.

The competition-facing payload is deliberately kept separate from the generic
LegalQA CSV writer.  A valid payload is a JSON object whose keys are question
identifiers and whose values contain exactly one ``answer`` ranking::

    {"question-id": {"answer": ["doc-1", "doc-2"]}}

The organizer contract does not impose a minimum answer length.  In
particular, ``{"answer": []}`` is a valid prediction and receives zero
precision/recall for that question under the published metric definition.

This module validates untrusted JSON/ZIP input without extracting archives,
supports exact question/corpus checks, and writes artifacts atomically.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, BinaryIO, Iterator, List, Optional, Set, Tuple
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

SUBMISSION_MEMBER_NAME = "submission.json"
DEFAULT_MAX_JSON_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_ZIP_BYTES = 256 * 1024 * 1024
_SUPPORTED_ZIP_COMPRESSION = {ZIP_STORED, ZIP_DEFLATED}
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class LegalIRSubmissionError(ValueError):
    """Raised when a LegalIR artifact violates the competition contract."""


class LegalIRSubmissionItem(BaseModel):
    """Internal immutable ranking; the wire fields are question key + ``answer``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: StrictStr
    documents: Tuple[StrictStr, ...]

    @field_validator("id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        """Reject blank or accidentally padded identifiers without rewriting them."""

        _validate_identifier(value, "question id")
        return value

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        """Require an unambiguous, ordered ranking of document identifiers."""

        for value in values:
            _validate_identifier(value, "document id")
        duplicate = _first_duplicate(values)
        if duplicate is not None:
            raise ValueError(
                f"documents must contain unique document ids; duplicate {duplicate!r}"
            )
        return values


def _validate_identifier(value: str, label: str) -> None:
    """Validate an already type-checked opaque competition identifier."""

    if not value:
        raise ValueError(f"{label} must not be empty")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if any(
        ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in value
    ):
        raise ValueError(f"{label} must not contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain valid Unicode scalar values") from exc


def _first_duplicate(values: Iterable[str]) -> Optional[str]:
    """Return the first repeated value, preserving diagnostic order."""

    seen: Set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _materialize_ids(
    values: Iterable[str],
    *,
    label: str,
    sort_unordered: bool = False,
) -> List[str]:
    """Validate an external ID collection and give it a deterministic order."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{label} must be an iterable of strings")
    materialized: List[Any] = list(values)
    for value in materialized:
        if not isinstance(value, str):
            raise TypeError(f"{label} must contain only strings")
        _validate_identifier(value, label.rstrip("s"))
    if sort_unordered and isinstance(values, (set, frozenset)):
        materialized.sort()
    duplicate = _first_duplicate(materialized)
    if duplicate is not None:
        raise LegalIRSubmissionError(f"{label} must be unique; duplicate {duplicate!r}")
    return materialized


def _format_id_preview(values: Iterable[str], *, limit: int = 5) -> str:
    """Format stable, bounded diagnostics for potentially large ID sets."""

    ordered = sorted(values)
    preview = ", ".join(repr(value) for value in ordered[:limit])
    if len(ordered) > limit:
        preview += f", ... ({len(ordered)} total)"
    return preview


def validate_legal_ir_submission(
    payload: object,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    allowed_document_ids: Optional[Iterable[str]] = None,
    require_complete_ranking: bool = False,
) -> List[LegalIRSubmissionItem]:
    """Validate a decoded official LegalIR submission payload.

    Args:
        payload: Decoded JSON value.  The root must be a non-empty object mapping
            each question ID to exactly ``{"answer": [document IDs...]}``.
            The answer array may be empty.
            A non-empty sequence containing only already validated
            :class:`LegalIRSubmissionItem` objects is also accepted as an
            internal hand-off between the loader, completion helper, and writer;
            decoded legacy arrays are never accepted.
        expected_question_ids: If supplied, require exact question coverage.
        allowed_document_ids: If supplied, reject document IDs outside this corpus.
        require_complete_ranking: Require every ranking to contain every allowed
            corpus document exactly once.  This option requires
            ``allowed_document_ids``.

    Returns:
        Validated items in their original question and ranking order.

    Raises:
        LegalIRSubmissionError: If any competition invariant is violated.
        TypeError: If an optional ID collection is not an iterable of strings.
    """

    items: List[LegalIRSubmissionItem] = []
    if (
        isinstance(payload, Sequence)
        and not isinstance(payload, (str, bytes, bytearray))
        and bool(payload)
        and all(isinstance(item, LegalIRSubmissionItem) for item in payload)
    ):
        for item in payload:
            if not isinstance(item, LegalIRSubmissionItem):
                raise LegalIRSubmissionError(
                    "internal submission hand-off contains an invalid item"
                )
            # Revalidate a fresh primitive snapshot.  Besides making ownership
            # explicit, this prevents a future mutable model field from being
            # shared across the loader/writer boundary.
            items.append(
                LegalIRSubmissionItem.model_validate(
                    {"id": item.id, "documents": tuple(item.documents)},
                    strict=True,
                )
            )
    else:
        if not isinstance(payload, Mapping):
            raise LegalIRSubmissionError("submission JSON root must be an object")
        if not payload:
            raise LegalIRSubmissionError(
                "submission must contain at least one question"
            )

        for raw_question_id, raw_entry in payload.items():
            if not isinstance(raw_question_id, str):
                raise LegalIRSubmissionError(
                    "submission question ids must be JSON object string keys"
                )
            try:
                _validate_identifier(raw_question_id, "question id")
            except ValueError as exc:
                raise LegalIRSubmissionError(
                    f"invalid submission question id {raw_question_id!r}: {exc}"
                ) from exc
            if not isinstance(raw_entry, Mapping):
                raise LegalIRSubmissionError(
                    f"submission entry for question {raw_question_id!r} must be "
                    "a JSON object"
                )
            if len(raw_entry) != 1 or "answer" not in raw_entry:
                raise LegalIRSubmissionError(
                    f"submission entry for question {raw_question_id!r} must "
                    "contain exactly the field 'answer'"
                )
            raw_documents = raw_entry["answer"]
            if not isinstance(raw_documents, list):
                raise LegalIRSubmissionError(
                    f"invalid submission entry for question {raw_question_id!r}: "
                    "answer must be a JSON array"
                )
            try:
                item = LegalIRSubmissionItem.model_validate(
                    {
                        "id": raw_question_id,
                        # The internal tuple keeps rankings deeply immutable.  It
                        # is created only after enforcing the public JSON array.
                        "documents": tuple(raw_documents),
                    },
                    strict=True,
                )
            except (TypeError, ValueError) as exc:
                raise LegalIRSubmissionError(
                    f"invalid submission entry for question {raw_question_id!r}: {exc}"
                ) from exc
            items.append(item)

    question_ids = [item.id for item in items]
    duplicate_question = _first_duplicate(question_ids)
    if duplicate_question is not None:
        raise LegalIRSubmissionError(
            f"question ids must be unique; duplicate {duplicate_question!r}"
        )

    expected_ids: Optional[List[str]] = None
    if expected_question_ids is not None:
        expected_ids = _materialize_ids(
            expected_question_ids,
            label="expected question ids",
            sort_unordered=True,
        )
        actual_set = set(question_ids)
        expected_set = set(expected_ids)
        missing = expected_set - actual_set
        unexpected = actual_set - expected_set
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {_format_id_preview(missing)}")
            if unexpected:
                details.append(f"unexpected: {_format_id_preview(unexpected)}")
            raise LegalIRSubmissionError(
                "submission question coverage mismatch (" + "; ".join(details) + ")"
            )

    allowed_ids: Optional[List[str]] = None
    if allowed_document_ids is not None:
        allowed_ids = _materialize_ids(
            allowed_document_ids,
            label="allowed document ids",
            sort_unordered=True,
        )
        allowed_set = set(allowed_ids)
        for item in items:
            unknown = set(item.documents) - allowed_set
            if unknown:
                raise LegalIRSubmissionError(
                    f"question {item.id!r} contains document ids outside the corpus: "
                    f"{_format_id_preview(unknown)}"
                )

    if require_complete_ranking:
        if allowed_ids is None:
            raise LegalIRSubmissionError(
                "require_complete_ranking requires allowed_document_ids"
            )
        allowed_set = set(allowed_ids)
        for item in items:
            missing_documents = allowed_set - set(item.documents)
            if missing_documents:
                raise LegalIRSubmissionError(
                    f"question {item.id!r} does not contain a complete corpus "
                    f"ranking; missing: {_format_id_preview(missing_documents)}"
                )

    # Return fresh models so no caller-supplied model instance is reused.
    return [
        LegalIRSubmissionItem(id=item.id, documents=tuple(item.documents))
        for item in items
    ]


def complete_legal_ir_rankings(
    payload: object,
    corpus_document_ids: Iterable[str],
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
) -> List[LegalIRSubmissionItem]:
    """Append missing corpus IDs to each valid ranking without changing its prefix.

    Sequence inputs preserve corpus order.  Sets and frozensets are sorted
    lexicographically so repeated runs remain deterministic.  This is retained
    as an explicit legacy/diagnostic helper only: completing a ranking is not an
    organizer requirement and can reduce the secondary Precision score.
    """

    corpus_ids = _materialize_ids(
        corpus_document_ids,
        label="corpus document ids",
        sort_unordered=True,
    )
    items = validate_legal_ir_submission(
        payload,
        expected_question_ids=expected_question_ids,
        allowed_document_ids=corpus_ids,
    )
    completed: List[LegalIRSubmissionItem] = []
    for item in items:
        ranked = list(item.documents)
        present = set(ranked)
        ranked.extend(doc_id for doc_id in corpus_ids if doc_id not in present)
        completed.append(LegalIRSubmissionItem(id=item.id, documents=tuple(ranked)))
    return completed


def _reject_json_constant(value: str) -> None:
    """Reject JavaScript-style NaN/Infinity extensions accepted by ``json``."""

    raise LegalIRSubmissionError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: List[Tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys instead of silently accepting the last value."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LegalIRSubmissionError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_json(data: bytes, *, source: str) -> object:
    """Decode strict UTF-8 JSON with bounded, actionable failures."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LegalIRSubmissionError(f"{source} must be valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except LegalIRSubmissionError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LegalIRSubmissionError(
            f"{source} is not valid strict JSON: {exc}"
        ) from exc


def _canonical_json_bytes(items: Sequence[LegalIRSubmissionItem]) -> bytes:
    """Serialize validated items as canonical UTF-8 JSON with one final newline."""

    payload = {item.id: {"answer": list(item.documents)} for item in items}
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _validate_size_limit(value: int, *, label: str) -> None:
    """Validate configurable byte limits while rejecting booleans as integers."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


@contextmanager
def _open_regular_binary(path: Path, *, max_bytes: int) -> Iterator[BinaryIO]:
    """Open a regular, non-symlink input file and enforce its stored size."""

    try:
        initial_metadata = path.lstat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise LegalIRSubmissionError(f"cannot inspect submission input: {exc}") from exc
    if stat.S_ISLNK(initial_metadata.st_mode):
        raise LegalIRSubmissionError("submission input must not be a symbolic link")
    # Reject FIFOs before ``os.open`` because opening a FIFO for reading can block.
    if not stat.S_ISREG(initial_metadata.st_mode):
        raise LegalIRSubmissionError("submission input must be a regular file")
    if initial_metadata.st_size > max_bytes:
        raise LegalIRSubmissionError(
            f"submission input exceeds the {max_bytes}-byte safety limit"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise LegalIRSubmissionError(
            f"cannot safely open submission input: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LegalIRSubmissionError("submission input must be a regular file")
        if metadata.st_size > max_bytes:
            raise LegalIRSubmissionError(
                f"submission input exceeds the {max_bytes}-byte safety limit"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_bounded(stream: IO[bytes], *, max_bytes: int, label: str) -> bytes:
    """Read at most one byte beyond a declared safety limit."""

    data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise LegalIRSubmissionError(
            f"{label} exceeds the {max_bytes}-byte safety limit"
        )
    return data


def _read_json_file(path: Path, *, max_json_bytes: int) -> bytes:
    """Read a standalone JSON submission without following symlinks."""

    with _open_regular_binary(path, max_bytes=max_json_bytes) as stream:
        return _read_bounded(
            stream,
            max_bytes=max_json_bytes,
            label="submission JSON",
        )


def _zip_member_is_link_or_special(info: ZipInfo) -> bool:
    """Detect Unix symlink/device metadata while allowing portable regular files."""

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    return file_type not in (0, stat.S_IFREG)


def _read_zip_member(
    path: Path,
    *,
    max_json_bytes: int,
    max_zip_bytes: int,
) -> bytes:
    """Inspect and read the sole JSON member without extracting the archive."""

    try:
        with _open_regular_binary(path, max_bytes=max_zip_bytes) as stream:
            with ZipFile(stream, "r") as archive:
                members = archive.infolist()
                if len(members) != 1:
                    raise LegalIRSubmissionError(
                        "submission ZIP must contain exactly one archive member"
                    )
                member = members[0]
                if member.filename != SUBMISSION_MEMBER_NAME:
                    raise LegalIRSubmissionError(
                        "submission ZIP member must be named exactly "
                        f"{SUBMISSION_MEMBER_NAME!r}"
                    )
                if member.is_dir() or _zip_member_is_link_or_special(member):
                    raise LegalIRSubmissionError(
                        "submission ZIP member must be a regular file"
                    )
                if member.flag_bits & 0x1:
                    raise LegalIRSubmissionError(
                        "encrypted submission ZIP members are not supported"
                    )
                if member.compress_type not in _SUPPORTED_ZIP_COMPRESSION:
                    raise LegalIRSubmissionError(
                        "submission ZIP uses an unsupported compression method"
                    )
                if member.file_size > max_json_bytes:
                    raise LegalIRSubmissionError(
                        "submission.json exceeds the "
                        f"{max_json_bytes}-byte safety limit"
                    )
                with archive.open(member, "r") as member_stream:
                    return _read_bounded(
                        member_stream,
                        max_bytes=max_json_bytes,
                        label=SUBMISSION_MEMBER_NAME,
                    )
    except LegalIRSubmissionError:
        raise
    except (BadZipFile, EOFError, RuntimeError, OSError) as exc:
        raise LegalIRSubmissionError(f"invalid submission ZIP: {exc}") from exc


def load_legal_ir_submission(
    input_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    allowed_document_ids: Optional[Iterable[str]] = None,
    require_complete_ranking: bool = False,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> List[LegalIRSubmissionItem]:
    """Load an official object-shaped JSON/ZIP into immutable internal items."""

    _validate_size_limit(max_json_bytes, label="max_json_bytes")
    _validate_size_limit(max_zip_bytes, label="max_zip_bytes")
    path = Path(input_path)
    suffix = path.suffix.casefold()
    if suffix == ".json":
        encoded = _read_json_file(path, max_json_bytes=max_json_bytes)
        source = "submission JSON"
    elif suffix == ".zip":
        encoded = _read_zip_member(
            path,
            max_json_bytes=max_json_bytes,
            max_zip_bytes=max_zip_bytes,
        )
        source = SUBMISSION_MEMBER_NAME
    else:
        raise LegalIRSubmissionError("submission input must use .json or .zip")
    decoded = _decode_json(encoded, source=source)
    return validate_legal_ir_submission(
        decoded,
        expected_question_ids=expected_question_ids,
        allowed_document_ids=allowed_document_ids,
        require_complete_ranking=require_complete_ranking,
    )


def _validate_output_path(path: Path, *, suffix: str) -> None:
    """Reject ambiguous output targets before creating a temporary file."""

    if path.suffix.casefold() != suffix:
        raise LegalIRSubmissionError(f"submission output path must use {suffix}")
    if path.is_symlink():
        raise LegalIRSubmissionError(
            "submission output target must not be a symbolic link"
        )
    if path.is_dir():
        raise LegalIRSubmissionError("submission output target must not be a directory")
    if path.exists() and not path.is_file():
        raise LegalIRSubmissionError("submission output target must be a regular file")


@contextmanager
def _atomic_output(path: Path) -> Iterator[BinaryIO]:
    """Yield a same-directory temporary stream and atomically replace on success."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    replaced = False
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            descriptor = -1
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck the destination immediately before replacement.  Replacing a
        # regular file is intentional; replacing a newly introduced symlink or
        # directory is not.
        if path.is_symlink():
            raise LegalIRSubmissionError(
                "submission output target became a symbolic link"
            )
        if path.exists() and not path.is_file():
            raise LegalIRSubmissionError(
                "submission output target is no longer a regular file"
            )
        temporary_path = Path(temporary_name)
        # Deterministic reruns need no replacement. On Windows, antivirus or a
        # just-closed ZIP reader can briefly deny replacing the destination.
        if path.is_file() and _files_equal(temporary_path, path):
            temporary_path.unlink()
        else:
            os.replace(temporary_name, path)
        replaced = True
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if not replaced:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _files_equal(left: Path, right: Path) -> bool:
    """Compare bounded artifact files without loading both into memory."""
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_stream, right.open("rb") as right_stream:
            while True:
                left_block = left_stream.read(1024 * 1024)
                right_block = right_stream.read(1024 * 1024)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError:
        return False


def _validated_for_write(
    payload: object,
    *,
    expected_question_ids: Optional[Iterable[str]],
    allowed_document_ids: Optional[Iterable[str]],
    complete_with_document_ids: Optional[Iterable[str]],
) -> List[LegalIRSubmissionItem]:
    """Apply the shared writer validation/completion policy."""

    if complete_with_document_ids is not None:
        if allowed_document_ids is not None:
            raise LegalIRSubmissionError(
                "pass either allowed_document_ids or complete_with_document_ids, "
                "not both"
            )
        return complete_legal_ir_rankings(
            payload,
            complete_with_document_ids,
            expected_question_ids=expected_question_ids,
        )
    return validate_legal_ir_submission(
        payload,
        expected_question_ids=expected_question_ids,
        allowed_document_ids=allowed_document_ids,
    )


def write_legal_ir_submission_json(
    payload: object,
    output_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    allowed_document_ids: Optional[Iterable[str]] = None,
    complete_with_document_ids: Optional[Iterable[str]] = None,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> None:
    """Validate and atomically write the official object-shaped JSON contract."""

    _validate_size_limit(max_json_bytes, label="max_json_bytes")
    path = Path(output_path)
    _validate_output_path(path, suffix=".json")
    items = _validated_for_write(
        payload,
        expected_question_ids=expected_question_ids,
        allowed_document_ids=allowed_document_ids,
        complete_with_document_ids=complete_with_document_ids,
    )
    encoded = _canonical_json_bytes(items)
    if len(encoded) > max_json_bytes:
        raise LegalIRSubmissionError(
            f"canonical submission JSON exceeds the {max_json_bytes}-byte safety limit"
        )
    with _atomic_output(path) as stream:
        stream.write(encoded)


def write_legal_ir_submission_zip(
    payload: object,
    output_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    allowed_document_ids: Optional[Iterable[str]] = None,
    complete_with_document_ids: Optional[Iterable[str]] = None,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> None:
    """Write a deterministic ZIP whose sole member uses the official contract."""

    _validate_size_limit(max_json_bytes, label="max_json_bytes")
    _validate_size_limit(max_zip_bytes, label="max_zip_bytes")
    path = Path(output_path)
    _validate_output_path(path, suffix=".zip")
    items = _validated_for_write(
        payload,
        expected_question_ids=expected_question_ids,
        allowed_document_ids=allowed_document_ids,
        complete_with_document_ids=complete_with_document_ids,
    )
    encoded = _canonical_json_bytes(items)
    if len(encoded) > max_json_bytes:
        raise LegalIRSubmissionError(
            f"{SUBMISSION_MEMBER_NAME} exceeds the {max_json_bytes}-byte safety limit"
        )
    member = ZipInfo(filename=SUBMISSION_MEMBER_NAME, date_time=_ZIP_TIMESTAMP)
    member.compress_type = ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = (stat.S_IFREG | 0o644) << 16
    with _atomic_output(path) as stream:
        with ZipFile(
            stream,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            archive.writestr(
                member,
                encoded,
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )
        if stream.tell() > max_zip_bytes:
            raise LegalIRSubmissionError(
                "canonical submission ZIP exceeds the "
                f"{max_zip_bytes}-byte safety limit"
            )


def write_legal_ir_submission(
    payload: object,
    output_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    allowed_document_ids: Optional[Iterable[str]] = None,
    complete_with_document_ids: Optional[Iterable[str]] = None,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> None:
    """Write JSON or ZIP based on the explicit output filename extension."""

    path = Path(output_path)
    if path.suffix.casefold() == ".json":
        write_legal_ir_submission_json(
            payload,
            path,
            expected_question_ids=expected_question_ids,
            allowed_document_ids=allowed_document_ids,
            complete_with_document_ids=complete_with_document_ids,
            max_json_bytes=max_json_bytes,
        )
        return
    if path.suffix.casefold() == ".zip":
        write_legal_ir_submission_zip(
            payload,
            path,
            expected_question_ids=expected_question_ids,
            allowed_document_ids=allowed_document_ids,
            complete_with_document_ids=complete_with_document_ids,
            max_json_bytes=max_json_bytes,
            max_zip_bytes=max_zip_bytes,
        )
        return
    raise LegalIRSubmissionError("submission output path must use .json or .zip")


def _paths_collide(input_path: Path, output_path: Path) -> bool:
    """Detect textual, resolved, and existing hard-link collisions."""

    try:
        if input_path.resolve(strict=True) == output_path.resolve(strict=False):
            return True
    except (FileNotFoundError, OSError):
        pass
    if input_path.exists() and output_path.exists():
        try:
            return os.path.samefile(input_path, output_path)
        except OSError:
            pass
    return False


def package_legal_ir_submission(
    input_json_path: str | Path,
    output_zip_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    allowed_document_ids: Optional[Iterable[str]] = None,
    require_complete_ranking: bool = False,
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES,
) -> None:
    """Validate a JSON file and package its canonical form as ``submission.json``."""

    source = Path(input_json_path)
    destination = Path(output_zip_path)
    if source.suffix.casefold() != ".json":
        raise LegalIRSubmissionError("package input path must use .json")
    _validate_output_path(destination, suffix=".zip")
    if _paths_collide(source, destination):
        raise LegalIRSubmissionError("submission input and output paths must differ")
    items = load_legal_ir_submission(
        source,
        expected_question_ids=expected_question_ids,
        allowed_document_ids=allowed_document_ids,
        require_complete_ranking=require_complete_ranking,
        max_json_bytes=max_json_bytes,
    )
    # ``items`` is already validated and deeply immutable.  Avoid consuming an
    # optional one-shot constraint iterator a second time in the ZIP writer.
    write_legal_ir_submission_zip(
        items,
        destination,
        max_json_bytes=max_json_bytes,
        max_zip_bytes=max_zip_bytes,
    )


__all__ = [
    "DEFAULT_MAX_JSON_BYTES",
    "DEFAULT_MAX_ZIP_BYTES",
    "SUBMISSION_MEMBER_NAME",
    "LegalIRSubmissionError",
    "LegalIRSubmissionItem",
    "complete_legal_ir_rankings",
    "load_legal_ir_submission",
    "package_legal_ir_submission",
    "validate_legal_ir_submission",
    "write_legal_ir_submission",
    "write_legal_ir_submission_json",
    "write_legal_ir_submission_zip",
]
