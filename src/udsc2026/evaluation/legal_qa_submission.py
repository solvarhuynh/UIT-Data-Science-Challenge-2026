"""Strict, deterministic artifacts for the official LegalQA submission format.

The Task 2 artifact is a ZIP archive whose sole member is ``submission.json``.
That member contains one JSON object.  Every root key is a question ID and its
value is an object containing exactly one ``answer`` field::

    {"question-id": {"answer": "Câu trả lời ..."}}

The superseded array form ``[{"id": ..., "answer": ...}]`` is deliberately
rejected.  :class:`LegalQASubmissionItem` remains an internal, immutable
representation so callers can safely load, inspect, and rewrite an artifact.

Competition identifiers are validated as opaque strings.  Answer text is never
trimmed, normalized, or otherwise rewritten: leading/trailing whitespace,
Unicode normalization form, line endings, and control characters are preserved.
Empty and whitespace-only answers are valid in the official schema; callers may
opt into a separate release-quality warning or error policy.

Untrusted archives are inspected without extraction.  The loader bounds both
stored and expanded sizes, rejects suspicious ZIP metadata, and parses strict
UTF-8 JSON without accepting duplicate keys or JavaScript numeric constants.
Writers validate first, serialize canonically, and replace destinations
atomically.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
import unicodedata
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import IO, Any, BinaryIO, Iterator, List, Optional, Set, Tuple
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

LEGAL_QA_SUBMISSION_MEMBER_NAME = "submission.json"
LEGAL_QA_DEFAULT_MAX_JSON_BYTES = 128 * 1024 * 1024
LEGAL_QA_DEFAULT_MAX_ZIP_BYTES = 256 * 1024 * 1024
LEGAL_QA_DEFAULT_MAX_COMPRESSION_RATIO = 200.0

_SUPPORTED_ZIP_COMPRESSION = {ZIP_STORED, ZIP_DEFLATED}
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class LegalQASubmissionError(ValueError):
    """Raised when a LegalQA artifact violates its official contract."""


class LegalQAEmptyAnswerWarning(UserWarning):
    """Warn that an officially valid artifact contains blank answers."""


class EmptyAnswerPolicy(str, Enum):
    """Optional release policy layered on top of the official schema."""

    ALLOW = "allow"
    WARN = "warn"
    ERROR = "error"


class LegalQASubmissionItem(BaseModel):
    """One immutable answer associated with one official question ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: StrictStr
    answer: StrictStr

    @field_validator("id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        """Reject ambiguous identifiers without rewriting valid identifiers."""

        _validate_identifier(value, "question id")
        return value

    @field_validator("answer")
    @classmethod
    def validate_answer_unicode(cls, value: str) -> str:
        """Require serializable Unicode while preserving every answer code point."""

        _validate_unicode_scalar_string(value, "answer")
        return value


def _validate_unicode_scalar_string(value: str, label: str) -> None:
    """Reject lone surrogate code points without normalizing the string."""

    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain valid Unicode scalar values") from exc


def _validate_identifier(value: str, label: str) -> None:
    """Validate an already type-checked opaque competition identifier."""

    if not value:
        raise ValueError(f"{label} must not be empty")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError(f"{label} must not contain control characters")
    _validate_unicode_scalar_string(value, label)


def _first_duplicate(values: Iterable[str]) -> Optional[str]:
    """Return the first repeated string while preserving diagnostic order."""

    seen: Set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _materialize_ids(values: Iterable[str], *, label: str) -> List[str]:
    """Consume an ID iterable exactly once and validate all of its values."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{label} must be an iterable of strings")
    materialized: List[Any] = list(values)
    for value in materialized:
        if not isinstance(value, str):
            raise TypeError(f"{label} must contain only strings")
        _validate_identifier(value, label.rstrip("s"))
    duplicate = _first_duplicate(materialized)
    if duplicate is not None:
        raise LegalQASubmissionError(f"{label} must be unique; duplicate {duplicate!r}")
    return materialized


def _format_id_preview(values: Iterable[str], *, limit: int = 5) -> str:
    """Format deterministic, bounded diagnostics for a collection of IDs."""

    ordered = sorted(values)
    preview = ", ".join(repr(value) for value in ordered[:limit])
    if len(ordered) > limit:
        preview += f", ... ({len(ordered)} total)"
    return preview


def _coerce_empty_answer_policy(
    policy: EmptyAnswerPolicy | str,
) -> EmptyAnswerPolicy:
    """Validate a release-policy option independently from artifact validity."""

    if isinstance(policy, EmptyAnswerPolicy):
        return policy
    if isinstance(policy, str):
        try:
            return EmptyAnswerPolicy(policy)
        except ValueError as exc:
            raise ValueError(
                "empty_answer_policy must be one of 'allow', 'warn', or 'error'"
            ) from exc
    raise TypeError("empty_answer_policy must be an EmptyAnswerPolicy or string")


def _apply_empty_answer_policy(
    items: Sequence[LegalQASubmissionItem],
    *,
    policy: EmptyAnswerPolicy,
) -> None:
    """Apply an opt-in release check without changing official schema rules."""

    # ``strip`` is used only as a predicate.  The original answer, including
    # whitespace and normalization form, is never replaced by the stripped value.
    empty_ids = [item.id for item in items if not item.answer.strip()]
    if not empty_ids or policy is EmptyAnswerPolicy.ALLOW:
        return
    message = (
        "submission contains blank answers for question ids: "
        f"{_format_id_preview(empty_ids)}"
    )
    if policy is EmptyAnswerPolicy.WARN:
        warnings.warn(message, LegalQAEmptyAnswerWarning, stacklevel=3)
        return
    raise LegalQASubmissionError(message + " (forbidden by release policy)")


def validate_legal_qa_submission(
    payload: object,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    empty_answer_policy: EmptyAnswerPolicy | str = EmptyAnswerPolicy.ALLOW,
) -> List[LegalQASubmissionItem]:
    """Validate a decoded official LegalQA submission payload.

    Args:
        payload: Decoded JSON.  Its root must be a non-empty object.  Every key
            is a question ID and every value must be exactly
            ``{"answer": <string>}``.
        expected_question_ids: If supplied, require exact question coverage.
            The iterable is consumed exactly once.
        empty_answer_policy: Optional release check.  ``allow`` is the official
            schema behavior; ``warn`` emits :class:`LegalQAEmptyAnswerWarning`;
            ``error`` rejects empty or whitespace-only strings.  Answer text is
            inspected but never modified.

    Returns:
        Fresh, deeply immutable items in source-object order.

    Raises:
        LegalQASubmissionError: If the payload violates an artifact invariant.
        TypeError: If a configuration iterable or policy has an invalid type.
        ValueError: If a release-policy value is unknown.
    """

    policy = _coerce_empty_answer_policy(empty_answer_policy)
    if not isinstance(payload, Mapping):
        raise LegalQASubmissionError(
            "submission JSON root must be an object mapping question IDs to "
            "answer objects; legacy array submissions are forbidden"
        )
    if not payload:
        raise LegalQASubmissionError("submission must contain at least one answer")

    items: List[LegalQASubmissionItem] = []
    for raw_question_id, raw_answer_object in payload.items():
        if not isinstance(raw_question_id, str):
            raise LegalQASubmissionError(
                "submission question IDs must be JSON object keys (strings)"
            )
        try:
            _validate_identifier(raw_question_id, "question id")
        except ValueError as exc:
            raise LegalQASubmissionError(
                f"invalid submission question id {raw_question_id!r}: {exc}"
            ) from exc
        if not isinstance(raw_answer_object, Mapping):
            raise LegalQASubmissionError(
                f"answer value for question {raw_question_id!r} must be a JSON object"
            )
        answer_object = dict(raw_answer_object)
        if set(answer_object) != {"answer"}:
            raise LegalQASubmissionError(
                f"answer object for question {raw_question_id!r} must contain "
                "exactly one field named 'answer'"
            )
        candidate: object = {
            "id": raw_question_id,
            "answer": answer_object["answer"],
        }
        try:
            item = LegalQASubmissionItem.model_validate(candidate, strict=True)
        except (TypeError, ValueError) as exc:
            raise LegalQASubmissionError(
                f"invalid answer object for question {raw_question_id!r}: {exc}"
            ) from exc
        items.append(item)

    question_ids = [item.id for item in items]

    if expected_question_ids is not None:
        expected_ids = _materialize_ids(
            expected_question_ids,
            label="expected question ids",
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
            raise LegalQASubmissionError(
                "submission question coverage mismatch (" + "; ".join(details) + ")"
            )

    _apply_empty_answer_policy(items, policy=policy)

    # Never reuse a model supplied by the caller.  This makes ownership explicit
    # and prevents future mutable fields from accidentally leaking by reference.
    return [LegalQASubmissionItem(id=item.id, answer=item.answer) for item in items]


def _reject_json_constant(value: str) -> None:
    """Reject JavaScript NaN/Infinity extensions accepted by :mod:`json`."""

    raise LegalQASubmissionError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: List[Tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys instead of silently taking the last value."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LegalQASubmissionError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _decode_json(data: bytes, *, source: str) -> object:
    """Decode strict UTF-8 JSON with actionable, bounded errors."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LegalQASubmissionError(f"{source} must be valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except LegalQASubmissionError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LegalQASubmissionError(
            f"{source} is not valid strict JSON: {exc}"
        ) from exc


def _canonical_json_bytes(items: Sequence[LegalQASubmissionItem]) -> bytes:
    """Serialize validated items as canonical UTF-8 JSON plus one final newline."""

    payload = {
        item.id: {"answer": item.answer}
        for item in sorted(items, key=lambda candidate: candidate.id)
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _materialize_payload_for_write(payload: object) -> object:
    """Convert internal item iterables to the official object wire shape.

    Raw lists/tuples of dictionaries are the obsolete wire contract and are
    rejected.  Only already validated item models receive this convenience.
    """

    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, (set, frozenset)):
        raise LegalQASubmissionError(
            "submission writer payload must have a deterministic order"
        )
    if isinstance(payload, (str, bytes, bytearray)) or not isinstance(
        payload, Iterable
    ):
        return payload
    materialized = list(payload)
    if any(not isinstance(item, LegalQASubmissionItem) for item in materialized):
        raise LegalQASubmissionError(
            "submission writer payload must be the official root object or an "
            "iterable of LegalQASubmissionItem; legacy raw arrays are forbidden"
        )
    items = [item for item in materialized if isinstance(item, LegalQASubmissionItem)]
    duplicate = _first_duplicate(item.id for item in items)
    if duplicate is not None:
        raise LegalQASubmissionError(
            f"question ids must be unique; duplicate {duplicate!r}"
        )
    return {item.id: {"answer": item.answer} for item in items}


def _validate_size_limit(value: int, *, label: str) -> None:
    """Validate byte limits while rejecting booleans as integers."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _validate_ratio_limit(value: float, *, label: str) -> None:
    """Validate a finite positive decompression-ratio limit."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")


@contextmanager
def _open_regular_binary(path: Path, *, max_bytes: int) -> Iterator[BinaryIO]:
    """Open a regular non-symlink input and enforce its stored byte size."""

    try:
        initial_metadata = path.lstat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise LegalQASubmissionError(f"cannot inspect submission input: {exc}") from exc
    if stat.S_ISLNK(initial_metadata.st_mode):
        raise LegalQASubmissionError("submission input must not be a symbolic link")
    # Reject FIFOs before os.open: opening one for reading could otherwise block.
    if not stat.S_ISREG(initial_metadata.st_mode):
        raise LegalQASubmissionError("submission input must be a regular file")
    if initial_metadata.st_size > max_bytes:
        raise LegalQASubmissionError(
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
        raise LegalQASubmissionError(
            f"cannot safely open submission input: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LegalQASubmissionError("submission input must be a regular file")
        if metadata.st_size > max_bytes:
            raise LegalQASubmissionError(
                f"submission input exceeds the {max_bytes}-byte safety limit"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_bounded(stream: IO[bytes], *, max_bytes: int, label: str) -> bytes:
    """Read at most one byte beyond a declared expanded-size limit."""

    data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise LegalQASubmissionError(
            f"{label} exceeds the {max_bytes}-byte safety limit"
        )
    return data


def _read_json_file(path: Path, *, max_json_bytes: int) -> bytes:
    """Read standalone JSON without following symbolic links."""

    with _open_regular_binary(path, max_bytes=max_json_bytes) as stream:
        return _read_bounded(
            stream,
            max_bytes=max_json_bytes,
            label="submission JSON",
        )


def _zip_member_is_link_or_special(info: ZipInfo) -> bool:
    """Detect Unix links/devices while allowing portable regular-file metadata."""

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    return file_type not in (0, stat.S_IFREG)


def _validate_zip_ratio(info: ZipInfo, *, max_compression_ratio: float) -> None:
    """Reject highly expanding members before any decompression occurs."""

    if info.file_size == 0:
        return
    if info.compress_size <= 0:
        raise LegalQASubmissionError(
            "submission ZIP member has an invalid compressed size"
        )
    ratio = info.file_size / info.compress_size
    if ratio > max_compression_ratio:
        raise LegalQASubmissionError(
            "submission ZIP exceeds the allowed decompression ratio "
            f"({ratio:.1f} > {max_compression_ratio:g})"
        )


def _read_zip_member(
    path: Path,
    *,
    max_json_bytes: int,
    max_zip_bytes: int,
    max_compression_ratio: float,
) -> bytes:
    """Inspect and read the sole JSON member without extracting the archive."""

    try:
        with _open_regular_binary(path, max_bytes=max_zip_bytes) as stream:
            with ZipFile(stream, "r") as archive:
                members = archive.infolist()
                if len(members) != 1:
                    raise LegalQASubmissionError(
                        "submission ZIP must contain exactly one archive member"
                    )
                member = members[0]
                if member.filename != LEGAL_QA_SUBMISSION_MEMBER_NAME:
                    raise LegalQASubmissionError(
                        "submission ZIP member must be named exactly "
                        f"{LEGAL_QA_SUBMISSION_MEMBER_NAME!r}"
                    )
                if member.is_dir() or _zip_member_is_link_or_special(member):
                    raise LegalQASubmissionError(
                        "submission ZIP member must be a regular file"
                    )
                if member.flag_bits & 0x1:
                    raise LegalQASubmissionError(
                        "encrypted submission ZIP members are not supported"
                    )
                if member.compress_type not in _SUPPORTED_ZIP_COMPRESSION:
                    raise LegalQASubmissionError(
                        "submission ZIP uses an unsupported compression method"
                    )
                if member.file_size > max_json_bytes:
                    raise LegalQASubmissionError(
                        f"{LEGAL_QA_SUBMISSION_MEMBER_NAME} exceeds the "
                        f"{max_json_bytes}-byte safety limit"
                    )
                _validate_zip_ratio(
                    member,
                    max_compression_ratio=max_compression_ratio,
                )
                with archive.open(member, "r") as member_stream:
                    return _read_bounded(
                        member_stream,
                        max_bytes=max_json_bytes,
                        label=LEGAL_QA_SUBMISSION_MEMBER_NAME,
                    )
    except LegalQASubmissionError:
        raise
    except (BadZipFile, EOFError, RuntimeError, OSError) as exc:
        raise LegalQASubmissionError(f"invalid submission ZIP: {exc}") from exc


def load_legal_qa_submission(
    input_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    empty_answer_policy: EmptyAnswerPolicy | str = EmptyAnswerPolicy.ALLOW,
    max_json_bytes: int = LEGAL_QA_DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = LEGAL_QA_DEFAULT_MAX_ZIP_BYTES,
    max_compression_ratio: float = LEGAL_QA_DEFAULT_MAX_COMPRESSION_RATIO,
) -> List[LegalQASubmissionItem]:
    """Load an official root-object JSON/ZIP into immutable internal items.

    JSON arrays, including the superseded ``[{"id", "answer"}]`` shape, are
    rejected before coverage or release-policy checks.
    """

    _validate_size_limit(max_json_bytes, label="max_json_bytes")
    _validate_size_limit(max_zip_bytes, label="max_zip_bytes")
    _validate_ratio_limit(max_compression_ratio, label="max_compression_ratio")
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
            max_compression_ratio=max_compression_ratio,
        )
        source = LEGAL_QA_SUBMISSION_MEMBER_NAME
    else:
        raise LegalQASubmissionError("submission input must use .json or .zip")
    decoded = _decode_json(encoded, source=source)
    return validate_legal_qa_submission(
        decoded,
        expected_question_ids=expected_question_ids,
        empty_answer_policy=empty_answer_policy,
    )


def _validate_output_path(path: Path, *, suffix: str) -> None:
    """Reject ambiguous output targets before creating a temporary file."""

    if path.suffix.casefold() != suffix:
        raise LegalQASubmissionError(f"submission output path must use {suffix}")
    if path.is_symlink():
        raise LegalQASubmissionError(
            "submission output target must not be a symbolic link"
        )
    if path.is_dir():
        raise LegalQASubmissionError("submission output target must not be a directory")
    if path.exists() and not path.is_file():
        raise LegalQASubmissionError("submission output target must be a regular file")


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
        # A target may change after initial validation.  Recheck immediately
        # before replacement so a newly introduced link/device is never followed.
        if path.is_symlink():
            raise LegalQASubmissionError(
                "submission output target became a symbolic link"
            )
        if path.exists() and not path.is_file():
            raise LegalQASubmissionError(
                "submission output target is no longer a regular file"
            )
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


def write_legal_qa_submission_json(
    payload: object,
    output_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    empty_answer_policy: EmptyAnswerPolicy | str = EmptyAnswerPolicy.ALLOW,
    max_json_bytes: int = LEGAL_QA_DEFAULT_MAX_JSON_BYTES,
) -> None:
    """Atomically write canonical official root-object ``submission.json``.

    ``payload`` may be the official mapping itself or an ordered iterable of
    :class:`LegalQASubmissionItem` returned by this module.  Raw arrays of
    dictionaries are never accepted as a compatibility format.
    """

    _validate_size_limit(max_json_bytes, label="max_json_bytes")
    path = Path(output_path)
    _validate_output_path(path, suffix=".json")
    items = validate_legal_qa_submission(
        _materialize_payload_for_write(payload),
        expected_question_ids=expected_question_ids,
        empty_answer_policy=empty_answer_policy,
    )
    encoded = _canonical_json_bytes(items)
    if len(encoded) > max_json_bytes:
        raise LegalQASubmissionError(
            f"canonical submission JSON exceeds the {max_json_bytes}-byte safety limit"
        )
    with _atomic_output(path) as stream:
        stream.write(encoded)


def write_legal_qa_submission_zip(
    payload: object,
    output_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    empty_answer_policy: EmptyAnswerPolicy | str = EmptyAnswerPolicy.ALLOW,
    max_json_bytes: int = LEGAL_QA_DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = LEGAL_QA_DEFAULT_MAX_ZIP_BYTES,
) -> None:
    """Atomically write the deterministic one-member official ZIP artifact.

    The accepted in-memory forms match :func:`write_legal_qa_submission_json`;
    the member bytes always use the official root-object wire shape.
    """

    _validate_size_limit(max_json_bytes, label="max_json_bytes")
    _validate_size_limit(max_zip_bytes, label="max_zip_bytes")
    path = Path(output_path)
    _validate_output_path(path, suffix=".zip")
    items = validate_legal_qa_submission(
        _materialize_payload_for_write(payload),
        expected_question_ids=expected_question_ids,
        empty_answer_policy=empty_answer_policy,
    )
    encoded = _canonical_json_bytes(items)
    if len(encoded) > max_json_bytes:
        raise LegalQASubmissionError(
            f"{LEGAL_QA_SUBMISSION_MEMBER_NAME} exceeds the "
            f"{max_json_bytes}-byte safety limit"
        )
    member = ZipInfo(
        filename=LEGAL_QA_SUBMISSION_MEMBER_NAME,
        date_time=_ZIP_TIMESTAMP,
    )
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
            raise LegalQASubmissionError(
                "canonical submission ZIP exceeds the "
                f"{max_zip_bytes}-byte safety limit"
            )


def write_legal_qa_submission(
    payload: object,
    output_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    empty_answer_policy: EmptyAnswerPolicy | str = EmptyAnswerPolicy.ALLOW,
    max_json_bytes: int = LEGAL_QA_DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = LEGAL_QA_DEFAULT_MAX_ZIP_BYTES,
) -> None:
    """Write canonical object JSON or ZIP according to the output extension."""

    path = Path(output_path)
    if path.suffix.casefold() == ".json":
        write_legal_qa_submission_json(
            payload,
            path,
            expected_question_ids=expected_question_ids,
            empty_answer_policy=empty_answer_policy,
            max_json_bytes=max_json_bytes,
        )
        return
    if path.suffix.casefold() == ".zip":
        write_legal_qa_submission_zip(
            payload,
            path,
            expected_question_ids=expected_question_ids,
            empty_answer_policy=empty_answer_policy,
            max_json_bytes=max_json_bytes,
            max_zip_bytes=max_zip_bytes,
        )
        return
    raise LegalQASubmissionError("submission output path must use .json or .zip")


def _paths_collide(input_path: Path, output_path: Path) -> bool:
    """Detect textual, resolved, and existing hard-link path collisions."""

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


def package_legal_qa_submission(
    input_json_path: str | Path,
    output_zip_path: str | Path,
    *,
    expected_question_ids: Optional[Iterable[str]] = None,
    empty_answer_policy: EmptyAnswerPolicy | str = EmptyAnswerPolicy.ALLOW,
    max_json_bytes: int = LEGAL_QA_DEFAULT_MAX_JSON_BYTES,
    max_zip_bytes: int = LEGAL_QA_DEFAULT_MAX_ZIP_BYTES,
) -> None:
    """Validate official object JSON and package its canonical ZIP form."""

    source = Path(input_json_path)
    destination = Path(output_zip_path)
    _validate_size_limit(max_json_bytes, label="max_json_bytes")
    _validate_size_limit(max_zip_bytes, label="max_zip_bytes")
    if source.suffix.casefold() != ".json":
        raise LegalQASubmissionError("package input path must use .json")
    _validate_output_path(destination, suffix=".zip")
    if _paths_collide(source, destination):
        raise LegalQASubmissionError("submission input and output paths must differ")
    items = load_legal_qa_submission(
        source,
        expected_question_ids=expected_question_ids,
        empty_answer_policy=empty_answer_policy,
        max_json_bytes=max_json_bytes,
    )
    # Constraints (including one-shot iterators) were already applied by load.
    write_legal_qa_submission_zip(
        items,
        destination,
        max_json_bytes=max_json_bytes,
        max_zip_bytes=max_zip_bytes,
    )


__all__ = [
    "LEGAL_QA_DEFAULT_MAX_COMPRESSION_RATIO",
    "LEGAL_QA_DEFAULT_MAX_JSON_BYTES",
    "LEGAL_QA_DEFAULT_MAX_ZIP_BYTES",
    "LEGAL_QA_SUBMISSION_MEMBER_NAME",
    "EmptyAnswerPolicy",
    "LegalQAEmptyAnswerWarning",
    "LegalQASubmissionError",
    "LegalQASubmissionItem",
    "load_legal_qa_submission",
    "package_legal_qa_submission",
    "validate_legal_qa_submission",
    "write_legal_qa_submission",
    "write_legal_qa_submission_json",
    "write_legal_qa_submission_zip",
]
