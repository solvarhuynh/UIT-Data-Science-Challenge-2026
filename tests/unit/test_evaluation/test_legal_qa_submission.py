"""Official object-contract, determinism, and security tests for LegalQA."""

from __future__ import annotations

import json
import os
import stat
import warnings
from collections.abc import Iterator
from pathlib import Path
from zipfile import (
    ZIP_BZIP2,
    ZIP_DEFLATED,
    ZIP_STORED,
    BadZipFile,
    ZipFile,
    ZipInfo,
)

import pytest
from pydantic import ValidationError

from udsc2026.evaluation.legal_qa_submission import (
    LEGAL_QA_SUBMISSION_MEMBER_NAME,
    EmptyAnswerPolicy,
    LegalQAEmptyAnswerWarning,
    LegalQASubmissionError,
    LegalQASubmissionItem,
    load_legal_qa_submission,
    package_legal_qa_submission,
    validate_legal_qa_submission,
    write_legal_qa_submission,
    write_legal_qa_submission_json,
    write_legal_qa_submission_zip,
)


def _payload() -> dict[str, dict[str, str]]:
    """Return official wire-format data in deliberately non-sorted ID order."""

    return {
        "q-2": {"answer": "Căn cứ Điều 2, người dân được đăng ký."},
        "câu-hỏi-1": {"answer": "Theo Điều 1\nHồ sơ gồm hai giấy tờ."},
    }


def _legacy_array_payload() -> list[dict[str, str]]:
    return [
        {"id": question_id, "answer": answer_object["answer"]}
        for question_id, answer_object in _payload().items()
    ]


def _json_bytes(
    payload: object,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> bytes:
    separators = (",", ":") if indent is None else None
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    ).encode("utf-8")


def _canonical_bytes(payload: object) -> bytes:
    return _json_bytes(payload, sort_keys=True) + b"\n"


def _write_zip(
    path: Path,
    members: list[tuple[str | ZipInfo, bytes]],
    *,
    compression: int = ZIP_DEFLATED,
) -> None:
    with ZipFile(path, "w", compression=compression) as archive:
        for name, content in members:
            archive.writestr(name, content)


def _mark_zip_as_encrypted(path: Path) -> None:
    """Set the encryption bit in both local and central ZIP headers."""

    encoded = bytearray(path.read_bytes())
    locations = (
        (encoded.find(b"PK\x03\x04"), 6),
        (encoded.find(b"PK\x01\x02"), 8),
    )
    assert all(position >= 0 for position, _ in locations)
    for position, flag_offset in locations:
        offset = position + flag_offset
        flags = int.from_bytes(encoded[offset : offset + 2], "little") | 0x1
        encoded[offset : offset + 2] = flags.to_bytes(2, "little")
    path.write_bytes(encoded)


class _OneShotIDs:
    """Iterable that fails if a function consumes it more than once."""

    def __init__(self, values: list[str]) -> None:
        self._values = values
        self._consumed = False

    def __iter__(self) -> Iterator[str]:
        if self._consumed:
            raise AssertionError("ID iterable was consumed more than once")
        self._consumed = True
        return iter(self._values)


def test_validation_accepts_exact_official_object_and_preserves_source_order() -> None:
    answer = "  Die\u0302̀u 1\r\n\tNBSP:\u00a0 FEFF:\ufeff NUL:\x00 cuối  "
    payload = {
        "q-2": {"answer": answer},
        "q-1": {"answer": ""},
    }

    items = validate_legal_qa_submission(
        payload,
        expected_question_ids={"q-1", "q-2"},
    )

    assert [item.id for item in items] == ["q-2", "q-1"]
    assert items[0].answer == answer
    assert items[0].answer.encode("utf-8") == answer.encode("utf-8")
    assert items[1].answer == ""


def test_validation_returns_detached_frozen_models() -> None:
    raw = {"q": {"answer": "before"}}
    item = validate_legal_qa_submission(raw)[0]
    raw["q"]["answer"] = "after"

    assert item.answer == "before"
    with pytest.raises(ValidationError, match="frozen"):
        item.answer = "changed"


@pytest.mark.parametrize(
    "payload",
    [
        _legacy_array_payload(),
        [],
        [{"q": {"answer": "a"}}],
        "{}",
        None,
        1,
        (),
        set(),
    ],
)
def test_validation_rejects_non_object_and_legacy_array_roots(payload: object) -> None:
    with pytest.raises(
        LegalQASubmissionError,
        match="root must be an object.*legacy array",
    ):
        validate_legal_qa_submission(payload)


def test_validation_rejects_empty_object() -> None:
    with pytest.raises(LegalQASubmissionError, match="at least one answer"):
        validate_legal_qa_submission({})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({7: {"answer": "a"}}, "question IDs.*strings"),
        ({"q": "answer"}, "must be a JSON object"),
        ({"q": ["answer"]}, "must be a JSON object"),
        ({"q": {}}, "exactly one field"),
        ({"q": {"answer": "a", "score": 1}}, "exactly one field"),
        ({"q": {"id": "q", "answer": "a"}}, "exactly one field"),
        ({"q": {"Answer": "a"}}, "exactly one field"),
        ({"q": {"answer": 7}}, "valid string"),
        ({"q": {"answer": True}}, "valid string"),
        ({"q": {"answer": None}}, "valid string"),
        ({"": {"answer": "a"}}, "must not be empty"),
        ({" q": {"answer": "a"}}, "surrounding whitespace"),
        ({"q ": {"answer": "a"}}, "surrounding whitespace"),
        ({"q\x00log": {"answer": "a"}}, "control"),
        ({"q\nlog": {"answer": "a"}}, "control"),
        ({"q\u0085log": {"answer": "a"}}, "control"),
        ({"q\u200blog": {"answer": "a"}}, "control"),
        ({"q\ufefflog": {"answer": "a"}}, "control"),
        ({"q\ud800": {"answer": "a"}}, "Unicode scalar"),
        ({"q": {"answer": "bad\udfff"}}, "Unicode scalar"),
    ],
)
def test_validation_rejects_malformed_question_entries(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(LegalQASubmissionError, match=message):
        validate_legal_qa_submission(payload)


def test_internal_model_is_strict_and_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LegalQASubmissionItem.model_validate({"id": 1, "answer": "a"})
    with pytest.raises(ValidationError):
        LegalQASubmissionItem.model_validate(
            {"id": "q", "answer": "a", "confidence": 1.0}
        )


def test_exact_coverage_reports_missing_and_unexpected_ids() -> None:
    with pytest.raises(LegalQASubmissionError) as error:
        validate_legal_qa_submission(
            _payload(),
            expected_question_ids=["q-2", "required-but-missing"],
        )

    message = str(error.value)
    assert "missing" in message and "required-but-missing" in message
    assert "unexpected" in message and "câu-hỏi-1" in message


def test_expected_id_iterable_is_strict_unique_and_consumed_once() -> None:
    ids = _OneShotIDs(["q-2", "câu-hỏi-1"])
    assert len(validate_legal_qa_submission(_payload(), expected_question_ids=ids)) == 2

    with pytest.raises(TypeError, match="iterable of strings"):
        validate_legal_qa_submission(_payload(), expected_question_ids="q-2")
    with pytest.raises(TypeError, match="only strings"):
        validate_legal_qa_submission(_payload(), expected_question_ids=["q-2", 2])
    with pytest.raises(LegalQASubmissionError, match="must be unique"):
        validate_legal_qa_submission(
            _payload(),
            expected_question_ids=["q-2", "q-2"],
        )
    with pytest.raises(ValueError, match="surrounding whitespace"):
        validate_legal_qa_submission(
            _payload(),
            expected_question_ids=["q-2", " padded"],
        )


def test_blank_answers_are_officially_valid_by_default() -> None:
    payload = {"empty": {"answer": ""}, "spaces": {"answer": " \n\t"}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        items = validate_legal_qa_submission(payload)

    assert [item.answer for item in items] == ["", " \n\t"]
    assert caught == []


def test_optional_release_policies_flag_blank_without_modifying_text() -> None:
    payload = {
        "q-empty": {"answer": ""},
        "q-whitespace": {"answer": " \n\t"},
    }

    with pytest.warns(LegalQAEmptyAnswerWarning, match="q-empty") as caught:
        items = validate_legal_qa_submission(
            payload,
            empty_answer_policy=EmptyAnswerPolicy.WARN,
        )
    assert len(caught) == 1
    assert "q-whitespace" in str(caught[0].message)
    assert items[1].answer == " \n\t"

    with pytest.raises(LegalQASubmissionError, match="release policy"):
        validate_legal_qa_submission(payload, empty_answer_policy="error")

    meaningful = validate_legal_qa_submission(
        {"q": {"answer": " nội dung "}},
        empty_answer_policy="error",
    )
    assert meaningful[0].answer == " nội dung "


def test_empty_answer_policy_option_is_strict() -> None:
    with pytest.raises(ValueError, match="allow.*warn.*error"):
        validate_legal_qa_submission(
            {"q": {"answer": ""}},
            empty_answer_policy="forbid",
        )
    with pytest.raises(TypeError, match="EmptyAnswerPolicy or string"):
        validate_legal_qa_submission(
            {"q": {"answer": ""}},
            empty_answer_policy=True,  # type: ignore[arg-type]
        )


def test_json_writer_emits_canonical_sorted_official_object(tmp_path: Path) -> None:
    output = tmp_path / "nested" / LEGAL_QA_SUBMISSION_MEMBER_NAME

    write_legal_qa_submission_json(_payload(), output)
    first = output.read_bytes()
    reversed_payload = dict(reversed(list(_payload().items())))
    write_legal_qa_submission_json(reversed_payload, output)

    assert output.read_bytes() == first
    assert first == _canonical_bytes(_payload())
    assert first.startswith(b'{"c')
    assert b'[{"id"' not in first
    assert b"c\xc3\xa2u-h\xe1\xbb\x8fi-1" in first
    assert not first.startswith(b"\xef\xbb\xbf")
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_json_roundtrip_preserves_answer_code_points_exactly(tmp_path: Path) -> None:
    answer = " \tNFD: a\u0301\r\nNBSP:\u00a0 FEFF:\ufeff NUL:\x00 cuối "
    payload = {"q": {"answer": answer}}
    path = tmp_path / "submission.json"

    write_legal_qa_submission_json(payload, path)
    loaded = load_legal_qa_submission(path)

    assert loaded[0].answer == answer
    assert loaded[0].answer.encode("utf-8") == answer.encode("utf-8")


def test_writer_accepts_one_shot_internal_items_but_rejects_raw_arrays(
    tmp_path: Path,
) -> None:
    consumed = False
    validated = validate_legal_qa_submission(_payload())

    def records() -> Iterator[LegalQASubmissionItem]:
        nonlocal consumed
        if consumed:
            raise AssertionError("item generator was consumed more than once")
        consumed = True
        yield from validated

    output = tmp_path / "submission.json"
    write_legal_qa_submission_json(
        records(),
        output,
        expected_question_ids=_OneShotIDs(["q-2", "câu-hỏi-1"]),
    )

    assert consumed
    assert {
        item.id: {"answer": item.answer} for item in load_legal_qa_submission(output)
    } == _payload()

    with pytest.raises(LegalQASubmissionError, match="legacy raw arrays"):
        write_legal_qa_submission_json(
            _legacy_array_payload(),
            tmp_path / "legacy.json",
        )
    with pytest.raises(LegalQASubmissionError, match="legacy raw arrays"):
        write_legal_qa_submission_json(
            (item for item in _legacy_array_payload()),
            tmp_path / "legacy-generator.json",
        )


def test_internal_writer_rejects_unordered_and_duplicate_items(tmp_path: Path) -> None:
    first = LegalQASubmissionItem(id="q", answer="first")
    second = LegalQASubmissionItem(id="q", answer="second")

    with pytest.raises(LegalQASubmissionError, match="deterministic order"):
        write_legal_qa_submission_json({first}, tmp_path / "unordered.json")
    with pytest.raises(LegalQASubmissionError, match="duplicate 'q'"):
        write_legal_qa_submission_json(
            iter([first, second]),
            tmp_path / "duplicate.json",
        )


def test_zip_writer_has_exact_member_and_deterministic_metadata(tmp_path: Path) -> None:
    first = tmp_path / "submission.zip"
    second = tmp_path / "different-name.zip"
    reversed_payload = dict(reversed(list(_payload().items())))

    write_legal_qa_submission_zip(_payload(), first)
    write_legal_qa_submission_zip(reversed_payload, second)

    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        assert archive.namelist() == [LEGAL_QA_SUBMISSION_MEMBER_NAME]
        assert archive.comment == b""
        member = archive.infolist()[0]
        assert member.filename == "submission.json"
        assert member.date_time == (1980, 1, 1, 0, 0, 0)
        assert member.compress_type == ZIP_DEFLATED
        assert member.extra == b""
        assert stat.S_IFMT(member.external_attr >> 16) == stat.S_IFREG
        assert archive.read(member) == _canonical_bytes(_payload())

    loaded = load_legal_qa_submission(first)
    assert {item.id: {"answer": item.answer} for item in loaded} == _payload()


def test_writers_enforce_size_limits_without_partial_replacement(
    tmp_path: Path,
) -> None:
    json_output = tmp_path / "submission.json"
    json_output.write_bytes(b"previous-json")
    with pytest.raises(LegalQASubmissionError, match="safety limit"):
        write_legal_qa_submission_json(
            _payload(),
            json_output,
            max_json_bytes=4,
        )
    assert json_output.read_bytes() == b"previous-json"

    zip_output = tmp_path / "submission.zip"
    zip_output.write_bytes(b"previous-zip")
    with pytest.raises(LegalQASubmissionError, match="submission.json.*safety limit"):
        write_legal_qa_submission_zip(
            _payload(),
            zip_output,
            max_json_bytes=4,
        )
    assert zip_output.read_bytes() == b"previous-zip"

    with pytest.raises(LegalQASubmissionError, match="ZIP.*safety limit"):
        write_legal_qa_submission_zip(
            _payload(),
            zip_output,
            max_zip_bytes=4,
        )
    assert zip_output.read_bytes() == b"previous-zip"
    assert not list(tmp_path.glob(".submission.zip.*.tmp"))


def test_generic_writer_dispatches_case_insensitive_json_and_zip(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "submission.JSON"
    zip_path = tmp_path / "submission.ZIP"

    write_legal_qa_submission(_payload(), json_path)
    write_legal_qa_submission(_payload(), zip_path)

    assert load_legal_qa_submission(json_path)[0].id == "câu-hỏi-1"
    assert load_legal_qa_submission(zip_path)[1].id == "q-2"
    with pytest.raises(LegalQASubmissionError, match=".json or .zip"):
        write_legal_qa_submission(_payload(), tmp_path / "submission.txt")


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        (b"not json", "not valid strict JSON"),
        (
            b'{"q":{"answer":"a"},"q":{"answer":"b"}}',
            "duplicate JSON object key 'q'",
        ),
        (
            b'{"q":{"answer":"a","answer":"b"}}',
            "duplicate JSON object key 'answer'",
        ),
        (b'{"q":{"answer":NaN}}', "non-standard JSON constant"),
        (b'{"q":{"answer":Infinity}}', "non-standard JSON constant"),
        (b"\xff", "valid UTF-8"),
        (b"\xef\xbb\xbf{}", "not valid strict JSON"),
        (b'{"q":{"answer":"a"}} trailing', "not valid strict JSON"),
        (b'{"\\ud800":{"answer":"a"}}', "Unicode scalar"),
        (b'{"q":{"answer":"\\udfff"}}', "Unicode scalar"),
        (_json_bytes(_legacy_array_payload()), "root must be an object"),
    ],
)
def test_json_loader_rejects_non_strict_unsafe_and_legacy_input(
    encoded: bytes,
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission.json"
    path.write_bytes(encoded)

    with pytest.raises(LegalQASubmissionError, match=message):
        load_legal_qa_submission(path)


def test_json_loader_accepts_valid_escaped_surrogate_pair(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    path.write_bytes(b'{"q":{"answer":"\\ud83d\\ude00"}}')

    assert load_legal_qa_submission(path)[0].answer == "😀"


def test_zip_loader_rejects_legacy_array_member(tmp_path: Path) -> None:
    path = tmp_path / "legacy.zip"
    _write_zip(
        path,
        [(LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_legacy_array_payload()))],
    )

    with pytest.raises(LegalQASubmissionError, match="root must be an object"):
        load_legal_qa_submission(path)


def test_loader_rejects_wrong_extension_directory_symlink_and_fifo(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "submission.txt"
    wrong.write_bytes(_json_bytes(_payload()))
    with pytest.raises(LegalQASubmissionError, match=".json or .zip"):
        load_legal_qa_submission(wrong)

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(LegalQASubmissionError, match="regular file"):
        load_legal_qa_submission(directory)

    target = tmp_path / "real.json"
    target.write_bytes(_json_bytes(_payload()))
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(LegalQASubmissionError, match="symbolic link"):
        load_legal_qa_submission(link)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.json"
        os.mkfifo(fifo)
        with pytest.raises(LegalQASubmissionError, match="regular file"):
            load_legal_qa_submission(fifo)


def test_loader_rejects_size_and_ratio_configuration_errors(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    path.write_bytes(_json_bytes(_payload()))

    with pytest.raises(LegalQASubmissionError, match="safety limit"):
        load_legal_qa_submission(path, max_json_bytes=4)
    for bad_limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            load_legal_qa_submission(
                path,
                max_json_bytes=bad_limit,  # type: ignore[arg-type]
            )
    for bad_ratio in (0, -1, True, float("inf"), float("nan"), "10"):
        with pytest.raises(ValueError, match="finite positive number"):
            load_legal_qa_submission(
                path,
                max_compression_ratio=bad_ratio,  # type: ignore[arg-type]
            )


def test_zip_loader_never_extracts_and_accepts_stored_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "submission.zip"
    _write_zip(
        path,
        [(LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
        compression=ZIP_STORED,
    )

    def reject_extraction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("loader must not extract archive members")

    monkeypatch.setattr(ZipFile, "extract", reject_extraction)
    monkeypatch.setattr(ZipFile, "extractall", reject_extraction)

    assert len(load_legal_qa_submission(path)) == 2
    assert not (tmp_path / LEGAL_QA_SUBMISSION_MEMBER_NAME).exists()


@pytest.mark.parametrize(
    "member_name",
    [
        "nested/submission.json",
        "../submission.json",
        "/submission.json",
        "other.json",
        "Submission.json",
    ],
)
def test_zip_loader_rejects_wrong_nested_and_zip_slip_member_names(
    member_name: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission.zip"
    _write_zip(path, [(member_name, _json_bytes(_payload()))])

    with pytest.raises(LegalQASubmissionError, match="named exactly"):
        load_legal_qa_submission(path)


def test_zip_loader_rejects_empty_extra_and_duplicate_members(tmp_path: Path) -> None:
    empty = tmp_path / "empty.zip"
    _write_zip(empty, [])
    with pytest.raises(LegalQASubmissionError, match="exactly one"):
        load_legal_qa_submission(empty)

    extra = tmp_path / "extra.zip"
    _write_zip(
        extra,
        [("folder/", b""), (LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
    )
    with pytest.raises(LegalQASubmissionError, match="exactly one"):
        load_legal_qa_submission(extra)

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write_zip(
            duplicate,
            [
                (LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_payload())),
                (LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_payload())),
            ],
        )
    with pytest.raises(LegalQASubmissionError, match="exactly one"):
        load_legal_qa_submission(duplicate)


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_zip_loader_rejects_symlink_and_special_members(
    file_type: int,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"special-{file_type}.zip"
    member = ZipInfo(LEGAL_QA_SUBMISSION_MEMBER_NAME)
    member.create_system = 3
    member.external_attr = (file_type | 0o600) << 16
    _write_zip(path, [(member, _json_bytes(_payload()))])

    with pytest.raises(LegalQASubmissionError, match="regular file"):
        load_legal_qa_submission(path)


def test_zip_loader_rejects_encrypted_member_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.zip"
    _write_zip(
        path,
        [(LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
        compression=ZIP_STORED,
    )
    _mark_zip_as_encrypted(path)

    with pytest.raises(LegalQASubmissionError, match="encrypted"):
        load_legal_qa_submission(path)


def test_zip_loader_rejects_unsupported_compression(tmp_path: Path) -> None:
    path = tmp_path / "submission.zip"
    try:
        _write_zip(
            path,
            [(LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
            compression=ZIP_BZIP2,
        )
    except RuntimeError:
        pytest.skip("bzip2 ZIP support is unavailable")

    with pytest.raises(LegalQASubmissionError, match="unsupported compression"):
        load_legal_qa_submission(path)


def test_zip_loader_rejects_invalid_and_oversized_archive_or_member(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not-a-zip")
    with pytest.raises(LegalQASubmissionError, match="invalid submission ZIP"):
        load_legal_qa_submission(invalid)

    valid = tmp_path / "valid.zip"
    _write_zip(valid, [(LEGAL_QA_SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))])
    with pytest.raises(LegalQASubmissionError, match="safety limit"):
        load_legal_qa_submission(valid, max_zip_bytes=4)
    with pytest.raises(LegalQASubmissionError, match="safety limit"):
        load_legal_qa_submission(valid, max_json_bytes=4)


def test_zip_loader_rejects_high_ratio_zip_bomb_before_decompression(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bomb.zip"
    payload = {"q": {"answer": "A" * 500_000}}
    encoded = _json_bytes(payload)
    _write_zip(
        path,
        [(LEGAL_QA_SUBMISSION_MEMBER_NAME, encoded)],
        compression=ZIP_DEFLATED,
    )

    with pytest.raises(LegalQASubmissionError, match="decompression ratio"):
        load_legal_qa_submission(
            path,
            max_json_bytes=len(encoded) + 1,
            max_compression_ratio=10,
        )


def test_corrupt_zip_member_crc_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "submission.zip"
    marker = _json_bytes(_payload())
    _write_zip(
        path,
        [(LEGAL_QA_SUBMISSION_MEMBER_NAME, marker)],
        compression=ZIP_STORED,
    )
    encoded = bytearray(path.read_bytes())
    offset = encoded.find(marker)
    assert offset >= 0
    encoded[offset] ^= 1
    path.write_bytes(encoded)

    with pytest.raises(LegalQASubmissionError, match="invalid submission ZIP"):
        load_legal_qa_submission(path)


def test_writer_rejects_directory_symlink_fifo_and_bad_extension(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "submission.json"
    directory.mkdir()
    with pytest.raises(LegalQASubmissionError, match="directory"):
        write_legal_qa_submission_json(_payload(), directory)

    target = tmp_path / "target.zip"
    link = tmp_path / "linked.zip"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(LegalQASubmissionError, match="symbolic link"):
        write_legal_qa_submission_zip(_payload(), link)

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "output.zip"
        os.mkfifo(fifo)
        with pytest.raises(LegalQASubmissionError, match="regular file"):
            write_legal_qa_submission_zip(_payload(), fifo)

    with pytest.raises(LegalQASubmissionError, match=".json"):
        write_legal_qa_submission_json(_payload(), tmp_path / "wrong.zip")
    with pytest.raises(LegalQASubmissionError, match=".zip"):
        write_legal_qa_submission_zip(_payload(), tmp_path / "wrong.json")


def test_atomic_writer_preserves_existing_file_and_cleans_temp_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "submission.json"
    output.write_bytes(b"previous")

    def fail_replace(source: str, destination: Path) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "udsc2026.evaluation.legal_qa_submission.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="simulated"):
        write_legal_qa_submission_json(_payload(), output)

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".submission.json.*.tmp"))


def test_validation_failure_does_not_touch_output_or_create_parent(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "submission.json"
    existing.write_bytes(b"previous")
    with pytest.raises(LegalQASubmissionError):
        write_legal_qa_submission_json(_legacy_array_payload(), existing)
    assert existing.read_bytes() == b"previous"

    absent_output = tmp_path / "absent" / "submission.json"
    with pytest.raises(LegalQASubmissionError):
        write_legal_qa_submission_json([], absent_output)
    assert not absent_output.parent.exists()


def test_packager_validates_and_canonicalizes_pretty_object_json(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(_json_bytes(_payload(), indent=4))
    output = tmp_path / "submission.zip"

    package_legal_qa_submission(
        source,
        output,
        expected_question_ids=["q-2", "câu-hỏi-1"],
    )

    loaded = load_legal_qa_submission(output)
    assert {item.id: {"answer": item.answer} for item in loaded} == _payload()
    with ZipFile(output) as archive:
        encoded = archive.read(LEGAL_QA_SUBMISSION_MEMBER_NAME)
    assert encoded == _canonical_bytes(_payload())


def test_packager_consumes_expected_id_iterator_once(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(_json_bytes(_payload()))
    output = tmp_path / "submission.zip"

    package_legal_qa_submission(
        source,
        output,
        expected_question_ids=_OneShotIDs(["q-2", "câu-hỏi-1"]),
    )

    assert len(load_legal_qa_submission(output)) == 2


def test_packager_emits_blank_warning_only_once(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(_json_bytes({"q": {"answer": ""}}))

    with pytest.warns(LegalQAEmptyAnswerWarning) as caught:
        package_legal_qa_submission(
            source,
            tmp_path / "submission.zip",
            empty_answer_policy="warn",
        )

    assert len(caught) == 1


def test_packager_rejects_hard_link_collision_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_json_bytes(_payload()))
    destination = tmp_path / "submission.zip"
    try:
        os.link(source, destination)
    except OSError:
        pytest.skip("hard links are unavailable on this platform")

    with pytest.raises(LegalQASubmissionError, match="paths must differ"):
        package_legal_qa_submission(source, destination)
    assert source.read_bytes() == _json_bytes(_payload())


def test_packager_rejects_bad_extensions_invalid_legacy_and_symlink(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "source.txt"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(LegalQASubmissionError, match="input path must use .json"):
        package_legal_qa_submission(wrong, tmp_path / "submission.zip")

    source = tmp_path / "source.json"
    source.write_bytes(_json_bytes(_legacy_array_payload()))
    with pytest.raises(LegalQASubmissionError, match="root must be an object"):
        package_legal_qa_submission(source, tmp_path / "submission.zip")

    source.write_bytes(_json_bytes(_payload()))
    with pytest.raises(LegalQASubmissionError, match="output path must use .zip"):
        package_legal_qa_submission(source, tmp_path / "submission.json")

    target = tmp_path / "real-source.json"
    target.write_bytes(_json_bytes(_payload()))
    linked = tmp_path / "linked-source.json"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(LegalQASubmissionError, match="symbolic link"):
        package_legal_qa_submission(linked, tmp_path / "linked-output.zip")


def test_missing_input_is_not_hidden_as_invalid_payload(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_legal_qa_submission(tmp_path / "missing.json")


def test_error_types_are_suitable_for_cli_boundaries() -> None:
    assert issubclass(LegalQASubmissionError, ValueError)
    assert issubclass(LegalQAEmptyAnswerWarning, UserWarning)
    assert issubclass(BadZipFile, Exception)
