"""Contract, determinism, and security tests for LegalIR submission artifacts."""

from __future__ import annotations

import json
import os
import stat
import warnings
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

from udsc2026.evaluation.legal_ir_submission import (
    SUBMISSION_MEMBER_NAME,
    LegalIRSubmissionError,
    LegalIRSubmissionItem,
    complete_legal_ir_rankings,
    load_legal_ir_submission,
    package_legal_ir_submission,
    validate_legal_ir_submission,
    write_legal_ir_submission,
    write_legal_ir_submission_json,
    write_legal_ir_submission_zip,
)


def _payload() -> dict[str, dict[str, list[str]]]:
    return {
        "q-2": {"answer": ["doc-3", "doc-1", "doc-2"]},
        "câu-hỏi-1": {"answer": ["doc-2", "doc-3", "doc-1"]},
    }


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _write_zip(
    path: Path,
    members: list[tuple[str | ZipInfo, bytes]],
    *,
    compression: int = ZIP_DEFLATED,
) -> None:
    with ZipFile(path, "w", compression=compression) as archive:
        for name, content in members:
            archive.writestr(name, content)


def test_model_and_payload_validation_preserve_rank_and_question_order() -> None:
    items = validate_legal_ir_submission(
        _payload(),
        expected_question_ids=["câu-hỏi-1", "q-2"],
        allowed_document_ids=["doc-1", "doc-2", "doc-3"],
        require_complete_ranking=True,
    )

    assert [item.id for item in items] == ["q-2", "câu-hỏi-1"]
    assert items[0].documents == ("doc-3", "doc-1", "doc-2")


def test_validated_model_is_deeply_immutable() -> None:
    item = LegalIRSubmissionItem(id="q-1", documents=["doc-1", "doc-2", "doc-3"])

    assert isinstance(item.documents, tuple)
    with pytest.raises(AttributeError):
        item.documents.append("doc-1")  # type: ignore[attr-defined]


def test_validated_items_can_be_revalidated_as_an_internal_handoff() -> None:
    first = validate_legal_ir_submission(_payload())
    second = validate_legal_ir_submission(first)

    assert second == first
    assert all(left is not right for left, right in zip(first, second))


@pytest.mark.parametrize("payload", [[], "{}", None, 1, (), set()])
def test_validation_requires_json_object_root(payload: object) -> None:
    with pytest.raises(LegalIRSubmissionError, match="root must be an object"):
        validate_legal_ir_submission(payload)


def test_validation_rejects_empty_submission() -> None:
    with pytest.raises(LegalIRSubmissionError, match="at least one question"):
        validate_legal_ir_submission({})


def test_validation_rejects_legacy_array_contract() -> None:
    legacy = [{"id": "q", "documents": ["1", "2", "3"]}]

    with pytest.raises(LegalIRSubmissionError, match="root must be an object"):
        validate_legal_ir_submission(legacy)


def test_json_and_zip_writers_reject_legacy_array_contract(tmp_path: Path) -> None:
    legacy = [{"id": "q", "documents": ["1", "2", "3"]}]

    with pytest.raises(LegalIRSubmissionError, match="root must be an object"):
        write_legal_ir_submission_json(legacy, tmp_path / "submission.json")
    with pytest.raises(LegalIRSubmissionError, match="root must be an object"):
        write_legal_ir_submission_zip(legacy, tmp_path / "submission.zip")
    assert not (tmp_path / "submission.json").exists()
    assert not (tmp_path / "submission.zip").exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"q": "not-an-object"}, "must be a JSON object"),
        ({"q": {"documents": ["1", "2", "3"]}}, "exactly.*answer"),
        ({"q": {}}, "exactly.*answer"),
        ({"q": {"answer": ["1", "2"]}}, "3 items"),
        (
            {"q": {"answer": ["1", "2", "3"], "scores": []}},
            "exactly.*answer",
        ),
        ({7: {"answer": ["1", "2", "3"]}}, "string keys"),
        ({"q": {"answer": [1, "2", "3"]}}, "valid string"),
        ({"q": {"answer": "123"}}, "JSON array"),
        ({"": {"answer": ["1", "2", "3"]}}, "must not be empty"),
        (
            {" q": {"answer": ["1", "2", "3"]}},
            "surrounding whitespace",
        ),
        (
            {"q": {"answer": ["1", "2 ", "3"]}},
            "surrounding whitespace",
        ),
        ({"q": {"answer": ["1", "2", ""]}}, "must not be empty"),
        ({"q\x00": {"answer": ["1", "2", "3"]}}, "control"),
        ({"q\nlog": {"answer": ["1", "2", "3"]}}, "control"),
        ({"q": {"answer": ["1", "2\tlog", "3"]}}, "control"),
        ({"q\u0085log": {"answer": ["1", "2", "3"]}}, "control"),
        (
            {"q\ud800": {"answer": ["1", "2", "3"]}},
            "Unicode scalar",
        ),
        ({"q": {"answer": ["1", "2", "1"]}}, "duplicate"),
    ],
)
def test_validation_rejects_malformed_items(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(LegalIRSubmissionError, match=message):
        validate_legal_ir_submission(payload)


def test_model_directly_uses_strict_strings_and_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LegalIRSubmissionItem.model_validate({"id": 1, "documents": ["1", "2", "3"]})


def test_decoded_payload_requires_answer_to_be_a_real_json_array() -> None:
    with pytest.raises(LegalIRSubmissionError, match="JSON array"):
        validate_legal_ir_submission({"q": {"answer": ("1", "2", "3")}})
    with pytest.raises(LegalIRSubmissionError, match="JSON array"):
        validate_legal_ir_submission({"q": {"answer": {"1", "2", "3"}}})
    with pytest.raises(ValidationError):
        LegalIRSubmissionItem.model_validate(
            {
                "id": "q",
                "documents": ["1", "2", "3"],
                "unknown": True,
            }
        )


def test_exact_question_coverage_reports_missing_and_unexpected_ids() -> None:
    with pytest.raises(LegalIRSubmissionError) as error:
        validate_legal_ir_submission(
            _payload(), expected_question_ids=["q-2", "required-but-missing"]
        )

    message = str(error.value)
    assert "missing" in message and "required-but-missing" in message
    assert "unexpected" in message and "câu-hỏi-1" in message


def test_optional_id_collections_are_strict_and_unique() -> None:
    with pytest.raises(TypeError, match="iterable of strings"):
        validate_legal_ir_submission(_payload(), expected_question_ids="q-2")
    with pytest.raises(TypeError, match="only strings"):
        validate_legal_ir_submission(_payload(), allowed_document_ids=["doc-1", 2])
    with pytest.raises(TypeError, match="only strings"):
        validate_legal_ir_submission(_payload(), allowed_document_ids={"doc-1", 2})
    with pytest.raises(LegalIRSubmissionError, match="must be unique"):
        validate_legal_ir_submission(
            _payload(), allowed_document_ids=["doc-1", "doc-1"]
        )


def test_validation_rejects_documents_outside_corpus() -> None:
    with pytest.raises(LegalIRSubmissionError, match="outside the corpus"):
        validate_legal_ir_submission(
            _payload(), allowed_document_ids=["doc-1", "doc-2"]
        )


def test_complete_ranking_validation_requires_corpus_and_all_ids() -> None:
    with pytest.raises(LegalIRSubmissionError, match="requires allowed_document_ids"):
        validate_legal_ir_submission(_payload(), require_complete_ranking=True)
    with pytest.raises(LegalIRSubmissionError, match="missing"):
        validate_legal_ir_submission(
            _payload(),
            allowed_document_ids=["doc-1", "doc-2", "doc-3", "doc-4"],
            require_complete_ranking=True,
        )


def test_completion_preserves_prefix_and_appends_in_corpus_order() -> None:
    payload = {"q": {"answer": ["d3", "d1", "d5"]}}
    completed = complete_legal_ir_rankings(payload, ["d1", "d2", "d3", "d4", "d5"])

    assert completed[0].documents == ("d3", "d1", "d5", "d2", "d4")


def test_completion_sorts_unordered_corpus_deterministically() -> None:
    payload = {"q": {"answer": ["d3", "d1", "d2"]}}

    completed = complete_legal_ir_rankings(payload, {"d4", "d2", "d3", "d1"})

    assert completed[0].documents == ("d3", "d1", "d2", "d4")


def test_completion_rejects_too_small_corpus_and_unknown_prefix_id() -> None:
    with pytest.raises(LegalIRSubmissionError, match="at least three"):
        complete_legal_ir_rankings(_payload(), ["doc-1", "doc-2"])
    with pytest.raises(LegalIRSubmissionError, match="outside the corpus"):
        complete_legal_ir_rankings(
            {"q": {"answer": ["d1", "d2", "unknown"]}},
            ["d1", "d2", "d3"],
        )


def test_json_writer_is_deterministic_compact_utf8_and_atomic(tmp_path: Path) -> None:
    output = tmp_path / "nested" / SUBMISSION_MEMBER_NAME

    write_legal_ir_submission_json(_payload(), output)
    first = output.read_bytes()
    write_legal_ir_submission_json(_payload(), output)

    assert output.read_bytes() == first
    assert b"c\xc3\xa2u-h\xe1\xbb\x8fi-1" in first
    assert not first.startswith(b"\xef\xbb\xbf")
    assert first.endswith(b"\n")
    assert b" " not in first
    assert json.loads(first) == _payload()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_json_writer_can_complete_every_ranking(tmp_path: Path) -> None:
    output = tmp_path / "submission.json"
    payload = {"q": {"answer": ["d3", "d1", "d2"]}}

    write_legal_ir_submission_json(
        payload,
        output,
        complete_with_document_ids=["d1", "d2", "d3", "d4"],
    )

    assert json.loads(output.read_text(encoding="utf-8"))["q"]["answer"] == [
        "d3",
        "d1",
        "d2",
        "d4",
    ]


def test_writer_rejects_conflicting_corpus_options(tmp_path: Path) -> None:
    with pytest.raises(LegalIRSubmissionError, match="either"):
        write_legal_ir_submission_json(
            _payload(),
            tmp_path / "submission.json",
            allowed_document_ids=["doc-1", "doc-2", "doc-3"],
            complete_with_document_ids=["doc-1", "doc-2", "doc-3"],
        )


def test_zip_writer_has_exact_member_stable_metadata_and_deterministic_bytes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "run-a.zip"
    second = tmp_path / "arbitrary-name.zip"

    write_legal_ir_submission_zip(_payload(), first)
    write_legal_ir_submission_zip(_payload(), second)

    assert first.read_bytes() == second.read_bytes()
    with ZipFile(first) as archive:
        assert archive.namelist() == [SUBMISSION_MEMBER_NAME]
        member = archive.infolist()[0]
        assert member.date_time == (1980, 1, 1, 0, 0, 0)
        assert member.compress_type == ZIP_DEFLATED
        assert stat.S_IFMT(member.external_attr >> 16) == stat.S_IFREG
        assert json.loads(archive.read(member)) == _payload()


def test_generic_writer_dispatches_json_and_zip(tmp_path: Path) -> None:
    json_path = tmp_path / "submission.JSON"
    zip_path = tmp_path / "anything.ZIP"

    write_legal_ir_submission(_payload(), json_path)
    write_legal_ir_submission(_payload(), zip_path)

    assert load_legal_ir_submission(json_path)[0].id == "q-2"
    assert load_legal_ir_submission(zip_path)[0].id == "q-2"
    with pytest.raises(LegalIRSubmissionError, match=".json or .zip"):
        write_legal_ir_submission(_payload(), tmp_path / "submission.txt")


def test_loader_applies_coverage_corpus_and_complete_checks(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    write_legal_ir_submission_json(_payload(), path)

    items = load_legal_ir_submission(
        path,
        expected_question_ids={"q-2", "câu-hỏi-1"},
        allowed_document_ids={"doc-1", "doc-2", "doc-3"},
        require_complete_ranking=True,
    )

    assert len(items) == 2


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        (b"not json", "not valid strict JSON"),
        (
            b'[{"id":"q","documents":["1","2","3"]}]',
            "root must be an object",
        ),
        (
            b'{"q":{"answer":["1","2","3"]},"q":{"answer":["4","5","6"]}}',
            "duplicate JSON object key",
        ),
        (
            b'{"q":{"answer":["1","2","3"],"answer":["4","5","6"]}}',
            "duplicate JSON object key",
        ),
        (
            b'{"q":{"answer":["1","2",NaN]}}',
            "non-standard JSON constant",
        ),
        (b"\xff", "valid UTF-8"),
        (b"\xef\xbb\xbf{}", "not valid strict JSON"),
    ],
)
def test_json_loader_rejects_non_strict_or_non_utf8_input(
    encoded: bytes,
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission.json"
    path.write_bytes(encoded)

    with pytest.raises(LegalIRSubmissionError, match=message):
        load_legal_ir_submission(path)


def test_loader_rejects_wrong_extension_directory_and_symlink(tmp_path: Path) -> None:
    wrong = tmp_path / "submission.txt"
    wrong.write_bytes(_json_bytes(_payload()))
    with pytest.raises(LegalIRSubmissionError, match=".json or .zip"):
        load_legal_ir_submission(wrong)

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(LegalIRSubmissionError, match="regular file"):
        load_legal_ir_submission(directory)

    target = tmp_path / "real.json"
    target.write_bytes(_json_bytes(_payload()))
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(LegalIRSubmissionError, match="symbolic link"):
        load_legal_ir_submission(link)


def test_loader_rejects_oversized_json_and_bad_size_options(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    path.write_bytes(_json_bytes(_payload()))

    with pytest.raises(LegalIRSubmissionError, match="safety limit"):
        load_legal_ir_submission(path, max_json_bytes=4)
    for bad_limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            load_legal_ir_submission(path, max_json_bytes=bad_limit)  # type: ignore[arg-type]


def test_writers_reject_oversized_canonical_json_and_zip(tmp_path: Path) -> None:
    json_output = tmp_path / "submission.json"
    zip_output = tmp_path / "submission.zip"

    with pytest.raises(LegalIRSubmissionError, match="safety limit"):
        write_legal_ir_submission_json(
            _payload(),
            json_output,
            max_json_bytes=4,
        )
    assert not json_output.exists()

    with pytest.raises(LegalIRSubmissionError, match="safety limit"):
        write_legal_ir_submission_zip(
            _payload(),
            zip_output,
            max_json_bytes=4,
        )
    assert not zip_output.exists()

    with pytest.raises(LegalIRSubmissionError, match="safety limit"):
        write_legal_ir_submission_zip(
            _payload(),
            zip_output,
            max_zip_bytes=4,
        )
    assert not zip_output.exists()


def test_zip_loader_never_extracts_and_accepts_stored_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "submission.zip"
    _write_zip(
        path,
        [(SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
        compression=ZIP_STORED,
    )

    def reject_extraction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("validator must not extract archives")

    monkeypatch.setattr(ZipFile, "extract", reject_extraction)
    monkeypatch.setattr(ZipFile, "extractall", reject_extraction)

    assert len(load_legal_ir_submission(path)) == 2
    assert not (tmp_path / SUBMISSION_MEMBER_NAME).exists()


@pytest.mark.parametrize(
    "member_name",
    ["nested/submission.json", "../submission.json", "/submission.json", "other.json"],
)
def test_zip_loader_rejects_wrong_or_zip_slip_member_names(
    member_name: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission.zip"
    _write_zip(path, [(member_name, _json_bytes(_payload()))])

    with pytest.raises(LegalIRSubmissionError, match="named exactly"):
        load_legal_ir_submission(path)


def test_zip_loader_rejects_extra_directory_and_duplicate_members(
    tmp_path: Path,
) -> None:
    extra = tmp_path / "extra.zip"
    _write_zip(
        extra,
        [("folder/", b""), (SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
    )
    with pytest.raises(LegalIRSubmissionError, match="exactly one"):
        load_legal_ir_submission(extra)

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _write_zip(
            duplicate,
            [
                (SUBMISSION_MEMBER_NAME, _json_bytes(_payload())),
                (SUBMISSION_MEMBER_NAME, _json_bytes(_payload())),
            ],
        )
    with pytest.raises(LegalIRSubmissionError, match="exactly one"):
        load_legal_ir_submission(duplicate)


def test_zip_loader_rejects_symlink_member(tmp_path: Path) -> None:
    path = tmp_path / "submission.zip"
    member = ZipInfo(SUBMISSION_MEMBER_NAME)
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    _write_zip(path, [(member, _json_bytes(_payload()))])

    with pytest.raises(LegalIRSubmissionError, match="regular file"):
        load_legal_ir_submission(path)


def test_zip_loader_rejects_unsupported_compression(tmp_path: Path) -> None:
    path = tmp_path / "submission.zip"
    try:
        _write_zip(
            path,
            [(SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
            compression=ZIP_BZIP2,
        )
    except RuntimeError:
        pytest.skip("bzip2 ZIP support is unavailable")

    with pytest.raises(LegalIRSubmissionError, match="unsupported compression"):
        load_legal_ir_submission(path)


def test_zip_loader_rejects_invalid_and_oversized_archives(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not-a-zip")
    with pytest.raises(LegalIRSubmissionError, match="invalid submission ZIP"):
        load_legal_ir_submission(invalid)

    valid = tmp_path / "valid.zip"
    _write_zip(valid, [(SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))])
    with pytest.raises(LegalIRSubmissionError, match="safety limit"):
        load_legal_ir_submission(valid, max_zip_bytes=4)
    with pytest.raises(LegalIRSubmissionError, match="safety limit"):
        load_legal_ir_submission(valid, max_json_bytes=4)


def test_writer_rejects_directory_symlink_and_bad_extension(tmp_path: Path) -> None:
    directory = tmp_path / "submission.json"
    directory.mkdir()
    with pytest.raises(LegalIRSubmissionError, match="directory"):
        write_legal_ir_submission_json(_payload(), directory)

    target = tmp_path / "target.zip"
    link = tmp_path / "linked.zip"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(LegalIRSubmissionError, match="symbolic link"):
        write_legal_ir_submission_zip(_payload(), link)

    with pytest.raises(LegalIRSubmissionError, match=".json"):
        write_legal_ir_submission_json(_payload(), tmp_path / "wrong.zip")
    with pytest.raises(LegalIRSubmissionError, match=".zip"):
        write_legal_ir_submission_zip(_payload(), tmp_path / "wrong.json")


def test_atomic_writer_preserves_existing_file_and_cleans_temp_on_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "submission.json"
    output.write_bytes(b"previous")

    def fail_replace(source: str, destination: Path) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "udsc2026.evaluation.legal_ir_submission.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="simulated"):
        write_legal_ir_submission_json(_payload(), output)

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".submission.json.*.tmp"))


def test_packager_validates_and_canonicalizes_json(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(_payload(), ensure_ascii=False, indent=4), encoding="utf-8"
    )
    output = tmp_path / "team-run.zip"

    package_legal_ir_submission(
        source,
        output,
        expected_question_ids=["q-2", "câu-hỏi-1"],
        allowed_document_ids=["doc-1", "doc-2", "doc-3"],
        require_complete_ranking=True,
    )

    assert load_legal_ir_submission(output)[1].id == "câu-hỏi-1"
    with ZipFile(output) as archive:
        encoded = archive.read(SUBMISSION_MEMBER_NAME)
    assert encoded == _json_bytes(_payload()) + b"\n"


def test_packager_accepts_one_shot_constraint_iterators(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(_json_bytes(_payload()))
    output = tmp_path / "submission.zip"

    package_legal_ir_submission(
        source,
        output,
        expected_question_ids=iter(["q-2", "câu-hỏi-1"]),
        allowed_document_ids=iter(["doc-1", "doc-2", "doc-3"]),
    )

    assert len(load_legal_ir_submission(output)) == 2


def test_packager_rejects_hard_link_collision_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "submission.json"
    source.write_bytes(_json_bytes(_payload()))
    destination = tmp_path / "submission.zip"
    try:
        os.link(source, destination)
    except OSError:
        pytest.skip("hard links are unavailable on this platform")

    with pytest.raises(LegalIRSubmissionError, match="paths must differ"):
        package_legal_ir_submission(source, destination)


def test_packager_rejects_wrong_input_extension_and_invalid_json(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "source.txt"
    wrong.write_text("[]", encoding="utf-8")
    with pytest.raises(LegalIRSubmissionError, match="input path must use .json"):
        package_legal_ir_submission(wrong, tmp_path / "output.zip")

    invalid = tmp_path / "source.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(LegalIRSubmissionError, match="at least one question"):
        package_legal_ir_submission(invalid, tmp_path / "output.zip")


def test_missing_input_is_not_silently_treated_as_invalid_payload(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_legal_ir_submission(tmp_path / "missing.json")


def test_corrupt_zip_member_crc_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "submission.zip"
    _write_zip(
        path,
        [(SUBMISSION_MEMBER_NAME, _json_bytes(_payload()))],
        compression=ZIP_STORED,
    )
    encoded = bytearray(path.read_bytes())
    marker = _json_bytes(_payload())
    offset = encoded.find(marker)
    assert offset >= 0
    encoded[offset] ^= 1
    path.write_bytes(encoded)

    with pytest.raises(LegalIRSubmissionError, match="invalid submission ZIP"):
        load_legal_ir_submission(path)


def test_frozen_item_prevents_top_level_assignment() -> None:
    item = LegalIRSubmissionItem(id="q", documents=["doc-1", "doc-2", "doc-3"])
    with pytest.raises(ValidationError, match="frozen"):
        item.id = "changed"


def test_bad_zip_exception_is_a_value_error_for_cli_boundaries() -> None:
    assert issubclass(LegalIRSubmissionError, ValueError)
    assert issubclass(BadZipFile, Exception)
