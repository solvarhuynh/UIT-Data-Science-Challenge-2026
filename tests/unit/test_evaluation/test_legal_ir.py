"""Tests for official and warm-up-specific LegalIR evaluation."""

import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.legal_ir import (
    LegalIRAggregate,
    LegalIREvaluationReport,
    LegalIRPrediction,
    LegalIRPredictionSet,
    LegalIRQueryDiagnostic,
    LegalIRReference,
    LegalIRReferenceSet,
    WarmupDataset,
    WarmupSample,
    evaluate_legal_ir,
    evaluate_warmup,
    evaluate_warmup_any_gold,
    legal_ir_precision,
    legal_ir_prediction_from_hits,
    legal_ir_recall,
    load_legal_ir_question_ids,
    load_warmup,
    normalize_legal_ir_matching_question,
    normalize_legal_ir_query,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _official_fixture() -> tuple[list[LegalIRReference], list[LegalIRPrediction]]:
    references = [
        LegalIRReference(id="q1", gold_documents=["gold-1", "gold-1b"]),
        LegalIRReference(id="q2", gold_documents=["gold-2"]),
        LegalIRReference(id="q3", gold_documents=["gold-3", "gold-3b"]),
        LegalIRReference(id="q4", gold_documents=["gold-4"]),
    ]
    predictions = [
        LegalIRPrediction(id="q1", documents=["gold-1", "a", "gold-1b"]),
        LegalIRPrediction(id="q2", documents=["a", "gold-2", "b"]),
        LegalIRPrediction(id="q3", documents=["a", "b", "c", "gold-3"]),
        LegalIRPrediction(id="q4", documents=["a", "b", "c"]),
    ]
    return references, predictions


def test_load_warmup_preserves_raw_query_and_normalizes_retrieval_query(
    tmp_path: Path,
) -> None:
    decomposed = unicodedata.normalize("NFD", "  Người\t lao động \n được nghỉ?  ")
    payload = {
        "001": {"question": decomposed, "answer": ["010", "011"]},
        "002": {"question": "Hỏi gì?", "answer": ["012"]},
    }
    path = tmp_path / "warmup.json"
    _write_json(path, payload)

    samples = load_warmup(path)

    assert [sample.id for sample in samples] == ["001", "002"]
    assert samples[0].raw_question == decomposed
    assert samples[0].question == "Người lao động được nghỉ?"
    assert unicodedata.is_normalized("NFC", samples[0].question)
    assert samples[0].gold_documents == ["010", "011"]


def test_question_manifest_loader_supports_phase_shapes_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.json"
    _write_json(
        mapping,
        {
            "001": {"question": "Câu một?"},
            "002": {"question": "Câu hai?", "answer": ["doc-2"]},
        },
    )
    assert load_legal_ir_question_ids(mapping) == ["001", "002"]

    unlabeled_phase = tmp_path / "unlabeled-phase.json"
    _write_json(
        unlabeled_phase,
        {
            "phase-001": {
                "question": "Question without released labels?",
                "answer": None,
            }
        },
    )
    assert load_legal_ir_question_ids(unlabeled_phase) == ["phase-001"]

    string_ids = tmp_path / "ids.json"
    _write_json(string_ids, ["002", "001"])
    assert load_legal_ir_question_ids(string_ids) == ["002", "001"]

    object_ids = tmp_path / "objects.json"
    _write_json(
        object_ids,
        [
            {"id": "001", "question": "Câu một?"},
            {"id": "002", "answer": ["doc-2"]},
        ],
    )
    assert load_legal_ir_question_ids(object_ids) == ["001", "002"]

    invalid_payloads = (
        [],
        ["001", {"id": "002"}],
        [{"id": "001"}, {"id": "001"}],
        {"001": {"answer": ["doc-1"]}},
        {"001": {"question": "Hỏi?", "answer": "doc-1"}},
        [{"id": "001", "extra": True}],
    )
    for index, payload in enumerate(invalid_payloads):
        invalid = tmp_path / f"invalid-manifest-{index}.json"
        _write_json(invalid, payload)
        with pytest.raises((TypeError, ValueError)):
            load_legal_ir_question_ids(invalid)


def test_warmup_root_mapping_is_strict_and_forbids_unknown_fields() -> None:
    validated = WarmupDataset.model_validate(
        {"q": {"question": "Câu hỏi?", "answer": ["doc"]}}, strict=True
    )
    assert validated.root["q"].answer == ["doc"]

    with pytest.raises(ValidationError, match="Extra inputs"):
        WarmupDataset.model_validate(
            {
                "q": {
                    "question": "Câu hỏi?",
                    "answer": ["doc"],
                    "unknown": True,
                }
            },
            strict=True,
        )
    with pytest.raises(ValidationError):
        WarmupDataset.model_validate(
            {"q": {"question": "Câu hỏi?", "answer": [123]}}, strict=True
        )
    with pytest.raises(ValidationError, match="must not be empty"):
        WarmupDataset.model_validate({}, strict=True)


@pytest.mark.parametrize(
    "payload",
    [
        {" q": {"question": "Hỏi?", "answer": ["doc"]}},
        {"q": {"question": "   ", "answer": ["doc"]}},
        {"q": {"question": "Hỏi?", "answer": []}},
        {"q": {"question": "Hỏi?", "answer": ["doc", "doc"]}},
        {"q": {"question": "Hỏi?", "answer": [" doc"]}},
    ],
)
def test_load_warmup_rejects_invalid_identifiers_questions_and_answers(
    tmp_path: Path,
    payload: object,
) -> None:
    path = tmp_path / "warmup.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="invalid warm-up schema"):
        load_warmup(path)


def test_load_warmup_rejects_duplicate_json_keys_and_non_standard_constants(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"q":{"question":"Một?","answer":["a"]},'
        '"q":{"question":"Hai?","answer":["b"]}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key: q"):
        load_warmup(duplicate)

    non_standard = tmp_path / "nan.json"
    non_standard.write_text('{"q":{"question":NaN,"answer":["a"]}}')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_warmup(non_standard)


def test_load_warmup_rejects_missing_or_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_warmup(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_warmup(malformed)


def test_query_normalization_is_strict_nfc_and_whitespace_aware() -> None:
    source = unicodedata.normalize("NFD", "  Bảo   hiểm\n xã hội  ")
    assert normalize_legal_ir_query(source) == "Bảo hiểm xã hội"
    with pytest.raises(TypeError, match="must be a string"):
        normalize_legal_ir_query(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unicode scalar"):
        normalize_legal_ir_query("question-\ud800")


def test_matching_normalization_matches_ensemble_contract() -> None:
    assert (
        normalize_legal_ir_matching_question("  ĐIỀU  5 — Người lao động? ")
        == "điều 5 người lao động"
    )


def test_prediction_and_reference_contracts_are_strict() -> None:
    with pytest.raises(ValidationError):
        LegalIRPrediction.model_validate({"id": 1, "documents": ["doc"]})
    with pytest.raises(ValidationError):
        LegalIRPrediction.model_validate({"id": "q", "documents": [1]})
    with pytest.raises(ValidationError, match="unique IDs"):
        LegalIRPrediction(id="q", documents=["doc", "doc"])
    with pytest.raises(ValidationError, match="surrounding whitespace"):
        LegalIRReference(id="q", gold_documents=[" doc "])
    with pytest.raises(ValidationError, match="control characters"):
        LegalIRPrediction(id="q\x00log", documents=[])
    with pytest.raises(ValidationError, match="control characters"):
        LegalIRPrediction(id="q", documents=["doc\u0085log"])
    with pytest.raises(ValidationError, match="Extra inputs"):
        LegalIRReference.model_validate(
            {"id": "q", "gold_documents": ["doc"], "answer": "extra"}
        )
    with pytest.raises(ValidationError):
        LegalIRReference.model_validate({"id": "q", "gold_documents": [1]})
    with pytest.raises(ValidationError, match="unique IDs"):
        LegalIRReference(id="q", gold_documents=["doc", "doc"])
    with pytest.raises(ValidationError, match="Unicode scalar"):
        LegalIRReference(id="q-\ud800", gold_documents=["doc"])

    legacy = LegalIRReference(id="q", gold_document="doc")
    assert legacy.gold_documents == ["doc"]
    assert legacy.model_dump() == {"id": "q", "gold_documents": ["doc"]}


def test_prediction_and_reference_root_sets_reject_duplicate_question_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate question ID"):
        LegalIRPredictionSet(
            root=[
                LegalIRPrediction(id="q", documents=[]),
                LegalIRPrediction(id="q", documents=["doc"]),
            ]
        )
    with pytest.raises(ValidationError, match="duplicate question ID"):
        LegalIRReferenceSet(
            root=[
                LegalIRReference(id="q", gold_documents=["a"]),
                LegalIRReference(id="q", gold_documents=["b"]),
            ]
        )
    with pytest.raises(ValidationError, match="must not be empty"):
        LegalIRPredictionSet(root=[])


def test_chunk_hits_collapse_to_unique_documents_in_best_chunk_order() -> None:
    hits = [
        RetrievalHit(
            chunk_id="a-1",
            doc_id="doc-a",
            text="A tốt nhất",
            rerank_score=0.9,
            rank=1,
        ),
        RetrievalHit(
            chunk_id="a-2",
            doc_id="doc-a",
            text="A thứ hai",
            rerank_score=0.8,
            rank=2,
        ),
        RetrievalHit(
            chunk_id="b-1",
            doc_id="doc-b",
            text="B",
            rerank_score=0.7,
            rank=3,
        ),
        RetrievalHit(
            chunk_id="c-1",
            doc_id="doc-c",
            text="C",
            rerank_score=0.6,
            rank=4,
        ),
    ]

    prediction = legal_ir_prediction_from_hits("q", hits, max_documents=2)

    assert prediction.documents == ["doc-a", "doc-b"]
    assert [hit.rank for hit in hits] == [1, 2, 3, 4]


def test_chunk_hit_adapter_rejects_ambiguous_handoff() -> None:
    hit = RetrievalHit(chunk_id="chunk", doc_id="doc", text="Nội dung", rank=2)
    with pytest.raises(ValueError, match="ranks must match"):
        legal_ir_prediction_from_hits("q", [hit])
    with pytest.raises(ValueError, match="positive integer"):
        legal_ir_prediction_from_hits("q", [], max_documents=0)
    with pytest.raises(TypeError, match=r"hits\[0\]"):
        legal_ir_prediction_from_hits("q", [object()])  # type: ignore[list-item]

    duplicate_chunk = hit.model_copy(update={"rank": 1})
    with pytest.raises(ValueError, match="duplicate chunk ID"):
        legal_ir_prediction_from_hits(
            "q",
            [duplicate_chunk, duplicate_chunk.model_copy(update={"rank": 2})],
        )


def test_official_metric_fixture_has_expected_macro_recall_and_precision() -> None:
    references, predictions = _official_fixture()

    report = evaluate_legal_ir(references, predictions)

    assert report.schema_version == "legal-ir-evaluation-v2"
    assert report.evaluation_mode == "official_set"
    assert report.metric_priority == ("recall", "precision")
    assert report.aggregate.sample_count == 4
    assert report.aggregate.multi_gold_sample_count == 2
    assert report.aggregate.recall == pytest.approx(0.625)
    assert report.aggregate.precision == pytest.approx(0.3125)
    assert [item.relevant_count for item in report.per_query] == [2, 1, 2, 1]
    assert [item.predicted_count for item in report.per_query] == [3, 3, 4, 3]
    assert [item.relevant_retrieved_count for item in report.per_query] == [2, 1, 1, 0]
    assert report.per_query[0].matched_documents == ["gold-1", "gold-1b"]
    assert report.per_query[2].missed_gold_documents == ["gold-3b"]
    assert report.per_query[2].false_positive_documents == ["a", "b", "c"]
    assert report.per_query[3].matched_documents == []


def test_official_metric_functions_use_sets_and_handle_empty_predictions() -> None:
    gold = ["gold-a", "gold-b"]
    assert legal_ir_recall(["x", "gold-a"], gold) == 0.5
    assert legal_ir_precision(["x", "gold-a"], gold) == 0.5
    assert legal_ir_recall(["gold-b", "x", "gold-a"], gold) == 1.0
    assert legal_ir_precision(["gold-b", "x", "gold-a"], gold) == pytest.approx(2 / 3)
    assert legal_ir_recall([], gold) == 0.0
    assert legal_ir_precision([], gold) == 0.0


def _official_scorer_reference(
    predictions: dict[str, dict[str, list[str]]], references: dict[str, list[str]]
) -> tuple[float, float]:
    """Call the published BTC scorer directly without modifying its source."""

    scorer_path = (
        Path(__file__).resolve().parents[3]
        / "docs/task1/Scoring-Program-Task-LegalIR/scoring.py"
    )
    spec = importlib.util.spec_from_file_location(
        "official_legal_ir_scorer", scorer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load official scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score = module.eval_retrieval(predictions, references)
    return float(score["recall"]), float(score["precision"])


def test_official_scorer_golden_parity_for_empty_overflow_and_multi_gold() -> None:
    references = {
        "q1": ["gold-1", "gold-1b"],
        "q2": ["gold-2"],
        "q3": ["gold-3"],
        "q4": ["gold-4", "gold-4b"],
    }
    predictions = {
        "q1": {"answer": ["gold-1"]},
        "q2": {"answer": []},
        # The gold document is present, but the official scorer gives zero.
        "q3": {"answer": ["a", "b", "c", "d", "e", "gold-3"]},
        "q4": {"answer": ["gold-4", "x", "gold-4b", "y", "z"]},
    }
    expected_recall, expected_precision = _official_scorer_reference(
        predictions, references
    )

    report = evaluate_legal_ir(
        [
            LegalIRReference(id=question_id, gold_documents=gold)
            for question_id, gold in references.items()
        ],
        [
            LegalIRPrediction(id=question_id, documents=record["answer"])
            for question_id, record in predictions.items()
        ],
    )

    assert report.aggregate.recall == pytest.approx(expected_recall)
    assert report.aggregate.precision == pytest.approx(expected_precision)
    assert report.per_query[1].recall == report.per_query[1].precision == 0.0
    assert report.per_query[2].recall == report.per_query[2].precision == 0.0
    assert report.per_query[2].predicted_documents == []


def test_six_predicted_documents_receive_zero_official_metrics() -> None:
    documents = ["a", "b", "c", "d", "e", "gold"]
    assert legal_ir_recall(documents, ["gold"]) == 0.0
    assert legal_ir_precision(documents, ["gold"]) == 0.0


@pytest.mark.parametrize(
    ("documents", "gold", "error"),
    [
        (["same", "same"], ["gold"], ValueError),
        ([""], ["gold"], ValueError),
        ([1], ["gold"], TypeError),
        ("gold", ["gold"], TypeError),
        (["gold"], [1], TypeError),
        (["gold"], [], ValueError),
        (["gold"], ["gold", "gold"], ValueError),
    ],
)
def test_official_metric_functions_reject_ambiguous_inputs(
    documents: object,
    gold: object,
    error: type[Exception],
) -> None:
    for metric in (legal_ir_recall, legal_ir_precision):
        with pytest.raises(error):
            metric(documents, gold)  # type: ignore[arg-type]


def test_official_evaluation_aligns_by_id_and_fingerprints_references() -> None:
    references, predictions = _official_fixture()
    original = evaluate_legal_ir(references, predictions)
    reordered_predictions = evaluate_legal_ir(references, list(reversed(predictions)))
    changed = evaluate_legal_ir(
        [
            references[0].model_copy(update={"gold_documents": ["changed"]}),
            *references[1:],
        ],
        predictions,
    )

    assert reordered_predictions == original
    assert len(original.dataset_fingerprint) == 64
    assert changed.dataset_fingerprint != original.dataset_fingerprint


def test_official_evaluation_rejects_duplicate_missing_and_unexpected_ids() -> None:
    references = [LegalIRReference(id="q1", gold_documents=["gold"])]
    with pytest.raises(ValueError, match="duplicate question ID in references"):
        evaluate_legal_ir(
            [references[0], references[0]],
            [LegalIRPrediction(id="q1", documents=[])],
        )
    with pytest.raises(ValueError, match="duplicate question ID in predictions"):
        evaluate_legal_ir(
            references,
            [
                LegalIRPrediction(id="q1", documents=[]),
                LegalIRPrediction(id="q1", documents=["other"]),
            ],
        )
    with pytest.raises(ValueError, match="missing IDs: q1.*unexpected IDs: q2"):
        evaluate_legal_ir(
            references,
            [LegalIRPrediction(id="q2", documents=[])],
        )
    with pytest.raises(ValueError, match="references must not be empty"):
        evaluate_legal_ir([], [])


def test_evaluation_revalidates_nested_lists_mutated_after_model_creation() -> None:
    prediction = LegalIRPrediction(id="q", documents=["a", "b"])
    prediction.documents.append("a")
    with pytest.raises(ValueError, match="unique IDs"):
        evaluate_legal_ir(
            [LegalIRReference(id="q", gold_documents=["gold"])],
            [prediction],
        )

    reference = LegalIRReference(id="q", gold_documents=["gold"])
    reference.gold_documents.append("gold")
    with pytest.raises(ValueError, match="unique IDs"):
        evaluate_legal_ir(
            [reference],
            [LegalIRPrediction(id="q", documents=[])],
        )

    sample = WarmupSample(
        id="q",
        raw_question="Câu hỏi?",
        question="Câu hỏi?",
        gold_documents=["gold"],
    )
    sample.gold_documents.append("gold")
    with pytest.raises(ValueError, match="unique IDs"):
        evaluate_warmup_any_gold(
            [sample],
            [LegalIRPrediction(id="q", documents=[])],
        )


def test_warmup_compatibility_entry_point_uses_official_set_metrics() -> None:
    sample = WarmupSample(
        id="q",
        raw_question="Câu hỏi?",
        question="Câu hỏi?",
        gold_documents=["first", "second"],
    )
    prediction = LegalIRPrediction(
        id="q",
        documents=["wrong", "second", "first"],
    )

    report = evaluate_warmup([sample], [prediction])
    legacy_report = evaluate_warmup_any_gold([sample], [prediction])

    assert report.evaluation_mode == "official_set"
    assert legacy_report == report
    assert report.aggregate.multi_gold_sample_count == 1
    assert report.aggregate.recall == 1.0
    assert report.aggregate.precision == pytest.approx(2 / 3)
    assert report.per_query[0].matched_documents == ["second", "first"]


def test_warmup_fingerprint_includes_preserved_question_source() -> None:
    first = WarmupSample(
        id="q",
        raw_question="Câu hỏi?",
        question="Câu hỏi?",
        gold_documents=["gold"],
    )
    second = WarmupSample(
        id="q",
        raw_question="  Câu  hỏi? ",
        question="Câu hỏi?",
        gold_documents=["gold"],
    )
    prediction = [LegalIRPrediction(id="q", documents=["gold"])]

    assert (
        evaluate_warmup_any_gold([first], prediction).dataset_fingerprint
        != evaluate_warmup_any_gold([second], prediction).dataset_fingerprint
    )


def test_report_contract_rejects_forged_aggregate() -> None:
    diagnostic = LegalIRQueryDiagnostic(
        id="q",
        gold_documents=["gold"],
        predicted_documents=["gold"],
        matched_documents=["gold"],
        missed_gold_documents=[],
        false_positive_documents=[],
        relevant_count=1,
        predicted_count=1,
        relevant_retrieved_count=1,
        recall=1.0,
        precision=1.0,
    )
    with pytest.raises(ValidationError, match="aggregate Recall"):
        LegalIREvaluationReport(
            evaluation_mode="official_set",
            dataset_fingerprint="0" * 64,
            aggregate=LegalIRAggregate(
                sample_count=1,
                recall=0.0,
                precision=1.0,
            ),
            per_query=[diagnostic],
        )


def test_warmup_sample_rejects_a_normalized_query_unrelated_to_raw() -> None:
    with pytest.raises(ValidationError, match="normalized form"):
        WarmupSample(
            id="q",
            raw_question="Câu hỏi gốc?",
            question="Câu hỏi khác?",
            gold_documents=["gold"],
        )
