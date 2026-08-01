"""Tests for strict benchmark/prediction/QA file boundaries."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from udsc2026.evaluation.loaders import (
    load_benchmark,
    load_predictions,
    load_qa_responses,
)
from udsc2026.evaluation.models import BenchmarkSample, PredictionSample

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "tv5"


def test_loads_jsonl_benchmark_and_both_prediction_formats() -> None:
    benchmark = load_benchmark(FIXTURE_DIR / "dev_benchmark.jsonl")
    before = load_predictions(FIXTURE_DIR / "predictions_before.json")
    after = load_predictions(FIXTURE_DIR / "predictions_after.jsonl")

    assert len(benchmark) == len(before) == len(after) == 12
    assert benchmark[0].question == "Bộ luật Lao động điều chỉnh quan hệ nào?"
    assert before[0].hits[0].hybrid_score == pytest.approx(0.91)
    assert after[0].hits[0].rerank_score == pytest.approx(0.98)


def test_loads_strict_qa_response_fixture() -> None:
    results = load_qa_responses(FIXTURE_DIR / "qa_responses.jsonl")
    assert len(results) == 2
    assert results[0].citations[0].is_verified is True
    assert results[0].trace_id == "tv5_dev_002"


def test_benchmark_model_rejects_duplicate_gold_ids_and_unknown_fields() -> None:
    base = {
        "question_id": "q1",
        "question": "Câu hỏi?",
        "answer": "Câu trả lời.",
        "gold_chunk_ids": ["a", "a"],
        "difficulty": "easy",
    }
    with pytest.raises(ValidationError, match="unique"):
        BenchmarkSample.model_validate(base)
    base["gold_chunk_ids"] = ["a"]
    base["unknown"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        BenchmarkSample.model_validate(base)


def test_prediction_model_allows_empty_hits_but_rejects_bad_latency() -> None:
    prediction = PredictionSample(question_id="q1", hits=[])
    assert prediction.hits == []
    with pytest.raises(ValidationError):
        PredictionSample(question_id="q1", latency_ms=-0.1)


def test_prediction_model_rejects_blank_or_duplicate_hit_ids() -> None:
    duplicate_hits = [
        {"chunk_id": "same", "doc_id": "doc", "text": "one"},
        {"chunk_id": "same", "doc_id": "doc", "text": "two"},
    ]
    with pytest.raises(ValidationError, match="must be unique"):
        PredictionSample.model_validate({"question_id": "q1", "hits": duplicate_hits})
    with pytest.raises(ValidationError, match="at least 1 character"):
        PredictionSample.model_validate(
            {
                "question_id": "q1",
                "hits": [{"chunk_id": " ", "doc_id": "doc", "text": "text"}],
            }
        )


def test_prediction_model_rejects_rank_that_disagrees_with_list_order() -> None:
    hits = [
        {
            "chunk_id": "gold",
            "doc_id": "doc",
            "text": "Gold but explicitly ranked second.",
            "rank": 2,
        },
        {
            "chunk_id": "wrong",
            "doc_id": "doc",
            "text": "Wrong but explicitly ranked first.",
            "rank": 1,
        },
    ]

    with pytest.raises(ValidationError, match="one-based list positions"):
        PredictionSample.model_validate({"question_id": "q1", "hits": hits})


def test_loader_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    record = {
        "question_id": "duplicate",
        "question": "Câu hỏi?",
        "answer": "Câu trả lời.",
        "gold_chunk_ids": ["chunk"],
        "difficulty": "easy",
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([record, record]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question_id"):
        load_benchmark(path)


def test_loader_rejects_malformed_jsonl_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"question_id":"q1"}\n{not-json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_predictions(path)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_loader_rejects_non_standard_numeric_constants(
    tmp_path: Path,
    suffix: str,
    constant: str,
) -> None:
    record = (
        '{"question_id":"q1","question":"Câu hỏi?","answer":"Câu trả lời.",'
        '"gold_chunk_ids":["chunk"],"difficulty":"easy",'
        f'"metadata":{{"invalid":{constant}}}}}'
    )
    path = tmp_path / f"benchmark{suffix}"
    payload = f"[{record}]" if suffix == ".json" else f"{record}\n"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_benchmark(path)


def test_loader_rejects_unknown_wrapper_keys_and_empty_files(tmp_path: Path) -> None:
    wrong_wrapper = tmp_path / "wrong.json"
    wrong_wrapper.write_text('{"items": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="only the key 'samples'"):
        load_benchmark(wrong_wrapper)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain any records"):
        load_benchmark(empty)


def test_loader_rejects_unsupported_extension_and_missing_file(
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "benchmark.txt"
    text_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported input format"):
        load_benchmark(text_file)
    with pytest.raises(FileNotFoundError):
        load_benchmark(tmp_path / "missing.json")


def test_qa_loader_rejects_unknown_fields_ignored_by_shared_contract(
    tmp_path: Path,
) -> None:
    record = {
        "answer": "Trả lời",
        "citations": [],
        "used_prompt_version": "v1",
        "retrieval_hits": [],
        "unexpected": "must not be ignored",
    }
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([record], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected fields"):
        load_qa_responses(path)
