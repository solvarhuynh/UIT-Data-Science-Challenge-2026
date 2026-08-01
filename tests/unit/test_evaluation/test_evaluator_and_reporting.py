"""Tests for aligned evaluation, comparison, and stable report output."""

import json
from pathlib import Path

import pytest

import udsc2026.evaluation.reporting as reporting_module
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.evaluator import (
    compare_reports,
    evaluate_predictions,
    evaluate_retrieval,
    validate_rerank_candidate_pools,
)
from udsc2026.evaluation.loaders import load_benchmark, load_predictions
from udsc2026.evaluation.models import (
    BenchmarkSample,
    EvaluationComparison,
    PredictionSample,
)
from udsc2026.evaluation.reporting import (
    render_report_markdown,
    write_report,
    write_report_bundle,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "tv5"


def _hit(chunk_id: str) -> RetrievalHit:
    return RetrievalHit(chunk_id=chunk_id, doc_id="doc", text=f"text {chunk_id}")


def test_evaluate_retrieval_contract_and_optional_metrics() -> None:
    report = evaluate_retrieval(
        predictions=[[_hit("a"), _hit("b")], [_hit("wrong"), _hit("c")]],
        gold_chunk_ids=[["a"], ["c"]],
        k_values=[2, 1],
        predicted_answers=["quyền lợi", "hồ sơ đầy đủ"],
        reference_answers=["quyền lợi", "hồ sơ được lưu trữ đầy đủ"],
        latencies_ms=[10, 20],
        name="unit",
    )
    assert report.k_values == [1, 2]
    assert report.retrieval.mrr == pytest.approx(0.75)
    assert report.retrieval.recall_at_k == {1: 0.5, 2: 1.0}
    assert report.qa is not None
    assert report.qa.rouge_l == pytest.approx(19 / 22)
    assert report.latency is not None
    assert report.latency.mean_ms == 15


def test_evaluate_retrieval_rejects_partial_or_misaligned_optional_data() -> None:
    with pytest.raises(ValueError, match="provided together"):
        evaluate_retrieval([[_hit("a")]], [["a"]], [1], predicted_answers=["answer"])
    with pytest.raises(ValueError, match="exactly 1"):
        evaluate_retrieval([[_hit("a")]], [["a"]], [1], latencies_ms=[1, 2])
    with pytest.raises(ValueError, match="equal length"):
        evaluate_retrieval([[_hit("a")]], [["a"], ["b"]], [1])
    with pytest.raises(ValueError, match="unique"):
        evaluate_retrieval([[_hit("a")]], [["a"]], [1, 1])
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        evaluate_retrieval([[_hit("a"), _hit("a")]], [["a"]], [1])


def test_evaluate_predictions_aligns_by_question_id_not_file_order() -> None:
    benchmark = [
        BenchmarkSample(
            question_id="q1",
            question="Một?",
            answer="Một",
            gold_chunk_ids=["a"],
            difficulty="easy",
        ),
        BenchmarkSample(
            question_id="q2",
            question="Hai?",
            answer="Hai",
            gold_chunk_ids=["b"],
            difficulty="easy",
        ),
    ]
    predictions = [
        PredictionSample(question_id="q2", hits=[_hit("b")]),
        PredictionSample(question_id="q1", hits=[_hit("a")]),
    ]
    report = evaluate_predictions(benchmark, predictions, [1])
    assert report.retrieval.mrr == 1.0


def test_evaluate_predictions_rejects_id_and_coverage_errors() -> None:
    benchmark = [
        BenchmarkSample(
            question_id="q1",
            question="Một?",
            answer="Một",
            gold_chunk_ids=["a"],
            difficulty="easy",
        )
    ]
    with pytest.raises(ValueError, match="missing IDs.*unexpected IDs"):
        evaluate_predictions(
            benchmark,
            [PredictionSample(question_id="other", hits=[])],
            [1],
        )
    with pytest.raises(ValueError, match="answers must be supplied"):
        evaluate_predictions(
            benchmark
            + [
                BenchmarkSample(
                    question_id="q2",
                    question="Hai?",
                    answer="Hai",
                    gold_chunk_ids=["b"],
                    difficulty="easy",
                )
            ],
            [
                PredictionSample(question_id="q1", hits=[], answer="Một"),
                PredictionSample(question_id="q2", hits=[], answer=None),
            ],
            [1],
        )


def test_fixture_comparison_shows_expected_rerank_improvement() -> None:
    benchmark = load_benchmark(FIXTURE_DIR / "dev_benchmark.jsonl")
    before = evaluate_predictions(
        benchmark,
        load_predictions(FIXTURE_DIR / "predictions_before.json"),
        [1, 3, 5],
        name="before",
    )
    after = evaluate_predictions(
        benchmark,
        load_predictions(FIXTURE_DIR / "predictions_after.jsonl"),
        [1, 3, 5],
        name="after",
    )
    comparison = compare_reports(before, after)

    assert comparison.delta.mrr > 0
    assert comparison.delta.recall_at_k[1] > 0
    assert comparison.delta.rouge_l is not None
    assert comparison.delta.rouge_l > 0
    assert comparison.delta.mean_latency_ms is not None
    assert comparison.delta.mean_latency_ms > 0


def test_rerank_comparison_rejects_new_or_mutated_candidates() -> None:
    before = [
        PredictionSample(
            question_id="q1",
            hits=[RetrievalHit(chunk_id="a", doc_id="doc", text="A", hybrid_score=0.4)],
        )
    ]
    introduced = [
        PredictionSample(
            question_id="q1",
            hits=[RetrievalHit(chunk_id="b", doc_id="doc", text="B", rerank_score=0.9)],
        )
    ]
    with pytest.raises(ValueError, match="introduced candidate"):
        validate_rerank_candidate_pools(before, introduced)

    mutated = [
        PredictionSample(
            question_id="q1",
            hits=[
                RetrievalHit(
                    chunk_id="a",
                    doc_id="doc",
                    text="Changed",
                    hybrid_score=0.4,
                    rerank_score=0.9,
                )
            ],
        )
    ]
    with pytest.raises(ValueError, match="changed preserved fields"):
        validate_rerank_candidate_pools(before, mutated)


def test_compare_reports_rejects_incompatible_metric_coverage() -> None:
    plain = evaluate_retrieval([[_hit("a")]], [["a"]], [1])
    with_qa = evaluate_retrieval(
        [[_hit("a")]],
        [["a"]],
        [1],
        predicted_answers=["a"],
        reference_answers=["a"],
    )
    with pytest.raises(ValueError, match="both reports"):
        compare_reports(plain, with_qa)


def test_compare_reports_rejects_different_dataset_fingerprints() -> None:
    first = evaluate_retrieval([[_hit("a")]], [["a"]], [1])
    second = evaluate_retrieval([[_hit("b")]], [["b"]], [1])

    with pytest.raises(ValueError, match="benchmark fingerprint"):
        compare_reports(first, second)


def test_prediction_evaluation_fingerprints_complete_benchmark() -> None:
    benchmark = load_benchmark(FIXTURE_DIR / "dev_benchmark.jsonl")
    predictions = load_predictions(FIXTURE_DIR / "predictions_before.json")
    original = evaluate_predictions(benchmark, predictions, [1])
    changed_benchmark = list(benchmark)
    changed_benchmark[0] = changed_benchmark[0].model_copy(
        update={"question": changed_benchmark[0].question + " (đã sửa)"}
    )
    changed = evaluate_predictions(changed_benchmark, predictions, [1])

    assert original.dataset_fingerprint != changed.dataset_fingerprint
    with pytest.raises(ValueError, match="benchmark fingerprint"):
        compare_reports(original, changed)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("mrr", 0.0),
        ("recall_at_k", {1: 0.0}),
        ("rouge_l", 0.0),
        ("mean_latency_ms", 0.0),
        ("p95_latency_ms", 0.0),
    ],
)
def test_comparison_model_rejects_forged_deltas(
    field: str,
    forged_value: object,
) -> None:
    before = evaluate_retrieval(
        [[_hit("wrong"), _hit("gold")]],
        [["gold"]],
        [1],
        predicted_answers=["sai"],
        reference_answers=["đúng"],
        latencies_ms=[10.0],
    )
    after = evaluate_retrieval(
        [[_hit("gold"), _hit("wrong")]],
        [["gold"]],
        [1],
        predicted_answers=["đúng"],
        reference_answers=["đúng"],
        latencies_ms=[20.0],
    )
    comparison = compare_reports(before, after)
    forged_delta = comparison.delta.model_copy(update={field: forged_value})

    with pytest.raises(ValueError, match="after minus before"):
        EvaluationComparison(
            before=before,
            after=after,
            delta=forged_delta,
        )


def test_write_report_is_utf8_deterministic_and_round_trippable(
    tmp_path: Path,
) -> None:
    report = evaluate_retrieval(
        [[_hit("điều-1")]], [["điều-1"]], [1], name="Tiếng Việt"
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report(report, json_path, markdown_path)
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    write_report(report, json_path, markdown_path)

    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
    assert json.loads(first_json)["name"] == "Tiếng Việt"
    assert "# Evaluation: Tiếng Việt" in first_markdown.decode("utf-8")


def test_write_report_rejects_same_output_path(tmp_path: Path) -> None:
    report = evaluate_retrieval([[_hit("a")]], [["a"]], [1])
    with pytest.raises(ValueError, match="must be different"):
        write_report(report, tmp_path / "same", tmp_path / "same")


def test_write_report_rejects_directory_target(tmp_path: Path) -> None:
    report = evaluate_retrieval([[_hit("a")]], [["a"]], [1])
    directory_target = tmp_path / "report.json"
    directory_target.mkdir()

    with pytest.raises(ValueError, match="must not be a directory"):
        write_report(report, directory_target, tmp_path / "report.md")


def test_report_bundle_rolls_back_every_replaced_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = evaluate_retrieval([[_hit("before")]], [["before"]], [1], name="new")
    after = evaluate_retrieval([[_hit("after")]], [["after"]], [1], name="new")
    targets = [
        tmp_path / "before.json",
        tmp_path / "before.md",
        tmp_path / "after.json",
        tmp_path / "after.md",
    ]
    original_bytes = {
        target: f"old-{index}".encode() for index, target in enumerate(targets, start=1)
    }
    for target, content in original_bytes.items():
        target.write_bytes(content)

    real_replace = reporting_module.os.replace
    replace_calls = 0

    def fail_second_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("forced commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(reporting_module.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="forced commit failure"):
        write_report_bundle(
            [
                (before, targets[0], targets[1]),
                (after, targets[2], targets[3]),
            ]
        )

    assert {target: target.read_bytes() for target in targets} == original_bytes
    assert {path.name for path in tmp_path.iterdir()} == {
        target.name for target in targets
    }


def test_comparison_markdown_includes_delta_table() -> None:
    before = evaluate_retrieval([[_hit("x"), _hit("gold")]], [["gold"]], [1, 2])
    after = evaluate_retrieval([[_hit("gold"), _hit("x")]], [["gold"]], [1, 2])
    markdown = render_report_markdown(compare_reports(before, after))
    assert "| MRR | 0.500000 | 1.000000 | 0.500000 |" in markdown
    assert "| Recall@1 | 0.000000 | 1.000000 | 1.000000 |" in markdown
