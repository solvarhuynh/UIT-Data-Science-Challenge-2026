"""CPU-only tests for the unified Task 1 operational runner."""

from pathlib import Path

from scripts.task1.run_legal_ir_pipeline import _load_yaml, _stage_list, build_parser


def test_runner_defaults_to_all_and_dry_run_flags() -> None:
    args = build_parser().parse_args(
        ["--questions", "questions.json", "--dry-run", "--max-questions", "20"]
    )

    assert args.stage == "all"
    assert args.max_questions == 20
    assert args.dry_run is True
    assert _stage_list(args.stage)[0] == "preflight"


def test_dry_run_can_resolve_config_without_question_data() -> None:
    args = build_parser().parse_args(["--dry-run"])

    assert args.questions is None
    assert args.dry_run is True


def test_baseline_config_is_validated_and_not_experimental() -> None:
    config = _load_yaml(Path("configs/task1_baseline.yaml"))

    assert config["schema_version"] == "task1-baseline-v1"
    assert config["reranker"]["name"] == "BAAI/bge-reranker-v2-m3"
    assert config["candidate"]["final_documents"] == 5
