"""End-to-end subprocess checks for the two model-independent CLIs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "tv5"


def test_evaluate_cli_writes_before_after_and_comparison(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--benchmark",
            str(FIXTURE_DIR / "dev_benchmark.jsonl"),
            "--before",
            str(FIXTURE_DIR / "predictions_before.json"),
            "--after",
            str(FIXTURE_DIR / "predictions_after.jsonl"),
            "--output-dir",
            str(tmp_path),
            "--k",
            "1",
            "3",
            "5",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {path.name for path in tmp_path.iterdir()} == {
        "before.json",
        "before.md",
        "after.json",
        "after.md",
        "comparison.json",
        "comparison.md",
    }
    comparison = json.loads((tmp_path / "comparison.json").read_text("utf-8"))
    assert comparison["report_type"] == "comparison"
    assert comparison["delta"]["mrr"] > 0


def test_evaluate_cli_single_run_uses_report_stem(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--benchmark",
            str(FIXTURE_DIR / "dev_benchmark.jsonl"),
            "--before",
            str(FIXTURE_DIR / "predictions_before.json"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_evaluate_cli_returns_two_for_invalid_input(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--benchmark",
            str(tmp_path / "missing.json"),
            "--before",
            str(FIXTURE_DIR / "predictions_before.json"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "evaluation error:" in result.stderr


@pytest.mark.parametrize(
    ("colliding_role", "output_name"),
    [
        ("benchmark", "before.json"),
        ("before", "after.json"),
        ("after", "comparison.json"),
    ],
)
def test_evaluate_cli_refuses_to_overwrite_any_input(
    tmp_path: Path,
    colliding_role: str,
    output_name: str,
) -> None:
    collision = tmp_path / output_name
    collision.write_text("input must remain untouched", encoding="utf-8")
    inputs = {
        "benchmark": FIXTURE_DIR / "dev_benchmark.jsonl",
        "before": FIXTURE_DIR / "predictions_before.json",
        "after": FIXTURE_DIR / "predictions_after.jsonl",
    }
    inputs[colliding_role] = collision

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--benchmark",
            str(inputs["benchmark"]),
            "--before",
            str(inputs["before"]),
            "--after",
            str(inputs["after"]),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must not overwrite an input file" in result.stderr
    assert collision.read_text(encoding="utf-8") == "input must remain untouched"
    assert list(tmp_path.iterdir()) == [collision]


def test_submission_cli_writes_configured_csv(tmp_path: Path) -> None:
    output = tmp_path / "submission.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "write_submission.py"),
            "--input",
            str(FIXTURE_DIR / "qa_responses.jsonl"),
            "--output",
            str(output),
            "--schema",
            str(FIXTURE_DIR / "submission_schema.json"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "citation_count" in output.read_text(encoding="utf-8-sig").splitlines()[0]


def test_submission_cli_returns_two_for_invalid_schema(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text('{"unknown": true}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "write_submission.py"),
            "--input",
            str(FIXTURE_DIR / "qa_responses.jsonl"),
            "--output",
            str(tmp_path / "submission.csv"),
            "--schema",
            str(schema),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "submission error:" in result.stderr


def test_submission_cli_refuses_symlink_collision_with_input(tmp_path: Path) -> None:
    input_link = tmp_path / "responses.jsonl"
    output = tmp_path / "submission.csv"
    output.write_text(
        (FIXTURE_DIR / "qa_responses.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    try:
        input_link.symlink_to(output)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "write_submission.py"),
            "--input",
            str(input_link),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must not overwrite an input file" in result.stderr
