"""Validate a downloaded Task 1 Beam result before publishing it locally."""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cloud.beam_source_contract import (  # noqa: E402
    compute_source_tree_sha256,
)

EXPECTED_ARCHIVE_SHA256 = (
    "21dda96ecab20686ef81b85d1f6813d8c6fa0d80f892e43c61604a3f90469921"
)
VOLUME_ROOT = PurePosixPath("/mnt/task1")
SMOKE_ROOT_BASE = VOLUME_ROOT / (
    "workspace/artifacts/task1/models/bge_reranker_finetune/beam_smoke"
)
FULL_ROOT_BASE = VOLUME_ROOT / (
    "workspace/artifacts/task1/models/bge_reranker_finetune/beam_full_oof"
)
PUBLIC_ROOT_BASE = VOLUME_ROOT / "workspace/artifacts/task1/beam_finetuned_oof_public"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_COUNT = 64
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_NESTED_SUBMISSION_BYTES = 64 * 1024 * 1024


def _load_json_value(bundle: zipfile.ZipFile, name: str) -> Any:
    return json.loads(bundle.read(name).decode("utf-8-sig"))


def _load_json_member(bundle: zipfile.ZipFile, name: str) -> dict[str, Any]:
    payload = _load_json_value(bundle, name)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _require_sha256(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"Beam summary has invalid {field}")
    return value


def _require_versioned_root(
    payload: dict[str, Any], field: str, base: PurePosixPath, contract: str
) -> PurePosixPath:
    value = payload.get(field)
    expected = base / contract[:24]
    if value != str(expected):
        raise ValueError(f"Beam summary {field} must equal {expected}")
    return expected


def _archive_name(path: PurePosixPath) -> str:
    return path.relative_to(VOLUME_ROOT).as_posix()


def _validate_member_names(names: list[str]) -> set[str]:
    if len(names) != len(set(names)):
        raise ValueError("Beam result contains duplicate ZIP member names")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Beam result contains an unsafe ZIP member: {name!r}")
    return set(names)


def _validate_nested_submission(content: bytes) -> None:
    if len(content) > MAX_NESTED_SUBMISSION_BYTES:
        raise ValueError("nested LegalIR submission ZIP exceeds the safety limit")
    with zipfile.ZipFile(io.BytesIO(content)) as submission:
        if submission.testzip() is not None:
            raise ValueError("nested LegalIR submission ZIP is corrupt")
        if submission.namelist() != ["submission.json"]:
            raise ValueError(
                "nested LegalIR submission must contain only submission.json"
            )
        payload = json.loads(submission.read("submission.json").decode("utf-8-sig"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError("nested LegalIR submission.json must be a nonempty object")


def validate_result_archive(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Beam result is missing or empty: {path}")
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Beam result ZIP exceeds the compressed-size safety limit")
    with zipfile.ZipFile(path) as bundle:
        members = bundle.infolist()
        if len(members) > MAX_MEMBER_COUNT:
            raise ValueError("Beam result ZIP contains too many members")
        if any(member.file_size > MAX_MEMBER_BYTES for member in members):
            raise ValueError("Beam result ZIP member exceeds the safety limit")
        if sum(member.file_size for member in members) > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError(
                "Beam result ZIP exceeds the uncompressed-size safety limit"
            )
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise ValueError(f"Beam result contains a corrupt entry: {corrupt}")
        names = _validate_member_names(bundle.namelist())
        summary_name = "results/full_run_summary.json"
        if summary_name not in names:
            raise ValueError(f"Beam result is missing {summary_name}")
        summary = _load_json_member(bundle, summary_name)

        if summary.get("schema_version") != "task1-p13-beam-full-v1":
            raise ValueError("Beam summary schema_version is invalid")
        if summary.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
            raise ValueError("Beam summary input archive SHA-256 is invalid")
        source_sha = _require_sha256(summary, "source_tree_sha256")
        local_source_sha = compute_source_tree_sha256(PROJECT_ROOT)
        if source_sha != local_source_sha:
            raise ValueError(
                "Beam result source hash does not match the current local code: "
                f"expected {local_source_sha}, got {source_sha}"
            )
        smoke_contract = _require_sha256(summary, "smoke_run_contract_sha256")
        full_contract = _require_sha256(summary, "run_contract_sha256")
        full_root = _require_versioned_root(
            summary, "full_root", FULL_ROOT_BASE, full_contract
        )
        public_root = _require_versioned_root(
            summary, "public_root", PUBLIC_ROOT_BASE, full_contract
        )
        if summary.get("download_path") != (
            "beam://udsc-task1-p13/results/task1_p13_beam_result.zip"
        ):
            raise ValueError("Beam summary download_path is invalid")
        dependencies = summary.get("dependency_versions")
        if (
            not isinstance(dependencies, dict)
            or not dependencies
            or not all(
                isinstance(key, str) and isinstance(value, str) and value
                for key, value in dependencies.items()
            )
        ):
            raise ValueError("Beam summary dependency_versions are invalid")

        full_relative = _archive_name(full_root)
        smoke_name = f"results/smoke/{smoke_contract}.json"
        required = {
            summary_name,
            smoke_name,
            f"{full_relative}/final_decision.json",
            f"{full_relative}/comparison.json",
            f"{full_relative}/comparison.md",
            f"{full_relative}/training_manifest.json",
            f"{full_relative}/oof_predictions.jsonl",
            *(f"{full_relative}/fold_{fold}/metrics.json" for fold in range(5)),
        }
        status = summary.get("status")
        if status == "PUBLIC_CANDIDATE_READY":
            public_relative = _archive_name(public_root)
            required.update(
                {
                    f"{public_relative}/predictions.json",
                    f"{public_relative}/report.json",
                    f"{public_relative}/submission.zip",
                }
            )
        elif status != "OOF_REJECTED":
            raise ValueError(f"Beam result has invalid completion status: {status!r}")
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Beam result is missing required artifacts: {missing}")

        decision_name = f"{full_relative}/final_decision.json"
        decision = _load_json_member(bundle, decision_name)
        if summary.get("decision") != decision:
            raise ValueError("Beam summary decision does not match final_decision.json")
        if status == "PUBLIC_CANDIDATE_READY":
            if not (
                summary.get("promotable") is True
                and decision.get("status") == "PROMOTE_CANDIDATE"
                and decision.get("promotable") is True
                and decision.get("complete_oof") is True
            ):
                raise ValueError("promoted Beam result has inconsistent gate decision")
        elif not (
            summary.get("promotable") is False
            and decision.get("status") == "REJECT_CHECKPOINT"
            and decision.get("promotable") is False
            and decision.get("complete_oof") is True
        ):
            raise ValueError("rejected Beam result has inconsistent gate decision")

        smoke = _load_json_member(bundle, smoke_name)
        expected_smoke_root = SMOKE_ROOT_BASE / smoke_contract[:24]
        if not (
            smoke.get("status") == "SMOKE_PASS"
            and smoke.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
            and smoke.get("source_tree_sha256") == source_sha
            and smoke.get("run_contract_sha256") == smoke_contract
            and smoke.get("smoke_root") == str(expected_smoke_root)
        ):
            raise ValueError("Beam smoke marker does not match the full run contract")

        for fold in range(5):
            metrics = _load_json_member(
                bundle, f"{full_relative}/fold_{fold}/metrics.json"
            )
            if not (
                metrics.get("status") == "COMPLETE"
                and metrics.get("fold") == fold
                and isinstance(metrics.get("validation_query_count"), int)
                and metrics["validation_query_count"] > 0
            ):
                raise ValueError(f"Beam fold {fold} metrics are incomplete")
        with bundle.open(f"{full_relative}/oof_predictions.jsonl") as stream:
            first_byte = stream.read(1)
        if not first_byte:
            raise ValueError("Beam OOF predictions are empty")

        if status == "PUBLIC_CANDIDATE_READY":
            public_relative = _archive_name(public_root)
            predictions = _load_json_value(
                bundle, f"{public_relative}/predictions.json"
            )
            if not isinstance(predictions, list) or not predictions:
                raise ValueError("Beam public predictions must be a nonempty array")
            _load_json_member(bundle, f"{public_relative}/report.json")
            _validate_nested_submission(
                bundle.read(f"{public_relative}/submission.zip")
            )
    return summary


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_beam_result.py RESULT.zip", file=sys.stderr)
        return 2
    path = Path(sys.argv[1]).resolve()
    try:
        summary = validate_result_archive(path)
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"BEAM RESULT INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"BEAM RESULT VALID: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
