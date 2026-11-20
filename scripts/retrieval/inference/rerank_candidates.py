"""Rerank dense retrieval candidates for organizer question mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pydantic import ValidationError  # noqa: E402

from udsc2026.evaluation.models import PredictionSample  # noqa: E402
from udsc2026.evaluation.reporting import write_run_manifest  # noqa: E402
from udsc2026.infrastructure.reranker import CrossEncoderClient  # noqa: E402
from udsc2026.infrastructure.modal_volume import (  # noqa: E402
    commit_mounted_volume_sync,
)
from udsc2026.retrieval.reranking import CrossEncoderReranker  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", default="models/aiteam-vietnamese-reranker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--max-length", type=_positive_int, default=1024)
    parser.add_argument("--candidate-k", type=_positive_int, default=200)
    parser.add_argument("--rerank-k", type=_positive_int, default=100)
    parser.add_argument("--save-every", type=_positive_int, default=100)
    parser.add_argument("--commit-modal-volume", action="store_true")
    parser.add_argument("--modal-volume-name", default="udsc")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("question source must be a non-empty ID mapping")
    questions: dict[str, str] = {}
    for raw_id, record in payload.items():
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("question IDs must be non-empty strings")
        if not isinstance(record, dict):
            raise ValueError(f"question {raw_id!r} must map to an object")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {raw_id!r} has no question text")
        questions[raw_id] = question.strip()
    return questions


def _load_existing_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                sample = PredictionSample.model_validate_json(line, strict=True)
            except ValidationError as exc:
                raise ValueError(f"invalid saved output at line {line_number}") from exc
            if sample.question_id in seen:
                raise ValueError(f"duplicate saved output ID {sample.question_id!r}")
            seen.add(sample.question_id)
            ids.append(sample.question_id)
    return ids


def _iter_candidates(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield PredictionSample.model_validate_json(line, strict=True)
            except ValidationError as exc:
                raise ValueError(f"invalid candidates at line {line_number}") from exc


def _serialize(sample: PredictionSample) -> str:
    return json.dumps(
        sample.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    if args.rerank_k > args.candidate_k:
        raise ValueError("--rerank-k must not exceed --candidate-k")
    if Path(args.model).resolve() != Path("models/aiteam-vietnamese-reranker").resolve():
        raise ValueError("only the selected AITeam reranker is enabled for production")
    try:
        candidate_manifest = json.loads(
            args.candidate_manifest.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate manifest is missing or invalid") from exc
    if not isinstance(candidate_manifest, dict):
        raise ValueError("candidate manifest must contain a JSON object")
    if candidate_manifest.get("complete") is not True:
        raise ValueError("candidate generation did not complete")
    if candidate_manifest.get("model_id") != "AITeamVN/Vietnamese_Embedding_v2":
        raise ValueError("candidates were not generated with the selected embedding")
    candidate_sha256 = _sha256(args.candidates)
    if candidate_manifest.get("output_sha256") != candidate_sha256:
        raise ValueError("candidate file does not match its generation manifest")
    if candidate_manifest.get("question_source_sha256") != _sha256(args.questions):
        raise ValueError("candidate questions do not match the supplied question file")

    model_manifest_path = Path("models/download_manifest.json")
    try:
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        selected_model = model_manifest["models"]["reranker"]
    except (KeyError, OSError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "selected reranker download manifest is missing or invalid"
        ) from exc
    if (
        not isinstance(selected_model, dict)
        or selected_model.get("repo_id") != "AITeamVN/Vietnamese_Reranker"
        or not isinstance(selected_model.get("resolved_revision"), str)
    ):
        raise ValueError("download manifest does not identify the selected reranker")

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    run_identity: dict[str, object] = {
        "schema_version": "reranker-run-v2",
        "questions_sha256": _sha256(args.questions),
        "candidate_predictions_sha256": candidate_sha256,
        "candidate_manifest_sha256": _sha256(args.candidate_manifest),
        "model": "AITeamVN/Vietnamese_Reranker",
        "model_revision": selected_model["resolved_revision"],
        "model_manifest_sha256": _sha256(model_manifest_path),
        "model_path": str(Path(args.model)),
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "fp16": args.fp16,
    }
    if args.output.resolve() == manifest_path.resolve():
        raise ValueError("reranked output and manifest must be different files")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {args.output}")
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"manifest exists; use a new output path: {manifest_path}")
    completed_ids = _load_existing_ids(args.output)
    if len(completed_ids) > len(questions):
        raise ValueError("saved output has more rows than the question source")
    if any(question_id not in questions for question_id in completed_ids):
        raise ValueError("saved output contains IDs outside the question source")
    previous_manifest: dict[str, object] | None = None
    if args.output.exists():
        if not manifest_path.is_file():
            raise ValueError("cannot resume without the original reranker manifest")
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot resume from an invalid reranker manifest") from exc
        if not isinstance(previous_manifest, dict):
            raise ValueError("reranker manifest must contain a JSON object")
        mismatches = {
            key: {"expected": value, "actual": previous_manifest.get(key)}
            for key, value in run_identity.items()
            if previous_manifest.get(key) != value
        }
        if mismatches and completed_ids:
            raise ValueError(f"reranker inputs changed; refusing to resume: {mismatches}")
        if previous_manifest.get("completed_count") != len(completed_ids):
            raise ValueError("reranked output rows do not match the saved checkpoint")
        if mismatches:
            print(
                "No reranked rows were saved; refreshing the empty checkpoint "
                f"for the current candidate manifest: {mismatches}",
                flush=True,
            )
        elif previous_manifest.get("complete") is True:
            if (
                len(completed_ids) != len(questions)
                or previous_manifest.get("output_sha256") != _sha256(args.output)
            ):
                raise ValueError("completed reranker output does not match its manifest")
            print(f"Reranked candidates are already complete: {args.output}", flush=True)
            return 0

    def save_manifest(*, complete: bool, completed_count: int, output_hash: str | None = None) -> None:
        write_run_manifest(
            {
                **run_identity,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "save_every": args.save_every,
                "completed_count": completed_count,
                "complete": complete,
                "output_sha256": output_hash,
            },
            manifest_path,
        )

    client = CrossEncoderClient(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=args.fp16,
    )
    reranker = CrossEncoderReranker(client)
    expected_prefix = iter(completed_ids)
    saved = 0
    seen_ids: set[str] = set()
    batch: list[PredictionSample] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(complete=False, completed_count=len(completed_ids))
    mode = "a" if args.resume else "x"
    with args.output.open(mode, encoding="utf-8", newline="\n") as output:
        for sample in _iter_candidates(args.candidates):
            if sample.question_id in seen_ids:
                raise ValueError(f"duplicate candidate ID: {sample.question_id}")
            seen_ids.add(sample.question_id)
            question = questions.get(sample.question_id)
            if question is None:
                raise ValueError(f"candidate ID missing from questions: {sample.question_id}")
            if saved < len(completed_ids):
                expected_id = next(expected_prefix)
                if sample.question_id != expected_id:
                    raise ValueError("saved output must be a prefix of candidate order")
                saved += 1
                continue
            if not sample.hits:
                raise ValueError(f"question {sample.question_id!r} has no candidates")
            candidates = sample.hits[: args.candidate_k]
            reranked = reranker.rerank(
                question, candidates[: args.rerank_k], min(args.rerank_k, len(candidates))
            )
            tail = [
                hit.model_copy(update={"rank": len(reranked) + offset})
                for offset, hit in enumerate(candidates[args.rerank_k :], start=1)
            ]
            ranked = reranked + tail
            batch.append(sample.model_copy(deep=True, update={"hits": ranked}))
            if len(batch) >= args.save_every:
                output.writelines(_serialize(item) for item in batch)
                output.flush()
                os.fsync(output.fileno())
                saved += len(batch)
                batch.clear()
                save_manifest(complete=False, completed_count=saved)
                if args.commit_modal_volume:
                    commit_mounted_volume_sync(args.modal_volume_name)
                print(f"saved={saved}", flush=True)
        if batch:
            output.writelines(_serialize(item) for item in batch)
            output.flush()
            os.fsync(output.fileno())
            saved += len(batch)
            save_manifest(complete=False, completed_count=saved)
            if args.commit_modal_volume:
                commit_mounted_volume_sync(args.modal_volume_name)
    if saved != len(questions) or seen_ids != set(questions):
        raise ValueError(
            f"candidate coverage mismatch: saved={saved}, questions={len(questions)}"
        )
    save_manifest(complete=True, completed_count=saved, output_hash=_sha256(args.output))
    if args.commit_modal_volume:
        commit_mounted_volume_sync(args.modal_volume_name)
    print(f"complete_questions={saved}")
    print(args.output)
    print(manifest_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"candidate reranking error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
