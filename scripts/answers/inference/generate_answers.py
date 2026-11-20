"""Generate checkpointed LegalQA answers from precomputed retrieval hits."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.models import PredictionSample  # noqa: E402
from udsc2026.evaluation.reporting import write_run_manifest  # noqa: E402
from udsc2026.infrastructure.llm.client import LLMClient  # noqa: E402
from udsc2026.infrastructure.llm.config import load_llm_config  # noqa: E402
from udsc2026.answer_generation.engine import QAEngine  # noqa: E402
from udsc2026.retrieval.parent_context import (  # noqa: E402
    JsonlParentStore,
    ParentContextExpander,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Atomically checkpoint answers and diagnostics after this many new rows.",
    )
    parser.add_argument(
        "--commit-modal-volume",
        action="store_true",
        help="Commit each completed checkpoint to the named Modal Volume.",
    )
    parser.add_argument(
        "--modal-volume-name",
        default="udsc",
        help="Modal Volume name used with --commit-modal-volume.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt-version", default="answer_system")
    parser.add_argument("--rag-template", default="answer_context")
    parser.add_argument("--parent-max-tokens", type=int, default=384)
    parser.add_argument("--parent-total-tokens", type=int, default=1152)
    parser.add_argument(
        "--parents-dir",
        type=Path,
        default=Path("data/processed/parents"),
        help="Directory containing the parent passages used to expand retrieved hits.",
    )
    parser.add_argument(
        "--parent-archive",
        type=Path,
        help="Source archive whose extracted parent passages are used for generation.",
    )
    parser.add_argument("--no-parent-context", action="store_true")
    return parser


def _load_questions(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty organizer JSON object")
    questions: list[tuple[str, str]] = []
    for question_id, record in payload.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question IDs must be non-empty strings")
        if not isinstance(record, dict):
            raise ValueError(f"question {question_id!r} must be an object")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no non-empty text")
        questions.append((question_id, question))
    return questions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths_collide(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _load_contexts(path: Path) -> dict[str, PredictionSample]:
    contexts: dict[str, PredictionSample] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            sample = PredictionSample.model_validate(json.loads(line))
            if sample.question_id in contexts:
                raise ValueError(
                    f"duplicate question {sample.question_id!r} at line {line_number}"
                )
            contexts[sample.question_id] = sample
    if not contexts:
        raise ValueError("contexts JSONL must not be empty")
    return contexts


def _load_existing(path: Path, *, resume: bool) -> list[dict[str, str]]:
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError(f"output exists; use --resume: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("existing output must be a JSON array")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"id", "answer"}:
            raise ValueError("existing predictions must contain exactly id and answer")
        question_id = item["id"]
        answer = item["answer"]
        if (
            not isinstance(question_id, str)
            or not question_id.strip()
            or question_id in seen
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError("existing predictions contain an invalid item")
        seen.add(question_id)
        result.append({"id": question_id, "answer": answer})
    return result


def _load_diagnostics(path: Path, *, resume: bool) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not resume:
        raise FileExistsError(f"diagnostics exists; use --resume: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError("existing diagnostics must be a JSON array of objects")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


async def _commit_modal_volume(name: str, *, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError(
            "--commit-modal-volume requires the Modal Python package"
        ) from exc
    await modal.Volume.from_name(name).commit.aio()


async def _run(args: argparse.Namespace) -> None:
    if args.top_n <= 0:
        raise ValueError("top-n must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.save_every <= 0:
        raise ValueError("save-every must be positive")
    if not args.modal_volume_name.strip():
        raise ValueError("modal-volume-name must not be blank")
    if args.parent_max_tokens <= 0 or args.parent_total_tokens <= 0:
        raise ValueError("parent token limits must be positive")
    if args.parent_max_tokens > args.parent_total_tokens:
        raise ValueError("parent-max-tokens must not exceed parent-total-tokens")

    questions = _load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    contexts = _load_contexts(args.contexts)
    try:
        context_manifest = json.loads(
            args.context_manifest.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("retrieval context manifest is missing or invalid") from exc
    if not isinstance(context_manifest, dict):
        raise ValueError("retrieval context manifest must contain a JSON object")
    if context_manifest.get("complete") is not True:
        raise ValueError("retrieval context generation did not complete")
    if context_manifest.get("model") != "AITeamVN/Vietnamese_Reranker":
        raise ValueError("QA contexts were not produced by the selected reranker")
    if context_manifest.get("output_sha256") != _sha256(args.contexts):
        raise ValueError("retrieval contexts do not match their run manifest")
    if context_manifest.get("questions_sha256") != _sha256(args.questions):
        raise ValueError("retrieval contexts use a different question file")
    missing_contexts = [
        question_id
        for question_id, _ in questions
        if question_id not in contexts
    ]
    if missing_contexts:
        raise ValueError(f"missing retrieval contexts for {missing_contexts[:5]}")

    if _paths_collide(args.output, args.diagnostics):
        raise ValueError("answer output and diagnostics must be different files")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    if any(
        _paths_collide(left, right)
        for left, right in (
            (args.output, manifest_path),
            (args.diagnostics, manifest_path),
        )
    ):
        raise ValueError("answer output, diagnostics, and manifest must be different files")
    prompts_root = Path(os.environ.get("UDSC2026_PROMPTS_PATH", "prompts"))
    prompt_files = (
        prompts_root / "system" / f"{args.prompt_version}.md",
        prompts_root / "rag_templates" / f"{args.rag_template}.md",
    )
    for prompt_file in prompt_files:
        if not prompt_file.is_file():
            raise FileNotFoundError(f"selected prompt file is missing: {prompt_file}")
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    model_manifest_path = Path("models/download_manifest.json")
    if not model_manifest_path.is_file():
        raise FileNotFoundError("selected model download manifest is missing")
    parent_archive_sha256 = (
        _sha256(args.parent_archive)
        if args.parent_archive is not None
        else None
    )
    run_identity: dict[str, object] = {
        "schema_version": "answer-generation-v2",
        "questions_sha256": _sha256(args.questions),
        "question_count": len(questions),
        "contexts_sha256": _sha256(args.contexts),
        "context_manifest_sha256": _sha256(args.context_manifest),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "parent_archive_sha256": parent_archive_sha256,
        "config_sha256": _sha256(config_path),
        "prompt_version": args.prompt_version,
        "prompt_template": args.rag_template,
        "prompt_sha256": [_sha256(path) for path in prompt_files],
        "top_n": args.top_n,
        "parent_max_tokens": args.parent_max_tokens,
        "parent_total_tokens": args.parent_total_tokens,
        "parent_context_enabled": not args.no_parent_context,
    }
    output_exists = args.output.exists()
    diagnostics_exists = args.diagnostics.exists()
    if (output_exists or diagnostics_exists) and not args.resume:
        raise FileExistsError("answer output exists; use --resume only for the same manifest")
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"answer manifest exists; use a new output path: {manifest_path}")
    predictions = _load_existing(args.output, resume=args.resume)
    diagnostics = _load_diagnostics(args.diagnostics, resume=args.resume)
    if output_exists != diagnostics_exists:
        raise ValueError("answer output and diagnostics checkpoints must both exist")
    if output_exists:
        if not manifest_path.is_file():
            raise ValueError("cannot resume answers without the original run manifest")
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("answer run manifest is invalid") from exc
        if not isinstance(previous_manifest, dict):
            raise ValueError("answer run manifest must contain a JSON object")
        mismatches = {
            key: {"expected": value, "actual": previous_manifest.get(key)}
            for key, value in run_identity.items()
            if previous_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"answer inputs changed; refusing to resume: {mismatches}")
        if previous_manifest.get("completed_count") != len(predictions):
            raise ValueError("answer rows do not match the saved generation checkpoint")
    elif manifest_path.exists():
        raise FileExistsError(f"manifest exists without answer files: {manifest_path}")

    selected_ids = {question_id for question_id, _ in questions}
    completed_ids = {item["id"] for item in predictions}
    if not completed_ids <= selected_ids:
        raise ValueError("existing output contains IDs outside the selected questions")
    expected_prefix = [question_id for question_id, _ in questions[: len(predictions)]]
    if [item["id"] for item in predictions] != expected_prefix:
        raise ValueError("saved answers must be a prefix of the selected question order")
    if [item.get("id") for item in diagnostics] != expected_prefix:
        raise ValueError("saved diagnostics must align with the answer checkpoint")

    context_expander = None
    if not args.no_parent_context:
        parents_dir = args.parents_dir
        if not parents_dir.is_absolute():
            parents_dir = PROJECT_ROOT / parents_dir
        context_expander = ParentContextExpander(
            JsonlParentStore(
                parents_dir,
                max_cached_documents=256,
            ),
            max_parent_tokens=args.parent_max_tokens,
            max_total_tokens=args.parent_total_tokens,
        )

    print("loading_llm", flush=True)
    load_started = time.perf_counter()
    engine = QAEngine(
        LLMClient(load_llm_config(args.config)),
        context_expander=context_expander,
    )
    print(f"llm_loaded_seconds={time.perf_counter() - load_started:.3f}", flush=True)

    def save_manifest(*, complete: bool, output_hash: str | None = None) -> None:
        write_run_manifest(
            {
                **run_identity,
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "completed_count": len(predictions),
                "complete": complete,
                "output_sha256": output_hash,
            },
            manifest_path,
        )

    if not manifest_path.exists():
        save_manifest(complete=False)

    pending_checkpoints = 0
    for index, (question_id, question) in enumerate(questions, 1):
        if question_id in completed_ids:
            continue
        sample = contexts[question_id]
        selected_hits = sample.hits[: args.top_n]
        if not selected_hits:
            raise ValueError(f"question {question_id!r} has no retrieval hits")
        started = time.perf_counter()
        response = await engine.generate_answer(
            question=question,
            contexts=selected_hits,
            prompt_version=args.prompt_version,
            rag_template=args.rag_template,
            trace_id=f"legalqa-{question_id}",
        )
        latency = time.perf_counter() - started
        answer = response.answer.strip()
        if not answer:
            raise ValueError(f"question {question_id!r} generated an empty answer")
        predictions.append({"id": question_id, "answer": answer})
        diagnostics.append(
            {
                "id": question_id,
                "latency_seconds": latency,
                "answer_characters": len(answer),
                "citation_count": len(response.citations),
                "warning_count": len(response.warnings),
                "warnings": response.warnings,
                "prompt_version": response.used_prompt_version,
            }
        )
        completed_ids.add(question_id)
        pending_checkpoints += 1
        if pending_checkpoints >= args.save_every:
            _write_json_atomic(args.output, predictions)
            _write_json_atomic(args.diagnostics, diagnostics)
            save_manifest(complete=False)
            await _commit_modal_volume(
                args.modal_volume_name,
                enabled=args.commit_modal_volume,
            )
            pending_checkpoints = 0
        print(
            f"processed={index}/{len(questions)} id={question_id} "
            f"latency_seconds={latency:.3f} answer_characters={len(answer)}",
            flush=True,
        )

    if pending_checkpoints:
        _write_json_atomic(args.output, predictions)
        _write_json_atomic(args.diagnostics, diagnostics)
        await _commit_modal_volume(
            args.modal_volume_name,
            enabled=args.commit_modal_volume,
        )
    if len(completed_ids) != len(questions):
        raise ValueError("generation ended without exact selected-question coverage")
    save_manifest(complete=True, output_hash=_sha256(args.output))
    await _commit_modal_volume(
        args.modal_volume_name,
        enabled=args.commit_modal_volume,
    )
    print(f"complete_questions={len(questions)}", flush=True)
    print(args.output, flush=True)


def main() -> int:
    args = build_parser().parse_args()
    try:
        asyncio.run(_run(args))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"LegalQA generation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
