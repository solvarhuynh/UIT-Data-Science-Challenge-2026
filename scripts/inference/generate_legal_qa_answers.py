"""Generate checkpointed LegalQA answers from precomputed retrieval hits."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.models import PredictionSample  # noqa: E402
from udsc2026.infrastructure.llm.client import LLMClient  # noqa: E402
from udsc2026.infrastructure.llm.config import load_llm_config  # noqa: E402
from udsc2026.qa.qa_engine import QAEngine  # noqa: E402
from udsc2026.retrieval.parent_context import (  # noqa: E402
    JsonlParentStore,
    ParentContextExpander,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/gpu.yaml"))
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt-version", default="legal_qa_v4")
    parser.add_argument("--rag-template", default="default_rag_v4")
    parser.add_argument("--parent-max-tokens", type=int, default=384)
    parser.add_argument("--parent-total-tokens", type=int, default=1152)
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


async def _run(args: argparse.Namespace) -> None:
    if args.top_n <= 0:
        raise ValueError("top-n must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.parent_max_tokens <= 0 or args.parent_total_tokens <= 0:
        raise ValueError("parent token limits must be positive")
    if args.parent_max_tokens > args.parent_total_tokens:
        raise ValueError("parent-max-tokens must not exceed parent-total-tokens")

    questions = _load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    contexts = _load_contexts(args.contexts)
    missing_contexts = [
        question_id
        for question_id, _ in questions
        if question_id not in contexts
    ]
    if missing_contexts:
        raise ValueError(f"missing retrieval contexts for {missing_contexts[:5]}")

    predictions = _load_existing(args.output, resume=args.resume)
    diagnostics = _load_diagnostics(args.diagnostics, resume=args.resume)
    selected_ids = {question_id for question_id, _ in questions}
    completed_ids = {item["id"] for item in predictions}
    if not completed_ids <= selected_ids:
        raise ValueError("existing output contains IDs outside the selected questions")

    context_expander = None
    if not args.no_parent_context:
        context_expander = ParentContextExpander(
            JsonlParentStore(
                PROJECT_ROOT / "data/processed_v3/parents",
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
        _write_json_atomic(args.output, predictions)
        _write_json_atomic(args.diagnostics, diagnostics)
        print(
            f"processed={index}/{len(questions)} id={question_id} "
            f"latency_seconds={latency:.3f} answer_characters={len(answer)}",
            flush=True,
        )

    if len(completed_ids) != len(questions):
        raise ValueError("generation ended without exact selected-question coverage")
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
