"""Batch reranking and deterministic prediction artifact helpers for TV5."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.models import BenchmarkSample, PredictionSample

SampleT = TypeVar("SampleT", BenchmarkSample, PredictionSample)


class CandidateReranker(Protocol):
    """Minimal reranker interface required by the offline batch workflow."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        """Return at most ``top_n`` candidates in descending relevance order."""


@dataclass(frozen=True)
class BatchRerankResult:
    """Normalized baseline, reranked output, and isolated reranker latency."""

    before: list[PredictionSample]
    after: list[PredictionSample]
    rerank_latencies_ms: list[float]
    candidate_pair_count: int

    @property
    def total_rerank_ms(self) -> float:
        """Return total measured model-scoring latency in milliseconds."""

        return sum(self.rerank_latencies_ms)


def rerank_prediction_samples(
    benchmark: Sequence[BenchmarkSample],
    predictions: Sequence[PredictionSample],
    reranker: CandidateReranker,
    *,
    candidate_k: int,
    top_n: int,
    clock: Callable[[], float] = time.perf_counter,
) -> BatchRerankResult:
    """Align inputs by question ID and rerank a bounded candidate pool per query."""

    _validate_positive_integer(candidate_k, "candidate_k")
    _validate_positive_integer(top_n, "top_n")
    if top_n > candidate_k:
        raise ValueError("top_n must not exceed candidate_k")
    if not benchmark:
        raise ValueError("benchmark must not be empty")

    benchmark_by_id = _index_by_question_id(benchmark, "benchmark")
    prediction_by_id = _index_by_question_id(predictions, "predictions")
    missing = sorted(set(benchmark_by_id).difference(prediction_by_id))
    extra = sorted(set(prediction_by_id).difference(benchmark_by_id))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing IDs: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected IDs: {', '.join(extra)}")
        raise ValueError(
            "prediction IDs do not match benchmark (" + "; ".join(details) + ")"
        )

    before: list[PredictionSample] = []
    after: list[PredictionSample] = []
    rerank_latencies_ms: list[float] = []
    candidate_pair_count = 0

    for reference in benchmark:
        prediction = prediction_by_id[reference.question_id]
        candidates = [
            hit.model_copy(deep=True) for hit in prediction.hits[:candidate_k]
        ]
        baseline = prediction.model_copy(deep=True, update={"hits": candidates})

        started = clock()
        reranked_hits = reranker.rerank(reference.question, candidates, top_n)
        elapsed_ms = max(0.0, (clock() - started) * 1000.0)
        after_latency = (
            prediction.latency_ms + elapsed_ms
            if prediction.latency_ms is not None
            else None
        )

        before.append(baseline)
        after.append(
            prediction.model_copy(
                deep=True,
                update={"hits": reranked_hits, "latency_ms": after_latency},
            )
        )
        rerank_latencies_ms.append(elapsed_ms)
        candidate_pair_count += len(candidates)

    return BatchRerankResult(
        before=before,
        after=after,
        rerank_latencies_ms=rerank_latencies_ms,
        candidate_pair_count=candidate_pair_count,
    )


def write_predictions(
    samples: Sequence[PredictionSample],
    path: str | Path,
) -> Path:
    """Atomically write strict prediction JSON or JSONL for evaluator reuse."""

    if not samples:
        raise ValueError("prediction output must not be empty")
    for index, sample in enumerate(samples):
        if not isinstance(sample, PredictionSample):
            raise TypeError(f"samples[{index}] must be a PredictionSample")

    target = Path(path)
    suffix = target.suffix.casefold()
    payloads = [sample.model_dump(mode="json", exclude_none=True) for sample in samples]
    if suffix == ".json":
        content = (
            json.dumps(
                {"predictions": payloads},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    elif suffix == ".jsonl":
        content = "".join(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for payload in payloads
        )
    else:
        raise ValueError("prediction output must use .json or .jsonl")

    _atomic_write_text(target, content)
    return target


def write_run_manifest(payload: dict[str, Any], path: str | Path) -> Path:
    """Atomically write one JSON run manifest with stable key ordering."""

    if not isinstance(payload, dict) or not payload:
        raise ValueError("run manifest payload must be a non-empty mapping")
    target = Path(path)
    if target.suffix.casefold() != ".json":
        raise ValueError("run manifest output must use .json")
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_write_text(target, content)
    return target


def _index_by_question_id(
    samples: Sequence[SampleT],
    label: str,
) -> dict[str, SampleT]:
    indexed: dict[str, SampleT] = {}
    for sample in samples:
        if sample.question_id in indexed:
            raise ValueError(f"duplicate question_id in {label}: {sample.question_id}")
        indexed[sample.question_id] = sample
    return indexed


def _validate_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _atomic_write_text(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ValueError(f"output target must not be a symbolic link: {path}")
    if path.is_dir():
        raise ValueError(f"output target must not be a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(content)
        os.replace(temporary_path, path)
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
