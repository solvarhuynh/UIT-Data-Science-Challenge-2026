"""Pure metric functions with explicit edge-case behavior."""

import math
import re
import unicodedata
from typing import Collection, List, Sequence, cast

from udsc2026.evaluation.models import LatencyStats

_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _validated_ids(values: Collection[str], *, name: str) -> List[str]:
    """Return stripped IDs after strict type/content validation."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a collection of string IDs")
    result: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        result.append(value.strip())
    return result


def reciprocal_rank(
    predicted_chunk_ids: Sequence[str],
    gold_chunk_ids: Collection[str],
) -> float:
    """Return reciprocal rank of the first relevant result, or zero if absent."""

    predictions = _validated_ids(
        predicted_chunk_ids,
        name="predicted_chunk_ids",
    )
    gold = set(_validated_ids(gold_chunk_ids, name="gold_chunk_ids"))
    if not gold:
        raise ValueError("gold_chunk_ids must not be empty")
    for rank, chunk_id in enumerate(predictions, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    predictions: Sequence[Sequence[str]],
    gold_chunk_ids: Sequence[Collection[str]],
) -> float:
    """Compute macro-average reciprocal rank across equally aligned samples."""

    _validate_parallel_sequences(predictions, gold_chunk_ids)
    scores = [
        reciprocal_rank(predicted, gold)
        for predicted, gold in zip(predictions, gold_chunk_ids)
    ]
    return sum(scores) / len(scores)


def recall_at_k(
    predicted_chunk_ids: Sequence[str],
    gold_chunk_ids: Collection[str],
    k: int,
) -> float:
    """Return the fraction of unique relevant chunks present in the first ``k`` hits."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    predictions = _validated_ids(
        predicted_chunk_ids,
        name="predicted_chunk_ids",
    )
    gold = set(_validated_ids(gold_chunk_ids, name="gold_chunk_ids"))
    if not gold:
        raise ValueError("gold_chunk_ids must not be empty")
    return len(set(predictions[:k]).intersection(gold)) / len(gold)


def mean_recall_at_k(
    predictions: Sequence[Sequence[str]],
    gold_chunk_ids: Sequence[Collection[str]],
    k: int,
) -> float:
    """Compute macro-average Recall@K over aligned samples."""

    _validate_parallel_sequences(predictions, gold_chunk_ids)
    scores = [
        recall_at_k(predicted, gold, k)
        for predicted, gold in zip(predictions, gold_chunk_ids)
    ]
    return sum(scores) / len(scores)


def _validate_parallel_sequences(
    left: Sequence[object],
    right: Sequence[object],
) -> None:
    """Validate the shared alignment rule used by macro metrics."""

    if not left:
        raise ValueError("at least one sample is required")
    if len(left) != len(right):
        raise ValueError("prediction and reference sample counts must match")


def _tokenize_vietnamese(text: str) -> List[str]:
    """Normalize canonical Unicode and tokenize words without removing accents."""

    if not isinstance(text, str):
        raise TypeError("ROUGE-L inputs must be strings")
    normalized = unicodedata.normalize("NFC", text).casefold()
    return _WORD_PATTERN.findall(normalized)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Compute longest-common-subsequence length with O(min(n, m)) memory."""

    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = [0] * (len(short) + 1)
    for long_token in long:
        current = [0]
        for index, short_token in enumerate(short, start=1):
            if long_token == short_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute sentence-level ROUGE-L F1 after Vietnamese Unicode normalization."""

    predicted_tokens = _tokenize_vietnamese(prediction)
    reference_tokens = _tokenize_vietnamese(reference)
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def mean_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute macro-average sentence-level ROUGE-L F1."""

    _validate_parallel_sequences(predictions, references)
    scores = [
        rouge_l_score(prediction, reference)
        for prediction, reference in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile from already sorted values."""

    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def aggregate_latencies(latencies_ms: Sequence[float]) -> LatencyStats:
    """Summarize finite non-negative millisecond measurements."""

    if not latencies_ms:
        raise ValueError("at least one latency value is required")
    values: List[float] = []
    for raw_value in cast(Sequence[object], latencies_ms):
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError("latency values must be real numbers")
        converted = float(raw_value)
        if not math.isfinite(converted) or converted < 0:
            raise ValueError("latency values must be finite and non-negative")
        values.append(converted)
    values.sort()
    total = sum(values)
    return LatencyStats(
        count=len(values),
        total_ms=total,
        min_ms=values[0],
        max_ms=values[-1],
        mean_ms=total / len(values),
        median_ms=_percentile(values, 0.5),
        p95_ms=_percentile(values, 0.95),
        p99_ms=_percentile(values, 0.99),
    )
