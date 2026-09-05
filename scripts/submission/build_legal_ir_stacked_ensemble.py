"""Build a leakage-safe supervised document ensemble for DSC2026 LegalIR.

The script consumes cached document rankings and cached DEk21 question
embeddings.  It never loads a neural model and never mutates the canonical
``data/processed_v3`` corpus.  Two labeled validation component files are
used in opposite train/evaluate directions before one final model is fitted
for the public questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, Sequence

import numpy as np
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import (  # type: ignore[import-untyped]  # noqa: E402
    normalize_legal_ir_matching_question,
)

OUTPUT_DOCUMENTS = 5
SOURCE_NAMES = ("dense", "bge", "knn", "bm25", "semantic")
TOP_THRESHOLDS = (1, 3, 5, 10, 20)
SEMANTIC_NEIGHBORS = 50
SEMANTIC_DOCUMENT_LIMIT = 100
MODEL_SEED = 2026
BASE_MODEL_BLEND_WEIGHT = 0.75
DOCUMENT_SCORE_FEATURE_SLICE = slice(46, 51)
MODEL_SETTINGS: dict[str, int | float | bool] = {
    "early_stopping": False,
    "l2_regularization": 1.0,
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 30,
    "random_state": MODEL_SEED,
}


@dataclass(frozen=True)
class Question:
    """One question and its optional organizer labels."""

    question_id: str
    text: str
    documents: tuple[str, ...]


@dataclass(frozen=True)
class SemanticRanking:
    """Document ranking transferred from nearby labeled questions."""

    documents: tuple[str, ...]
    scores: dict[str, float]


@dataclass(frozen=True)
class FeatureBatch:
    """Flat feature matrix plus boundaries needed to rank each question."""

    features: np.ndarray
    labels: np.ndarray
    query_ids: tuple[str, ...]
    query_documents: tuple[tuple[str, ...], ...]
    query_slices: tuple[tuple[int, int], ...]
    fallback_scores: np.ndarray


DocumentScores = dict[str, dict[str, tuple[float, float]]]


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"input does not exist: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"input must be UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty without surrounding whitespace")
    if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def load_questions(path: Path, *, require_labels: bool) -> dict[str, Question]:
    """Load an organizer-shaped question mapping without coercing IDs."""

    payload = _load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"question root must be a non-empty object: {path}")
    questions: dict[str, Question] = {}
    for raw_id, raw_record in payload.items():
        question_id = _identifier(raw_id, label="question ID")
        if not isinstance(raw_record, dict):
            raise TypeError(f"question {question_id!r} must be an object")
        text = raw_record.get("question")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"question {question_id!r} has no usable text")
        raw_documents = raw_record.get("answer")
        if raw_documents is None and not require_labels:
            documents: tuple[str, ...] = ()
        else:
            if not isinstance(raw_documents, list) or not raw_documents:
                raise ValueError(f"question {question_id!r} needs answer labels")
            documents = tuple(
                _identifier(value, label=f"answer for {question_id!r}")
                for value in raw_documents
            )
            if len(documents) != len(set(documents)):
                raise ValueError(f"answer for {question_id!r} contains duplicates")
            if len(documents) > OUTPUT_DOCUMENTS:
                raise ValueError(
                    f"answer for {question_id!r} exceeds {OUTPUT_DOCUMENTS} documents"
                )
        questions[question_id] = Question(question_id, text.strip(), documents)
    return questions


def load_components(
    path: Path,
    *,
    expected_ids: Sequence[str],
    allowed_document_ids: set[str],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Load four cached ranking sources with exact question coverage."""

    payload = _load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"component root must be a non-empty array: {path}")
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for index, raw_row in enumerate(payload):
        if not isinstance(raw_row, dict):
            raise TypeError(f"component row {index} must be an object")
        question_id = _identifier(raw_row.get("id"), label=f"component row {index} ID")
        if question_id in result:
            raise ValueError(f"duplicate component question ID {question_id!r}")
        row: dict[str, tuple[str, ...]] = {}
        for source in SOURCE_NAMES[:-1]:
            values = raw_row.get(source)
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"component {question_id!r} needs non-empty source {source!r}"
                )
            documents = tuple(
                _identifier(value, label=f"{source} document for {question_id!r}")
                for value in values
            )
            if len(documents) != len(set(documents)):
                raise ValueError(
                    f"component {question_id!r} source {source!r} has duplicates"
                )
            unknown = set(documents).difference(allowed_document_ids)
            if unknown:
                raise ValueError(
                    f"component {question_id!r} has unknown documents: "
                    f"{sorted(unknown)[:5]}"
                )
            row[source] = documents
        result[question_id] = row
    expected = list(expected_ids)
    if set(result) != set(expected) or len(result) != len(expected):
        missing = sorted(set(expected).difference(result))
        extra = sorted(set(result).difference(expected))
        raise ValueError(
            "component question coverage mismatch "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return result


def load_embeddings(
    embeddings_path: Path,
    ids_path: Path,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load a finite normalized embedding matrix and its exact row IDs."""

    raw_ids = _load_json(ids_path)
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("embedding ID root must be a non-empty array")
    ids = tuple(_identifier(value, label="embedding question ID") for value in raw_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("embedding question IDs contain duplicates")
    try:
        matrix = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid embedding matrix {embeddings_path}: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] < 1:
        raise ValueError(
            "embedding shape must be (len(ids), positive_dimension), got "
            f"{matrix.shape} for {len(ids)} IDs"
        )
    array = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("embedding matrix contains NaN or Infinity")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0):
        raise ValueError("embedding matrix contains a zero vector")
    array = array / norms[:, None]
    return array, ids


def load_document_scores(
    path: Path,
    *,
    expected_ids: Sequence[str],
    allowed_document_ids: set[str],
) -> DocumentScores:
    """Stream chunk results and retain max dense/BGE scores per document."""

    results: DocumentScores = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid score JSONL line {line_number} in {path}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"score row {line_number} must be an object")
            question_id = _identifier(
                row.get("question_id"), label=f"score row {line_number} ID"
            )
            if question_id in results:
                raise ValueError(f"duplicate score question ID {question_id!r}")
            raw_hits = row.get("hits")
            if not isinstance(raw_hits, list) or not raw_hits:
                raise ValueError(f"score row {question_id!r} needs non-empty hits")
            dense_by_document: dict[str, float] = {}
            rerank_by_document: dict[str, float] = {}
            for hit_index, hit in enumerate(raw_hits):
                if not isinstance(hit, dict):
                    raise TypeError(
                        f"score hit {question_id!r}[{hit_index}] must be an object"
                    )
                document_id = _identifier(
                    hit.get("doc_id"), label=f"score hit for {question_id!r}"
                )
                if document_id not in allowed_document_ids:
                    raise ValueError(
                        f"score row {question_id!r} has unknown "
                        f"document {document_id!r}"
                    )
                dense = hit.get("dense_score")
                rerank = hit.get("rerank_score")
                if (
                    isinstance(dense, bool)
                    or not isinstance(dense, (int, float))
                    or not math.isfinite(float(dense))
                    or isinstance(rerank, bool)
                    or not isinstance(rerank, (int, float))
                    or not math.isfinite(float(rerank))
                ):
                    raise ValueError(
                        f"score hit {question_id!r}[{hit_index}] is not finite"
                    )
                dense_by_document[document_id] = max(
                    dense_by_document.get(document_id, -math.inf), float(dense)
                )
                rerank_by_document[document_id] = max(
                    rerank_by_document.get(document_id, -math.inf), float(rerank)
                )
            results[question_id] = {
                document_id: (
                    dense_by_document[document_id],
                    rerank_by_document[document_id],
                )
                for document_id in dense_by_document
            }
    expected = set(expected_ids)
    if set(results) != expected:
        missing = sorted(expected.difference(results))
        extra = sorted(set(results).difference(expected))
        raise ValueError(
            f"score question coverage mismatch missing={missing[:5]} extra={extra[:5]}"
        )
    return results


def _document_prior(
    training: dict[str, Question], excluded_normalized: set[str]
) -> Counter[str]:
    prior: Counter[str] = Counter()
    for question in training.values():
        if normalize_legal_ir_matching_question(question.text) in excluded_normalized:
            continue
        prior.update(question.documents)
    return prior


def semantic_document_rankings(
    *,
    queries: dict[str, Question],
    query_embeddings: np.ndarray,
    query_embedding_ids: Sequence[str],
    training: dict[str, Question],
    training_embeddings: np.ndarray,
    training_embedding_ids: Sequence[str],
    excluded_normalized: set[str],
    neighbor_count: int = SEMANTIC_NEIGHBORS,
    document_limit: int = SEMANTIC_DOCUMENT_LIMIT,
) -> dict[str, SemanticRanking]:
    """Transfer labels from semantically nearby non-validation questions."""

    if neighbor_count <= 0 or document_limit <= 0:
        raise ValueError("semantic limits must be positive")
    query_index = {value: index for index, value in enumerate(query_embedding_ids)}
    training_index = {
        value: index for index, value in enumerate(training_embedding_ids)
    }
    if set(training) != set(training_index):
        raise ValueError("training question and embedding ID sets differ")
    missing_queries = set(queries).difference(query_index)
    if missing_queries:
        raise ValueError(
            f"query embeddings are missing IDs: {sorted(missing_queries)[:5]}"
        )
    eligible_ids = [
        question_id
        for question_id, question in training.items()
        if normalize_legal_ir_matching_question(question.text)
        not in excluded_normalized
    ]
    if not eligible_ids:
        raise ValueError("semantic exclusions removed every training question")
    eligible_rows = np.asarray(
        [training_index[question_id] for question_id in eligible_ids], dtype=np.int64
    )
    eligible_matrix = training_embeddings[eligible_rows]
    limit = min(neighbor_count, len(eligible_ids))
    results: dict[str, SemanticRanking] = {}
    ordered_query_ids = list(queries)
    for offset in range(0, len(ordered_query_ids), 128):
        batch_ids = ordered_query_ids[offset : offset + 128]
        batch_rows = np.asarray(
            [query_index[question_id] for question_id in batch_ids], dtype=np.int64
        )
        similarities = query_embeddings[batch_rows] @ eligible_matrix.T
        for row_index, question_id in enumerate(batch_ids):
            row = similarities[row_index]
            if limit == len(eligible_ids):
                nearest = np.arange(len(eligible_ids))
            else:
                nearest = np.argpartition(row, -limit)[-limit:]
            nearest = nearest[np.argsort(-row[nearest], kind="stable")]
            document_scores: defaultdict[str, float] = defaultdict(float)
            for rank, local_index in enumerate(nearest, 1):
                similarity = max(float(row[local_index]), 0.0)
                transfer_weight = similarity / math.log2(rank + 1.0)
                source_question = training[eligible_ids[int(local_index)]]
                for document_id in source_question.documents:
                    document_scores[document_id] += transfer_weight
            ordered = sorted(
                document_scores,
                key=lambda document_id: (-document_scores[document_id], document_id),
            )[:document_limit]
            peak = document_scores[ordered[0]] if ordered else 1.0
            normalized_scores = {
                document_id: document_scores[document_id] / peak
                for document_id in ordered
            }
            results[question_id] = SemanticRanking(
                documents=tuple(ordered), scores=normalized_scores
            )
    return results


def _query_flags(text: str) -> tuple[float, ...]:
    normalized = normalize_legal_ir_matching_question(text)
    tokens = normalized.split()
    has_digit = float(any(char.isdigit() for char in normalized))
    has_structure = float(
        any(term in normalized for term in ("điều", "khoản", "điểm", "chương"))
    )
    has_quantity = float(
        any(
            term in normalized
            for term in (
                "bao nhiêu",
                "mức phạt",
                "xử phạt",
                "thời hạn",
                "thời gian",
                "tỷ lệ",
            )
        )
    )
    has_instrument = float(
        any(
            term in normalized
            for term in (
                "luật",
                "nghị định",
                "thông tư",
                "quyết định",
                "bộ luật",
            )
        )
    )
    return (
        math.log1p(len(tokens)),
        has_digit,
        has_structure,
        has_quantity,
        has_instrument,
    )


def build_feature_batch(
    *,
    questions: dict[str, Question],
    components: dict[str, dict[str, tuple[str, ...]]],
    semantic: dict[str, SemanticRanking],
    prior: Counter[str],
    document_scores: DocumentScores,
) -> FeatureBatch:
    """Create deterministic pointwise features for every candidate document."""

    rows: list[list[float]] = []
    labels: list[int] = []
    query_ids: list[str] = []
    query_documents: list[tuple[str, ...]] = []
    query_slices: list[tuple[int, int]] = []
    fallback_scores: list[float] = []
    for question_id, question in questions.items():
        source_documents = dict(components[question_id])
        source_documents["semantic"] = semantic[question_id].documents
        rank_maps = {
            source: {document_id: rank for rank, document_id in enumerate(documents, 1)}
            for source, documents in source_documents.items()
        }
        candidates = tuple(
            dict.fromkeys(
                document_id
                for source in SOURCE_NAMES
                for document_id in source_documents[source]
            )
        )
        if len(candidates) < OUTPUT_DOCUMENTS:
            raise ValueError(
                f"question {question_id!r} has fewer than {OUTPUT_DOCUMENTS} candidates"
            )
        flags = _query_flags(question.text)
        query_scores = document_scores[question_id]
        max_dense = max((value[0] for value in query_scores.values()), default=0.0)
        max_rerank = max((value[1] for value in query_scores.values()), default=0.0)
        start = len(rows)
        for document_id in candidates:
            reciprocal: list[float] = []
            logarithmic: list[float] = []
            presence: list[float] = []
            thresholds: list[float] = []
            for source in SOURCE_NAMES:
                rank = rank_maps[source].get(document_id)
                reciprocal.append(0.0 if rank is None else 1.0 / rank)
                logarithmic.append(0.0 if rank is None else 1.0 / math.log2(rank + 1.0))
                presence.append(float(rank is not None))
                thresholds.extend(
                    float(rank is not None and rank <= threshold)
                    for threshold in TOP_THRESHOLDS
                )
            count = sum(presence)
            semantic_score = semantic[question_id].scores.get(document_id, 0.0)
            dense_score, rerank_score = query_scores.get(document_id, (0.0, 0.0))
            clipped_rerank = min(max(rerank_score, 1e-6), 1.0 - 1e-6)
            global_features = [
                count,
                sum(reciprocal),
                max(reciprocal),
                sum(logarithmic),
                math.log1p(prior[document_id]),
                semantic_score,
                dense_score,
                dense_score - max_dense,
                rerank_score,
                math.log(clipped_rerank / (1.0 - clipped_rerank)),
                rerank_score - max_rerank,
                float(
                    document_id in rank_maps["dense"]
                    and document_id in rank_maps["bge"]
                ),
                float(
                    document_id in rank_maps["bge"] and document_id in rank_maps["bm25"]
                ),
                float(
                    document_id in rank_maps["knn"]
                    and document_id in rank_maps["semantic"]
                ),
            ]
            interactions = [
                reciprocal_value * flag
                for reciprocal_value in reciprocal
                for flag in flags[1:]
            ]
            rows.append(
                reciprocal
                + logarithmic
                + presence
                + thresholds
                + global_features
                + list(flags)
                + interactions
            )
            labels.append(int(document_id in question.documents))
            fallback_scores.append(sum(reciprocal))
        stop = len(rows)
        query_ids.append(question_id)
        query_documents.append(candidates)
        query_slices.append((start, stop))
    features = np.asarray(rows, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.uint8)
    if not np.isfinite(features).all():
        raise ValueError("generated features contain NaN or Infinity")
    return FeatureBatch(
        features=features,
        labels=label_array,
        query_ids=tuple(query_ids),
        query_documents=tuple(query_documents),
        query_slices=tuple(query_slices),
        fallback_scores=np.asarray(fallback_scores, dtype=np.float32),
    )


def _fit_model(
    batch: FeatureBatch, *, positive_weight: float
) -> HistGradientBoostingClassifier:
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive weight must be finite and greater than zero")
    if not np.any(batch.labels == 1) or not np.any(batch.labels == 0):
        raise ValueError("training features need both positive and negative labels")
    model = HistGradientBoostingClassifier(**MODEL_SETTINGS)
    weights = np.where(batch.labels == 1, positive_weight, 1.0)
    model.fit(batch.features, batch.labels, sample_weight=weights)
    return model


def rank_feature_batch(
    model: HistGradientBoostingClassifier, batch: FeatureBatch
) -> list[dict[str, object]]:
    """Rank exactly five unique documents per question."""

    probabilities = model.predict_proba(batch.features)[:, 1]
    predictions: list[dict[str, object]] = []
    for question_id, documents, (start, stop) in zip(
        batch.query_ids,
        batch.query_documents,
        batch.query_slices,
        strict=True,
    ):
        local = list(range(stop - start))
        local.sort(
            key=lambda index: (
                -float(probabilities[start + index]),
                -float(batch.fallback_scores[start + index]),
                documents[index],
            )
        )
        predictions.append(
            {
                "id": question_id,
                "documents": [documents[index] for index in local[:OUTPUT_DOCUMENTS]],
            }
        )
    return predictions


def without_document_score_features(batch: FeatureBatch) -> FeatureBatch:
    """Return an equivalent batch with the five optional raw-score features off."""

    features = batch.features.copy()
    features[:, DOCUMENT_SCORE_FEATURE_SLICE] = 0.0
    return FeatureBatch(
        features=features,
        labels=batch.labels,
        query_ids=batch.query_ids,
        query_documents=batch.query_documents,
        query_slices=batch.query_slices,
        fallback_scores=batch.fallback_scores,
    )


def rank_blended_feature_batch(
    *,
    base_model: HistGradientBoostingClassifier,
    scored_model: HistGradientBoostingClassifier,
    base_batch: FeatureBatch,
    scored_batch: FeatureBatch,
    base_weight: float = BASE_MODEL_BLEND_WEIGHT,
) -> list[dict[str, object]]:
    """Blend within-query model ranks so probability calibration cannot drift."""

    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("base model blend weight must be between zero and one")
    if (
        base_batch.query_ids != scored_batch.query_ids
        or base_batch.query_documents != scored_batch.query_documents
        or base_batch.query_slices != scored_batch.query_slices
    ):
        raise ValueError("blended feature batches must describe identical candidates")
    base_probabilities = base_model.predict_proba(base_batch.features)[:, 1]
    scored_probabilities = scored_model.predict_proba(scored_batch.features)[:, 1]
    predictions: list[dict[str, object]] = []
    for question_id, documents, (start, stop) in zip(
        scored_batch.query_ids,
        scored_batch.query_documents,
        scored_batch.query_slices,
        strict=True,
    ):
        candidate_count = stop - start
        base_ranks = np.empty(candidate_count, dtype=np.float32)
        scored_ranks = np.empty(candidate_count, dtype=np.float32)
        for values, ranks in (
            (base_probabilities[start:stop], base_ranks),
            (scored_probabilities[start:stop], scored_ranks),
        ):
            order = np.argsort(-values, kind="stable")
            ranks[order] = np.arange(candidate_count, 0, -1, dtype=np.float32)
            ranks /= candidate_count
        blended = base_weight * base_ranks + (1.0 - base_weight) * scored_ranks
        order = sorted(
            range(candidate_count),
            key=lambda index: (-float(blended[index]), documents[index]),
        )
        predictions.append(
            {
                "id": question_id,
                "documents": [documents[index] for index in order[:OUTPUT_DOCUMENTS]],
            }
        )
    return predictions


def official_metrics(
    predictions: Sequence[dict[str, object]], references: dict[str, Question]
) -> dict[str, float | int]:
    """Compute the organizer's macro set Recall and Precision."""

    recalls: list[float] = []
    precisions: list[float] = []
    for row in predictions:
        question_id = _identifier(row.get("id"), label="prediction ID")
        raw_documents = row.get("documents")
        if not isinstance(raw_documents, list):
            raise TypeError("prediction documents must be an array")
        documents = [
            _identifier(value, label=f"prediction for {question_id!r}")
            for value in raw_documents
        ]
        if len(documents) != len(set(documents)) == OUTPUT_DOCUMENTS:
            raise ValueError(
                "each prediction must contain exactly five unique documents"
            )
        gold = set(references[question_id].documents)
        matched = len(gold.intersection(documents))
        recalls.append(matched / len(gold))
        precisions.append(matched / len(documents))
    return {
        "precision": sum(precisions) / len(precisions),
        "question_count": len(recalls),
        "recall": sum(recalls) / len(recalls),
    }


def apply_exact_overlay(
    predictions: list[dict[str, object]],
    target_questions: dict[str, Question],
    labeled_sources: Sequence[dict[str, Question]],
) -> int:
    """Put labels first only for exact normalized train/public duplicates."""

    labels: defaultdict[str, list[str]] = defaultdict(list)
    for source in labeled_sources:
        for question in source.values():
            key = normalize_legal_ir_matching_question(question.text)
            for document_id in question.documents:
                if document_id not in labels[key]:
                    labels[key].append(document_id)
                if len(labels[key]) > OUTPUT_DOCUMENTS:
                    raise ValueError(
                        "exact-match label sources disagree on more than five documents"
                    )
    matches = 0
    for row in predictions:
        question_id = _identifier(row.get("id"), label="prediction ID")
        exact = labels.get(
            normalize_legal_ir_matching_question(target_questions[question_id].text)
        )
        if not exact:
            continue
        raw_documents = row.get("documents")
        assert isinstance(raw_documents, list)
        documents = [*exact, *(item for item in raw_documents if item not in exact)]
        row["documents"] = documents[:OUTPUT_DOCUMENTS]
        matches += 1
    return matches


def _without_normalized_groups(
    questions: dict[str, Question], excluded: set[str]
) -> dict[str, Question]:
    """Remove complete normalized-question groups from a labeled source."""

    return {
        question_id: question
        for question_id, question in questions.items()
        if normalize_legal_ir_matching_question(question.text) not in excluded
    }


def _copy_predictions(
    predictions: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    for row in predictions:
        documents = row.get("documents")
        if not isinstance(documents, list):
            raise TypeError("prediction documents must be an array")
        copied.append({"id": row.get("id"), "documents": list(documents)})
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    def command(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    commit_result = command("rev-parse", "HEAD")
    status_result = command("status", "--porcelain")
    return {
        "commit": (
            commit_result.stdout.strip()
            if commit_result.returncode == 0
            else "unavailable"
        ),
        "dirty": status_result.returncode != 0 or bool(status_result.stdout.strip()),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _corpus_ids(path: Path) -> list[str]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError("corpus manifest must be a non-empty string array")
    ids = [_identifier(value, label="corpus document ID") for value in payload]
    if len(ids) != len(set(ids)):
        raise ValueError("corpus manifest contains duplicate IDs")
    return ids


def _fold_resources(
    *,
    questions: dict[str, Question],
    components_path: Path,
    score_path: Path,
    corpus: set[str],
    training: dict[str, Question],
    train_embeddings: np.ndarray,
    train_embedding_ids: Sequence[str],
) -> FeatureBatch:
    excluded = {
        normalize_legal_ir_matching_question(question.text)
        for question in questions.values()
    }
    components = load_components(
        components_path,
        expected_ids=list(questions),
        allowed_document_ids=corpus,
    )
    document_scores = load_document_scores(
        score_path,
        expected_ids=list(questions),
        allowed_document_ids=corpus,
    )
    semantic = semantic_document_rankings(
        queries=questions,
        query_embeddings=train_embeddings,
        query_embedding_ids=train_embedding_ids,
        training=training,
        training_embeddings=train_embeddings,
        training_embedding_ids=train_embedding_ids,
        excluded_normalized=excluded,
    )
    return build_feature_batch(
        questions=questions,
        components=components,
        semantic=semantic,
        prior=_document_prior(training, excluded),
        document_scores=document_scores,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Cross-validate, fit the final stacker, and write an audited candidate."""

    if args.output.resolve() == args.report.resolve():
        raise ValueError("prediction and report outputs must differ")
    training = load_questions(args.train, require_labels=True)
    public = load_questions(args.questions, require_labels=False)
    overlay_sources = [
        load_questions(path, require_labels=True) for path in args.overlay_labels
    ]
    fold_a = load_questions(args.fold_a_questions, require_labels=True)
    fold_b = load_questions(args.fold_b_questions, require_labels=True)
    if set(fold_a).intersection(fold_b):
        raise ValueError("validation folds must have disjoint question IDs")
    if not set(fold_a).union(fold_b).issubset(training):
        raise ValueError("validation fold IDs must come from the training dataset")
    corpus_ids = _corpus_ids(args.corpus_manifest)
    corpus = set(corpus_ids)
    train_embeddings, train_embedding_ids = load_embeddings(
        args.train_embeddings, args.train_embedding_ids
    )
    public_embeddings, public_embedding_ids = load_embeddings(
        args.public_embeddings, args.public_embedding_ids
    )
    if set(public_embedding_ids) != set(public):
        raise ValueError("public question and embedding ID sets differ")

    print("building leakage-safe fold A features", flush=True)
    batch_a = _fold_resources(
        questions=fold_a,
        components_path=args.fold_a_components,
        score_path=args.fold_a_scores,
        corpus=corpus,
        training=training,
        train_embeddings=train_embeddings,
        train_embedding_ids=train_embedding_ids,
    )
    print("building leakage-safe fold B features", flush=True)
    batch_b = _fold_resources(
        questions=fold_b,
        components_path=args.fold_b_components,
        score_path=args.fold_b_scores,
        corpus=corpus,
        training=training,
        train_embeddings=train_embeddings,
        train_embedding_ids=train_embedding_ids,
    )
    base_batch_a = without_document_score_features(batch_a)
    base_batch_b = without_document_score_features(batch_b)
    base_model_a = _fit_model(base_batch_a, positive_weight=args.positive_weight)
    base_model_b = _fit_model(base_batch_b, positive_weight=args.positive_weight)
    scored_model_a = _fit_model(batch_a, positive_weight=args.positive_weight)
    scored_model_b = _fit_model(batch_b, positive_weight=args.positive_weight)
    predictions_b = rank_blended_feature_batch(
        base_model=base_model_a,
        scored_model=scored_model_a,
        base_batch=base_batch_b,
        scored_batch=batch_b,
    )
    predictions_a = rank_blended_feature_batch(
        base_model=base_model_b,
        scored_model=scored_model_b,
        base_batch=base_batch_a,
        scored_batch=batch_a,
    )
    metrics_a = official_metrics(predictions_a, fold_a)
    metrics_b = official_metrics(predictions_b, fold_b)
    pooled_core = {
        "precision": (float(metrics_a["precision"]) + float(metrics_b["precision"]))
        / 2.0,
        "question_count": int(metrics_a["question_count"])
        + int(metrics_b["question_count"]),
        "recall": (float(metrics_a["recall"]) + float(metrics_b["recall"])) / 2.0,
    }
    excluded_a = {
        normalize_legal_ir_matching_question(question.text)
        for question in fold_a.values()
    }
    excluded_b = {
        normalize_legal_ir_matching_question(question.text)
        for question in fold_b.values()
    }
    overlay_predictions_a = _copy_predictions(predictions_a)
    overlay_predictions_b = _copy_predictions(predictions_b)
    overlay_a_count = apply_exact_overlay(
        overlay_predictions_a,
        fold_a,
        [*overlay_sources, _without_normalized_groups(training, excluded_a)],
    )
    overlay_b_count = apply_exact_overlay(
        overlay_predictions_b,
        fold_b,
        [*overlay_sources, _without_normalized_groups(training, excluded_b)],
    )
    overlay_metrics_a = official_metrics(overlay_predictions_a, fold_a)
    overlay_metrics_b = official_metrics(overlay_predictions_b, fold_b)
    pooled_overlay = {
        "precision": (
            float(overlay_metrics_a["precision"])
            + float(overlay_metrics_b["precision"])
        )
        / 2.0,
        "question_count": int(overlay_metrics_a["question_count"])
        + int(overlay_metrics_b["question_count"]),
        "recall": (
            float(overlay_metrics_a["recall"]) + float(overlay_metrics_b["recall"])
        )
        / 2.0,
    }

    public_components = load_components(
        args.components,
        expected_ids=list(public),
        allowed_document_ids=corpus,
    )
    public_scores = load_document_scores(
        args.public_scores,
        expected_ids=list(public),
        allowed_document_ids=corpus,
    )
    print("building public semantic-neighbor features", flush=True)
    public_semantic = semantic_document_rankings(
        queries=public,
        query_embeddings=public_embeddings,
        query_embedding_ids=public_embedding_ids,
        training=training,
        training_embeddings=train_embeddings,
        training_embedding_ids=train_embedding_ids,
        excluded_normalized=set(),
    )
    public_batch = build_feature_batch(
        questions=public,
        components=public_components,
        semantic=public_semantic,
        prior=_document_prior(training, set()),
        document_scores=public_scores,
    )
    combined_scored = FeatureBatch(
        features=np.concatenate((batch_a.features, batch_b.features)),
        labels=np.concatenate((batch_a.labels, batch_b.labels)),
        query_ids=(),
        query_documents=(),
        query_slices=(),
        fallback_scores=np.concatenate(
            (batch_a.fallback_scores, batch_b.fallback_scores)
        ),
    )
    combined_base = without_document_score_features(combined_scored)
    public_base_batch = without_document_score_features(public_batch)
    final_base_model = _fit_model(combined_base, positive_weight=args.positive_weight)
    final_scored_model = _fit_model(
        combined_scored, positive_weight=args.positive_weight
    )
    predictions = rank_blended_feature_batch(
        base_model=final_base_model,
        scored_model=final_scored_model,
        base_batch=public_base_batch,
        scored_batch=public_batch,
    )
    exact_matches = apply_exact_overlay(
        predictions, public, [training, *overlay_sources]
    )
    prediction_ids = [str(row["id"]) for row in predictions]
    if prediction_ids != list(public):
        raise ValueError("prediction order or coverage differs from public questions")
    for row in predictions:
        raw_documents = row["documents"]
        assert isinstance(raw_documents, list)
        if len(raw_documents) != len(set(raw_documents)) == OUTPUT_DOCUMENTS:
            raise ValueError("final prediction does not contain five unique documents")
        if not set(raw_documents).issubset(corpus):
            raise ValueError("final prediction contains a document outside the corpus")

    _atomic_json(args.output, predictions)
    input_paths = {
        "components": args.components,
        "corpus_manifest": args.corpus_manifest,
        "fold_a_components": args.fold_a_components,
        "fold_a_questions": args.fold_a_questions,
        "fold_a_scores": args.fold_a_scores,
        "fold_b_components": args.fold_b_components,
        "fold_b_questions": args.fold_b_questions,
        "fold_b_scores": args.fold_b_scores,
        "public_embedding_ids": args.public_embedding_ids,
        "public_embeddings": args.public_embeddings,
        "public_scores": args.public_scores,
        "questions": args.questions,
        "train": args.train,
        "train_embedding_ids": args.train_embedding_ids,
        "train_embeddings": args.train_embeddings,
    }
    input_paths.update(
        {
            f"overlay_labels_{index}": path
            for index, path in enumerate(args.overlay_labels, 1)
        }
    )
    report: dict[str, object] = {
        "schema_version": "legal-ir-stacked-ensemble-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_validation": {
            "core": {
                "fold_a_trained_on_fold_b": metrics_a,
                "fold_b_trained_on_fold_a": metrics_b,
                "pooled": pooled_core,
            },
            "with_external_exact_overlay": {
                "fold_a": {
                    "exact_match_count": overlay_a_count,
                    "metrics": overlay_metrics_a,
                },
                "fold_b": {
                    "exact_match_count": overlay_b_count,
                    "metrics": overlay_metrics_b,
                },
                "pooled": pooled_overlay,
            },
        },
        "exact_normalized_label_overlay_count": exact_matches,
        "feature_count": int(batch_a.features.shape[1]),
        "git": _git_state(),
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "input_sha256": {
            name: _sha256(path) for name, path in sorted(input_paths.items())
        },
        "model": {
            "base_model_blend_weight": BASE_MODEL_BLEND_WEIGHT,
            "classes": [
                "HistGradientBoostingClassifier-without-raw-scores",
                "HistGradientBoostingClassifier-with-raw-scores",
            ],
            "positive_weight": args.positive_weight,
            "settings": MODEL_SETTINGS,
        },
        "output": {
            "documents_per_question": OUTPUT_DOCUMENTS,
            "path": str(args.output),
            "question_count": len(predictions),
            "sha256": _sha256(args.output),
        },
        "semantic_transfer": {
            "document_limit": SEMANTIC_DOCUMENT_LIMIT,
            "neighbor_count": SEMANTIC_NEIGHBORS,
            "validation_group_exclusion": "normalized-question group",
        },
    }
    _atomic_json(args.report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--fold-a-questions", type=Path, required=True)
    parser.add_argument("--fold-a-components", type=Path, required=True)
    parser.add_argument("--fold-a-scores", type=Path, required=True)
    parser.add_argument("--fold-b-questions", type=Path, required=True)
    parser.add_argument("--fold-b-components", type=Path, required=True)
    parser.add_argument("--fold-b-scores", type=Path, required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--train-embedding-ids", type=Path, required=True)
    parser.add_argument("--public-embeddings", type=Path, required=True)
    parser.add_argument("--public-embedding-ids", type=Path, required=True)
    parser.add_argument("--public-scores", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument(
        "--overlay-labels",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional organizer-labeled mapping used only for normalized exact "
            "question matches; may be repeated."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--positive-weight", type=float, default=80.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalIR stacked ensemble error: {exc}", file=sys.stderr)
        return 2
    cv = report["cross_validation"]
    assert isinstance(cv, dict)
    core = cv["core"]
    assert isinstance(core, dict)
    pooled = core["pooled"]
    assert isinstance(pooled, dict)
    print(
        "valid "
        f"questions={report['output']['question_count']} "  # type: ignore[index]
        f"cv_recall={float(pooled['recall']):.6f} "
        f"cv_precision={float(pooled['precision']):.6f} "
        f"exact_overlays={report['exact_normalized_label_overlay_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
