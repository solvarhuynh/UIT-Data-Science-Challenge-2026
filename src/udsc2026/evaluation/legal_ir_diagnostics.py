"""Document-level LegalIR diagnostics kept separate from official scoring.

Candidate diagnostics deliberately inspect deep chunk rankings.  They never
apply the organizer's one-to-five-document penalty; that penalty is applied
only after collapsing the final ranked documents for official evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from udsc2026.evaluation.legal_ir import (
    OFFICIAL_MAX_DOCUMENTS,
    LegalIRPrediction,
    LegalIRReference,
    evaluate_legal_ir,
)
from udsc2026.evaluation.models import PredictionSample

DEFAULT_CANDIDATE_CHUNK_DEPTHS = (5, 10, 20, 50, 100, 200)
OPTIONAL_CANDIDATE_CHUNK_DEPTH = 500


def _index_references(
    references: Sequence[LegalIRReference],
) -> dict[str, LegalIRReference]:
    indexed: dict[str, LegalIRReference] = {}
    for index, reference in enumerate(references):
        if not isinstance(reference, LegalIRReference):
            raise TypeError(f"references[{index}] must be a LegalIRReference")
        if reference.id in indexed:
            raise ValueError(f"duplicate question ID in references: {reference.id}")
        indexed[reference.id] = reference
    if not indexed:
        raise ValueError("references must not be empty")
    return indexed


def _index_predictions(
    predictions: Sequence[PredictionSample],
) -> dict[str, PredictionSample]:
    indexed: dict[str, PredictionSample] = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, PredictionSample):
            raise TypeError(f"predictions[{index}] must be a PredictionSample")
        if prediction.question_id in indexed:
            raise ValueError(
                f"duplicate question ID in predictions: {prediction.question_id}"
            )
        indexed[prediction.question_id] = prediction
    return indexed


def _require_exact_coverage(
    references: dict[str, LegalIRReference],
    predictions: dict[str, PredictionSample],
) -> None:
    missing = sorted(set(references).difference(predictions))
    unexpected = sorted(set(predictions).difference(references))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing IDs: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected IDs: " + ", ".join(unexpected))
        message = "; ".join(details)
        raise ValueError(f"prediction IDs do not match LegalIR references ({message})")


def document_ids_from_hits(
    prediction: PredictionSample,
    *,
    chunk_depth: int | None = None,
    max_documents: int | None = None,
) -> list[str]:
    """Collapse a chunk prefix into unique document IDs in rank order."""

    if chunk_depth is not None and chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive or None")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive or None")

    document_ids: list[str] = []
    seen: set[str] = set()
    hits = prediction.hits if chunk_depth is None else prediction.hits[:chunk_depth]
    for hit in hits:
        if hit.doc_id in seen:
            continue
        seen.add(hit.doc_id)
        document_ids.append(hit.doc_id)
        if max_documents is not None and len(document_ids) >= max_documents:
            break
    return document_ids


def evaluate_legal_ir_document_diagnostics(
    references: Sequence[LegalIRReference],
    predictions: Sequence[PredictionSample],
    *,
    candidate_predictions: Sequence[PredictionSample] | None = None,
    final_document_limit: int = OFFICIAL_MAX_DOCUMENTS,
    candidate_chunk_depths: Sequence[int] = DEFAULT_CANDIDATE_CHUNK_DEPTHS,
) -> dict[str, Any]:
    """Compute official document scores plus deep candidate-pool diagnostics.

    ``final_document_limit`` is intentionally capped at the official maximum.
    In contrast, candidate diagnostics use all requested chunk depths and only
    collapse duplicate document IDs; they do not impose an answer-length
    penalty.
    """

    if (
        isinstance(final_document_limit, bool)
        or not isinstance(final_document_limit, int)
        or not 1 <= final_document_limit <= OFFICIAL_MAX_DOCUMENTS
    ):
        raise ValueError(
            "final_document_limit must be an integer between 1 and "
            f"{OFFICIAL_MAX_DOCUMENTS}"
        )
    normalized_depths = sorted(set(candidate_chunk_depths))
    if not normalized_depths or any(
        isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0
        for depth in normalized_depths
    ):
        raise ValueError("candidate_chunk_depths must contain positive integers")

    by_reference = _index_references(references)
    by_prediction = _index_predictions(predictions)
    by_candidate_prediction = _index_predictions(
        predictions if candidate_predictions is None else candidate_predictions
    )
    _require_exact_coverage(by_reference, by_prediction)
    _require_exact_coverage(by_reference, by_candidate_prediction)
    ordered_references = list(references)
    if OPTIONAL_CANDIDATE_CHUNK_DEPTH not in normalized_depths and all(
        len(by_candidate_prediction[reference.id].hits)
        >= OPTIONAL_CANDIDATE_CHUNK_DEPTH
        for reference in ordered_references
    ):
        normalized_depths.append(OPTIONAL_CANDIDATE_CHUNK_DEPTH)

    final_predictions = [
        LegalIRPrediction(
            id=reference.id,
            documents=document_ids_from_hits(
                by_prediction[reference.id],
                max_documents=final_document_limit,
            ),
        )
        for reference in ordered_references
    ]
    official = evaluate_legal_ir(ordered_references, final_predictions)
    official_by_id = {item.id: item for item in official.per_query}

    candidate_recall: dict[str, float] = {}
    mean_unique_documents: dict[str, float] = {}
    complete_query_counts: dict[str, int] = {}
    for depth in normalized_depths:
        recalls: list[float] = []
        unique_counts: list[int] = []
        complete_count = 0
        for reference in ordered_references:
            prediction = by_candidate_prediction[reference.id]
            candidate_documents = document_ids_from_hits(
                prediction,
                chunk_depth=depth,
            )
            gold = set(reference.gold_documents)
            recalls.append(len(gold.intersection(candidate_documents)) / len(gold))
            unique_counts.append(len(candidate_documents))
            if len(prediction.hits) >= depth:
                complete_count += 1
        key = str(depth)
        candidate_recall[key] = sum(recalls) / len(recalls)
        mean_unique_documents[key] = sum(unique_counts) / len(unique_counts)
        complete_query_counts[key] = complete_count

    gold_absent_from_candidate_count = 0
    gold_present_but_final_dropped_count = 0
    multi_gold_recalls: list[float] = []
    per_query: list[dict[str, Any]] = []
    for reference, final_prediction in zip(ordered_references, final_predictions):
        all_candidate_documents = set(
            document_ids_from_hits(by_candidate_prediction[reference.id])
        )
        final_documents = set(final_prediction.documents)
        gold = set(reference.gold_documents)
        absent = gold.difference(all_candidate_documents)
        dropped = gold.intersection(all_candidate_documents).difference(final_documents)
        gold_absent_from_candidate_count += len(absent)
        gold_present_but_final_dropped_count += len(dropped)
        diagnostic = official_by_id[reference.id]
        if len(reference.gold_documents) > 1:
            multi_gold_recalls.append(diagnostic.recall)
        per_query.append(
            {
                "id": reference.id,
                "candidate_chunk_count": len(
                    by_candidate_prediction[reference.id].hits
                ),
                "final_documents": list(final_prediction.documents),
                "gold_absent_from_candidate_documents": sorted(absent),
                "gold_present_but_final_dropped_documents": sorted(dropped),
                "official_recall": diagnostic.recall,
                "official_precision": diagnostic.precision,
            }
        )

    report: dict[str, Any] = {
        "schema_version": "legal-ir-document-diagnostics-v1",
        "official": official.model_dump(mode="json"),
        "official_recall": official.aggregate.recall,
        "official_precision": official.aggregate.precision,
        "final_document_limit": final_document_limit,
        "candidate_doc_recall_at_k": candidate_recall,
        "mean_unique_docs_at_chunk_depth": mean_unique_documents,
        "candidate_depth_complete_query_count": complete_query_counts,
        "gold_absent_from_candidate_count": gold_absent_from_candidate_count,
        "gold_present_but_final_dropped_count": gold_present_but_final_dropped_count,
        "multi_gold_recall": (
            sum(multi_gold_recalls) / len(multi_gold_recalls)
            if multi_gold_recalls
            else None
        ),
        "multi_gold_sample_count": len(multi_gold_recalls),
        "per_query": per_query,
    }
    for depth_key, value in candidate_recall.items():
        report[f"candidate_doc_recall_at_{depth_key}"] = value
    for depth_key, value in mean_unique_documents.items():
        report[f"mean_unique_docs_at_chunk_depth_{depth_key}"] = value
    return report


def compare_legal_ir_document_diagnostics(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Build a compact before/after delta for one aligned candidate pool."""

    before_official = before.get("official")
    after_official = after.get("official")
    if not isinstance(before_official, dict) or not isinstance(after_official, dict):
        raise ValueError("both diagnostics must contain an official report")
    before_aggregate = before_official.get("aggregate")
    after_aggregate = after_official.get("aggregate")
    if not isinstance(before_aggregate, dict) or not isinstance(after_aggregate, dict):
        raise ValueError("both diagnostics must contain official aggregates")
    if before_aggregate.get("sample_count") != after_aggregate.get("sample_count"):
        raise ValueError("document diagnostics must cover the same sample count")
    return {
        "schema_version": "legal-ir-document-diagnostics-comparison-v1",
        "sample_count": before_aggregate["sample_count"],
        "before": {
            "official_recall": before["official_recall"],
            "official_precision": before["official_precision"],
            "multi_gold_recall": before["multi_gold_recall"],
        },
        "after": {
            "official_recall": after["official_recall"],
            "official_precision": after["official_precision"],
            "multi_gold_recall": after["multi_gold_recall"],
        },
        "delta": {
            "official_recall": after["official_recall"] - before["official_recall"],
            "official_precision": after["official_precision"]
            - before["official_precision"],
            "multi_gold_recall": (
                None
                if before["multi_gold_recall"] is None
                or after["multi_gold_recall"] is None
                else after["multi_gold_recall"] - before["multi_gold_recall"]
            ),
        },
    }


__all__ = [
    "DEFAULT_CANDIDATE_CHUNK_DEPTHS",
    "OPTIONAL_CANDIDATE_CHUNK_DEPTH",
    "compare_legal_ir_document_diagnostics",
    "document_ids_from_hits",
    "evaluate_legal_ir_document_diagnostics",
]
