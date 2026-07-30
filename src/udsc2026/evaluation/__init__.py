"""Deterministic, model-independent evaluation utilities for LegalIR/LegalQA."""

from udsc2026.evaluation.evaluator import (
    compare_reports,
    evaluate_predictions,
    evaluate_retrieval,
    validate_rerank_candidate_pools,
)
from udsc2026.evaluation.loaders import (
    load_benchmark,
    load_predictions,
    load_qa_responses,
)
from udsc2026.evaluation.metrics import (
    aggregate_latencies,
    mean_recall_at_k,
    mean_reciprocal_rank,
    mean_rouge_l,
    recall_at_k,
    reciprocal_rank,
    rouge_l_score,
)
from udsc2026.evaluation.models import (
    BenchmarkSample,
    EvaluationComparison,
    EvaluationReport,
    PredictionSample,
)
from udsc2026.evaluation.reporting import write_report, write_report_bundle
from udsc2026.evaluation.submission import (
    SubmissionColumn,
    SubmissionRow,
    SubmissionSchema,
    write_submission,
)

__all__ = [
    "BenchmarkSample",
    "EvaluationComparison",
    "EvaluationReport",
    "PredictionSample",
    "SubmissionColumn",
    "SubmissionRow",
    "SubmissionSchema",
    "aggregate_latencies",
    "compare_reports",
    "evaluate_predictions",
    "evaluate_retrieval",
    "load_benchmark",
    "load_predictions",
    "load_qa_responses",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "mean_rouge_l",
    "recall_at_k",
    "reciprocal_rank",
    "rouge_l_score",
    "write_report",
    "write_report_bundle",
    "write_submission",
    "validate_rerank_candidate_pools",
]
