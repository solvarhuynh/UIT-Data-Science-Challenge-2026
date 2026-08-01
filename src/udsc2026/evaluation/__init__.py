"""Evaluation, reporting, submission, and synthetic benchmark utilities."""

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
from udsc2026.evaluation.synthetic_generator import (
    SyntheticQA,
    generate_synthetic_benchmark,
    load_legal_chunks,
    write_synthetic_benchmark,
)

__all__ = [
    "BenchmarkSample",
    "EvaluationComparison",
    "EvaluationReport",
    "PredictionSample",
    "SubmissionColumn",
    "SubmissionRow",
    "SubmissionSchema",
    "SyntheticQA",
    "aggregate_latencies",
    "compare_reports",
    "evaluate_predictions",
    "evaluate_retrieval",
    "generate_synthetic_benchmark",
    "load_benchmark",
    "load_legal_chunks",
    "load_predictions",
    "load_qa_responses",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "mean_rouge_l",
    "recall_at_k",
    "reciprocal_rank",
    "rouge_l_score",
    "validate_rerank_candidate_pools",
    "write_report",
    "write_report_bundle",
    "write_submission",
    "write_synthetic_benchmark",
]
