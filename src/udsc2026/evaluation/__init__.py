"""Shared evaluation interfaces used by the competition CLIs."""

from udsc2026.evaluation.evaluator import (
    compare_reports,
    evaluate_predictions,
)
from udsc2026.evaluation.retrieval_metrics import LegalIRReference, evaluate_legal_ir, load_warmup
from udsc2026.evaluation.retrieval_candidates import (
    AGGREGATION_METHODS,
    AggregatedDocument,
    DEFAULT_DENSE_RRF_K,
    DEFAULT_DENSE_SIGNAL_WEIGHT,
    ScoreField,
    aggregate_document_candidates,
    aggregate_to_legal_ir_prediction,
)
from udsc2026.evaluation.retrieval_diagnostics import (
    compare_legal_ir_document_diagnostics,
    document_ids_from_hits,
    evaluate_legal_ir_document_diagnostics,
)
from udsc2026.evaluation.loaders import load_benchmark, load_predictions
from udsc2026.evaluation.models import PredictionSample
from udsc2026.evaluation.reporting import (
    write_report_bundle,
    write_run_manifest,
)
