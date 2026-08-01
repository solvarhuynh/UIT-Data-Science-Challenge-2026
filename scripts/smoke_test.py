"""Fast, offline smoke checks for the TV5 runtime package.

The checks deliberately use a fake reranker scorer and tiny in-memory objects:
no model checkpoint, GPU, VectorDB, Redis, or network connection is required.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import yaml

MINIMUM_PYTHON = (3, 10)
REQUIRED_BACKEND_EXTRAS = {"llm", "rerank", "retrieval"}
REQUIRED_ENV_KEYS = {
    "BACKEND_EXTRAS",
    "BACKEND_PORT",
    "BM25_INDEX_PATH",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DEVICE",
    "EMBEDDING_MAX_LENGTH",
    "EMBEDDING_NORMALIZE_EMBEDDINGS",
    "FAISS_COLLECTION_NAME",
    "FAISS_INDEX_PATH",
    "LLM_BACKEND",
    "LLM_DEVICE",
    "LLM_DTYPE",
    "LLM_MAX_NEW_TOKENS",
    "LLM_REPETITION_PENALTY",
    "LLM_STREAM",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
    "LLM_TOP_P",
    "LEGAL_IR_SUBMISSION_PATH",
    "LEGAL_QA_SUBMISSION_PATH",
    "MODEL_EMBEDDER_PATH",
    "MODEL_LLM_PATH",
    "QDRANT_URL",
    "QDRANT_COLLECTION_NAME",
    "REDIS_URL",
    "RERANKER_BATCH_SIZE",
    "RERANKER_DEVICE",
    "RERANKER_ENABLED",
    "RERANKER_LOCAL_FILES_ONLY",
    "RERANKER_MAX_LENGTH",
    "RERANKER_TOP_N",
    "RERANKER_MODEL_PATH",
    "SUBMISSION_PATH",
    "UDSC2026_CONFIG_PATH",
    "VECTOR_DB_TYPE",
    "VECTOR_STORE_DIR",
}
REQUIRED_DOCKERIGNORE_PATTERNS = {
    ".cache/",
    ".git",
    "data/processed/",
    "data/raw/",
    "data/task1/",
    "data/task2/",
    "data/vector_store/",
    "frontend/",
    "models/",
    "**/node_modules/",
}


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one independently reportable smoke check."""

    name: str
    passed: bool
    detail: str


def _project_root(mode: str) -> Path:
    if mode == "container":
        return Path("/app")
    return Path(__file__).resolve().parents[1]


def _check_python(_: Path, mode: str) -> str:
    current = sys.version_info[:2]
    if current < MINIMUM_PYTHON:
        required = ".".join(map(str, MINIMUM_PYTHON))
        actual = ".".join(map(str, current))
        raise RuntimeError(f"Python {required}+ required; found {actual}")
    if mode == "container" and hasattr(os, "geteuid") and os.geteuid() == 0:
        raise RuntimeError("container runtime must not run as root")
    return f"Python {sys.version.split()[0]}; mode={mode}"


def _check_core_contract(_: Path, __: str) -> str:
    from udsc2026.contracts.retrieval import RetrievalHit

    hit = RetrievalHit(
        chunk_id="smoke-chunk",
        doc_id="smoke-doc",
        text="Điều khoản kiểm tra",
        metadata={"source": "smoke"},
        hybrid_score=0.75,
    )
    if hit.chunk_id != "smoke-chunk" or hit.metadata["source"] != "smoke":
        raise RuntimeError("RetrievalHit did not preserve required fields")
    return "udsc2026 package and RetrievalHit contract import successfully"


def _check_config(root: Path, _: str) -> str:
    config_path = root / "configs" / "base.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError("configs/base.yaml must contain a YAML mapping")
    missing = {"embedding", "llm", "vector_db", "hybrid", "reranker"} - config.keys()
    if missing:
        raise ValueError(f"configs/base.yaml missing sections: {sorted(missing)}")
    return f"loaded {config_path} with required sections"


def _parse_env_example(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if not key or any(character.isspace() for character in key):
            raise ValueError(f"{path}:{line_number}: invalid environment key")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key}")
        values[key] = value.strip()
    return values


def _check_env_example(root: Path, mode: str) -> str:
    if mode == "container":
        return "host-only .env.example validation skipped in container"

    env_path = root / ".env.example"
    if not env_path.is_file():
        raise FileNotFoundError(f"missing environment template: {env_path}")
    values = _parse_env_example(env_path)
    missing = REQUIRED_ENV_KEYS - values.keys()
    if missing:
        raise ValueError(f".env.example missing keys: {sorted(missing)}")
    empty = sorted(key for key in REQUIRED_ENV_KEYS if not values[key])
    if empty:
        raise ValueError(f".env.example has empty required values: {empty}")

    configured_extras = {
        extra.strip() for extra in values["BACKEND_EXTRAS"].split(",") if extra.strip()
    }
    missing_extras = REQUIRED_BACKEND_EXTRAS - configured_extras
    if missing_extras:
        raise ValueError(
            ".env.example BACKEND_EXTRAS missing production extras: "
            f"{sorted(missing_extras)}"
        )
    return f"validated {len(values)} unique environment variables"


def _check_dockerignore(root: Path, mode: str) -> str:
    if mode == "container":
        return "host-only .dockerignore validation skipped in container"

    ignore_path = root / ".dockerignore"
    patterns = {
        line.strip()
        for line in ignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = REQUIRED_DOCKERIGNORE_PATTERNS - patterns
    if missing:
        raise ValueError(f".dockerignore missing safety patterns: {sorted(missing)}")
    return "model weights, corpora, indexes, frontend, and Git data are excluded"


class _FakeRerankerClient:
    """Deterministic scorer implementing the RerankerClient protocol."""

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        del query, batch_size
        return [float(index) for index, _ in enumerate(documents)]


def _check_reranker(_: Path, __: str) -> str:
    reranker_module = importlib.import_module("udsc2026.retrieval.reranking")
    client_module = importlib.import_module("udsc2026.infrastructure.reranker")
    reranker_type = getattr(reranker_module, "CrossEncoderReranker")
    getattr(client_module, "CrossEncoderClient")
    getattr(client_module, "RerankerClient")

    from udsc2026.contracts.retrieval import RetrievalHit

    candidates = [
        RetrievalHit(
            chunk_id="lower",
            doc_id="doc-1",
            text="short",
            metadata={"citation": "Điều 1"},
            hybrid_score=0.9,
        ),
        RetrievalHit(
            chunk_id="higher",
            doc_id="doc-2",
            text="longer",
            metadata={"citation": "Điều 2"},
            hybrid_score=0.8,
        ),
    ]
    reranked = reranker_type(_FakeRerankerClient()).rerank(
        "câu hỏi smoke",
        candidates,
        top_n=1,
    )
    if len(reranked) != 1 or reranked[0].chunk_id != "higher":
        raise RuntimeError("fake-score reranking returned an unexpected order")
    if reranked[0].metadata != {"citation": "Điều 2"}:
        raise RuntimeError("reranking did not preserve citation metadata")
    if reranked[0].rank != 1 or reranked[0].rerank_score is None:
        raise RuntimeError("reranking did not assign rank/rerank_score")
    return "public reranker API works with an offline deterministic scorer"


def _check_evaluation(_: Path, __: str) -> str:
    module = importlib.import_module("udsc2026.evaluation")
    expected_symbols = {
        "BenchmarkSample",
        "EvaluationComparison",
        "EvaluationReport",
        "LegalIREvaluationReport",
        "LegalIRPrediction",
        "LegalIRReference",
        "LegalIRSubmissionError",
        "LegalIRSubmissionItem",
        "LegalQAEvaluationReport",
        "LegalQAPrediction",
        "LegalQASubmissionError",
        "LegalQASubmissionItem",
        "LegalQAWarmupSample",
        "PredictionSample",
        "SubmissionColumn",
        "SubmissionRow",
        "SubmissionSchema",
        "SyntheticQA",
        "aggregate_latencies",
        "compare_reports",
        "complete_legal_ir_rankings",
        "evaluate_legal_ir",
        "evaluate_legal_qa",
        "evaluate_predictions",
        "evaluate_retrieval",
        "evaluate_warmup_any_gold",
        "generate_synthetic_benchmark",
        "legal_ir_prediction_from_hits",
        "load_benchmark",
        "load_legal_chunks",
        "load_legal_ir_submission",
        "load_legal_qa_submission",
        "load_legal_qa_question_ids",
        "load_legal_qa_warmup",
        "load_predictions",
        "load_qa_responses",
        "load_warmup",
        "mean_recall_at_k",
        "mean_reciprocal_rank",
        "mean_rouge_l",
        "meteor_diagnostic_score",
        "recall_at_k",
        "reciprocal_rank",
        "rouge_l_score",
        "rouge_l_f1_score",
        "validate_legal_ir_submission",
        "validate_legal_qa_submission",
        "validate_rerank_candidate_pools",
        "write_legal_ir_submission",
        "write_legal_qa_submission",
        "write_report",
        "write_report_bundle",
        "write_submission",
        "write_synthetic_benchmark",
    }
    declared_exports = set(getattr(module, "__all__", ()))
    missing = sorted(
        symbol
        for symbol in expected_symbols
        if symbol not in declared_exports or not hasattr(module, symbol)
    )
    if missing:
        raise AttributeError(f"udsc2026.evaluation missing public API: {missing}")

    reference_type = getattr(module, "LegalIRReference")
    prediction_type = getattr(module, "LegalIRPrediction")
    report = getattr(module, "evaluate_legal_ir")(
        [
            reference_type(id="q1", gold_document="gold-1"),
            reference_type(id="q2", gold_document="gold-2"),
        ],
        [
            prediction_type(id="q1", documents=["gold-1", "a", "b"]),
            prediction_type(id="q2", documents=["a", "b", "c", "gold-2"]),
        ],
    )
    if report.aggregate.mrr != 0.625 or report.aggregate.recall_at_3 != 0.5:
        raise RuntimeError("LegalIR document-level metrics returned unexpected values")
    items = getattr(module, "validate_legal_ir_submission")(
        {"q1": {"answer": ["gold-1", "a", "b"]}},
        expected_question_ids=["q1"],
        allowed_document_ids=["gold-1", "a", "b"],
        require_complete_ranking=True,
    )
    if items[0].documents != ("gold-1", "a", "b"):
        raise RuntimeError("LegalIR submission validation changed ranking order")
    try:
        getattr(module, "validate_legal_ir_submission")(
            [{"id": "q1", "documents": ["gold-1", "a", "b"]}]
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("LegalIR validator accepted the obsolete array wire shape")

    legal_qa_reference_type = getattr(module, "LegalQAWarmupSample")
    legal_qa_prediction_type = getattr(module, "LegalQAPrediction")
    legal_qa_report = getattr(module, "evaluate_legal_qa")(
        [
            legal_qa_reference_type(
                id="q1",
                question="Quy định nào?",
                answer="Điều 1 quy định quyền của người lao động.",
            )
        ],
        [
            legal_qa_prediction_type(
                id="q1",
                answer="Điều 1 quy định quyền của người lao động.",
            )
        ],
    )
    if legal_qa_report.aggregate.rouge_l != 1.0:
        raise RuntimeError("LegalQA diagnostic ROUGE-L returned an unexpected value")
    if not 0 < legal_qa_report.aggregate.meteor <= 1:
        raise RuntimeError("LegalQA diagnostic METEOR returned an unexpected value")
    legal_qa_items = getattr(module, "validate_legal_qa_submission")(
        {"q1": {"answer": "  Giữ nguyên\n ﻿"}},
        expected_question_ids=["q1"],
    )
    if legal_qa_items[0].answer != "  Giữ nguyên\n ﻿":
        raise RuntimeError("LegalQA submission validation changed answer text")
    try:
        getattr(module, "validate_legal_qa_submission")(
            [{"id": "q1", "answer": "obsolete"}]
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("LegalQA validator accepted the obsolete array wire shape")
    return (
        f"imported all {len(expected_symbols)} evaluation API symbols; "
        "LegalIR and LegalQA metric/submission contracts passed"
    )


def _check_prompt_and_api(root: Path, _: str) -> str:
    from udsc2026.qa.prompt_builder import PromptBuilder

    prompt = PromptBuilder(prompts_root=root / "prompts").build_prompt(
        "Câu hỏi smoke?",
        [],
    )
    if "Câu hỏi smoke?" not in prompt:
        raise RuntimeError("PromptBuilder did not render the configured prompt files")

    app_module = importlib.import_module("udsc2026.api.app")
    route_paths = {route.path for route in app_module.app.routes}
    missing_routes = {"/health", "/ready"} - route_paths
    if missing_routes:
        raise RuntimeError(f"FastAPI shell missing routes: {sorted(missing_routes)}")
    return "prompt files load and FastAPI health/readiness routes are registered"


def _check_container_dependencies(_: Path, mode: str) -> str:
    if mode != "container":
        return "container-only optional dependency imports skipped on host"

    modules = (
        "accelerate",
        "faiss",
        "pyvi",
        "qdrant_client",
        "rank_bm25",
        "sentence_transformers",
        "torch",
        "transformers",
    )
    for module_name in modules:
        importlib.import_module(module_name)
    return f"imported {len(modules)} packaged ML/retrieval dependencies"


def _run_check(
    name: str,
    check: Callable[[Path, str], str],
    root: Path,
    mode: str,
) -> CheckResult:
    try:
        detail = check(root, mode)
    except Exception as error:  # noqa: BLE001 - aggregate all smoke failures
        return CheckResult(
            name=name, passed=False, detail=f"{type(error).__name__}: {error}"
        )
    return CheckResult(name=name, passed=True, detail=detail)


def run_checks(mode: str) -> list[CheckResult]:
    """Run all lightweight checks and return their structured outcomes."""

    root = _project_root(mode)
    if mode == "host":
        source_root = str(root / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

    checks: list[tuple[str, Callable[[Path, str], str]]] = [
        ("python-runtime", _check_python),
        ("core-contract", _check_core_contract),
        ("base-config", _check_config),
        ("env-template", _check_env_example),
        ("docker-context", _check_dockerignore),
        ("reranker", _check_reranker),
        ("evaluation", _check_evaluation),
        ("prompt-api", _check_prompt_and_api),
        ("container-dependencies", _check_container_dependencies),
    ]
    return [_run_check(name, check, root, mode) for name, check in checks]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("host", "container"),
        default="host",
        help="Use host repository paths or the packaged /app container layout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable lines.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = run_checks(args.mode)

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False))
    else:
        for result in results:
            label = "PASS" if result.passed else "FAIL"
            print(f"[{label}] {result.name}: {result.detail}")

    failures = sum(not result.passed for result in results)
    if failures:
        print(
            f"TV5 smoke test failed: {failures}/{len(results)} check(s)",
            file=sys.stderr,
        )
        return 1
    if not args.json:
        print(f"TV5 smoke test passed: {len(results)}/{len(results)} check(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
