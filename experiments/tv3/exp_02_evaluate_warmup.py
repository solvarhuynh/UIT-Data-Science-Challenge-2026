"""Warmup evaluation script for Task 2 (LegalQA).

Measures METEOR and ROUGE-L scores for QAEngine answers against the BTC
reference answers in ``data/warmup/warmup_task2.json``.

Cross-validation design (prevents prompt overfitting):
  - 80% of warmup questions are used for prompt tuning (DEV split).
  - 20% are held out and never looked at during tuning (TEST split).
  - Only compare METEOR on TEST split to decide if a prompt change is good.

Usage (from repo root with .venv active)::

    python experiments/tv3/exp_02_evaluate_warmup.py
    python experiments/tv3/exp_02_evaluate_warmup.py --split test
    python experiments/tv3/exp_02_evaluate_warmup.py --prompt-version legal_qa_v2
    python experiments/tv3/exp_02_evaluate_warmup.py --max-samples 20 --split dev

Dependencies (install once)::

    pip install nltk rouge-score
    python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo root discovery – allow running from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from udsc2026.contracts.retrieval import RetrievalHit  # noqa: E402
from udsc2026.qa.prompt_builder import PromptBuilder  # noqa: E402
from udsc2026.qa.qa_engine import QAEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WARMUP_PATH = _REPO_ROOT / "data" / "warmup" / "warmup_task2.json"
_PROMPTS_ROOT = _REPO_ROOT / "prompts"
_RANDOM_SEED = 42
_DEV_RATIO = 0.8  # 80% dev, 20% held-out test


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_meteor(hypothesis: str, reference: str) -> float:
    """Compute METEOR score between hypothesis and reference strings.

    Args:
        hypothesis: Model-generated answer.
        reference: Gold reference answer from BTC.

    Returns:
        METEOR score in [0, 1].
    """
    if not hypothesis or not reference:
        return 0.0
    try:
        from nltk.translate.meteor_score import single_meteor_score
        import nltk

        try:
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
        except Exception:
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower().split()

        return float(single_meteor_score(ref_tokens, hyp_tokens))
    except Exception as exc:
        logger.debug("METEOR score error: %s", exc)
        return 0.0


def _compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 between hypothesis and reference strings.

    Requires ``rouge-score`` package.

    Args:
        hypothesis: Model-generated answer.
        reference: Gold reference answer from BTC.

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        result = scorer.score(reference, hypothesis)
        return float(result["rougeL"].fmeasure)
    except ImportError:
        logger.error(
            "rouge-score is not installed. Run: pip install rouge-score"
        )
        return 0.0


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_warmup(path: Path) -> list[dict]:
    """Load warmup_task2.json and return a flat list of samples.

    Args:
        path: Path to warmup_task2.json.

    Returns:
        List of dicts with keys ``id``, ``question``, ``answer``.
    """
    with path.open(encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    samples = []
    for sample_id, content in raw.items():
        samples.append(
            {
                "id": sample_id,
                "question": content["question"],
                "answer": content["answer"],
            }
        )
    logger.info("Loaded %d warmup samples from '%s'.", len(samples), path)
    return samples


def _split_warmup(
    samples: list[dict],
    dev_ratio: float = _DEV_RATIO,
    seed: int = _RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    """Split samples into reproducible DEV and TEST splits.

    The split is deterministic: same seed always produces the same split so
    different prompt versions can be compared on the same held-out TEST set.

    Args:
        samples: Full warmup sample list.
        dev_ratio: Fraction of samples assigned to the dev (tuning) split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(dev_samples, test_samples)``.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * dev_ratio)
    dev, test = shuffled[:cut], shuffled[cut:]
    logger.info(
        "Split: %d DEV (tuning) | %d TEST (held-out).", len(dev), len(test)
    )
    return dev, test


# ---------------------------------------------------------------------------
# Mock LLM for fast offline evaluation
# ---------------------------------------------------------------------------

class _MockLLM:
    """Lightweight mock that returns the question back as a 'no-context' answer.

    Useful for quickly verifying the evaluation pipeline before running the
    real GPU-heavy model.
    """

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return "Dựa trên dữ liệu pháp lý được cung cấp, không có đủ căn cứ để trả lời câu hỏi này."


def _build_mock_context(question: str) -> list[RetrievalHit]:
    """Return a single dummy RetrievalHit for testing without a real retriever.

    In production this list comes from TV2's retrieval pipeline.

    Args:
        question: The question text (used as a stand-in passage).

    Returns:
        List with one placeholder ``RetrievalHit``.
    """
    return [
        RetrievalHit(
            chunk_id="mock-chunk-0",
            doc_id="mock-doc-0",
            text=f"[Văn bản thử nghiệm cho câu hỏi: {question}]",
            score=1.0,
        )
    ]


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

import asyncio

def evaluate(
    samples: list[dict],
    engine: QAEngine,
    prompt_version: str,
    rag_template: str,
    use_mock_context: bool = True,
    max_samples: Optional[int] = None,
    min_viable_samples: int = 20,
    stuck_busy_limit: int = 10,
) -> dict:
    """Run QAEngine on each sample and aggregate METEOR / ROUGE-L scores.

    Args:
        samples: List of ``{"id", "question", "answer"}`` dicts.
        engine: Initialised ``QAEngine`` instance.
        prompt_version: System prompt version to use (e.g. ``"legal_qa_v2"``).
        rag_template: RAG template name (e.g. ``"default_rag_v2"``).
        use_mock_context: When True use a placeholder context (no real retriever).
        max_samples: Cap the number of samples to evaluate (for quick checks).
        min_viable_samples: Abort early when zero generations have succeeded
            after this many samples (the model cannot generate at all).
        stuck_busy_limit: Abort early when this many consecutive "busy" failures
            occur (the LLM client is wedged after a timeout).

    Returns:
        Dict with ``meteor_mean``, ``rouge_l_mean``, per-sample ``results`` and
        success/failure diagnostics.
    """
    if max_samples is not None:
        samples = samples[:max_samples]

    meteor_scores: list[float] = []
    rouge_l_scores: list[float] = []
    success_meteor: list[float] = []
    success_rouge: list[float] = []
    results: list[dict] = []
    failures: list[dict] = []
    success_count = 0
    consecutive_busy = 0
    aborted = False
    abort_reason: Optional[str] = None

    for i, sample in enumerate(samples, start=1):
        qid = sample["id"]
        question = sample["question"]
        reference = sample["answer"]

        # Nạp context cho QAEngine:
        # - Khi dùng MockLLM: nạp context giả lập đơn giản (chỉ để test pipeline).
        # - Khi dùng mô hình thật (--use-real-model): dùng tạm nội dung đáp án chuẩn
        #   của BTC làm "sách luật" để Qwen3 có ngữ cảnh đọc và sinh câu trả lời.
        #   Đây là giải pháp tạm thời trước khi TV2 (Retrieval) được tích hợp.
        if use_mock_context:
            contexts = _build_mock_context(question)
        else:
            contexts = _build_mock_context(reference)

        try:
            response = asyncio.run(
                engine.generate_answer(
                    question=question,
                    contexts=contexts,
                    prompt_version=prompt_version,
                    rag_template=rag_template,
                )
            )
            hypothesis = response.answer
            success_count += 1
            consecutive_busy = 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("QAEngine failed for id=%s: %s", qid, exc)
            error_type = _classify_error(exc)
            failures.append({"id": qid, "error": str(exc), "type": error_type})
            consecutive_busy = consecutive_busy + 1 if error_type == "busy" else 0
            hypothesis = ""

            if error_type == "busy" and consecutive_busy >= stuck_busy_limit:
                aborted = True
                abort_reason = (
                    f"LLM client is wedged: {consecutive_busy} consecutive 'busy' "
                    f"failures. Aborting to skip the remaining "
                    f"{len(samples) - i} samples that cannot generate."
                )
                logger.error(abort_reason)
                break

        m = _compute_meteor(hypothesis, reference)
        r = _compute_rouge_l(hypothesis, reference)
        meteor_scores.append(m)
        rouge_l_scores.append(r)
        if hypothesis:
            success_meteor.append(m)
            success_rouge.append(r)

        results.append(
            {
                "id": qid,
                "question": question,
                "reference": reference,
                "hypothesis": hypothesis,
                "meteor": round(m, 4),
                "rouge_l": round(r, 4),
            }
        )

        if i % 20 == 0 or i == len(samples):
            current_meteor = sum(meteor_scores) / len(meteor_scores)
            current_rouge = sum(rouge_l_scores) / len(rouge_l_scores)
            logger.info(
                "[%d/%d] Running avg — METEOR: %.4f | ROUGE-L: %.4f",
                i,
                len(samples),
                current_meteor,
                current_rouge,
            )

        if success_count == 0 and i >= min_viable_samples:
            aborted = True
            abort_reason = (
                f"No successful generation after {i} samples. The model cannot "
                "generate with the current device/dtype/quantize configuration "
                "(check VRAM). Aborting the remaining samples."
            )
            logger.error(abort_reason)
            break

    n = len(meteor_scores)
    return {
        "n_samples": n,
        "success_count": success_count,
        "failure_count": len(failures),
        "failures": failures,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "meteor_mean": round(sum(meteor_scores) / n, 4) if n else 0.0,
        "rouge_l_mean": round(sum(rouge_l_scores) / n, 4) if n else 0.0,
        "meteor_mean_success": (
            round(sum(success_meteor) / len(success_meteor), 4)
            if success_meteor
            else 0.0
        ),
        "rouge_l_mean_success": (
            round(sum(success_rouge) / len(success_rouge), 4)
            if success_rouge
            else 0.0
        ),
        "prompt_version": prompt_version,
        "rag_template": rag_template,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _classify_error(exc: BaseException) -> str:
    """Bucket a generation exception into a coarse category for reporting."""
    message = str(exc).lower()
    if "still processing" in message:
        return "busy"
    if "out of memory" in message:
        return "oom"
    if isinstance(exc, TimeoutError) or "exceeded" in message:
        return "timeout"
    return "other"


def _count_failure_types(failures: list[dict]) -> dict[str, int]:
    """Count failures grouped by their classified error type."""
    counts: dict[str, int] = {}
    for failure in failures:
        kind = str(failure.get("type", "other"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _missing_nltk_resources() -> list[str]:
    """Return the nltk data resources (punkt/wordnet) that are not installed."""
    import nltk

    missing: list[str] = []
    for resource in ("tokenizers/punkt", "corpora/wordnet"):
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)
    return missing


def _print_score_zero_hint(report: dict) -> None:
    """Print an accurate diagnostic instead of a misleading one when scores are 0."""
    print("\n  ⚠️  METEOR = 0.0 — nguyên nhân cần kiểm tra:")
    if report.get("success_count", 0) == 0:
        print("     Không có câu trả lời nào được sinh thành công.")
        failure_types = _count_failure_types(report.get("failures", []))
        if failure_types:
            print("     Phân loại lỗi generation:")
            for kind, count in sorted(failure_types.items(), key=lambda kv: -kv[1]):
                print(f"       - {kind}: {count}")
        print(
            "     Điều này thường do cấu hình model (device/dtype/quantize) "
            "không phù hợp VRAM, không phải do nltk/rouge-score."
        )
        return
    missing = _missing_nltk_resources()
    if missing:
        print("     nltk/rouge-score đã cài nhưng thiếu dữ liệu nltk sau:")
        print(f"       {', '.join(missing)}")
        print(
            '     Chạy: python -c "import nltk; nltk.download(\'punkt\'); '
            "nltk.download('wordnet')\""
        )
    else:
        print(
            "     nltk/rouge-score sẵn sàng; xem các hypothesis để đánh giá "
            "chất lượng câu trả lời."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QAEngine on warmup_task2.json using METEOR and ROUGE-L."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help=(
            "Which split to evaluate on. "
            "'dev' = 80%% tuning split (default). "
            "'test' = 20%% held-out split (use sparingly). "
            "'all' = full warmup set."
        ),
    )
    parser.add_argument(
        "--prompt-version",
        default="legal_qa_v2",
        help="System prompt version (file under prompts/system/ without .md).",
    )
    parser.add_argument(
        "--rag-template",
        default="default_rag_v2",
        help="RAG template name (file under prompts/rag_templates/ without .md).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        default=False,
        help=(
            "Load the real LLM from ./models/qwen3-legal instead of the mock. "
            "Requires CUDA and model weights."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for the real model: cuda, cpu or mps.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="float16",
        help="Weight dtype when --quantize is 'none'.",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="4bit",
        help=(
            "bitsandbytes quantization so the model fits small GPUs "
            "(defaults to '4bit' for 4 GB GPUs like GTX 1650)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-generation timeout in seconds.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save per-sample results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the evaluation script."""
    args = _parse_args()

    # ── Load warmup data ──────────────────────────────────────────────────
    if not _WARMUP_PATH.exists():
        logger.error(
            "Warmup file not found at '%s'. "
            "Download it from the BTC portal and place it in data/warmup/.",
            _WARMUP_PATH,
        )
        sys.exit(1)

    all_samples = _load_warmup(_WARMUP_PATH)
    dev_samples, test_samples = _split_warmup(all_samples)

    split_map = {"dev": dev_samples, "test": test_samples, "all": all_samples}
    samples = split_map[args.split]
    logger.info(
        "Evaluating on split='%s' (%d samples) | prompt='%s' | rag='%s'",
        args.split,
        len(samples),
        args.prompt_version,
        args.rag_template,
    )

    # ── Build QAEngine ────────────────────────────────────────────────────
    if args.use_real_model:
        from udsc2026.infrastructure.llm.client import LLMClient
        from udsc2026.infrastructure.llm.config import LLMConfig
        from udsc2026.qa.citation_parser import CitationParser

        llm_config = LLMConfig(
            model_path="./models/qwen3-legal",
            backend="transformers",
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantize,
            max_new_tokens=1024,
            temperature=0.1,
            timeout_seconds=args.timeout,
        )
        llm = LLMClient(llm_config)
        logger.info(
            "Real model loaded from '%s' (device=%s, dtype=%s, quantize=%s).",
            llm_config.model_path,
            args.device,
            args.dtype,
            args.quantize,
        )
    else:
        llm = _MockLLM()  # type: ignore[assignment]
        from udsc2026.qa.citation_parser import CitationParser
        logger.info("Using MockLLM (pass --use-real-model to use real weights).")

    from udsc2026.qa.citation_parser import CitationParser

    engine = QAEngine(
        llm_client=llm,  # type: ignore[arg-type]
        prompt_builder=PromptBuilder(prompts_root=_PROMPTS_ROOT),
        citation_parser=CitationParser(),
    )

    # ── Run evaluation ────────────────────────────────────────────────────
    report = evaluate(
        samples=samples,
        engine=engine,
        prompt_version=args.prompt_version,
        rag_template=args.rag_template,
        use_mock_context=not args.use_real_model,
        max_samples=args.max_samples,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EVALUATION REPORT — Split: {args.split.upper()}")
    print("=" * 60)
    print(f"  Prompt Version : {report['prompt_version']}")
    print(f"  RAG Template   : {report['rag_template']}")
    print(f"  Samples        : {report['n_samples']}")
    print(f"  Success        : {report['success_count']} / {report['n_samples']}")
    print(f"  Failed         : {report['failure_count']}")
    print(f"  METEOR (mean)  : {report['meteor_mean']:.4f}  ← Độ đo chính BTC")
    print(f"  ROUGE-L (mean) : {report['rouge_l_mean']:.4f}  ← Độ đo phụ BTC")
    if report["success_count"] > 0:
        print(
            "  METEOR (gen ok)  : "
            f"{report['meteor_mean_success']:.4f}  ← chỉ các mẫu sinh được"
        )
        print(
            "  ROUGE-L (gen ok) : "
            f"{report['rouge_l_mean_success']:.4f}  ← chỉ các mẫu sinh được"
        )
    print("=" * 60)

    if report.get("aborted"):
        print(f"\n  ⛔ Bị dừng sớm: {report['abort_reason']}")

    if report["meteor_mean"] == 0.0:
        _print_score_zero_hint(report)

    # ── Optionally save results ───────────────────────────────────────────
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        logger.info("Results saved to '%s'.", args.output_json)


if __name__ == "__main__":
    main()
