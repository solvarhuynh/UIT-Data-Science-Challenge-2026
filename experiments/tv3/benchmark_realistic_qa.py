"""Realistic Benchmark for TV3 QA Engine using Mock Hybrid Retrieval (BM25 + Dense BKAI).

Uses MockHybridRetriever to perform true Hybrid Search (Sparse BM25 + Dense Vector Embedding + RRF)
over parent context files (parents/*.jsonl) and feeds top-K hits to QAEngine.

Includes anti-OOM local controls:
--max_context_len: Truncates context text per hit (e.g. 1200 chars)
--max_new_tokens: Controls Qwen3 response token cap (default: 512)
--mode: 'hybrid', 'bm25', or 'dense'

Usage (Local / Kaggle):
    python experiments/tv3/benchmark_realistic_qa.py --limit 10 --mode hybrid --top_k 2
"""

import asyncio
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Thêm src vào Python path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

# Thêm đường dẫn rouge_score của BTC
BTC_QA_DIR = BASE_DIR / "docs/Scoring-Program-Task-LegalQA"
sys.path.insert(0, str(BTC_QA_DIR))

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.llm.client import LLMClient
from udsc2026.infrastructure.llm.config import LLMConfig
from udsc2026.qa.qa_engine import QAEngine

# Import mock hybrid retriever
sys.path.insert(0, str(BASE_DIR / "experiments/tv3"))
from mock_hybrid_retriever import MockHybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("benchmark_realistic_qa")

DEFAULT_EMBEDDING_MODEL = "huyydangg/DEk21_hcmute_embedding_v2"
DEFAULT_LLM_MODEL = "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Enable PyTorch VRAM memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_processing_manifest(parents_dir: Path) -> dict:
    manifest_path = parents_dir.parent / "metadata" / "processing_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_local_model(requested: str, default_id: str, local_dir: Path) -> str:
    if requested == default_id and local_dir.is_dir():
        return str(local_dir)
    return requested


def _truncate_contexts(
    contexts: list[RetrievalHit],
    max_context_len: int,
) -> list[RetrievalHit]:
    """Apply the prompt budget only after candidate reranking."""

    output: list[RetrievalHit] = []
    for hit in contexts:
        if len(hit.text) <= max_context_len:
            output.append(hit)
            continue
        output.append(
            hit.model_copy(
                deep=True,
                update={"text": hit.text[:max_context_len].rstrip() + "..."},
            )
        )
    return output


def _configure_utf8_stdio() -> None:
    """Keep Vietnamese CLI help/logs readable on legacy Windows code pages."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _rerank_candidates(
    reranker: Any | None,
    question: str,
    contexts: list[RetrievalHit],
    *,
    top_k: int,
    allow_fallback: bool,
) -> tuple[list[RetrievalHit], bool, str | None]:
    """Rerank or return a bounded, explicitly reported fallback."""

    if reranker is None or len(contexts) <= 1:
        return contexts[:top_k], False, None
    try:
        output = reranker.rerank(question, candidates=contexts, top_n=top_k)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError("TV5 reranking failed") from exc
        return contexts[:top_k], False, f"{type(exc).__name__}: {exc}"
    return output, True, None


def _ensure_metric_dependencies() -> None:
    """Fail before model work when BTC metric dependencies are unavailable."""

    try:
        import nltk
        from rouge_score import rouge_scorer as _rouge_scorer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "BTC metrics require the GPU extra: pip install -e .[gpu]"
        ) from exc

    for resource, package in (
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Unable to download NLTK resource: {package}")


def calculate_btc_scores(predictions: dict[str, str], references: dict[str, str]) -> dict[str, float]:
    """Tính điểm ROUGE-L và METEOR theo đúng thuật toán scoring.py của BTC."""
    _ensure_metric_dependencies()
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    rouge_scoring = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    qids = list(predictions.keys())
    rouge_scores = []
    meteor_scores = []

    for qid in qids:
        pred_text = str(predictions[qid])
        gold_text = str(references[qid])
        r_score = rouge_scoring.score(gold_text, pred_text)['rougeL'].fmeasure
        rouge_scores.append(r_score)
        m_score = meteor_score([gold_text.split()], pred_text.split())
        meteor_scores.append(m_score)

    mean_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    mean_meteor = float(np.mean(meteor_scores)) if meteor_scores else 0.0
    return {"rougeL": mean_rouge, "meteor": mean_meteor}


async def main() -> None:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run TV3 Realistic Hybrid Benchmark")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng câu hỏi cần test (0 = tất cả)")
    parser.add_argument("--top_k", type=int, default=3, help="Số đoạn luật truyền cho LLM (default: 3)")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "bm25", "dense"], help="Retrieval mode")
    parser.add_argument(
        "--max_docs",
        type=int,
        default=0,
        help="Giới hạn số parent records nạp (0 = nạp toàn bộ corpus)",
    )
    parser.add_argument("--max_context_len", type=int, default=1200, help="Giới hạn ký tự tối đa cho mỗi context (chống OOM)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Giới hạn số token Qwen3 sinh ra (chống OOM)")
    parser.add_argument("--dataset", type=str, default="", help="Đường dẫn file train.json")
    parser.add_argument("--parents_dir", type=str, default="", help="Đường dẫn thư mục parents chứa file jsonl")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(BASE_DIR / "artifacts/tv3/cache"),
        help="Thư mục cache ngoài data/processed_v3",
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_LLM_MODEL, help="Tên model Qwen trên HuggingFace hoặc đường dẫn local")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Tên model embedding trên HuggingFace hoặc đường dẫn local")
    parser.add_argument("--quantization", type=str, default="none", choices=["4bit", "8bit", "none"], help="Mức nén quantization (Kaggle T4 x2 khuyên dùng 'none')")
    parser.add_argument("--device", type=str, default="cuda", help="cuda hoặc cpu")
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="legal_qa_v2",
        help="System prompt version (file under prompts/system/ without .md).",
    )
    parser.add_argument(
        "--rag-template",
        type=str,
        default="default_rag_v2",
        help="RAG template name (file under prompts/rag_templates/ without .md).",
    )
    parser.add_argument("--use_reranker", action="store_true", help="Bật Cross-Encoder Reranker của TV5")
    parser.add_argument("--reranker_model", type=str, default=DEFAULT_RERANKER_MODEL, help="Mô hình Reranker của TV5")
    parser.add_argument("--candidate_k", type=int, default=50, help="Số ứng viên đưa vào reranker")
    parser.add_argument("--decompose_queries", action="store_true", help="Bật thử nghiệm tách sub-query (mặc định tắt)")
    parser.add_argument("--allow_remote_reranker", action="store_true", help="Cho phép tải reranker từ Hugging Face nếu chưa có local")
    parser.add_argument("--allow_reranker_fallback", action="store_true", help="Cho phép fallback RRF khi reranker lỗi; mặc định fail-fast")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE_DIR / "experiments/tv3/hybrid_benchmark_report.json",
        help="Đường dẫn báo cáo JSON; đặt tên riêng cho mỗi prompt để so sánh.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    if args.candidate_k < args.top_k:
        parser.error("--candidate_k must be greater than or equal to --top_k")
    if args.max_docs < 0:
        parser.error("--max_docs must be non-negative")
    if args.max_context_len <= 0 or args.max_new_tokens <= 0:
        parser.error("context and generation limits must be positive")
    _ensure_metric_dependencies()

    # 1. Load train dataset
    train_path = Path(args.dataset) if args.dataset else None
    if not train_path or not train_path.exists():
        candidates = [
            BASE_DIR / "data/raw/btc/LegalQA/train.json",
            BASE_DIR / "data/task2/public/train.json",
            BASE_DIR / "data/task2/warmup.json",
            Path("/kaggle/input/datasets/phamthequan/uit-ds-task2-test/train.json"),
        ]
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for train_match in kaggle_input.glob("**/train.json"):
                candidates.insert(0, train_match)
            for warmup_match in kaggle_input.glob("**/warmup.json"):
                candidates.append(warmup_match)
        for cand in candidates:
            if cand and cand.exists():
                train_path = cand
                break

    if not train_path or not train_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy file train.json hoặc warmup.json! "
            "Vui lòng truyền --dataset <đường_dẫn_file>"
        )

    logger.info("Đang đọc dữ liệu tham chiếu từ: %s", train_path)
    with open(train_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    qids = list(raw_data.keys())
    if args.limit > 0:
        qids = qids[:args.limit]
    logger.info("📝 Số câu hỏi cần test: %d", len(qids))

    # 2. Khởi tạo MockHybridRetriever (BM25 + Dense Vector + RRF)
    parents_dir = (
        Path(args.parents_dir)
        if args.parents_dir
        else BASE_DIR / "data/processed_v3/parents"
    )
    embedding_model_path = _resolve_local_model(
        args.embedding_model,
        DEFAULT_EMBEDDING_MODEL,
        BASE_DIR / "models/dek21-v2",
    )

    logger.info("⏳ Đang khởi tạo MockHybridRetriever với embedding model='%s' (mode=%s)...", embedding_model_path, args.mode)
    retriever = MockHybridRetriever(
        parents_dir=parents_dir,
        embedding_model_path=embedding_model_path,
        device=args.device if torch.cuda.is_available() else "cpu",
        cache_dir=args.cache_dir,
        max_docs=args.max_docs,
        enable_dense=args.mode != "bm25",
        enable_query_decomposition=args.decompose_queries,
    )
    effective_parents_dir = retriever.parents_dir
    processing_manifest = _load_processing_manifest(effective_parents_dir)

    effective_device = args.device if torch.cuda.is_available() else "cpu"

    # 3. Khởi tạo CrossEncoderReranker chính thức của TV5 (nếu được bật)
    tv5_reranker = None
    reranker_error = None
    reranker_success_count = 0
    reranker_fallback_count = 0
    reranker_model_path = _resolve_local_model(
        args.reranker_model,
        DEFAULT_RERANKER_MODEL,
        BASE_DIR / "models/reranker",
    )
    if args.use_reranker:
        try:
            from udsc2026.infrastructure.reranker import CrossEncoderClient
            from udsc2026.retrieval.reranking import CrossEncoderReranker

            logger.info(
                "⏳ Đang cấu hình CrossEncoderReranker TV5 từ '%s'...",
                reranker_model_path,
            )
            client = CrossEncoderClient(
                reranker_model_path,
                device=effective_device,
                batch_size=8,
                max_length=1024,
                local_files_only=(
                    Path(reranker_model_path).is_dir()
                    or not args.allow_remote_reranker
                ),
                use_fp16=effective_device.startswith("cuda"),
            )
            tv5_reranker = CrossEncoderReranker(client)
            logger.info("✅ Reranker đã cấu hình; weights sẽ được lazy-load khi chấm.")
        except Exception as exc:
            reranker_error = f"{type(exc).__name__}: {exc}"
            if not args.allow_reranker_fallback:
                raise RuntimeError("Không thể cấu hình TV5 reranker") from exc
            reranker_fallback_count += 1
            logger.warning("Không thể cấu hình TV5 Reranker: %s", exc)

    # 4. Khởi tạo Qwen LLM
    model_path = _resolve_local_model(
        args.model_path,
        DEFAULT_LLM_MODEL,
        BASE_DIR / "models/qwen3-legal",
    )

    logger.info("⏳ Đang khởi tạo LLM từ '%s' (quantization=%s)...", model_path, args.quantization)
    llm_config = LLMConfig(
        model_path=model_path,
        backend="transformers",
        device=effective_device,
        dtype="float16",
        quantization=args.quantization if torch.cuda.is_available() else "none",
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=300.0,
    )
    llm_client = LLMClient(llm_config)
    qa_engine = QAEngine(llm_client=llm_client, use_chat_template=True)

    predictions: dict[str, str] = {}
    references: dict[str, str] = {}
    failures: dict[str, str] = {}
    skipped: dict[str, str] = {}
    retrieval_traces: dict[str, dict] = {}
    start_time = time.monotonic()
    success_count = 0

    print("\n" + "=" * 70)
    reranker_label = " + TV5 Reranker" if args.use_reranker else ""
    print(
        "🚀 BENCHMARK THỰC TẾ HYBRID "
        f"(BM25 + Dense {embedding_model_path}{reranker_label} + LLM)"
    )
    print(
        f"   mode={args.mode} | top_k={args.top_k} | use_reranker={args.use_reranker} | "
        f"prompt={args.prompt_version}/{args.rag_template} | "
        f"quantization={args.quantization} | limit={args.limit}"
    )
    print("=" * 70)

    for idx, qid in enumerate(qids, start=1):
        item = raw_data[qid]
        question = item["question"]
        gold_answer = item["answer"]

        if not gold_answer or not gold_answer.strip():
            skipped[qid] = "empty_gold_answer"
            continue

        # 5. Retrieve a reproducible candidate pool before optional reranking.
        search_k = args.candidate_k if tv5_reranker is not None else args.top_k
        retrieved_docs = retriever.search(question, top_k=search_k, mode=args.mode)

        # 6. Preserve full parent hits for fair reranking. Prompt truncation is
        # applied only after the reranker has selected its final top-k.
        contexts = []
        for rank, doc in enumerate(retrieved_docs):
            raw_text = str(doc.get("text") or "").strip()
            if not raw_text:
                continue
            parent_id = str(doc.get("parent_id") or "").strip() or None
            metadata = doc.get("metadata")

            hit = RetrievalHit(
                chunk_id=parent_id or f"ret_{qid}_{rank}",
                parent_id=parent_id,
                doc_id=str(doc.get("doc_id") or "unknown"),
                text=raw_text,
                score=doc.get("retrieval_score", 0.0),
                source=doc.get("source"),
                law_name=doc.get("law_name"),
                article=doc.get("article"),
                clause=doc.get("clause"),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
            contexts.append(hit)

        retrieval_traces[qid] = {
            "before": [
                {
                    "chunk_id": hit.chunk_id,
                    "parent_id": hit.parent_id,
                    "score": hit.score,
                }
                for hit in contexts
            ]
        }

        # 7. Rerank bằng CrossEncoderReranker chính thức của TV5 (nếu bật)
        try:
            contexts, reranker_applied, current_reranker_error = _rerank_candidates(
                tv5_reranker,
                question,
                contexts,
                top_k=args.top_k,
                allow_fallback=args.allow_reranker_fallback,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"TV5 reranking failed for QID {qid}") from exc
        if reranker_applied:
            reranker_success_count += 1
        if current_reranker_error is not None:
            reranker_error = current_reranker_error
            reranker_fallback_count += 1
            logger.warning(
                "Reranking TV5 thất bại; vô hiệu hóa cho phần còn lại: %s",
                current_reranker_error,
            )
            tv5_reranker = None

        contexts = _truncate_contexts(contexts, args.max_context_len)
        retrieval_traces[qid]["after"] = [
            {
                "chunk_id": hit.chunk_id,
                "parent_id": hit.parent_id,
                "rank": hit.rank,
                "retrieval_score": hit.score,
                "rerank_score": hit.rerank_score,
            }
            for hit in contexts
        ]

        if not contexts:
            skipped[qid] = "no_retrieval_context"
            continue

        logger.info("[%d/%d] QID: %s | Question: %s... | %d contexts",
                    idx, len(qids), qid, question[:40], len(contexts))

        # Dọn dẹp VRAM rác trước khi gọi GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            qa_response = await qa_engine.generate_answer(
                question=question,
                contexts=contexts,
                prompt_version=args.prompt_version,
                rag_template=args.rag_template,
                trace_id=f"hybrid_{qid}",
            )
            pred_answer = qa_response.answer
            predictions[qid] = pred_answer
            references[qid] = gold_answer
            success_count += 1
            logger.info("  -> Output (%d chars): %s...", len(pred_answer), pred_answer[:60])
        except Exception as exc:
            logger.error("  -> Lỗi QID %s: %s", qid, exc)
            failures[qid] = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - start_time

    if predictions:
        scores = calculate_btc_scores(predictions, references)
        refused_count = sum(
            1 for pred in predictions.values() 
            if "không có đủ căn cứ" in pred.lower() or len(pred.strip()) <= 85
        )
        answered_count = len(predictions) - refused_count

        print("\n" + "=" * 70)
        print(
            "📊 KẾT QUẢ BENCHMARK HYBRID "
            f"(BM25 + DENSE {embedding_model_path}{reranker_label} + LLM)"
        )
        print("=" * 70)
        print(f"🔹 Tổng số câu hỏi test:         {len(qids)}")
        print(f"✅ Trả lời đầy đủ (Chi tiết ý):  {answered_count} / {len(qids)} ({answered_count / len(qids) * 100:.1f}%)")
        print(f"⚠️ Từ chối ('Không đủ căn cứ'):  {refused_count} / {len(qids)} ({refused_count / len(qids) * 100:.1f}%)")
        print(f"🔥 ROUGE-L Score:                {scores['rougeL']:.4f}")
        print(f"🎯 METEOR Score:                 {scores['meteor']:.4f}")
        print(f"⚙️  Retrieval Mode:              {args.mode}")
        print(f"📚 Top-K Retrieval:              {args.top_k}")
        print(f"⏱️  Thời gian trung bình:         {elapsed / max(1, success_count):.2f} giây/câu")
        print("=" * 70)

        output_path = args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        details = {}
        for qid in qids:
            if qid in predictions:
                pred_str = str(predictions[qid])
                gold_str = str(references.get(qid, ""))
                is_refused = "không có đủ căn cứ" in pred_str.lower() or len(pred_str.strip()) <= 85
                details[qid] = {
                    "question": raw_data[qid]["question"] if isinstance(raw_data[qid], dict) else raw_data[qid],
                    "prediction": pred_str,
                    "reference": gold_str,
                    "is_refused": is_refused,
                }

        report = {
            "num_samples": success_count,
            "answered_count": answered_count,
            "refused_count": refused_count,
            "elapsed_seconds": elapsed,
            "avg_seconds_per_sample": elapsed / max(1, success_count),
            "metrics": scores,
            "mode": args.mode,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "quantization": args.quantization,
            "prompt_version": args.prompt_version,
            "rag_template": args.rag_template,
            "dataset_path": str(train_path.resolve()),
            "dataset_sha256": _sha256(train_path),
            "parents_dir": str(effective_parents_dir),
            "processed_corpus_tree_hash": processing_manifest.get(
                "processed_corpus_tree_hash"
            ),
            "processed_counts": processing_manifest.get("counts"),
            "processed_git_commit": processing_manifest.get("git_commit"),
            "embedding_model": embedding_model_path,
            "llm_model": model_path,
            "device": effective_device,
            "query_decomposition_enabled": args.decompose_queries,
            "reranker_requested": args.use_reranker,
            "reranker_effective": reranker_success_count > 0,
            "reranker_model": reranker_model_path if args.use_reranker else None,
            "reranker_success_count": reranker_success_count,
            "reranker_fallback_count": reranker_fallback_count,
            "reranker_error": reranker_error,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "failures": failures,
            "skipped": skipped,
            "retrieval_traces": retrieval_traces,
            "details": details,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"🎉 Đã lưu báo cáo chi tiết + câu trả lời đầy đủ ({len(details)} câu) vào file: {output_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
