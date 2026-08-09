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

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
import numpy as np
import torch

# Thêm src vào Python path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

# Thêm đường dẫn rouge_score của BTC
BTC_QA_DIR = BASE_DIR / "docs/Scoring-Program-Task-LegalQA"
sys.path.insert(0, str(BTC_QA_DIR))

import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.llm.client import LLMClient
from udsc2026.infrastructure.llm.config import LLMConfig
from udsc2026.qa.qa_engine import QAEngine

# Import mock hybrid retriever
sys.path.insert(0, str(BASE_DIR / "experiments/tv3"))
from mock_hybrid_retriever import MockHybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("benchmark_realistic_qa")

# Enable PyTorch VRAM memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Download NLTK data nếu chưa có
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def calculate_btc_scores(predictions: dict[str, str], references: dict[str, str]) -> dict[str, float]:
    """Tính điểm ROUGE-L và METEOR theo đúng thuật toán scoring.py của BTC."""
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


async def main():
    parser = argparse.ArgumentParser(description="Run TV3 Realistic Hybrid Benchmark")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng câu hỏi cần test (0 = tất cả)")
    parser.add_argument("--top_k", type=int, default=3, help="Số đoạn luật truyền cho LLM (default: 3)")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "bm25", "dense"], help="Retrieval mode")
    parser.add_argument("--max_docs", type=int, default=0, help="Giới hạn số parent docs nạp (0 = nạp tất cả 8.532)")
    parser.add_argument("--max_context_len", type=int, default=1200, help="Giới hạn ký tự tối đa cho mỗi context (chống OOM)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Giới hạn số token Qwen3 sinh ra (chống OOM)")
    parser.add_argument("--dataset", type=str, default="", help="Đường dẫn file train.json")
    parser.add_argument("--parents_dir", type=str, default="", help="Đường dẫn thư mục parents chứa file jsonl")
    parser.add_argument("--model_path", type=str, default="thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2", help="Tên model Qwen (<4B) trên HuggingFace hoặc đường dẫn local (mặc định: thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2)")
    parser.add_argument("--embedding_model", type=str, default="huyydangg/DEk21_hcmute_embedding_v2", help="Tên model trên HuggingFace (ví dụ: huyydangg/DEk21_hcmute_embedding_v2) hoặc đường dẫn local")
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
    parser.add_argument("--qid", type=str, default="", help="Lọc duy nhất 1 QID cụ thể để test/debug (ví dụ: --qid 101625)")
    parser.add_argument("--cache_dir", type=str, default="", help="Đường dẫn lưu DB SQLite và Vector cache (ví dụ: /run/media/quan/New Volume/uit_data/cache)")
    parser.add_argument("--verbose", action="store_true", help="In log chi tiết từng giai đoạn (BM25, Dense, RRF, TV5 Rerank, LLM Contexts)")
    parser.add_argument("--use_reranker", action="store_true", help="Bật Cross-Encoder Reranker của TV5 (BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3", help="Mô hình Reranker của TV5")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE_DIR / "experiments/tv3/hybrid_benchmark_report.json",
        help="Đường dẫn báo cáo JSON; đặt tên riêng cho mỗi prompt để so sánh.",
    )
    args = parser.parse_args()

    # 1. Load train dataset
    train_path = Path(args.dataset) if args.dataset else None
    if not train_path or not train_path.exists():
        candidates = [
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
        raise FileNotFoundError(f"Không tìm thấy file train.json hoặc warmup.json! Vui lòng truyền --dataset <đường_dẫn_file>")

    logger.info("Đang đọc dữ liệu tham chiếu từ: %s", train_path)
    with open(train_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    qids = list(raw_data.keys())
    if args.qid:
        if args.qid in raw_data:
            qids = [args.qid]
            logger.info("🎯 Đang chọn riêng QID=%s để debug...", args.qid)
        else:
            raise KeyError(f"Không tìm thấy QID='{args.qid}' trong file dataset {train_path}!")
    elif args.limit > 0:
        qids = qids[:args.limit]
    logger.info("📝 Số câu hỏi cần test: %d", len(qids))

    # 2. Khởi tạo MockHybridRetriever (BM25 + Dense Vector + RRF)
    parents_dir = Path(args.parents_dir) if args.parents_dir else BASE_DIR / "data/processed/parents"
    embedding_model_path = args.embedding_model
    if not embedding_model_path.startswith("huyydangg/") and not Path(embedding_model_path).exists():
        local_bkai = BASE_DIR / "models/bkai-bi-encoder"
        if local_bkai.exists():
            embedding_model_path = str(local_bkai)

    logger.info("⏳ Đang khởi tạo MockHybridRetriever với embedding model='%s' (mode=%s)...", embedding_model_path, args.mode)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    retriever = MockHybridRetriever(
        parents_dir=parents_dir,
        embedding_model_path=embedding_model_path,
        device=args.device if torch.cuda.is_available() else "cpu",
        cache_dir=cache_dir,
        max_docs=args.max_docs,
    )

    # 3. Khởi tạo CrossEncoderReranker chính thức của TV5 (nếu được bật)
    tv5_reranker = None
    if args.use_reranker:
        try:
            from udsc2026.infrastructure.reranker import CrossEncoderClient
            from udsc2026.retrieval.reranking import CrossEncoderReranker
            logger.info("⏳ Đang khởi tạo CrossEncoderReranker TV5 từ '%s'...", args.reranker_model)
            client = CrossEncoderClient(args.reranker_model, device=args.device if torch.cuda.is_available() else "cpu")
            tv5_reranker = CrossEncoderReranker(client)
            logger.info("✅ Đã nạp thành công Reranker chính thức của TV5!")
        except Exception as e:
            logger.warning("Không thể nạp TV5 Reranker (%s). Sẽ tiếp tục dùng RRF scores mặc định.", e)

    # 4. Khởi tạo Qwen LLM
    model_path = args.model_path
    local_qwen = BASE_DIR / "models/qwen3-legal"
    if local_qwen.exists():
        model_path = str(local_qwen)

    logger.info("⏳ Đang khởi tạo LLM từ '%s' (quantization=%s)...", model_path, args.quantization)
    llm_config = LLMConfig(
        model_path=model_path,
        backend="transformers",
        device=args.device if torch.cuda.is_available() else "cpu",
        dtype="float16",
        quantization=args.quantization if torch.cuda.is_available() else "none",
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=300.0,
    )
    llm_client = LLMClient(llm_config)
    qa_engine = QAEngine(llm_client=llm_client, use_chat_template=True)

    predictions: dict[str, str] = {}
    references: dict[str, str] = {}
    start_time = time.monotonic()
    success_count = 0

    print("\n" + "=" * 70)
    print(f"🚀 BENCHMARK THỰC TẾ HYBRID (BM25 + Dense {args.embedding_model} + TV5 Reranker + LLM)")
    print(
        f"   mode={args.mode} | top_k={args.top_k} | use_reranker={args.use_reranker} | "
        f"prompt={args.prompt_version}/{args.rag_template} | "
        f"quantization={args.quantization} | limit={len(qids)}"
    )
    print("=" * 70)

    for idx, qid in enumerate(qids, start=1):
        item = raw_data[qid]
        question = item["question"]
        gold_answer = item["answer"]

        if not gold_answer or not gold_answer.strip():
            continue

        # 5. Tìm kiếm Hybrid (Kéo 20 ứng viên nếu dùng TV5 Reranker, hoặc top_k nếu không)
        search_k = 20 if tv5_reranker is not None else args.top_k
        retrieved_docs = retriever.search(question, top_k=search_k, mode=args.mode)

        if args.verbose:
            print(f"\n--- [DEBUG VERBOSE QID: {qid}] ---")
            print(f"❓ Câu hỏi: {question}")
            print(f"🎯 Đáp án tham chiếu BTC:\n{gold_answer[:300]}...")
            print(f"\n🔍 [Giai đoạn Retrieval] Rút được {len(retrieved_docs)} candidates:")
            for rank_i, doc_item in enumerate(retrieved_docs[:5], start=1):
                print(f"   Candidate {rank_i} (RRF Score: {doc_item.get('retrieval_score', 0):.4f}): {doc_item.get('text', '')[:150]}...")

        # 6. Chuyển đổi thành RetrievalHit với chiến lược Context Budget
        TOTAL_CHAR_BUDGET = 10000 if args.max_context_len == 1200 else args.max_context_len * len(retrieved_docs)
        n_docs = len(retrieved_docs)
        contexts = []
        for rank, doc in enumerate(retrieved_docs):
            raw_text = doc["text"]
            weight = max(0.10, 0.35 - rank * 0.05) if n_docs > 1 else 1.0
            char_limit = int(TOTAL_CHAR_BUDGET * weight)
            if len(raw_text) > char_limit:
                raw_text = raw_text[:char_limit] + "..."

            hit = RetrievalHit(
                chunk_id=doc.get("parent_id", f"ret_{qid}_{rank}"),
                doc_id=doc.get("doc_id", "unknown"),
                text=raw_text,
                score=doc.get("retrieval_score", 0.0),
                law_name=doc.get("law_name"),
                article=doc.get("article"),
            )
            contexts.append(hit)

        # 7. Rerank bằng CrossEncoderReranker chính thức của TV5 (nếu bật)
        if tv5_reranker is not None and len(contexts) > 1:
            try:
                contexts = tv5_reranker.rerank(question, candidates=contexts, top_n=args.top_k)
                if args.verbose:
                    print(f"\n⚡ [Giai đoạn TV5 Reranking] Top-{len(contexts)} sau khi Rerank:")
                    for rk_i, ctx_item in enumerate(contexts, start=1):
                        print(f"   Reranked Hit {rk_i} (Score: {ctx_item.rerank_score}): {ctx_item.text[:150]}...")
            except Exception as e:
                logger.warning("Reranking TV5 thất bại: %s", e)

        if not contexts:
            continue

        logger.info("[%d/%d] QID: %s | Question: %s... | %d contexts",
                    idx, len(qids), qid, question[:40], len(contexts))

        if not contexts:
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

    elapsed = time.monotonic() - start_time

    if predictions:
        scores = calculate_btc_scores(predictions, references)
        refused_count = sum(
            1 for pred in predictions.values() 
            if "không có đủ căn cứ" in pred.lower() or len(pred.strip()) <= 85
        )
        answered_count = len(predictions) - refused_count

        print("\n" + "=" * 70)
        print(f"📊 KẾT QUẢ BENCHMARK HYBRID (BM25 + DENSE {args.embedding_model} + LLM)")
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
            "quantization": args.quantization,
            "prompt_version": args.prompt_version,
            "rag_template": args.rag_template,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": details,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"🎉 Đã lưu báo cáo chi tiết + câu trả lời đầy đủ ({len(details)} câu) vào file: {output_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
