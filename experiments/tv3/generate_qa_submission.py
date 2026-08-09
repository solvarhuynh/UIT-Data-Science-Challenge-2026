"""Local & Kaggle Submission Pipeline Generator for Task 2 (LegalQA).

Executes end-to-end QA pipeline over test data (public_official.json) and formats
the outputs into a valid BTC submission archive: 'predictions/submission_task2.zip'.

Supports optional sample limiting (--limit N) for fast local dry-runs.

Usage:
    python experiments/tv3/generate_qa_submission.py --limit 10
    python experiments/tv3/generate_qa_submission.py --limit 0  # Full 4,002 questions
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path

# Thêm src vào Python path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "src"))

import torch

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.legal_qa_submission import (
    LegalQASubmissionItem,
    write_legal_qa_submission_zip,
)
from udsc2026.infrastructure.llm.client import LLMClient
from udsc2026.infrastructure.llm.config import LLMConfig
from udsc2026.qa.qa_engine import QAEngine

# Import mock hybrid retriever
sys.path.insert(0, str(BASE_DIR / "experiments/tv3"))
from mock_hybrid_retriever import MockHybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("generate_qa_submission")


async def main():
    parser = argparse.ArgumentParser(description="Generate Task 2 Submission Zip")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng câu hỏi cần sinh (0 = tất cả câu trong public test)")
    parser.add_argument("--top_k", type=int, default=5, help="Số đoạn luật truyền cho LLM")
    parser.add_argument("--max_docs", type=int, default=0, help="Số parent docs tối đa để nạp (0 = nạp tất cả)")
    parser.add_argument("--test_file", type=str, default="", help="Đường dẫn file public_official.json")
    parser.add_argument("--parents_dir", type=str, default="", help="Đường dẫn thư mục parents")
    parser.add_argument("--model_path", type=str, default="thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2", help="Model LLM")
    parser.add_argument("--embedding_model", type=str, default="huyydangg/DEk21_hcmute_embedding_v2", help="Model Embedding")
    parser.add_argument("--output_zip", type=str, default="predictions/submission_task2.zip", help="Đường dẫn file zip đầu ra")
    parser.add_argument("--quantization", type=str, default="none", choices=["4bit", "8bit", "none"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. Tìm file public test
    test_path = Path(args.test_file) if args.test_file else None
    if not test_path or not test_path.exists():
        candidates = [
            BASE_DIR / "data/task2/public/public-official.json",
            BASE_DIR / "data/task2/public/public_official.json",
            BASE_DIR / "data/task2/public/train.json",
        ]
        for cand in candidates:
            if cand.exists():
                test_path = cand
                break

    if not test_path or not test_path.exists():
        raise FileNotFoundError("Không tìm thấy file public_official.json hoặc public_test.json")

    logger.info("Đang đọc câu hỏi từ đề thi: %s", test_path)
    with open(test_path, "r", encoding="utf-8") as f:
        raw_test = json.load(f)

    # Đề thi có dạng dict {qid: {question: ...}} hoặc list
    if isinstance(raw_test, dict):
        items = list(raw_test.items())
    elif isinstance(raw_test, list):
        items = [(str(item.get("id", idx)), item) for idx, item in enumerate(raw_test)]

    if args.limit > 0:
        items = items[:args.limit]
        logger.info("🧪 CHẾ ĐỘ THỬ NGHIỆM: Sinh nộp bài cho %d câu đầu tiên", len(items))
    else:
        logger.info("🚀 CHẾ ĐỘ THI ĐẤU THẬT: Sinh nộp bài cho toàn bộ %d câu", len(items))

    # 2. Khởi tạo Retriever
    parents_dir = Path(args.parents_dir) if args.parents_dir else BASE_DIR / "data/processed/parents"
    embedding_model_path = args.embedding_model
    if not embedding_model_path.startswith("huyydangg/") and not Path(embedding_model_path).exists():
        local_bkai = BASE_DIR / "models/bkai-bi-encoder"
        if local_bkai.exists():
            embedding_model_path = str(local_bkai)

    logger.info("⏳ Đang khởi tạo MockHybridRetriever (embedding='%s')...", embedding_model_path)
    retriever = MockHybridRetriever(
        parents_dir=parents_dir,
        embedding_model_path=embedding_model_path,
        device=args.device if torch.cuda.is_available() else "cpu",
        max_docs=args.max_docs,
    )

    # 3. Khởi tạo LLM (Ưu tiên thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2 hoặc local)
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
        max_new_tokens=512,
        timeout_seconds=300.0,
    )
    llm_client = LLMClient(llm_config)
    qa_engine = QAEngine(llm_client=llm_client, use_chat_template=True)

    # 4. Tích hợp RAGOrchestrator chính thức của TV5 và Cache
    from udsc2026.api.cache import CacheClient
    from udsc2026.api.orchestrator import RAGOrchestrator
    from udsc2026.contracts.api import QueryRequest

    class OfficialBridgeRetriever:
        """Bridge Retriever kết hợp Selected Contexts của BTC và Hybrid Search cho TV5 RAGOrchestrator."""
        def __init__(self, fallback_retriever):
            self.fallback_retriever = fallback_retriever

        def search(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]:
            return self.fallback_retriever.search(query, top_k=top_k)

    cache = CacheClient()
    bridge_retriever = OfficialBridgeRetriever(retriever)
    orchestrator = RAGOrchestrator(
        retriever=bridge_retriever,
        qa_engine=qa_engine,
        cache=cache,
    )

    submission_items = []
    start_time = time.monotonic()

    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU CHẠY PIPELINE CHÍNH THỨC TV5 + TV3 (LEGAL QA)")
    print("=" * 70)

    for idx, (qid, data) in enumerate(items, start=1):
        question = data["question"] if isinstance(data, dict) else data

        # 1. Kiểm tra xem có Selected Context / Golden Context từ BTC cho QID này không
        ctx_file = BASE_DIR / f"data/task2/public/selected-contexts/context_{qid}.json"
        contexts = []
        if ctx_file.exists():
            try:
                with open(ctx_file, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                passage = c_data.get("passage", "")
                if passage:
                    contexts.append(
                        RetrievalHit(
                            chunk_id=f"sel_{qid}",
                            doc_id=str(qid),
                            text=passage[:3000],
                            score=1.0,
                            law_name=c_data.get("name") or c_data.get("link"),
                        )
                    )
            except Exception as e:
                logger.warning("Không thể đọc selected context file %s: %s", ctx_file, e)

        # 2. Nếu không có selected context, fallback sang Hybrid Retriever
        if not contexts:
            retrieved_docs = retriever.search(question, top_k=args.top_k, mode="hybrid")
            # Áp dụng chiến lược Context Budget (Dense X Retrieval paper):
            # Tổng ngân sách ký tự = 10000, phân bổ ưu tiên cho context xếp hạng cao hơn.
            # Tối ưu cho Kaggle T4 GPU 16GB VRAM.
            TOTAL_CHAR_BUDGET = 10000
            n_docs = len(retrieved_docs)
            contexts = []
            for rank, doc in enumerate(retrieved_docs):
                # Rank 0 (top 1) được 35% ngân sách, rank cuối được 10%
                weight = max(0.10, 0.35 - rank * 0.05) if n_docs > 1 else 1.0
                char_limit = int(TOTAL_CHAR_BUDGET * weight)
                contexts.append(
                    RetrievalHit(
                        chunk_id=doc.get("parent_id", f"hit_{idx}"),
                        doc_id=doc.get("doc_id", "doc"),
                        text=doc["text"][:char_limit],
                        score=doc.get("retrieval_score", 0.0),
                        law_name=doc.get("law_name"),
                        article=doc.get("article"),
                    )
                )

        print(f"\n📌 [Câu {idx}/{len(items)}] QID: {qid}")
        print(f"❓ Câu hỏi: {question}")
        print(f"📚 Contexts tìm thấy cho Bước 2 ({len(contexts)} văn bản):")
        for c_idx, hit in enumerate(contexts, start=1):
            source_tag = " [Golden BTC]" if hit.chunk_id.startswith("sel_") else " [Hybrid Search]"
            law_title = hit.law_name or hit.doc_id or "Văn bản luật"
            snippet = (hit.text or "").replace("\n", " ").strip()[:100]
            print(f"   🔹 Context {c_idx}{source_tag} | {law_title} (Score: {hit.score:.4f})")
            print(f"      Trích đoạn: \"{snippet}...\"")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Gọi trực tiếp qua QAEngine đã được nâng cấp
            qa_res = await qa_engine.generate_answer(
                question=question,
                contexts=contexts,
                trace_id=f"sub_{qid}",
            )
            answer_text = qa_res.answer
        except Exception as exc:
            logger.error("Lỗi khi sinh đáp án QID %s: %s", qid, exc)
            answer_text = "Dựa trên dữ liệu pháp lý được cung cấp, không có đủ căn cứ để trả lời câu hỏi này."

        submission_items.append(
            LegalQASubmissionItem(
                id=str(qid),
                answer=answer_text,
            )
        )
        logger.info("  -> Answer (%d chars): %s...", len(answer_text), answer_text[:60])

    elapsed = time.monotonic() - start_time
    output_zip = Path(args.output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # 4. Ghi file submission.zip chuẩn 100% định dạng BTC
    write_legal_qa_submission_zip(submission_items, output_zip)

    # 5. Kiểm tra nếu đề thi có sẵn đáp án mẫu -> Gọi TRỰC TIẾP code chấm điểm BTC (scoring.py)!
    y_pred = {item.id: {"answer": item.answer} for item in submission_items}
    y_true = {}
    for qid, data in items:
        if isinstance(data, dict) and "answer" in data and data["answer"]:
            y_true[str(qid)] = data["answer"]

    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH TẠO FILE NỘP BÀI CHÍNH THỨC!")
    print("=" * 70)
    print(f"🔹 Số câu đã tạo:      {len(submission_items)} câu")
    print(f"⏱️  Thời gian xử lý:   {elapsed:.1f} giây ({elapsed/len(items):.2f}s/câu)")
    print(f"📦 File nộp bài ZIP:   {output_zip.resolve()}")

    if len(y_true) == len(y_pred):
        btc_scoring_script = BASE_DIR / "docs/Scoring-Program-Task-LegalQA/scoring.py"
        if btc_scoring_script.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("btc_scoring", str(btc_scoring_script))
            btc_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(btc_module)
            eval_qa = btc_module.eval_qa

            print("-" * 70)
            print("📊 ĐÁNH GIÁ ĐIỂM SỐ BỞI CHƯƠNG TRÌNH CHẤM ĐIỂM CHÍNH THỨC CỦA BTC")
            print("-" * 70)
            scores = eval_qa(y_pred, y_true)
            print(f"🔥 ROUGE-L Score:       {scores.get('rouge', 0.0):.4f}")
            print(f"🎯 METEOR Score:        {scores.get('meteor', 0.0):.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
