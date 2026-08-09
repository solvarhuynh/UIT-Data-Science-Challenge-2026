"""Golden Context Benchmark for TV3 QA Engine.

Evaluates QAEngine (Qwen3) against ground truth answers using BTC's official scoring metrics
(ROUGE-L and METEOR) when supplied with 100% accurate (Golden) context.

Usage:
    python experiments/tv3/benchmark_golden_qa.py --limit 10
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("benchmark_golden_qa")

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

        # ROUGE-L F1 score
        r_score = rouge_scoring.score(gold_text, pred_text)['rougeL'].fmeasure
        rouge_scores.append(r_score)

        # METEOR score (split by space)
        m_score = meteor_score([gold_text.split()], pred_text.split())
        meteor_scores.append(m_score)

    mean_rouge = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    mean_meteor = float(np.mean(meteor_scores)) if meteor_scores else 0.0

    return {
        "rougeL": mean_rouge,
        "meteor": mean_meteor,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run TV3 Golden Context Benchmark")
    parser.add_argument("--limit", type=int, default=10, help="Số lượng mẫu cần test (0 = tất cả)")
    parser.add_argument("--dataset", type=str, default="", help="Đường dẫn file dataset (mặc định ưu tiên train.json)")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="Mức nén quantization")
    args = parser.parse_args()

    # Determine dataset path
    dataset_path = args.dataset
    if not dataset_path:
        train_path = BASE_DIR / "data/task2/public/train.json"
        warmup_path = BASE_DIR / "data/warmup/warmup_task2.json"
        if train_path.exists():
            dataset_path = str(train_path)
        elif warmup_path.exists():
            dataset_path = str(warmup_path)
        else:
            raise FileNotFoundError("Không tìm thấy file dataset train.json hoặc warmup_task2.json")

    logger.info("Đang đọc dữ liệu tham chiếu từ: %s", dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    qids = list(raw_data.keys())
    if args.limit > 0:
        qids = qids[:args.limit]
        logger.info("🧪 CHẾ ĐỘ BENCHMARK THỬ NGHIỆM: %d câu hỏi", len(qids))
    else:
        logger.info("🚀 CHẾ ĐỘ BENCHMARK FULL: %d câu hỏi", len(qids))

    model_path = str(BASE_DIR / "models/qwen3-legal")
    logger.info("⏳ Đang khởi tạo Qwen3 từ '%s' (quantization=%s)...", model_path, args.quantization)

    llm_config = LLMConfig(
        model_path=model_path,
        backend="transformers",
        device="cuda",
        dtype="float16",
        quantization=args.quantization,
        timeout_seconds=300.0,
    )
    llm_client = LLMClient(llm_config)
    qa_engine = QAEngine(llm_client=llm_client, use_chat_template=True)

    predictions: dict[str, str] = {}
    references: dict[str, str] = {}

    start_time = time.monotonic()
    success_count = 0

    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU CHẠY GOLDEN CONTEXT BENCHMARK CHO TV3")
    print("=" * 60)

    for idx, qid in enumerate(qids, start=1):
        item = raw_data[qid]
        question = item["question"]
        gold_answer = item["answer"]

        # Nếu đáp án trong file tham chiếu là null/rỗng thì bỏ qua
        if not gold_answer or not gold_answer.strip():
            continue

        # Giả lập Golden Context (coi như TV2/TV5 đã tìm đúng 100% đoạn luật)
        golden_hit = RetrievalHit(
            chunk_id=f"golden_chunk_{qid}",
            doc_id=f"golden_doc_{qid}",
            text=gold_answer,
            score=1.0,
        )

        logger.info("[%d/%d] QID: %s | Question: %s...", idx, len(qids), qid, question[:50])

        try:
            qa_response = await qa_engine.generate_answer(
                question=question,
                contexts=[golden_hit],
                trace_id=f"bench_{qid}",
            )
            pred_answer = qa_response.answer
            predictions[qid] = pred_answer
            references[qid] = gold_answer
            success_count += 1
            logger.info("  -> Output: %s...", pred_answer[:80])
        except Exception as exc:
            logger.error("  -> Lỗi khi sinh câu trả lời cho QID %s: %s", qid, exc)

    elapsed = time.monotonic() - start_time
    logger.info("Chạy xong %d mẫu trong %.2f giây (trung bình %.2fs/mẫu)", success_count, elapsed, elapsed / max(1, success_count))

    if predictions:
        scores = calculate_btc_scores(predictions, references)

        print("\n" + "=" * 60)
        print("📊 KẾT QUẢ BENCHMARK GOLDEN CONTEXT (THƯỚC ĐO BTC)")
        print("=" * 60)
        print(f"🔹 Số câu test thành công: {success_count} / {len(qids)}")
        print(f"🔥 ROUGE-L Score:           {scores['rougeL']:.4f}")
        print(f"🎯 METEOR Score:            {scores['meteor']:.4f}")
        print(f"⏱️  Thời gian trung bình:    {elapsed / max(1, success_count):.2f} giây/câu")
        print("=" * 60)

        # Lưu báo cáo JSON
        output_report_path = BASE_DIR / "experiments/tv3/golden_benchmark_report.json"
        report_data = {
            "num_samples": success_count,
            "elapsed_seconds": elapsed,
            "avg_seconds_per_sample": elapsed / max(1, success_count),
            "metrics": scores,
            "quantization": args.quantization,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"🎉 Đã lưu báo cáo chi tiết vào: {output_report_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
