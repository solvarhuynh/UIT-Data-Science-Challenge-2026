# Qwen3-VL-Reranker-2B

`Qwen/Qwen3-VL-Reranker-2B` là reranker đa phương thức thuộc họ Qwen3-VL.
Pipeline truyền query và chunk pháp luật dưới dạng text; khả năng đọc ảnh/bảng
của model vẫn có thể dùng cho tài liệu đa phương thức.

- Model card: [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
- Tham số: khoảng 2B, dưới giới hạn 4B của cuộc thi
- Local path: `models/qwen3-vl-reranker-2b`
- Output: raw relevance score; điểm cao hơn được xếp trước
- Revision thực tế: `models/download_manifest.json`

## Tải và benchmark

```powershell
python download_models.py --profile task1-qwen --only reranker
```

Không copy file từ `models/reranker` cũ sang thư mục mới. GPU được khuyến nghị;
CPU chỉ phù hợp smoke test. Benchmark lại latency, VRAM và Recall/Precision;
không gán kết quả của reranker 0.6B/BGE cho checkpoint này.

```powershell
python scripts/evaluation/benchmark_reranker.py `
  --benchmark data/processed_v3/benchmarks/synthetic_qa.jsonl `
  --candidates artifacts/tv2/dense_predictions.jsonl `
  --model models/qwen3-vl-reranker-2b --device cuda `
  --batch-size 4 --max-length 1024 --candidate-k 50 --top-n 5 --fp16 `
  --output-dir artifacts/tv5/qwen3-vl-reranker-2b
```
