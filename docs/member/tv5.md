# TV5 - Cross-Encoder Reranker, Evaluation Benchmark & Dockerization

## 1. Tổng quan vai trò

TV5 phụ trách lớp xếp hạng cuối, benchmark chất lượng và đóng gói hệ thống. Trọng tâm là Cross-Encoder Reranker Pipeline nhận candidates từ Hybrid Search, chấm lại cặp `(query, document)` và đưa context liên quan nhất lên đầu. TV5 cũng chịu trách nhiệm Evaluation Benchmark, script submission và Dockerization để hệ thống chạy ổn định ở môi trường local/CI.

TV5 là người đo chất lượng truy hồi, đo latency và đảm bảo pipeline có thể build/run lặp lại.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `RerankerClient` trong `src/udsc2026/infrastructure/reranker/`, load cross-encoder/reranker từ local path hoặc `configs/`.
- [ ] Viết `CrossEncoderReranker` trong `src/udsc2026/retrieval/reranking/`, nhận query và `list[RetrievalHit]`, trả danh sách đã sắp xếp lại.
- [ ] Hỗ trợ batch scoring với `batch_size`, `top_n`, `device` để giảm latency.
- [ ] Bảo toàn metadata citation, `chunk_id`, `article`, `clause`, score gốc và score mới sau rerank.
- [ ] Viết benchmark trong `src/udsc2026/evaluation/` để đo MRR, Recall@K, Precision@K và latency trước/sau rerank.
- [ ] Viết script evaluation trong `scripts/` để chạy test set, gọi pipeline và xuất báo cáo.
- [ ] Viết submission writer sinh `submission.csv` theo schema cuộc thi/CodaLab khi schema được cố định.
- [ ] Tạo Dockerfile multi-stage cho backend/frontend nếu cần; không copy `models/` hoặc vector store lớn vào image.
- [ ] Viết `docker-compose.yml` để chạy backend, frontend và VectorDB local theo biến môi trường.
- [ ] Hỗ trợ config `.env.example` cho port, model path, vector DB URL và cache URL.

## 3. Quy chuẩn Code & API Contract

### Clean Code Standard

- Code ngắn gọn, tách `RerankerClient`, `CrossEncoderReranker`, metric calculators và submission writer.
- Dùng type hints đầy đủ và Pydantic schema chung cho retrieval/evaluation input output.
- Không copy-paste metric logic; mỗi metric nên có hàm riêng, test được.
- Không đưa model weight, data raw lớn hoặc vector store vào Docker image.
- Không tạo schema `RetrievalHit` riêng; bảo toàn contract chung từ đầu vào đến đầu ra.

### API Contract

Reranker:

```python
from udsc2026.contracts.retrieval import RetrievalHit

rerank(query: str, candidates: list[RetrievalHit], top_n: int) -> list[RetrievalHit]
```

Candidate cần có:

```text
chunk_id, text, dense_score, sparse_score, hybrid_score, metadata
```

Output bổ sung nếu schema cho phép:

```text
rerank_score, final_score, rank
```

Submission writer:

```python
write_submission(results: list[QAResponse], output_path: str) -> None
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Reranker chạy được với top-K từ Hybrid Search.
- [ ] Metadata citation được giữ nguyên sau rerank.
- [ ] Benchmark so sánh trước/sau rerank bằng MRR, Recall@K và latency.
- [ ] Script evaluation sinh được report và `submission.csv` từ test set mẫu.
- [ ] Docker build thành công và image không chứa model weights/data lớn.
- [ ] `docker-compose.yml` chạy được backend, frontend và VectorDB local.
