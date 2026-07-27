# TV5 - Rerank & DevOps: Cross-Encoder, Docker, CodaLab Submission

## 1. Tổng quan vai trò

TV5 chịu trách nhiệm tối ưu lớp xếp hạng cuối và đóng gói hệ thống để chạy ổn định. Trọng tâm là Reranking Pipeline: nhận top-K kết quả từ Hybrid Search, dùng Cross-Encoder hoặc reranker tương đương để chấm lại từng cặp `(query, document)` và đưa document liên quan nhất lên đầu. Đây là lớp giúp cải thiện đáng kể precision cho LegalIR và chất lượng context cho LegalQA. Phần bổ sung là DevOps: Docker multi-stage, `docker-compose.yml`, script chạy benchmark và sinh `submission.csv` chuẩn CodaLab.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `RerankerClient` trong `src/udsc2026/infrastructure/reranker/`, load model từ local hoặc cấu hình trong `configs/`.
- [ ] Viết `CrossEncoderReranker` trong `src/udsc2026/retrieval/reranking/`, nhận query và candidates từ TV3 Hybrid Search.
- [ ] Hỗ trợ batch scoring để giảm latency, có tham số `batch_size`, `top_n`, `device`.
- [ ] Bảo toàn toàn bộ metadata citation khi rerank, không tạo object mới làm mất `chunk_id` hoặc `article`.
- [ ] Viết benchmark trong `src/udsc2026/evaluation/` để đo MRR, Recall@K, latency trước và sau rerank.
- [ ] Viết script trong `scripts/` để chạy test set, gọi pipeline end-to-end và xuất kết quả.
- [ ] Tạo Dockerfile multi-stage cho backend/frontend nếu cần, không copy `models/` hoặc `data/vector_store/` lớn vào image.
- [ ] Viết `docker-compose.yml` để chạy backend, frontend và VectorDB local theo biến môi trường.

## 3. API Contract & Dữ liệu giao tiếp

Reranker nhận:

```python
rerank(query: str, candidates: list[RetrievalHit], top_n: int) -> list[RetrievalHit]
```

Mỗi candidate cần có:

```text
chunk_id, text, dense_score, sparse_score, hybrid_score, metadata
```

Output cần bổ sung:

```text
rerank_score, final_score, rank
```

Submission writer nhận:

```python
write_submission(results: list[QAResponse], output_path: str) -> None
```

File đầu ra:

```text
submission.csv
```

Schema cụ thể theo CodaLab sẽ được cố định trong `configs/evaluation/` hoặc tài liệu cuộc thi.

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Reranker chạy được với top-K từ Hybrid Search.
- [ ] Metadata citation được giữ nguyên sau rerank.
- [ ] Có benchmark so sánh trước/sau rerank bằng MRR và Recall@K.
- [ ] Docker build thành công và image không chứa model weights lớn.
- [ ] `docker-compose.yml` chạy được backend, frontend và VectorDB.
- [ ] Script evaluation sinh được `submission.csv` từ test set mẫu.
