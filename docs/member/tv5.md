# TV5 - Reranking & MLOps Specialist

## 1. Tổng quan vai trò

TV5 phụ trách Reranking Pipeline, Evaluation Benchmark và MLOps packaging cho hệ thống RAG Pháp luật DSC2026. Trọng tâm là tích hợp Cross-Encoder để tái xếp hạng top-K kết quả từ Hybrid Search, giữ nguyên metadata citation, đo chất lượng hệ thống và đóng gói để chạy lặp lại trên môi trường local/CI/CodaLab.

TV5 không làm Web Frontend và không viết QA prompt. Vai trò chính là nâng chất lượng context trước khi vào LLM, đo điểm bằng benchmark và bảo đảm hệ thống có thể build/run ổn định.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `RerankerClient` trong `src/udsc2026/infrastructure/reranker/`, load Cross-Encoder từ local path hoặc config.
- [ ] Viết `CrossEncoderReranker` trong `src/udsc2026/retrieval/reranking/`, nhận query và candidates từ Hybrid Search, trả danh sách đã sắp xếp lại.
- [ ] Hỗ trợ batch scoring với `batch_size`, `device`, `max_length`, `top_n` để kiểm soát latency.
- [ ] Bảo toàn metadata citation gồm `chunk_id`, `doc_id`, `law_name`, `article`, `clause`, `point`, `source`, `parent_id`.
- [ ] Giữ score gốc từ Dense/Sparse/Hybrid và bổ sung `rerank_score`, `final_score`, `rank`.
- [ ] Viết Evaluation Benchmark trong `src/udsc2026/evaluation/` để đo MRR, Recall@K, ROUGE-L và latency.
- [ ] Hỗ trợ benchmark trên synthetic dataset của TV4 và test set thật khi có.
- [ ] Viết script `scripts/evaluate.py` để chạy pipeline, xuất report JSON/Markdown và so sánh trước/sau rerank.
- [ ] Viết script xuất `submission.csv` cho CodaLab theo schema cuộc thi khi schema được cố định.
- [ ] Tạo Multi-stage Docker Image cho backend: stage build dependencies, stage runtime gọn nhẹ.
- [ ] Không copy model weights, raw data hoặc vector store lớn vào Docker image; mount bằng volume hoặc cấu hình path.
- [ ] Viết `docker-compose.yml` để chạy backend, VectorDB, Redis cache và các service phụ trợ cần thiết.
- [ ] Chuẩn hóa `.env.example` cho model path, vector DB URL, Redis URL, device, batch size, port và submission path.
- [ ] Viết smoke test hoặc CI command để kiểm tra import, config, reranker, evaluation và Docker build cơ bản.

## 3. Quy chuẩn Clean Code & API Contract

### Clean Code bắt buộc

- Áp dụng triệt để DRY, tối ưu số dòng code và không tạo class/interface dư thừa nếu chưa có nhu cầu thật.
- Dùng type hinting đầy đủ cho mọi input/output; evaluation sample, metric result và submission row phải có schema rõ ràng.
- Mỗi hàm chỉ làm một trách nhiệm: score rerank, sort result, compute metric, write report hoặc write submission.
- Không copy-paste metric logic; MRR, Recall@K và ROUGE-L phải là các hàm riêng, test được.
- Không định nghĩa lại `RetrievalHit` hoặc `QAResponse`; dùng contract chung.
- Không đưa model weights, raw data lớn, cache hoặc vector index vào Docker image.

### API Contract

Reranker:

```python
from udsc2026.contracts.retrieval import RetrievalHit

def rerank(
    query: str,
    candidates: list[RetrievalHit],
    top_n: int,
) -> list[RetrievalHit]: ...
```

Candidate input cần có:

```text
chunk_id, text, dense_score, sparse_score, hybrid_score, metadata
```

Output rerank cần bổ sung nếu schema cho phép:

```text
rerank_score, final_score, rank
```

Evaluation:

```python
def evaluate_retrieval(
    predictions: list[list[RetrievalHit]],
    gold_chunk_ids: list[list[str]],
    k_values: list[int],
) -> EvaluationReport: ...
```

Submission writer:

```python
def write_submission(results: list[QAResponse], output_path: str) -> None: ...
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Cross-Encoder reranker chạy được với top-K candidates từ Hybrid Search.
- [ ] Metadata citation được giữ nguyên sau rerank, không mất `chunk_id` hoặc điều/khoản.
- [ ] Output có `rerank_score`, `final_score` và `rank` rõ ràng.
- [ ] Benchmark đo được MRR, Recall@K, ROUGE-L và latency trên synthetic dataset của TV4.
- [ ] Script evaluation sinh được report trước/sau rerank.
- [ ] Script xuất được `submission.csv` theo schema CodaLab đã thống nhất.
- [ ] Docker multi-stage build thành công và image không chứa model weights/data lớn.
- [ ] `docker-compose.yml` chạy được backend, VectorDB và Redis cache bằng `.env.example`.
