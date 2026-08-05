# TV5 - Reranking & MLOps Specialist

## 1. Tổng quan vai trò

TV5 phụ trách Reranking Pipeline, Evaluation Benchmark và MLOps packaging cho hệ thống RAG Pháp luật DSC2026. Trọng tâm là tích hợp Cross-Encoder để tái xếp hạng top-K kết quả từ Hybrid Search, giữ nguyên metadata citation, đo chất lượng hệ thống và đóng gói để chạy lặp lại trên môi trường local/CI/CodaLab.

Frontend không nằm trên đường chấm điểm và không phải điều kiện để nộp Warm-up.
Theo phân công project hiện tại, TV5 vẫn phụ trách workstream Web UI/UX riêng cho
demo/tích hợp; nhiệm vụ đó độc lập với pipeline reranking/evaluation/MLOps. TV5
không viết QA prompt.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [x] Viết `RerankerClient` trong `src/udsc2026/infrastructure/reranker/`, load Cross-Encoder từ local path hoặc config.
- [x] Viết `CrossEncoderReranker` trong `src/udsc2026/retrieval/reranking/`, nhận query và candidates từ Hybrid Search, trả danh sách đã sắp xếp lại.
- [x] Hỗ trợ batch scoring với `batch_size`, `device`, `max_length`, `top_n` để kiểm soát latency.
- [x] Bảo toàn metadata citation gồm `chunk_id`, `doc_id`, `law_name`, `article`, `clause`, `point`, `source`, `parent_id`.
- [x] Giữ score gốc từ Dense/Sparse/Hybrid và bổ sung `rerank_score`, `final_score`, `rank`.
- [x] Viết Evaluation Benchmark trong `src/udsc2026/evaluation/` để đo metric phát triển MRR, Recall@K, ROUGE-L, latency; metric LegalIR chính thức ở cấp `document_id`; và METEOR/ROUGE-L diagnostic cho LegalQA.
- [x] Hỗ trợ benchmark synthetic của TV4 và adapter riêng cho `data/task1/warmup.json`, `data/task2/warmup.json`; test set public/private dùng cùng contract theo từng task khi BTC phát hành.
- [x] Viết `scripts/evaluation/evaluate.py` cho benchmark chung, `scripts/evaluation/evaluate_legal_ir.py` cho macro Recall/Precision multi-gold và `scripts/evaluation/evaluate_legal_qa.py` cho diagnostic METEOR/ROUGE-L.
- [x] Viết writer/validator `submission.zip` → `submission.json` đúng schema object chính thức của LegalIR và LegalQA; giữ CSV tổng quát dưới nhãn legacy, không dùng để nộp task nào.
- [x] Tạo Multi-stage Docker Image cho backend: stage build dependencies, stage runtime gọn nhẹ.
- [x] Không copy model weights, raw data hoặc vector store lớn vào Docker image; mount bằng volume hoặc cấu hình path.
- [x] Viết `docker-compose.yml` để chạy backend, VectorDB, Redis cache và các service phụ trợ cần thiết.
- [x] Chuẩn hóa `.env.example` cho model path, vector DB URL, Redis URL, device, batch size, port, Task 1/Task 2 submission path và legacy CSV path.
- [x] Viết smoke test/CI command kiểm tra import, config, reranker, evaluation/submission của hai task và cấu hình Docker.

## 3. Quy chuẩn Clean Code & API Contract

### Clean Code bắt buộc

- Áp dụng triệt để DRY, tối ưu số dòng code và không tạo class/interface dư thừa nếu chưa có nhu cầu thật.
- Dùng type hinting đầy đủ cho mọi input/output; evaluation sample, metric result và submission row phải có schema rõ ràng.
- Mỗi hàm chỉ làm một trách nhiệm: score rerank, sort result, compute metric, write report hoặc write submission.
- Không copy-paste metric logic; MRR, Recall@K, METEOR và ROUGE-L phải là các hàm riêng, test được và gắn rõ metric profile.
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

LegalIR submission writer:

```python
def write_legal_ir_submission(
    payload: object,
    output_path: str,
    *,
    expected_question_ids: list[str] | None = None,
    allowed_document_ids: list[str] | None = None,
) -> None: ...
```

LegalQA evaluator và submission writer:

```python
def evaluate_legal_qa(
    references: Sequence[LegalQAWarmupSample],
    predictions: Sequence[LegalQAPrediction],
) -> LegalQAEvaluationReport: ...

def write_legal_qa_submission(
    payload: object,
    output_path: str,
    *,
    expected_question_ids: Iterable[str] | None = None,
) -> None: ...
```

Wire format LegalIR là object
`{"<question_id>": {"answer": ["<document_id>", "..."]}}`; wire format
LegalQA là object `{"<question_id>": {"answer": "<câu trả lời>"}}`. Mỗi file
ZIP chỉ chứa duy nhất `submission.json`. Array/list model nếu có chỉ là biểu
diễn prediction nội bộ trước bước serialize. Writer CSV `write_submission()`
được giữ riêng dưới nhãn legacy và không phải format nộp chính thức của Task 1
hoặc Task 2.

METEOR/ROUGE-L local của Task 2 chỉ phục vụ diagnostic. Report bắt buộc ghi
`evaluation_scope="local_diagnostic"` và `official_scorer_parity=false`; chưa
được tuyên bố parity cho đến khi BTC công bố implementation/config scorer ẩn
hoặc cung cấp golden result để đối chiếu.

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [x] Cross-Encoder reranker nhận được top-K candidates từ Hybrid Search và có test scorer deterministic; inference checkpoint thật cần artifact model.
- [x] Metadata citation được giữ nguyên sau rerank, không mất `chunk_id` hoặc điều/khoản.
- [x] Output có `rerank_score`, `final_score` và `rank` rõ ràng.
- [x] Benchmark đo được MRR, Recall@K, ROUGE-L và latency trên synthetic dataset của TV4.
- [x] Script evaluation sinh được report trước/sau rerank.
- [x] Audit/evaluator LegalQA đọc canonical `data/task2/warmup.json`, kiểm tra exact coverage và sinh report diagnostic có fingerprint.
- [x] Script xuất/validate được `artifacts/task1/submission.zip` LegalIR và `artifacts/task2/submission.zip` LegalQA theo hai schema object tách biệt.
- [ ] Docker multi-stage **build thực tế** thành công trong máy có Docker daemon; static config và host smoke đã có nhưng không thay thế build.
- [ ] Stack backend/VectorDB/Redis **chạy thực tế** với model/index được mount; Compose config đã được kiểm tra tĩnh.

Chi tiết dữ liệu, contract và lệnh vận hành:

- LegalIR: [`docs/tv5_legalir_warmup.md`](../tv5_legalir_warmup.md);
- LegalQA: [`docs/tv5_legalqa_warmup.md`](../tv5_legalqa_warmup.md).
