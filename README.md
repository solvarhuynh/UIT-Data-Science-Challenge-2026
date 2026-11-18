# UDSC 2026 — LegalIR và LegalQA

Mã nguồn thi UIT Data Science Challenge 2026 của HCMUTE-SHIPCODE. Repo tập trung vào hai đường chạy: truy hồi `document_id` cho LegalIR và sinh câu trả lời có căn cứ cho LegalQA. Mã được tổ chức theo `src/udsc2026` (thư viện), `scripts/` (CLI), `configs/` và `prompts/`. Dữ liệu BTC, chỉ mục, checkpoint và dự đoán được giữ ngoài Git.

## Trạng thái có thể kiểm chứng

- Các CLI đánh giá và đóng gói submission đã được kiểm tra `--help` trên Python 3.12.6. Đây là kiểm tra nhập module và tham số, **chưa phải** chạy end-to-end.
- Checkout hiện thiếu `data/raw/btc/LegalIR/train.json`, selected contexts và manifest corpus `data/processed_v3/metadata/manifest.json`; vì vậy chưa thể xác nhận lại điểm hoặc train hợp lệ. `data/` hiện có một bản processed corpus ở ngay thư mục gốc, nhưng manifest của bản này ghi `semantic_completeness_gate_passed=false` và `quality_gate_passed=false`; chưa được dùng làm V3 canonical.
- `scripts/inference/run_legal_ir.py` cũ chỉ tìm artifact cache rồi báo stage đã qua; nó không chạy retrieval/reranking. Đường chạy thật là các CLI riêng ở dưới.
- Điểm, bài học từ sáu repo và chín poster nằm ở [GUIDE.md](GUIDE.md); chỉ dẫn cho agent ở [AGENT.md](AGENT.md). Các báo cáo từng đội ở `docs/comparisons/`.

## Cài đặt

Python 3.10–3.12. Trên máy dùng GPU, cài bản PyTorch/CUDA phù hợp với driver trước khi cài các extra nặng.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -e ".[data,retrieval,models]"
```

Cho LLM/reranker, cài thêm `.[llm,rerank]`. Model được nạp từ `models/` theo `configs/gpu.yaml`; repo không chứa weights. `python download_models.py --help` liệt kê profile tải model. Không tải model hoặc build lại corpus trước khi xác minh nguồn BTC và dung lượng.

## Dữ liệu và chỉ mục

Đặt dữ liệu BTC theo schema gốc dưới `data/raw/btc/`. Chạy ETL vào **candidate**, xem audit/manifest, rồi mới promotion sang `data/processed_v3`; không ghi đè V3 đã fingerprint.

```powershell
python scripts/data_prep/run_ingestion.py --raw-directory data/raw/btc --processed-root data/processed_candidate
python scripts/data_prep/index_chunks.py --chunks-dir data/processed_v3/chunks --config-env gpu --vector-db-type faiss
python scripts/data_prep/build_bm25.py --chunks data/processed_v3/chunks --output data/vector_store/bm25/index.json
```

## Task 1 — LegalIR

Metric chính là macro Recall@5 trên **document ID**; macro Precision là tie-break. Không chọn model theo chunk recall đơn lẻ. Giữ tách biệt train và validation theo nhóm câu hỏi chuẩn hóa.

```powershell
python scripts/evaluation/build_strict_legal_ir_cv.py --input data/raw/btc/LegalIR/train.json --output-dir artifacts/task1/cv
python scripts/evaluation/generate_dense_candidates.py --help
python scripts/evaluation/benchmark_reranker.py --help
python scripts/evaluation/analyze_legal_ir_candidates.py --help
```

Sau khi có dense và BGE predictions cùng ID set, tạo ensemble. Ở fold validation, `--exclude-question-sources` phải trỏ đến nguồn ID validation để không dùng nhãn của chính fold.

```powershell
python scripts/inference/build_legal_ir_ensemble.py --questions data/task1/questions.json --dense artifacts/task1/dense.jsonl --reranked artifacts/task1/bge.jsonl --labeled data/raw/btc/LegalIR/train.json --exclude-question-sources data/task1/questions.json --contexts-dir data/raw/btc/LegalIR/selected-contexts --output artifacts/task1/predictions.json
python scripts/evaluation/evaluate_legal_ir.py --references data/task1/references.json --predictions artifacts/task1/predictions.json --output artifacts/task1/evaluation.json
python scripts/evaluation/compare_runs.py --task legal-ir --baseline artifacts/task1/baseline_report.json --candidate artifacts/task1/evaluation.json --questions data/task1/questions.json --output artifacts/task1/uplift.json
python scripts/inference/write_legal_ir_submission.py --input artifacts/task1/predictions.json --questions data/task1/questions.json --output artifacts/task1/submission.zip
python scripts/inference/validate_legal_ir_submission.py --input artifacts/task1/submission.zip --questions data/task1/questions.json
```

Các đường dẫn `data/task1/questions.json` và `references.json` là input do người chạy cung cấp theo phase, không có sẵn trong repo. Khi suy luận tập test, không dùng `--exclude-question-sources` để loại toàn bộ train labels; chỉ loại nguồn trùng ID với tập đang đánh giá. Không dùng nhãn test/public/private để fit hoặc tune.

## Task 2 — LegalQA

Đánh giá retrieval evidence trước khi tinh chỉnh generator. `build_extractive_legal_qa.py` là baseline không cần LLM; `generate_legal_qa_answers.py` dùng contexts đã xếp hạng và checkpoint local.

```powershell
python scripts/inference/build_extractive_legal_qa.py --dense artifacts/task2/dense.jsonl --reranked artifacts/task2/reranked.jsonl --parents-dir data/processed_v3/parents --output artifacts/task2/extractive.json
python scripts/inference/generate_legal_qa_answers.py --questions data/task2/questions.json --contexts artifacts/task2/dense.jsonl --output artifacts/task2/answers.json --diagnostics artifacts/task2/diagnostics.json --config configs/gpu.yaml
python scripts/evaluation/evaluate_legal_qa.py --references data/task2/references.json --predictions artifacts/task2/answers.json --output artifacts/task2/candidate_report.json
python scripts/evaluation/compare_runs.py --task legal-qa --baseline artifacts/task2/baseline_report.json --candidate artifacts/task2/candidate_report.json --questions data/task2/questions.json --output artifacts/task2/uplift.json
python scripts/inference/write_legal_qa_submission.py --input artifacts/task2/answers.json --questions data/task2/questions.json --output artifacts/task2/submission.zip
python scripts/inference/validate_legal_qa_submission.py --input artifacts/task2/submission.zip --questions data/task2/questions.json
```

Task 2 ưu tiên METEOR; scorer nội bộ không được gọi là điểm Private. `compare_runs.py` lấy bootstrap theo nhóm câu hỏi và chỉ mở cổng nội bộ khi có ít nhất 100 nhóm, gain dương và `P(Δ>0) ≥ 0,70`; đây không phải xác suất Private. Đảm bảo tổng tham số model đang hoạt động thỏa rule cuộc thi. Các lệnh đánh giá/đóng gói đã qua smoke với dữ liệu nhỏ tạm thời; inference/training chưa chạy end-to-end do thiếu input và weights.

## Cấu trúc

```text
configs/             Cấu hình baseline và GPU
docs/comparisons/    Sáu báo cáo so sánh repo đối thủ
prompts/             Prompt LegalQA được chọn
scripts/data_prep/   ETL và đánh chỉ mục
scripts/evaluation/  Scorer, grouped CV, phân tích candidate
scripts/inference/   Ensemble, QA, đóng gói và kiểm định submission
scripts/training/    Fine-tune có kiểm soát
src/udsc2026/       Contracts, ingestion, retrieval, evaluation, QA
```

Mỗi thay đổi metric cần lưu input hash, corpus manifest, checkpoint, split, scorer và dự đoán từng query. Chỉ chi GPU sau khi baseline và cổng xác suất cải thiện trong [AGENT.md](AGENT.md) có dữ liệu để đánh giá.

## Giấy phép

Xem [LICENSE](LICENSE).
