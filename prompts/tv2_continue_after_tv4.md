# Prompt nhắc tiếp tục TV2 sau khi TV4 bàn giao context_id

```text
TV4 đã bàn giao JSONL LegalChunk mới và đã bổ sung metadata.context_id theo
contract đã thống nhất. Hãy tiếp tục các nhiệm vụ TV2 còn dang dở trong repo
UDSC2026 theo đúng thứ tự:

1. Prompt 2 — chạy/hoàn thiện scripts/validate_chunk_mapping.py:
   - đọc toàn bộ JSONL từ --chunks-dir;
   - validate LegalChunk;
   - kiểm tra duplicate chunk_id, text rỗng, thiếu source/doc_id;
   - kiểm tra metadata.context_id map được toàn bộ answer.context_id của
     LegalIR train.json;
   - ghi data/processed/metadata/chunk_mapping_report.json;
   - exit code khác 0 nếu duplicate hoặc orphan ground-truth;
   - không sửa raw data/chunks.

2. Chỉ khi Prompt 2 pass, thực hiện Prompt 5 — tạo/chạy
   scripts/benchmark_retrieval_internal.py:
   - hỗ trợ dense, sparse, hybrid;
   - mặc định top_k=5;
   - map hit về metadata.context_id, không đoán field khác;
   - báo Recall@5, Precision@5, số miss và top 10 câu hỏi thấp nhất;
   - ghi data/reports/tv2_retrieval_internal_<mode>.json;
   - không sửa index/chunks.

3. Sau đó chạy Prompt 7:
   - toàn bộ tests/retrieval/;
   - validator mapping;
   - index_chunks.py trên corpus thật và xác nhận manifest;
   - benchmark cả dense/sparse/hybrid;
   - cập nhật chỉ mục TV2 trong PROJECT_STATUS.md bằng ngày chạy,
     corpus_hash, Recall@5 và Precision@5;
   - không cập nhật phần thành viên khác;
   - không commit raw data, index lớn hoặc model weights.

Trước khi code, hãy đọc lại contract LegalChunk/RetrievalHit và in ra một
record mẫu có metadata.context_id để xác nhận schema. Nếu context_id vẫn
thiếu, tên field khác, hoặc train.json không có schema question_id ->
{question, answer:[context_id]}, hãy dừng và hỏi tôi. Nếu môi trường thiếu
model hoặc Docker, báo lỗi cụ thể và không tự nới lỏng validation.

Sau mỗi prompt, báo cáo file đã tạo/sửa, lệnh đã chạy, kết quả test và blocker
trước khi chuyển bước tiếp theo.
```
