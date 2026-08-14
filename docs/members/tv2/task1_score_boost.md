# Báo cáo tăng điểm và tăng hạng TV2 — Nhiệm vụ 1 LegalIR

Tài liệu này ghi lại các hướng tối ưu giúp Nhiệm vụ 1 tăng Recall, tăng điểm
và cải thiện thứ hạng: retrieval, tạo candidate, gộp document, hard-negative
mining và reranking. Các kết luận chất lượng chỉ được ghi khi có artifact hoặc
bài kiểm tra tương ứng; không suy diễn từ kết quả smoke sang điểm leaderboard.

## Trạng thái tổng quan

| Hạng mục | Trạng thái | Bằng chứng chính |
|---|---|---|
| Tương thích official scorer | Hoàn thành | `artifacts/task1/evaluation/p1_report.json`; golden tests |
| Chia dữ liệu 5-fold chống leakage | Hoàn thành | `artifacts/task1/evaluation/strict_cv_v2/folds.json` |
| Dense retrieval HCMUTE | Hoàn thành | manifest dense/index và cấu hình Nhiệm vụ 1 |
| Gộp chunk thành document candidate | Hoàn thành | cache document candidate 7,000 query |
| Tạo hard/semi-hard negatives | Hoàn thành | mining manifest và prerequisite gate |
| Fine-tune BGE reranker | Đang thực hiện | local dry-run hợp lệ; chờ kết quả đánh giá đầy đủ 5 folds |
| Synthetic legal queries | Chưa thực hiện — chỉ chạy khi cần | chỉ mở nếu reranker không còn cải thiện đủ |
| Chạy public pipeline và tạo submission | Tiếp theo | chỉ chạy sau khi chọn stack cuối |

## Nền retrieval và dữ liệu chuẩn

**Nhiệm vụ**

- Xây nền retrieval ổn định để các bước benchmark và reranking dùng cùng một
  corpus, schema và cách tính điểm.

**Ý tưởng thực hiện**

- Dùng embedding client, FAISS/Qdrant, BM25, hybrid search, citation metadata,
  manifest và regression tests.
- Giữ `data/processed_v3` là corpus chuẩn, chỉ đọc trong các experiment.

**Đã thực hiện**

- Hoàn thiện các module retrieval/index, `scripts/data_prep/index_chunks.py`
  và các kiểm tra contract/index.

**Tác dụng**

- Candidate, index và kết quả benchmark có thể tái lập; thí nghiệm không làm
  thay đổi corpus chuẩn.

## Chọn embedding model cho dense retrieval

**Nhiệm vụ**

- Chọn model phù hợp để tìm candidate document cho LegalIR.

**Ý tưởng thực hiện**

- Dùng cache và manifest để kiểm soát chất lượng retrieval và reuse index.

**Đã thực hiện**

- Chốt `huyydangg/DEk21_hcmute_embedding_v2` tại `models/dek21-v2` làm dense
  retrieval chính; embedding có 768 chiều.
- Dùng embedding client, index script và config Nhiệm vụ 1 để model/index luôn
  khớp nhau.

**Tác dụng**

- Candidate dense ổn định hơn, tạo đầu vào tốt cho reranker và tránh đổi index
  không kiểm soát.

## Chọn số lượng candidate và gộp chunk theo document

**Nhiệm vụ**

- Giữ đủ candidate đúng trước reranking nhưng không đưa nhiều chunk trùng lặp
  vào model.

**Ý tưởng thực hiện**

- Lấy raw chunk K=500, gộp các chunk cùng document và giữ top 200 document;
  mỗi document giữ tối đa 2 evidence chunks.

**Đã thực hiện**

- Cập nhật `scripts/evaluation/generate_dense_candidates.py` và
  `scripts/evaluation/collapse_task1_candidates.py` theo hướng streaming.
- Tạo cache document candidate cho 7,000 training query tại
  `artifacts/task1/candidates/train7000_document_candidates.jsonl`.

**Tác dụng**

- Giữ recall ở tầng retrieval, giảm duplicate chunk và giảm chi phí reranking.

## Chia dữ liệu 5-fold để đánh giá không leakage

**Nhiệm vụ**

- Đánh giá model trên câu hỏi chưa được dùng để train.

**Ý tưởng thực hiện**

- Chia 7,000 câu hỏi thành 5 folds; các câu normalized trùng nhóm luôn ở cùng
  một fold.

**Đã thực hiện**

- Dùng `artifacts/task1/evaluation/strict_cv_v2/folds.json` và evaluation
  utilities liên quan.
- Mỗi lần train dùng 5,600 query và đánh giá 1,400 query còn lại.

**Tác dụng**

- Giảm leakage, giúp kết quả đánh giá đáng tin cậy trước khi chọn model.

## Tạo hard/semi-hard negative samples cho reranker

**Nhiệm vụ**

- Tạo dữ liệu train khó thay vì chỉ dùng negative ngẫu nhiên.

**Ý tưởng thực hiện**

- Positive là document đúng; hard và semi-hard negative là document sai nhưng
  được retrieval xếp cao hoặc có nội dung gần nghĩa.

**Đã thực hiện**

- Dùng `scripts/training/mine_task1_negatives.py` và
  `scripts/evaluation/check_task1_p13_prerequisites.py`.
- Đã tạo khoảng 514k negative records cho 5 folds; gate xác nhận đủ candidate,
  positive và negative khó, không có leakage.

**Tác dụng**

- Reranker học phân biệt các văn bản pháp lý dễ gây nhầm tốt hơn và tập trung
  vào lỗi retrieval thực tế.

## Fine-tune BGE reranker bằng hard negatives

**Nhiệm vụ**

- Cải thiện thứ tự document sau dense retrieval.

**Ý tưởng thực hiện**

- Fine-tune `BAAI/bge-reranker-v2-m3` bằng query, document đúng và document
  sai nhưng khó; đánh giá bằng 5 folds.

**Đã thực hiện**

- Hoàn thiện `scripts/training/finetune_task1_bge_reranker.py` với 2
  epochs/fold, candidate depth 200, batch 4, gradient accumulation 4 và max
  length 512.
- Local dry-run đã kiểm tra dữ liệu và split; đang thực hiện full 5-fold
  training.

**Tác dụng**

- Đẩy document liên quan lên thứ hạng cao hơn và kiểm tra khả năng generalize
  trước khi đổi stack cuối.

## Bổ sung synthetic legal queries nếu reranker không còn cải thiện

**Nhiệm vụ**

- Chỉ tăng dữ liệu train khi fine-tuned reranker chưa cải thiện đủ.

**Ý tưởng thực hiện**

- Sinh query từ legal corpus, không tạo synthetic label cho public questions
  và luôn ưu tiên organizer queries.

**Đã thực hiện**

- Chưa thực hiện — chỉ chạy khi cần.

**Tác dụng**

- Bổ sung dữ liệu ở vùng train còn yếu mà không mở rộng experiment không cần
  thiết.

## Chạy pipeline cuối và tạo submission

**Nhiệm vụ**

- Dùng stack đã chọn để tạo file nộp LegalIR đúng format.

**Ý tưởng thực hiện**

- Dense retrieval → document candidates → reranking → top document IDs →
  `submission.zip` → validation.

**Đã thực hiện**

- Tiếp theo, dùng `scripts/task1/run_legal_ir_pipeline.py`,
  `scripts/submission/write_legal_ir_submission.py` và
  `scripts/submission/validate_legal_ir_submission.py` sau khi review xong
  full evaluation.

**Tác dụng**

- Chuyển pipeline đã benchmark thành output có thể kiểm tra trước khi nộp.

## Kết quả đã đo trên cache diagnostic

| Metric | Giá trị |
|---|---:|
| CandidateDocRecall@5 | 0.769833 |
| CandidateDocRecall@100 | 0.946000 |
| CandidateDocRecall@200 | 0.965000 |
| Số document unique trung bình ở depth 200 | 61.258 |
| Retrieval misses | 11 |

Đây là diagnostic cache 500 query, không phải điểm leaderboard.

## Kiểm tra trước khi bàn giao

```powershell
python -m compileall -q src scripts
git diff --check
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier unit
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier data
```
