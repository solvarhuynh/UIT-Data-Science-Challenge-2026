# BAAI/bge-reranker-v2-m3

## Tổng quan

`BAAI/bge-reranker-v2-m3` là mô hình **cross-encoder reranker đa ngôn ngữ** do [Beijing Academy of Artificial Intelligence (BAAI)](https://huggingface.co/BAAI) phát triển. Model nhận đồng thời một query và một passage, sau đó dự đoán mức độ liên quan trực tiếp giữa hai chuỗi.

> **Vai trò trong UDSC2026:** model không thay thế embedding model hoặc vector database. Nó được dùng ở bước sau dense/sparse/hybrid retrieval để sắp xếp lại một tập ứng viên nhỏ, chất lượng cao.

Model card chính thức: [BAAI/bge-reranker-v2-m3 trên Hugging Face](https://huggingface.co/BAAI/bge-reranker-v2-m3).

## Đặc điểm chính

| Thuộc tính | Thông tin |
|---|---|
| Nhà phát triển | BAAI |
| Kiến trúc | Cross-encoder, sequence classification |
| Base model | BGE-M3 / XLM-RoBERTa family |
| Ngôn ngữ | Multilingual, phù hợp cho cross-lingual retrieval; cần benchmark riêng với tiếng Việt pháp lý |
| Maximum sequence length | Tối đa 8192 tokens theo model card/configuration; pipeline thực tế có thể đặt giới hạn thấp hơn để kiểm soát latency |
| Kích thước | Khoảng 0.6B tham số |
| Định dạng | Safetensors, F32 trên model card |
| License | Apache-2.0 |
| Đầu ra | Một relevance logit cho mỗi cặp query–passage |

### Cross-encoder khác gì bi-encoder?

Bi-encoder mã hóa query và document độc lập thành vector để tìm kiếm nhanh trên FAISS/Qdrant. Cross-encoder đọc cả hai chuỗi trong cùng một lượt inference nên có thể mô hình hóa tương tác chi tiết giữa từ khóa, điều khoản và ngữ cảnh, nhưng chi phí tính toán cao hơn.

Luồng sử dụng điển hình:

```text
Query
  │
  ├─ Dense/Sparse/Hybrid retrieval → top-N candidates
  │
  └─ bge-reranker-v2-m3(query, candidate) → relevance score
                                      │
                                      └─ sort / threshold → top-k context cho QA
```

## Use-cases trong Legal IR / QA

- Rerank top 20–100 chunk do dense retrieval hoặc hybrid retrieval trả về.
- Ưu tiên passage thực sự trả lời câu hỏi thay vì chỉ chứa từ khóa giống query.
- Phân biệt các điều khoản gần giống nhau trong văn bản luật.
- Rerank kết quả LegalIR trước khi tính Recall@k, MRR hoặc nDCG.
- Lọc context trước khi đưa vào LegalQA để giảm context nhiễu cho generator.
- Hỗ trợ query tiếng Việt và passage có ngôn ngữ khác trong các thử nghiệm cross-lingual.

Không nên chạy model trên toàn bộ corpus. Hãy dùng embedding/BM25 để tạo candidate set trước, vì cross-encoder cần một lần tính điểm cho từng cặp query–passage.

## So sánh định tính

| Tiêu chí | `bge-reranker-v2-m3` | `bge-reranker-large` | Các checkpoint BGE reranker cũ hơn / bản nội bộ |
|---|---|---|---|
| Đa ngôn ngữ | Mạnh, là mục tiêu thiết kế chính | Thiên về tiếng Anh/Trung hơn | Phụ thuộc checkpoint |
| Cross-lingual | Có thể thử nghiệm trực tiếp | Không phải ưu điểm chính | Thường hạn chế hơn |
| Ngữ cảnh dài | Hỗ trợ tối đa 8192 tokens | Thường phải giới hạn ngắn hơn theo checkpoint/pipeline | Phụ thuộc cấu hình |
| Độ chính xác ngữ nghĩa | Tốt cho rerank candidate đa ngôn ngữ | Có thể tốt trên dữ liệu tiếng Anh/Trung phù hợp | Không đồng nhất |
| Tốc độ | Tốt hơn các reranker LLM lớn, nhưng vẫn chậm hơn bi-encoder | Có thể nhanh/chậm tùy phần cứng và batch | Thường nhẹ hơn nếu model nhỏ |
| Chi phí inference | Cao hơn retrieval vector | Cao | Thấp hơn nếu candidate/model nhỏ |

Các so sánh trên là định tính. Quyết định dùng trong pipeline chính thức phải dựa trên benchmark tiếng Việt pháp lý của dự án, cùng candidate set, batch size, max length và phần cứng.

## Hướng dẫn thử nghiệm

### Mục tiêu

Đánh giá model theo hai chiều: mức cải thiện chất lượng xếp hạng và chi phí latency/bộ nhớ so với baseline không rerank.

Candidate set phải được giữ cố định giữa baseline và reranker. Nên thử top-20, top-50 và top-100 candidates do dense/sparse/hybrid retrieval tạo ra.

Các biến cần ghi lại:

- Revision model, tokenizer và phiên bản dữ liệu.
- `max_length`, batch size, device và precision.
- Số candidate/query, số kết quả giữ lại và chiến lược truncate.
- Chunking strategy và nhãn ground truth.

### Checklist benchmark

**Chất lượng**

- [ ] Chạy baseline không rerank.
- [ ] Chạy `bge-reranker-v2-m3` trên cùng candidate set.
- [ ] Đo Recall@5/10/20, MRR@10 và nDCG@k cho LegalIR.
- [ ] Đo metric LegalQA theo yêu cầu của BTC.
- [ ] Phân tích riêng query ngắn/dài, nhiều điều kiện và query cross-lingual.
- [ ] Review lỗi nhầm điều khoản, ngoại lệ, thời hạn và chủ thể pháp lý.

**Hiệu năng**

- [ ] Warm-up trước khi đo.
- [ ] Ghi model-load time, p50/p95/p99 latency và throughput.
- [ ] Ghi peak RAM/VRAM theo batch size.
- [ ] Đo trên đúng phần cứng triển khai mục tiêu.
- [ ] So sánh candidate count 20/50/100 và max length 256/512/1024.

**Tái lập**

- [ ] Lưu revision model, metadata benchmark và random seed.
- [ ] Lưu số liệu theo từng query, không chỉ giá trị trung bình.
- [ ] Lưu các false positive/false negative tiêu biểu.

## Scoring và thresholding

Model trả về một **relevance logit** cho mỗi cặp `[query, passage]`. Điểm càng cao thì passage càng liên quan; khi xếp hạng, sort giảm dần theo raw logit.

Có thể ánh xạ logit `s` về `[0, 1]` bằng sigmoid:

```text
p = sigmoid(s) = 1 / (1 + exp(-s))
```

Raw logit phù hợp để sort trong cùng model/batch. Sigmoid score dễ hiển thị và đặt threshold hơn, nhưng không mặc nhiên là xác suất đã calibration.

> Không nên mặc định dùng threshold `0.5`. Hãy chọn threshold trên development set, sau đó xác nhận trên holdout/test set.

Quy trình đề xuất:

1. Chia dữ liệu thành development và holdout/test.
2. Lưu raw logit và sigmoid score cho từng candidate.
3. Thử dải threshold, ví dụ 0.1–0.9.
4. Chọn theo mục tiêu: ưu tiên recall để hạn chế bỏ sót điều khoản, hoặc precision để giảm context nhiễu cho QA.
5. Kiểm tra độ ổn định theo loại query và độ dài passage.

Nếu pipeline giữ cố định top-k, threshold dùng như bộ lọc bổ sung sau khi sort. Cần định nghĩa fallback khi threshold làm rỗng candidate set.

### Mẫu báo cáo

| Model | Candidates/query | Max length | Device | Recall@10 | MRR@10 | QA metric | p50 (ms) | p95 (ms) | Peak VRAM |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Baseline, không rerank | 50 | — | — | | | | | | |
| `bge-reranker-v2-m3` | 50 | 512 | CPU/GPU | | | | | | |

## Trade-off latency và tài nguyên

- Chi phí tăng gần tuyến tính theo số cặp được rerank: `số query × số candidates/query`.
- `max_length=8192` hữu ích cho văn bản dài nhưng có thể làm tăng đáng kể bộ nhớ và latency. Với chunk đã cắt ngắn, nên benchmark các mức 256/512/1024 trước.
- Batch inference thường tăng throughput nhưng cần theo dõi peak RAM/VRAM.
- FP16/BF16 có thể giảm latency và bộ nhớ trên GPU; phải kiểm tra sai khác điểm số và chất lượng trước khi áp dụng.
- CPU phù hợp smoke test hoặc batch nhỏ; production retrieval nên đo p50/p95 latency trên đúng phần cứng triển khai.

## Tài liệu tham khảo

- [Model card chính thức](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [FlagEmbedding – repository chính thức](https://github.com/FlagOpen/FlagEmbedding)
- [BGE-M3 paper](https://arxiv.org/abs/2402.03216)
