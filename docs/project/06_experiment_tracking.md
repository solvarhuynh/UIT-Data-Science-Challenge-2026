# Experiment Tracking

## Mục tiêu

Ghi log mọi thí nghiệm tinh chỉnh hệ thống RAG (retrieval, chunking, reranking, generation) theo một chuẩn chung, tránh chạy trùng thí nghiệm, và cho phép so sánh nhanh giữa các phiên bản khi viết báo cáo/bài báo cho vòng chấm.

Mọi thành viên khi chạy xong một thí nghiệm **bắt buộc** thêm 1 dòng vào bảng log tương ứng bên dưới trước khi merge PR. Không ghi log = thí nghiệm coi như chưa tồn tại.

## Quy ước đặt tên Experiment Name

```text
<module>_<mô_tả_ngắn>_<vNN>
```

Ví dụ: `retrieval_bkai_baseline_v1`, `chunk_512_overlap_80_v2`, `rerank_crossencoder_top20_v1`, `qa_qwen3_fewshot_v3`.

## Bảng log thí nghiệm chính

| Experiment Name | Module | Dataset | Config (Chunk Size/Overlap, top_k, model...) | Metrics | Result | Người thực hiện | Ngày | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| retrieval_bkai_baseline_v1 | Retrieval (Dense) | `data/processed/chunks/v1` | chunk_size=512, overlap=80, top_k=10, model=bkai-bi-encoder | Recall@10, MRR@10 | Recall@10=0.71, MRR@10=0.58 | TV2 | 2026-07-01 | Baseline chưa hybrid |
| retrieval_hybrid_bm25_dense_v1 | Retrieval (Hybrid) | `data/processed/chunks/v1` | dense_weight=0.6, sparse_weight=0.4, top_k=10 | Recall@10, MRR@10 | Recall@10=0.79, MRR@10=0.64 | TV3 | 2026-07-05 | Cải thiện so với dense-only |
| rerank_crossencoder_top20_v1 | Reranking | `data/processed/chunks/v1` | retrieve_top_k=20, rerank_top_k=5, model=cross-encoder | Recall@5, nDCG@5 | Recall@5=0.83, nDCG@5=0.77 | TV5 | 2026-07-10 | Baseline cross-encoder |
| qa_qwen3_zero_shot_v1 | Generation | `data/eval/qa_set_v1` | model=qwen3-1.7b-legal, temperature=0.2 | EM, F1, Citation Accuracy | EM=0.42, F1=0.61, CitAcc=0.70 | TV3 | 2026-07-12 | Chưa few-shot |
| *(thêm dòng mới ở đây)* | | | | | | | | |

> Ghi chú: bảng trên chỉ là ví dụ minh hoạ tiến trình từ Baseline (dense-only) → Hybrid → Cross Encoder → Generation. Xoá các dòng ví dụ khi bắt đầu log thật, giữ lại header.

## Định nghĩa cột

- **Experiment Name**: định danh duy nhất, theo quy ước ở trên.
- **Module**: `Ingestion / Retrieval (Dense) / Retrieval (Sparse) / Retrieval (Hybrid) / Reranking / Generation / End-to-End`.
- **Dataset**: đường dẫn hoặc version của tập dữ liệu dùng để chạy thí nghiệm (nên trỏ tới `data/processed/...` hoặc `data/eval/...` có version rõ ràng, không dùng dữ liệu chưa qua ETL).
- **Config**: các siêu tham số ảnh hưởng kết quả — với ingestion/retrieval là `chunk_size`, `chunk_overlap`, `top_k`, `distance metric`; với reranking là `retrieve_top_k`, `rerank_top_k`, checkpoint model; với generation là `temperature`, `max_new_tokens`, prompt template version.
- **Metrics**: tên metric dùng để đánh giá (`Recall@K`, `MRR@K`, `nDCG@K`, `EM`, `F1`, `Citation Accuracy`, `Latency`).
- **Result**: giá trị số đo được, nên kèm đơn vị hoặc thang đo.
- **Người thực hiện**: tên/TV phụ trách chạy thí nghiệm.
- **Ngày**: ngày chạy (định dạng `YYYY-MM-DD`).
- **Ghi chú**: nhận xét ngắn, lý do tăng/giảm so với thí nghiệm trước, hoặc hướng thử tiếp theo.

## Bảng theo dõi tiến trình Retrieval (Baseline → Cross Encoder)

Bảng riêng để so sánh trực quan tiến trình cải thiện pipeline retrieval qua từng giai đoạn.

| Giai đoạn | Experiment Name | Recall@10 | MRR@10 | nDCG@10 | So với baseline |
|---|---|---|---|---|---|
| 1. Baseline Dense (BKAI) | retrieval_bkai_baseline_v1 | | | | — |
| 2. + Sparse (BM25) | retrieval_bm25_v1 | | | | |
| 3. + Hybrid (Dense+Sparse) | retrieval_hybrid_bm25_dense_v1 | | | | |
| 4. + Cross-Encoder Reranking | rerank_crossencoder_top20_v1 | | | | |

## Quy trình bắt buộc khi chạy thí nghiệm

1. Kiểm tra bảng log trước — nếu config gần giống thí nghiệm đã có, không chạy lại, chỉ chạy phần khác biệt.
2. Chạy thí nghiệm trong `experiments/tvX/` tương ứng, không sửa trực tiếp `src/udsc2026/`.
3. Lưu kết quả thô (log, output JSON) vào `experiments/tvX/results/<experiment_name>/`.
4. Thêm 1 dòng vào bảng log trong file này (PR riêng hoặc kèm PR code, nhưng phải có trong cùng PR).
5. Nếu thí nghiệm là điểm mốc quan trọng (ví dụ vượt baseline rõ rệt), gắn thẻ trong Ghi chú: `[MILESTONE]`.

## Liên quan

- Version model dùng trong thí nghiệm: xem `docs/models/model_registry.md`.
- Định dạng dữ liệu đầu vào: xem `docs/project/11_data_pipeline.md`.
- Kiến trúc tổng thể: xem `docs/project/10_system_design.md`.
