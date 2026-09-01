# Hướng dẫn cấu trúc kho lưu trữ: `data/`, `artifacts/`, `models/`

Tài liệu này là bản hướng dẫn định hướng súc tích dành cho AI agent. Tài liệu mô tả ý nghĩa của từng khu vực cấp cao nhất (top-level), các đường dẫn nào là chuẩn xác (authoritative), và các tệp nào thường có dung lượng lớn/được sinh tự động. Không tự suy diễn nhãn (labels) hay trạng thái production chỉ dựa vào tên tệp; hãy sử dụng manifests và báo cáo (reports) khi có sẵn.

## 1. `data/` — Dữ liệu nguồn, tập ngữ liệu đã xử lý, và các chỉ mục (indexes)

`data/` là không gian làm việc về dữ liệu. Nó chứa các tập dữ liệu nguồn, các tệp ngữ liệu (corpus) đã chuẩn hóa/tiền xử lý, và các chỉ mục truy xuất (retrieval indexes). Các tệp có thể có dung lượng lớn và thường không được đưa vào quản lý phiên bản mã nguồn (source control / git).

```text
data/
├── README.md
├── raw/
│   ├── btc/
│   │   ├── LegalIR/
│   │   └── LegalQA/
│   ├── UDSC2026_Plan.pdf
│   └── .gitkeep
├── processed_v3/
│   ├── benchmarks/
│   ├── chunks/
│   ├── documents/
│   ├── metadata/
│   ├── parents/
│   └── .gitkeep
└── vector_store/
    ├── bm25/
    ├── faiss/
    │   └── legal_chunks_dek21_v2_768/
    ├── qdrant/
    └── .gitkeep
```

### Cách hiểu và sử dụng `data/`

- `raw/` là khu vực dữ liệu đầu vào gốc. `raw/btc/LegalIR/` và `raw/btc/LegalQA/` chứa dữ liệu của ban tổ chức/đề bài; không chỉnh sửa trực tiếp (in-place) tại đây.
- `processed_v3/` là tập ngữ liệu đã chuẩn hóa được dùng cho việc truy xuất và đánh giá. Các lớp ngữ nghĩa tiêu biểu gồm documents, parent documents, chunks, metadata và benchmark fixtures.
- `vector_store/` chứa các backend/dữ liệu hiện thực hóa cho truy xuất. FAISS là định dạng chỉ mục vector cục bộ chính; BM25 và Qdrant là các backend hoặc bản xuất dữ liệu riêng biệt.
- `task1/` và `task2/` chứa các dữ liệu đầu vào khởi động/hỗ trợ theo từng tác vụ cụ thể. Hướng dẫn này chủ ý không quy định cứng ngữ nghĩa theo từng tác vụ cho `data/`.

## 2. `artifacts/task1/` — Minh chứng, kết quả production, và lịch sử thực nghiệm của Task 1

Chỉ có cây thư mục artifact của Task 1 được ghi chép tại đây. `artifacts/task2/` nằm ngoài phạm vi này và không được sửa đổi khi đang thực hiện Task 1.

```text
artifacts/task1/
├── submission.zip                         # tệp nộp bài cuối cùng; cần bảo vệ
├── corpus_document_ids.json                # đối chiếu/kiểm tra tính hợp lệ của document-ID
├── public_benchmark_adapter.jsonl         # minh chứng public adapter
├── production paths under recovery_096/
├── candidates/                             # đầu vào ứng viên / tái lập thực nghiệm
├── cv_strict/                              # minh chứng strict grouped-CV
├── evaluation/                             # độ đo (metrics), điều kiện kiểm tra (gates), chẩn đoán
├── models/                                 # minh chứng mô hình / checkpoints của Task 1
├── training/                               # minh chứng huấn luyện / khai thác negative mẫu khó
├── recovery_096/                            # cây nguồn gốc / khôi phục V3A chính
├── research/                               # các thử nghiệm cũ (legacy), bị loại bỏ, chẩn đoán, kiểm thử nhanh (smoke)
├── archive/                                # dữ liệu lịch sử chưa xử lý / chưa hoàn chỉnh
├── raw_k200.jsonl + manifest               # bộ nhớ đệm (cache) train lịch sử dung lượng lớn
└── raw_k500.jsonl + manifest               # bộ nhớ đệm (cache) train lịch sử dung lượng lớn
```

### Bộ tệp Production của Task 1 (các đường dẫn chuẩn tương thích hệ thống)

Quy ước production hiện tại chủ ý giữ nguyên các đường dẫn đã thiết lập này vì các script khởi chạy (launchers) và manifests đang tham chiếu đến chúng:

```text
artifacts/task1/recovery_096/
├── final_public_v3/
│   ├── public_bm25_v3_exact.jsonl
│   ├── public_knn_word_v3_exact.jsonl
│   ├── public_knn_char_v3_exact.jsonl
│   ├── public_v3_source_report.json
│   ├── public_raw_k500.jsonl + manifest
│   ├── public_bge_k200_compatible.jsonl + manifest
│   ├── public_adaptive_k500.jsonl + report
│   ├── public_candidate_union.jsonl + manifest
│   ├── public_shortlist_evidence.jsonl + report
│   ├── public_frozen_features.jsonl + report
│   └── public_v3a_predictions.json + public_v3a_policy_report.json
├── v3_residual/
│   ├── shortlist_evidence.jsonl + shortlist_report.json       # minh chứng train
│   ├── frozen_features.jsonl + frozen_features_report.json    # minh chứng train
│   └── policy/
│       ├── v3a_final_fit.joblib
│       ├── v3a_final_fit_manifest.json
│       └── policy_training_report.json
├── adaptive_k500_v1/                         # mô hình fit cuối / báo cáo adaptive
├── baseline_093_oof/                         # baseline, OOF và bộ nhớ đệm nguồn
├── public_anchor_093/                        # baseline tái lập và public IDs
└── models/inference_matched_reranker_v1_fold0/ # checkpoint lịch sử / tái lập kết quả
```

Các thư mục gốc khác của Task 1 (`candidates`, `cv_strict`, `evaluation`, `training`, `research`, `archive`) là các minh chứng hoặc lịch sử, không mặc nhiên là production. Bản báo cáo (report)/manifest là cơ sở có thẩm quyền quyết định về trạng thái, phân chia tập dữ liệu (split), điểm số và nguồn gốc (provenance). Các tệp JSONL dung lượng lớn có thể là các bản cache có thể tái tạo lại được; không bao giờ xóa chúng chỉ dựa vào tên tệp.

### Các tệp Task 1 được bảo vệ (Protected files)

Trước khi thực hiện dọn dẹp hoặc di chuyển đường dẫn, hãy xác minh các tệp sau vẫn tồn tại:

- `artifacts/task1/submission.zip`
- `recovery_096/final_public_v3/public_v3a_predictions.json`
- `recovery_096/final_public_v3/public_v3a_policy_report.json`
- `recovery_096/final_public_v3/public_frozen_features.jsonl` và báo cáo tương ứng
- `recovery_096/final_public_v3/public_shortlist_report.json`
- `recovery_096/final_public_v3/public_candidate_union_manifest.json`
- `recovery_096/final_public_v3/public_adaptive_k500_report.json`
- `recovery_096/v3_residual/policy/v3a_final_fit.joblib` và manifest
- `recovery_096/public_anchor_093/reproduced_093_submission.zip`

## 3. `models/` — Tài nguyên mô hình cục bộ

```text
models/
├── dek21-v2/
├── qwen3-legal/
└── reranker/
```

Mỗi thư mục mô hình thông thường chứa các tệp cấu hình/tokenizer của Hugging Face và một hoặc nhiều tệp `model.safetensors`. Các thư mục `.cache/` là trạng thái tải xuống/bộ nhớ đệm cục bộ và bản thân chúng không phải là nguồn gốc mô hình.

- `models/dek21-v2/`: mô hình dense embedding được sử dụng bởi luồng truy xuất dạng DEK21/FAISS. Kiểm tra `config.json`, siêu dữ liệu sentence-transformers, các tệp tokenizer và `model.safetensors`.
- `models/qwen3-legal/`: tài nguyên mô hình ngôn ngữ pháp lý Qwen, bao gồm cấu hình sinh (generation config), mẫu chat (chat template), tokenizer và trọng số mô hình.
- `models/reranker/`: mô hình reranker cục bộ chuẩn được sử dụng bởi bộ chấm điểm đặc trưng đóng băng (frozen-feature scorer) của V3. Hash của mô hình và ngữ nghĩa suy luận được ghi lại trong báo cáo frozen-feature; không tùy tiện thay thế hoặc chỉnh sửa mô hình này.

Thư mục `artifacts/task1/models/` riêng biệt là nơi lưu trữ minh chứng/checkpoints của Task 1 và khác biệt với các thư mục mô hình runtime cục bộ này. Một script khởi chạy có thể sử dụng `models/reranker/` trong khi một báo cáo lịch sử lại tham chiếu đến checkpoint nằm trong `artifacts/task1/recovery_096/models/`.

## 4. Luồng dữ liệu xuyên suốt các thư mục (Cross-directory data flow)

```text
data/raw + data/processed_v3
        │
        ├── chỉ mục truy xuất: data/vector_store/
        │                         │
        │                         └── Artifacts ứng viên Task 1
        │                              artifacts/task1/recovery_096/final_public_v3/
        │
        └── đầu vào mô hình: models/dek21-v2 hoặc models/reranker
                                      │
                                      └── đặc trưng đóng băng (frozen features) → Chính sách V3A
                                          → public_v3a_predictions.json
                                          → artifacts/task1/submission.zip
```

## 5. Quy tắc dành cho AI Agent

1. Coi `data/` là nơi lưu trữ đầu vào/chỉ mục, không phải là nơi ghi kết quả đầu ra của mô hình.
2. Coi `artifacts/task1/recovery_096/final_public_v3/` và các đường dẫn chính sách/mô hình được bảo vệ ở trên là production trừ khi manifest chỉ định khác.
3. Không sử dụng `artifacts/task2/**` cho các quyết định của Task 1.
4. Không đọc nhãn/đáp án public (public labels/gold) chỉ để chạy suy luận tập public; các báo cáo tập public ghi nhận rõ ràng việc chạy không cần nhãn (label-free).
5. Trước khi di chuyển/xóa một đường dẫn, hãy tìm kiếm các tham chiếu trong `scripts/`, `src/`, `configs/`, `docs/`, `Makefile`, và `pyproject.toml`.
6. Ưu tiên sử dụng mã băm (hash) trong manifest/report hơn là dấu thời gian (timestamps) hoặc tên tệp khi quyết định xem hai kết quả đầu ra có bị trùng lặp hay không.
7. Tuyệt đối không commit token API, thông tin xác thực, nội dung `.cache/`, hoặc trọng số mô hình trừ khi được yêu cầu rõ ràng.

## 6. Các lệnh định hướng nhanh

```bash
# Chỉ hiển thị cấu trúc thư mục (tránh xuất nội dung tệp lớn)
find data -maxdepth 2 -type d | sort
find artifacts/task1 -maxdepth 3 -type d | sort
find models -maxdepth 2 -type d | sort

# Tìm kiếm các nơi sử dụng trước khi thay đổi đường dẫn
rg -n "artifacts/task1|data/vector_store|models/reranker|models/dek21-v2|models/qwen3-legal" scripts src configs docs Makefile pyproject.toml
```
