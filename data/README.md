# Dữ liệu cuộc thi UIT-DSC 2026

Thư mục này tách dữ liệu theo tác vụ để code không phải suy luận
từ tên file tạm trong `docs/`.

```text
data/
├── task1/
│   └── warmup.json   # LegalIR: id -> {question, answer: [document_id, ...]}
├── task2/
│   └── warmup.json   # LegalQA: id -> {question, answer: string}
├── raw/                  # Kho context BTC khi được bàn giao (không commit)
├── processed/            # Output ETL/chunking
└── vector_store/         # Index cục bộ (không commit)
```

Tài liệu hướng dẫn `.docx` vẫn ở `docs/`; chỉ dữ liệu máy đọc được
đặt dưới `data/`.

## Tính toàn vẹn Warm-up

| Tác vụ | Số câu | SHA-256 |
| --- | ---: | --- |
| Task 1 – LegalIR | 500 | `fadfbcab2923c085980239396d5564c16c232714f42004a36ed2a3d93962c0f9` |
| Task 2 – LegalQA | 500 | `b824e4f18bd9181c021498a28e402b7374d6d559f2ff28caa4120a9d932f82c5` |

ID là chuỗi opaque. Không ép sang số, không sắp xếp lại theo giá trị số và
không ghi đè dữ liệu gốc sau khi normalize. Reference answer chỉ dùng cho
evaluation/audit; artifact oracle phải mang nhãn `DO_NOT_SUBMIT`.

Task 1 hỗ trợ multi-gold: toàn bộ danh sách `answer` là tập gold hợp lệ, gồm cả
37 câu Warm-up có từ 2 đến 4 document. Metric chính là macro Recall và metric
phụ/tiebreak là macro Precision; không lấy riêng `answer[0]`, không ép prediction
đủ ba document và không nối toàn bộ corpus theo mặc định.

Hai file Warm-up đều là root mapping có cả `question` và gold `answer`; chúng
không được nộp trực tiếp. Theo contract Codabench đã kiểm tra ngày 02/08/2026,
wire submission cũng là root object keyed by question ID nhưng mỗi value chỉ
có `answer`: Task 1 là danh sách document IDs không trùng, vẫn giữ thứ tự giảm
dần relevance; Task 2 là answer string.
Writer trong `scripts/` chịu trách nhiệm chuyển prediction nội bộ sang đúng
shape và loại field `question`/trace trước khi đóng ZIP.
