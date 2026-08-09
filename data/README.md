# Dữ liệu UDSC 2026

`data/processed_v3` là **corpus processed chính thức duy nhất** của repository.
Mã nguồn, cấu hình, Docker và pipeline GPU đều được khóa về đường dẫn này để
không index nhầm một bản dữ liệu cũ.

## Cấu trúc chuẩn

```text
data/
├── raw/
│   └── btc/                    Dữ liệu nguồn được BTC bàn giao
├── processed_v3/
│   ├── documents/              Văn bản đã chuẩn hóa, một file mỗi document
│   ├── chunks/                 Child chunks dùng cho retrieval/indexing
│   ├── parents/                Parent context, tra cứu bằng parent_id
│   ├── benchmarks/
│   │   └── synthetic_qa.jsonl  Benchmark nội bộ có provenance
│   └── metadata/
│       ├── processing_manifest.json
│       ├── disk_audit_report.json
│       └── validation_report.json
├── vector_store/               FAISS/Qdrant/BM25 local, có thể tái tạo
├── task1/                      Input/fixture LegalIR khi được cung cấp
└── task2/                      Input/fixture LegalQA khi được cung cấp
```

Không dùng `data/processed`, `processed_v2`, `processed_v3_failed_*` hoặc một
cây `data/data/processed` được giải nén từ gói cũ. Archive như `data.zip` không
được runtime tự động đọc.

## Số liệu V3 đã audit

| Hạng mục | Giá trị |
| --- | ---: |
| Document files/records | 8.532 / 8.532 |
| Chunk files/records | 8.532 / 1.270.356 |
| Parent files/records | 8.532 / 184.548 |
| Synthetic benchmark nội bộ | 100 câu, 7 loại câu hỏi |
| Tổng dung lượng | 2.905.166.051 bytes (2,706 GiB) |
| Missing source token / intraline bigram | 0 / 0 |
| Invalid / duplicate / orphan / empty output | 0 |
| Oversized chunk / missing metadata | 0 / 0 |

```text
source_corpus_hash:             cda01fcb55da1e5656f1190eb35f07d77314ed72220d13aa6a1f421737d202ed
processed_corpus_tree_hash:     80fb33ff1133ce2583097cc5a98ddc9739240a60bfe11c979892d5412dcda647
benchmark_sha256:               80f27b47e81aa40b25ca55ab2fe7fb7edd8fc65388f41fdae841a48a104892d3
source_chunk_corpus_sha256:     d2aa542f1f45ad9f310bceb43063aed3058d2e81edbb15ecf18525c3c6d6bbc7
```

`integrity_gate_passed=true`. Semantic audit vẫn báo 20 context chính thức có
`passage` rỗng ngay trong nguồn BTC, gồm 6 gold context của LegalIR và 9 câu có
toàn bộ gold rỗng. V3 tạo placeholder có nhãn rõ ràng; không dùng chuỗi JSON/URL
làm nội dung giả và không tự suy diễn câu chữ pháp luật.

1.359 document được gắn cờ review cấu trúc nhưng vẫn có searchable output; chúng
không bị loại khỏi corpus. Benchmark synthetic chỉ dùng cho smoke/evaluation nội
bộ, không thay thế tập test chính thức của BTC.

## Chuyển dữ liệu sang máy GPU

Dữ liệu lớn bị loại khỏi Git bởi `.gitignore`. Copy nguyên thư mục
`data/processed_v3`, bao gồm đủ năm thư mục con sau:

```text
documents/
chunks/
parents/
benchmarks/
metadata/
```

Không chỉ copy `chunks/`: runtime LegalQA cần `parents/`, còn preflight cần
manifest, disk audit và benchmark để đối chiếu hash. Sau khi copy, chạy đúng
pipeline trong [TV5 GPU Runbook](../docs/members/tv5/tv5_gpu_runbook.md); script
sẽ tính lại tree hash trước khi index.

## Tạo lại corpus

Không ghi đè trực tiếp V3 đang dùng. Luôn sinh vào một thư mục mới, ví dụ:

```powershell
python scripts/data_prep/run_ingestion.py `
  --raw-directory data/raw/btc `
  --processed-root data/processed_candidate `
  --chunk-size 192 `
  --chunk-overlap 32 `
  --benchmark-count 100 `
  --benchmark-seed 2026
```

Chỉ thay corpus canonical sau khi disk audit, source coverage, benchmark hashes,
gold-document coverage và preflight đều đạt yêu cầu. Mọi index/candidate cũ phải
được build lại khi corpus hash hoặc chunk ID thay đổi.

## Dữ liệu Warm-up và submission

ID câu hỏi/document là chuỗi opaque: không ép sang số và không tự sắp xếp lại.
LegalIR hỗ trợ multi-gold; phải giữ toàn bộ danh sách gold khi đánh giá. Gold
answer chỉ dùng để audit/evaluation và không được đưa vào artifact nộp bài.

Submission của hai task đều là `submission.zip` chứa duy nhất
`submission.json`, nhưng schema khác nhau. Dùng writer và validator trong
`scripts/submission/` thay vì tự đóng ZIP thủ công.
