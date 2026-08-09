# Data Pipeline chi tiết: LegalIR và LegalQA

Tài liệu này mô tả contract kỹ thuật từ dữ liệu BTC đến corpus V3. Luồng phát
hành bắt buộc là:

```text
raw → processed_candidate → audit + benchmark → promote nguyên cây → processed_v3
```

ETL không ghi trực tiếp vào V3. Mọi thành phần indexing, retrieval, reranking và
QA chỉ đọc V3.

## 1. Extract

Nguồn mặc định là `data/raw/btc`. BTC adapter quét có thứ tự các context LegalIR
và LegalQA, chuẩn hóa mỗi nguồn thành `RawDocument`:

```text
doc_id, source_path, title, raw_text, file_format, metadata
```

Reader hỗ trợ JSON/JSONL/TXT/PDF/DOCX. Lỗi từng file và ID trùng được ghi vào
`metadata/extract_errors.json`; pipeline tiếp tục để báo cáo đầy đủ, nhưng audit
sẽ quyết định candidate có đủ điều kiện phát hành hay không.

`metadata/manifest.json` lưu danh sách context, quan hệ fixture LegalIR/LegalQA,
duplicate/orphan và hash corpus nguồn.

## 2. Clean

Cleaner chuẩn hóa Unicode NFC, khoảng trắng và lỗi OCR an toàn; loại page marker,
header/footer lặp và mục lục theo quy tắc có audit. Cleaner phải giữ dấu câu,
thứ tự dòng và các mốc Chương/Mục/Điều/Khoản/Điểm cần cho parser.

Mỗi kết quả được ghi thành `documents/<doc_id>.json`. Nội dung bị loại và lý do
được lưu để disk audit phát hiện thao tác làm mất nội dung có rủi ro.

## 3. Parse cấu trúc pháp luật

Parser state-machine nhận diện theo thứ tự:

```text
Văn bản → Chương → Mục → Điều → Khoản → Điểm
```

Regex được neo theo dòng và chỉ nhận Khoản/Điểm trong ngữ cảnh Điều phù hợp để
giảm false positive. Parser cũng ghi `source_family`, `structure_status` và
`structure_warnings` để các trường hợp không chắc chắn có thể review.

## 4. Parent–child chunking

Cấu hình chính thức:

```yaml
chunk_size: 192
chunk_overlap: 32
```

Nguyên tắc:

- Parent đại diện cho context Điều hoặc context fallback đủ rộng.
- Child ưu tiên đơn vị Khoản/Điểm; chỉ phần quá dài mới chia theo câu rồi token,
  vẫn nằm trong cùng đơn vị pháp lý.
- Mọi child có `chunk_id`, `doc_id`, `parent_id`, text và citation metadata.
- `parent_text` trong child luôn là `null`. Parent text chỉ xuất hiện trong
  `parents/<doc_id>.jsonl`, tránh lặp hàng triệu lần.
- Consumer retrieve/rerank child trước, sau đó tra `parent_id` để mở rộng context.

### Không làm mất văn bản khó parse

| Trạng thái | Cách xử lý |
| --- | --- |
| `structured` | Chunk theo cấu trúc pháp luật đã nhận diện |
| `partial` | Giữ chunk hợp lệ, bổ sung cảnh báo và manual review |
| `unstructured_fallback` | Chunk toàn văn theo đoạn/câu, gắn `fallback_chunking=true` |
| `empty_source` | Giữ ID nguồn rỗng bằng chunk có `structure_type=empty_source_placeholder` và nhãn rõ ràng |

Placeholder không được dùng làm bằng chứng pháp luật và khiến semantic/quality
gate cảnh báo. Fallback/partial vẫn được index nhưng trạng thái phải đi theo
metadata để audit và phân tích lỗi.

## 5. Load vào candidate mới

Output mặc định là `data/processed_candidate`. Cả API thư viện và CLI đều từ
chối một output root không rỗng; đây là bảo vệ chống trộn hoặc ghi đè corpus.

```text
data/processed_candidate/
├── documents/*.json
├── chunks/*.jsonl
├── parents/*.jsonl
├── benchmarks/synthetic_qa.jsonl
└── metadata/*.json
```

Lệnh chuẩn từ repository root:

```powershell
python scripts/data_prep/run_ingestion.py `
  --raw-directory data/raw/btc `
  --processed-root data/processed_candidate `
  --chunk-size 192 `
  --chunk-overlap 32 `
  --benchmark-count 100 `
  --benchmark-seed 2026 `
  --progress-every 100
```

Benchmark được sinh sau disk audit và chỉ khi integrity gate đạt. Candidate
nghiệm thu không dùng `--skip-benchmark` hay `--allow-missing-benchmark-types`.

## 6. Audit và provenance

Pipeline đọc lại output trên đĩa thay vì tin vào bộ đếm trong RAM. Các artifact
chính gồm:

| Artifact | Nội dung quyết định |
| --- | --- |
| `metadata/validation_report.json` | Schema, empty/oversized, metadata, orphan, duplicate, coverage và trạng thái cấu trúc |
| `metadata/disk_audit_report.json` | Counts thực tế, parent mapping, zero-loss token/bigram, task coverage, tree hash và các gate |
| `metadata/processing_manifest.json` | Git commit, nguồn, cấu hình 192/32, counts, tree hash và benchmark SHA-256 |
| `metadata/manual_review_*.json` | Phân loại regex/OCR/fallback/partial cần kiểm tra |
| `benchmarks/synthetic_qa.jsonl` | Q&A synthetic có gold chunk/citation và provenance corpus |

Điều kiện tối thiểu để phát hành:

- `integrity_gate_passed=true` trong audit và processing manifest;
- `integrity_failures=[]`;
- số file document/chunk/parent và số record khớp manifest;
- không invalid JSON, ID trùng, orphan, chunk rỗng/quá dài, thiếu metadata hoặc
  `parent_text` bị lặp trong child;
- benchmark hash/count/source-chunk hash khớp manifest;
- semantic issues, source rỗng, fallback và partial đã được review có chủ đích.

`semantic_completeness_gate_passed=false` có thể phản ánh passage BTC rỗng ngay
từ nguồn. Trường hợp đó không tự động biến thành lỗi integrity, nhưng phải được
ghi nhận trước khi phát hành; tuyệt đối không bịa nội dung để làm gate xanh.

## 7. Promote và sử dụng

Sau khi nghiệm thu, người chịu trách nhiệm release di chuyển nguyên cây candidate
sang `data/processed_v3`. V3 phải chưa tồn tại; không merge từng thư mục và không
sửa file sau promotion.

```powershell
if (Test-Path -LiteralPath data/processed_v3) {
  throw "data/processed_v3 đã tồn tại; dừng để bảo vệ corpus bất biến"
}
Move-Item -LiteralPath data/processed_candidate -Destination data/processed_v3
```

Quyền sử dụng sau phát hành:

| Consumer | Dữ liệu chỉ đọc |
| --- | --- |
| TV2 | `data/processed_v3/chunks` để index child |
| TV3 | `data/processed_v3/parents` để mở rộng context theo `parent_id` |
| TV5 | chunks, benchmark và manifests để rerank/evaluate |
| TV1 | corpus V3 và index đã version hóa để orchestration/API |

Index, candidate retrieval và kết quả benchmark phải được tạo lại nếu tree hash,
benchmark hash hoặc model revision thay đổi. Report của các lần chạy downstream
đặt dưới `artifacts/` hoặc `outputs/`, không ghi vào V3.

## Liên quan

- [Kiến trúc data pipeline](../architecture/data_pipeline.md)
- [TV4 setup](../members/tv4/tv4_setup.md)
- [Nhiệm vụ TV4](../members/tv4/tv4.md)
- [TV2 setup](../members/tv2/tv2_setup.md)
- [TV5 GPU runbook](../members/tv5/tv5_gpu_runbook.md)
- [Retrieval contract](../../src/udsc2026/contracts/retrieval.py)
