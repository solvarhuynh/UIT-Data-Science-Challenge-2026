# TV4 - Data & UI: ETL Pipeline, Legal Chunking, Citation Viewer

## 1. Tổng quan vai trò

TV4 phụ trách chất lượng dữ liệu đầu vào, tức nền móng của toàn bộ hệ thống RAG. Nếu chunk sai ranh giới Điều/Khoản hoặc mất metadata, TV2 khó index đúng, TV3 không thể citation chính xác và TV5 không đánh giá được. Trọng tâm là xây Data ETL Pipeline cho file text pháp luật thô từ ban tổ chức, dùng regex và rule-based parser để bóc tách cấu trúc `Luật -> Chương -> Điều -> Khoản -> Điểm`. Phần bổ sung là nâng cấp UI/UX React, đặc biệt render Markdown, Citation Viewer và responsive layout.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết reader trong `src/udsc2026/ingestion/readers/` để đọc `.txt`, `.json`, `.jsonl` hoặc format BTC cung cấp.
- [ ] Viết cleaner trong `src/udsc2026/ingestion/cleaners/` để chuẩn hóa Unicode, khoảng trắng, xuống dòng, header/footer và noise văn bản phổ biến.
- [ ] Viết parser trong `src/udsc2026/ingestion/legal_structure/` nhận diện tên luật, chương, mục, điều, khoản, điểm bằng regex có test case.
- [ ] Viết chunker trong `src/udsc2026/ingestion/chunking/`, ưu tiên không cắt ngang điều/khoản; nếu đoạn quá dài thì chia theo câu và giữ metadata cha.
- [ ] Xuất `data/processed/chunks/*.jsonl` để TV2 index, mỗi dòng là một chunk hoàn chỉnh.
- [ ] Tạo report validation trong `data/processed/metadata/`, gồm số document, số chunk, chunk rỗng, chunk quá dài và metadata thiếu.
- [ ] Nâng cấp frontend: component render Markdown answer, component `CitationViewer` hiển thị nguồn luật, điều/khoản và đoạn trích.
- [ ] Làm responsive cho màn hình laptop và mobile, tránh UI bị vỡ khi citation dài.

## 3. API Contract & Dữ liệu giao tiếp

ETL output cho TV2/TV3/TV5:

```json
{
  "chunk_id": "doc001_article_10_clause_1",
  "doc_id": "doc001",
  "text": "Nội dung điều khoản...",
  "metadata": {
    "law_name": "Bộ luật Lao động 2019",
    "chapter": "Chương II",
    "article": "Điều 10",
    "clause": "Khoản 1",
    "point": null,
    "source": "data/raw/btc/..."
  }
}
```

UI Citation Viewer nhận:

```text
citations: list[Citation]
```

Trong đó `Citation` cần có `chunk_id`, `law_name`, `article`, `clause`, `quote`.

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] ETL chạy được từ `data/raw/` sang `data/processed/chunks/`.
- [ ] Regex parser có test cho ít nhất Luật, Chương, Điều, Khoản, Điểm.
- [ ] Chunk không rỗng, không mất dấu tiếng Việt và có metadata citation.
- [ ] TV2 có thể index output JSONL mà không cần sửa tay.
- [ ] Citation Viewer render đúng đoạn trích và không làm vỡ layout.
- [ ] Có validation report để phát hiện dữ liệu lỗi trước khi build index.
