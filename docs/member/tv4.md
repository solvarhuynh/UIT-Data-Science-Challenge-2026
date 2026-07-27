# TV4 - Data ETL, Legal Structure Parsing, UI/UX Advanced

## Mục tiêu & Phạm vi công việc

- [ ] Xây Data ETL Pipeline từ dữ liệu thô của ban tổ chức sang dữ liệu chuẩn.
- [ ] Bóc tách cấu trúc pháp luật: văn bản, chương, mục, điều, khoản, điểm.
- [ ] Làm sạch encoding, khoảng trắng, header/footer và noise trong văn bản.
- [ ] Sinh chunk có metadata đầy đủ để retrieval và citation dùng được.
- [ ] Phát triển UI/UX nâng cao: hiển thị citation, markdown rendering, trạng thái streaming.

## Thư mục mã nguồn phụ trách

- [ ] `src/dsc2026_legal/ingestion/readers/`
- [ ] `src/dsc2026_legal/ingestion/cleaners/`
- [ ] `src/dsc2026_legal/ingestion/legal_structure/`
- [ ] `src/dsc2026_legal/ingestion/chunking/`
- [ ] `frontend/`
- [ ] `data/raw/`
- [ ] `data/processed/`

## API/Interface đầu ra cần bàn giao

- [ ] `DocumentReader.read(path: str) -> RawLegalDocument`.
- [ ] `LegalTextCleaner.clean(text: str) -> str`.
- [ ] `LegalStructureParser.parse(document: RawLegalDocument) -> StructuredLegalDocument`.
- [ ] `Chunker.chunk(document: StructuredLegalDocument) -> list[LegalChunk]`.
- [ ] UI component `CitationList` hiển thị nguồn, điều, khoản và đoạn trích liên quan.

## Checklist nghiệm thu công việc

- [ ] Pipeline đọc được dữ liệu mẫu trong `data/raw/`.
- [ ] Chunk có `chunk_id`, `doc_id`, `law_name`, `article`, `clause`, `text`.
- [ ] Không mất dấu tiếng Việt sau bước làm sạch.
- [ ] File processed có thể dùng trực tiếp cho TV2 index và TV3 citation.
- [ ] UI render được markdown answer và danh sách citation rõ ràng.
