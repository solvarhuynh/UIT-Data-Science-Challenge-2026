# TV4 - Data ETL Pipeline & Full Web Frontend

## 1. Tổng quan vai trò

TV4 phụ trách hai mảng: Data ETL Pipeline và 100% Web Frontend React + TailwindCSS. Về data, TV4 đảm bảo văn bản pháp luật được parse đúng cấu trúc `Luật -> Chương -> Điều -> Khoản -> Điểm`, chunk không mất metadata và sẵn sàng cho TV2 index. Về frontend, TV4 sở hữu toàn bộ UI: search box, chat panel, kết nối API/SSE, render Markdown, citation viewer và trạng thái loading/error.

TV4 là người đảm bảo người dùng có giao diện để hỏi đáp pháp luật và nhìn thấy citation rõ ràng.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết reader trong `src/udsc2026/ingestion/readers/` để đọc `.txt`, `.json`, `.jsonl` hoặc format BTC cung cấp.
- [ ] Viết cleaner trong `src/udsc2026/ingestion/cleaners/` để chuẩn hóa Unicode, khoảng trắng, xuống dòng, header/footer và noise.
- [ ] Viết parser trong `src/udsc2026/ingestion/legal_structure/` để nhận diện tên luật, chương, mục, điều, khoản, điểm.
- [ ] Viết chunker trong `src/udsc2026/ingestion/chunking/`, ưu tiên không cắt ngang điều/khoản; nếu quá dài thì chia theo câu và giữ metadata cha.
- [ ] Xuất `data/processed/chunks/*.jsonl` cho TV2 index, mỗi dòng là một legal chunk hoàn chỉnh.
- [ ] Tạo validation report trong `data/processed/metadata/`: số document, số chunk, chunk rỗng, chunk quá dài, metadata thiếu.
- [ ] Xây frontend React + TailwindCSS trong `frontend/` gồm layout chính, `SearchBox`, `ChatPanel`, `MessageBubble`, `LoadingState`, `ErrorState`.
- [ ] Viết API client frontend để gọi `/query` và xử lý SSE stream từ backend TV1.
- [ ] Render Markdown answer an toàn, không làm vỡ layout với danh sách/điều khoản dài.
- [ ] Xây `CitationViewer` hiển thị `law_name`, `article`, `clause`, `quote`, `source` và liên kết citation với answer.
- [ ] Làm responsive cho laptop và mobile, đảm bảo citation dài, answer dài và lỗi API đều hiển thị rõ.

## 3. Quy chuẩn Code & API Contract

### Clean Code Standard

- Code ETL ngắn gọn, tách reader, cleaner, parser, chunker và validator.
- Code frontend tách component rõ: UI component, API client, state handling, render citation.
- Dùng Pydantic/Typing cho ETL output; với TypeScript frontend thì dùng type/interface cho request/response.
- Không copy-paste component hoặc parser regex trùng lặp; tách helper khi lặp lại có ý nghĩa.
- Không để dead code, mock UI cũ, console log dư thừa hoặc component không dùng.

### Data Contract

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

Frontend request:

```json
{"question": "Điều kiện ly hôn đơn phương là gì?", "top_k": 5, "stream": true}
```

Frontend response cần render:

```text
answer, citations, retrieval_hits, latency_ms, cache_hit, warnings
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] ETL chạy được từ `data/raw/` sang `data/processed/chunks/`.
- [ ] Parser có test cho Luật, Chương, Điều, Khoản, Điểm.
- [ ] Chunk không rỗng, không mất dấu tiếng Việt và có metadata citation.
- [ ] TV2 index được output JSONL mà không cần sửa tay.
- [ ] Frontend chạy được bằng `npm install` và `npm run dev`.
- [ ] UI gửi được câu hỏi đến backend, xử lý loading/error và render answer Markdown.
- [ ] Citation Viewer hiển thị đúng nguồn, điều/khoản, quote và không vỡ layout trên mobile.
