# TV3 - Local LLM Qwen3, Prompt System, BM25 Hybrid

## Mục tiêu & Phạm vi công việc

- [ ] Tích hợp Local LLM `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`.
- [ ] Thiết kế prompt system ép trả lời dựa trên context, có trích dẫn điều luật.
- [ ] Xây cơ chế chống hallucination: từ chối khi không đủ căn cứ, ghi rõ nguồn.
- [ ] Cấu hình Sparse Retrieval bằng BM25.
- [ ] Viết thuật toán Hybrid Search kết hợp dense score, sparse score và rerank score.

## Thư mục mã nguồn phụ trách

- [ ] `src/dsc2026_legal/qa/`
- [ ] `src/dsc2026_legal/infrastructure/llm/`
- [ ] `prompts/`
- [ ] `src/dsc2026_legal/retrieval/sparse/`
- [ ] `src/dsc2026_legal/retrieval/hybrid/`

## API/Interface đầu ra cần bàn giao

- [ ] `LLMClient.generate(prompt: str, stream: bool = False) -> str | Iterator[str]`.
- [ ] `PromptBuilder.build(question: str, contexts: list[RetrievalHit]) -> str`.
- [ ] `CitationExtractor.extract(answer: str, contexts: list[RetrievalHit]) -> list[Citation]`.
- [ ] `BM25Retriever.search(query: str, top_k: int) -> list[RetrievalHit]`.
- [ ] `HybridRetriever.search(query: str, top_k: int, weights: dict) -> list[RetrievalHit]`.

## Checklist nghiệm thu công việc

- [ ] Qwen3 được load từ `./models/qwen3-legal`.
- [ ] Prompt template có version và lưu trong `prompts/`.
- [ ] Câu trả lời có citation hoặc nêu rõ không tìm thấy căn cứ.
- [ ] BM25 chạy độc lập với dense retrieval.
- [ ] Hybrid search có cấu hình trọng số trong `configs/`.
