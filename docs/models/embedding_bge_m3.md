# BGE-M3 và vai trò trong Legal Retrieval

[Legal Retrieval Model](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder)

## Tổng quan

BGE-M3 là embedding model của BAAI với ba đặc điểm chính: đa ngôn ngữ, đa chức năng và đa mức độ chi tiết. Dù baseline hiện tại của UDSC2026 dùng `bkai-foundation-models/vietnamese-bi-encoder`, kiến thức về BGE-M3 vẫn hữu ích để thiết kế retrieval enterprise, đặc biệt khi cần nâng cấp sang model hỗ trợ dense, sparse và multi-vector trong cùng một kiến trúc.

## Ba đặc điểm chính

### Multi-Lingual

BGE-M3 hỗ trợ nhiều ngôn ngữ trong cùng một vector space. Với tài liệu pháp luật Việt Nam, điểm quan trọng là khả năng biểu diễn tiếng Việt ổn định và vẫn giữ được quan hệ ngữ nghĩa khi query có cách diễn đạt khác với văn bản gốc.

### Multi-Functionality

BGE-M3 hỗ trợ ba kiểu truy hồi:

- Dense retrieval: vector ngữ nghĩa cho semantic search.
- Sparse retrieval: trọng số từ khóa dạng learnable sparse vector.
- Multi-vector retrieval: nhiều vector cho một đoạn để so khớp chi tiết hơn.

Trong dự án hiện tại, ta triển khai dense bằng BKAI và sparse bằng BM25. Thiết kế `retrieval/` nên giữ interface đủ rộng để sau này thay BM25 bằng sparse embedding nếu cần.

### Multi-Granularity

Model có thể biểu diễn văn bản ở nhiều cấp: câu, đoạn, passage và tài liệu. Điều này quan trọng với legal corpus vì câu trả lời thường cần đúng cấp `Điều`, `Khoản`, `Điểm`, không chỉ đúng văn bản tổng thể.

## Vì sao phù hợp với legal document

- Văn bản luật có thuật ngữ chính xác, cần sparse hoặc keyword matching.
- Người dùng hỏi bằng ngôn ngữ đời thường, cần dense semantic matching.
- Citation cần metadata theo cấu trúc pháp luật, nên chunk phải giữ đơn vị pháp lý nhỏ.
- Các điều khoản dài cần mô hình hóa passage tốt thay vì chỉ sentence-level.

## Bài học áp dụng cho BKAI Bi-encoder

Khi dùng BKAI làm dense retriever, cần giữ các nguyên tắc sau:

- Chunk theo cấu trúc pháp luật, không cắt giữa điều/khoản nếu tránh được.
- Lưu metadata giàu ngữ cảnh: `law_name`, `chapter`, `article`, `clause`, `point`.
- Dùng Hybrid Search để bù điểm yếu của dense retrieval trong truy vấn có số điều và tên văn bản.
- Benchmark theo query pháp lý thực tế, không chỉ cosine score.

## Interface hạ tầng

```python
class EmbeddingClient:
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        ...
```

Embedding client không phụ thuộc trực tiếp vào FastAPI. Backend chỉ gọi qua service/retriever để giữ khả năng thay model.
