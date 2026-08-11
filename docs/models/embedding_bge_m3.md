# BGE-M3 và vai trò trong Legal Retrieval

[HCMUTE Embedding v2](https://huggingface.co/huyydangg/DEk21_hcmute_embedding_v2)

> **Status: `DEFERRED_EXPERIMENT`.** BGE-M3 không thuộc Task1 production hiện
> tại, không được tải/build index và chưa được thêm vào preflight hay config.
> Chỉ xem xét nếu P12/P13/P15 vẫn plateau theo CandidateDocRecall hoặc
> official OOF Recall.

## Tổng quan

Tài liệu này ghi chú cách tích hợp `huyydangg/DEk21_hcmute_embedding_v2` cho dense retrieval tiếng Việt. Các nguyên tắc về dense, sparse và hybrid retrieval vẫn áp dụng khi mở rộng kiến trúc.

## Ba đặc điểm chính

### Multi-Lingual

BGE-M3 hỗ trợ nhiều ngôn ngữ trong cùng một vector space. Với tài liệu pháp luật Việt Nam, điểm quan trọng là khả năng biểu diễn tiếng Việt ổn định và vẫn giữ được quan hệ ngữ nghĩa khi query có cách diễn đạt khác với văn bản gốc.

### Multi-Functionality

BGE-M3 hỗ trợ ba kiểu truy hồi:

- Dense retrieval: vector ngữ nghĩa cho semantic search.
- Sparse retrieval: trọng số từ khóa dạng learnable sparse vector.
- Multi-vector retrieval: nhiều vector cho một đoạn để so khớp chi tiết hơn.

Trong dự án hiện tại, ta triển khai dense bằng HCMUTE Embedding v2 và sparse bằng BM25. Thiết kế `retrieval/` nên giữ interface đủ rộng để sau này thay BM25 bằng sparse embedding nếu cần.

### Multi-Granularity

Model có thể biểu diễn văn bản ở nhiều cấp: câu, đoạn, passage và tài liệu. Điều này quan trọng với legal corpus vì câu trả lời thường cần đúng cấp `Điều`, `Khoản`, `Điểm`, không chỉ đúng văn bản tổng thể.

## Vì sao phù hợp với legal document

- Văn bản luật có thuật ngữ chính xác, cần sparse hoặc keyword matching.
- Người dùng hỏi bằng ngôn ngữ đời thường, cần dense semantic matching.
- Citation cần metadata theo cấu trúc pháp luật, nên chunk phải giữ đơn vị pháp lý nhỏ.
- Các điều khoản dài cần mô hình hóa passage tốt thay vì chỉ sentence-level.

## Bài học áp dụng cho HCMUTE embedding v2

Khi dùng HCMUTE Embedding v2 làm dense retriever, cần giữ các nguyên tắc sau:

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
