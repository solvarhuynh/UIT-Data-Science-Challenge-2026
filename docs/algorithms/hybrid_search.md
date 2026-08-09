# Hybrid Search cho LegalIR

## Mục tiêu

Hybrid Search kết hợp truy hồi ngữ nghĩa bằng Dense Retrieval và truy hồi từ khóa bằng Sparse Retrieval để tăng độ chính xác cho văn bản pháp luật. Với dự án UDSC2026, Dense dùng `huyydangg/DEk21_hcmute_embedding_v2`, Sparse dùng BM25, kết quả cuối có thể đi qua Reranker trước khi chuyển sang QA.

## Thành phần

- Dense Retriever: mã hóa query và chunk thành vector, tìm các đoạn có nghĩa gần nhau.
- Sparse Retriever: dùng BM25 để bắt đúng thuật ngữ pháp lý, số điều, tên luật, cụm từ khóa.
- Hybrid Merger: hợp nhất, chuẩn hóa điểm, khử trùng lặp và xếp hạng.
- Reranker: chấm lại top kết quả bằng Cross-Encoder hoặc model rerank tương đương.

## BM25

BM25 ưu tiên tài liệu có từ khóa khớp query, đồng thời điều chỉnh theo độ dài tài liệu.

```text
score(q, d) = sum IDF(t) * (tf(t, d) * (k1 + 1)) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))
```

Khuyến nghị mặc định:

- `k1 = 1.2` đến `1.8`
- `b = 0.75`
- `top_k_sparse = 50`

## Dense Retrieval

Dense Retrieval dùng embedding để tìm đoạn có quan hệ ngữ nghĩa với câu hỏi, kể cả khi không trùng từ khóa. Với luật Việt Nam, dense giúp bắt các cách diễn đạt tương đương như "vượt đèn đỏ" và "không chấp hành tín hiệu giao thông".

Điểm thường dùng:

```text
dense_score = cosine_similarity(query_embedding, chunk_embedding)
```

## Chuẩn hóa điểm

BM25 và cosine similarity có thang điểm khác nhau, nên cần normalize trước khi trộn.

```text
normalized_score = (score - min_score) / (max_score - min_score + epsilon)
```

Nếu một nhánh không có kết quả, điểm của nhánh đó bằng `0`.

## Công thức Hybrid

Công thức baseline:

```text
hybrid_score = alpha * dense_norm + beta * sparse_norm + overlap_bonus
```

Trong đó:

- `alpha`: trọng số dense, mặc định `0.55`
- `beta`: trọng số sparse, mặc định `0.45`
- `overlap_bonus`: cộng thêm `0.05` đến `0.10` nếu chunk xuất hiện ở cả dense và sparse

Ràng buộc:

```text
alpha + beta = 1.0
```

Ví dụ:

```text
dense_norm = 0.86
sparse_norm = 0.72
overlap_bonus = 0.05
hybrid_score = 0.55 * 0.86 + 0.45 * 0.72 + 0.05 = 0.847
```

## Quy trình xử lý

1. Nhận query từ FastAPI.
2. Chạy Dense Retrieval bằng HCMUTE embedding v2.
3. Chạy BM25 Retrieval trên cùng tập chunk đã chuẩn hóa.
4. Gộp kết quả theo `chunk_id`.
5. Normalize `dense_score` và `sparse_score`.
6. Tính `hybrid_score`.
7. Sort giảm dần theo `hybrid_score`.
8. Chuyển top-n sang Reranker nếu được bật.

## Interface đề xuất

```python
from udsc2026.contracts.retrieval import RetrievalHit

class HybridRetriever:
    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]:
        ...
```

`RetrievalHit` phải được import từ `src/udsc2026/contracts/retrieval.py`; Hybrid Search không tự định nghĩa schema riêng. Contract chung cần giữ đủ metadata để QA sinh citation và để TV5 rerank:

```text
chunk_id, doc_id, law_name, article, clause, text, dense_score, sparse_score, hybrid_score
```

## Gợi ý tuning

- Query có số điều, tên nghị định, mã văn bản: tăng `beta`.
- Query diễn đạt đời thường, ít thuật ngữ chính xác: tăng `alpha`.
- LegalQA cần citation chắc: giữ `overlap_bonus` để ưu tiên kết quả được cả hai nhánh đồng thuận.

