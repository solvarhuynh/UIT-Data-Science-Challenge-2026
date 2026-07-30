# Hướng dẫn Setup & Vận hành TV2 — Retrieval Pipeline

Tài liệu này là hướng dẫn vận hành cho thành viên TV2 và các thành viên tích hợp
retrieval. Chạy các prompt theo đúng thứ tự, kiểm tra sau mỗi bước và tạo một commit
rollback trước khi chuyển sang bước tiếp theo.

## 1. Chuẩn bị môi trường

Từ thư mục gốc repository, dùng Python 3.10-3.12 (chưa hỗ trợ Python 3.13+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -c requirements_runtime.txt
python -m pip install -c requirements_runtime.txt -e .
python -m pip install -c requirements_runtime.txt -r requirements_dev.txt
```

Không commit `.venv/`. Nếu chưa dùng editable install, đặt tạm package path:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Dependency cài theo nhu cầu:

```powershell
python -m pip install -c requirements_runtime.txt sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements_runtime.txt qdrant-client faiss-cpu
python -m pip install -c requirements_runtime.txt rank-bm25 pyvi
python -m pip install -c requirements_runtime.txt pyyaml
```

Kiểm tra nền tảng:

```powershell
python -c "import pydantic, yaml; print('base deps ok')"
pytest tests/ -v --tb=short
```

## 2. Lộ trình prompt và verify

| Tasks | Phạm vi | Verify sau khi hoàn tất |
|---|---|---|
| 0 | `LegalChunk` contract | `python -c "from udsc2026.contracts import LegalChunk, RetrievalHit; print('ok')"` |
| 1 | `EmbeddingClient` BKAI local | `python -c "from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient; print('ok')"` |
| 2 | Qdrant/FAISS adapter | `python -c "from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter; print('ok')"` |
| 3 | `DenseRetriever` | `python -c "from udsc2026.retrieval.dense.dense_retriever import DenseRetriever; print('ok')"` |
| 4 | `BM25Retriever` + tokenizer | `python -c "from udsc2026.retrieval.sparse.tokenizer import tokenize_vi; print(tokenize_vi('Điều 10 Bộ luật Lao động'))"` |
| 5 | `HybridRetriever` + fusion | `python -c "from udsc2026.retrieval.hybrid import fuse_scores; print(fuse_scores([], []))"` |
| 6 | Indexing script | Chạy script index với JSONL TV4 và kiểm tra `data/vector_store/`. |
| 7 | Hợp nhất config | Kiểm tra đủ `embedding`, `vector_db`, `sparse`, `hybrid`. |
| 8 | Unit tests | `python -m pytest tests/unit/test_retrieval -v` phải pass 100%. |
| 9 | Đối chiếu DoD | Sửa mọi mục FAIL trước khi bàn giao. |

Sau mỗi verify pass, commit đúng nhóm prompt. Không chạy prompt sau nếu verify bước hiện
tại còn fail.

## 3. Cấu trúc và trách nhiệm

- `src/udsc2026/contracts/`: `LegalChunk` từ TV4 và `RetrievalHit` dùng chung.
- `src/udsc2026/infrastructure/embedding/`: load BKAI bi-encoder local và encode query/chunks.
- `src/udsc2026/infrastructure/vector_db/`: interface backend-neutral, Qdrant và FAISS.
- `src/udsc2026/retrieval/dense/`: query embedding và dense search.
- `src/udsc2026/retrieval/sparse/`: tokenizer tiếng Việt và BM25 độc lập.
- `src/udsc2026/retrieval/hybrid/`: normalize, merge và weighted score fusion; không rerank.
- `configs/base.yaml`: model path, vector DB và hybrid settings.
- `data/vector_store/`: FAISS/BM25 index local; không commit dữ liệu index lớn.

TV2 cung cấp các API `search(query, top_k, filters=None)` cho dense, sparse và hybrid.
TV1 có thể gọi hybrid để lấy context; TV5 nhận `list[RetrievalHit]` để rerank.

## Tóm tắt vị trí file

# Cấu trúc các file tv2 đã làm việc

udsc2026/
├── pyproject.toml              # ĐÃ SỬA: thêm [build-system] + [project]
├── requirements_dev.txt        # SẼ ĐƯỢC AGENT BỔ SUNG dần theo bảng trên (Prompt 1/2/4/7)
├── .venv/                      # môi trường ảo, KHÔNG commit vào git
├── configs/
│   ├── base.yaml                # SẼ CÓ THÊM section embedding/vector_db/sparse/hybrid
│   └── development.yaml
├── data/
│   ├── processed/chunks/        # sample_dev.jsonl do Prompt 6 tạo (dev only, xoá khi TV4 có data thật)
│   └── vector_store/            # output của Prompt 6: faiss/, bm25/ (qdrant chạy ngoài Docker)
├── scripts/
│   └── index_chunks.py          # Prompt 6
├── src/udsc2026/
│   ├── contracts/                 # Prompt 0: chunk.py (LegalChunk) + retrieval.py (RetrievalHit, có sẵn)
│   ├── infrastructure/
│   │   ├── config.py              # Prompt 7
│   │   ├── embedding/             # Prompt 1
│   │   └── vector_db/             # Prompt 2
│   └── retrieval/
│       ├── dense/                 # Prompt 3
│       ├── sparse/                # Prompt 4
│       └── hybrid/                # Prompt 5
├── tests/unit/test_retrieval/     # Prompt 8
└── docs
    └── tv2_setup.md               # chính là file bạn đang đọc

## 4. Quy tắc tích hợp

`LegalChunk` phải khớp JSONL output của TV4. Nếu TV4 đổi schema, cập nhật contract sau khi
thống nhất với TV4. Không định nghĩa lại `RetrievalHit` trong retriever.

`final_score` của TV2 là điểm hybrid dùng để xếp candidate; TV5 có thể bổ sung
`rerank_score` theo pipeline của mình. Không đưa cross-encoder, QA generation, API route hoặc
frontend vào TV2.

## 5. Chạy thử sau khi có index

```powershell
python scripts/index_chunks.py --vector-db-type faiss --chunks-dir data/processed/chunks
python -c "from udsc2026.retrieval.hybrid import search; print(search('Điều 10 Bộ luật Lao động quy định gì?', 5))"
python -m pytest tests/unit/test_retrieval -v
```

## 6. Git workflow

Không commit thẳng vào `main`. Với nhánh TV2 dùng nhánh `tv2` theo yêu cầu điều phối; các
commit nên nhỏ, rõ và theo nhóm:

```powershell
git switch tv2
git add <nhóm-file>
git commit -m "feat(tv2): <mô tả nhóm>"
git push origin tv2
```

Trước PR, kiểm tra `git status`, test liên quan, không có `.env`, `.venv`, dữ liệu raw hoặc
file index lớn. Tham khảo đầy đủ tại `docs/project/04_git_workflow.md`.
