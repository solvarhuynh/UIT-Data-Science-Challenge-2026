# TV2 Progress Report

Tài liệu này tổng hợp các prompt TV2 đã được xử lý từ `prompts.md`, theo đúng trạng thái thực tế của repo sau khi hoàn thành từng bước.

## 1. Prompt 0 - Readiness check trước khi đụng vào data mới

Yêu cầu:
- Audit read-only toàn bộ corpus mock hiện có.
- Đọc config embedding, vector DB, BM25, hybrid.
- Đối chiếu contract `LegalChunk` và `RetrievalHit`.
- Chạy test `tests/retrieval/` và báo cáo hiện trạng.

Mình đã thực hiện:
- Đọc dữ liệu mock trong `data/processed/chunks/`.
- Kiểm tra `data/raw/btc/` và xác nhận chưa có data BTC thật để ingest.
- Chạy test retrieval với môi trường tạm ổn định hơn để tránh lỗi temp của Windows.
- Ghi nhận rõ contract hiện tại đủ để chạy trên mock, nhưng chưa đủ điều kiện để nhận JSONL thật từ TV4 nếu chưa có mapping context chuẩn.

Tác dụng sau khi làm xong:
- Có ảnh chụp trạng thái ban đầu của hệ thống retrieval.
- Xác định đúng điểm nghẽn trước khi tối ưu: thiếu data thật, thiếu manifest/versioning, thiếu bài benchmark theo corpus BTC.

Ảnh hưởng tới gì:
- Không sửa code ở bước này.
- Không làm thay đổi hành vi runtime.
- Chỉ tạo baseline để các prompt sau có mốc so sánh.

## 2. Prompt 1 - Nâng cấp `scripts/index_chunks.py` với versioning theo corpus/model hash

Yêu cầu:
- Tính `corpus_hash` và `model_hash`.
- Ghi manifest cạnh index.
- Hỗ trợ `--force`.
- Không rebuild nếu index đã up to date.
- Có test versioning.

Mình đã thực hiện:
- Thêm logic hash corpus và hash model/config.
- Thêm manifest JSON cho FAISS, Qdrant và BM25.
- Thêm cơ chế bỏ qua rebuild khi corpus/model không đổi.
- Thêm `--force` để ép build lại khi cần.
- Tạo test riêng cho versioning index.
- Cập nhật trạng thái TV2 trong `PROJECT_STATUS.md`.

Tác dụng sau khi làm xong:
- Index không còn là file “mù” nữa, mà có metadata để truy vết.
- Build lại index tiết kiệm thời gian hơn khi corpus không đổi.
- Dễ rollback và dễ debug khi đổi corpus hoặc model.

Ảnh hưởng tới gì:
- Không đổi signature `search()`.
- Không làm thay đổi contract dữ liệu.
- Tăng thêm file manifest đi kèm index, nên pipeline đã có thêm một lớp kiểm tra trạng thái.

File runtime chính:

- `scripts/data_prep/index_chunks.py`

File test:

- `tests/unit/test_retrieval/test_index_versioning.py`
- `tests/unit/test_retrieval/test_build_bm25_cli.py`
- `tests/retrieval/test_vector_db_adapters.py`

Chức năng:

- Đọc từng `LegalChunk` từ thư mục JSONL và tính `corpus_hash` ổn định từ
  `chunk_id` cùng nội dung `text` đã sắp xếp.
- Tạo `model_hash` từ nhận diện/cấu hình embedding để phân biệt index tạo bởi
  các model hoặc thiết lập khác nhau.
- Ghi manifest cạnh FAISS/Qdrant và BM25, gồm corpus/model hash, số chunk,
  collection, đường dẫn BM25 và git commit nếu có.
- So sánh manifest hiện có trước khi build; nếu corpus/model không đổi thì
  thoát thành công mà không embed/index lại. Cờ `--force` ép rebuild.

## 3. Prompt 3 - Benchmark và tăng độ bền của `EmbeddingClient`

Yêu cầu:
- Đo khả năng encode batch trên corpus lớn.
- Xử lý lỗi batch rõ ràng thay vì làm sập toàn bộ pipeline.
- Giữ nguyên shape/output contract.
- Có test cho lỗi batch.

Mình đã thực hiện:
- Bổ sung phương thức resilient cho embedding batch.
- Thêm script benchmark embedding để đo thời gian và lỗi batch.
- Thêm test mô phỏng batch lỗi có kiểm soát.

Tác dụng sau khi làm xong:
- Retrieval pipeline chịu lỗi tốt hơn khi gặp text quá dài hoặc batch lỗi cục bộ.
- Có công cụ đo tốc độ thật trước khi đẩy corpus BTC lớn vào hệ thống.

Ảnh hưởng tới gì:
- Encode vẫn trả đúng output shape như cũ.
- Chỉ thêm một nhánh xử lý an toàn hơn, không phá luồng gọi hiện tại.
- Giảm rủi ro fail toàn bộ index khi chỉ một vài chunk lỗi.

File runtime chính:

- `src/udsc2026/infrastructure/embedding/bkai_client.py`
- `scripts/data_prep/benchmark_embedding.py`

File test:

- `tests/retrieval/test_embedding_client.py`

Chức năng:

- Mã hóa query và document theo batch cấu hình được, vẫn giữ nguyên số chiều
  và dạng dữ liệu embedding mà các retriever hiện tại sử dụng.
- Đo thời gian toàn bộ, thời gian trung bình mỗi batch và throughput để đánh
  giá khả năng chạy trên corpus BTC lớn.
- Khi batch lỗi, log rõ chunk/batch lỗi và tiếp tục phần dữ liệu còn lại theo
  luồng resilient, thay vì dừng toàn bộ quá trình indexing.

## 4. Prompt 4 - Regression test BM25 tiếng Việt với thuật ngữ pháp lý thật

Yêu cầu:
- Bảo vệ tokenizer tiếng Việt và các cụm pháp lý quan trọng.
- Thêm test cho điều, khoản, điểm, số hiệu văn bản, viết tắt pháp lý.
- Không sửa logic nếu hành vi hiện tại đúng.

Mình đã thực hiện:
- Bổ sung test BM25 tokenizer cho các case pháp lý Việt Nam.
- Kiểm tra tokenization không làm mất số điều/khoản/điểm và số hiệu văn bản.
- Giữ nguyên logic tokenizer nếu case đã đúng ý nghĩa yêu cầu.

Tác dụng sau khi làm xong:
- Giảm nguy cơ BM25 khớp sai trên văn bản pháp lý thật.
- Bắt lỗi regression sớm khi thay đổi tokenizer hoặc regex xử lý pháp lý.

Ảnh hưởng tới gì:
- Không đổi hành vi runtime nếu case cũ đã đúng.
- Chủ yếu là tăng độ an toàn bằng test regression.

File runtime chính:

- `src/udsc2026/retrieval/sparse/tokenizer.py`
- `src/udsc2026/retrieval/sparse/bm25_retriever.py`

File test:

- `tests/retrieval/test_bm25_tokenizer.py`
- `tests/retrieval/test_bm25_retriever.py`
- `tests/unit/test_retrieval/test_bm25_persistence.py`

Chức năng:

- Chuẩn hóa và tách token tiếng Việt nhưng vẫn bảo toàn dấu, số điều/khoản/
  điểm, số hiệu văn bản và các thuật ngữ pháp lý cần thiết cho truy hồi.
- Xây dựng BM25 từ LegalChunk, truy vấn theo điểm sparse, rồi lưu/nạp index
  mà không làm thay đổi kết quả truy hồi hoặc metadata chunk.
- Regression tests khóa các trường hợp pháp lý thực tế để thay đổi regex hay
  tokenizer sau này không âm thầm làm giảm chất lượng khớp.

## 5. Prompt 6 - Guard top-k và giữ citation metadata

Yêu cầu:
- Chặn trả dư hơn `top_k`.
- Raise lỗi nếu search trả nhiều hơn giới hạn.
- Bảo toàn metadata citation từ chunk sang `RetrievalHit`.
- Có test cho overflow và preservation metadata.

Mình đã thực hiện:
- Thêm validation ở tầng search để phát hiện kết quả dư.
- Bổ sung test cho trường hợp backend trả quá số lượng cho phép.
- Bổ sung test kiểm tra metadata citation được giữ nguyên.

Tác dụng sau khi làm xong:
- Hệ thống fail sớm nếu có lỗi logic ở retrieval.
- TV5 và TV1 nhận đầu ra sạch hơn, không phải tự đoán hoặc tự chữa metadata bị mất.

Ảnh hưởng tới gì:
- Đây là thay đổi hành vi có chủ đích: lỗi vượt `top_k` không còn bị nuốt âm thầm.
- Giảm rủi ro sai format khi đi tới bước submission.

File runtime chính:

- `src/udsc2026/retrieval/dense/dense_retriever.py`
- `src/udsc2026/retrieval/sparse/bm25_retriever.py`
- `src/udsc2026/retrieval/hybrid/hybrid_retriever.py`
- `src/udsc2026/retrieval/hybrid/score_fusion.py`

File test:

- `tests/retrieval/test_dense_retriever.py`
- `tests/retrieval/test_bm25_retriever.py`
- `tests/retrieval/test_hybrid_retriever.py`
- `tests/retrieval/test_score_fusion.py`
- `tests/unit/test_retrieval/test_hybrid_and_dense.py`

Chức năng:

- Dense, sparse và hybrid trả tối đa đúng `top_k` `RetrievalHit`; số lượng vượt
  giới hạn được xem là lỗi logic để phát hiện sớm trước tầng TV1/TV5.
- Hybrid hợp nhất kết quả dense và BM25 theo score fusion/candidate policy,
  đồng thời duy trì thứ hạng và điểm thành phần phục vụ debug.
- Khi chuyển chunk thành `RetrievalHit`, các field citation như `chunk_id`,
  `doc_id`, `text`, `source`, `law_name`, `article`, `clause` và `metadata`
  phải giữ nguyên giá trị gốc nếu chunk có cung cấp.
