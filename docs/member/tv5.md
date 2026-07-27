# TV5 - Reranking, Docker, Submission

## Mục tiêu & Phạm vi công việc

- [ ] Dựng module Reranking bằng Cross-Encoder để lọc lại top kết quả truy xuất.
- [ ] Chuẩn hóa interface nhận candidate từ hybrid retrieval và trả danh sách đã xếp lại.
- [ ] Tối ưu latency reranking theo batch size, top-n và device config.
- [ ] Đóng gói Docker cho backend, frontend và dịch vụ phụ trợ nếu cần.
- [ ] Thiết lập luồng xuất file submission cho CodaLab.

## Thư mục mã nguồn phụ trách

- [ ] `src/dsc2026_legal/retrieval/reranking/`
- [ ] `src/dsc2026_legal/infrastructure/reranker/`
- [ ] `src/dsc2026_legal/evaluation/`
- [ ] `scripts/`
- [ ] `docker/`

## API/Interface đầu ra cần bàn giao

- [ ] `Reranker.rerank(query: str, candidates: list[RetrievalHit], top_n: int) -> list[RetrievalHit]`.
- [ ] `SubmissionWriter.write(results: list[QAResponse], output_path: str) -> None`.
- [ ] Docker command/script để build và chạy backend.
- [ ] Docker command/script để build và chạy frontend.
- [ ] Batch evaluation script nhận input test set và sinh output submission.

## Checklist nghiệm thu công việc

- [ ] Reranking giữ nguyên metadata cần cho citation.
- [ ] Có benchmark latency cho top-20 và top-50 candidates.
- [ ] Docker build không copy model weights vào image.
- [ ] Submission file đúng schema CodaLab yêu cầu.
- [ ] Có script chạy end-to-end từ dữ liệu test đến file submission.
