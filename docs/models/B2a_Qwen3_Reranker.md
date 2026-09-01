# Giải thích model B2a: Qwen3-Reranker-0.6B

## Mục đích

`Qwen/Qwen3-Reranker-0.6B` là model reranker dạng cross-encoder: model nhận
đồng thời câu hỏi và một tài liệu rồi trả về điểm liên quan. Trong Workflow B
B2a, model được dùng để xếp hạng trực tiếp 77 tài liệu ứng viên cho từng câu
hỏi.

## Cấu hình đã khóa

- Chế độ: `ZERO_SHOT` (không dùng nhãn huấn luyện, không fine-tune).
- Kích thước ứng viên: K77, retrieval và embedding giữ nguyên.
- Đầu ra: đúng top-5 tài liệu; không one-swap, threshold, NO_OP hay P1/P4/P5.
- Metric: set-based macro Recall (chính), macro Precision (phụ).
- Revision đã tải: `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
- Model card khai báo khoảng 0.6B tham số, hơn 100 ngôn ngữ và context 32K.

## Instruction duy nhất

> Given a Vietnamese legal question, determine whether the Document contains
> legal provisions relevant to answering the Query.

Không được thử nhiều instruction rồi chọn theo Recall.

## Trạng thái chạy

Đã hoàn tất kiểm tra provenance và hash snapshot cục bộ. Bước smoke kỹ thuật
GPU bị chặn vì runtime hiện không có CUDA/GPU; vì vậy chưa chạy context audit,
full inference, freeze top-5 hoặc đánh giá F1–F4. Không có nhãn Fold0/public và
không có metric khoa học nào được tính.

Khi có GPU phù hợp, cần chạy lại đúng revision, instruction và cấu hình đã khóa,
sau đó freeze toàn bộ dự đoán trước khi mở nhãn F1–F4.
