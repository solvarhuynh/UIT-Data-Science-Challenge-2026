# Experiment Roadmaps for Workflow B – Member (TV4)

## Overview

- Tài liệu này mô tả các thí nghiệm **điểm‑leverage** thuộc **TV4** (Member Researcher) nhằm cải thiện Recall / top‑5 cuối cùng.
- Các thí nghiệm không tự động chạy; chúng được kích hoạt theo các điều kiện trong bảng.

## Bảng Thí nghiệm

| ID | Thí nghiệm | Trạng thái | Khi nào chạy | Tại sao chạy | Cần GPU? | Khả năng tăng điểm | Điều kiện STOP | Bước tiếp theo |
|----|------------|------------|--------------|--------------|----------|-------------------|----------------|----------------|
| M1 | Hoàn thành Direct Learning‑to‑Rank | ACTIVE | Ngay khi hợp đồng B1 đóng băng có sẵn | Kiểm tra Direct LTR cải thiện top‑5 so với Workflow‑A? | Có (Modal GPU) | HIGH | Nếu không cải thiện >1% → STOP | Đánh giá OOF, top‑5, báo cáo Recall/Precision |
| M2 | Mở rộng tính năng cho Direct LTR | CONDITIONAL | Sau khi M1 có kết quả | Thêm tính năng (P4 probability, metadata, độ dài tài liệu…) có giúp cải thiện? | Có (GPU tùy chọn) | MEDIUM‑HIGH | Nếu không thấy cải thiện → STOP | Báo cáo phân tích tính năng, mô hình LTR cập nhật |
| M3 | LTR kết hợp Semantic | CONDITIONAL | Khi B2a‑0 (TV2) có điểm semantic và M1 hợp lệ | Kiểm tra việc kết hợp điểm Qwen semantic có tăng điểm không? | Có (GPU cho scoring) | HIGH | Nếu không có boost → STOP | Prototype fusion, báo cáo so sánh LTR vs Semantic vs Fusion |
| M4 | Cải thiện Hard‑Negative Ranking | CONDITIONAL | Sau M1 hoặc phân tích lỗi B2a | Hard‑negative có tăng khả năng phân biệt không? | Có (GPU cho training) | MEDIUM‑HIGH | Nếu không cải thiện → STOP | Curriculum hard‑negative, mô hình ranking, báo cáo |
| M5 | So sánh kiến trúc ranking thay thế | CONDITIONAL | Khi M1 baseline đã ổn định | Kiểm tra LambdaRank, pairwise, listwise có vượt M1? | Có (GPU tùy chọn) | MEDIUM | Nếu không vượt → STOP | Triển khai comparator, benchmark, khuyến nghị |
| M7 | Cải thiện Candidate Retrieval | DEFERRED / CONDITIONAL | Khi M6 cho thấy giới hạn K77 và được cấp phép | Kiểm tra hybrid sparse+dense, embedding mạnh hơn, late interaction | Có (GPU) | HIGH | Nếu không cải thiện → STOP | Prototype retrieval, đánh giá pool, artifact cập nhật |

---
## Quy tắc phân GPU

Nếu chỉ có **MỘT** nhánh GPU quan trọng → **TV2** sở hữu.
Nếu có **HAI** hoặc hơn nhánh GPU độc lập → **TV2** giữ nhánh có **expected score leverage** hoặc **architecture risk** cao nhất, **TV4** có thể nhận nhánh thứ hai.

---
## Chủ sở hữu Fine‑tuning

Qwen fine‑tuning / LoRA **thuộc TV2** nếu được ủy quyền vì nó thay đổi kiến trúc neural chính.
TV4 chỉ cung cấp evidence (hard‑negative, error analysis, ranking diagnostics) nhưng **KHÔNG** sở hữu fine‑tuning.

---
## Địa chỉ báo cáo

- eports/task1/tv2/ ← artifact TV2
- eports/task1/tv4/ ← artifact TV4
- eports/task1/shared/ ← định nghĩa bất biến
- eports/task1/progress_log.md ← tiến độ chung

---
*All technical terms (Zero‑shot, Fine‑tuning, LoRA, Reranker, Learning‑to‑Rank, Embedding, Retrieval, Hard negative, Prediction freeze, Inference, Cross‑encoder, Long‑context, Fusion) are kept in English and explained the first time they appear.*
