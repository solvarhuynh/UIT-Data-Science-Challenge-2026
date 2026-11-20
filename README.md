# UDSC 2026

Hệ thống truy hồi tài liệu pháp luật, sinh câu trả lời có dẫn chứng và tạo gói theo schema của Ban tổ chức.

## Mô hình

- Embedding: `AITeamVN/Vietnamese_Embedding_v2` (1.024 chiều; truy vấn 256 token, tài liệu 2.048 token).
- Reranker: `AITeamVN/Vietnamese_Reranker`.
- Sinh câu trả lời: `Qwen/Qwen3.5-2B`.

Trọng số nằm trong `models/`; cấu hình trong `configs/`. Dữ liệu, index và kết quả được giữ trong `data/`, ngoài Git.

## Cài đặt

Yêu cầu Python 3.12. Trong PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[answers,data,modal,models]"
python scripts/shared/download_models.py --profile all
```

Đăng nhập Modal và gắn Volume tên `udsc`. Đưa dữ liệu đã chuẩn bị cùng ba thư mục model vào Volume theo cùng cấu trúc `data/` và `models/` của repo.

## Dựng index

Runner yêu cầu A10 hoặc L4; Modal đã cấp L4 cho LegalIR hiện tại. Index phải hoàn tất đủ 1.270.356 chunks và có manifest hợp lệ trước khi chạy retrieval.

```powershell
python -m modal run --detach scripts/data_prep/run_modal_index.py --batch-size 16 --checkpoint-chunks 100000
```

Lượt dựng index chạy ở chế độ detach để việc đóng terminal không hủy job. Theo dõi trạng thái trong Modal dashboard hoặc `python -m modal app logs <APP_ID>`. Nếu job dừng ngoài checkpoint, chạy lại cùng lệnh kèm `--resume`; index chỉ tiếp tục khi hash nguồn, model, cấu hình và corpus khớp checkpoint đã lưu.

Nếu index đã hoàn tất nhưng bước kiểm tra/ghi checksum sau đó bị lỗi, chạy CPU finalizer để hoàn tất manifest mà không dựng lại embeddings:

```powershell
python -m modal run --detach scripts/data_prep/run_modal_index.py --finalize-only
```

## LegalIR

Chạy 5-fold đánh giá trên train:

```powershell
python -m modal run scripts/retrieval/inference/run_modal.py --dataset train
```

Tạo và kiểm tra gói cho câu hỏi public:

```powershell
python -m modal run scripts/retrieval/inference/run_modal.py --dataset public
```

Runner lấy 200 ứng viên dense cho mỗi câu, chấm lại 100 ứng viên đầu bằng AITeam Reranker, rồi đo coverage@K và đối chiếu dense-only, reranker-only, hybrid RRF trên cùng pool qua 5 folds cùng paired grouped bootstrap. Gói cuối giữ 5 tài liệu/câu. Hybrid dùng trọng số dense/reranker/BM25 là `0,5/1,0/0,3` với `RRF k=60`; gói public được lưu dưới `data/submissions/retrieval/` trên Volume.

## LegalQA

Sinh và chấm câu train:

```powershell
python -m modal run scripts/answers/inference/run_modal.py --dataset train --limit 500 --save-every 100
```

500 câu được chọn ngẫu nhiên cố định theo seed 2026. Runner chấm Qwen và baseline trích dẫn trên cùng câu hỏi, sau đó ghi paired grouped bootstrap `P(Δ METEOR > 0)`. Nếu kết quả có ích, mở rộng sang 7.000 câu bằng `--limit 7000`.

Sinh và đóng gói 1.000 câu public:

```powershell
python -m modal run scripts/answers/inference/run_modal.py --dataset public --save-every 100
```

Các checkpoint được ghi theo từng 100 câu. Gói LegalQA được kiểm tra số ID và schema trước khi lưu tại `data/submissions/answers/submission.zip` trên Volume.

## Kết quả và giới hạn đánh giá

- LegalIR được chấm bằng Recall/Precision theo scorer BTC; mục tiêu tham khảo là macro Recall ít nhất 0,95.
- LegalQA báo METEOR và ROUGE-L riêng theo scorer BTC; mục tiêu 0,65 phải được hiểu theo metric được chỉ rõ trong báo cáo, không tự gộp hai metric.
- Kết quả train là chẩn đoán trên dữ liệu có nhãn, không phải ước lượng chắc chắn cho tập ẩn. Public chỉ dùng để tạo gói, không dùng chọn model hay tham số.
- `P(Δ > 0)` từ bootstrap nhóm là xác suất theo các mẫu bootstrap trên benchmark đã đo, không phải xác suất hiệu năng trên leaderboard riêng.

## Bố cục

- `src/udsc2026/`: hợp đồng dữ liệu, ingestion, retrieval, sinh câu trả lời và đánh giá.
- `scripts/data_prep/`: xử lý dữ liệu và dựng index.
- `scripts/retrieval/`: sinh/rerank candidates, đánh giá và đóng gói LegalIR.
- `scripts/answers/`: sinh, chấm và đóng gói LegalQA.
- `configs/`, `prompts/`: cấu hình và mẫu prompt đang dùng.
