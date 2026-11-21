<div align="center">
  <img src="assets/banner.svg" alt="UIT Data Science Challenge 2026 - HCMUTE-ShipCODE" width="50%" />
  <h1>HCMUTE-ShipCODE · UDSC 2026</h1>
  <p><strong>Truy hồi và hỏi đáp trên văn bản pháp luật tiếng Việt, có căn cứ và kiểm chứng được.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Competition-Top%2010-2ea44f?style=flat-square" alt="Top 10" />
    <img src="https://img.shields.io/badge/Task%201-LegalIR-2563eb?style=flat-square" alt="LegalIR" />
    <img src="https://img.shields.io/badge/Task%202-LegalQA-7c3aed?style=flat-square" alt="LegalQA" />
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.12" />
  </p>
</div>

> **Kết quả cuộc thi:** đội đạt **Top 10** UDSC 2026. Điểm cuối cùng được công bố: **Task 1 - LegalIR: 0.954**; **Task 2 - LegalQA: 0.652**.

## Tổng quan

HCMUTE-ShipCODE xây dựng hệ thống xử lý câu hỏi pháp luật tiếng Việt cho **UIT Data Science Challenge 2026**. Hệ thống giải quyết hai tác vụ: tìm văn bản pháp luật liên quan (**LegalIR**) và sinh câu trả lời dựa trên ngữ cảnh truy hồi (**LegalQA**).

Pipeline giữ cấu trúc pháp lý trong khâu phân đoạn, kết hợp nhiều tín hiệu truy hồi, xếp hạng lại ứng viên và tạo câu trả lời có căn cứ. Các thành phần được tổ chức thành CLI vận hành, cấu hình và công cụ đóng gói theo schema cuộc thi.

## Kết quả cuộc thi

| Tác vụ | Bài toán | Điểm cuối cùng |
|---|---|---:|
| **Task 1 - LegalIR** | Truy hồi văn bản pháp luật liên quan | **0.954** |
| **Task 2 - LegalQA** | Hỏi đáp pháp luật dựa trên ngữ cảnh | **0.652** |
| **Thứ hạng** | UIT Data Science Challenge 2026 | **Top 10** |

Giấy khen: [`assets/UDSC.jpg`](assets/UDSC.jpg).

## Nhóm thực hiện

| Thành viên | Phạm vi |
|---|---|
| Hoàng Long | Tích hợp hệ thống và điều phối |
| Trung Nghĩa (Leader) | Truy hồi thông tin |
| Thế Quân | Hỏi đáp và mô hình ngôn ngữ |
| Trung Khang | Kỹ thuật dữ liệu và xử lý corpus |
| Nguyên Khang | Reranking, đánh giá và vận hành |

## Phương pháp

### Dữ liệu và xử lý văn bản

Corpus gồm **8.532 tài liệu**, được xử lý thành **1.270.356 chunks** và **184.548 ngữ cảnh cha (parents)**. Bộ phân tích nhận diện các cấp pháp lý như chương, mục, điều, khoản và điểm. Chunk ngắn phục vụ truy hồi; parent context cung cấp ngữ cảnh rộng hơn cho bước trả lời.

### Task 1 - LegalIR

1. Mã hóa câu hỏi và chunks bằng `AITeamVN/Vietnamese_Embedding_v2`, tạo ứng viên dense từ FAISS.
2. Dùng BM25 làm kênh lexical bổ sung cho số hiệu văn bản và thuật ngữ pháp lý.
3. Hợp nhất thứ hạng bằng Reciprocal Rank Fusion (RRF), rồi chấm lại ứng viên bằng `AITeamVN/Vietnamese_Reranker`.
4. Tổng hợp thứ hạng về tài liệu và xuất tối đa năm tài liệu cho mỗi câu hỏi.

### Task 2 - LegalQA

Pipeline truy hồi và reranking chọn bằng chứng, mở rộng ngữ cảnh từ parent store, rồi đưa câu hỏi và ngữ cảnh vào `Qwen/Qwen3.5-2B`. Prompt hướng mô hình bám sát bằng chứng và giữ thông tin trích dẫn phù hợp. Đầu ra được kiểm tra và đóng gói theo schema LegalQA.

### Mô hình

| Vai trò | Mô hình |
|---|---|
| Embedding | [`AITeamVN/Vietnamese_Embedding_v2`](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) |
| Reranker | [`AITeamVN/Vietnamese_Reranker`](https://huggingface.co/AITeamVN/Vietnamese_Reranker) |
| Sinh câu trả lời | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) |

## Tái lập pipeline

### Cài đặt

Yêu cầu Python 3.12. Dữ liệu và trọng số mô hình được quản lý ngoài Git; đặt theo cấu trúc `data/` và `models/` mà cấu hình yêu cầu. Pipeline GPU chạy qua Modal, ưu tiên A10 và dùng L4 dự phòng.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[answers,data,modal,models]"
python scripts/shared/download_models.py --profile all
```

### Dựng index

Xác minh manifest index hoàn tất trước khi tạo candidates.

```powershell
python -m modal run --detach scripts/data_prep/run_modal_index.py --batch-size 16 --checkpoint-chunks 100000
```

### Chạy LegalIR

```powershell
python -m modal run scripts/retrieval/inference/run_modal.py --dataset train --candidate-k 500 --rerank-k 200
python -m modal run scripts/retrieval/inference/run_modal.py --dataset public --candidate-k 500 --rerank-k 200
```

### Chạy LegalQA

```powershell
python -m modal run scripts/answers/inference/run_modal.py --dataset train --limit 500 --candidate-k 500 --rerank-k 200 --save-every 100
python -m modal run scripts/answers/inference/run_modal.py --dataset public --save-every 100
```

Đây là các lệnh vận hành theo cấu hình hiện hành. Nhãn/reference chỉ dành cho scorer và chẩn đoán; candidate generation, reranking và answer generation nhận mapping câu hỏi. Kết quả mỗi lần chạy cần gắn với input, model, config, scorer và split tương ứng.


## Cấu trúc repository

```text
src/udsc2026/
├── ingestion/          # Đọc, chuẩn hóa, phân tích cấu trúc và chia đoạn văn bản
├── retrieval/          # Dense/sparse retrieval, hợp nhất, context và reranking
├── answer_generation/  # Prompt, sinh câu trả lời và kiểm tra trích dẫn
├── evaluation/         # Scorer, phân tích và hợp đồng submission
└── infrastructure/     # Cấu hình, model clients và vector store
scripts/
├── data_prep/          # Chuẩn bị dữ liệu và dựng index
├── retrieval/          # Inference, đánh giá và đóng gói LegalIR
├── answers/            # Sinh, chấm và đóng gói LegalQA
└── shared/             # Tiện ích vận hành dùng chung
configs/                # Cấu hình pipeline
prompts/                # Prompt sinh câu trả lời
data/                   # Dữ liệu, index và predictions (ngoài Git)
models/                 # Trọng số mô hình (ngoài Git)
assets/                  # Banner, logo và giấy khen
```

## Ghi nhận

- **Cuộc thi:** UIT Data Science Challenge 2026.
- **Đội:** HCMUTE-ShipCODE.
- **Thành tích:** Top 10; Task 1 **0.954**, Task 2 **0.652**.

## Trích dẫn và giấy phép

Khi sử dụng pipeline trong nghiên cứu, vui lòng ghi mã commit, tập dữ liệu, cấu hình mô hình và scorer dùng cho kết quả được báo cáo. Điều khoản sử dụng mã nguồn nằm trong [`LICENSE`](LICENSE); dữ liệu cuộc thi và model weights có thể có giấy phép riêng.

