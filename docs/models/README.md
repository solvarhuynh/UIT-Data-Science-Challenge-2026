# Chọn model

## Quyết định hiện tại

Giữ model đang cấu hình làm mốc và chỉ tải ứng viên khi chuẩn bị chạy đánh giá. Repo top cung cấp hướng để chọn thí nghiệm, nhưng điểm của họ là kết quả cả pipeline nên không thể suy ra lợi ích của riêng một model. Chưa có kết quả paired OOF trên cùng dữ liệu của repo này để khẳng định xác suất cải thiện 70%.

| Vai trò | Mốc hiện tại | Ứng viên ưu tiên | Bằng chứng và điều kiện |
|---|---|---|---|
| LegalIR embedding | `huyydangg/DEk21_hcmute_embedding_v2` | [`AITeamVN/Vietnamese_Embedding_v2`](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) | Được nhiều repo top dùng hoặc fine-tune. Khác số chiều với mốc hiện tại (1024 so với 768 theo model card đã rà); cần xác nhận pooling/format và dựng lại index. |
| LegalIR reranker | `BAAI/bge-reranker-v2-m3` | [`AITeamVN/Vietnamese_Reranker`](https://huggingface.co/AITeamVN/Vietnamese_Reranker), sau đó [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | AITeamVN xuất hiện trong pipeline top; Endgame và Artiz dùng Qwen3 reranker cho scoring cấp heading/document. So sánh trên cùng candidate list; Qwen3 có scoring contract riêng. |
| LegalIR embedding | DEk21 | [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Có trong tài liệu Endgame. Cần tuân thủ query instruction, pooling và chuẩn hóa theo model card; dựng index riêng. |
| LegalIR embedding | DEk21 | [`darklethelong/vnlegal-lal`](https://huggingface.co/darklethelong/vnlegal-lal) | Model card báo cáo benchmark truy hồi pháp luật Zalo, không phải benchmark cuộc thi này. Cần query prefix, last-token pooling và adapter riêng. Chỉ thử sau các ứng viên tương thích hơn. |
| LegalQA generator | `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | GhienAram báo cáo dùng model trong pipeline LoRA + reranker + nhiều câu trả lời/MBR. Đây không phải phép so sánh model đơn lẻ. Repo đó pin revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; Transformers hiện ghim trong repo này cần được kiểm tra tương thích trước khi dùng. |

Model legal-QA GRPO hiện tại có kết quả nội bộ ghi trong tài liệu repo, nhưng cần xác minh license và điều khoản dữ liệu trước khi công bố hoặc sử dụng. Với mọi checkpoint, kiểm tra license, model card, quy định cuộc thi, số tham số hoạt động và dung lượng trước khi tải/chạy.

## Quy trình chọn model

1. Lưu mốc hiện tại và pin revision Hub cho từng ứng viên.
2. LegalIR: so embedding bằng cùng corpus/chunking/split/metric, dựng index riêng cho từng encoder; so reranker trên cùng danh sách ứng viên.
3. LegalQA: giữ retrieval/evidence/prompt/scorer cố định khi so generator; đo METEOR cùng các metric và lỗi căn cứ.
4. Tính chênh lệch theo từng nhóm câu hỏi, bootstrap khoảng tin cậy và `P(Δ > 0)`. Chỉ coi xác suất cải thiện đạt 70% nếu dữ liệu OOF cùng split hỗ trợ; đây không phải xác suất đạt điểm Private.
5. Ghi latency, VRAM đỉnh, index size, lỗi và revision. Chỉ đổi cấu hình production sau khi ứng viên đạt metric mục tiêu, tài nguyên và quy tắc cuộc thi.

Không gộp model 2B với reranker lớn nếu tổng tham số hoạt động vượt giới hạn 4B của cuộc thi. Các model dưới đây chỉ là lựa chọn tải, không làm thay đổi cấu hình production. Bằng chứng repo top được tóm tắt trong [`docs/comparisons/`](../comparisons/).

## Tải model

```powershell
python -m pip install -e ".[gpu]"
python scripts/download_models.py --list
python scripts/download_models.py --only task1-current
python scripts/download_models.py --only task1-aiteam
```

Mỗi lần tải ghi revision SHA đã phân giải vào `models/download_manifest.json`. Mặc định downloader không tải model nào.
