# Models Documentation

Thư mục này gom toàn bộ tài liệu liên quan đến model, checkpoint, tối ưu inference và quy ước sử dụng model trong hệ thống.

## Cấu trúc

- `model_registry.md`: nguồn đăng ký checkpoint chính thức cho Retriever, Generator và Reranker.
- `qwen3_vl_reranker_2b.md`: hướng dẫn tải và benchmark reranker hiện tại.
- `DEk21_hcmute_embedding_v2.md`: thông tin và cách dùng embedding model chính thức.
- `embedding_bge_m3.md`: ghi chú kiến trúc retrieval cũ, chỉ dùng để tham khảo.
- `qwen3_vl_reranker_2b.md`: hướng dẫn, giới hạn và cách benchmark Qwen3-VL-Reranker-2B.
- `llm_optimization.md`: chiến lược tối ưu Qwen3, LoRA/QLoRA và inference.

## Quy ước

- Tài liệu quản lý checkpoint, version, license và giới hạn tham số đặt trong `model_registry.md`.
- Tài liệu tối ưu hoặc thử nghiệm model cụ thể đặt thành file riêng trong thư mục này.
- Không commit model weights vào `docs/` hoặc `models/`; chỉ ghi metadata, link tải và hướng dẫn tái tạo.
