# LLM Optimization cho Qwen3 Legal

[Qwen3 Legal Model](https://huggingface.co/thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2)

## Mục tiêu

Tài liệu này tổng hợp các khái niệm tối ưu LLM còn hữu ích từ kiến trúc cũ và viết lại cho stack hiện tại: local generator `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`, FastAPI backend, React frontend, streaming và citation.

## LoRA

LoRA, Low-Rank Adaptation, giữ nguyên trọng số model gốc và thêm các ma trận trainable nhỏ.

```text
W_new = W + delta_W
delta_W = B * A
```

Lợi ích:

- Giảm số tham số cần train.
- Giảm VRAM so với full fine-tuning.
- Dễ lưu adapter riêng cho domain pháp luật.

Tham số quan trọng:

- `r`: rank của adapter, rank cao học tốt hơn nhưng tốn VRAM hơn.
- `lora_alpha`: hệ số scale, thường đặt quanh `2 * r`.
- `lora_dropout`: chống overfit khi dataset nhỏ.
- `target_modules`: thường gồm `q_proj`, `k_proj`, `v_proj`, `o_proj`, có thể thêm MLP modules.

## QLoRA và Quantization

Quantization giảm số bit lưu trọng số model, ví dụ FP16 xuống 8-bit hoặc 4-bit. QLoRA kết hợp base model 4-bit với LoRA adapter FP16/BF16.

Lợi ích:

- Chạy hoặc fine-tune model trên GPU nhỏ hơn.
- Giảm dung lượng VRAM.
- Phù hợp khi cần thử nghiệm nhanh trong `experiments/`.

Rủi ro:

- Có thể giảm chất lượng nếu quantization quá mạnh.
- Cần kiểm tra hallucination và citation correctness sau khi đổi precision.

## GQA

Grouped Query Attention chia nhiều query heads dùng chung key/value heads. Cơ chế này giảm KV cache và tăng tốc inference cho long context.

Ý nghĩa với LegalQA:

- Context pháp luật thường dài.
- Streaming cần latency ổn định.
- KV cache nhỏ hơn giúp phục vụ nhiều request hơn.

## Prompt formatting

Prompt phải ép model trả lời dựa trên context truy hồi, không suy diễn ngoài nguồn.

Template khuyến nghị:

```text
SYSTEM:
Bạn là trợ lý pháp luật Việt Nam. Chỉ trả lời dựa trên CONTEXT.
Nếu CONTEXT không đủ căn cứ, nói rõ không tìm thấy căn cứ pháp luật trong dữ liệu được cung cấp.
Luôn trích dẫn theo dạng [law_name, article, clause] khi có căn cứ.

CONTEXT:
{retrieved_chunks}

QUESTION:
{question}

ANSWER:
```

## Quy tắc sinh câu trả lời

- Ưu tiên câu trả lời ngắn, đúng trọng tâm.
- Mỗi kết luận pháp lý quan trọng phải có citation.
- Không bịa số điều, ngày ban hành hoặc tên văn bản.
- Nếu context mâu thuẫn, nêu rõ có nhiều nguồn khác nhau và ưu tiên văn bản mới hơn nếu metadata có ngày hiệu lực.

## Inference config baseline

```yaml
model_path: ./models/qwen3-legal
max_new_tokens: 1024
temperature: 0.1
top_p: 0.9
repetition_penalty: 1.05
stream: true
```

Temperature thấp giúp câu trả lời pháp lý ổn định hơn. Các thay đổi cấu hình phải được lưu trong `configs/` để tái lập thí nghiệm.
