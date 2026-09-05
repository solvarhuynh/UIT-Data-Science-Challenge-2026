# TV5 — Task 2 P14: listwise retrieval + Qwen LegalQA SFT

## Kết luận audit

- Hai lần nộp hiện có đạt METEOR `0.488623` và `0.511932`.
- Bản `0.511932` đã dùng 41 câu public trùng chính xác câu train.
- Held-out 1.000 câu của source-only fusion đạt khoảng `0.491` METEOR.
- Oracle chọn một cửa sổ tốt nhất trong 16 parent chỉ đạt khoảng `0.573`.
- Parent cross-encoder HN16 cũ chọn đúng positive top-1 `272/957`
  (`28.42%`), thấp hơn retrieval rank gốc `279/957` (`29.15%`).

Do đó không thể kỳ vọng chạm `0.6` chỉ bằng cách đổi `top_k`. P14 dùng hai
tầng: listwise parent ranking và Qwen SFT để tổng hợp/rút gọn đúng văn phong
đáp án. Mốc `0.6` là cổng đo held-out, không phải lời bảo đảm điểm private.

## Artifact đã chuẩn bị

Dataset SFT hiện có tại:

```text
artifacts/task2/training/task2_qwen_sft_train.jsonl
```

Nó gồm 5.747 câu train, mỗi câu có ba teacher-selected parent nhưng được trả
về thứ tự retrieval để model không học mẹo “context đầu tiên luôn đúng”. Audit
xác nhận không có ID nào thuộc 1.000 câu held-out. Toàn bộ 5.747 mẫu đã qua
tokenization preflight ở cấu hình `3072/1800/1024` token.

## 1. Cài môi trường GPU

Từ thư mục gốc repo:

```powershell
python -m pip install -e ".[gpu,llm]"
```

## 2. Train listwise parent reranker

```powershell
python scripts/training/train_legal_qa_parent_crossencoder.py train `
  --train-data artifacts/task2/training/parent_ce_hn16_train.jsonl `
  --eval-data artifacts/task2/training/parent_ce_hn16_eval.jsonl `
  --model-dir artifacts/task2/models/dek21-parent-crossencoder-cv1/checkpoint-best `
  --output-dir artifacts/task2/models/dek21-parent-listwise-p14 `
  --objective listwise `
  --target-temperature 0.10 `
  --epochs 3 `
  --learning-rate 1e-5 `
  --batch-size 2 `
  --eval-batch-size 2 `
  --gradient-accumulation 8 `
  --max-length 256 `
  --device cuda `
  --amp
```

Score đúng 1.000 câu held-out:

```powershell
python scripts/training/train_legal_qa_parent_crossencoder.py score `
  --questions data/raw/btc/LegalQA/train.json `
  --candidates artifacts/task2/train500-bge-reranker-all/predictions_after.jsonl `
  --candidates artifacts/task2/train500b-bge-reranker-all/predictions_after.jsonl `
  --parents-dir data/processed_v3/parents `
  --checkpoint artifacts/task2/models/dek21-parent-listwise-p14/checkpoint-best `
  --output artifacts/task2/evaluation/p14_listwise_strict1000/ranked_parents.jsonl `
  --candidate-k 50 `
  --batch-size 32 `
  --max-length 256 `
  --device cuda `
  --amp
```

Sanity-check source-only; nếu kết quả thấp hơn `0.491`, không promote reranker
mới và vẫn dùng HN16 fusion cũ cho tầng Qwen:

```powershell
python scripts/evaluation/evaluate_legal_qa_parent_scores_grid.py `
  --scores artifacts/task2/evaluation/p14_listwise_strict1000/ranked_parents.jsonl `
  --questions data/raw/btc/LegalQA/train.json `
  --question-ids artifacts/task2/training/parent_ce_eval_ids.json `
  --output-dir artifacts/task2/evaluation/p14_listwise_grid `
  --top-parents 1 2 3 4 `
  --max-parent-tokens 384 512 768
```

## 3. Fine-tune Qwen bằng LoRA

RTX 4090/5090 nên dùng BF16. Lệnh lưu adapter từng epoch và merged model cuối:

```powershell
python scripts/training/finetune_task2_qwen_lora.py train `
  --train-data artifacts/task2/training/task2_qwen_sft_train.jsonl `
  --model-dir models/qwen3-legal `
  --output-dir artifacts/task2/models/qwen-legalqa-lora-p14 `
  --merge-output artifacts/task2/models/qwen-legalqa-merged-p14 `
  --epochs 2 `
  --batch-size 1 `
  --gradient-accumulation 16 `
  --learning-rate 2e-4 `
  --max-length 3072 `
  --max-context-tokens 1800 `
  --max-answer-tokens 1024 `
  --lora-rank 32 `
  --lora-alpha 64 `
  --dtype bfloat16 `
  --device cuda
```

## 4. Sinh và chấm strict held-out

Không bật known-answer overlay ở bước này. Như vậy 1.000 câu vẫn thật sự
held-out:

```powershell
python scripts/training/finetune_task2_qwen_lora.py generate `
  --questions data/raw/btc/LegalQA/train.json `
  --question-ids artifacts/task2/training/parent_ce_eval_ids.json `
  --rankings artifacts/task2/evaluation/p14_listwise_strict1000/ranked_parents.jsonl `
  --model-dir artifacts/task2/models/qwen-legalqa-merged-p14 `
  --output artifacts/task2/evaluation/p14_qwen_strict1000/predictions.json `
  --diagnostics artifacts/task2/evaluation/p14_qwen_strict1000/diagnostics.json `
  --batch-size 4 `
  --top-parents 3 `
  --max-parent-words 768 `
  --max-context-tokens 2400 `
  --max-new-tokens 1024 `
  --ce-weight 0.60 `
  --retrieval-weight 0.40 `
  --rrf-k 20 `
  --dtype bfloat16 `
  --device cuda `
  --resume
```

Cổng promote bắt buộc:

```powershell
python scripts/training/finetune_task2_qwen_lora.py evaluate `
  --questions data/raw/btc/LegalQA/train.json `
  --question-ids artifacts/task2/training/parent_ce_eval_ids.json `
  --predictions artifacts/task2/evaluation/p14_qwen_strict1000/predictions.json `
  --output artifacts/task2/evaluation/p14_qwen_strict1000/metrics.json `
  --minimum-meteor 0.60
```

Nếu script ghi `REJECTED`, không tạo bản nộp. Khi đó thử checkpoint epoch 1
hoặc quay về rankings HN16 cũ; tuyệt đối không dùng đáp án held-out để chọn
context hay sửa tay.

## 5. Public inference và đóng gói

Chỉ thực hiện sau khi strict gate báo `PROMOTED`. Trước tiên score public bằng
checkpoint listwise với candidate
`artifacts/task2/public-bge-reranker-all/predictions_after.jsonl`:

```powershell
python scripts/training/train_legal_qa_parent_crossencoder.py score `
  --questions data/raw/btc/LegalQA/public-official.json `
  --candidates artifacts/task2/public-bge-reranker-all/predictions_after.jsonl `
  --parents-dir data/processed_v3/parents `
  --checkpoint artifacts/task2/models/dek21-parent-listwise-p14/checkpoint-best `
  --output artifacts/task2/candidates/p14_listwise_public/ranked_parents.jsonl `
  --candidate-k 50 `
  --batch-size 32 `
  --max-length 256 `
  --device cuda `
  --amp
```

Sau đó sinh answer, áp dụng lại exact organizer overlay hợp lệ và đóng gói:

```powershell
python scripts/training/finetune_task2_qwen_lora.py generate `
  --questions data/raw/btc/LegalQA/public-official.json `
  --rankings artifacts/task2/candidates/p14_listwise_public/ranked_parents.jsonl `
  --model-dir artifacts/task2/models/qwen-legalqa-merged-p14 `
  --output artifacts/task2/candidates/p14_qwen_public/predictions.json `
  --diagnostics artifacts/task2/candidates/p14_qwen_public/diagnostics.json `
  --known-answers data/raw/btc/LegalQA/train.json `
  --batch-size 4 `
  --top-parents 3 `
  --max-parent-words 768 `
  --max-context-tokens 2400 `
  --max-new-tokens 1024 `
  --ce-weight 0.60 `
  --retrieval-weight 0.40 `
  --rrf-k 20 `
  --dtype bfloat16 `
  --device cuda `
  --resume

python scripts/submission/write_legal_qa_submission.py `
  --input artifacts/task2/candidates/p14_qwen_public/predictions.json `
  --questions data/raw/btc/LegalQA/public-official.json `
  --empty-answer-policy error `
  --output artifacts/task2/ready_to_submit_p14/submission.zip

python scripts/submission/validate_legal_qa_submission.py `
  --input artifacts/task2/ready_to_submit_p14/submission.zip `
  --questions data/raw/btc/LegalQA/public-official.json
```

File nộp duy nhất là
`artifacts/task2/ready_to_submit_p14/submission.zip`.
