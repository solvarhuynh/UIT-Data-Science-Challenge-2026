"""Debug trực tiếp với transformers: bỏ qua LLMClient, gọi thẳng model.generate()."""

import os
import sys
import json
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from udsc2026.qa.prompt_builder import PromptBuilder
from udsc2026.contracts.retrieval import RetrievalHit

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
model_path = os.path.join(base_dir, "models/qwen3-legal")
submission_path = os.path.join(base_dir, "data/warmup/submission.json")
warmup_path = os.path.join(base_dir, "data/warmup/warmup_task2.json")

# Lấy câu test đầu tiên bị rỗng
with open(submission_path, "r", encoding="utf-8") as f:
    sub_data = json.load(f)
with open(warmup_path, "r", encoding="utf-8") as f:
    warmup_data = json.load(f)

test_qid = next(qid for qid, item in sub_data.items() if not item.get("answer", "").strip())
w_item = warmup_data[test_qid]
q_text = w_item["question"]
ref_text = w_item["answer"]
print(f"Test với QID: {test_qid}")
print(f"Câu hỏi: {q_text}\n")

# Build prompt
prompt_builder = PromptBuilder(prompts_root=Path(os.path.join(base_dir, "prompts")))
context = RetrievalHit(chunk_id=f"ctx_{test_qid}", doc_id=f"doc_{test_qid}", text=ref_text, score=1.0)
prompt = prompt_builder.build_prompt(
    question=q_text, contexts=[context],
    prompt_version="legal_qa_v2", rag_template="default_rag_v2",
)

# Load model trực tiếp
print("Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    ),
)
model.eval()

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[1]
print(f"\n📊 Số token input: {input_len}")

# ---- Test 1: Không có stopping criteria, không timeout ----
print("\n=== TEST 1: generate() không có stopping criteria ===")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = output_ids[0][input_len:]
print(f"Số token output (raw): {len(generated_ids)}")
print(f"Output token ids (10 đầu): {generated_ids[:10].tolist()}")
decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Output decoded (skip_special_tokens=True): '{decoded[:500]}'")

decoded_no_skip = tokenizer.decode(generated_ids, skip_special_tokens=False)
print(f"Output decoded (skip_special_tokens=False): '{decoded_no_skip[:500]}'")

# ---- Test 2: Chat template ----
print("\n=== TEST 2: Dùng apply_chat_template (Qwen3 chat format) ===")
messages = [
    {"role": "system", "content": "Bạn là Trợ lý Pháp luật Việt Nam. Trả lời dựa trên context được cung cấp."},
    {"role": "user", "content": f"CONTEXT:\n{ref_text}\n\nCÂU HỎI: {q_text}"},
]
try:
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Chat prompt (200 ký tự đầu): {chat_prompt[:200]}")

    chat_inputs = tokenizer(chat_prompt, return_tensors="pt")
    chat_inputs = {k: v.to("cuda") for k, v in chat_inputs.items()}
    chat_input_len = chat_inputs["input_ids"].shape[1]
    print(f"Số token chat input: {chat_input_len}")

    with torch.no_grad():
        chat_output_ids = model.generate(
            **chat_inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    chat_generated = chat_output_ids[0][chat_input_len:]
    print(f"Số token output chat: {len(chat_generated)}")
    chat_decoded = tokenizer.decode(chat_generated, skip_special_tokens=True)
    print(f"Output chat decoded: '{chat_decoded[:500]}'")
except Exception as e:
    print(f"Lỗi apply_chat_template: {e}")
