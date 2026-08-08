"""Debug: In prompt thực sự được truyền vào LLM cho câu ID cụ thể."""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from udsc2026.qa.prompt_builder import PromptBuilder
from udsc2026.contracts.retrieval import RetrievalHit

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
submission_path = os.path.join(base_dir, "data/warmup/submission.json")
warmup_path = os.path.join(base_dir, "data/warmup/warmup_task2.json")

with open(submission_path, "r", encoding="utf-8") as f:
    sub_data = json.load(f)

with open(warmup_path, "r", encoding="utf-8") as f:
    warmup_data = json.load(f)

# Lấy câu đầu tiên bị rỗng
test_qid = None
for qid, item in sub_data.items():
    if not item.get("answer", "").strip():
        test_qid = qid
        break

if test_qid is None:
    print("Không tìm thấy câu nào bị rỗng!")
    exit()

w_item = warmup_data[test_qid]
q_text = w_item["question"]
ref_text = w_item["answer"]

print(f"QID: {test_qid}")
print(f"Câu hỏi: {q_text}")
print(f"Đáp án gốc (500 ký tự đầu): {ref_text[:500]}")
print()

context = RetrievalHit(
    chunk_id=f"ctx_{test_qid}",
    doc_id=f"doc_{test_qid}",
    text=ref_text,
    score=1.0,
)

prompt_builder = PromptBuilder(
    prompts_root=Path(os.path.join(base_dir, "prompts"))
)

prompt = prompt_builder.build_prompt(
    question=q_text,
    contexts=[context],
    prompt_version="legal_qa_v2",
    rag_template="default_rag_v2",
)

print("=" * 60)
print("PROMPT GỬI VÀO MODEL:")
print("=" * 60)
print(prompt)
print("=" * 60)
print(f"Tổng độ dài prompt: {len(prompt)} ký tự")
