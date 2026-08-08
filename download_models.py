import os

from huggingface_hub import snapshot_download

# Định nghĩa đường dẫn đích dựa theo cấu trúc thư mục của bạn
embedder_dir = "./models/hcmute-embedding-v2"
llm_dir = "./models/qwen3-legal"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(embedder_dir, exist_ok=True)
os.makedirs(llm_dir, exist_ok=True)

print("Đang tải HCMUTE embedding v2...")
snapshot_download(
    repo_id="huyydangg/DEk21_hcmute_embedding_v2", local_dir=embedder_dir
)

print("\nĐang tải Qwen3 1.7B Legal...")
snapshot_download(
    repo_id="thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2", local_dir=llm_dir
)

print("\nHoàn tất tải models!")

