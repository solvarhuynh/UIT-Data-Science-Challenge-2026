"""Load the approved Qwen3 checkpoint and run one bounded CUDA generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.infrastructure.llm.client import LLMClient  # noqa: E402
from udsc2026.infrastructure.llm.config import load_llm_config  # noqa: E402


def main() -> int:
    config = load_llm_config(ROOT / "configs" / "gpu.yaml").model_copy(
        update={"max_new_tokens": 64, "timeout_seconds": 180.0}
    )
    client = LLMClient(config)
    answer = client.generate_messages(
        [
            {
                "role": "system",
                "content": "Bạn là trợ lý pháp luật. Trả lời cực kỳ ngắn gọn.",
            },
            {
                "role": "user",
                "content": "Hãy trả lời đúng một từ: OK",
            },
        ]
    )
    if not answer.strip():
        raise RuntimeError("Qwen3 smoke generation returned an empty answer")
    print(
        json.dumps(
            {
                "model_path": config.model_path,
                "device": config.device,
                "dtype": config.dtype,
                "answer_preview": answer[:200],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
