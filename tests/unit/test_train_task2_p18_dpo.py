"""Model-free contracts for the P18 preference optimizer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/training/train_task2_p18_dpo.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p18_dpo_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [10 + index for index, _ in enumerate(text.split())]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **_: object,
    ) -> list[int]:
        content = " ".join(message["content"] for message in messages)
        return [3, 4] + self.encode(content)


def test_load_preferences_rejects_duplicates(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "preferences.jsonl"
    row = {
        "id": "q1",
        "question": "Câu hỏi?",
        "contexts": [{"text": "Căn cứ pháp luật."}],
        "chosen": "Đáp án đúng.",
        "rejected": "Đáp án kém.",
    }
    path.write_text(
        json.dumps(row, ensure_ascii=False)
        + "\n"
        + json.dumps(row, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate preference ID"):
        script._load_preferences(path)


def test_load_preferences_skips_ties(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "preferences.jsonl"
    tied = {
        "id": "tie",
        "question": "Câu hỏi?",
        "contexts": [{"text": "Căn cứ."}],
        "chosen": "Giống nhau.",
        "rejected": "Giống nhau.",
    }
    useful = {**tied, "id": "useful", "rejected": "Khác."}
    path.write_text(
        json.dumps(tied, ensure_ascii=False)
        + "\n"
        + json.dumps(useful, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    rows = script._load_preferences(path)

    assert [row["id"] for row in rows] == ["useful"]


def test_preference_pair_has_identical_prompt_and_bounded_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load()
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(script.qwen, "_context_block", lambda _: "context words")
    monkeypatch.setattr(
        script.qwen,
        "_truncate_context",
        lambda _tokenizer, value, limit: " ".join(value.split()[:limit]),
    )
    row = {
        "id": "q1",
        "question": "question words",
        "contexts": [{"text": "context words"}],
        "chosen": "one two three four five",
        "rejected": "six seven",
    }

    encoded = script.encode_preference(
        tokenizer,
        row,
        max_length=64,
        max_context_tokens=8,
        max_answer_tokens=3,
    )

    chosen_prompt = encoded["chosen_ids"][: encoded["chosen_labels"].count(-100)]
    rejected_prompt = encoded["rejected_ids"][
        : encoded["rejected_labels"].count(-100)
    ]
    assert chosen_prompt == rejected_prompt
    assert len(encoded["chosen_ids"]) <= 64
    assert encoded["chosen_labels"][-1] == tokenizer.eos_token_id
    assert encoded["rejected_labels"][-1] == tokenizer.eos_token_id


def test_sequence_logps_are_length_normalized() -> None:
    torch = pytest.importorskip("torch")
    script = _load()
    labels = torch.tensor([[-100, 1, 1], [-100, 1, -100]])
    logits = torch.zeros((2, 3, 3))

    result = script._sequence_average_logps(torch, logits, labels)

    expected = -torch.log(torch.tensor(3.0))
    assert result.tolist() == pytest.approx([float(expected), float(expected)])


def test_default_contract_is_conservative() -> None:
    script = _load()
    args = script.build_parser().parse_args(
        [
            "--preference-data",
            "preferences.jsonl",
            "--model-dir",
            "base",
            "--initial-adapter",
            "adapter",
            "--output-dir",
            "output",
        ]
    )

    assert args.epochs == 1
    assert args.learning_rate == pytest.approx(5e-6)
    assert args.beta == pytest.approx(0.2)
    assert args.sft_weight == pytest.approx(0.1)
    assert args.max_length == 2560
