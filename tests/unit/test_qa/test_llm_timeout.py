"""Unit tests for bounded blocking LLM calls without loading a model."""

import threading

import pytest

from udsc2026.infrastructure.llm.client import LLMClient, _run_with_timeout
from udsc2026.infrastructure.llm.config import LLMConfig


def test_run_with_timeout_returns_fast_result_and_preserves_errors() -> None:
    completed = threading.Event()
    assert (
        _run_with_timeout(
            lambda: "answer",
            1.0,
            on_complete=completed.set,
        )
        == "answer"
    )
    assert completed.is_set()

    def fail() -> str:
        raise ValueError("model failed")

    with pytest.raises(ValueError, match="model failed"):
        _run_with_timeout(fail, 1.0)


def test_run_with_timeout_bounds_blocking_operation() -> None:
    release = threading.Event()
    try:
        with pytest.raises(TimeoutError, match="exceeded"):
            _run_with_timeout(lambda: release.wait(5.0), 0.01)
    finally:
        release.set()


def test_timed_out_client_rejects_overlap_until_worker_finishes() -> None:
    release = threading.Event()
    finished = threading.Event()
    client = object.__new__(LLMClient)
    client._config = LLMConfig(  # type: ignore[attr-defined]
        backend="vllm",
        timeout_seconds=0.01,
    )
    client._generation_lock = threading.Lock()  # type: ignore[attr-defined]

    def blocked_generate(_prompt: str) -> str:
        release.wait(1.0)
        finished.set()
        return "done"

    client._generate_vllm = blocked_generate  # type: ignore[method-assign]

    with pytest.raises(TimeoutError):
        client.generate("first")
    with pytest.raises(RuntimeError, match="earlier generation"):
        client.generate("overlap")

    release.set()
    assert finished.wait(1.0)
    assert client._generation_lock.acquire(timeout=1.0)  # type: ignore[attr-defined]
    client._generation_lock.release()  # type: ignore[attr-defined]
    assert client.generate("after completion") == "done"


def test_is_busy_reflects_an_in_flight_generation() -> None:
    release = threading.Event()
    client = object.__new__(LLMClient)
    client._config = LLMConfig(  # type: ignore[attr-defined]
        backend="vllm",
        timeout_seconds=0.05,
    )
    client._generation_lock = threading.Lock()  # type: ignore[attr-defined]

    def blocked_generate(_prompt: str) -> str:
        release.wait(1.0)
        return "done"

    client._generate_vllm = blocked_generate  # type: ignore[method-assign]

    assert client.is_busy() is False
    try:
        with pytest.raises(TimeoutError):
            client.generate("first")
        assert client.is_busy() is True
    finally:
        release.set()
    assert client._generation_lock.acquire(timeout=1.0)  # type: ignore[attr-defined]
    client._generation_lock.release()  # type: ignore[attr-defined]
    assert client.is_busy() is False


def test_qwen_chat_messages_use_tokenizer_template() -> None:
    seen: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return "rendered-chat-prompt"

    client = object.__new__(LLMClient)
    client._tokenizer = FakeTokenizer()  # type: ignore[attr-defined]

    rendered = client._render_chat_messages(
        [
            {"role": "system", "content": "Ground answers."},
            {"role": "user", "content": "Question"},
        ]
    )

    assert rendered == "rendered-chat-prompt"
    assert seen["kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
