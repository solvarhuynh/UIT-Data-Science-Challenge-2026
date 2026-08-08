"""Unit tests for the lazy CrossEncoder scoring client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import udsc2026.infrastructure.reranker.client as client_module
from udsc2026.infrastructure.reranker import (
    CrossEncoderClient,
    RerankerDependencyError,
    RerankerModelLoadError,
    RerankerScoringError,
)


class FakeCrossEncoder:
    """Small in-memory substitute for sentence-transformers.CrossEncoder."""

    instances: list["FakeCrossEncoder"] = []

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None,
        max_length: int,
        local_files_only: bool,
        automodel_args: dict[str, Any],
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.automodel_args = automodel_args
        self.predict_calls: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.predict_calls.append({"pairs": pairs, **kwargs})
        return [float(index) / 10 for index in range(1, len(pairs) + 1)]


@pytest.fixture(autouse=True)
def clear_fake_instances() -> None:
    FakeCrossEncoder.instances.clear()


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    cross_encoder_class: type[Any] = FakeCrossEncoder,
) -> None:
    fake_module = SimpleNamespace(CrossEncoder=cross_encoder_class)
    fake_torch = SimpleNamespace(float16="fake-float16")

    def fake_import(name: str) -> object:
        return fake_torch if name == "torch" else fake_module

    monkeypatch.setattr(client_module, "import_module", fake_import)


@pytest.mark.unit
def test_initialization_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_imported(_: str) -> None:
        raise AssertionError("sentence-transformers was imported eagerly")

    monkeypatch.setattr(client_module, "import_module", fail_if_imported)
    client = CrossEncoderClient("org/legal-cross-encoder")

    assert client.is_loaded is False


@pytest.mark.unit
def test_empty_documents_do_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_imported(_: str) -> None:
        raise AssertionError("empty scoring should not load the model")

    monkeypatch.setattr(client_module, "import_module", fail_if_imported)
    client = CrossEncoderClient("models/local-reranker")

    assert client.score("câu hỏi", []) == []
    assert client.is_loaded is False


@pytest.mark.unit
def test_loads_once_and_forwards_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    client = CrossEncoderClient(
        "org/legal-cross-encoder",
        device="cuda:0",
        batch_size=16,
        max_length=384,
        local_files_only=True,
    )

    first = client.score("quyền nghỉ phép?", ["văn bản 1", "văn bản 2"])
    second = client.score("câu hỏi khác", ["văn bản 3"], batch_size=4)

    assert first == [0.1, 0.2]
    assert second == [0.1]
    assert len(FakeCrossEncoder.instances) == 1
    model = FakeCrossEncoder.instances[0]
    assert model.model_name_or_path == "org/legal-cross-encoder"
    assert model.device == "cuda:0"
    assert model.max_length == 384
    assert model.local_files_only is True
    assert model.automodel_args == {}
    assert model.predict_calls[0] == {
        "pairs": [
            ("quyền nghỉ phép?", "văn bản 1"),
            ("quyền nghỉ phép?", "văn bản 2"),
        ],
        "batch_size": 16,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    assert model.predict_calls[1]["batch_size"] == 4
    assert client.is_loaded is True


@pytest.mark.unit
def test_fp16_is_forwarded_to_transformer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sentence_transformers(monkeypatch)
    client = CrossEncoderClient(
        "org/legal-cross-encoder",
        device="cuda:0",
        use_fp16=True,
    )

    assert client.score("cau hoi", ["van ban"]) == [0.1]
    assert FakeCrossEncoder.instances[0].automodel_args == {
        "torch_dtype": "fake-float16"
    }


@pytest.mark.unit
def test_missing_optional_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(_: str) -> None:
        raise ModuleNotFoundError("No module named 'sentence_transformers'")

    monkeypatch.setattr(client_module, "import_module", missing_dependency)
    client = CrossEncoderClient("org/legal-cross-encoder")

    with pytest.raises(RerankerDependencyError, match="pip install"):
        client.score("câu hỏi", ["văn bản"])


@pytest.mark.unit
def test_model_load_failure_is_wrapped_with_model_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenCrossEncoder:
        def __init__(self, *_: Any, **__: Any) -> None:
            raise OSError("missing weights")

    _install_fake_sentence_transformers(monkeypatch, BrokenCrossEncoder)
    client = CrossEncoderClient("missing/model")

    with pytest.raises(RerankerModelLoadError, match="missing/model"):
        client.score("câu hỏi", ["văn bản"])


@pytest.mark.unit
def test_predict_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenPredictCrossEncoder(FakeCrossEncoder):
        def predict(
            self,
            pairs: list[tuple[str, str]],
            **kwargs: Any,
        ) -> list[float]:
            raise RuntimeError("out of memory")

    _install_fake_sentence_transformers(monkeypatch, BrokenPredictCrossEncoder)
    client = CrossEncoderClient("models/local")

    with pytest.raises(RerankerScoringError, match="1 candidate"):
        client.score("câu hỏi", ["văn bản"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_scores", "message"),
    [
        ([0.1], "1 score"),
        ([[0.1, 0.9], [0.3, 0.7]], "not a scalar"),
        ([0.1, float("nan")], "must be finite"),
        ([0.1, True], "not a scalar"),
    ],
)
def test_rejects_invalid_model_outputs(
    monkeypatch: pytest.MonkeyPatch,
    raw_scores: object,
    message: str,
) -> None:
    class InvalidOutputCrossEncoder(FakeCrossEncoder):
        def predict(self, *_: Any, **__: Any) -> object:
            return raw_scores

    _install_fake_sentence_transformers(monkeypatch, InvalidOutputCrossEncoder)
    client = CrossEncoderClient("models/local")

    with pytest.raises(RerankerScoringError, match=message):
        client.score("câu hỏi", ["văn bản 1", "văn bản 2"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name_or_path": ""},
        {"model_name_or_path": "model", "device": ""},
        {"model_name_or_path": "model", "batch_size": 0},
        {"model_name_or_path": "model", "batch_size": True},
        {"model_name_or_path": "model", "max_length": -1},
        {"model_name_or_path": "model", "local_files_only": "yes"},
        {"model_name_or_path": "model", "use_fp16": "yes"},
        {"model_name_or_path": "model", "device": "cpu", "use_fp16": True},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CrossEncoderClient(**kwargs)


@pytest.mark.unit
def test_rejects_invalid_score_inputs() -> None:
    client = CrossEncoderClient("models/local")

    with pytest.raises(ValueError, match="query"):
        client.score(" ", ["văn bản"])
    with pytest.raises(TypeError, match="sequence"):
        client.score("câu hỏi", "không phải danh sách")
    with pytest.raises(ValueError, match=r"documents\[1\]"):
        client.score("câu hỏi", ["hợp lệ", " "])
    with pytest.raises(ValueError, match="batch_size"):
        client.score("câu hỏi", ["văn bản"], batch_size=0)
