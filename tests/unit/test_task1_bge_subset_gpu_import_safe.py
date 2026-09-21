from pathlib import Path
from types import SimpleNamespace

import scripts.modal.task1_bge_subset_gpu as runner
from scripts.modal.task1_bge_subset_gpu import LOCAL_ROOT_SENTINEL, resolve_local_root


def test_modal_shallow_source_path_uses_sentinel() -> None:
    assert resolve_local_root(Path("/root/task1_bge_subset_gpu.py")) == LOCAL_ROOT_SENTINEL


def test_local_source_path_resolves_repository_root() -> None:
    source = Path("D:/udsc2026/scripts/modal/task1_bge_subset_gpu.py")
    assert resolve_local_root(source) == Path("D:/udsc2026")


def test_production_dispatch_uses_one_spawn(monkeypatch) -> None:
    calls = []

    class FakeRunSubset:
        def get_current_stats(self):
            return SimpleNamespace(num_running_inputs=0, backlog=0)

        def spawn(self, *args):
            calls.append(args)
            return SimpleNamespace(object_id="fc-test", _app_id="ap-test")

    monkeypatch.setattr(runner, "run_subset", FakeRunSubset())
    call = runner.spawn_production_call(b"payload", 16, 256)

    assert call.object_id == "fc-test"
    assert calls == [(b"payload", "production", None, 16, 256)]


def test_production_dispatch_refuses_active_duplicate(monkeypatch) -> None:
    class FakeRunSubset:
        def get_current_stats(self):
            return SimpleNamespace(num_running_inputs=1, backlog=0)

        def spawn(self, *args):
            raise AssertionError("duplicate production spawn")

    monkeypatch.setattr(runner, "run_subset", FakeRunSubset())
    assert runner.spawn_production_call(b"payload", 16, 256) is None


def test_reattach_uses_existing_function_call_only(monkeypatch) -> None:
    calls = []

    class FakeFunctionCall:
        def get(self):
            calls.append("get")
            return {"status": "COMPLETE"}

    class FakeFunctionCallFactory:
        @staticmethod
        def from_id(call_id):
            calls.append(call_id)
            return FakeFunctionCall()

    monkeypatch.setattr(runner.modal, "FunctionCall", FakeFunctionCallFactory)
    assert runner.reattach_call("fc-existing") == {"status": "COMPLETE"}
    assert calls == ["fc-existing", "get"]
