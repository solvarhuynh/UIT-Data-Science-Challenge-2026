"""Run the gated Task 2 P15 continuation pipeline on Modal.

P15 reuses the verified P14 corpus/model archive and the already paid-for
retrieval outputs.  One GPU allocation performs smoke, a disjoint development
gate, final continuation training, generation, and submission validation.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, Callable

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

APP_NAME = "udsc-task2-p15-modal"
VOLUME_NAME = "udsc-task2-p15-modal"
VOLUME_MOUNT = "/mnt/task2"
REMOTE_CODE_ROOT = Path("/root/udsc2026")
P15_ROOT = Path(VOLUME_MOUNT) / "p15"
P15_RESULT_ROOT = P15_ROOT / "results"
P15_SUMMARY = P15_RESULT_ROOT / "p15_run_summary.json"
REMOTE_P14_ARCHIVE = "input/task2_p14_beam_input.tar.zst"
REMOTE_P14_MANIFEST = "input/task2_p14_beam_input.tar.manifest.json"
REMOTE_P15_BUNDLE = "p15/input/task2_p15_modal_input.zip"
REMOTE_P15_MANIFEST = "p15/input/task2_p15_modal_input.manifest.json"
REMOTE_RESULT = "p15/results/task2_p15_modal_result.zip"
DEFAULT_P14_ARCHIVE: Path
DEFAULT_P15_BUNDLE: Path
DEFAULT_DOWNLOAD_DIR: Path
MODAL_GPU_PRIORITY = ["H100"]
TORCH_VERSION = "2.10.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYTHON_PACKAGES = (
    "accelerate>=1.1,<2.0",
    "huggingface-hub>=1.3,<2.0",
    "nltk>=3.8,<4.0",
    "numpy==1.26.4",
    "peft>=0.17,<1.0",
    "pydantic>=2.8,<3.0",
    "pyyaml>=6.0",
    "rapidfuzz>=3.9,<4.0",
    "rouge-score>=0.1,<1.0",
    "safetensors>=0.5,<1.0",
    "scikit-learn>=1.5,<2.0",
    "sentencepiece>=0.2,<1.0",
    "tqdm>=4.66,<5.0",
    "transformers==5.0.0",
    "zstandard==0.23.0",
)


def _is_project_root(candidate: Path) -> bool:
    return (
        (candidate / "pyproject.toml").is_file()
        and (candidate / "scripts/cloud/beam_task2_p14.py").is_file()
        and (candidate / "scripts/cloud/modal_task2_p15.py").is_file()
        and (candidate / "src").is_dir()
    )


def _resolve_project_root() -> Path:
    module = Path(__file__).expanduser().resolve()
    remote = REMOTE_CODE_ROOT.resolve()
    candidates = [
        remote,
        module.parent,
        *module.parents,
        Path.cwd().resolve(),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen and _is_project_root(candidate):
            return candidate
        seen.add(key)
    raise RuntimeError("Cannot locate the Task 2 P15 project root")


PROJECT_ROOT = _resolve_project_root()
DEFAULT_P14_ARCHIVE = PROJECT_ROOT / "artifacts/task2/task2_p14_beam_input.tar.zst"
DEFAULT_P15_BUNDLE = PROJECT_ROOT / "artifacts/task2/task2_p15_modal_input.zip"
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "artifacts/task2/modal_p15_download"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_local_input(
    path: str | Path, expected_schema: str | None = None
) -> tuple[Path, Path, dict[str, Any]]:
    archive = Path(path).expanduser().resolve()
    manifest_path = archive.with_suffix(".manifest.json")
    if not archive.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"archive or manifest is missing: {archive}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be an object: {manifest_path}")
    if expected_schema and manifest.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected manifest schema: {manifest_path}")
    if int(manifest.get("archive_bytes", -1)) != archive.stat().st_size:
        raise ValueError(f"archive size mismatch: {archive}")
    if str(manifest.get("archive_sha256", "")) != _sha256(archive):
        raise ValueError(f"archive SHA-256 mismatch: {archive}")
    return archive, manifest_path, manifest


class _ProviderImageStub:
    def __init__(self, **_: Any) -> None:
        pass

    def add_python_packages(self, _: list[str]) -> _ProviderImageStub:
        return self


class _ProviderVolumeStub:
    def __init__(self, **_: Any) -> None:
        pass


class _ProviderTaskPolicyStub:
    def __init__(self, **_: Any) -> None:
        pass


class _ProviderHandlerStub:
    def __init__(self, handler: Callable[..., Any]) -> None:
        self._handler = handler
        self.parent = types.SimpleNamespace(handler=None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._handler(*args, **kwargs)

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._handler(*args, **kwargs)


def _provider_function_stub(**_: Any) -> Callable[[Callable[..., Any]], Any]:
    def decorate(handler: Callable[..., Any]) -> _ProviderHandlerStub:
        return _ProviderHandlerStub(handler)

    return decorate


def _load_pipeline() -> Any:
    os.environ["UDSC_TASK2_PROVIDER"] = "modal"
    os.environ["UDSC_BEAM_GPU"] = "H100,A100,L40S"
    for root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    fake_beam = types.ModuleType("beam")
    fake_beam.Image = _ProviderImageStub
    fake_beam.Volume = _ProviderVolumeStub
    fake_beam.function = _provider_function_stub
    fake_beta9 = types.ModuleType("beta9")
    fake_beta9.TaskPolicy = _ProviderTaskPolicyStub
    sys.modules["beam"] = fake_beam
    sys.modules["beta9"] = fake_beta9
    pipeline = importlib.import_module("scripts.cloud.beam_task2_p14")
    if pipeline.PROVIDER_NAME != "modal":
        raise RuntimeError("shared Task 2 core loaded with the wrong provider")
    return pipeline


def _read_volume_json(volume: Any, remote_path: str) -> dict[str, Any] | None:
    try:
        content = b"".join(volume.read_file(remote_path))
    except FileNotFoundError:
        return None
    if not content:
        return None
    payload = json.loads(content.decode("utf-8-sig"))
    return payload if isinstance(payload, dict) else None


def _p15_input_state() -> dict[str, Any]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P15_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P15_MANIFEST
    if not archive.is_file() or not manifest_path.is_file():
        return {"status": "P15_INPUT_MISSING"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = str(manifest.get("archive_sha256", ""))
    observed = _sha256(archive)
    return {
        "status": "P15_INPUT_READY" if expected == observed else "P15_INPUT_INVALID",
        "expected_sha256": expected,
        "observed_sha256": observed,
        "archive_bytes": archive.stat().st_size,
    }


def _extract_p15_bundle() -> tuple[Path, dict[str, Any]]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P15_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P15_MANIFEST
    state = _p15_input_state()
    if state["status"] != "P15_INPUT_READY":
        raise RuntimeError("P15 input is not verified: " + json.dumps(state))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = P15_ROOT / "extracted" / str(manifest["archive_sha256"])[:24]
    ready = root / ".ready.json"
    if ready.is_file():
        return root, manifest
    root.mkdir(parents=True, exist_ok=True)
    expected_members = {
        str(row["archive_path"]): row for row in manifest.get("members", [])
    }
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        if set(names) != set(expected_members):
            raise ValueError("P15 archive member contract mismatch")
        for info in bundle.infolist():
            destination = (root / info.filename).resolve()
            if root.resolve() not in destination.parents:
                raise ValueError(f"unsafe P15 member path: {info.filename}")
        bundle.extractall(root)
    for name, member in expected_members.items():
        path = root / name
        if path.stat().st_size != int(member["bytes"]):
            raise ValueError(f"P15 member size mismatch: {name}")
        if _sha256(path) != str(member["sha256"]):
            raise ValueError(f"P15 member SHA-256 mismatch: {name}")
    ready.write_text(
        json.dumps(
            {"schema_version": 1, "archive_sha256": manifest["archive_sha256"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, manifest


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _training_command(
    pipeline: Any,
    *,
    train_data: Path,
    base_model: Path,
    initial_adapter: Path,
    output: Path,
    batch_size: int,
    gradient_accumulation: int,
    smoke_only: bool = False,
) -> list[str]:
    command = pipeline._python(
        "scripts/training/finetune_task2_qwen_lora.py",
        "train",
        "--train-data",
        str(train_data),
        "--model-dir",
        str(base_model),
        "--initial-adapter",
        str(initial_adapter),
        "--output-dir",
        str(output),
        "--epochs",
        "1",
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation",
        str(gradient_accumulation),
        "--learning-rate",
        "1e-5",
        "--max-length",
        "3072",
        "--max-context-tokens",
        "1800",
        "--max-answer-tokens",
        "1024",
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--log-every",
        "10",
        "--resume",
    )
    if smoke_only:
        command.remove("--resume")
        command.append("--smoke-only")
    return command


def _generation_command(
    pipeline: Any,
    *,
    questions: str,
    rankings: Path,
    model: Path,
    base_model: Path,
    output: Path,
    diagnostics: Path,
    batch_size: int,
    question_ids: Path | None = None,
    max_new_tokens: int = 1024,
) -> list[str]:
    command = pipeline._python(
        "scripts/training/finetune_task2_qwen_lora.py",
        "generate",
        "--questions",
        questions,
        "--rankings",
        str(rankings),
        "--model-dir",
        str(model),
        "--base-model-dir",
        str(base_model),
        "--output",
        str(output),
        "--diagnostics",
        str(diagnostics),
        "--batch-size",
        str(batch_size),
        "--top-parents",
        "1",
        "--max-parent-words",
        "512",
        "--max-context-tokens",
        "1800",
        "--max-new-tokens",
        str(max_new_tokens),
        "--ce-weight",
        "0",
        "--retrieval-weight",
        "1",
        "--rrf-k",
        "0",
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--resume",
    )
    if question_ids is not None:
        command.extend(["--question-ids", str(question_ids)])
    return command


def _profile_command(
    pipeline: Any,
    *,
    extracted: Path,
    dev_qwen: Path,
    public_qwen: Path | None,
    output: Path,
) -> list[str]:
    command = pipeline._python(
        "scripts/evaluation/select_task2_p15_profile.py",
        "--dev-questions",
        "data/raw/btc/LegalQA/train.json",
        "--dev-ids",
        str(extracted / "training/adaptation_dev_ids.json"),
        "--dev-qwen",
        str(dev_qwen),
        "--dev-extractive",
        str(extracted / "predictions/dev_extractive.json"),
        "--output-dir",
        str(output),
    )
    if public_qwen is None:
        command.append("--selection-only")
    else:
        command.extend(
            [
                "--public-questions",
                "data/raw/btc/LegalQA/public-official.json",
                "--public-qwen",
                str(public_qwen),
                "--public-extractive",
                str(extracted / "predictions/public_extractive.json"),
            ]
        )
    return command


def _run_pipeline(*, smoke_only: bool) -> dict[str, Any]:
    pipeline = _load_pipeline()
    pipeline._extract_workspace()
    source_sha = pipeline._overlay_source()
    pipeline._download_qwen()
    gpu = pipeline._gpu_manifest()
    pipeline._require_free_vram(gpu)
    extracted, p15_manifest = _extract_p15_bundle()
    os.environ["NLTK_DATA"] = str(extracted / "nltk_data")
    total_gib = int(gpu["vram_bytes"]) / 1024**3
    batch_size = 4 if total_gib >= 70 else 2 if total_gib >= 40 else 1
    gradient_accumulation = 16 // batch_size
    generation_batch = 16 if total_gib >= 70 else 8 if total_gib >= 40 else 4
    local_base = pipeline._localize_directory(pipeline.QWEN_DIR, "p15_qwen_base")
    local_initial = pipeline._localize_directory(
        extracted / "initial_adapter", "p15_initial_adapter"
    )
    smoke_summary_path = P15_RESULT_ROOT / "smoke_summary.json"
    cached_smoke = (
        _read_json(smoke_summary_path) if smoke_summary_path.is_file() else {}
    )
    reuse_smoke = (
        cached_smoke.get("status") == "SMOKE_PASS"
        and cached_smoke.get("source_sha256") == source_sha
        and cached_smoke.get("p15_archive_sha256")
        == p15_manifest["archive_sha256"]
        and int(gpu["vram_bytes"]) >= int(cached_smoke.get("vram_bytes", 0))
    )
    if reuse_smoke:
        print("P15_SMOKE_ALREADY_VALID_FOR_THIS_GPU", flush=True)
    else:
        smoke_output = Path("/tmp/udsc-task2-p15-runtime/smoke_adapter")
        pipeline._run(
            _training_command(
                pipeline,
                train_data=extracted / "training/adaptation_train.jsonl",
                base_model=local_base,
                initial_adapter=local_initial,
                output=smoke_output,
                batch_size=batch_size,
                gradient_accumulation=gradient_accumulation,
                smoke_only=True,
            )
        )
        dev_ids = json.loads(
            (extracted / "training/adaptation_dev_ids.json").read_text(
                encoding="utf-8-sig"
            )
        )
        if not isinstance(dev_ids, list) or not dev_ids:
            raise ValueError("P15 smoke requires at least one development ID")
        smoke_ids = Path("/tmp/udsc-task2-p15-runtime/smoke_ids.json")
        _write_json(smoke_ids, [str(dev_ids[0])])
        pipeline._run(
            _generation_command(
                pipeline,
                questions="data/raw/btc/LegalQA/train.json",
                rankings=extracted / "rankings/dev_rankings.jsonl",
                model=smoke_output,
                base_model=local_base,
                output=Path(
                    "/tmp/udsc-task2-p15-runtime/smoke_predictions.json"
                ),
                diagnostics=Path(
                    "/tmp/udsc-task2-p15-runtime/smoke_diagnostics.json"
                ),
                batch_size=1,
                question_ids=smoke_ids,
                max_new_tokens=16,
            )
        )
        cached_smoke = {
            "schema_version": "task2-p15-modal-smoke-v1",
            "status": "SMOKE_PASS",
            "source_sha256": source_sha,
            "p15_archive_sha256": p15_manifest["archive_sha256"],
            "vram_bytes": int(gpu["vram_bytes"]),
            "gpu_name": gpu["name"],
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
        }
        _write_json(smoke_summary_path, cached_smoke)
    if smoke_only:
        return {**cached_smoke, "gpu": gpu, "reused": reuse_smoke}

    contract_payload = {
        "schema": "task2-p15-continuation-v1",
        "p15_archive_sha256": p15_manifest["archive_sha256"],
        "source_sha256": source_sha,
        "learning_rate": 1e-5,
        "epochs": 1,
        "effective_batch": 16,
    }
    contract = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True).encode()
    ).hexdigest()[:24]
    run_root = P15_RESULT_ROOT / "runs" / contract
    run_root.mkdir(parents=True, exist_ok=True)
    dev_adapter = run_root / "dev_adapter"
    pipeline._run(
        _training_command(
            pipeline,
            train_data=extracted / "training/adaptation_train.jsonl",
            base_model=local_base,
            initial_adapter=local_initial,
            output=dev_adapter,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
        )
    )
    dev_predictions = run_root / "dev_qwen_predictions.json"
    pipeline._run(
        _generation_command(
            pipeline,
            questions="data/raw/btc/LegalQA/train.json",
            rankings=extracted / "rankings/dev_rankings.jsonl",
            model=dev_adapter,
            base_model=local_base,
            output=dev_predictions,
            diagnostics=run_root / "dev_qwen_diagnostics.json",
            batch_size=generation_batch,
            question_ids=extracted / "training/adaptation_dev_ids.json",
        )
    )
    baseline_profile = run_root / "baseline_profile"
    baseline_qwen = extracted / "training/baseline_dev_predictions.json"
    pipeline._run(
        _profile_command(
            pipeline,
            extracted=extracted,
            dev_qwen=baseline_qwen,
            public_qwen=None,
            output=baseline_profile,
        )
    )
    dev_profile = run_root / "dev_profile"
    pipeline._run(
        _profile_command(
            pipeline,
            extracted=extracted,
            dev_qwen=dev_predictions,
            public_qwen=None,
            output=dev_profile,
        )
    )
    baseline_report = _read_json(baseline_profile / "profile_report.json")
    dev_report = _read_json(dev_profile / "profile_report.json")
    baseline_metrics = baseline_report["selected_metrics"]
    dev_metrics = dev_report["selected_metrics"]
    meteor_gain = float(dev_metrics["meteor"]) - float(baseline_metrics["meteor"])
    rouge_gain = float(dev_metrics["rouge_l"]) - float(baseline_metrics["rouge_l"])
    gate_pass = meteor_gain >= 0.003 and rouge_gain >= -0.015
    if not gate_pass:
        summary = {
            "schema_version": "task2-p15-modal-result-v1",
            "provider": "modal",
            "status": "HELDOUT_REJECTED",
            "contract": contract,
            "baseline_dev_metrics": baseline_metrics,
            "p15_dev_metrics": dev_metrics,
            "meteor_gain": meteor_gain,
            "rouge_l_gain": rouge_gain,
            "gate": {"minimum_meteor_gain": 0.003, "minimum_rouge_gain": -0.015},
            "gpu": gpu,
            "run_root": str(run_root),
        }
        _write_json(P15_SUMMARY, summary)
        _package_result(run_root, summary)
        return summary

    final_adapter = run_root / "final_adapter"
    pipeline._run(
        _training_command(
            pipeline,
            train_data=extracted / "training/adaptation_all.jsonl",
            base_model=local_base,
            initial_adapter=local_initial,
            output=final_adapter,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
        )
    )
    public_qwen = run_root / "public_qwen_predictions.json"
    pipeline._run(
        _generation_command(
            pipeline,
            questions="data/raw/btc/LegalQA/public-official.json",
            rankings=extracted / "rankings/public_rankings.jsonl",
            model=final_adapter,
            base_model=local_base,
            output=public_qwen,
            diagnostics=run_root / "public_qwen_diagnostics.json",
            batch_size=generation_batch,
        )
    )
    final_profile = run_root / "final_profile"
    pipeline._run(
        _profile_command(
            pipeline,
            extracted=extracted,
            dev_qwen=dev_predictions,
            public_qwen=public_qwen,
            output=final_profile,
        )
    )
    submission = run_root / "submission.zip"
    pipeline._run(
        pipeline._python(
            "scripts/submission/write_legal_qa_submission.py",
            "--input",
            str(final_profile / "public_predictions.json"),
            "--questions",
            "data/raw/btc/LegalQA/public-official.json",
            "--empty-answer-policy",
            "error",
            "--output",
            str(submission),
        )
    )
    pipeline._run(
        pipeline._python(
            "scripts/submission/validate_legal_qa_submission.py",
            "--input",
            str(submission),
            "--questions",
            "data/raw/btc/LegalQA/public-official.json",
        )
    )
    summary = {
        "schema_version": "task2-p15-modal-result-v1",
        "provider": "modal",
        "status": "PUBLIC_CANDIDATE_READY",
        "contract": contract,
        "baseline_dev_metrics": baseline_metrics,
        "p15_dev_metrics": dev_metrics,
        "meteor_gain": meteor_gain,
        "rouge_l_gain": rouge_gain,
        "selected_profile": dev_report["selected_profile"],
        "submission_sha256": _sha256(submission),
        "gpu": gpu,
        "run_root": str(run_root),
    }
    _write_json(P15_SUMMARY, summary)
    _package_result(run_root, summary)
    return summary


def _package_result(run_root: Path, summary: dict[str, Any]) -> Path:
    output = Path(VOLUME_MOUNT) / REMOTE_RESULT
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    include: list[Path] = [P15_SUMMARY]
    for relative in (
        "dev_qwen_predictions.json",
        "dev_qwen_diagnostics.json",
        "baseline_profile/profile_report.json",
        "dev_profile/profile_report.json",
        "final_profile/profile_report.json",
        "final_profile/public_predictions.json",
        "submission.zip",
        "final_adapter/adapter_config.json",
        "final_adapter/adapter_model.safetensors",
        "final_adapter/training_manifest.json",
    ):
        candidate = run_root / relative
        if candidate.is_file():
            include.append(candidate)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as bundle:
            for path in include:
                archive_name = (
                    "p15_run_summary.json"
                    if path == P15_SUMMARY
                    else path.relative_to(run_root).as_posix()
                )
                bundle.write(path, archive_name)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("P15 result bundle is corrupt")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    summary["result_archive"] = str(output)
    summary["result_archive_sha256"] = _sha256(output)
    _write_json(P15_SUMMARY, summary)
    return output


def _validate_result(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("downloaded P15 result is corrupt")
        if "p15_run_summary.json" not in bundle.namelist():
            raise ValueError("downloaded P15 result lacks its summary")
        summary = json.loads(bundle.read("p15_run_summary.json").decode("utf-8"))
    if summary.get("status") not in {"PUBLIC_CANDIDATE_READY", "HELDOUT_REJECTED"}:
        raise ValueError("downloaded P15 result has an invalid status")
    return summary


if modal is not None:
    app = modal.App(APP_NAME)
    task2_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    pipeline_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgomp1")
        .pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX_URL)
        .pip_install(*PYTHON_PACKAGES)
    )
    for _relative in ("scripts", "src", "configs"):
        _source = PROJECT_ROOT / _relative
        if _source.is_dir():
            pipeline_image = pipeline_image.add_local_dir(
                str(_source),
                remote_path=(REMOTE_CODE_ROOT / _relative).as_posix(),
                copy=True,
            )
    pipeline_image = pipeline_image.add_local_file(
        str(PROJECT_ROOT / "pyproject.toml"),
        remote_path=(REMOTE_CODE_ROOT / "pyproject.toml").as_posix(),
        copy=True,
    )

    @app.function(
        image=pipeline_image,
        cpu=2,
        memory=4096,
        timeout=20 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def input_probe() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()
        return {
            "p14": pipeline._input_state(verify_hash=True),
            "p15": _p15_input_state(),
        }

    @app.function(
        image=pipeline_image,
        cpu=4,
        memory=16384,
        timeout=60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def model_prepare() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()
        pipeline._download_qwen()
        task2_volume.commit()
        return {"status": "MODEL_READY", "repo": pipeline.QWEN_REPO}

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=45 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def smoke() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()
        try:
            with pipeline._exclusive_run():
                return _run_pipeline(smoke_only=True)
        finally:
            task2_volume.commit()

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=8 * 60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def run_all() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()
        try:
            with pipeline._exclusive_run():
                return _run_pipeline(smoke_only=False)
        finally:
            task2_volume.commit()

    def _upload_inputs(p14_archive: str, p15_bundle: str) -> None:
        p14, p14_manifest, p14_meta = _validated_local_input(p14_archive)
        p15, p15_manifest, p15_meta = _validated_local_input(
            p15_bundle, "task2-p15-modal-input-v1"
        )
        state = input_probe.remote()
        p14_ready = (
            state.get("p14", {}).get("status") == "INPUT_READY"
            and state["p14"].get("expected_sha256") == p14_meta["archive_sha256"]
        )
        p15_ready = (
            state.get("p15", {}).get("status") == "P15_INPUT_READY"
            and state["p15"].get("expected_sha256") == p15_meta["archive_sha256"]
        )
        if p14_ready and p15_ready:
            print("MODAL_P15_INPUTS_ALREADY_READY", flush=True)
            return
        with task2_volume.batch_upload(force=True) as batch:
            if not p14_ready:
                batch.put_file(str(p14), f"/{REMOTE_P14_ARCHIVE}")
                batch.put_file(str(p14_manifest), f"/{REMOTE_P14_MANIFEST}")
            if not p15_ready:
                batch.put_file(str(p15), f"/{REMOTE_P15_BUNDLE}")
                batch.put_file(str(p15_manifest), f"/{REMOTE_P15_MANIFEST}")
        verified = input_probe.remote()
        if verified.get("p14", {}).get("status") != "INPUT_READY":
            raise RuntimeError("Modal P14 input verification failed")
        if verified.get("p15", {}).get("status") != "P15_INPUT_READY":
            raise RuntimeError("Modal P15 input verification failed")

    def _download(download_dir: str) -> tuple[Path, dict[str, Any]]:
        destination = Path(download_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "task2_p15_modal_result.zip"
        temporary = output.with_name(f".{output.name}.partial")
        try:
            with temporary.open("wb") as stream:
                for chunk in task2_volume.read_file(REMOTE_RESULT):
                    stream.write(chunk)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        summary = _validate_result(output)
        if summary["status"] == "PUBLIC_CANDIDATE_READY":
            with zipfile.ZipFile(output) as bundle:
                content = bundle.read("submission.zip")
            with zipfile.ZipFile(io.BytesIO(content)) as submission:
                if submission.namelist() != ["submission.json"]:
                    raise ValueError("P15 submission contract is invalid")
            (destination / "submission.zip").write_bytes(content)
            print(f"MODAL_P15_SUBMISSION_READY={destination / 'submission.zip'}")
        print("MODAL_P15_RESULT=" + json.dumps(summary, ensure_ascii=False))
        return output, summary

    @app.local_entrypoint()
    def main(
        stage: str = "run",
        p14_archive: str = str(DEFAULT_P14_ARCHIVE),
        p15_bundle: str = str(DEFAULT_P15_BUNDLE),
        download_dir: str = str(DEFAULT_DOWNLOAD_DIR),
    ) -> None:
        selected = stage.strip().lower()
        if selected == "check":
            _validated_local_input(p14_archive)
            _validated_local_input(p15_bundle, "task2-p15-modal-input-v1")
            print(
                json.dumps(
                    {
                        "status": "LOCAL_CHECK_PASS",
                        "app": APP_NAME,
                        "gpu_priority": MODAL_GPU_PRIORITY,
                    },
                    indent=2,
                )
            )
            return
        if selected == "status":
            payload = _read_volume_json(task2_volume, REMOTE_RESULT.replace(
                "task2_p15_modal_result.zip", "p15_run_summary.json"
            ))
            print("MODAL_P15_STATUS=" + json.dumps(payload or {"status": "NO_RESULT"}))
            return
        if selected == "download":
            _download(download_dir)
            return
        if selected not in {"prepare", "smoke", "run"}:
            raise ValueError(
                "stage must be check, prepare, smoke, run, status, or download"
            )
        _upload_inputs(p14_archive, p15_bundle)
        model = model_prepare.remote()
        if model.get("status") != "MODEL_READY":
            raise RuntimeError("Modal P15 model preparation failed")
        if selected == "prepare":
            print("MODAL_P15_PREPARE_COMPLETE")
            return
        if selected == "smoke":
            result = smoke.remote()
            if result.get("status") != "SMOKE_PASS":
                raise RuntimeError("Modal P15 smoke failed: " + json.dumps(result))
            print("MODAL_P15_SMOKE_COMPLETE")
            return
        result = run_all.remote()
        if result.get("status") not in {"PUBLIC_CANDIDATE_READY", "HELDOUT_REJECTED"}:
            raise RuntimeError("Modal P15 run failed: " + json.dumps(result))
        _download(download_dir)

else:
    app = None
    task2_volume = None


if __name__ == "__main__" and modal is None:
    raise SystemExit(
        "Modal SDK is missing. Install it with: python -m pip install -U modal"
    )
