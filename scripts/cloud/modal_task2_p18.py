"""Run the gated P18 length-normalized DPO experiment on Modal H100."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:
    modal = None  # type: ignore[assignment]

_module_path = Path(__file__).resolve()
for _root in (
    Path("/root/udsc2026"),
    Path.cwd().resolve(),
    _module_path.parent,
    *_module_path.parents,
):
    if (_root / "scripts").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.cloud import modal_task2_p15 as core  # noqa: E402

APP_NAME = "udsc-task2-p18-modal"
VOLUME_NAME = core.VOLUME_NAME
VOLUME_MOUNT = core.VOLUME_MOUNT
P18_ROOT = Path(VOLUME_MOUNT) / "p18"
P18_RESULT_ROOT = P18_ROOT / "results"
P18_SUMMARY = P18_RESULT_ROOT / "p18_run_summary.json"
REMOTE_P18_BUNDLE = "p18/input/task2_p18_modal_input.zip"
REMOTE_P18_MANIFEST = "p18/input/task2_p18_modal_input.manifest.json"
REMOTE_RESULT = "p18/results/task2_p18_modal_result.zip"
DEFAULT_P18_BUNDLE = core.PROJECT_ROOT / "artifacts/task2/task2_p18_modal_input.zip"
DEFAULT_DOWNLOAD_DIR = core.PROJECT_ROOT / "artifacts/task2/modal_p18_download"
MODAL_GPU_PRIORITY = ["H100"]

EXPECTED_TRAIN_PREFERENCES = 780
EXPECTED_ALL_PREFERENCES = 995
MINIMUM_METEOR_GAIN = 0.010
MINIMUM_DEV_METEOR = 0.605
MINIMUM_ROUGE_GAIN = -0.015
DPO_CONFIG = {
    "epochs": 1,
    "batch_size": 2,
    "reference_batch_size": 4,
    "gradient_accumulation": 8,
    "learning_rate": 5e-6,
    "beta": 0.2,
    "sft_weight": 0.1,
    "max_length": 2560,
    "max_context_tokens": 1500,
    "max_answer_tokens": 768,
}


def _p18_input_state() -> dict[str, Any]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P18_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P18_MANIFEST
    if not archive.is_file() or not manifest_path.is_file():
        return {"status": "P18_INPUT_MISSING"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = str(manifest.get("archive_sha256", ""))
    observed = core._sha256(archive)
    return {
        "status": "P18_INPUT_READY" if expected == observed else "P18_INPUT_INVALID",
        "expected_sha256": expected,
        "observed_sha256": observed,
        "archive_bytes": archive.stat().st_size,
    }


def _extract_p18_bundle() -> tuple[Path, dict[str, Any]]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P18_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P18_MANIFEST
    state = _p18_input_state()
    if state["status"] != "P18_INPUT_READY":
        raise RuntimeError("P18 input is not verified: " + json.dumps(state))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = P18_ROOT / "extracted" / str(manifest["archive_sha256"])[:24]
    ready = root / ".ready.json"
    if ready.is_file():
        return root, manifest
    root.mkdir(parents=True, exist_ok=True)
    expected = {str(row["archive_path"]): row for row in manifest.get("members", [])}
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        if set(names) != set(expected):
            raise ValueError("P18 archive member contract mismatch")
        for info in bundle.infolist():
            destination = (root / info.filename).resolve()
            if root.resolve() not in destination.parents:
                raise ValueError(f"unsafe P18 member path: {info.filename}")
        bundle.extractall(root)
    for name, row in expected.items():
        path = root / name
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"P18 member size mismatch: {name}")
        if core._sha256(path) != str(row["sha256"]):
            raise ValueError(f"P18 member hash mismatch: {name}")
    core._write_json(
        ready,
        {"schema_version": 1, "archive_sha256": manifest["archive_sha256"]},
    )
    return root, manifest


def _contract(p15_hash: str, p18_hash: str, source_hash: str) -> str:
    payload = {
        "schema": "task2-p18-length-normalized-dpo-v1",
        "p15_hash": p15_hash,
        "p18_hash": p18_hash,
        "source_hash": source_hash,
        "dpo": DPO_CONFIG,
        "gate": {
            "minimum_meteor_gain": MINIMUM_METEOR_GAIN,
            "minimum_dev_meteor": MINIMUM_DEV_METEOR,
            "minimum_rouge_gain": MINIMUM_ROUGE_GAIN,
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _training_command(
    pipeline: Any,
    *,
    preferences: Path,
    base_model: Path,
    initial_adapter: Path,
    output: Path,
) -> list[str]:
    return pipeline._python(
        "scripts/training/train_task2_p18_dpo.py",
        "--preference-data",
        str(preferences),
        "--model-dir",
        str(base_model),
        "--initial-adapter",
        str(initial_adapter),
        "--output-dir",
        str(output),
        "--epochs",
        str(DPO_CONFIG["epochs"]),
        "--batch-size",
        str(DPO_CONFIG["batch_size"]),
        "--reference-batch-size",
        str(DPO_CONFIG["reference_batch_size"]),
        "--gradient-accumulation",
        str(DPO_CONFIG["gradient_accumulation"]),
        "--learning-rate",
        str(DPO_CONFIG["learning_rate"]),
        "--beta",
        str(DPO_CONFIG["beta"]),
        "--sft-weight",
        str(DPO_CONFIG["sft_weight"]),
        "--max-length",
        str(DPO_CONFIG["max_length"]),
        "--max-context-tokens",
        str(DPO_CONFIG["max_context_tokens"]),
        "--max-answer-tokens",
        str(DPO_CONFIG["max_answer_tokens"]),
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--log-every",
        "5",
    )


def _generation_command(
    pipeline: Any,
    *,
    questions: str,
    rankings: Path,
    adapter: Path,
    base_model: Path,
    output: Path,
    diagnostics: Path,
    question_ids: Path | None = None,
) -> list[str]:
    return core._generation_command(
        pipeline,
        questions=questions,
        rankings=rankings,
        model=adapter,
        base_model=base_model,
        output=output,
        diagnostics=diagnostics,
        batch_size=16,
        question_ids=question_ids,
        max_new_tokens=1024,
    )


def _profile_command(
    pipeline: Any,
    *,
    p15_inputs: Path,
    dev_qwen: Path,
    output: Path,
    public_qwen: Path | None = None,
) -> list[str]:
    command = pipeline._python(
        "scripts/evaluation/select_task2_p15_profile.py",
        "--dev-questions",
        "data/raw/btc/LegalQA/train.json",
        "--dev-ids",
        str(p15_inputs / "training/adaptation_dev_ids.json"),
        "--dev-qwen",
        str(dev_qwen),
        "--dev-extractive",
        str(p15_inputs / "predictions/dev_extractive.json"),
        "--output-dir",
        str(output),
        "--meteor-weight",
        "1",
        "--rouge-weight",
        "0",
        "--maximum-rouge-drop",
        "1",
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
                str(p15_inputs / "predictions/public_extractive.json"),
            ]
        )
    return command


def _preference_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                ids.add(str(row["id"]))
    return ids


def _validate_preference_split(p18_inputs: Path) -> dict[str, int]:
    from scripts.training import train_task2_p18_dpo as trainer

    train_path = p18_inputs / "preferences/train.jsonl"
    all_path = p18_inputs / "preferences/all.jsonl"
    train_rows = trainer._load_preferences(train_path)
    all_rows = trainer._load_preferences(all_path)
    if len(train_rows) != EXPECTED_TRAIN_PREFERENCES:
        raise ValueError(
            f"expected {EXPECTED_TRAIN_PREFERENCES} train preferences, "
            f"found {len(train_rows)}"
        )
    if len(all_rows) != EXPECTED_ALL_PREFERENCES:
        raise ValueError(
            f"expected {EXPECTED_ALL_PREFERENCES} all preferences, "
            f"found {len(all_rows)}"
        )
    train_ids = _preference_ids(train_path)
    all_ids = _preference_ids(all_path)
    if not train_ids < all_ids:
        raise ValueError("P18 train preference IDs are not a strict subset")
    return {"train": len(train_rows), "all": len(all_rows)}


def _prepare_runtime() -> tuple[
    Any,
    Path,
    Path,
    Path,
    dict[str, Any],
    str,
    str,
    str,
    Path,
]:
    pipeline = core._load_pipeline()
    pipeline._extract_workspace()
    source_hash = pipeline._overlay_source()
    pipeline._download_qwen()
    gpu = pipeline._gpu_manifest()
    pipeline._require_free_vram(gpu)
    p15_inputs, p15_manifest = core._extract_p15_bundle()
    p18_inputs, p18_manifest = _extract_p18_bundle()
    os.environ["NLTK_DATA"] = str(p15_inputs / "nltk_data")
    local_base = pipeline._localize_directory(pipeline.QWEN_DIR, "p18_qwen_base")
    local_adapter = pipeline._localize_directory(
        p15_inputs / "initial_adapter", "p18_initial_adapter"
    )
    return (
        pipeline,
        p15_inputs,
        p18_inputs,
        local_base,
        gpu,
        source_hash,
        str(p15_manifest["archive_sha256"]),
        str(p18_manifest["archive_sha256"]),
        local_adapter,
    )


def _run_all() -> dict[str, Any]:
    runtime = _prepare_runtime()
    (
        pipeline,
        p15_inputs,
        p18_inputs,
        local_base,
        gpu,
        source_hash,
        p15_hash,
        p18_hash,
        local_adapter,
    ) = runtime
    contract = _contract(p15_hash, p18_hash, source_hash)
    run_root = P18_RESULT_ROOT / "runs" / contract
    run_root.mkdir(parents=True, exist_ok=True)

    baseline_profile = run_root / "baseline_profile"
    if not (baseline_profile / "profile_report.json").is_file():
        pipeline._run(
            _profile_command(
                pipeline,
                p15_inputs=p15_inputs,
                dev_qwen=p15_inputs / "training/baseline_dev_predictions.json",
                output=baseline_profile,
            )
        )

    dev_adapter = run_root / "dev_adapter"
    pipeline._run(
        _training_command(
            pipeline,
            preferences=p18_inputs / "preferences/train.jsonl",
            base_model=local_base,
            initial_adapter=local_adapter,
            output=dev_adapter,
        )
    )
    dev_qwen = run_root / "dev_qwen_predictions.json"
    pipeline._run(
        _generation_command(
            pipeline,
            questions="data/raw/btc/LegalQA/train.json",
            rankings=p15_inputs / "rankings/dev_rankings.jsonl",
            adapter=dev_adapter,
            base_model=local_base,
            output=dev_qwen,
            diagnostics=run_root / "dev_qwen_diagnostics.json",
            question_ids=p15_inputs / "training/adaptation_dev_ids.json",
        )
    )
    dev_profile = run_root / "dev_profile"
    if not (dev_profile / "profile_report.json").is_file():
        pipeline._run(
            _profile_command(
                pipeline,
                p15_inputs=p15_inputs,
                dev_qwen=dev_qwen,
                output=dev_profile,
            )
        )

    baseline_report = core._read_json(baseline_profile / "profile_report.json")
    dev_report = core._read_json(dev_profile / "profile_report.json")
    baseline_metrics = baseline_report["selected_metrics"]
    dev_metrics = dev_report["selected_metrics"]
    meteor_gain = float(dev_metrics["meteor"]) - float(baseline_metrics["meteor"])
    rouge_gain = float(dev_metrics["rouge_l"]) - float(baseline_metrics["rouge_l"])
    gate_pass = (
        meteor_gain >= MINIMUM_METEOR_GAIN
        and float(dev_metrics["meteor"]) >= MINIMUM_DEV_METEOR
        and rouge_gain >= MINIMUM_ROUGE_GAIN
    )
    summary: dict[str, Any] = {
        "schema_version": "task2-p18-modal-result-v1",
        "provider": "modal",
        "status": "DEV_PASS" if gate_pass else "DEV_REJECTED",
        "contract": contract,
        "baseline_dev_profile": baseline_report["selected_profile"],
        "baseline_dev_metrics": baseline_metrics,
        "p18_dev_profile": dev_report["selected_profile"],
        "p18_dev_metrics": dev_metrics,
        "meteor_gain": meteor_gain,
        "rouge_l_gain": rouge_gain,
        "gate": {
            "minimum_meteor_gain": MINIMUM_METEOR_GAIN,
            "minimum_dev_meteor": MINIMUM_DEV_METEOR,
            "minimum_rouge_gain": MINIMUM_ROUGE_GAIN,
        },
        "dpo_config": DPO_CONFIG,
        "gpu": gpu,
        "run_root": str(run_root),
    }
    if not gate_pass:
        core._write_json(P18_SUMMARY, summary)
        _package_result(run_root, summary)
        return summary

    final_adapter = run_root / "final_adapter"
    pipeline._run(
        _training_command(
            pipeline,
            preferences=p18_inputs / "preferences/all.jsonl",
            base_model=local_base,
            initial_adapter=local_adapter,
            output=final_adapter,
        )
    )
    public_qwen = run_root / "public_qwen_predictions.json"
    pipeline._run(
        _generation_command(
            pipeline,
            questions="data/raw/btc/LegalQA/public-official.json",
            rankings=p15_inputs / "rankings/public_rankings.jsonl",
            adapter=final_adapter,
            base_model=local_base,
            output=public_qwen,
            diagnostics=run_root / "public_qwen_diagnostics.json",
        )
    )
    final_profile = run_root / "final_profile"
    pipeline._run(
        _profile_command(
            pipeline,
            p15_inputs=p15_inputs,
            dev_qwen=dev_qwen,
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
    summary.update(
        {
            "status": "PUBLIC_CANDIDATE_READY",
            "selected_profile": dev_report["selected_profile"],
            "submission_sha256": core._sha256(submission),
        }
    )
    core._write_json(P18_SUMMARY, summary)
    _package_result(run_root, summary)
    return summary


def _package_result(run_root: Path, summary: dict[str, Any]) -> Path:
    output = Path(VOLUME_MOUNT) / REMOTE_RESULT
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    include: list[Path] = [P18_SUMMARY]
    relatives = (
        "dev_qwen_predictions.json",
        "dev_qwen_diagnostics.json",
        "baseline_profile/profile_report.json",
        "dev_profile/profile_report.json",
        "dev_adapter/adapter_config.json",
        "dev_adapter/adapter_model.safetensors",
        "dev_adapter/training_manifest.json",
        "public_qwen_predictions.json",
        "public_qwen_diagnostics.json",
        "final_profile/profile_report.json",
        "final_profile/public_predictions.json",
        "submission.zip",
        "final_adapter/adapter_config.json",
        "final_adapter/adapter_model.safetensors",
        "final_adapter/training_manifest.json",
    )
    include.extend(run_root / name for name in relatives if (run_root / name).is_file())
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as bundle:
            for path in include:
                name = (
                    "p18_run_summary.json"
                    if path == P18_SUMMARY
                    else path.relative_to(run_root).as_posix()
                )
                bundle.write(path, name)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("P18 result archive is corrupt")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    summary["result_archive"] = str(output)
    summary["result_archive_sha256"] = core._sha256(output)
    core._write_json(P18_SUMMARY, summary)
    return output


def _validate_result(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("downloaded P18 result is corrupt")
        summary = json.loads(bundle.read("p18_run_summary.json").decode("utf-8"))
    if summary.get("status") not in {"DEV_REJECTED", "PUBLIC_CANDIDATE_READY"}:
        raise ValueError("downloaded P18 result has an invalid status")
    return summary


if modal is not None:
    app = modal.App(APP_NAME)
    task2_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    pipeline_image = core.pipeline_image

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
        pipeline = core._load_pipeline()
        return {
            "p14": pipeline._input_state(verify_hash=True),
            "p15": core._p15_input_state(),
            "p18": _p18_input_state(),
        }

    @app.function(
        image=pipeline_image,
        cpu=4,
        memory=16384,
        timeout=60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def prepare_inputs() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = core._load_pipeline()
        pipeline._extract_workspace()
        source_hash = pipeline._overlay_source()
        p15_inputs, p15_manifest = core._extract_p15_bundle()
        p18_inputs, p18_manifest = _extract_p18_bundle()
        counts = _validate_preference_split(p18_inputs)
        adapter = p15_inputs / "initial_adapter"
        required = (
            adapter / "adapter_config.json",
            adapter / "adapter_model.safetensors",
            p15_inputs / "training/adaptation_dev_ids.json",
            p15_inputs / "training/baseline_dev_predictions.json",
            p15_inputs / "rankings/dev_rankings.jsonl",
            p15_inputs / "rankings/public_rankings.jsonl",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "P18 prerequisite is missing: " + ", ".join(missing)
            )
        return {
            "status": "P18_PREPARED",
            "source_sha256": source_hash,
            "p15_archive_sha256": p15_manifest["archive_sha256"],
            "p18_archive_sha256": p18_manifest["archive_sha256"],
            "preferences": counts,
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
        pipeline = core._load_pipeline()
        pipeline._download_qwen()
        task2_volume.commit()
        return {"status": "MODEL_READY", "repo": pipeline.QWEN_REPO}

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
        pipeline = core._load_pipeline()
        try:
            with pipeline._exclusive_run():
                return _run_all()
        finally:
            task2_volume.commit()

    def _upload_inputs(p14_archive: str, p15_bundle: str, p18_bundle: str) -> None:
        p14, p14_manifest, p14_meta = core._validated_local_input(p14_archive)
        p15, p15_manifest, p15_meta = core._validated_local_input(
            p15_bundle, "task2-p15-modal-input-v1"
        )
        p18, p18_manifest, p18_meta = core._validated_local_input(
            p18_bundle, "task2-p18-modal-input-v1"
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
        p18_ready = (
            state.get("p18", {}).get("status") == "P18_INPUT_READY"
            and state["p18"].get("expected_sha256") == p18_meta["archive_sha256"]
        )
        if p14_ready and p15_ready and p18_ready:
            print("MODAL_P18_INPUTS_ALREADY_READY", flush=True)
            return
        with task2_volume.batch_upload(force=True) as batch:
            if not p14_ready:
                batch.put_file(str(p14), f"/{core.REMOTE_P14_ARCHIVE}")
                batch.put_file(str(p14_manifest), f"/{core.REMOTE_P14_MANIFEST}")
            if not p15_ready:
                batch.put_file(str(p15), f"/{core.REMOTE_P15_BUNDLE}")
                batch.put_file(str(p15_manifest), f"/{core.REMOTE_P15_MANIFEST}")
            if not p18_ready:
                batch.put_file(str(p18), f"/{REMOTE_P18_BUNDLE}")
                batch.put_file(str(p18_manifest), f"/{REMOTE_P18_MANIFEST}")
        verified = input_probe.remote()
        expected_statuses = {
            "p14": "INPUT_READY",
            "p15": "P15_INPUT_READY",
            "p18": "P18_INPUT_READY",
        }
        for name, status in expected_statuses.items():
            if verified.get(name, {}).get("status") != status:
                raise RuntimeError(f"Modal {name.upper()} input verification failed")

    def _download(download_dir: str) -> tuple[Path, dict[str, Any]]:
        destination = Path(download_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "task2_p18_modal_result.zip"
        partial = output.with_name(f".{output.name}.partial")
        try:
            with partial.open("wb") as stream:
                for chunk in task2_volume.read_file(REMOTE_RESULT):
                    stream.write(chunk)
            os.replace(partial, output)
        finally:
            partial.unlink(missing_ok=True)
        summary = _validate_result(output)
        if summary["status"] == "PUBLIC_CANDIDATE_READY":
            with zipfile.ZipFile(output) as result:
                content = result.read("submission.zip")
            with zipfile.ZipFile(io.BytesIO(content)) as submission:
                if submission.namelist() != ["submission.json"]:
                    raise ValueError("P18 submission contract is invalid")
            submission_path = destination / "submission.zip"
            submission_path.write_bytes(content)
            print(f"MODAL_P18_SUBMISSION_READY={submission_path}")
        print("MODAL_P18_RESULT=" + json.dumps(summary, ensure_ascii=False))
        return output, summary

    @app.local_entrypoint()
    def main(
        stage: str = "run",
        p14_archive: str = str(core.DEFAULT_P14_ARCHIVE),
        p15_bundle: str = str(core.DEFAULT_P15_BUNDLE),
        p18_bundle: str = str(DEFAULT_P18_BUNDLE),
        download_dir: str = str(DEFAULT_DOWNLOAD_DIR),
    ) -> None:
        selected = stage.strip().lower()
        if selected == "check":
            core._validated_local_input(p14_archive)
            core._validated_local_input(p15_bundle, "task2-p15-modal-input-v1")
            core._validated_local_input(p18_bundle, "task2-p18-modal-input-v1")
            print(
                json.dumps(
                    {
                        "status": "LOCAL_CHECK_PASS",
                        "app": APP_NAME,
                        "gpu_priority": MODAL_GPU_PRIORITY,
                        "dpo": DPO_CONFIG,
                    },
                    indent=2,
                )
            )
            return
        if selected == "status":
            payload = core._read_volume_json(
                task2_volume, "p18/results/p18_run_summary.json"
            )
            print("MODAL_P18_STATUS=" + json.dumps(payload or {"status": "NO_RESULT"}))
            return
        if selected == "download":
            _download(download_dir)
            return
        if selected not in {"prepare", "run"}:
            raise ValueError("stage must be check, prepare, run, status, or download")
        _upload_inputs(p14_archive, p15_bundle, p18_bundle)
        prepared = prepare_inputs.remote()
        if prepared.get("status") != "P18_PREPARED":
            raise RuntimeError("Modal P18 preparation failed")
        model = model_prepare.remote()
        if model.get("status") != "MODEL_READY":
            raise RuntimeError("Modal P18 model preparation failed")
        if selected == "prepare":
            print("MODAL_P18_PREPARE_COMPLETE")
            return
        result = run_all.remote()
        if result.get("status") not in {"DEV_REJECTED", "PUBLIC_CANDIDATE_READY"}:
            raise RuntimeError("Modal P18 run failed: " + json.dumps(result))
        _download(download_dir)

else:
    app = None
    task2_volume = None


if __name__ == "__main__" and modal is None:
    raise SystemExit(
        "Modal SDK is missing. Install it with: python -m pip install -U modal"
    )
