"""Run the gated Task 2 P16 retrieval/generation sweep on Modal H100."""

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

# Modal imports the entrypoint itself from /root while repository code is
# mounted at /root/udsc2026. Make the shared P15 core importable before any
# decorated remote function is hydrated.
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

APP_NAME = "udsc-task2-p16-modal"
VOLUME_NAME = core.VOLUME_NAME
VOLUME_MOUNT = core.VOLUME_MOUNT
P16_ROOT = Path(VOLUME_MOUNT) / "p16"
P16_RESULT_ROOT = P16_ROOT / "results"
P16_SUMMARY = P16_RESULT_ROOT / "p16_run_summary.json"
REMOTE_P16_BUNDLE = "p16/input/task2_p16_modal_input.zip"
REMOTE_P16_MANIFEST = "p16/input/task2_p16_modal_input.manifest.json"
REMOTE_RESULT = "p16/results/task2_p16_modal_result.zip"
DEFAULT_P16_BUNDLE = core.PROJECT_ROOT / "artifacts/task2/task2_p16_modal_input.zip"
DEFAULT_DOWNLOAD_DIR = core.PROJECT_ROOT / "artifacts/task2/modal_p16_download"
MODAL_GPU_PRIORITY = ["H100"]

# P14 already generated adapter_top2_fusion. Every paid experiment below
# changes one meaningful variable and is screened on the same untouched IDs.
DEV_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "base_top2_fusion",
        "model": "base",
        "top_parents": 2,
        "max_parent_words": 512,
        "max_context_tokens": 1800,
        "ce_weight": 0.30,
        "retrieval_weight": 0.70,
    },
    {
        "name": "base_top3_fusion",
        "model": "base",
        "top_parents": 3,
        "max_parent_words": 384,
        "max_context_tokens": 1800,
        "ce_weight": 0.30,
        "retrieval_weight": 0.70,
    },
    {
        "name": "adapter_top3_fusion",
        "model": "adapter",
        "top_parents": 3,
        "max_parent_words": 384,
        "max_context_tokens": 1800,
        "ce_weight": 0.30,
        "retrieval_weight": 0.70,
    },
    {
        "name": "adapter_top3_ce",
        "model": "adapter",
        "top_parents": 3,
        "max_parent_words": 384,
        "max_context_tokens": 1800,
        "ce_weight": 1.0,
        "retrieval_weight": 0.0,
    },
)


def _p16_input_state() -> dict[str, Any]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P16_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P16_MANIFEST
    if not archive.is_file() or not manifest_path.is_file():
        return {"status": "P16_INPUT_MISSING"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = str(manifest.get("archive_sha256", ""))
    observed = core._sha256(archive)
    return {
        "status": "P16_INPUT_READY" if expected == observed else "P16_INPUT_INVALID",
        "expected_sha256": expected,
        "observed_sha256": observed,
        "archive_bytes": archive.stat().st_size,
    }


def _extract_p16_bundle() -> tuple[Path, dict[str, Any]]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P16_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P16_MANIFEST
    state = _p16_input_state()
    if state["status"] != "P16_INPUT_READY":
        raise RuntimeError("P16 input is not verified: " + json.dumps(state))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = P16_ROOT / "extracted" / str(manifest["archive_sha256"])[:24]
    ready = root / ".ready.json"
    if ready.is_file():
        return root, manifest
    root.mkdir(parents=True, exist_ok=True)
    expected = {str(row["archive_path"]): row for row in manifest.get("members", [])}
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        if set(names) != set(expected):
            raise ValueError("P16 archive member contract mismatch")
        for info in bundle.infolist():
            destination = (root / info.filename).resolve()
            if root.resolve() not in destination.parents:
                raise ValueError(f"unsafe P16 member path: {info.filename}")
        bundle.extractall(root)
    for name, row in expected.items():
        path = root / name
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"P16 member size mismatch: {name}")
        if core._sha256(path) != str(row["sha256"]):
            raise ValueError(f"P16 member hash mismatch: {name}")
    core._write_json(
        ready,
        {"schema_version": 1, "archive_sha256": manifest["archive_sha256"]},
    )
    return root, manifest


def _generation_command(
    pipeline: Any,
    *,
    questions: str,
    rankings: Path,
    model: Path,
    base_model: Path,
    output: Path,
    diagnostics: Path,
    config: dict[str, Any],
    question_ids: Path | None = None,
    known_answers: str | None = None,
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
        "16",
        "--top-parents",
        str(config["top_parents"]),
        "--max-parent-words",
        str(config["max_parent_words"]),
        "--max-context-tokens",
        str(config["max_context_tokens"]),
        "--max-new-tokens",
        "1024",
        "--ce-weight",
        str(config["ce_weight"]),
        "--retrieval-weight",
        str(config["retrieval_weight"]),
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
    if known_answers is not None:
        command.extend(["--known-answers", known_answers])
    return command


def _profile_command(
    pipeline: Any,
    *,
    question_ids: Path,
    qwen: Path,
    extractive: Path,
    output: Path,
    public_qwen: Path | None = None,
    public_extractive: Path | None = None,
) -> list[str]:
    command = pipeline._python(
        "scripts/evaluation/select_task2_p15_profile.py",
        "--dev-questions",
        "data/raw/btc/LegalQA/train.json",
        "--dev-ids",
        str(question_ids),
        "--dev-qwen",
        str(qwen),
        "--dev-extractive",
        str(extractive),
        "--output-dir",
        str(output),
        "--meteor-weight",
        "1",
        "--rouge-weight",
        "0",
        "--maximum-rouge-drop",
        "1",
    )
    if public_qwen is None or public_extractive is None:
        command.append("--selection-only")
    else:
        command.extend(
            [
                "--public-questions",
                "data/raw/btc/LegalQA/public-official.json",
                "--public-qwen",
                str(public_qwen),
                "--public-extractive",
                str(public_extractive),
            ]
        )
    return command


def _read_report(path: Path) -> dict[str, Any]:
    return core._read_json(path / "profile_report.json")


def _contract(p16_hash: str, source_hash: str) -> str:
    payload = {
        "schema": "task2-p16-rag-sweep-v1",
        "p16_hash": p16_hash,
        "source_hash": source_hash,
        "experiments": DEV_EXPERIMENTS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _prepare_runtime() -> tuple[Any, Path, Path, Path, dict[str, Any], str]:
    pipeline = core._load_pipeline()
    pipeline._extract_workspace()
    source_hash = pipeline._overlay_source()
    pipeline._download_qwen()
    gpu = pipeline._gpu_manifest()
    pipeline._require_free_vram(gpu)
    p15_root, _ = core._extract_p15_bundle()
    p16_root, p16_manifest = _extract_p16_bundle()
    os.environ["NLTK_DATA"] = str(p15_root / "nltk_data")
    local_base = pipeline._localize_directory(pipeline.QWEN_DIR, "p16_qwen_base")
    local_adapter = pipeline._localize_directory(
        p15_root / "initial_adapter", "p16_initial_adapter"
    )
    return pipeline, p16_root, local_base, local_adapter, gpu, source_hash


def _run_dev() -> dict[str, Any]:
    pipeline, inputs, local_base, local_adapter, gpu, source_hash = _prepare_runtime()
    manifest = json.loads(
        (Path(VOLUME_MOUNT) / REMOTE_P16_MANIFEST).read_text(encoding="utf-8-sig")
    )
    contract = _contract(str(manifest["archive_sha256"]), source_hash)
    run_root = P16_RESULT_ROOT / "runs" / contract
    run_root.mkdir(parents=True, exist_ok=True)
    dev_ids = inputs / "training/dev_ids.json"
    heldout_rankings = inputs / "rankings/heldout_rankings.jsonl"
    extractive = inputs / "predictions/heldout_extractive.json"

    baseline_profile = run_root / "dev_baseline_profile"
    if not (baseline_profile / "profile_report.json").is_file():
        pipeline._run(
            _profile_command(
                pipeline,
                question_ids=dev_ids,
                qwen=inputs / "predictions/heldout_qwen.json",
                extractive=extractive,
                output=baseline_profile,
            )
        )
    baseline_report = _read_report(baseline_profile)
    experiment_rows: list[dict[str, Any]] = []
    for config in DEV_EXPERIMENTS:
        experiment = run_root / "dev" / str(config["name"])
        predictions = experiment / "predictions.json"
        diagnostics = experiment / "diagnostics.json"
        profile = experiment / "profile"
        model = local_base if config["model"] == "base" else local_adapter
        pipeline._run(
            _generation_command(
                pipeline,
                questions="data/raw/btc/LegalQA/train.json",
                rankings=heldout_rankings,
                model=model,
                base_model=local_base,
                output=predictions,
                diagnostics=diagnostics,
                config=config,
                question_ids=dev_ids,
            )
        )
        if not (profile / "profile_report.json").is_file():
            pipeline._run(
                _profile_command(
                    pipeline,
                    question_ids=dev_ids,
                    qwen=predictions,
                    extractive=extractive,
                    output=profile,
                )
            )
        report = _read_report(profile)
        experiment_rows.append(
            {
                "config": config,
                "selected_profile": report["selected_profile"],
                "selected_metrics": report["selected_metrics"],
            }
        )
    best = max(
        experiment_rows,
        key=lambda row: (
            float(row["selected_metrics"]["meteor"]),
            float(row["selected_metrics"]["rouge_l"]),
            str(row["config"]["name"]),
        ),
    )
    baseline_metrics = baseline_report["selected_metrics"]
    meteor_gain = float(best["selected_metrics"]["meteor"]) - float(
        baseline_metrics["meteor"]
    )
    rouge_gain = float(best["selected_metrics"]["rouge_l"]) - float(
        baseline_metrics["rouge_l"]
    )
    passed = meteor_gain >= 0.003 and rouge_gain >= -0.06
    summary = {
        "schema_version": "task2-p16-modal-result-v1",
        "provider": "modal",
        "status": "DEV_PASS" if passed else "DEV_REJECTED",
        "contract": contract,
        "baseline": {
            "selected_profile": baseline_report["selected_profile"],
            "selected_metrics": baseline_metrics,
        },
        "best": best,
        "meteor_gain": meteor_gain,
        "rouge_l_gain": rouge_gain,
        "experiments": experiment_rows,
        "gate": {"minimum_meteor_gain": 0.003, "minimum_rouge_gain": -0.06},
        "gpu": gpu,
        "run_root": str(run_root),
    }
    core._write_json(P16_SUMMARY, summary)
    _package_result(run_root, summary)
    return summary


def _run_all() -> dict[str, Any]:
    dev = _run_dev()
    if dev["status"] != "DEV_PASS":
        return dev
    pipeline, inputs, local_base, local_adapter, gpu, _ = _prepare_runtime()
    run_root = Path(str(dev["run_root"]))
    config = dict(dev["best"]["config"])
    model = local_base if config["model"] == "base" else local_adapter
    heldout_ids = inputs / "training/heldout_ids.json"
    extractive = inputs / "predictions/heldout_extractive.json"

    baseline_profile = run_root / "full_baseline_profile"
    if not (baseline_profile / "profile_report.json").is_file():
        pipeline._run(
            _profile_command(
                pipeline,
                question_ids=heldout_ids,
                qwen=inputs / "predictions/heldout_qwen.json",
                extractive=extractive,
                output=baseline_profile,
            )
        )
    full_predictions = run_root / "full" / "predictions.json"
    full_diagnostics = run_root / "full" / "diagnostics.json"
    pipeline._run(
        _generation_command(
            pipeline,
            questions="data/raw/btc/LegalQA/train.json",
            rankings=inputs / "rankings/heldout_rankings.jsonl",
            model=model,
            base_model=local_base,
            output=full_predictions,
            diagnostics=full_diagnostics,
            config=config,
            question_ids=heldout_ids,
        )
    )
    full_profile = run_root / "full" / "profile"
    if not (full_profile / "profile_report.json").is_file():
        pipeline._run(
            _profile_command(
                pipeline,
                question_ids=heldout_ids,
                qwen=full_predictions,
                extractive=extractive,
                output=full_profile,
            )
        )
    baseline_report = _read_report(baseline_profile)
    full_report = _read_report(full_profile)
    baseline_metrics = baseline_report["selected_metrics"]
    full_metrics = full_report["selected_metrics"]
    meteor_gain = float(full_metrics["meteor"]) - float(baseline_metrics["meteor"])
    full_pass = (
        float(full_metrics["meteor"]) >= 0.58
        and meteor_gain >= 0.01
        and float(full_metrics["rouge_l"]) >= 0.40
    )
    if not full_pass:
        summary = {
            **dev,
            "status": "FULL_HELDOUT_REJECTED",
            "full_baseline": {
                "selected_profile": baseline_report["selected_profile"],
                "selected_metrics": baseline_metrics,
            },
            "full_candidate": {
                "selected_profile": full_report["selected_profile"],
                "selected_metrics": full_metrics,
            },
            "full_meteor_gain": meteor_gain,
            "full_gate": {
                "minimum_meteor": 0.58,
                "minimum_meteor_gain": 0.01,
                "minimum_rouge_l": 0.40,
            },
            "gpu": gpu,
        }
        core._write_json(P16_SUMMARY, summary)
        _package_result(run_root, summary)
        return summary

    public_predictions = run_root / "public" / "qwen_predictions.json"
    public_diagnostics = run_root / "public" / "qwen_diagnostics.json"
    pipeline._run(
        _generation_command(
            pipeline,
            questions="data/raw/btc/LegalQA/public-official.json",
            rankings=inputs / "rankings/public_rankings.jsonl",
            model=model,
            base_model=local_base,
            output=public_predictions,
            diagnostics=public_diagnostics,
            config=config,
            known_answers="data/raw/btc/LegalQA/train.json",
        )
    )
    final_profile = run_root / "public" / "profile"
    if not (final_profile / "public_predictions.json").is_file():
        pipeline._run(
            _profile_command(
                pipeline,
                question_ids=heldout_ids,
                qwen=full_predictions,
                extractive=extractive,
                output=final_profile,
                public_qwen=public_predictions,
                public_extractive=inputs / "predictions/public_extractive.json",
            )
        )
    submission = run_root / "public" / "submission.zip"
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
        **dev,
        "status": "PUBLIC_CANDIDATE_READY",
        "full_baseline": {
            "selected_profile": baseline_report["selected_profile"],
            "selected_metrics": baseline_metrics,
        },
        "full_candidate": {
            "selected_profile": full_report["selected_profile"],
            "selected_metrics": full_metrics,
        },
        "full_meteor_gain": meteor_gain,
        "submission_sha256": core._sha256(submission),
        "gpu": gpu,
    }
    core._write_json(P16_SUMMARY, summary)
    _package_result(run_root, summary)
    return summary


def _package_result(run_root: Path, summary: dict[str, Any]) -> Path:
    output = Path(VOLUME_MOUNT) / REMOTE_RESULT
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    include = [P16_SUMMARY]
    for candidate in run_root.rglob("*"):
        if candidate.is_file() and (
            candidate.name in {"profile_report.json", "submission.zip"}
            or candidate.name.endswith("diagnostics.json")
        ):
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
                name = (
                    "p16_run_summary.json"
                    if path == P16_SUMMARY
                    else path.relative_to(run_root).as_posix()
                )
                bundle.write(path, name)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("P16 result archive is corrupt")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _validate_result(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("downloaded P16 result is corrupt")
        summary = json.loads(bundle.read("p16_run_summary.json").decode("utf-8"))
    allowed = {
        "DEV_PASS",
        "DEV_REJECTED",
        "FULL_HELDOUT_REJECTED",
        "PUBLIC_CANDIDATE_READY",
    }
    if summary.get("status") not in allowed:
        raise ValueError("downloaded P16 result has an invalid status")
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
            "p16": _p16_input_state(),
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
        timeout=3 * 60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def dev_sweep() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = core._load_pipeline()
        try:
            with pipeline._exclusive_run():
                return _run_dev()
        finally:
            task2_volume.commit()

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=6 * 60 * 60,
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

    def _upload_inputs(p14_archive: str, p15_bundle: str, p16_bundle: str) -> None:
        p14, p14_manifest, p14_meta = core._validated_local_input(p14_archive)
        p15, p15_manifest, p15_meta = core._validated_local_input(
            p15_bundle, "task2-p15-modal-input-v1"
        )
        p16, p16_manifest, p16_meta = core._validated_local_input(
            p16_bundle, "task2-p16-modal-input-v1"
        )
        state = input_probe.remote()
        ready = {
            "p14": state.get("p14", {}).get("status") == "INPUT_READY"
            and state["p14"].get("expected_sha256") == p14_meta["archive_sha256"],
            "p15": state.get("p15", {}).get("status") == "P15_INPUT_READY"
            and state["p15"].get("expected_sha256") == p15_meta["archive_sha256"],
            "p16": state.get("p16", {}).get("status") == "P16_INPUT_READY"
            and state["p16"].get("expected_sha256") == p16_meta["archive_sha256"],
        }
        if all(ready.values()):
            print("MODAL_P16_INPUTS_ALREADY_READY", flush=True)
            return
        with task2_volume.batch_upload(force=True) as batch:
            if not ready["p14"]:
                batch.put_file(str(p14), f"/{core.REMOTE_P14_ARCHIVE}")
                batch.put_file(str(p14_manifest), f"/{core.REMOTE_P14_MANIFEST}")
            if not ready["p15"]:
                batch.put_file(str(p15), f"/{core.REMOTE_P15_BUNDLE}")
                batch.put_file(str(p15_manifest), f"/{core.REMOTE_P15_MANIFEST}")
            if not ready["p16"]:
                batch.put_file(str(p16), f"/{REMOTE_P16_BUNDLE}")
                batch.put_file(str(p16_manifest), f"/{REMOTE_P16_MANIFEST}")
        verified = input_probe.remote()
        expected_status = {
            "p14": "INPUT_READY",
            "p15": "P15_INPUT_READY",
            "p16": "P16_INPUT_READY",
        }
        for name, status in expected_status.items():
            if verified.get(name, {}).get("status") != status:
                raise RuntimeError(f"Modal {name.upper()} input verification failed")

    def _download(download_dir: str) -> tuple[Path, dict[str, Any]]:
        destination = Path(download_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "task2_p16_modal_result.zip"
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
            with zipfile.ZipFile(output) as result:
                submission_name = next(
                    name
                    for name in result.namelist()
                    if name.endswith("submission.zip")
                )
                content = result.read(submission_name)
            with zipfile.ZipFile(io.BytesIO(content)) as submission:
                if submission.namelist() != ["submission.json"]:
                    raise ValueError("P16 submission contract is invalid")
            (destination / "submission.zip").write_bytes(content)
            print(f"MODAL_P16_SUBMISSION_READY={destination / 'submission.zip'}")
        print("MODAL_P16_RESULT=" + json.dumps(summary, ensure_ascii=False))
        return output, summary

    @app.local_entrypoint()
    def main(
        stage: str = "dev",
        p14_archive: str = str(core.DEFAULT_P14_ARCHIVE),
        p15_bundle: str = str(core.DEFAULT_P15_BUNDLE),
        p16_bundle: str = str(DEFAULT_P16_BUNDLE),
        download_dir: str = str(DEFAULT_DOWNLOAD_DIR),
    ) -> None:
        selected = stage.strip().lower()
        if selected == "check":
            core._validated_local_input(p14_archive)
            core._validated_local_input(p15_bundle, "task2-p15-modal-input-v1")
            core._validated_local_input(p16_bundle, "task2-p16-modal-input-v1")
            print(
                json.dumps(
                    {
                        "status": "LOCAL_CHECK_PASS",
                        "app": APP_NAME,
                        "gpu_priority": MODAL_GPU_PRIORITY,
                        "experiments": DEV_EXPERIMENTS,
                    },
                    indent=2,
                )
            )
            return
        if selected == "status":
            payload = core._read_volume_json(
                task2_volume, "p16/results/p16_run_summary.json"
            )
            print("MODAL_P16_STATUS=" + json.dumps(payload or {"status": "NO_RESULT"}))
            return
        if selected == "download":
            _download(download_dir)
            return
        if selected not in {"prepare", "dev", "run"}:
            raise ValueError(
                "stage must be check, prepare, dev, run, status, or download"
            )
        _upload_inputs(p14_archive, p15_bundle, p16_bundle)
        prepared = model_prepare.remote()
        if prepared.get("status") != "MODEL_READY":
            raise RuntimeError("Modal P16 model preparation failed")
        if selected == "prepare":
            print("MODAL_P16_PREPARE_COMPLETE")
            return
        result = dev_sweep.remote() if selected == "dev" else run_all.remote()
        allowed = (
            {"DEV_PASS", "DEV_REJECTED"}
            if selected == "dev"
            else {"DEV_REJECTED", "FULL_HELDOUT_REJECTED", "PUBLIC_CANDIDATE_READY"}
        )
        if result.get("status") not in allowed:
            raise RuntimeError("Modal P16 run failed: " + json.dumps(result))
        _download(download_dir)

else:
    app = None
    task2_volume = None


if __name__ == "__main__" and modal is None:
    raise SystemExit(
        "Modal SDK is missing. Install it with: python -m pip install -U modal"
    )
