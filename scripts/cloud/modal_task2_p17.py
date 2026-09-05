"""Run the leakage-safe P17 answer-selector experiment on one Modal H100."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
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

APP_NAME = "udsc-task2-p17-modal"
VOLUME_NAME = core.VOLUME_NAME
VOLUME_MOUNT = core.VOLUME_MOUNT
P17_ROOT = Path(VOLUME_MOUNT) / "p17"
P17_RESULT_ROOT = P17_ROOT / "results"
P17_SUMMARY = P17_RESULT_ROOT / "p17_run_summary.json"
REMOTE_P17_BUNDLE = "p17/input/task2_p17_modal_input.zip"
REMOTE_P17_MANIFEST = "p17/input/task2_p17_modal_input.manifest.json"
REMOTE_RESULT = "p17/results/task2_p17_modal_result.zip"
DEFAULT_P17_BUNDLE = core.PROJECT_ROOT / "artifacts/task2/task2_p17_modal_input.zip"
DEFAULT_DOWNLOAD_DIR = core.PROJECT_ROOT / "artifacts/task2/modal_p17_download"
MODAL_GPU_PRIORITY = ["H100"]
FOLDS = 5
EPOCHS = 2
BASELINE_OFFICIAL_METEOR = 0.5659266467010807
MINIMUM_OFFICIAL_METEOR = 0.585
MINIMUM_FAST_METEOR = 0.58
MINIMUM_FOLD_FAST_METEOR = 0.55


def _p17_input_state() -> dict[str, Any]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P17_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P17_MANIFEST
    if not archive.is_file() or not manifest_path.is_file():
        return {"status": "P17_INPUT_MISSING"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = str(manifest.get("archive_sha256", ""))
    observed = core._sha256(archive)
    return {
        "status": "P17_INPUT_READY" if expected == observed else "P17_INPUT_INVALID",
        "expected_sha256": expected,
        "observed_sha256": observed,
        "archive_bytes": archive.stat().st_size,
    }


def _extract_p17_bundle() -> tuple[Path, dict[str, Any]]:
    archive = Path(VOLUME_MOUNT) / REMOTE_P17_BUNDLE
    manifest_path = Path(VOLUME_MOUNT) / REMOTE_P17_MANIFEST
    state = _p17_input_state()
    if state["status"] != "P17_INPUT_READY":
        raise RuntimeError("P17 input is not verified: " + json.dumps(state))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = P17_ROOT / "extracted" / str(manifest["archive_sha256"])[:24]
    ready = root / ".ready.json"
    if ready.is_file():
        return root, manifest
    root.mkdir(parents=True, exist_ok=True)
    expected = {str(row["archive_path"]): row for row in manifest.get("members", [])}
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        if set(names) != set(expected):
            raise ValueError("P17 archive member contract mismatch")
        for info in bundle.infolist():
            destination = (root / info.filename).resolve()
            if root.resolve() not in destination.parents:
                raise ValueError(f"unsafe P17 member path: {info.filename}")
        bundle.extractall(root)
    for name, row in expected.items():
        path = root / name
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"P17 member size mismatch: {name}")
        if core._sha256(path) != str(row["sha256"]):
            raise ValueError(f"P17 member hash mismatch: {name}")
    core._write_json(
        ready,
        {"schema_version": 1, "archive_sha256": manifest["archive_sha256"]},
    )
    return root, manifest


def _contract(p17_hash: str, source_hash: str) -> str:
    payload = {
        "schema": "task2-p17-answer-selector-v1",
        "p17_hash": p17_hash,
        "source_hash": source_hash,
        "folds": FOLDS,
        "epochs": EPOCHS,
        "base": "dek21-parent-crossencoder-hn16-v2/checkpoint-best",
        "target_temperature": 0.05,
        "freeze_layers": 8,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _prepare_runtime() -> tuple[Any, Path, Path, dict[str, Any], str]:
    pipeline = core._load_pipeline()
    pipeline._extract_workspace()
    source_hash = pipeline._overlay_source()
    inputs, _ = _extract_p17_bundle()
    gpu = pipeline._gpu_manifest()
    pipeline._require_free_vram(gpu)
    base = pipeline._localize_directory(
        pipeline.WORKSPACE
        / "artifacts/task2/models/dek21-parent-crossencoder-hn16-v2"
        / "checkpoint-best",
        "p17_parent_base",
    )
    return pipeline, inputs, base, gpu, source_hash


def _train_command(
    pipeline: Any,
    *,
    train_data: Path,
    eval_data: Path | None,
    base: Path,
    output: Path,
    seed: int,
) -> list[str]:
    command = pipeline._python(
        "scripts/training/train_legal_qa_parent_crossencoder.py",
        "train",
        "--train-data",
        str(train_data),
        "--model-dir",
        str(base),
        "--output-dir",
        str(output),
        "--objective",
        "listwise",
        "--target-temperature",
        "0.05",
        "--epochs",
        str(EPOCHS),
        "--learning-rate",
        "8e-6",
        "--batch-size",
        "4",
        "--eval-batch-size",
        "8",
        "--gradient-accumulation",
        "2",
        "--max-length",
        "256",
        "--freeze-layers",
        "8",
        "--freeze-embeddings",
        "--device",
        "cuda",
        "--amp",
        "--num-workers",
        "2",
        "--seed",
        str(seed),
        "--resume",
    )
    if eval_data is None:
        command.append("--fit-all")
    else:
        command.extend(["--eval-data", str(eval_data)])
    return command


def _score_command(
    pipeline: Any,
    *,
    questions: Path,
    bank: Path,
    checkpoint: Path,
    output: Path,
    ids: Path | None = None,
    fold: int | None = None,
) -> list[str]:
    command = pipeline._python(
        "scripts/training/train_task2_p17_answer_selector.py",
        "score",
        "--questions",
        str(questions),
        "--candidate-bank",
        str(bank),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--batch-size",
        "128",
        "--max-length",
        "256",
        "--device",
        "cuda",
        "--amp",
    )
    if ids is not None:
        command.extend(["--question-ids", str(ids)])
    if fold is not None:
        command.extend(["--fold", str(fold)])
    return command


def _official_evaluate_command(
    pipeline: Any,
    *,
    questions: Path,
    ids: Path,
    predictions: Path,
    output: Path,
) -> list[str]:
    return pipeline._python(
        "scripts/training/finetune_task2_qwen_lora.py",
        "evaluate",
        "--questions",
        str(questions),
        "--question-ids",
        str(ids),
        "--predictions",
        str(predictions),
        "--output",
        str(output),
        "--minimum-meteor",
        "0",
    )


def _run_oof() -> tuple[dict[str, Any], Path, Path, Path, Any, Path, dict[str, Any]]:
    pipeline, inputs, base, gpu, source_hash = _prepare_runtime()
    manifest = json.loads(
        (Path(VOLUME_MOUNT) / REMOTE_P17_MANIFEST).read_text(encoding="utf-8-sig")
    )
    contract = _contract(str(manifest["archive_sha256"]), source_hash)
    run_root = P17_RESULT_ROOT / "runs" / contract
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"task2-p17-{contract}-"))
    prepared = temporary / "prepared"
    questions = pipeline.WORKSPACE / "data/raw/btc/LegalQA/train.json"
    bank = inputs / "heldout/candidate_bank.jsonl"
    ids = inputs / "heldout/question_ids.json"
    pipeline._run(
        pipeline._python(
            "scripts/training/train_task2_p17_answer_selector.py",
            "prepare",
            "--questions",
            str(questions),
            "--question-ids",
            str(ids),
            "--candidate-bank",
            str(bank),
            "--output-dir",
            str(prepared),
            "--folds",
            str(FOLDS),
        )
    )
    score_paths: list[Path] = []
    fold_manifests: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        print(f"P17_FOLD_START {fold + 1}/{FOLDS}", flush=True)
        fold_output = temporary / f"model_fold_{fold}"
        pipeline._run(
            _train_command(
                pipeline,
                train_data=prepared / f"fold_{fold}_train.jsonl",
                eval_data=prepared / f"fold_{fold}_valid.jsonl",
                base=base,
                output=fold_output,
                seed=2026 + fold,
            )
        )
        training_manifest = json.loads(
            (fold_output / "training_manifest.json").read_text(encoding="utf-8")
        )
        fold_manifests.append(
            {
                "fold": fold,
                "best_epoch": training_manifest["best_epoch"],
                "best_metric": training_manifest["best_metric"],
                "history": training_manifest["history"],
            }
        )
        score_path = run_root / "oof" / f"fold_{fold}_scores.jsonl"
        pipeline._run(
            _score_command(
                pipeline,
                questions=questions,
                bank=bank,
                checkpoint=fold_output / "checkpoint-best",
                output=score_path,
                ids=prepared / f"fold_{fold}_valid_ids.json",
                fold=fold,
            )
        )
        score_paths.append(score_path)
        shutil.rmtree(fold_output)
        print(f"P17_FOLD_COMPLETE {fold + 1}/{FOLDS}", flush=True)

    selector_dir = run_root / "oof" / "selector"
    select_command = pipeline._python(
        "scripts/training/train_task2_p17_answer_selector.py",
        "select-oof",
        "--candidate-bank",
        str(bank),
        "--split-manifest",
        str(prepared / "split_manifest.json"),
        "--output-dir",
        str(selector_dir),
    )
    for path in score_paths:
        select_command.extend(["--scores", str(path)])
    pipeline._run(select_command)
    os.environ["NLTK_DATA"] = str(inputs / "nltk_data")
    official_path = selector_dir / "official_metrics.json"
    pipeline._run(
        _official_evaluate_command(
            pipeline,
            questions=questions,
            ids=ids,
            predictions=selector_dir / "oof_predictions.json",
            output=official_path,
        )
    )
    selector_report = json.loads(
        (selector_dir / "selector_report.json").read_text(encoding="utf-8")
    )
    official = json.loads(official_path.read_text(encoding="utf-8"))
    fast_meteor = float(selector_report["oof_fast_metrics"]["meteor"])
    official_meteor = float(official["meteor"])
    minimum_fold = min(
        float(row["meteor"]) for row in selector_report["fold_metrics"].values()
    )
    passed = (
        official_meteor >= MINIMUM_OFFICIAL_METEOR
        and fast_meteor >= MINIMUM_FAST_METEOR
        and minimum_fold >= MINIMUM_FOLD_FAST_METEOR
    )
    summary = {
        "schema_version": "task2-p17-modal-result-v1",
        "provider": "modal",
        "status": "OOF_PASS" if passed else "OOF_REJECTED",
        "contract": contract,
        "oof_fast_metrics": selector_report["oof_fast_metrics"],
        "oof_official_metrics": {
            "meteor": official_meteor,
            "rouge_l": float(official["rouge_l"]),
        },
        "baseline_official_meteor": BASELINE_OFFICIAL_METEOR,
        "official_meteor_gain": official_meteor - BASELINE_OFFICIAL_METEOR,
        "minimum_fold_fast_meteor": minimum_fold,
        "selected_alpha": selector_report["selected_alpha"],
        "fold_training": fold_manifests,
        "gate": {
            "minimum_official_meteor": MINIMUM_OFFICIAL_METEOR,
            "minimum_fast_meteor": MINIMUM_FAST_METEOR,
            "minimum_fold_fast_meteor": MINIMUM_FOLD_FAST_METEOR,
        },
        "gpu": gpu,
        "run_root": str(run_root),
    }
    core._write_json(P17_SUMMARY, summary)
    shutil.rmtree(temporary, ignore_errors=True)
    return summary, run_root, inputs, base, pipeline, questions, gpu


def _run_all() -> dict[str, Any]:
    summary, run_root, inputs, base, pipeline, questions, gpu = _run_oof()
    if summary["status"] != "OOF_PASS":
        _package_result(run_root, summary)
        return summary

    temporary = Path(tempfile.mkdtemp(prefix="task2-p17-full-"))
    try:
        prepared = temporary / "prepared"
        bank = inputs / "heldout/candidate_bank.jsonl"
        ids = inputs / "heldout/question_ids.json"
        pipeline._run(
            pipeline._python(
                "scripts/training/train_task2_p17_answer_selector.py",
                "prepare",
                "--questions",
                str(questions),
                "--question-ids",
                str(ids),
                "--candidate-bank",
                str(bank),
                "--output-dir",
                str(prepared),
                "--folds",
                str(FOLDS),
            )
        )
        full_model = temporary / "full_model"
        pipeline._run(
            _train_command(
                pipeline,
                train_data=prepared / "all.jsonl",
                eval_data=None,
                base=base,
                output=full_model,
                seed=2031,
            )
        )
        public_bank = run_root / "public" / "candidate_bank.jsonl"
        public_questions = (
            pipeline.WORKSPACE / "data/raw/btc/LegalQA/public-official.json"
        )
        pipeline._run(
            pipeline._python(
                "scripts/training/train_task2_p17_answer_selector.py",
                "build-bank",
                "--questions",
                str(public_questions),
                "--qwen",
                str(inputs / "public/qwen.json"),
                "--extractive",
                str(inputs / "public/extractive.json"),
                "--output",
                str(public_bank),
            )
        )
        public_scores = run_root / "public" / "scores.jsonl"
        pipeline._run(
            _score_command(
                pipeline,
                questions=public_questions,
                bank=public_bank,
                checkpoint=full_model / "checkpoint-final",
                output=public_scores,
            )
        )
        public_predictions = run_root / "public" / "predictions.json"
        pipeline._run(
            pipeline._python(
                "scripts/training/train_task2_p17_answer_selector.py",
                "select-public",
                "--candidate-bank",
                str(public_bank),
                "--scores",
                str(public_scores),
                "--selector-report",
                str(run_root / "oof/selector/selector_report.json"),
                "--output",
                str(public_predictions),
            )
        )
        submission = run_root / "public" / "submission.zip"
        pipeline._run(
            pipeline._python(
                "scripts/submission/write_legal_qa_submission.py",
                "--input",
                str(public_predictions),
                "--questions",
                str(public_questions),
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
                str(public_questions),
            )
        )
        summary = {
            **summary,
            "status": "PUBLIC_CANDIDATE_READY",
            "submission_sha256": core._sha256(submission),
            "gpu": gpu,
        }
        core._write_json(P17_SUMMARY, summary)
        _package_result(run_root, summary)
        return summary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _package_result(run_root: Path, summary: dict[str, Any]) -> Path:
    output = Path(VOLUME_MOUNT) / REMOTE_RESULT
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    include = [P17_SUMMARY]
    allowed_names = {
        "selector_report.json",
        "official_metrics.json",
        "oof_predictions.json",
        "predictions.json",
        "submission.zip",
    }
    include.extend(
        path
        for path in run_root.rglob("*")
        if path.is_file() and path.name in allowed_names
    )
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
                    "p17_run_summary.json"
                    if path == P17_SUMMARY
                    else path.relative_to(run_root).as_posix()
                )
                bundle.write(path, name)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("P17 result archive is corrupt")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _validate_result(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("downloaded P17 result is corrupt")
        summary = json.loads(bundle.read("p17_run_summary.json").decode("utf-8"))
    allowed = {"OOF_PASS", "OOF_REJECTED", "PUBLIC_CANDIDATE_READY"}
    if summary.get("status") not in allowed:
        raise ValueError("downloaded P17 result has an invalid status")
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
            "p17": _p17_input_state(),
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
        inputs, manifest = _extract_p17_bundle()
        base = (
            pipeline.WORKSPACE
            / "artifacts/task2/models/dek21-parent-crossencoder-hn16-v2"
            / "checkpoint-best"
        )
        required_model_files = (
            base / "config.json",
            base / "model.safetensors",
            base / "tokenizer_config.json",
        )
        missing = [str(path) for path in required_model_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "P17 parent checkpoint is incomplete: " + ", ".join(missing)
            )
        temporary = Path(tempfile.mkdtemp(prefix="task2-p17-prepare-"))
        try:
            prepared = temporary / "prepared"
            pipeline._run(
                pipeline._python(
                    "scripts/training/train_task2_p17_answer_selector.py",
                    "prepare",
                    "--questions",
                    str(pipeline.WORKSPACE / "data/raw/btc/LegalQA/train.json"),
                    "--question-ids",
                    str(inputs / "heldout/question_ids.json"),
                    "--candidate-bank",
                    str(inputs / "heldout/candidate_bank.jsonl"),
                    "--output-dir",
                    str(prepared),
                    "--folds",
                    str(FOLDS),
                )
            )
            pipeline._run(
                pipeline._python(
                    "scripts/training/train_legal_qa_parent_crossencoder.py",
                    "audit",
                    "--train-data",
                    str(prepared / "fold_0_train.jsonl"),
                    "--eval-data",
                    str(prepared / "fold_0_valid.jsonl"),
                )
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return {
            "status": "P17_PREPARED",
            "workspace": str(pipeline.WORKSPACE),
            "inputs": str(inputs),
            "archive_sha256": manifest["archive_sha256"],
            "source_sha256": source_hash,
            "parent_checkpoint": str(base),
        }

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=4 * 60 * 60,
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

    def _upload_inputs(p14_archive: str, p17_bundle: str) -> None:
        p14, p14_manifest, p14_meta = core._validated_local_input(p14_archive)
        p17, p17_manifest, p17_meta = core._validated_local_input(
            p17_bundle, "task2-p17-modal-input-v1"
        )
        state = input_probe.remote()
        p14_ready = (
            state.get("p14", {}).get("status") == "INPUT_READY"
            and state["p14"].get("expected_sha256") == p14_meta["archive_sha256"]
        )
        p17_ready = (
            state.get("p17", {}).get("status") == "P17_INPUT_READY"
            and state["p17"].get("expected_sha256") == p17_meta["archive_sha256"]
        )
        if p14_ready and p17_ready:
            print("MODAL_P17_INPUTS_ALREADY_READY", flush=True)
            return
        with task2_volume.batch_upload(force=True) as batch:
            if not p14_ready:
                batch.put_file(str(p14), f"/{core.REMOTE_P14_ARCHIVE}")
                batch.put_file(str(p14_manifest), f"/{core.REMOTE_P14_MANIFEST}")
            if not p17_ready:
                batch.put_file(str(p17), f"/{REMOTE_P17_BUNDLE}")
                batch.put_file(str(p17_manifest), f"/{REMOTE_P17_MANIFEST}")
        verified = input_probe.remote()
        if verified.get("p14", {}).get("status") != "INPUT_READY":
            raise RuntimeError("Modal P14 input verification failed")
        if verified.get("p17", {}).get("status") != "P17_INPUT_READY":
            raise RuntimeError("Modal P17 input verification failed")

    def _download(download_dir: str) -> tuple[Path, dict[str, Any]]:
        destination = Path(download_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "task2_p17_modal_result.zip"
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
                submission_name = next(
                    name
                    for name in result.namelist()
                    if name.endswith("submission.zip")
                )
                content = result.read(submission_name)
            with zipfile.ZipFile(io.BytesIO(content)) as submission:
                if submission.namelist() != ["submission.json"]:
                    raise ValueError("P17 submission contract is invalid")
            submission_path = destination / "submission.zip"
            submission_path.write_bytes(content)
            print(f"MODAL_P17_SUBMISSION_READY={submission_path}")
        print("MODAL_P17_RESULT=" + json.dumps(summary, ensure_ascii=False))
        return output, summary

    @app.local_entrypoint()
    def main(
        stage: str = "run",
        p14_archive: str = str(core.DEFAULT_P14_ARCHIVE),
        p17_bundle: str = str(DEFAULT_P17_BUNDLE),
        download_dir: str = str(DEFAULT_DOWNLOAD_DIR),
    ) -> None:
        selected = stage.strip().lower()
        if selected == "check":
            core._validated_local_input(p14_archive)
            core._validated_local_input(p17_bundle, "task2-p17-modal-input-v1")
            print(
                json.dumps(
                    {
                        "status": "LOCAL_CHECK_PASS",
                        "app": APP_NAME,
                        "gpu_priority": MODAL_GPU_PRIORITY,
                        "folds": FOLDS,
                        "epochs": EPOCHS,
                    },
                    indent=2,
                )
            )
            return
        if selected == "status":
            payload = core._read_volume_json(
                task2_volume, "p17/results/p17_run_summary.json"
            )
            print("MODAL_P17_STATUS=" + json.dumps(payload or {"status": "NO_RESULT"}))
            return
        if selected == "download":
            _download(download_dir)
            return
        if selected not in {"prepare", "run"}:
            raise ValueError("stage must be check, prepare, run, status, or download")
        _upload_inputs(p14_archive, p17_bundle)
        prepared = prepare_inputs.remote()
        if prepared.get("status") != "P17_PREPARED":
            raise RuntimeError("Modal P17 preparation failed")
        if selected == "prepare":
            print("MODAL_P17_PREPARE_COMPLETE")
            return
        result = run_all.remote()
        if result.get("status") not in {"OOF_REJECTED", "PUBLIC_CANDIDATE_READY"}:
            raise RuntimeError("Modal P17 run failed: " + json.dumps(result))
        _download(download_dir)

else:
    app = None
    task2_volume = None


if __name__ == "__main__" and modal is None:
    raise SystemExit(
        "Modal SDK is missing. Install it with: python -m pip install -U modal"
    )
