"""Run the inference-only Task 2 P19 top-parent sweep on Modal H100.

P19 never trains the registered Qwen model.  It reuses the verified P14
adapter and official BTC rankings, selects a top-parent policy on disjoint dev
IDs, confirms it on the full held-out set, and only then generates public
answers.  The final CPU selector uses no reference at inference time.
"""

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

from scripts.cloud import modal_task2_p16 as p16  # noqa: E402

core = p16.core
APP_NAME = "udsc-task2-p19-modal"
VOLUME_NAME = core.VOLUME_NAME
VOLUME_MOUNT = core.VOLUME_MOUNT
P19_ROOT = Path(VOLUME_MOUNT) / "p19"
P19_RESULT_ROOT = P19_ROOT / "results"
P19_SUMMARY = P19_RESULT_ROOT / "p19_run_summary.json"
REMOTE_RESULT = "p19/results/task2_p19_modal_result.zip"
DEFAULT_DOWNLOAD_DIR = core.PROJECT_ROOT / "artifacts/task2/modal_p19_download"
MODAL_GPU_PRIORITY = ["H100"]

DEV_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "adapter_top1_ce",
        "model": "adapter",
        "top_parents": 1,
        "max_parent_words": 512,
        "max_context_tokens": 1800,
        "ce_weight": 1.0,
        "retrieval_weight": 0.0,
    },
    {
        "name": "adapter_top1_retrieval",
        "model": "adapter",
        "top_parents": 1,
        "max_parent_words": 512,
        "max_context_tokens": 1800,
        "ce_weight": 0.0,
        "retrieval_weight": 1.0,
    },
    {
        "name": "adapter_top1_fusion",
        "model": "adapter",
        "top_parents": 1,
        "max_parent_words": 512,
        "max_context_tokens": 1800,
        "ce_weight": 0.3,
        "retrieval_weight": 0.7,
    },
)
BASELINE_SELECTOR_OOF = {
    "meteor": 0.5683329701423645,
    "rouge_l": 0.46632450819015503,
}


def _contract(p16_hash: str, source_hash: str) -> str:
    payload = {
        "schema": "task2-p19-top1-inference-v1",
        "p16_hash": p16_hash,
        "source_hash": source_hash,
        "experiments": DEV_EXPERIMENTS,
        "baseline_selector_oof": BASELINE_SELECTOR_OOF,
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
    p16_root, _ = p16._extract_p16_bundle()
    os.environ["NLTK_DATA"] = str(p15_root / "nltk_data")
    local_base = pipeline._localize_directory(pipeline.QWEN_DIR, "p19_qwen_base")
    local_adapter = pipeline._localize_directory(
        p15_root / "initial_adapter", "p19_initial_adapter"
    )
    return pipeline, p16_root, local_base, local_adapter, gpu, source_hash


def _profile_report(path: Path) -> dict[str, Any]:
    return core._read_json(path / "profile_report.json")


def _run_profile(
    pipeline: Any,
    *,
    ids: Path,
    qwen: Path,
    extractive: Path,
    output: Path,
) -> dict[str, Any]:
    if not (output / "profile_report.json").is_file():
        pipeline._run(
            p16._profile_command(
                pipeline,
                question_ids=ids,
                qwen=qwen,
                extractive=extractive,
                output=output,
            )
        )
    return _profile_report(output)


def _generate(
    pipeline: Any,
    *,
    questions: str,
    rankings: Path,
    adapter: Path,
    base: Path,
    output: Path,
    diagnostics: Path,
    config: dict[str, Any],
    ids: Path | None = None,
    known_answers: str | None = None,
) -> None:
    pipeline._run(
        p16._generation_command(
            pipeline,
            questions=questions,
            rankings=rankings,
            model=adapter,
            base_model=base,
            output=output,
            diagnostics=diagnostics,
            config=config,
            question_ids=ids,
            known_answers=known_answers,
        )
    )


def _run_dev() -> dict[str, Any]:
    pipeline, inputs, base, adapter, gpu, source_hash = _prepare_runtime()
    p16_manifest = json.loads(
        (Path(VOLUME_MOUNT) / p16.REMOTE_P16_MANIFEST).read_text(
            encoding="utf-8-sig"
        )
    )
    contract = _contract(str(p16_manifest["archive_sha256"]), source_hash)
    run_root = P19_RESULT_ROOT / "runs" / contract
    run_root.mkdir(parents=True, exist_ok=True)
    dev_ids = inputs / "training/dev_ids.json"
    rankings = inputs / "rankings/heldout_rankings.jsonl"
    extractive = inputs / "predictions/heldout_extractive.json"
    baseline = _run_profile(
        pipeline,
        ids=dev_ids,
        qwen=inputs / "predictions/heldout_qwen.json",
        extractive=extractive,
        output=run_root / "dev_baseline_profile",
    )

    experiments: list[dict[str, Any]] = []
    for config in DEV_EXPERIMENTS:
        root = run_root / "dev" / str(config["name"])
        predictions = root / "predictions.json"
        _generate(
            pipeline,
            questions="data/raw/btc/LegalQA/train.json",
            rankings=rankings,
            adapter=adapter,
            base=base,
            output=predictions,
            diagnostics=root / "diagnostics.json",
            config=config,
            ids=dev_ids,
        )
        report = _run_profile(
            pipeline,
            ids=dev_ids,
            qwen=predictions,
            extractive=extractive,
            output=root / "profile",
        )
        experiments.append(
            {
                "config": config,
                "selected_profile": report["selected_profile"],
                "selected_metrics": report["selected_metrics"],
            }
        )
    best = max(
        experiments,
        key=lambda row: (
            float(row["selected_metrics"]["meteor"]),
            float(row["selected_metrics"]["rouge_l"]),
            str(row["config"]["name"]),
        ),
    )
    baseline_metrics = baseline["selected_metrics"]
    meteor_gain = float(best["selected_metrics"]["meteor"]) - float(
        baseline_metrics["meteor"]
    )
    rouge_gain = float(best["selected_metrics"]["rouge_l"]) - float(
        baseline_metrics["rouge_l"]
    )
    passed = meteor_gain >= 0.002 and rouge_gain >= -0.04
    summary = {
        "schema_version": "task2-p19-modal-result-v1",
        "provider": "modal",
        "status": "DEV_PASS" if passed else "DEV_REJECTED",
        "contract": contract,
        "baseline": {
            "selected_profile": baseline["selected_profile"],
            "selected_metrics": baseline_metrics,
        },
        "best": best,
        "meteor_gain": meteor_gain,
        "rouge_l_gain": rouge_gain,
        "experiments": experiments,
        "gate": {"minimum_meteor_gain": 0.002, "minimum_rouge_gain": -0.04},
        "gpu": gpu,
        "run_root": str(run_root),
        "external_data": False,
        "augmentation": False,
    }
    core._write_json(P19_SUMMARY, summary)
    _package_result(run_root)
    return summary


def _build_scored_bank(
    pipeline: Any,
    *,
    inputs: Path,
    ids: Path,
    qwen: Path,
    output: Path,
) -> Path:
    selector_dir = output / "p15_selector"
    pipeline._run(
        pipeline._python(
            "scripts/evaluation/tune_task2_p15_answers.py",
            "--questions",
            "data/raw/btc/LegalQA/train.json",
            "--question-ids",
            str(ids),
            "--qwen",
            str(qwen),
            "--extractive",
            str(inputs / "predictions/heldout_extractive.json"),
            "--output-dir",
            str(selector_dir),
            "--metric",
            "fast",
            "--meteor-weight",
            "1",
            "--rouge-weight",
            "0",
        )
    )
    return selector_dir / "candidate_bank.jsonl"


def _run_all() -> dict[str, Any]:
    dev = _run_dev()
    if dev["status"] != "DEV_PASS":
        return dev
    pipeline, inputs, base, adapter, gpu, _ = _prepare_runtime()
    run_root = Path(str(dev["run_root"]))
    config = dict(dev["best"]["config"])
    heldout_ids = inputs / "training/heldout_ids.json"
    extractive = inputs / "predictions/heldout_extractive.json"
    baseline = _run_profile(
        pipeline,
        ids=heldout_ids,
        qwen=inputs / "predictions/heldout_qwen.json",
        extractive=extractive,
        output=run_root / "full_baseline_profile",
    )
    full_root = run_root / "full"
    full_predictions = full_root / "qwen_predictions.json"
    _generate(
        pipeline,
        questions="data/raw/btc/LegalQA/train.json",
        rankings=inputs / "rankings/heldout_rankings.jsonl",
        adapter=adapter,
        base=base,
        output=full_predictions,
        diagnostics=full_root / "qwen_diagnostics.json",
        config=config,
        ids=heldout_ids,
    )
    full_profile = _run_profile(
        pipeline,
        ids=heldout_ids,
        qwen=full_predictions,
        extractive=extractive,
        output=full_root / "profile",
    )
    bank = _build_scored_bank(
        pipeline,
        inputs=inputs,
        ids=heldout_ids,
        qwen=full_predictions,
        output=full_root,
    )
    research_report = full_root / "p19_selector_research.json"
    pipeline._run(
        pipeline._python(
            "scripts/evaluation/research_task2_p19_selector.py",
            "--candidate-bank",
            str(bank),
            "--questions",
            "data/raw/btc/LegalQA/train.json",
            "--output",
            str(research_report),
        )
    )
    research = core._read_json(research_report)
    selector_metrics = research["methods"]["numeric_extra_trees_t0"]
    profile_gain = float(full_profile["selected_metrics"]["meteor"]) - float(
        baseline["selected_metrics"]["meteor"]
    )
    selector_gain = float(selector_metrics["meteor"]) - float(
        BASELINE_SELECTOR_OOF["meteor"]
    )
    passed = (
        profile_gain >= 0.003
        and selector_gain >= 0.002
        and float(selector_metrics["rouge_l"]) >= 0.44
    )
    if not passed:
        summary = {
            **dev,
            "status": "FULL_HELDOUT_REJECTED",
            "full_baseline": {
                "selected_profile": baseline["selected_profile"],
                "selected_metrics": baseline["selected_metrics"],
            },
            "full_candidate": {
                "selected_profile": full_profile["selected_profile"],
                "selected_metrics": full_profile["selected_metrics"],
            },
            "selector_oof": selector_metrics,
            "profile_meteor_gain": profile_gain,
            "selector_meteor_gain": selector_gain,
            "full_gate": {
                "minimum_profile_gain": 0.003,
                "minimum_selector_gain": 0.002,
                "minimum_selector_rouge_l": 0.44,
            },
            "gpu": gpu,
        }
        core._write_json(P19_SUMMARY, summary)
        _package_result(run_root)
        return summary

    selector = full_root / "selector.joblib"
    pipeline._run(
        pipeline._python(
            "scripts/training/train_task2_p19_uplift_selector.py",
            "train",
            "--questions",
            "data/raw/btc/LegalQA/train.json",
            "--candidate-bank",
            str(bank),
            "--output",
            str(selector),
        )
    )
    public_root = run_root / "public"
    public_qwen = public_root / "qwen_predictions.json"
    _generate(
        pipeline,
        questions="data/raw/btc/LegalQA/public-official.json",
        rankings=inputs / "rankings/public_rankings.jsonl",
        adapter=adapter,
        base=base,
        output=public_qwen,
        diagnostics=public_root / "qwen_diagnostics.json",
        config=config,
        known_answers="data/raw/btc/LegalQA/train.json",
    )
    public_predictions = public_root / "predictions.json"
    pipeline._run(
        pipeline._python(
            "scripts/training/train_task2_p19_uplift_selector.py",
            "select",
            "--questions",
            "data/raw/btc/LegalQA/public-official.json",
            "--qwen",
            str(public_qwen),
            "--extractive",
            str(inputs / "predictions/public_extractive.json"),
            "--selector",
            str(selector),
            "--known-answers",
            "data/raw/btc/LegalQA/train.json",
            "--output",
            str(public_predictions),
            "--diagnostics",
            str(public_root / "selector_diagnostics.json"),
        )
    )
    submission = public_root / "submission.zip"
    pipeline._run(
        pipeline._python(
            "scripts/submission/write_legal_qa_submission.py",
            "--input",
            str(public_predictions),
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
            "selected_profile": baseline["selected_profile"],
            "selected_metrics": baseline["selected_metrics"],
        },
        "full_candidate": {
            "selected_profile": full_profile["selected_profile"],
            "selected_metrics": full_profile["selected_metrics"],
        },
        "selector_oof": selector_metrics,
        "profile_meteor_gain": profile_gain,
        "selector_meteor_gain": selector_gain,
        "submission_sha256": core._sha256(submission),
        "gpu": gpu,
    }
    core._write_json(P19_SUMMARY, summary)
    _package_result(run_root)
    return summary


def _package_result(run_root: Path) -> Path:
    output = Path(VOLUME_MOUNT) / REMOTE_RESULT
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    include = [P19_SUMMARY]
    allowed = {
        "profile_report.json",
        "p19_selector_research.json",
        "selector.manifest.json",
        "selector_diagnostics.json",
        "submission.zip",
    }
    for candidate in run_root.rglob("*"):
        if candidate.is_file() and (
            candidate.name in allowed or candidate.name.endswith("diagnostics.json")
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
                    "p19_run_summary.json"
                    if path == P19_SUMMARY
                    else path.relative_to(run_root).as_posix()
                )
                bundle.write(path, name)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("P19 result archive is corrupt")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _validate_result(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("downloaded P19 result is corrupt")
        summary = json.loads(bundle.read("p19_run_summary.json").decode("utf-8"))
    allowed = {
        "DEV_PASS",
        "DEV_REJECTED",
        "FULL_HELDOUT_REJECTED",
        "PUBLIC_CANDIDATE_READY",
    }
    if summary.get("status") not in allowed:
        raise ValueError("downloaded P19 result has an invalid status")
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
            "p16": p16._p16_input_state(),
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
        p16_file, p16_manifest, p16_meta = core._validated_local_input(
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
            print("MODAL_P19_INPUTS_ALREADY_READY", flush=True)
            return
        with task2_volume.batch_upload(force=True) as batch:
            if not ready["p14"]:
                batch.put_file(str(p14), f"/{core.REMOTE_P14_ARCHIVE}")
                batch.put_file(str(p14_manifest), f"/{core.REMOTE_P14_MANIFEST}")
            if not ready["p15"]:
                batch.put_file(str(p15), f"/{core.REMOTE_P15_BUNDLE}")
                batch.put_file(str(p15_manifest), f"/{core.REMOTE_P15_MANIFEST}")
            if not ready["p16"]:
                batch.put_file(str(p16_file), f"/{p16.REMOTE_P16_BUNDLE}")
                batch.put_file(str(p16_manifest), f"/{p16.REMOTE_P16_MANIFEST}")
        verified = input_probe.remote()
        expected = {
            "p14": "INPUT_READY",
            "p15": "P15_INPUT_READY",
            "p16": "P16_INPUT_READY",
        }
        for name, status in expected.items():
            if verified.get(name, {}).get("status") != status:
                raise RuntimeError(f"Modal {name.upper()} input verification failed")

    def _download(download_dir: str) -> tuple[Path, dict[str, Any]]:
        destination = Path(download_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "task2_p19_modal_result.zip"
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
                name = next(
                    value
                    for value in result.namelist()
                    if value.endswith("submission.zip")
                )
                content = result.read(name)
            with zipfile.ZipFile(io.BytesIO(content)) as submission:
                if submission.namelist() != ["submission.json"]:
                    raise ValueError("P19 submission contract is invalid")
            (destination / "submission.zip").write_bytes(content)
            print(f"MODAL_P19_SUBMISSION_READY={destination / 'submission.zip'}")
        print("MODAL_P19_RESULT=" + json.dumps(summary, ensure_ascii=False))
        return output, summary

    @app.local_entrypoint()
    def main(
        stage: str = "dev",
        p14_archive: str = str(core.DEFAULT_P14_ARCHIVE),
        p15_bundle: str = str(core.DEFAULT_P15_BUNDLE),
        p16_bundle: str = str(p16.DEFAULT_P16_BUNDLE),
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
                        "external_data": False,
                        "augmentation": False,
                    },
                    indent=2,
                )
            )
            return
        if selected == "status":
            payload = core._read_volume_json(
                task2_volume, "p19/results/p19_run_summary.json"
            )
            print("MODAL_P19_STATUS=" + json.dumps(payload or {"status": "NO_RESULT"}))
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
            raise RuntimeError("Modal P19 model preparation failed")
        if selected == "prepare":
            print("MODAL_P19_PREPARE_COMPLETE")
            return
        result = dev_sweep.remote() if selected == "dev" else run_all.remote()
        allowed = (
            {"DEV_PASS", "DEV_REJECTED"}
            if selected == "dev"
            else {"DEV_REJECTED", "FULL_HELDOUT_REJECTED", "PUBLIC_CANDIDATE_READY"}
        )
        if result.get("status") not in allowed:
            raise RuntimeError("Modal P19 run failed: " + json.dumps(result))
        _download(download_dir)

else:
    app = None
    task2_volume = None


if __name__ == "__main__" and modal is None:
    raise SystemExit(
        "Modal SDK is missing. Install it with: python -m pip install -U modal"
    )
