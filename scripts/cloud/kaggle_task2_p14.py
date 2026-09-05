"""Run the Task 2 P14 pipeline on Kaggle without Beam-specific services.

The script consumes the existing ``task2_p14_beam_input.tar.zst`` because that
archive is provider-neutral: it contains only repository-relative data,
candidate rankings, and the parent checkpoint.  Runtime outputs are written to
Kaggle's persistent notebook output directory and every completed stage is
skipped on a later invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_REPO = "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/task2/kaggle_p14"
ARCHIVE_NAME = "task2_p14_beam_input.tar.zst"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qwen-dir", type=Path)
    parser.add_argument(
        "--stage",
        choices=("preflight", "parent", "qwen", "strict", "public", "all"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Do not use torch DataParallel even when Kaggle exposes two GPUs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _find_archive(explicit: Path | None) -> Path:
    if explicit is not None:
        archive = explicit.expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Task 2 archive is missing: {archive}")
        return archive
    candidates = [PROJECT_ROOT / f"artifacts/task2/{ARCHIVE_NAME}"]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(sorted(kaggle_input.rglob(ARCHIVE_NAME)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot find {ARCHIVE_NAME}; pass its Kaggle path with --archive"
    )


def _extract_input(archive: Path, workspace: Path, *, dry_run: bool) -> None:
    marker = workspace / ".task2_p14_kaggle_input.json"
    archive_hash = _sha256(archive)
    if marker.is_file() and _read_json(marker).get("archive_sha256") == archive_hash:
        print(f"KAGGLE_INPUT_READY sha256={archive_hash}", flush=True)
        return
    if dry_run:
        print(f"WOULD_EXTRACT {archive} -> {workspace}", flush=True)
        return
    try:
        import zstandard
    except ImportError as exc:
        raise ImportError(
            "install zstandard before extracting the Task 2 input"
        ) from exc
    workspace.mkdir(parents=True, exist_ok=True)
    print(
        f"extracting {archive} ({archive.stat().st_size / 1024**3:.2f} GiB)",
        flush=True,
    )
    with archive.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as bundle:
                bundle.extractall(workspace, filter="data")
    _write_json(marker, {"archive": str(archive), "archive_sha256": archive_hash})
    print("KAGGLE_INPUT_EXTRACTED", flush=True)


def _environment(workspace: Path) -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH", "")
    python_roots = [str(workspace), str(workspace / "src")]
    if existing:
        python_roots.append(existing)
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(python_roots),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def _run(
    command: Sequence[str],
    *,
    workspace: Path,
    dry_run: bool,
    accepted: set[int] | None = None,
) -> int:
    print("RUN:\n" + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=workspace,
        env=_environment(workspace),
        check=False,
    )
    allowed = {0} if accepted is None else accepted
    print(f"PROCESS EXIT: {completed.returncode}", flush=True)
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, list(command))
    return completed.returncode


def _python(*arguments: str) -> list[str]:
    return [sys.executable, *arguments]


def _gpu_preflight(*, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"count": 2, "names": ["dry-run T4", "dry-run T4"]}
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle allocated no CUDA GPU; select GPU T4 x2")
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    capabilities = [torch.cuda.get_device_capability(index) for index in range(count)]
    manifest = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "count": count,
        "names": names,
        "capabilities": capabilities,
    }
    print("KAGGLE_GPU_JSON=" + json.dumps(manifest), flush=True)
    if any(capability < (7, 0) for capability in capabilities):
        raise RuntimeError(
            "this PyTorch build requires CUDA capability >= 7.0; use T4 x2, not P100"
        )
    probe = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    result = probe @ probe
    torch.cuda.synchronize()
    del probe, result
    torch.cuda.empty_cache()
    print("KAGGLE_CUDA_COMPUTE_PASS", flush=True)
    return manifest


def _download_qwen(qwen_dir: Path, *, dry_run: bool) -> None:
    required = (qwen_dir / "config.json", qwen_dir / "model.safetensors")
    if all(path.is_file() and path.stat().st_size > 0 for path in required):
        print(f"QWEN_READY {qwen_dir}", flush=True)
        return
    if dry_run:
        print(f"WOULD_DOWNLOAD {QWEN_REPO} -> {qwen_dir}", flush=True)
        return
    from huggingface_hub import snapshot_download

    qwen_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {QWEN_REPO}", flush=True)
    snapshot_download(repo_id=QWEN_REPO, local_dir=qwen_dir)
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("Qwen download did not produce the required model files")


def _clean_incomplete(directory: Path, completion: Path, *, dry_run: bool) -> None:
    if completion.is_file() or not directory.exists() or dry_run:
        return
    shutil.rmtree(directory)


def _score_parents(
    *,
    workspace: Path,
    checkpoint: Path,
    output: Path,
    public: bool,
    dry_run: bool,
) -> None:
    questions = (
        "data/raw/btc/LegalQA/public-official.json"
        if public
        else "data/raw/btc/LegalQA/train.json"
    )
    candidates = (
        ["artifacts/task2/public-bge-reranker-all/predictions_after.jsonl"]
        if public
        else [
            "artifacts/task2/train500-bge-reranker-all/predictions_after.jsonl",
            "artifacts/task2/train500b-bge-reranker-all/predictions_after.jsonl",
        ]
    )
    command = _python(
        "scripts/training/train_legal_qa_parent_crossencoder.py",
        "score",
        "--questions",
        questions,
    )
    for candidate in candidates:
        command.extend(("--candidates", candidate))
    command.extend(
        (
            "--parents-dir",
            "data/processed_v3/parents",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--candidate-k",
            "50",
            "--batch-size",
            "16",
            "--max-length",
            "256",
            "--device",
            "cuda",
            "--amp",
        )
    )
    _run(command, workspace=workspace, dry_run=dry_run)


def _run_parent(workspace: Path, output: Path, *, dry_run: bool) -> Path:
    parent = output / "parent_listwise"
    manifest = parent / "training_manifest.json"
    _clean_incomplete(parent, manifest, dry_run=dry_run)
    if not manifest.is_file():
        _run(
            _python(
                "scripts/training/train_legal_qa_parent_crossencoder.py",
                "train",
                "--train-data",
                "artifacts/task2/training/parent_ce_hn16_train.jsonl",
                "--eval-data",
                "artifacts/task2/training/parent_ce_hn16_eval.jsonl",
                "--model-dir",
                "artifacts/task2/models/dek21-parent-crossencoder-hn16-v2/checkpoint-best",
                "--output-dir",
                str(parent),
                "--objective",
                "listwise",
                "--target-temperature",
                "0.10",
                "--epochs",
                "3",
                "--learning-rate",
                "1e-5",
                "--batch-size",
                "2",
                "--eval-batch-size",
                "2",
                "--gradient-accumulation",
                "8",
                "--max-length",
                "256",
                "--device",
                "cuda",
                "--amp",
            ),
            workspace=workspace,
            dry_run=dry_run,
        )
    checkpoint = parent / "checkpoint-best"
    strict_rankings = output / "strict_rankings.jsonl"
    if not strict_rankings.is_file():
        _score_parents(
            workspace=workspace,
            checkpoint=checkpoint,
            output=strict_rankings,
            public=False,
            dry_run=dry_run,
        )
    extractive = output / "extractive_predictions.json"
    if not extractive.is_file():
        _run(
            _python(
                "scripts/submission/build_legal_qa_parent_scores.py",
                "--scores",
                str(strict_rankings),
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--output",
                str(extractive),
                "--top-parents",
                "2",
                "--max-parent-tokens",
                "512",
                "--max-total-tokens",
                "1024",
                "--ce-weight",
                "0.60",
                "--retrieval-weight",
                "0.40",
                "--rrf-k",
                "20",
            ),
            workspace=workspace,
            dry_run=dry_run,
        )
    metrics = output / "extractive_metrics.json"
    if not metrics.is_file():
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "evaluate",
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--predictions",
                str(extractive),
                "--output",
                str(metrics),
                "--minimum-meteor",
                "0",
            ),
            workspace=workspace,
            dry_run=dry_run,
        )
    return checkpoint


def _run_qwen(
    workspace: Path,
    output: Path,
    qwen_dir: Path,
    *,
    gpu_count: int,
    single_gpu: bool,
    epochs: int,
    dry_run: bool,
) -> Path:
    adapter = output / "qwen_lora"
    manifest = adapter / "training_manifest.json"
    _clean_incomplete(adapter, manifest, dry_run=dry_run)
    if not manifest.is_file():
        use_two = gpu_count >= 2 and not single_gpu
        command = _python(
            "scripts/training/finetune_task2_qwen_lora.py",
            "train",
            "--train-data",
            "artifacts/task2/training/task2_qwen_sft_train.jsonl",
            "--model-dir",
            str(qwen_dir),
            "--output-dir",
            str(adapter),
            "--epochs",
            str(epochs),
            "--batch-size",
            "2" if use_two else "1",
            "--gradient-accumulation",
            "8" if use_two else "16",
            "--learning-rate",
            "2e-4",
            "--max-length",
            "2048",
            "--max-context-tokens",
            "1200",
            "--max-answer-tokens",
            "768",
            "--lora-rank",
            "32",
            "--lora-alpha",
            "64",
            "--dtype",
            "float16",
            "--device",
            "cuda",
        )
        if use_two:
            command.append("--data-parallel")
        _run(command, workspace=workspace, dry_run=dry_run)
    return adapter


def _generate(
    *,
    workspace: Path,
    adapter: Path,
    rankings: Path,
    output: Path,
    diagnostics: Path,
    public: bool,
    dry_run: bool,
) -> None:
    command = _python(
        "scripts/training/finetune_task2_qwen_lora.py",
        "generate",
        "--questions",
        (
            "data/raw/btc/LegalQA/public-official.json"
            if public
            else "data/raw/btc/LegalQA/train.json"
        ),
    )
    if not public:
        command.extend(
            (
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
            )
        )
    command.extend(
        (
            "--rankings",
            str(rankings),
            "--model-dir",
            str(adapter),
            "--output",
            str(output),
            "--diagnostics",
            str(diagnostics),
        )
    )
    if public:
        command.extend(("--known-answers", "data/raw/btc/LegalQA/train.json"))
    command.extend(
        (
            "--batch-size",
            "2",
            "--top-parents",
            "3",
            "--max-parent-words",
            "768",
            "--max-context-tokens",
            "2400",
            "--max-new-tokens",
            "1024",
            "--ce-weight",
            "0.60",
            "--retrieval-weight",
            "0.40",
            "--rrf-k",
            "20",
            "--dtype",
            "float16",
            "--device",
            "cuda",
            "--resume",
        )
    )
    _run(command, workspace=workspace, dry_run=dry_run)


def _run_strict(
    workspace: Path,
    output: Path,
    adapter: Path,
    *,
    dry_run: bool,
) -> bool:
    predictions = output / "strict_predictions.json"
    diagnostics = output / "strict_diagnostics.json"
    if not predictions.is_file():
        _generate(
            workspace=workspace,
            adapter=adapter,
            rankings=output / "strict_rankings.jsonl",
            output=predictions,
            diagnostics=diagnostics,
            public=False,
            dry_run=dry_run,
        )
    metrics = output / "strict_metrics.json"
    if not metrics.is_file():
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "evaluate",
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--predictions",
                str(predictions),
                "--output",
                str(metrics),
                "--minimum-meteor",
                "0.60",
            ),
            workspace=workspace,
            dry_run=dry_run,
            accepted={0, 1},
        )
    if dry_run:
        return True
    return _read_json(metrics).get("status") == "PROMOTED"


def _run_public(
    workspace: Path,
    output: Path,
    checkpoint: Path,
    adapter: Path,
    *,
    dry_run: bool,
) -> Path:
    rankings = output / "public_rankings.jsonl"
    if not rankings.is_file():
        _score_parents(
            workspace=workspace,
            checkpoint=checkpoint,
            output=rankings,
            public=True,
            dry_run=dry_run,
        )
    predictions = output / "public_predictions.json"
    diagnostics = output / "public_diagnostics.json"
    if not predictions.is_file():
        _generate(
            workspace=workspace,
            adapter=adapter,
            rankings=rankings,
            output=predictions,
            diagnostics=diagnostics,
            public=True,
            dry_run=dry_run,
        )
    submission = output / "submission.zip"
    if not submission.is_file():
        _run(
            _python(
                "scripts/submission/write_legal_qa_submission.py",
                "--input",
                str(predictions),
                "--questions",
                "data/raw/btc/LegalQA/public-official.json",
                "--empty-answer-policy",
                "error",
                "--output",
                str(submission),
            ),
            workspace=workspace,
            dry_run=dry_run,
        )
        _run(
            _python(
                "scripts/submission/validate_legal_qa_submission.py",
                "--input",
                str(submission),
                "--questions",
                "data/raw/btc/LegalQA/public-official.json",
            ),
            workspace=workspace,
            dry_run=dry_run,
        )
    return submission


def _package_result(output: Path, summary: dict[str, Any], *, dry_run: bool) -> Path:
    archive = output / "task2_p14_kaggle_result.zip"
    if dry_run:
        return archive
    _write_json(output / "run_summary.json", summary)
    temporary = archive.with_name(f".{archive.name}.partial")
    temporary.unlink(missing_ok=True)
    include_roots = [output / "parent_listwise/checkpoint-best", output / "qwen_lora"]
    include_files = [
        output / "run_summary.json",
        output / "strict_rankings.jsonl",
        output / "extractive_predictions.json",
        output / "extractive_metrics.json",
        output / "strict_predictions.json",
        output / "strict_diagnostics.json",
        output / "strict_metrics.json",
        output / "public_rankings.jsonl",
        output / "public_predictions.json",
        output / "public_diagnostics.json",
        output / "submission.zip",
    ]
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as bundle:
        for path in include_files:
            if path.is_file():
                bundle.write(path, arcname=path.relative_to(output).as_posix())
        for root in include_roots:
            if root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file() and path.name != "training_state.pt":
                        bundle.write(path, arcname=path.relative_to(output).as_posix())
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Kaggle result archive is corrupt")
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    workspace = args.workspace.resolve()
    output = args.output_dir.resolve()
    qwen_dir = (
        args.qwen_dir.resolve()
        if args.qwen_dir is not None
        else workspace / "models/qwen3-legal"
    )
    try:
        archive = _find_archive(args.archive)
        _extract_input(archive, workspace, dry_run=args.dry_run)
        output.mkdir(parents=True, exist_ok=True)
        gpu = _gpu_preflight(dry_run=args.dry_run)
        if args.stage == "preflight":
            return 0
        checkpoint = output / "parent_listwise/checkpoint-best"
        adapter = output / "qwen_lora"
        if args.stage in {"parent", "all"}:
            checkpoint = _run_parent(workspace, output, dry_run=args.dry_run)
        if args.stage in {"qwen", "all"}:
            _download_qwen(qwen_dir, dry_run=args.dry_run)
            adapter = _run_qwen(
                workspace,
                output,
                qwen_dir,
                gpu_count=int(gpu["count"]),
                single_gpu=args.single_gpu,
                epochs=args.epochs,
                dry_run=args.dry_run,
            )
        promoted = False
        if args.stage in {"strict", "all"}:
            promoted = _run_strict(
                workspace, output, adapter, dry_run=args.dry_run
            )
        if args.stage == "public":
            strict_metrics = output / "strict_metrics.json"
            if not strict_metrics.is_file() or _read_json(strict_metrics).get(
                "status"
            ) != "PROMOTED":
                raise RuntimeError(
                    "strict METEOR gate has not passed; refusing public run"
                )
            promoted = True
        submission = None
        if args.stage in {"public", "all"} and promoted:
            submission = _run_public(
                workspace,
                output,
                checkpoint,
                adapter,
                dry_run=args.dry_run,
            )
        status = (
            "PUBLIC_CANDIDATE_READY"
            if submission is not None
            else "HELDOUT_REJECTED"
            if args.stage == "all" and not promoted
            else f"{args.stage.upper()}_COMPLETE"
        )
        summary = {
            "schema_version": "task2-p14-kaggle-v1",
            "status": status,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "archive_sha256": _sha256(archive),
            "gpu": gpu,
            "epochs": args.epochs,
            "data_parallel": int(gpu["count"]) >= 2 and not args.single_gpu,
            "output_dir": str(output),
            "submission": str(submission) if submission else None,
        }
        result = _package_result(output, summary, dry_run=args.dry_run)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"KAGGLE_RESULT={result}", flush=True)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Task 2 Kaggle failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
