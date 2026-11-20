"""Run the verified retrieval-to-answer pipeline on the selected GPU stack."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import modal

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = Path("/workspace")
VOLUME_NAME = "udsc"
VOLUME_PATH = "/mnt/udsc"

volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.1,<2",
        "faiss-cpu>=1.8,<2",
        "modal==1.6.4",
        "nltk>=3.8,<4",
        "numpy>=1.26,<2",
        "pydantic>=2.8",
        "pyvi>=0.1,<1",
        "pyyaml>=6",
        "rapidfuzz>=3.9,<4",
        "scikit-learn>=1.5,<2",
        "sentence-transformers==5.6.0",
        "sentencepiece>=0.2,<1",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "tqdm>=4.66,<5",
        "transformers==5.23.0",
    )
    .run_commands(
        "python -c \"import nltk; nltk.download('wordnet', quiet=True); "
        "nltk.download('omw-1.4', quiet=True)\""
    )
    .env(
        {
            "PYTHONPATH": "/workspace/src",
            "UDSC2026_CONFIG_DIR": "/workspace/configs",
            "TOKENIZERS_PARALLELISM": "false",
            "UDSC2026_PROMPTS_PATH": "/workspace/prompts",
            "LLM_DTYPE": "float16",
        }
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/workspace/src")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/workspace/configs")
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "retrieval" / "inference"),
        remote_path="/workspace/scripts/retrieval/inference",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "data_prep"),
        remote_path="/workspace/scripts/data_prep",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "answers" / "inference"),
        remote_path="/workspace/scripts/answers/inference",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "answers" / "evaluation"),
        remote_path="/workspace/scripts/answers/evaluation",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "answers" / "submission"),
        remote_path="/workspace/scripts/answers/submission",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "shared"),
        remote_path="/workspace/scripts/shared",
    )
    .add_local_dir(str(PROJECT_ROOT / "prompts"), remote_path="/workspace/prompts")
)

app = modal.App("udsc-answer-pipeline")


def _preflight_reranker_tokenizer() -> None:
    """Validate the selected cross-encoder tokenizer before GPU retrieval work."""
    model_path = Path("models/aiteam-vietnamese-reranker")
    if not model_path.is_dir():
        raise FileNotFoundError(f"selected reranker directory is missing: {model_path}")
    try:
        import sentencepiece  # noqa: F401
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        encoded = tokenizer("Kiểm tra tokenizer reranker.", add_special_tokens=True)
    except Exception as exc:
        raise RuntimeError(
            "Selected reranker tokenizer preflight failed. The Modal image must "
            "include sentencepiece and the mounted model must contain its tokenizer "
            f"files: {type(exc).__name__}: {exc}"
        ) from exc
    if not encoded.get("input_ids"):
        raise RuntimeError("Selected reranker tokenizer returned no token IDs")
    print(
        "Reranker tokenizer preflight passed "
        f"({type(tokenizer).__name__}, {len(encoded['input_ids'])} token(s)).",
        flush=True,
    )


@app.function(
    image=image,
    gpu=["A10", "L4"],
    cpu=4,
    memory=12288,
    timeout=24 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def run_pipeline(
    dataset: str = "train",
    limit: int = 500,
    save_every: int = 100,
    candidate_k: int = 500,
    rerank_k: int = 200,
    resume: bool = False,
) -> None:
    if save_every <= 0 or save_every > 500:
        raise ValueError("save-every must be from 1 to 500")
    if candidate_k <= 0 or rerank_k <= 0 or rerank_k > candidate_k:
        raise ValueError(
            "candidate-k and rerank-k must be positive, with rerank-k <= candidate-k"
        )
    if dataset not in {"train", "public"}:
        raise ValueError("dataset must be 'train' or 'public'")

    os.chdir(VOLUME_PATH)
    _preflight_reranker_tokenizer()
    source_questions = Path("data/raw/btc/LegalQA") / (
        "train.json" if dataset == "train" else "public-official.json"
    )
    if not source_questions.is_file():
        raise FileNotFoundError(f"LegalQA question file is missing: {source_questions}")
    source_payload = json.loads(source_questions.read_text(encoding="utf-8-sig"))
    if not isinstance(source_payload, dict) or not source_payload:
        raise ValueError("LegalQA questions must be a non-empty ID mapping")
    if dataset == "train":
        if limit <= 0 or limit > len(source_payload):
            raise ValueError(f"limit must be from 1 to {len(source_payload)}")
        selected_records = list(source_payload.items())
        if limit < len(selected_records):
            selected_records = random.Random(2026).sample(selected_records, limit)
        run_name = f"train_{len(selected_records)}"
    else:
        selected_records = list(source_payload.items())
        run_name = "public"
    run_name = f"{run_name}/candidate_k{candidate_k}_rerank_k{rerank_k}"
    candidate_root = Path("data/predictions/selected/legalqa") / run_name
    candidate_root.mkdir(parents=True, exist_ok=True)
    questions = candidate_root / "questions.json"
    query_mapping = {
        question_id: {"question": record["question"]}
        for question_id, record in selected_records
        if isinstance(record, dict) and isinstance(record.get("question"), str)
    }
    if len(query_mapping) != len(selected_records):
        raise ValueError("selected LegalQA records contain invalid question text")
    questions.write_text(
        json.dumps(query_mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    references: Path | None = None
    if dataset == "train":
        references = candidate_root / "references.json"
        references.write_text(
            json.dumps(dict(selected_records), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    candidates = candidate_root / "candidates.jsonl"
    candidate_manifest = candidate_root / "candidates.manifest.json"
    reranked = candidate_root / "reranked.jsonl"
    reranker_manifest = candidate_root / "reranked.manifest.json"
    answers = candidate_root / "answers.json"
    diagnostics = candidate_root / "answers.diagnostics.json"
    answer_manifest = candidate_root / "answers.manifest.json"

    command = [
        sys.executable,
        "/workspace/scripts/retrieval/inference/generate_candidates.py",
        "--question-mapping",
        str(questions),
        "--config-env",
        "gpu",
        "--candidate-k",
        str(candidate_k),
        "--query-batch-size",
        "64",
        "--save-every",
        "500",
        "--commit-modal-volume",
        "--modal-volume-name",
        VOLUME_NAME,
        "--output",
        str(candidates),
        "--manifest",
        str(candidate_manifest),
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True, cwd=VOLUME_PATH)

    command = [
        sys.executable,
        "/workspace/scripts/retrieval/inference/rerank_candidates.py",
        "--questions",
        str(questions),
        "--candidates",
        str(candidates),
        "--candidate-manifest",
        str(candidate_manifest),
        "--model",
        "models/aiteam-vietnamese-reranker",
        "--candidate-k",
        str(candidate_k),
        "--rerank-k",
        str(rerank_k),
        "--batch-size",
        "8",
        "--max-length",
        "1024",
        "--save-every",
        "500",
        "--commit-modal-volume",
        "--modal-volume-name",
        VOLUME_NAME,
        "--fp16",
        "--output",
        str(reranked),
        "--manifest",
        str(reranker_manifest),
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True, cwd=VOLUME_PATH)

    parents_archive = Path("data/processed/parents_aiteam.tar.gz")
    if not parents_archive.is_file():
        raise FileNotFoundError(f"parent archive is missing: {parents_archive}")
    extraction_root = Path("/tmp/udsc2026-answer-parents")
    parents_dir = extraction_root / "parents"
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True)
    with tarfile.open(parents_archive, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError(f"unsafe parent archive entry: {member.name}")
        archive.extractall(extraction_root, filter="data")

    command = [
        sys.executable,
        "/workspace/scripts/answers/inference/generate_answers.py",
        "--questions",
        str(questions),
        "--contexts",
        str(reranked),
        "--context-manifest",
        str(reranker_manifest),
        "--output",
        str(answers),
        "--diagnostics",
        str(diagnostics),
        "--manifest",
        str(answer_manifest),
        "--config",
        str(PROJECT_ROOT / "configs/gpu.yaml"),
        "--save-every",
        str(save_every),
        "--parents-dir",
        str(parents_dir),
        "--parent-archive",
        str(parents_archive),
        "--commit-modal-volume",
        "--modal-volume-name",
        VOLUME_NAME,
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True, cwd=VOLUME_PATH)

    if dataset == "train":
        assert references is not None
        report_root = Path("data/reports/answers/selected") / run_name
        report_root.mkdir(parents=True, exist_ok=True)
        score_report = report_root / "qwen_metrics.json"
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/answers/evaluation/score_answers.py",
                "--references",
                str(references),
                "--predictions",
                str(answers),
                "--output",
                str(score_report),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        extractive = candidate_root / "extractive_answers.json"
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/answers/inference/build_extractive_answers.py",
                "--dense",
                str(candidates),
                "--reranked",
                str(reranked),
                "--parents-dir",
                str(parents_dir),
                "--question-mapping",
                str(questions),
                "--output",
                str(extractive),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        extractive_report = report_root / "extractive_metrics.json"
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/answers/evaluation/score_answers.py",
                "--references",
                str(references),
                "--predictions",
                str(extractive),
                "--output",
                str(extractive_report),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/shared/compare_results.py",
                "--task",
                "legal-qa",
                "--baseline",
                str(extractive_report),
                "--candidate",
                str(score_report),
                "--questions",
                str(questions),
                "--output",
                str(report_root / "qwen_vs_extractive.json"),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        selector_dir = candidate_root / "selector"
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/answers/evaluation/select_answers.py",
                "--questions",
                str(questions),
                "--references",
                str(references),
                "--qwen",
                str(answers),
                "--extractive",
                str(extractive),
                "--fold-manifest",
                "data/splits/answers/folds.json",
                "--output-dir",
                str(selector_dir),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        selector_predictions = selector_dir / "selector_oof_predictions.json"
        selector_report = report_root / "selector_metrics.json"
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/answers/evaluation/score_answers.py",
                "--references",
                str(references),
                "--predictions",
                str(selector_predictions),
                "--output",
                str(selector_report),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        for baseline, comparison in (
            (extractive_report, "selector_vs_extractive.json"),
            (score_report, "selector_vs_qwen.json"),
        ):
            subprocess.run(
                [
                    sys.executable,
                    "/workspace/scripts/shared/compare_results.py",
                    "--task",
                    "legal-qa",
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(selector_report),
                    "--questions",
                    str(questions),
                    "--output",
                    str(report_root / comparison),
                ],
                check=True,
                cwd=VOLUME_PATH,
            )
    else:
        package = Path("data/submissions/answers/submission.zip")
        package.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/answers/submission/package_submission.py",
                "--input",
                str(answers),
                "--questions",
                str(questions),
                "--output",
                str(package),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/answers/submission/validate_package.py",
                "--input",
                str(package),
                "--questions",
                str(questions),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
    volume.commit()


@app.local_entrypoint()
def main(
    dataset: str = "train",
    limit: int = 500,
    save_every: int = 100,
    candidate_k: int = 500,
    rerank_k: int = 200,
    resume: bool = False,
) -> None:
    run_pipeline.remote(
        dataset=dataset,
        limit=limit,
        save_every=save_every,
        candidate_k=candidate_k,
        rerank_k=rerank_k,
        resume=resume,
    )
