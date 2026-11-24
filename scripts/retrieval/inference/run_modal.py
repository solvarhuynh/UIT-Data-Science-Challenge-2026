"""Build selected retrieval rankings and BTC-compatible output packages."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = Path("/workspace")
VOLUME_NAME = "udsc"
VOLUME_PATH = "/mnt/udsc"

volume = modal.Volume.from_name(VOLUME_NAME)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "faiss-cpu>=1.8,<2",
        "accelerate>=1.1,<2",
        "modal==1.6.4",
        "numpy>=1.26,<2",
        "pydantic>=2.8",
        "pyvi>=0.1,<1",
        "pyyaml>=6",
        "scikit-learn>=1.5,<2",
        "sentence-transformers==5.6.0",
        "sentencepiece>=0.2,<1",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "tqdm>=4.66,<5",
        "transformers==5.23.0",
    )
    .env(
        {
            "PYTHONPATH": "/workspace/src",
            "UDSC2026_CONFIG_DIR": "/workspace/configs",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/workspace/src")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/workspace/configs")
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "retrieval" / "inference"),
        remote_path="/workspace/scripts/retrieval/inference",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "retrieval" / "training"),
        remote_path="/workspace/scripts/retrieval/training",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "data_prep"),
        remote_path="/workspace/scripts/data_prep",
    )
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "modal==1.6.4",
        "numpy>=1.26,<2",
        "pydantic>=2.8",
        "pyyaml>=6",
        "scikit-learn>=1.5,<2",
    )
    .env({"PYTHONPATH": "/workspace/src"})
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/workspace/src")
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "retrieval" / "inference"),
        remote_path="/workspace/scripts/retrieval/inference",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "retrieval" / "evaluation"),
        remote_path="/workspace/scripts/retrieval/evaluation",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "retrieval" / "submission"),
        remote_path="/workspace/scripts/retrieval/submission",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "shared"),
        remote_path="/workspace/scripts/shared",
    )
)
app = modal.App("udsc-retrieval-pipeline")
TRAIN_POOL = Path(
    "data/predictions/selected/legalir/train/candidate_k500_rerank_k200"
)
TRAIN_SPLIT = Path("data/splits/retrieval/folds.json")
FINAL_TRAINING = Path("data/training/retrieval/production")
PRODUCTION_ROOT = Path(
    "data/predictions/selected/legalir/production/candidate_k500_rerank_k200"
)


def _legal_ir_dir() -> Path:
    path = Path("data/raw/btc/Legall-IR")
    if not path.is_dir():
        raise FileNotFoundError(
            f"canonical LegalIR directory is missing: {path}; "
            "do not create a duplicate LegalIR directory"
        )
    legacy = Path("data/raw/btc/LegalIR")
    if legacy.exists():
        raise ValueError(
            "duplicate LegalIR directories found; remove data/raw/btc/LegalIR"
        )
    return path


def _preflight_reranker_tokenizer() -> None:
    """Fail before retrieval work if the selected reranker tokenizer is unusable."""
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
    image=gpu_image,
    gpu=["A10", "L4"],
    cpu=4,
    memory=16384,
    timeout=12 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def generate_rankings(
    dataset: str = "train",
    candidate_k: int = 500,
    rerank_k: int = 200,
    resume: bool = False,
) -> None:
    if dataset not in {"train", "public"}:
        raise ValueError("dataset must be 'train' or 'public'")
    if candidate_k <= 0 or rerank_k <= 0 or rerank_k > candidate_k:
        raise ValueError(
            "candidate_k and rerank_k must be positive, with rerank_k <= candidate_k"
        )
    os.chdir(VOLUME_PATH)
    _preflight_reranker_tokenizer()
    source_questions = _legal_ir_dir() / (
        "train.json" if dataset == "train" else "public-official.json"
    )
    if not source_questions.is_file():
        raise FileNotFoundError(f"LegalIR question file is missing: {source_questions}")
    output_root = Path("data/predictions/selected/legalir") / dataset
    if (candidate_k, rerank_k) != (200, 100):
        output_root /= f"candidate_k{candidate_k}_rerank_k{rerank_k}"
    output_root.mkdir(parents=True, exist_ok=True)
    source_payload = json.loads(source_questions.read_text(encoding="utf-8-sig"))
    if not isinstance(source_payload, dict) or not source_payload:
        raise ValueError("LegalIR question source must be a non-empty ID mapping")
    questions = output_root / "questions.json"
    question_mapping = {
        question_id: {"question": record["question"]}
        for question_id, record in source_payload.items()
        if isinstance(record, dict) and isinstance(record.get("question"), str)
    }
    if len(question_mapping) != len(source_payload):
        raise ValueError("LegalIR question source contains invalid question text")
    questions.write_text(
        json.dumps(question_mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dense = output_root / "dense.jsonl"
    dense_manifest = output_root / "dense.manifest.json"
    reranked = output_root / "reranked.jsonl"
    reranker_manifest = output_root / "reranked.manifest.json"

    command = [
        sys.executable,
        "/workspace/scripts/retrieval/inference/generate_candidates.py",
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
        str(dense),
        "--manifest",
        str(dense_manifest),
    ]
    command.extend(["--question-mapping", str(questions.resolve())])
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True, cwd=VOLUME_PATH)

    command = [
        sys.executable,
        "/workspace/scripts/retrieval/inference/rerank_candidates.py",
        "--questions",
        str(questions.resolve()),
        "--candidates",
        str(dense),
        "--candidate-manifest",
        str(dense_manifest),
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
    volume.commit()


@app.function(
    image=cpu_image,
    cpu=4,
    memory=16384,
    timeout=8 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def evaluate_or_package(
    dataset: str = "train",
    candidate_k: int = 500,
    rerank_k: int = 200,
) -> None:
    if dataset not in {"train", "public"}:
        raise ValueError("dataset must be 'train' or 'public'")
    if candidate_k <= 0 or rerank_k <= 0 or rerank_k > candidate_k:
        raise ValueError(
            "candidate_k and rerank_k must be positive, with rerank_k <= candidate_k"
        )
    os.chdir(VOLUME_PATH)
    question_path = _legal_ir_dir() / (
        "train.json" if dataset == "train" else "public-official.json"
    )
    output_root = Path("data/predictions/selected/legalir") / dataset
    variant_name = f"candidate_k{candidate_k}_rerank_k{rerank_k}"
    if (candidate_k, rerank_k) != (200, 100):
        output_root /= variant_name
    questions = output_root / "questions.json"
    dense = output_root / "dense.jsonl"
    reranked = output_root / "reranked.jsonl"
    combined = output_root / "ranking.json"

    def combine(
        questions: Path,
        dense_weight: float,
        reranker_weight: float,
        bm25_weight: float,
        output: Path,
    ) -> None:
        command = [
            sys.executable,
            "/workspace/scripts/retrieval/inference/combine_rankings.py",
            "--questions",
            str(questions),
            "--dense",
            str(dense),
            "--reranked",
            str(reranked),
            "--contexts-dir",
            "data/processed/documents",
            "--output",
            str(output),
            "--dense-weight",
            str(dense_weight),
            "--reranker-weight",
            str(reranker_weight),
            "--bm25-weight",
            str(bm25_weight),
            "--rrf-k",
            "60",
        ]
        subprocess.run(command, check=True, cwd=VOLUME_PATH)

    if dataset == "public":
        combine(questions, 0.5, 1.0, 0.3, combined)
        package = Path("data/submissions/retrieval/submission.zip")
        package.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/retrieval/submission/package_submission.py",
                "--input",
                str(combined),
                "--questions",
                str(question_path),
                "--output",
                str(package),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/retrieval/submission/validate_package.py",
                "--input",
                str(package),
                "--questions",
                str(question_path),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
    else:
        fold_root = Path("data/splits/retrieval")
        prediction_root = Path("data/predictions/selected/legalir/folds")
        report_root = Path("data/reports/retrieval/selected")
        if (candidate_k, rerank_k) != (200, 100):
            prediction_root /= variant_name
            report_root /= variant_name
        report_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/retrieval/evaluation/analyze_candidates.py",
                "--gold",
                str(question_path),
                "--predictions",
                str(dense),
                "--manifest",
                str(output_root / "dense.manifest.json"),
                "--strict-folds",
                "data/splits/retrieval/folds.json",
                "--output-dir",
                str(report_root / "candidate_analysis"),
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        variants = {
            "dense_only": (1.0, 0.0, 0.0),
            "reranker_only": (0.0, 1.0, 0.0),
            "hybrid": (0.5, 1.0, 0.3),
        }
        reports: dict[str, Path] = {}
        for variant, weights in variants.items():
            variant_root = prediction_root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            full_predictions = variant_root / "unpartitioned.json"
            # These rankings depend only on each question, fixed model outputs,
            # and the corpus. They use no fold labels, so build once and split
            # by question ID instead of re-reading the large candidate files
            # and document corpus separately for each fold.
            combine(questions, *weights, full_predictions)
            full_rows = json.loads(full_predictions.read_text(encoding="utf-8"))
            if not isinstance(full_rows, list):
                raise ValueError("combined predictions must be a JSON array")
            predictions_by_id = {
                row["id"]: row
                for row in full_rows
                if isinstance(row, dict) and isinstance(row.get("id"), str)
            }
            if len(predictions_by_id) != len(full_rows):
                raise ValueError(
                    "combined predictions contain invalid or duplicate IDs"
                )
            for fold in range(5):
                fold_dir = variant_root / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                fold_questions = json.loads(
                    (fold_root / f"fold_{fold}/questions.json").read_text(
                        encoding="utf-8"
                    )
                )
                if not isinstance(fold_questions, dict):
                    raise ValueError(f"fold {fold} questions must be an ID mapping")
                fold_ids = list(fold_questions)
                if not set(fold_ids) <= predictions_by_id.keys():
                    raise ValueError(
                        f"combined predictions are missing fold {fold} IDs"
                    )
                fold_rows = [predictions_by_id[question_id] for question_id in fold_ids]
                (fold_dir / "predictions.json").write_text(
                    json.dumps(fold_rows, ensure_ascii=False, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
            full_predictions.unlink()
            full_predictions.with_suffix(".manifest.json").unlink(missing_ok=True)
            merged = variant_root / "predictions.json"
            subprocess.run(
                [
                    sys.executable,
                    "/workspace/scripts/retrieval/evaluation/merge_fold_predictions.py",
                    "--folds-dir",
                    str(variant_root),
                    "--filename",
                    "predictions.json",
                    "--questions",
                    str(question_path),
                    "--fold-count",
                    "5",
                    "--output",
                    str(merged),
                ],
                check=True,
                cwd=VOLUME_PATH,
            )
            report = report_root / f"{variant}.json"
            reports[variant] = report
            subprocess.run(
                [
                    sys.executable,
                    "/workspace/scripts/retrieval/evaluation/score_rankings.py",
                    "--references",
                    str(question_path),
                    "--predictions",
                    str(merged),
                    "--output",
                    str(report),
                ],
                check=True,
                cwd=VOLUME_PATH,
            )

        comparisons = (
            ("dense_only", "reranker_only", "reranker_vs_dense"),
            ("dense_only", "hybrid", "hybrid_vs_dense"),
            ("reranker_only", "hybrid", "hybrid_vs_reranker"),
        )
        for baseline, candidate, name in comparisons:
            subprocess.run(
                [
                    sys.executable,
                    "/workspace/scripts/shared/compare_results.py",
                    "--task",
                    "legal-ir",
                    "--baseline",
                    str(reports[baseline]),
                    "--candidate",
                    str(reports[candidate]),
                    "--questions",
                    str(question_path),
                    "--output",
                    str(report_root / f"{name}.json"),
                ],
                check=True,
                cwd=VOLUME_PATH,
            )
    volume.commit()


@app.function(
    image=cpu_image,
    cpu=8,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def evaluate_bce_oof(loss: str = "bce") -> None:
    """Build and score complete out-of-fold retrieval predictions."""

    if loss not in {"bce", "softmax"}:
        raise ValueError("loss must be bce or softmax")

    os.chdir(VOLUME_PATH)
    question_path = _legal_ir_dir() / "train.json"
    dense = TRAIN_POOL / "dense.jsonl"
    contexts = Path("data/processed/documents")
    root = Path("data/predictions/selected/legalir/folds") / loss
    all_questions = json.loads(
        (TRAIN_POOL / "questions.json").read_text(encoding="utf-8")
    )
    fold_spec = json.loads(
        TRAIN_SPLIT.read_text(encoding="utf-8")
    )
    from udsc2026.evaluation.models import PredictionSample
    from udsc2026.evaluation.retrieval_candidates import aggregate_to_legal_ir_prediction
    for fold in range(5):
        fold_dir = root / f"fold_{fold}"
        manifest = json.loads(
            (fold_dir / "reranked.manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("complete") is not True or manifest.get("completed_count") != 1400:
            raise ValueError(f"BCE fold {fold} rerank is incomplete")
        fold_ids = fold_spec["folds"][fold]["validation_ids"]
        fold_questions = root / f"fold_{fold}" / "questions.json"
        fold_questions.write_text(
            json.dumps(
                {question_id: all_questions[question_id] for question_id in fold_ids},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        output = fold_dir / "predictions.json"
        rows = []
        seen = set()
        with (fold_dir / "reranked.jsonl").open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                sample = PredictionSample.model_validate_json(line, strict=True)
                if sample.question_id in seen:
                    raise ValueError(f"duplicate BCE fold {fold} question")
                seen.add(sample.question_id)
                prediction = aggregate_to_legal_ir_prediction(
                    sample,
                    method="rrf_cap2_dense_signal",
                    max_documents=5,
                    evidence_limit=3,
                    rrf_k=60,
                    dense_signal_weight=0.30,
                    score_field="rerank_score",
                )
                rows.append(prediction.model_dump(mode="json"))
        if seen != set(fold_ids):
            raise ValueError(f"BCE fold {fold} question coverage mismatch")
        output.write_text(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    merged = root / "predictions.json"
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/evaluation/merge_fold_predictions.py",
            "--folds-dir", str(root),
            "--filename", "predictions.json",
            "--questions", str(question_path),
            "--fold-count", "5",
            "--output", str(merged),
        ],
        check=True,
        cwd=VOLUME_PATH,
    )
    report = Path("data/reports/retrieval/selected") / f"{loss}_oof.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/evaluation/score_rankings.py",
            "--references", str(question_path),
            "--predictions", str(merged),
            "--output", str(report),
        ],
        check=True,
        cwd=VOLUME_PATH,
    )
    volume.commit()


@app.function(
    image=cpu_image,
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={VOLUME_PATH: volume},
    max_containers=1,
)
def prepare_union() -> None:
    """Build and score a label-free dense/lexical candidate union on CPU."""

    os.chdir(VOLUME_PATH)
    dense_root = TRAIN_POOL
    dense = dense_root / "dense.jsonl"
    dense_manifest = dense_root / "dense.manifest.json"
    questions = dense_root / "questions.json"
    chunks_archive = Path("data/processed/chunks_aiteam.tar.gz")
    gold = _legal_ir_dir() / "train.json"
    folds = TRAIN_SPLIT
    required = (dense, dense_manifest, questions, chunks_archive, gold, folds)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("union preflight missing: " + ", ".join(missing))

    output_root = Path("data/predictions/selected/legalir/union/candidate_k700_rerank_k200")
    union = output_root / "union.jsonl"
    union_manifest = output_root / "union.manifest.json"
    report_root = Path("data/reports/retrieval/selected/union_candidate")

    if union.is_file() or union_manifest.is_file():
        if not (union.is_file() and union_manifest.is_file()):
            raise RuntimeError(
                "union output is partial. Remove the incomplete union artifact before retrying"
            )
        manifest = json.loads(union_manifest.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise RuntimeError("union manifest exists but is incomplete")
        observed_hash = hashlib.sha256(union.read_bytes()).hexdigest()
        if observed_hash != manifest.get("output_sha256"):
            raise RuntimeError("union output hash does not match its manifest")
        print("Existing complete union artifact matches its manifest. Skipping rebuild.", flush=True)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/retrieval/inference/build_union_candidates.py",
                "--questions", str(questions.resolve()),
                "--dense", str(dense.resolve()),
                "--dense-manifest", str(dense_manifest.resolve()),
                "--chunks-archive", str(chunks_archive.resolve()),
                "--output", str(union.resolve()),
                "--manifest", str(union_manifest.resolve()),
                "--dense-k", "500",
                "--lexical-k", "200",
                "--candidate-k", "700",
                "--rrf-k", "60",
            ],
            check=True,
            cwd=VOLUME_PATH,
        )

    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/evaluation/analyze_candidates.py",
            "--gold", str(gold.resolve()),
            "--predictions", str(union.resolve()),
            "--manifest", str(union_manifest.resolve()),
            "--strict-folds", str(folds.resolve()),
            # The union is capped at 700, but a question can legitimately
            # have fewer unique chunks when the dense and lexical lists
            # overlap.  analyze_candidates adds the actual cached depth to
            # this list, so requesting 700 here would reject a valid output
            # such as a 502-hit union before producing any report.
            "--k", "5", "10", "20", "50", "100", "200", "500",
            "--output-dir", str(report_root),
            "--aggregation", "first_occurrence",
            "--score-field", "hybrid_score",
            "--final-documents", "5",
        ],
        check=True,
        cwd=VOLUME_PATH,
    )
    volume.commit()


@app.function(
    image=cpu_image,
    cpu=8,
    memory=32768,
    timeout=8 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def probe_slot5() -> None:
    """Run the leakage-safe slot-five rescue on the cached five-fold rerankings."""

    os.chdir(VOLUME_PATH)
    gold = _legal_ir_dir() / "train.json"
    folds = Path("data/splits/retrieval/folds.json")
    reranked_root = Path("data/predictions/selected/legalir/folds/softmax")
    output_root = Path("data/predictions/selected/legalir/slot5_guard")
    if not gold.is_file():
        raise FileNotFoundError(f"LegalIR train references are missing: {gold}")
    if not folds.is_file():
        raise FileNotFoundError(f"strict grouped folds are missing: {folds}")
    for fold in range(5):
        input_path = reranked_root / f"fold_{fold}" / "reranked.jsonl"
        manifest_path = reranked_root / f"fold_{fold}" / "reranked.manifest.json"
        if not input_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"completed softmax rerank artifact is missing for fold {fold}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True or manifest.get("completed_count") != 1400:
            raise ValueError(f"softmax fold {fold} rerank is not complete")
        if manifest.get("candidate_k") != 500 or manifest.get("rerank_k") != 200:
            raise ValueError(f"softmax fold {fold} has unexpected candidate/rerank depth")
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/evaluation/run_slot5_probe.py",
            "--gold",
            str(gold),
            "--folds",
            str(folds),
            "--reranked-root",
            str(reranked_root),
            "--output-root",
            str(output_root),
            "--overwrite",
        ],
        check=True,
        cwd=VOLUME_PATH,
    )
    volume.commit()


@app.function(
    image=cpu_image,
    cpu=4,
    memory=16384,
    timeout=8 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def aggregate_final(
    dataset: str = "public",
    candidate_k: int = 500,
    rerank_k: int = 200,
) -> None:
    """Apply the locked CPU recipe and validate the LegalIR package."""

    if dataset not in {"train", "public"}:
        raise ValueError("dataset must be 'train' or 'public'")
    if candidate_k <= 0 or rerank_k <= 0 or rerank_k > candidate_k:
        raise ValueError("candidate_k and rerank_k must be positive, with rerank_k <= candidate_k")
    os.chdir(VOLUME_PATH)
    root = PRODUCTION_ROOT
    questions = root / "questions.json"
    reranked = root / "reranked.jsonl"
    reranked_manifest = root / "reranked.manifest.json"
    submission = root / "submission_candidates.json"
    contexts = root / "contexts.jsonl"
    context_manifest = root / "contexts.manifest.json"
    source_questions = _legal_ir_dir() / (
        "train.json" if dataset == "train" else "public-official.json"
    )
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/evaluation/aggregate_production.py",
            "--questions", str(questions),
            "--reranked", str(reranked),
            "--reranked-manifest", str(reranked_manifest),
            "--submission", str(submission),
            "--contexts", str(contexts),
            "--context-manifest", str(context_manifest),
            "--method", "rrf_cap2_dense_signal",
            "--score-field", "rerank_score",
            "--max-documents", "5",
            "--evidence-limit", "3",
            "--rrf-k", "60",
            "--dense-rrf-k", "30",
            "--dense-signal-weight", "0.30",
        ],
        check=True,
        cwd=VOLUME_PATH,
    )
    package = Path("data/submissions/retrieval/production.zip")
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/submission/package_submission.py",
            "--input", str(submission),
            "--questions", str(source_questions),
            "--output", str(package),
        ],
        check=True,
        cwd=VOLUME_PATH,
    )
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/submission/validate_package.py",
            "--input", str(package),
            "--questions", str(source_questions),
        ],
        check=True,
        cwd=VOLUME_PATH,
    )
    volume.commit()


@app.function(
    image=gpu_image,
    gpu=["A10", "L4"],
    cpu=4,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={VOLUME_PATH: volume},
    max_containers=1,
)
def train_and_rerank_fold(fold: int, loss: str, resume: bool) -> None:
    from udsc2026.training.provenance import read_json, sha256

    os.chdir(VOLUME_PATH)
    _preflight_reranker_tokenizer()
    dense = TRAIN_POOL / "dense.jsonl"
    questions = TRAIN_POOL / "questions.json"
    dense_manifest = TRAIN_POOL / "dense.manifest.json"
    base_manifest = TRAIN_POOL / "reranked.manifest.json"
    metadata = read_json(dense_manifest)
    base = read_json(base_manifest)
    if (
        sha256(dense) != metadata["output_sha256"]
        or sha256(questions) != metadata["question_source_sha256"]
        or sha256(dense_manifest) != base["candidate_manifest_sha256"]
    ):
        raise ValueError("cached candidates, questions and base reranker disagree")
    training = Path("data/training/retrieval") / loss / f"fold_{fold}"
    ranking = Path("data/predictions/selected/legalir/folds") / loss / f"fold_{fold}"
    resume_flag = ["--resume"] if resume else []
    training_report = training / "training_report.json"
    report = None
    if training_report.is_file():
        report = json.loads(training_report.read_text(encoding="utf-8"))
    if not (report and report.get("complete") is True):
        subprocess.run(
            [
                sys.executable, "/workspace/scripts/retrieval/training/train_reranker.py",
                "--groups-dir", "data/training/retrieval",
                "--folds", str(TRAIN_SPLIT),
                "--fold", str(fold),
                "--loss", loss,
                "--output-dir", str(training),
                "--modal-volume", VOLUME_NAME,
                *resume_flag,
            ],
            check=True,
        )
    if not training_report.is_file():
        raise FileNotFoundError(f"missing completed training report: {training_report}")
    ranking.mkdir(parents=True, exist_ok=True)
    rerank_command = [
        sys.executable,
        "/workspace/scripts/retrieval/inference/rerank_candidates.py",
        "--questions", str(questions.resolve()),
        "--candidates", str(dense.resolve()),
        "--candidate-manifest", str(dense_manifest.resolve()),
        "--training-report", str(training_report.resolve()),
        "--base-reranker-manifest", str(base_manifest.resolve()),
        "--folds", str(TRAIN_SPLIT),
        "--model", str((training / "checkpoint").resolve()),
        "--candidate-k", "500",
        "--rerank-k", "200",
        "--batch-size", "8",
        "--max-length", "1024",
        "--save-every", "500",
        "--commit-modal-volume", "--modal-volume-name", VOLUME_NAME,
        "--fp16",
        "--output", str((ranking / "reranked.jsonl").resolve()),
        "--manifest", str((ranking / "reranked.manifest.json").resolve()),
        *resume_flag,
    ]
    subprocess.run(rerank_command, check=True, cwd=VOLUME_PATH)
    volume.commit()


@app.function(
    image=gpu_image,
    gpu=["A10", "L4"],
    cpu=4,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={VOLUME_PATH: volume},
    max_containers=1,
)
def train_hard_negative_all(resume: bool = False) -> None:
    """Mine dense semantic confusers, train five held-out folds, and rerank them."""

    os.chdir(VOLUME_PATH)
    _preflight_reranker_tokenizer()
    groups_dir = Path("data/training/retrieval/hard_negative")
    if not (groups_dir / "groups.jsonl").is_file() or not (
        groups_dir / "manifest.json"
    ).is_file():
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/retrieval/training/prepare_hard_negative_training.py",
                "--train", "data/raw/btc/Legall-IR/train.json",
                "--folds", str(TRAIN_SPLIT),
                "--dense", str(TRAIN_POOL / "dense.jsonl"),
                "--processed-root", "data/processed",
                "--base-groups", "data/training/retrieval/groups.jsonl",
                "--base-manifest", "data/training/retrieval/manifest.json",
                "--output-dir", str(groups_dir),
                "--max-hard", "2",
                "--max-semi-hard", "2",
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        volume.commit()

    base_manifest = TRAIN_POOL / "reranked.manifest.json"
    for fold in range(5):
        training = groups_dir / f"fold_{fold}"
        training_report = training / "training_report.json"
        resume_flag = ["--resume"] if resume else []
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/retrieval/training/train_reranker.py",
                "--groups-dir", str(groups_dir),
                "--folds", str(TRAIN_SPLIT),
                "--fold", str(fold),
                "--loss", "softmax",
                "--output-dir", str(training),
                "--modal-volume", VOLUME_NAME,
                *resume_flag,
            ],
            check=True,
            cwd=VOLUME_PATH,
        )
        output_dir = Path(
            "data/predictions/selected/legalir/folds"
        ) / "hard_negative" / f"fold_{fold}"
        output_dir.mkdir(parents=True, exist_ok=True)
        rerank_command = [
            sys.executable,
            "/workspace/scripts/retrieval/inference/rerank_candidates.py",
            "--questions", str((TRAIN_POOL / "questions.json").resolve()),
            "--candidates", str((TRAIN_POOL / "dense.jsonl").resolve()),
            "--candidate-manifest", str((TRAIN_POOL / "dense.manifest.json").resolve()),
            "--training-report", str(training_report.resolve()),
            "--base-reranker-manifest", str(base_manifest.resolve()),
            "--folds", str(TRAIN_SPLIT),
            "--model", str((training / "checkpoint").resolve()),
            "--candidate-k", "500",
            "--rerank-k", "200",
            "--batch-size", "8",
            "--max-length", "1024",
            "--save-every", "500",
            "--commit-modal-volume", "--modal-volume-name", VOLUME_NAME,
            "--fp16",
            "--output", str((output_dir / "reranked.jsonl").resolve()),
            "--manifest", str((output_dir / "reranked.manifest.json").resolve()),
            *resume_flag,
        ]
        subprocess.run(rerank_command, check=True, cwd=VOLUME_PATH)
        volume.commit()


@app.function(
    image=gpu_image,
    gpu=["A10", "L4"],
    cpu=4,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={VOLUME_PATH: volume},
    max_containers=1,
)
def train_and_rerank_final(
    dataset: str = "public",
    candidate_k: int = 500,
    rerank_k: int = 200,
    resume: bool = False,
) -> None:
    """Train on every usable grouped train query, then rank the output mapping."""

    from udsc2026.training.provenance import read_json

    if dataset not in {"train", "public"}:
        raise ValueError("dataset must be 'train' or 'public'")
    if candidate_k <= 0 or rerank_k <= 0 or rerank_k > candidate_k:
        raise ValueError("candidate_k and rerank_k must be positive, with rerank_k <= candidate_k")

    os.chdir(VOLUME_PATH)
    _preflight_reranker_tokenizer()
    FINAL_TRAINING.mkdir(parents=True, exist_ok=True)
    training_report = FINAL_TRAINING / "training_report.json"
    resume_flag = ["--resume"] if resume else []
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/retrieval/training/train_reranker.py",
            "--groups-dir", "data/training/retrieval",
            "--folds", str(TRAIN_SPLIT),
            "--all-queries",
            "--loss", "softmax",
            "--output-dir", str(FINAL_TRAINING),
            "--modal-volume", VOLUME_NAME,
            *resume_flag,
        ],
        check=True,
    )
    report = read_json(training_report)
    if report.get("scope") != "all" or report.get("complete") is not True:
        raise ValueError("final reranker training did not produce a complete all-query report")

    source_questions = _legal_ir_dir() / (
        "train.json" if dataset == "train" else "public-official.json"
    )
    source_payload = json.loads(source_questions.read_text(encoding="utf-8-sig"))
    if not isinstance(source_payload, dict) or not source_payload:
        raise ValueError("LegalIR question source must be a non-empty ID mapping")
    PRODUCTION_ROOT.mkdir(parents=True, exist_ok=True)
    questions = PRODUCTION_ROOT / "questions.json"
    questions.write_text(
        json.dumps(
            {
                question_id: {"question": record["question"]}
                for question_id, record in source_payload.items()
                if isinstance(record, dict) and isinstance(record.get("question"), str)
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    dense = PRODUCTION_ROOT / "dense.jsonl"
    dense_manifest = PRODUCTION_ROOT / "dense.manifest.json"
    reranked = PRODUCTION_ROOT / "reranked.jsonl"
    reranked_manifest = PRODUCTION_ROOT / "reranked.manifest.json"
    candidate_command = [
        sys.executable,
        "/workspace/scripts/retrieval/inference/generate_candidates.py",
        "--config-env", "gpu",
        "--candidate-k", str(candidate_k),
        "--query-batch-size", "64",
        "--save-every", "500",
        "--commit-modal-volume", "--modal-volume-name", VOLUME_NAME,
        "--question-mapping", str(questions.resolve()),
        "--output", str(dense),
        "--manifest", str(dense_manifest),
        *resume_flag,
    ]
    subprocess.run(candidate_command, check=True, cwd=VOLUME_PATH)
    rerank_command = [
        sys.executable,
        "/workspace/scripts/retrieval/inference/rerank_candidates.py",
        "--questions", str(questions.resolve()),
        "--candidates", str(dense),
        "--candidate-manifest", str(dense_manifest),
        "--training-report", str(training_report),
        "--model", str(FINAL_TRAINING / "checkpoint"),
        "--candidate-k", str(candidate_k),
        "--rerank-k", str(rerank_k),
        "--batch-size", "8",
        "--max-length", str(report["max_length"]),
        "--save-every", "500",
        "--commit-modal-volume", "--modal-volume-name", VOLUME_NAME,
        "--fp16",
        "--output", str(reranked),
        "--manifest", str(reranked_manifest),
        *resume_flag,
    ]
    subprocess.run(rerank_command, check=True, cwd=VOLUME_PATH)
    volume.commit()


@app.local_entrypoint()
def train_fold(fold: int, loss: str, resume: bool = False) -> None:
    if fold not in range(5) or loss not in {"softmax", "bce"}:
        raise ValueError("choose a fold from 0 to 4 and loss softmax or bce")
    train_and_rerank_fold.remote(fold, loss, resume)


@app.local_entrypoint()
def evaluate_bce() -> None:
    evaluate_bce_oof.remote()


@app.local_entrypoint()
def evaluate_oof(loss: str = "softmax") -> None:
    evaluate_bce_oof.remote(loss)


@app.local_entrypoint()
def union_prepare() -> None:
    prepare_union.remote()


@app.local_entrypoint()
def slot5_probe() -> None:
    probe_slot5.remote()


@app.local_entrypoint()
def train_hard_negative(resume: bool = False) -> None:
    train_hard_negative_all.remote(resume)


@app.local_entrypoint()
def train_final(
    dataset: str = "public",
    candidate_k: int = 500,
    rerank_k: int = 200,
    resume: bool = False,
) -> None:
    train_and_rerank_final.remote(dataset, candidate_k, rerank_k, resume)


@app.local_entrypoint()
def aggregate(
    dataset: str = "public",
    candidate_k: int = 500,
    rerank_k: int = 200,
) -> None:
    aggregate_final.remote(dataset, candidate_k, rerank_k)


@app.local_entrypoint()
def main(
    dataset: str = "train",
    candidate_k: int = 500,
    rerank_k: int = 200,
    resume: bool = False,
) -> None:
    generate_rankings.remote(
        dataset=dataset,
        candidate_k=candidate_k,
        rerank_k=rerank_k,
        resume=resume,
    )
    evaluate_or_package.remote(
        dataset=dataset,
        candidate_k=candidate_k,
        rerank_k=rerank_k,
    )
