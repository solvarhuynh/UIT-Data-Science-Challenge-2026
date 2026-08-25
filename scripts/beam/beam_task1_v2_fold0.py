"""Beam runner for one Task1 V2A train + fold0 inference/evaluation path.

The CLI preflight is local and CPU-only.  The default branch submits exactly
one remote Beam job when the user explicitly runs this file without
``--preflight``; this task never invokes that branch.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if "--preflight" not in sys.argv:
        raise

    class Image:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass

    class Volume:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass

    def function(**kwargs):  # type: ignore[no-redef]
        def decorate(fn):
            return fn

        return decorate


ROOT = Path(__file__).resolve().parents[2]
VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
LOCAL_V2 = ROOT / "scripts/beam/task1_v2"
CPU_PREP = ROOT / "scripts/beam/beam_task1_v2_prepare_cpu.py"
QUESTIONS = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
CANDIDATES = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
PAYLOADS = ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
BASE_MODEL = ROOT / "models/reranker"
OUT = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v2/fold0_v2a"
REMOTE_OUT = RUNTIME / "artifacts/task1/recovery_096/inference_matched_reranker_v2/fold0_v2a"

IMAGE = Image(
    python_version="python3.11",
    python_packages=[
        "torch",
        "transformers==5.0.0",
        "tokenizers",
        "safetensors",
        "accelerate>=1.1,<2",
        "numpy",
        "tqdm",
        "pyvi",
    ],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prepared_data(data_dir: Path, candidates: Path, evidence: Path) -> dict:
    report_path = data_dir / "docgroups_report.json"
    train_path = data_dir / "train_docgroups.jsonl"
    eval_path = data_dir / "fold0_eval_docgroups.jsonl"
    if any(not path.is_file() for path in [report_path, train_path, eval_path]):
        raise RuntimeError(f"prepared V2 data is incomplete: {data_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "CLEAN":
        raise RuntimeError(f"prepared V2 data is not clean: {report.get('status')}")
    if int(report.get("eval_query_count", -1)) != 1400:
        raise RuntimeError("prepared V2 eval query count is not 1400")
    if [int(value) for value in report.get("train_folds", [])] != [1, 2, 3, 4]:
        raise RuntimeError("prepared V2 train folds are not [1,2,3,4]")
    if int(report.get("missing_evidence_count", report.get("missing_payload_or_raw_text", -1))) != 0:
        raise RuntimeError("prepared V2 data has missing evidence")
    if float(report.get("true_s2_selector_share", -1.0)) != 1.0:
        raise RuntimeError("prepared V2 true-S2 share is not 1.0")
    expected = {
        "candidate_refs_full_sha256": sha256(candidates),
        "train_docgroups_sha256": sha256(train_path),
        "fold0_eval_docgroups_sha256": sha256(eval_path),
        "evidence_py_sha256": sha256(evidence),
    }
    mismatches = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
    if mismatches:
        raise RuntimeError(f"prepared V2 hash handoff mismatch: {mismatches}")
    return report


def preflight() -> dict[str, object]:
    required = [
        QUESTIONS,
        FOLDS,
        CANDIDATES,
        BASELINE,
        PAYLOADS,
        BASE_MODEL / "config.json",
        CPU_PREP,
        LOCAL_V2 / "__init__.py",
        LOCAL_V2 / "evidence.py",
        LOCAL_V2 / "losses.py",
        LOCAL_V2 / "build_v2_docgroups.py",
        LOCAL_V2 / "train_v2_fold0.py",
        LOCAL_V2 / "score_v2_fold0.py",
        LOCAL_V2 / "selective_fusion_v2.py",
        LOCAL_V2 / "evaluate_v2_fold0.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("V2 preflight missing: " + ", ".join(missing))
    compile_targets = [LOCAL_V2 / name for name in [
        "evidence.py", "losses.py", "build_v2_docgroups.py", "train_v2_fold0.py",
        "score_v2_fold0.py", "selective_fusion_v2.py", "evaluate_v2_fold0.py",
    ]] + [CPU_PREP, Path(__file__).resolve()]
    for path in compile_targets:
        subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2, select_true_s2_prepared
    from scripts.beam.task1_v2.build_v2_docgroups import choose_train_docs
    from scripts.beam.task1_v2.losses import multi_positive_pairwise_loss
    import torch

    toy = select_true_s2(
        "quyền sử dụng đất",
        [
            {"chunk_id": "z", "text": "Nội dung không liên quan đến thủ tục."},
            {"chunk_id": "a", "text": "Quyền sử dụng đất được cấp theo quy định pháp luật."},
        ],
    )
    if not toy or toy[0]["chunk_id"] != "a":
        raise AssertionError("true-S2 toy selector failed")
    prepared = prepare_document([
        {"chunk_id": "z", "text": "Nội dung không liên quan đến thủ tục."},
        {"chunk_id": "a", "text": "Quyền sử dụng đất được cấp theo quy định pháp luật."},
    ])
    if select_true_s2("quyền sử dụng đất", prepared.chunks) != select_true_s2_prepared("quyền sử dụng đất", prepared):
        raise AssertionError("prepared true-S2 selector is not equivalent to direct selector")
    positive, negative = choose_train_docs(
        [{"doc_id": "A", "union_rank": 1}, {"doc_id": "B", "union_rank": 2}, {"doc_id": "C", "union_rank": 3}],
        {"A", "B"},
        ["C"],
        1,
    )
    if {row["doc_id"] for row in positive} != {"A", "B"} or len(negative) != 1:
        raise AssertionError("multi-gold positive retention failed")
    good = multi_positive_pairwise_loss(torch.tensor([2.0, 1.5]), torch.tensor([-1.0, 0.0]))
    bad = multi_positive_pairwise_loss(torch.tensor([-1.0, 0.0]), torch.tensor([2.0, 1.5]))
    if not bool(good < bad):
        raise AssertionError("pairwise loss ordering failed")
    candidate_preflight = subprocess.run(
        [
            sys.executable,
            str(LOCAL_V2 / "build_v2_docgroups.py"),
            "--questions", str(QUESTIONS), "--folds", str(FOLDS),
            "--candidates", str(CANDIDATES), "--baseline", str(BASELINE),
            "--payloads", str(PAYLOADS), "--output-dir", str(OUT / "data"), "--preflight",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads((BASE_MODEL / "config.json").read_text(encoding="utf-8"))
    if int(config.get("num_labels", len(config.get("id2label", {})) or -1)) != 1:
        raise ValueError("models/reranker is not single-logit")
    if config.get("model_type") != "xlm-roberta":
        raise ValueError("models/reranker is not the expected XLM-R model")
    cpu_preflight = subprocess.run(
        [sys.executable, str(CPU_PREP), "--preflight"],
        check=True,
        capture_output=True,
        text=True,
    )
    runner_source = Path(__file__).read_text(encoding="utf-8")
    run_source = runner_source.rsplit("@function(", 1)[1]
    if "build_v2_docgroups.py" in run_source:
        raise AssertionError("GPU runner still materializes V2 data")
    if '"--batch-size", "4"' not in run_source or '"--gradient-checkpointing"' not in run_source:
        raise AssertionError("GPU runner memory-safe training settings missing")
    return {
        "status": "PREFLIGHT_PASS",
        "candidate_preflight": json.loads(candidate_preflight.stdout),
        "true_s2_toy_relevant_first": True,
        "multi_gold_all_positives_retained": True,
        "pairwise_good_loss_lower": True,
        "prepared_selector_equivalence": True,
        "cpu_preparation_preflight": json.loads(cpu_preflight.stdout),
        "gpu_runner_does_not_materialize": True,
        "gpu_runner_batch4_gradient_checkpointing": True,
        "shared_selector_import": select_true_s2.__name__ == "select_true_s2",
        "base_model_single_logit": True,
        "gpu_launched": False,
        "remote_submitted": False,
        "protected_artifacts_overwritten": False,
        "candidate_sha256": sha256(CANDIDATES),
    }


@function(
    name="udsc-task1-v2-fold0-score-first",
    cpu=8,
    memory="32Gi",
    gpu="RTX5090",
    image=IMAGE,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def run():
    required = [
        RUNTIME / "scripts/beam/task1_v2/evidence.py",
        RUNTIME / "scripts/beam/task1_v2/losses.py",
        RUNTIME / "scripts/beam/task1_v2/train_v2_fold0.py",
        RUNTIME / "scripts/beam/task1_v2/score_v2_fold0.py",
        RUNTIME / "scripts/beam/task1_v2/evaluate_v2_fold0.py",
        RUNTIME / "scripts/beam/task1_v2/selective_fusion_v2.py",
        RUNTIME / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl",
        RUNTIME / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
        RUNTIME / "models/reranker/config.json",
        REMOTE_OUT / "data/train_docgroups.jsonl",
        REMOTE_OUT / "data/fold0_eval_docgroups.jsonl",
        REMOTE_OUT / "data/docgroups_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("remote V2 input missing: " + ", ".join(missing))
    config = json.loads((RUNTIME / "models/reranker/config.json").read_text(encoding="utf-8"))
    if int(config.get("num_labels", len(config.get("id2label", {})) or -1)) != 1:
        raise RuntimeError("remote base reranker is not single-logit")
    if config.get("model_type") != "xlm-roberta":
        raise RuntimeError("remote base reranker is not XLM-R")
    validate_prepared_data(
        REMOTE_OUT / "data",
        RUNTIME / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl",
        RUNTIME / "scripts/beam/task1_v2/evidence.py",
    )
    if REMOTE_OUT.exists() and any(path.name != "data" for path in REMOTE_OUT.iterdir()):
        raise RuntimeError(f"refusing to reuse non-data V2 output: {REMOTE_OUT}")
    REMOTE_OUT.mkdir(parents=True, exist_ok=True)

    def call(command: list[str]) -> None:
        print("[V2 RUN] " + " ".join(map(str, command)), flush=True)
        result = subprocess.run(command, cwd=str(RUNTIME))
        if result.returncode:
            raise RuntimeError(f"V2 command failed ({result.returncode}): {command}")

    data_dir = REMOTE_OUT / "data"
    call([
        "python", "scripts/beam/task1_v2/train_v2_fold0.py",
        "--docgroups", str(data_dir / "train_docgroups.jsonl"),
        "--base-model", "models/reranker", "--output-dir", str(REMOTE_OUT / "model"),
        "--learning-rate", "5e-7", "--epochs", "1", "--batch-size", "4",
        "--batch-queries", "1", "--gradient-accumulation", "8", "--margin", "0.0",
        "--max-length", "512", "--gradient-checkpointing", "--allow-training",
    ])
    call([
        "python", "scripts/beam/task1_v2/score_v2_fold0.py",
        "--docgroups", str(data_dir / "fold0_eval_docgroups.jsonl"),
        "--model", str(REMOTE_OUT / "model"),
        "--doc-scores", str(REMOTE_OUT / "doc_scores.jsonl"),
        "--predictions", str(REMOTE_OUT / "raw_v2_predictions.jsonl"),
        "--batch-size", "16", "--max-length", "512",
    ])
    call([
        "python", "scripts/beam/task1_v2/evaluate_v2_fold0.py",
        "--raw-predictions", str(REMOTE_OUT / "raw_v2_predictions.jsonl"),
        "--doc-scores", str(REMOTE_OUT / "doc_scores.jsonl"),
        "--baseline", "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
        "--report", str(REMOTE_OUT / "final_report.json"),
        "--fusion-predictions", str(REMOTE_OUT / "best_selective_fusion_predictions.jsonl"),
    ])
    (REMOTE_OUT / "remote_run_manifest.json").write_text(
        json.dumps({
            "status": "COMPLETE",
            "gpu_launched": True,
            "remote_submitted": True,
            "output_dir": str(REMOTE_OUT),
            "selector": "true_s2_bm25_within_document_v2",
            "objective": "multi-positive document-level pairwise ranking",
            "raw_predictions": "raw_v2_predictions.jsonl",
            "selective_fusion": "diagnostic_only",
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print(json.dumps(preflight(), indent=2))
    else:
        run.remote()
