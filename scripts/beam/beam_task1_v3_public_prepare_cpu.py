"""CPU Beam launcher for public V3 candidate/shortlist preparation only."""
from __future__ import annotations

import json
import ast
import subprocess
import sys
from pathlib import Path

try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if "--preflight" not in sys.argv:
        raise
    class Image:
        def __init__(self, **kwargs): pass
    class Volume:
        def __init__(self, **kwargs): pass
    def function(**kwargs):
        return lambda fn: fn

ROOT = Path(__file__).resolve().parents[2]
# When this launcher is invoked as ``python scripts/beam/...py``, Python puts
# ``scripts/beam`` (not the repository root) on sys.path.  Add the root so the
# local read-only preflight can import the namespace package ``scripts.*``;
# Beam's remote wrapper already supplies the repository root, so this is
# harmless there.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
OUT = RUNTIME / "artifacts/task1/recovery_096/final_public_v3"
IMAGE = Image(python_version="python3.11", python_packages=["pyvi"])
LOCAL_PUBLIC = ROOT / "artifacts/task1/recovery_096/final_public_v3"


def _stage(status: str, **details: object) -> dict[str, object]:
    return {"status": status, **details}


def master_preflight() -> dict[str, object]:
    """Read-only DAG gate; it never loads models or launches another stage."""
    public = ROOT / "artifacts/task1/recovery_096/final_public_v3"
    raw = public / "public_raw_k500.jsonl"
    raw_manifest = public / "public_raw_k500_manifest.json"
    bge = public / "public_bge_k200_compatible.jsonl"
    bge_manifest = public / "public_bge_k200_compatible_manifest.json"
    union = public / "public_candidate_union.jsonl"
    union_manifest = public / "public_candidate_union_manifest.json"
    adaptive_report = public / "public_adaptive_k500_report.json"
    adaptive_output = public / "public_adaptive_k500.jsonl"
    shortlist = public / "public_shortlist_evidence.jsonl"
    train_report = ROOT / "artifacts/task1/recovery_096/v3_residual/frozen_features_report.json"
    v3a_manifest = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit_manifest.json"
    v3a_model = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib"
    baseline = ROOT / "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip"
    stages: dict[str, dict[str, object]] = {}
    try:
        manifest = json.loads(raw_manifest.read_text(encoding="utf-8")) if raw_manifest.is_file() else {}
        stages["fresh_public_k500"] = _stage("PASS" if raw.is_file() and int(manifest.get("question_count", -1)) == 1000 and int(manifest.get("candidate_k", -1)) == 500 else "MISSING", output=str(raw), manifest=str(raw_manifest))
    except (OSError, ValueError, TypeError):
        stages["fresh_public_k500"] = _stage("BLOCKED", reason="invalid or missing manifest")
    try:
        manifest = json.loads(bge_manifest.read_text(encoding="utf-8")) if bge_manifest.is_file() else {}
        stages["fresh_bge_k200_compatibility"] = _stage("PASS" if bge.is_file() and manifest.get("schema_version") == "public-bge-k200-compatible-v1" and manifest.get("status") == "PUBLIC_BGE_K200_COMPATIBLE_PASS" else "MISSING", output=str(bge), manifest=str(bge_manifest))
    except (OSError, ValueError, TypeError):
        stages["fresh_bge_k200_compatibility"] = _stage("BLOCKED", reason="invalid or missing compatible manifest")
    try:
        manifest = json.loads(union_manifest.read_text(encoding="utf-8")) if union_manifest.is_file() else {}
        stages["compatible_union"] = _stage("PASS" if union.is_file() and manifest.get("schema_version") == "public-candidate-union-v2" and manifest.get("source_keys") == ["adaptive_k500", "bm25", "knn_word", "knn_char"] else "MISSING", output=str(union), manifest=str(union_manifest))
    except (OSError, ValueError, TypeError):
        stages["compatible_union"] = _stage("BLOCKED", reason="invalid or missing union manifest")
    stages["adaptive_public"] = _stage("PASS" if adaptive_report.is_file() and adaptive_output.is_file() else "MISSING", report=str(adaptive_report), output=str(adaptive_output))
    stages["baseline_contract"] = _stage("PASS" if baseline.is_file() else "MISSING", baseline=str(baseline))
    stages["public_shortlist"] = _stage("PASS" if shortlist.is_file() else "MISSING", output=str(shortlist))
    try:
        from scripts.beam.task1_v3_residual.score_frozen_features import parity_smoke
        parity = parity_smoke(ROOT / "models/reranker", train_report)
        stages["frozen_scorer_train_parity"] = _stage(parity.get("status", "BLOCKED"), details=parity)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        stages["frozen_scorer_train_parity"] = _stage("BLOCKED", reason=f"parity smoke unavailable: {exc}")
    try:
        v3a = json.loads(v3a_manifest.read_text(encoding="utf-8")) if v3a_manifest.is_file() else {}
        replay = v3a.get("validation_replay", v3a.get("fold0_replay", {}))
        fit = v3a.get("production_final_fit", {})
        ready = v3a_model.is_file() and replay.get("status") == "PASS" and int(replay.get("mismatch_count", -1)) == 0 and fit.get("no_model_selection_performed") is True
        stages["v3a_final_fit"] = _stage("PASS" if ready else "BLOCKED", manifest=str(v3a_manifest), model=str(v3a_model))
    except (OSError, ValueError, TypeError):
        stages["v3a_final_fit"] = _stage("BLOCKED", reason="invalid or missing final-fit manifest")
    stages["public_policy_ready"] = _stage("MISSING", reason="public frozen features/policy output not yet produced")
    stages["submission_input_ready"] = _stage("MISSING", reason="public V3A prediction artifact not yet produced")
    return {
        "status": "MASTER_PREFLIGHT_PASS" if all(item["status"] == "PASS" for item in stages.values()) else "MASTER_PREFLIGHT_BLOCKED",
        "stages": stages,
        "labels_answers_gold_read": False,
        "model_loaded": False,
        "gpu_launched": False,
        "training": False,
        "retrieval_rerun": False,
        "remote_submitted": False,
    }


def _public_contract(root: Path) -> dict[str, object]:
    from scripts.beam.task1_v3_residual.prepare_public_v3 import verify_public_contract

    public = root / "artifacts/task1/recovery_096/final_public_v3"
    return verify_public_contract(
        root / "data/raw/btc/LegalIR/public-official.json",
        public / "public_candidate_union.jsonl",
        root / "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip",
        root / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json",
        public / "public_adaptive_k500_report.json",
        public / "public_adaptive_k500.jsonl",
        root / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_final_fit_manifest.json",
        root / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_final_fit.joblib",
        root / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit_manifest.json",
        root / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib",
        root / "artifacts/task1/recovery_096/v3_residual/frozen_features_report.json",
        root / "artifacts/task1/recovery_096/v3_residual/policy/policy_training_report.json",
        root / "artifacts/task1/recovery_096/public_anchor_093/producer_manifest.json",
        root / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
        public / "public_candidate_union_manifest.json",
        public / "public_raw_k500.jsonl",
    )


def preflight() -> dict[str, object]:
    required = [
        ROOT / "scripts/beam/task1_v3_residual/prepare_public_v3.py",
        ROOT / "scripts/beam/task1_v3_residual/build_actions.py",
        ROOT / "scripts/beam/task1_v3_residual/common.py",
        ROOT / "scripts/beam/task1_v2/evidence.py",
        ROOT / "scripts/beam/beam_task1_v3_public_adaptive_k200.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("public V3 CPU source missing:\n" + "\n".join(missing))
    source = (ROOT / "scripts/beam/task1_v3_residual/prepare_public_v3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    if imported_modules.intersection({"torch", "transformers"}):
        raise AssertionError("public preparation must not load a neural model or GPU")
    contract = _public_contract(ROOT)
    production_ready = bool(contract["production_adapter_verified"])
    result = {
        "status": "PREFLIGHT_PASS" if production_ready else "PREFLIGHT_BLOCKED_UNVERIFIED_CONTRACT",
        "cpu": 16,
        "memory": "64Gi",
        "production_adapter_verified": production_ready,
        "public_v3a_feature_contract_reproducible": bool(contract["checks"].get("adaptive_feature_parity") and contract["checks"].get("train_public_schema_parity")),
        "real_train_parity_pass": bool(contract["checks"].get("real_train_parity_pass")),
        "remote_execution_allowed": production_ready,
        "missing": contract["missing"],
        "contract_checks": contract["checks"],
        "gpu_parameter_present": False,
        "model_loading": False,
        "retrieval_rerun": False,
        "remote_submitted": False,
    }
    result["master_dag"] = master_preflight()
    return result


def _sync_code() -> None:
    pairs = [
        ("scripts/beam/__init__.py", "scripts/beam/__init__.py"),
        ("scripts/beam/task1_v2/__init__.py", "scripts/beam/task1_v2/__init__.py"),
        ("scripts/beam/task1_v3_residual/prepare_public_v3.py", "scripts/beam/task1_v3_residual/prepare_public_v3.py"),
        ("scripts/beam/task1_v3_residual/build_actions.py", "scripts/beam/task1_v3_residual/build_actions.py"),
        ("scripts/beam/task1_v3_residual/common.py", "scripts/beam/task1_v3_residual/common.py"),
        ("scripts/beam/task1_v2/evidence.py", "scripts/beam/task1_v2/evidence.py"),
    ]
    for local_rel, remote_rel in pairs:
        local = ROOT / local_rel
        remote = RUNTIME / remote_rel
        remote.parent.mkdir(parents=True, exist_ok=True)
        remote.write_text(local.read_text(encoding="utf-8"), encoding="utf-8")


@function(name="udsc-task1-v3-public-prepare-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=0, headless=True)
def prepare_public_cpu() -> None:
    _sync_code()
    contract = _public_contract(RUNTIME)
    if not contract["production_adapter_verified"]:
        raise RuntimeError(json.dumps({"status": "PUBLIC_PREP_BLOCKED_UNVERIFIED_CONTRACT", **contract}))
    command = ["python", "scripts/beam/task1_v3_residual/prepare_public_v3.py", "--questions", "data/raw/btc/LegalIR/public-official.json", "--union", "artifacts/task1/recovery_096/final_public_v3/public_candidate_union.jsonl", "--union-manifest", "artifacts/task1/recovery_096/final_public_v3/public_candidate_union_manifest.json", "--baseline-zip", "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip", "--payloads", "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json", "--output-dir", "artifacts/task1/recovery_096/final_public_v3"]
    if subprocess.run(command, cwd=str(RUNTIME)).returncode:
        raise RuntimeError("public V3 preparation failed")


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print(json.dumps(preflight(), indent=2))
    else:
        prepare_public_cpu.remote()
