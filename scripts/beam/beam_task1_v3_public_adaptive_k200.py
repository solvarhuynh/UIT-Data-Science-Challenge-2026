"""Beam GPU producer for the fresh public BGE-compatible dense K200 cache.

This is an inference-only compatibility adapter.  It scores exactly the first
200 chunks from the fresh, label-free public raw K500 artifact with the
historical ``models/reranker`` contract.  It never reads train/gold/answer
fields, never retrains, and refuses to overwrite an existing output.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if "--preflight" not in sys.argv and "--self-test" not in sys.argv:
        raise

    class Image:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class Volume:
        def __init__(self, **kwargs: Any) -> None:
            pass

    def function(**kwargs: Any):
        return lambda fn: fn


ROOT = Path(__file__).resolve().parents[2]
VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
PUBLIC_QUESTIONS = RUNTIME / "data/raw/btc/LegalIR/public-official.json"
RAW_K500 = RUNTIME / "artifacts/task1/recovery_096/final_public_v3/public_raw_k500.jsonl"
PAYLOADS = RUNTIME / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
MODEL = RUNTIME / "models/reranker"
MODEL_CANDIDATES = (MODEL,)
OUTPUT = RUNTIME / "artifacts/task1/recovery_096/final_public_v3/public_bge_k200_compatible.jsonl"
MANIFEST = RUNTIME / "artifacts/task1/recovery_096/final_public_v3/public_bge_k200_compatible_manifest.json"
TEMPORARY_OUTPUT = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
EXPECTED_QUERIES = 1000
K200 = 200
MAX_LENGTH = 512
BATCH_SIZE = 64

# This is the historical ``models/reranker`` model used by the frozen V3
# features.  The remote resolver accepts only this exact contract.  The only
# accepted Beam location is deliberately canonical: no checked-in reranker
# checkpoint or historical archive was byte-proven equivalent during the
# 2026-08-24 forensic audit.
EXPECTED_MODEL_FILES = {
    "config.json": "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a",
    "model.safetensors": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
    "tokenizer.json": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
    "tokenizer_config.json": "7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
}
EXPECTED_MODEL_CONTRACT = {
    "model_type": "xlm-roberta",
    "architecture": "XLMRobertaForSequenceClassification",
    "tokenizer_class": "XLMRobertaTokenizer",
    "num_labels": 1,
    "max_length": MAX_LENGTH,
    "minimum_max_position_embeddings": MAX_LENGTH,
}

IMAGE = Image(
    python_version="python3.11",
    python_packages=["torch", "transformers==5.0.0", "accelerate>=1.1,<2"],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _log(message: str) -> None:
    print(message, flush=True)


def _model_fingerprint(model: Path) -> dict[str, Any]:
    """Return the exact load contract or raise without loading transformers."""
    missing = [name for name in EXPECTED_MODEL_FILES if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required model file(s): {', '.join(missing)}")
    observed_hashes = {name: sha256(model / name) for name in EXPECTED_MODEL_FILES}
    mismatches = {
        name: {"expected": EXPECTED_MODEL_FILES[name], "actual": observed_hashes[name]}
        for name in EXPECTED_MODEL_FILES
        if observed_hashes[name] != EXPECTED_MODEL_FILES[name]
    }
    if mismatches:
        raise ValueError(f"historical model SHA256 mismatch: {json.dumps(mismatches, sort_keys=True)}")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((model / "tokenizer_config.json").read_text(encoding="utf-8"))
    num_labels = int(config.get("num_labels", len(config.get("id2label", {}))))
    architecture = (config.get("architectures") or [None])[0]
    checks = {
        "model_type": config.get("model_type") == EXPECTED_MODEL_CONTRACT["model_type"],
        "architecture": architecture == EXPECTED_MODEL_CONTRACT["architecture"],
        "num_labels": num_labels == EXPECTED_MODEL_CONTRACT["num_labels"],
        "max_position_embeddings": int(config.get("max_position_embeddings", 0))
        >= EXPECTED_MODEL_CONTRACT["minimum_max_position_embeddings"],
        "tokenizer_class": tokenizer_config.get("tokenizer_class")
        == EXPECTED_MODEL_CONTRACT["tokenizer_class"],
        "tokenizer_max_length": int(tokenizer_config.get("model_max_length", 0)) >= MAX_LENGTH,
        "scoring_max_length": MAX_LENGTH == EXPECTED_MODEL_CONTRACT["max_length"],
    }
    if not all(checks.values()):
        raise ValueError(f"historical model contract mismatch: {json.dumps(checks, sort_keys=True)}")
    return {
        "path": str(model),
        "files_sha256": observed_hashes,
        "model_type": config["model_type"],
        "architecture": architecture,
        "num_labels": num_labels,
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "tokenizer_class": tokenizer_config["tokenizer_class"],
        "max_length": MAX_LENGTH,
    }


def _resolve_model() -> tuple[Path, dict[str, Any]]:
    """Resolve only a byte-proven historical model; never trust folder names."""
    failures: list[dict[str, str]] = []
    for model in MODEL_CANDIDATES:
        try:
            return model, _model_fingerprint(model)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"path": str(model), "reason": str(exc)})
    raise FileNotFoundError(
        "no hash-proven historical reranker at an accepted remote location: "
        + json.dumps(failures, sort_keys=True)
    )


def iter_payloads(path: Path) -> Iterable[dict[str, Any]]:
    """Stream the repository's large payloads object without loading it all."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            block = handle.read(1 << 20)
            if block:
                buffer += block
            else:
                eof = True

        while '"payloads"' not in buffer and not eof:
            refill()
        marker = buffer.find('"payloads"')
        if marker < 0:
            raise ValueError(f"payloads field missing: {path}")
        buffer = buffer[marker + len('"payloads"'):]
        while "{" not in buffer and not eof:
            refill()
        opening = buffer.find("{")
        if opening < 0:
            raise ValueError("payloads field is not an object")
        buffer = buffer[opening + 1:]
        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()
            if not buffer:
                raise ValueError("unexpected EOF in payloads object")
            if buffer[0] == "}":
                return
            while True:
                try:
                    _, position = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            buffer = buffer[position:].lstrip()
            if not buffer.startswith(":"):
                raise ValueError("invalid payload separator")
            buffer = buffer[1:].lstrip()
            while True:
                try:
                    value, position = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            if not isinstance(value, dict):
                raise ValueError("payload row is not an object")
            yield value
            buffer = buffer[position:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]


def _load_questions(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("public-official.json must be an object")
    result: dict[str, str] = {}
    for query_id, row in value.items():
        if not isinstance(row, dict) or "question" not in row:
            raise ValueError(f"public question lacks question text: {query_id}")
        result[str(query_id)] = str(row["question"])
    if len(result) != EXPECTED_QUERIES or len(result) != len(value):
        raise ValueError(f"public questions must contain {EXPECTED_QUERIES} unique IDs")
    return result


def _load_raw(path: Path, expected_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", row.get("question_id", "")))
            if not query_id or query_id in result:
                raise ValueError(f"raw K500 duplicate/empty query at line {line_number}: {query_id}")
            if query_id not in expected_ids:
                raise ValueError(f"raw K500 has non-public query ID: {query_id}")
            hits = row.get("hits")
            if not isinstance(hits, list) or len(hits) != 500:
                raise ValueError(f"raw K500 must contain exactly 500 hits: {query_id}")
            normalized: list[dict[str, Any]] = []
            for rank, hit in enumerate(hits, 1):
                if not isinstance(hit, dict) or not hit.get("doc_id") or not hit.get("chunk_id"):
                    raise ValueError(f"raw K500 hit lacks doc_id/chunk_id: {query_id}:{rank}")
                if int(hit.get("rank", rank)) != rank:
                    raise ValueError(f"raw K500 rank is not ordered 1-based: {query_id}:{rank}")
                normalized.append({
                    "doc_id": str(hit["doc_id"]),
                    "chunk_id": str(hit["chunk_id"]),
                    "dense_rank": rank,
                    "dense_score": float(hit.get("dense_score", hit.get("score", 0.0))),
                    "text": str(hit.get("text", "")),
                })
            result[query_id] = normalized[:K200]
    if set(result) != expected_ids or len(result) != EXPECTED_QUERIES:
        raise ValueError("fresh public raw K500 coverage does not match public questions")
    return result


def _validate_scored_cache(path: Path, expected_ids: set[str]) -> None:
    """Validate a completed K200 cache before publishing or recovering it."""
    observed_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", ""))
            if not query_id or query_id in observed_ids or row.get("question_id") != query_id:
                raise ValueError(f"invalid scored cache query row at line {line_number}")
            if set(row) & {"answer", "gold", "label", "fold"}:
                raise ValueError(f"scored cache contains forbidden field at line {line_number}")
            hits = row.get("hits")
            if not isinstance(hits, list) or len(hits) != K200:
                raise ValueError(f"scored cache must contain {K200} hits at line {line_number}")
            ranks = set()
            for hit in hits:
                if not isinstance(hit, dict) or not hit.get("doc_id") or not hit.get("chunk_id"):
                    raise ValueError(f"scored cache hit missing IDs at line {line_number}")
                rank = int(hit.get("dense_rank", 0))
                if rank < 1 or rank > K200 or rank in ranks or "bge_score" not in hit:
                    raise ValueError(f"scored cache hit contract invalid at line {line_number}")
                float(hit["bge_score"])
                ranks.add(rank)
            observed_ids.add(query_id)
    if observed_ids != expected_ids or len(observed_ids) != EXPECTED_QUERIES:
        raise ValueError("scored cache coverage does not match public questions")


def _publish_with_retry(temporary: Path, target: Path, *, label: str) -> None:
    """Retry transient Beam Volume rename failures without allowing overwrite."""
    for attempt in range(1, 9):
        if target.exists():
            raise RuntimeError(f"refusing to overwrite existing {label}: {target}")
        try:
            temporary.replace(target)
            return
        except BlockingIOError as exc:
            if attempt == 8:
                raise RuntimeError(f"unable to publish {label} after {attempt} attempts: {target}") from exc
            delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
            _log(f"[WRITE] {label} publish busy; retry {attempt}/8 in {delay:.1f}s")
            time.sleep(delay)


def _fill_missing_text(rows: dict[str, list[dict[str, Any]]], payloads: Path) -> None:
    wanted = {hit["chunk_id"] for hits in rows.values() for hit in hits if not hit["text"].strip()}
    if not wanted:
        _log("[PAYLOAD] raw K500 already contains all required chunk text")
        return
    _log(f"[PAYLOAD] indexing required chunk text: {len(wanted)} chunk(s)")
    found: dict[str, tuple[str, str]] = {}
    scanned = 0
    for payload in iter_payloads(payloads):
        scanned += 1
        chunk_id = str(payload.get("chunk_id", ""))
        if chunk_id in wanted:
            found[chunk_id] = (str(payload.get("doc_id", "")), str(payload.get("text", "")))
            if len(found) == len(wanted):
                break
        if scanned % 50000 == 0:
            _log(f"[PAYLOAD] resolved {len(found)}/{len(wanted)} after {scanned} payload rows")
    for query_id, hits in rows.items():
        for hit in hits:
            if hit["text"].strip():
                continue
            doc_id, text = found.get(hit["chunk_id"], ("", ""))
            if doc_id != hit["doc_id"] or not text.strip():
                raise ValueError(f"payload text missing or doc mismatch: {query_id}:{hit['chunk_id']}")
            hit["text"] = text
    _log(f"[PAYLOAD] resolved {len(found)}/{len(wanted)} required chunk text")


def _score(model_path: Path, questions: dict[str, str], rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for public BGE K200 inference")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to("cuda").eval()
    if int(getattr(model.config, "num_labels", -1)) != 1:
        raise ValueError("historical public BGE model must be single-logit")
    total_pairs = sum(len(hits) for hits in rows.values())
    _log(f"[SCORE] pairs={total_pairs}")
    outputs: list[dict[str, Any]] = []
    completed_pairs = 0
    last_reported = 0
    with torch.inference_mode():
        for query_id in sorted(rows, key=lambda value: int(value) if value.isdigit() else value):
            hits = rows[query_id]
            scores: list[float] = []
            for start in range(0, len(hits), BATCH_SIZE):
                batch = hits[start:start + BATCH_SIZE]
                encoded = tokenizer(
                    [questions[query_id]] * len(batch),
                    [hit["text"] for hit in batch],
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(**encoded).logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise ValueError(f"historical BGE scorer must emit [batch,1], got {tuple(logits.shape)}")
                scores.extend(float(value) for value in logits[:, 0].float().cpu().tolist())
                completed_pairs += len(batch)
                if completed_pairs - last_reported >= 10000 or completed_pairs == total_pairs:
                    _log(f"[SCORE] {completed_pairs}/{total_pairs}")
                    last_reported = completed_pairs
            scored = []
            for hit, score in zip(hits, scores):
                scored.append({key: value for key, value in hit.items() if key != "text"} | {"bge_score": score})
            scored.sort(key=lambda hit: (-float(hit["bge_score"]), int(hit["dense_rank"]), str(hit["doc_id"]), str(hit["chunk_id"])))
            outputs.append({"query_id": query_id, "question_id": query_id, "hits": scored})
    return outputs


def _remote_preflight() -> tuple[Path, dict[str, Any]]:
    """Check every remote prerequisite before parsing inputs or loading a model."""
    _log("[BOOT] container started")
    _log("[PREFLIGHT] PUBLIC_ADAPTIVE_K200_REMOTE_PREFLIGHT")
    blockers: list[dict[str, str]] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only in Beam image
        blockers.append({"check": "torch_import", "reason": repr(exc)})
    else:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        _log(f"[GPU] CUDA_VISIBLE_DEVICES={cuda_visible}")
        if not torch.cuda.is_available():
            blockers.append({"check": "cuda", "reason": "torch.cuda.is_available() is false"})
        else:
            _log(f"[GPU] {torch.cuda.get_device_name(0)}")
    if not VOLUME_ROOT.is_dir():
        blockers.append({"check": "volume_mount", "reason": f"missing: {VOLUME_ROOT}"})
    if not RUNTIME.is_dir():
        blockers.append({"check": "runtime", "reason": f"missing: {RUNTIME}"})
    else:
        _log("[VOLUME] runtime mounted")
    for name, path in (("public_official", PUBLIC_QUESTIONS), ("public_raw_k500", RAW_K500), ("payloads", PAYLOADS)):
        if not path.is_file():
            blockers.append({"check": name, "reason": f"missing: {path}"})
    if OUTPUT.exists() or MANIFEST.exists():
        blockers.append({"check": "overwrite_guard", "reason": f"output or manifest already exists: {OUTPUT} / {MANIFEST}"})
    model_path: Path | None = None
    fingerprint: dict[str, Any] | None = None
    try:
        model_path, fingerprint = _resolve_model()
        _log(
            "[MODEL] historical reranker verified: "
            + json.dumps(fingerprint, sort_keys=True)
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        blockers.append({"check": "historical_reranker", "reason": str(exc)})
    if blockers:
        raise RuntimeError(
            "PUBLIC_ADAPTIVE_K200_REMOTE_PREFLIGHT_BLOCKED "
            + json.dumps({"status": "PUBLIC_ADAPTIVE_K200_REMOTE_PREFLIGHT", "blockers": blockers}, sort_keys=True)
        )
    assert model_path is not None and fingerprint is not None
    _log("PUBLIC_ADAPTIVE_K200_REMOTE_PREFLIGHT_PASS")
    return model_path, fingerprint


def _beam_local_cli_environment() -> dict[str, Any]:
    """Report local CLI compatibility without opening a Beam connection."""
    packages: dict[str, str] = {}
    for package in ("beam-client", "beta9", "websockets"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    python_minor = sys.version_info[:2]
    compatible_python = python_minor in {(3, 11), (3, 12)}
    complete = all(value != "NOT_INSTALLED" for value in packages.values())
    marker = "BEAM_LOCAL_CLI_ENV_PASS" if compatible_python and complete else "BEAM_LOCAL_CLI_ENV_BLOCKED"
    return {
        "status": marker,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "beam_version": packages["beam-client"],
        "beta9_version": packages["beta9"],
        "websockets_version": packages["websockets"],
        "openssl_version": ssl.OPENSSL_VERSION,
        "reason": "Python 3.11 or 3.12 plus beam-client, beta9, and websockets are required for the dedicated local CLI environment.",
    }


def preflight() -> dict[str, Any]:
    required = [ROOT / "scripts/evaluation/generate_dense_candidates.py", ROOT / "scripts/beam/archive/run_adaptive_k500_rescue.py"]
    missing = [str(path) for path in required if not path.is_file()]
    local_inputs = {
        "public_questions": ROOT / "data/raw/btc/LegalIR/public-official.json",
        "fresh_public_k500": ROOT / "artifacts/task1/recovery_096/final_public_v3/public_raw_k500.jsonl",
        "payloads": ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json",
    }
    missing_inputs = [name for name, path in local_inputs.items() if not path.is_file()]
    local_model = ROOT / "models/reranker"
    try:
        local_fingerprint = _model_fingerprint(local_model)
        model_status = "PASS"
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        local_fingerprint = {"path": str(local_model), "error": str(exc)}
        model_status = "BLOCKED"
    cli_environment = _beam_local_cli_environment()
    return {
        "status": "PREFLIGHT_PASS" if not missing and not missing_inputs and model_status == "PASS" else "PREFLIGHT_BLOCKED",
        "gpu": "A10G",
        "required_source_files": [str(path) for path in required],
        "missing_source_files": missing,
        "local_input_files": {name: {"path": str(path), "present": path.is_file()} for name, path in local_inputs.items()},
        "missing_local_inputs": missing_inputs,
        "runtime_inputs": [str(PUBLIC_QUESTIONS), str(RAW_K500), str(PAYLOADS)],
        "expected_beam_volume_paths": {"volume_root": str(VOLUME_ROOT), "runtime": str(RUNTIME), "accepted_model_locations": [str(path) for path in MODEL_CANDIDATES]},
        "output": str(OUTPUT),
        "historical_model_local_provenance": {"status": model_status, "canonical_local_path": str(local_model), "fingerprint": local_fingerprint, "expected_remote_files_sha256": EXPECTED_MODEL_FILES},
        "remote_model_upload_required": True,
        "remote_model_upload_reason": "No checked-in/downstream reranker candidate was byte-proven equal to models/reranker; local preflight does not inspect the live Beam volume.",
        "beam_local_cli_environment": cli_environment,
        "historical_model_contract": {"model": "models/reranker", **EXPECTED_MODEL_CONTRACT, "batch_size": BATCH_SIZE, "local_files_only": True, "fp16": True},
        "labels_answers_folds_read": False,
        "training": False,
        "retrieval": False,
        "remote_submitted": False,
    }


def self_test() -> dict[str, bool]:
    toy = {"doc_id": "d", "chunk_id": "c", "dense_rank": 1, "dense_score": 0.1, "bge_score": 2.0}
    ordered = sorted([toy, {**toy, "doc_id": "a", "chunk_id": "a", "bge_score": 2.0}], key=lambda hit: (-float(hit["bge_score"]), int(hit["dense_rank"]), str(hit["doc_id"]), str(hit["chunk_id"])))
    try:
        fingerprint = _model_fingerprint(ROOT / "models/reranker")
        model_contract = fingerprint["files_sha256"] == EXPECTED_MODEL_FILES
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        model_contract = False
    return {"canonical_tie_break": [item["doc_id"] for item in ordered] == ["a", "d"], "historical_model_contract": model_contract, "no_train_field_names": not {"answer", "gold", "label", "fold"} & set(toy)}


@function(name="udsc-task1-v3-public-adaptive-k200", cpu=8, memory="32Gi", gpu="A10G", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=0, headless=True)
def run_public_adaptive_k200() -> None:
    model_path, model_fingerprint = _remote_preflight()
    questions = _load_questions(PUBLIC_QUESTIONS)
    if TEMPORARY_OUTPUT.is_file():
        _log(f"[RECOVER] validating completed temporary cache: {TEMPORARY_OUTPUT}")
        _validate_scored_cache(TEMPORARY_OUTPUT, set(questions))
        _log("[RECOVER] temporary cache is complete; publishing without rescoring")
        _publish_with_retry(TEMPORARY_OUTPUT, OUTPUT, label="public BGE K200 output")
    else:
        raw = _load_raw(RAW_K500, set(questions))
        _log(f"[INPUT] public K500 verified: {len(raw)} queries x {K200} scored candidates")
        _fill_missing_text(raw, PAYLOADS)
        _log("[MODEL] loading...")
        outputs = _score(model_path, questions, raw)
        if len(outputs) != EXPECTED_QUERIES or any(len(row["hits"]) != K200 for row in outputs):
            raise RuntimeError("public compatible BGE output coverage is incomplete")
        _log("[WRITE] validating")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with TEMPORARY_OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
            for row in outputs:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        _validate_scored_cache(TEMPORARY_OUTPUT, set(questions))
        _publish_with_retry(TEMPORARY_OUTPUT, OUTPUT, label="public BGE K200 output")
    manifest = {
        "schema_version": "public-bge-k200-compatible-v1",
        "status": "PUBLIC_BGE_K200_COMPATIBLE_PASS",
        "compatibility": {"status": "FRESH_RAW_K500_PREFIX_EXACT", "alignment": "exact_doc_id_and_chunk_id_set; dense_rank_1_to_200_preserved", "old_cache_rejected_as_provenance": True},
        "query_count": EXPECTED_QUERIES,
        "candidate_k": K200,
        "top_n": K200,
        "inputs": {"public_questions": str(PUBLIC_QUESTIONS), "raw_k500": str(RAW_K500), "raw_k500_sha256": sha256(RAW_K500), "payloads": str(PAYLOADS), "payloads_sha256": sha256(PAYLOADS)},
        "model": {"path": str(model_path), "fingerprint": model_fingerprint, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "local_files_only": True, "fp16": True, "single_logit": True, "preprocessing": "pair(question,raw_chunk_text); truncation=True"},
        "output_sha256": sha256(OUTPUT),
        "no_public_labels_used": True,
        "no_training": True,
        "no_retrieval": True,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _log("[DONE] PUBLIC_BGE_K200_COMPATIBLE_PASS")


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        result = preflight()
        print(result["beam_local_cli_environment"]["status"])
        print(json.dumps(result, indent=2))
    elif "--self-test" in sys.argv:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()):
            raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "gpu_launched": False}, indent=2))
    else:
        run_public_adaptive_k200.remote()
