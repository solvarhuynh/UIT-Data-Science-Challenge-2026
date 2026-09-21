"""Fail-closed local CPU fallback for the frozen PV1 BGE delta.

The model calls and frozen-row resolver are reused from the canonical Modal
runner.  This file only supplies the CPU execution wrapper needed after the
Modal runtime failed before user code could start.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from scripts.modal import task1_bge_subset_gpu as canonical  # noqa: E402


DEFAULT_DELTA = ROOT / "private_task1/experiments/private_pv1/bge/private_pv1_bge_delta_worklist.jsonl"
DEFAULT_QUESTIONS = ROOT / "private_task1/input/private-official.json"
DEFAULT_OLD_WORKLIST = ROOT / "private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl"
DEFAULT_OLD_SCORES = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/results/production/scores.jsonl"
DEFAULT_CHUNKS_V3 = ROOT / "data/processed_v3/chunks"
DEFAULT_CHUNKS_PV1 = ROOT / "data/processed_pv1/chunks"
DEFAULT_BASE = ROOT / "models/reranker"
DEFAULT_FT = ROOT / "outputs/task1/bge_ft_model/bge_m3_finetuned"
DEFAULT_OUT = ROOT / "private_task1/experiments/private_pv1/bge/local_cpu_fallback"
SCORE_TOLERANCE = 1e-5


def read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = gzip.compress(path.read_bytes(), compresslevel=9)
    rows, full_sha, _ = canonical.load_worklist(payload, None)
    return rows, full_sha


def selected_sha(rows: list[dict[str, Any]]) -> str:
    raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    return canonical.sha256_bytes(raw)


def load_questions(path: Path, query_ids: set[str]) -> tuple[bytes, dict[str, str]]:
    payload = gzip.compress(path.read_bytes(), compresslevel=9)
    return payload, canonical.load_questions(query_ids, payload)


def validate_worklist(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], int], int]:
    expected: dict[tuple[str, str], int] = {}
    for row in rows:
        identity = (str(row["query_id"]), str(row["document_id"]))
        if identity in expected:
            raise RuntimeError(f"duplicate q-doc identity: {identity[0]}/{identity[1]}")
        if str(row.get("selector", "")) != canonical.SELECTOR:
            raise RuntimeError(f"selector contract mismatch: {identity[0]}/{identity[1]}")
        if str(row.get("aggregation", "")) != canonical.AGGREGATION:
            raise RuntimeError(f"aggregation contract mismatch: {identity[0]}/{identity[1]}")
        if int(row.get("max_length", canonical.MAX_LENGTH)) != canonical.MAX_LENGTH:
            raise RuntimeError(f"max_length contract mismatch: {identity[0]}/{identity[1]}")
        selected = row.get("selected_chunk_ids")
        count = int(row.get("expected_inference_units", 0))
        if not isinstance(selected, list) or len(selected) != count or not 1 <= count <= 3:
            raise RuntimeError(f"invalid frozen selector contract: {identity[0]}/{identity[1]}")
        selected = [str(value) for value in selected]
        if len(set(selected)) != len(selected):
            raise RuntimeError(f"duplicate frozen selected chunks: {identity[0]}/{identity[1]}")
        if any(not chunk_id.startswith(identity[1] + "_") for chunk_id in selected):
            raise RuntimeError(f"cross-document frozen chunk: {identity[0]}/{identity[1]}")
        expected[identity] = count
    return expected, sum(expected.values())


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return canonical.sha256_file(path)


def model_contract(base_path: Path, ft_path: Path) -> dict[str, Any]:
    base_config = canonical.validate_bge_directory(base_path, "BGE_BASE")
    base_revision = canonical.resolve_revision(base_path, required=True)
    if base_revision != canonical.BASE_MODEL_REVISION:
        raise RuntimeError(f"BGE_BASE_REVISION_MISMATCH: {base_revision}")
    ft = canonical.validate_ft_artifact(ft_path)
    return {
        "base": {
            "model_id": canonical.MODEL_ID,
            "path": str(base_path),
            "revision": base_revision,
            "config_sha256": canonical.sha256_file(base_path / "config.json"),
            "num_labels": base_config["_resolved_num_labels"],
        },
        "ft": ft,
    }


def select_parity_rows(
    source_worklist: Path,
    excluded_worklist: Path | None,
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    rows, source_sha = read_rows(source_worklist)
    excluded: set[tuple[str, str]] = set()
    if excluded_worklist is not None:
        excluded_rows, _ = read_rows(excluded_worklist)
        excluded = {(str(row["query_id"]), str(row["document_id"])) for row in excluded_rows}
    chosen = [
        row for row in rows
        if (str(row["query_id"]), str(row["document_id"])) not in excluded
    ][:limit]
    if len(chosen) != limit:
        raise RuntimeError(f"parity sample unavailable: {len(chosen)}/{limit}")
    return chosen, source_sha


def run_score(
    rows: list[dict[str, Any]],
    *,
    source_worklist_sha: str,
    questions_path: Path,
    chunks_root: Path,
    base_path: Path,
    ft_path: Path,
    batch_size: int,
    checkpoint_path: Path,
    output_path: Path,
    execution_source: str,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    wall_started = time.perf_counter()
    expected, total_units = validate_worklist(rows)
    question_ids = {str(row["query_id"]) for row in rows}
    questions_payload, questions = load_questions(questions_path, question_ids)
    models = model_contract(base_path, ft_path)
    document_cache: dict[str, dict[str, str]] = {}
    resolved, resolved_units = canonical.resolve_frozen_rows(
        rows, questions, document_cache, chunks_root
    )
    if resolved_units != total_units:
        raise RuntimeError(f"frozen unit mismatch: {resolved_units}/{total_units}")

    identity = {
        "mode": execution_source,
        "worklist_sha256": source_worklist_sha,
        "selected_worklist_sha256": selected_sha(rows),
        "q_doc_count": str(len(rows)),
        "inference_unit_count": str(total_units),
        "contract_sha256": canonical.contract_sha256(),
        "selector": canonical.SELECTOR,
        "aggregation": canonical.AGGREGATION,
        "max_length": str(canonical.MAX_LENGTH),
        "base_model_id": models["base"]["model_id"],
        "base_model_revision": models["base"]["revision"],
        "ft_model_id": models["ft"]["model_id"],
        "ft_artifact_id": models["ft"]["artifact_id"],
        "ft_weight_sha256": models["ft"]["weight_sha256"],
        "ft_config_sha256": models["ft"]["config_sha256"],
        "runtime_namespace": f"local_cpu_{execution_source}",
    }
    store = canonical.DurableBgeStore(checkpoint_path, identity, lambda: None)
    completed = store.completed_identities()
    if not completed.issubset(expected):
        store.close()
        raise RuntimeError("CHECKPOINT_HAS_UNKNOWN_IDENTITIES")
    remaining = [
        row for row in rows
        if (str(row["query_id"]), str(row["document_id"])) not in completed
    ]
    completed_units = sum(expected[key] for key in completed)
    model_load_seconds = 0.0
    inference_seconds = 0.0
    inference_batches = 0
    inference_forward_calls = 0
    try:
        if remaining:
            import torch
            from sentence_transformers import CrossEncoder

            load_started = time.perf_counter()
            ft_model = CrossEncoder(
                str(ft_path), num_labels=1, max_length=canonical.MAX_LENGTH,
                device="cpu", local_files_only=True,
            )
            base_model = CrossEncoder(
                str(base_path), num_labels=1, max_length=canonical.MAX_LENGTH,
                device="cpu", local_files_only=True,
            )
            ft_model.model.eval()
            base_model.model.eval()
            model_load_seconds = time.perf_counter() - load_started
            with torch.inference_mode():
                for offset in range(0, len(remaining), 256):
                    current_rows = remaining[offset : offset + 256]
                    current, current_units = canonical.resolve_frozen_rows(
                        current_rows, questions, document_cache, chunks_root
                    )
                    if current_units != sum(expected[(str(row["query_id"]), str(row["document_id"]))] for row in current_rows):
                        raise RuntimeError("block unit mismatch")
                    pairs: list[list[str]] = []
                    offsets: list[tuple[int, int]] = []
                    for item in current:
                        start = len(pairs)
                        pairs.extend([[item["question"], text] for text in item["selected_texts"]])
                        offsets.append((start, len(pairs)))
                    score_started = time.perf_counter()
                    ft_raw = ft_model.predict(
                        pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
                    )
                    base_raw = base_model.predict(
                        pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
                    )
                    inference_seconds += time.perf_counter() - score_started
                    inference_batches += 2
                    inference_forward_calls += 2 * math.ceil(len(pairs) / batch_size)
                    checkpoint_batch: list[dict[str, Any]] = []
                    for item, (start, end) in zip(current, offsets):
                        ft_scores = [float(x) for x in ft_raw[start:end]]
                        base_scores = [float(x) for x in base_raw[start:end]]
                        if not all(math.isfinite(x) for x in [*ft_scores, *base_scores]):
                            raise RuntimeError(f"non-finite score: {item['query_id']}/{item['document_id']}")
                        checkpoint_batch.append({**item, "ft_scores": ft_scores, "base_scores": base_scores})
                    store.commit_batch(checkpoint_batch)
                    done = len(completed) + min(offset + len(current_rows), len(remaining))
                    done_units = completed_units + sum(
                        expected[(str(item["query_id"]), str(item["document_id"]))]
                        for item in remaining[: offset + len(current_rows)]
                    )
                    print(
                        f"CPU_BGE_PROGRESS completed={done}/{len(rows)} units={done_units}/{total_units} "
                        f"elapsed={time.perf_counter() - wall_started:.1f}", flush=True
                    )
            del ft_model, base_model
        store.validate_complete(expected)
        output_rows: list[dict[str, Any]] = []
        for row in rows:
            query_id = str(row["query_id"])
            document_id = str(row["document_id"])
            result = store.result_for(query_id, document_id)
            output_rows.append({
                "query_id": query_id,
                "document_id": document_id,
                **result,
                "model_id": canonical.MODEL_ID,
                "base_model_revision": models["base"]["revision"],
                "ft_model_id": models["ft"]["model_id"],
                "ft_artifact_id": models["ft"]["artifact_id"],
                "ft_weight_sha256": models["ft"]["weight_sha256"],
                "ft_config_sha256": models["ft"]["config_sha256"],
                "aggregation": canonical.AGGREGATION,
                "selector": canonical.SELECTOR,
                "execution_source": execution_source,
            })
        output_sha = atomic_jsonl(output_path, output_rows)
    finally:
        store.close()
    return {
        "status": "PASS",
        "execution_source": execution_source,
        "q_doc_count": len(rows),
        "inference_unit_count": total_units,
        "completed_qdocs": len(rows),
        "completed_units": total_units,
        "worklist_sha256": source_worklist_sha,
        "selected_worklist_sha256": identity["selected_worklist_sha256"],
        "contract_sha256": identity["contract_sha256"],
        "model_contract": models,
        "selector": canonical.SELECTOR,
        "aggregation": canonical.AGGREGATION,
        "max_length": canonical.MAX_LENGTH,
        "device": "cpu",
        "gpu_used": False,
        "modal_used": False,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "inference_batches": inference_batches,
        "inference_forward_calls": inference_forward_calls,
        "document_cache_count": len(document_cache),
        "total_wall_seconds": time.perf_counter() - wall_started,
        "units_per_second_inference": total_units / inference_seconds if inference_seconds else 0.0,
        "units_per_second_wall": total_units / (time.perf_counter() - wall_started),
        "output": str(output_path),
        "output_sha256": output_sha,
        "checkpoint": str(checkpoint_path),
        "checkpoint_generation": identity,
        "question_source": str(questions_path),
        "chunk_source": str(chunks_root),
    }


def compare_reference(output_path: Path, reference_path: Path) -> dict[str, Any]:
    actual = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference = {}
    for line in reference_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            reference[(str(row["query_id"]), str(row["document_id"]))] = row
    missing = 0
    identity_mismatches = 0
    chunk_identity_mismatches = 0
    base_diffs: list[float] = []
    ft_diffs: list[float] = []
    base_doc_diffs: list[float] = []
    ft_doc_diffs: list[float] = []
    score_mismatch_rows = 0
    aggregation_mismatches = 0
    actual_by_query: dict[str, list[dict[str, Any]]] = {}
    reference_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in actual:
        key = (str(row["query_id"]), str(row["document_id"]))
        ref = reference.get(key)
        if ref is None:
            missing += 1
            continue
        actual_by_query.setdefault(key[0], []).append(row)
        reference_by_query.setdefault(key[0], []).append(ref)
        if key != (str(ref["query_id"]), str(ref["document_id"])):
            identity_mismatches += 1
        if row.get("selected_chunk_ids") != ref.get("selected_chunk_ids"):
            chunk_identity_mismatches += 1
        for actual_values, ref_values, bucket in (
            (row["base_chunk_scores"], ref["base_chunk_scores"], base_diffs),
            (row["ft_chunk_scores"], ref["ft_chunk_scores"], ft_diffs),
        ):
            if len(actual_values) != len(ref_values):
                chunk_identity_mismatches += 1
                continue
            bucket.extend(abs(float(a) - float(b)) for a, b in zip(actual_values, ref_values))
        base_doc_diffs.append(abs(float(row["bge_base_score"]) - float(ref["bge_base_score"])))
        ft_doc_diffs.append(abs(float(row["bge_ft_score"]) - float(ref["bge_ft_score"])))
        if abs(float(row["bge_base_score"]) - max(map(float, row["base_chunk_scores"]))) > 0:
            aggregation_mismatches += 1
        if abs(float(row["bge_ft_score"]) - max(map(float, row["ft_chunk_scores"]))) > 0:
            aggregation_mismatches += 1
        if (
            base_doc_diffs[-1] > SCORE_TOLERANCE
            or ft_doc_diffs[-1] > SCORE_TOLERANCE
            or any(diff > SCORE_TOLERANCE for diff in base_diffs[-len(row["base_chunk_scores"]):])
            or any(diff > SCORE_TOLERANCE for diff in ft_diffs[-len(row["ft_chunk_scores"]):])
        ):
            score_mismatch_rows += 1

    ordering_mismatches = 0
    for query_id, actual_rows in actual_by_query.items():
        ref_rows = reference_by_query.get(query_id, [])
        if len(actual_rows) < 2 or len(ref_rows) != len(actual_rows):
            continue
        actual_order = [str(row["document_id"]) for row in sorted(actual_rows, key=lambda row: (-float(row["bge_ft_score"]), str(row["document_id"])))]
        ref_order = [str(row["document_id"]) for row in sorted(ref_rows, key=lambda row: (-float(row["bge_ft_score"]), str(row["document_id"]))) ]
        if actual_order != ref_order:
            ordering_mismatches += 1

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "max_abs_diff": max(values) if values else 0.0,
            "mean_abs_diff": sum(values) / len(values) if values else 0.0,
        }

    gate = (
        len(actual) == len(reference) or len(actual) > 0
    ) and missing == 0 and identity_mismatches == 0 and chunk_identity_mismatches == 0
    gate = gate and score_mismatch_rows == 0 and aggregation_mismatches == 0 and ordering_mismatches == 0
    result = {
        "status": "PASS" if gate else "FAIL",
        "gate": "PASS" if gate else "FAIL",
        "reference": str(reference_path),
        "actual_rows": len(actual),
        "reference_rows_available": len(reference),
        "compared_rows": len(actual) - missing,
        "missing_reference_rows": missing,
        "identity_mismatches": identity_mismatches,
        "chunk_identity_mismatches": chunk_identity_mismatches,
        "base_chunk_score_matches_within_tolerance": sum(diff <= SCORE_TOLERANCE for diff in base_diffs),
        "ft_chunk_score_matches_within_tolerance": sum(diff <= SCORE_TOLERANCE for diff in ft_diffs),
        "base_chunk_scores_compared": len(base_diffs),
        "ft_chunk_scores_compared": len(ft_diffs),
        "base_chunk_score_summary": summary(base_diffs),
        "ft_chunk_score_summary": summary(ft_diffs),
        "base_document_score_summary": summary(base_doc_diffs),
        "ft_document_score_summary": summary(ft_doc_diffs),
        "score_mismatch_rows": score_mismatch_rows,
        "aggregation_mismatches": aggregation_mismatches,
        "ordering_mismatches": ordering_mismatches,
        "tolerance": SCORE_TOLERANCE,
    }
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("parity", "delta"), required=True)
    p.add_argument("--worklist", type=Path, default=DEFAULT_DELTA)
    p.add_argument("--exclude-worklist", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--reference-output", type=Path, default=DEFAULT_OLD_SCORES)
    p.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--chunks-root", type=Path)
    p.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    p.add_argument("--ft-model", type=Path, default=DEFAULT_FT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parity-metrics", type=Path)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.mode == "parity":
        sample_limit = args.limit or 12
        rows, source_sha = select_parity_rows(args.worklist, args.exclude_worklist, sample_limit)
    else:
        rows, source_sha = read_rows(args.worklist)
        if args.limit is not None:
            raise SystemExit("delta mode must score the complete frozen delta; omit --limit")
        if len(rows) != 297:
            raise RuntimeError(f"delta q-doc mismatch: {len(rows)}/297")
    chunks_root = args.chunks_root or (DEFAULT_CHUNKS_V3 if args.mode == "parity" else DEFAULT_CHUNKS_PV1)
    metrics = run_score(
        rows,
        source_worklist_sha=source_sha,
        questions_path=args.questions_file,
        chunks_root=chunks_root,
        base_path=args.base_model,
        ft_path=args.ft_model,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        execution_source="LOCAL_CPU_PARITY" if args.mode == "parity" else "LOCAL_CPU_FALLBACK",
    )
    if args.mode == "parity":
        comparison = compare_reference(args.output, args.reference_output)
        metrics["cpu_bge_parity"] = comparison
        if args.parity_metrics is not None:
            args.parity_metrics.parent.mkdir(parents=True, exist_ok=True)
            args.parity_metrics.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.parity_metrics is not None:
        parity = json.loads(args.parity_metrics.read_text(encoding="utf-8"))
        if parity.get("gate") != "PASS":
            raise RuntimeError("CPU_BGE_PARITY_GATE_FAIL")
        metrics["cpu_bge_parity_gate"] = "PASS"
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
