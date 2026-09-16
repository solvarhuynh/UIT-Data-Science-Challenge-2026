"""Locate, verify, and prepare the canonical Workflow C1-I input handoff.

This is a CPU-only, read/verify/copy task.  It never regenerates retrieval,
action, or prediction artifacts.  JSONL rows are parsed through the C0-A
structural reader with only identity/schema fields selected; label-bearing
payload fields are skipped.  The handoff contains exact byte copies and a
manifest, not decoded labels.

Lifecycle: VERSIONED_NEW_FILE / KEEP_REPRODUCIBILITY.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_MAP = ROOT / "docs/task1/workflow_c/workflow_c_evidence_map.md"
HANDOFF_ROOT = ROOT / "handoff/task1_workflow_c_c1i_inputs"
EXPECTED_QUERY_COUNT = 7000
EXPECTED_FOLD_COUNT = 1400
EXPECTED_ACTION_ROWS = 215422

FOLDS_REL = Path("artifacts/task1/evaluation/strict_cv_v2/folds.json")
BASELINE_REL = Path("artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl")
ACTIONS_REL = Path("artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl")
CANDIDATES_REL = Path("artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl")

EXPECTED_SHA = {
    BASELINE_REL: "1272cb9e8b465f433c725674081084f084a60cdb9807d6ca6a3aa3d9acc1e7d5",
    ACTIONS_REL: "7b7acbdd5c26c1e7b21cba42c84900f7894b8fc3dff1475941534345a140eb3a",
    CANDIDATES_REL: "e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6",
}


def load_structural_reader() -> Any:
    path = ROOT / "scripts/analysis/workflow_c/workflow_c_c0a_oracle_opportunity_anatomy.py"
    spec = importlib.util.spec_from_file_location("c0a_structural_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import canonical structural reader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_key(query_id: str) -> tuple[int, int | str]:
    return (0, int(query_id)) if query_id.isdigit() else (1, query_id)


def safe_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} is boolean")
    return int(value)


def add_error(errors: list[str], message: str, limit: int = 25) -> None:
    if len(errors) < limit:
        errors.append(message)


def verify_folds(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    """Validate membership only; no train/gold payload is involved."""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    if not isinstance(value, dict) or not isinstance(value.get("folds"), list):
        raise ValueError("folds.json must contain a folds array")
    all_ids: list[str] = []
    counts: Counter[int] = Counter()
    record_keys: set[str] = set()
    for record in value["folds"]:
        if not isinstance(record, dict):
            raise ValueError("fold record is not an object")
        record_keys.update(record)
        fold = safe_int(record.get("fold"), "fold")
        ids = record.get("validation_ids")
        if not isinstance(ids, list):
            raise ValueError(f"fold {fold} validation_ids is not a list")
        counts[fold] += len(ids)
        all_ids.extend(str(query_id) for query_id in ids)
    duplicates = len(all_ids) - len(set(all_ids))
    fold_map: dict[str, int] = {}
    for record in value["folds"]:
        fold = safe_int(record["fold"], "fold")
        for query_id in record["validation_ids"]:
            query_id = str(query_id)
            if query_id in fold_map:
                add_error(errors, f"duplicate query ID {query_id}")
            fold_map[query_id] = fold
    expected_folds = set(range(5))
    if set(counts) != expected_folds:
        add_error(errors, f"fold set {sorted(counts)} != [0,1,2,3,4]")
    for fold in range(5):
        if counts[fold] != EXPECTED_FOLD_COUNT:
            add_error(errors, f"fold {fold} count {counts[fold]} != {EXPECTED_FOLD_COUNT}")
    if len(all_ids) != EXPECTED_QUERY_COUNT:
        add_error(errors, f"query count {len(all_ids)} != {EXPECTED_QUERY_COUNT}")
    if duplicates:
        add_error(errors, f"duplicate query IDs: {duplicates}")
    return fold_map, {
        "row_or_query_count": len(all_ids),
        "schema_summary": {
            "kind": "JSON object containing folds array",
            "root_keys": sorted(value),
            "fold_record_keys": sorted(record_keys),
            "validation_id_type": "stringified query IDs",
        },
        "fold_counts": {str(fold): counts[fold] for fold in range(5)},
        "duplicate_query_id_count": duplicates,
        "errors": errors,
    }


def verify_baseline(path: Path, reader: Any, fold_map: dict[str, int]) -> dict[str, Any]:
    """Decode query_id/fold/top5 only; skip embedded gold_documents."""
    rows, stats = reader.stream_target_jsonl(path, set(fold_map), {"fold", "top5"})
    seen: set[str] = set()
    seen_rows: set[str] = set()
    duplicate_rows = 0
    schema_keys: set[str] = set()
    errors: list[str] = []
    top5_count = Counter()
    for line_number, query_id, row, keys in rows:
        schema_keys.update(keys)
        if query_id in seen_rows:
            duplicate_rows += 1
        seen_rows.add(query_id)
        try:
            if safe_int(row.get("fold"), "baseline.fold") != fold_map[query_id]:
                raise ValueError("fold mismatch")
            top5 = row.get("top5")
            if not isinstance(top5, list) or len(top5) != 5:
                raise ValueError("top5 must contain exactly 5 IDs")
            docs = [str(value) for value in top5]
            if len(set(docs)) != 5:
                raise ValueError("top5 contains duplicate document IDs")
            if query_id in seen:
                raise ValueError("duplicate query ID")
            seen.add(query_id)
            top5_count[len(docs)] += 1
        except (KeyError, TypeError, ValueError) as exc:
            add_error(errors, f"{path}:{line_number}:{query_id}:{exc}")
    errors.extend(stats.get("errors", []))
    missing = len(set(fold_map) - seen)
    if missing:
        add_error(errors, f"missing baseline query IDs: {missing}")
    return {
        "row_or_query_count": stats["nonempty_row_count"],
        "unique_query_count": len(seen),
        "schema_summary": {
            "kind": "JSONL object",
            "selected_fields": ["query_id", "fold", "top5"],
            "required_fields": ["query_id", "fold", "top5"],
            "skipped_label_bearing_fields": ["gold_documents"],
            "top5_length_distribution": {str(key): top5_count[key] for key in sorted(top5_count)},
            "observed_top_level_keys": sorted(schema_keys | stats.get("target_top_level_keys", [])),
        },
        "structurally_skipped_non_target_rows": stats["non_target_rows_structurally_skipped"],
        "duplicate_query_id_count": duplicate_rows,
        "missing_query_id_count": missing,
        "errors": errors,
    }


def verify_actions(path: Path, reader: Any, fold_map: dict[str, int], baseline: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Decode only action identity/schema; skip gain/label fields."""
    # Baseline is not needed for this handoff contract; retaining the optional
    # parameter makes the identity check easy to extend without changing the
    # byte-level handoff.
    rows, stats = reader.stream_target_jsonl(
        path,
        set(fold_map),
        {"fold", "incoming_doc_id", "dropped_doc_id", "drop_rank", "baseline_top5", "features"},
    )
    seen_queries: set[str] = set()
    action_ids: set[tuple[str, str, int, str]] = set()
    schema_keys: set[str] = set()
    feature_sizes: Counter[int] = Counter()
    drop_ranks: Counter[int] = Counter()
    errors: list[str] = []
    for line_number, query_id, row, keys in rows:
        schema_keys.update(keys)
        try:
            if safe_int(row.get("fold"), "action.fold") != fold_map[query_id]:
                raise ValueError("fold mismatch")
            incoming = str(row["incoming_doc_id"])
            dropped = str(row["dropped_doc_id"])
            rank = safe_int(row.get("drop_rank"), "action.drop_rank")
            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError("features missing")
            feature_sizes[len(features)] += 1
            if len(features) != 58:
                raise ValueError("feature count is not 58")
            if rank not in (4, 5):
                raise ValueError("drop_rank is not 4 or 5")
            drop_ranks[rank] += 1
            identity = (query_id, incoming, rank, dropped)
            if identity in action_ids:
                raise ValueError("duplicate action identity")
            action_ids.add(identity)
            seen_queries.add(query_id)
        except (KeyError, TypeError, ValueError) as exc:
            add_error(errors, f"{path}:{line_number}:{query_id}:{exc}")
    errors.extend(stats.get("errors", []))
    missing = len(set(fold_map) - seen_queries)
    if missing:
        add_error(errors, f"missing action query IDs: {missing}")
    duplicate_rows = stats["target_row_count"] - len(action_ids)
    return {
        "row_or_query_count": stats["nonempty_row_count"],
        "unique_query_count": len(seen_queries),
        "action_count": len(action_ids),
        "schema_summary": {
            "kind": "JSONL object",
            "selected_fields": ["query_id", "fold", "incoming_doc_id", "dropped_doc_id", "drop_rank", "baseline_top5", "features"],
            "required_fields": ["query_id", "fold", "incoming_doc_id", "dropped_doc_id", "drop_rank", "baseline_top5", "features"],
            "skipped_label_bearing_fields": ["baseline_recall", "gain", "label"],
            "feature_size_distribution": {str(key): feature_sizes[key] for key in sorted(feature_sizes)},
            "drop_rank_distribution": {str(key): drop_ranks[key] for key in sorted(drop_ranks)},
            "observed_top_level_keys": sorted(schema_keys | stats.get("target_top_level_keys", [])),
        },
        "structurally_skipped_non_target_rows": stats["non_target_rows_structurally_skipped"],
        "duplicate_action_identity_count": duplicate_rows,
        "missing_query_id_count": missing,
        "errors": errors,
    }


def verify_candidates(path: Path, reader: Any, fold_map: dict[str, int]) -> dict[str, Any]:
    """Validate the 200-document candidate cap and source support/ranks."""
    rows, stats = reader.stream_target_jsonl(
        path,
        set(fold_map),
        {"fold", "candidates"},
    )
    seen: set[str] = set()
    schema_keys: set[str] = set()
    candidate_lengths: Counter[int] = Counter()
    support_counts: Counter[int] = Counter()
    errors: list[str] = []
    allowed_sources = {"adaptive_k500", "bm25", "knn_word", "knn_char"}
    for line_number, query_id, row, keys in rows:
        schema_keys.update(keys)
        try:
            if safe_int(row.get("fold"), "candidate.fold") != fold_map[query_id]:
                raise ValueError("fold mismatch")
            if query_id in seen:
                raise ValueError("duplicate query ID")
            seen.add(query_id)
            candidates = row.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError("candidates is not a list")
            candidate_lengths[len(candidates)] += 1
            if len(candidates) != 200:
                raise ValueError("candidate cap is not exactly 200")
            docs: set[str] = set()
            ranks: set[int] = set()
            for item in candidates:
                if not isinstance(item, dict):
                    raise ValueError("candidate item is not an object")
                doc_id = str(item.get("doc_id", ""))
                union_rank = safe_int(item.get("union_rank"), "candidate.union_rank")
                support = safe_int(item.get("source_support"), "candidate.source_support")
                source_ranks = item.get("source_ranks")
                if not doc_id or union_rank < 1 or union_rank > 200:
                    raise ValueError("invalid doc_id/union_rank")
                if doc_id in docs or union_rank in ranks:
                    raise ValueError("duplicate doc_id/union_rank")
                if not isinstance(source_ranks, dict):
                    raise ValueError("source_ranks missing")
                if set(source_ranks) - allowed_sources:
                    raise ValueError("unknown source-rank key")
                if support != len(source_ranks) or not 1 <= support <= 4:
                    raise ValueError("source_support does not match source_ranks")
                for source, rank in source_ranks.items():
                    rank_value = safe_int(rank, f"source_ranks.{source}")
                    if rank_value < 1:
                        raise ValueError("source rank must be positive")
                docs.add(doc_id)
                ranks.add(union_rank)
                support_counts[support] += 1
        except (KeyError, TypeError, ValueError) as exc:
            add_error(errors, f"{path}:{line_number}:{query_id}:{exc}")
    errors.extend(stats.get("errors", []))
    missing = len(set(fold_map) - seen)
    if missing:
        add_error(errors, f"missing candidate query IDs: {missing}")
    return {
        "row_or_query_count": stats["nonempty_row_count"],
        "unique_query_count": len(seen),
        "schema_summary": {
            "kind": "JSONL object with nested candidates array",
            "selected_fields": ["query_id", "fold", "candidates[].doc_id", "candidates[].union_rank", "candidates[].source_support", "candidates[].source_ranks"],
            "required_fields": ["query_id", "fold", "candidates"],
            "candidate_cap": 200,
            "candidate_length_distribution": {str(key): candidate_lengths[key] for key in sorted(candidate_lengths)},
            "source_support_distribution": {str(key): support_counts[key] for key in sorted(support_counts)},
            "allowed_source_rank_keys": sorted(allowed_sources),
            "observed_top_level_keys": sorted(schema_keys | stats.get("target_top_level_keys", [])),
        },
        "structurally_skipped_non_target_rows": stats["non_target_rows_structurally_skipped"],
        "missing_query_id_count": missing,
        "errors": errors,
    }


def copy_and_hash(source: Path, destination: Path) -> tuple[str, str, bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = sha256(source)
    handoff_hash = sha256(destination)
    identical = source_hash == handoff_hash and source.stat().st_size == destination.stat().st_size
    if not identical:
        destination.unlink(missing_ok=True)
    return source_hash, handoff_hash, identical


def main() -> None:
    reader = load_structural_reader()
    source_paths = {
        FOLDS_REL: ROOT / FOLDS_REL,
        BASELINE_REL: ROOT / BASELINE_REL,
        ACTIONS_REL: ROOT / ACTIONS_REL,
        CANDIDATES_REL: ROOT / CANDIDATES_REL,
    }
    missing = [path for path in source_paths.values() if not path.exists()]
    if missing:
        print("TASK1 C1-I CANONICAL ARTIFACT HANDOFF PREPARATION")
        print("\nStatus:\nBLOCKED_CANONICAL_SOURCE_NOT_FOUND")
        print(f"\nCanonical artifacts found:\n{4 - len(missing)}/4")
        print("\nfolds.json:\nMISSING" if source_paths[FOLDS_REL] in missing else "\nfolds.json:\nFOUND")
        print("\nfolds.json SHA:\nN/A")
        print("\nfolds structural contract:\nN/A")
        print("\nbaseline predictions:\nMISSING" if source_paths[BASELINE_REL] in missing else "\nbaseline predictions:\nFOUND")
        print("\nbaseline SHA match:\nN/A")
        print("\nactions.jsonl:\nMISSING" if source_paths[ACTIONS_REL] in missing else "\nactions.jsonl:\nFOUND")
        print("\nactions SHA match:\nN/A")
        print("\ncandidate_refs_full.jsonl:\nMISSING" if source_paths[CANDIDATES_REL] in missing else "\ncandidate_refs_full.jsonl:\nFOUND")
        print("\ncandidate refs SHA match:\nN/A")
        print("\nByte-identical handoff copies:\n0/4\n\nHandoff directory:\nN/A\n\nHandoff manifest:\nN/A\n\nArtifacts regenerated:\nNO\n\nOriginal artifacts modified:\nNO\n\nFold0 labels used:\nNO\n\nPublic labels used:\nNO\n\nGPU:\nNO\n\nModal:\nNO\n\nSafe next action:\nRECOVER_CANONICAL_SOURCE\n\nSTOP.")
        return

    fold_map, fold_info = verify_folds(source_paths[FOLDS_REL])
    baseline_info = verify_baseline(source_paths[BASELINE_REL], reader, fold_map)
    actions_info = verify_actions(source_paths[ACTIONS_REL], reader, fold_map)
    candidates_info = verify_candidates(source_paths[CANDIDATES_REL], reader, fold_map)
    infos = {FOLDS_REL: fold_info, BASELINE_REL: baseline_info, ACTIONS_REL: actions_info, CANDIDATES_REL: candidates_info}
    structural_errors: list[str] = []
    for rel_path, info in infos.items():
        structural_errors.extend(info.get("errors", []))
    if baseline_info["row_or_query_count"] != EXPECTED_QUERY_COUNT or baseline_info["unique_query_count"] != EXPECTED_QUERY_COUNT:
        add_error(structural_errors, "baseline row/query count is not 7,000")
    if actions_info["row_or_query_count"] != EXPECTED_ACTION_ROWS or actions_info["action_count"] != EXPECTED_ACTION_ROWS:
        add_error(structural_errors, "actions row/action count is not 215,422")
    if candidates_info["row_or_query_count"] != EXPECTED_QUERY_COUNT or candidates_info["unique_query_count"] != EXPECTED_QUERY_COUNT:
        add_error(structural_errors, "candidate query count is not 7,000")
    source_hashes = {rel_path: sha256(path) for rel_path, path in source_paths.items()}
    expected_failures = [rel_path for rel_path, expected in EXPECTED_SHA.items() if source_hashes[rel_path] != expected]
    if expected_failures:
        add_error(structural_errors, "known canonical SHA mismatch: " + ", ".join(path.as_posix() for path in expected_failures))
    if structural_errors:
        print("TASK1 C1-I CANONICAL ARTIFACT HANDOFF PREPARATION")
        print("\nStatus:\nBLOCKED_IDENTITY_MISMATCH")
        print("\nCanonical artifacts found:\n4/4")
        print("\nfolds.json:\nFOUND")
        print(f"\nfolds.json SHA:\n{source_hashes[FOLDS_REL]}")
        print("\nfolds structural contract:\nFAIL")
        print("\nbaseline predictions:\nFOUND\n\nbaseline SHA match:\n" + ("NO" if BASELINE_REL in expected_failures else "YES"))
        print("\nactions.jsonl:\nFOUND\n\nactions SHA match:\n" + ("NO" if ACTIONS_REL in expected_failures else "YES"))
        print("\ncandidate_refs_full.jsonl:\nFOUND\n\ncandidate refs SHA match:\n" + ("NO" if CANDIDATES_REL in expected_failures else "YES"))
        print("\nByte-identical handoff copies:\n0/4\n\nHandoff directory:\nN/A\n\nHandoff manifest:\nN/A\n\nArtifacts regenerated:\nNO\n\nOriginal artifacts modified:\nNO\n\nFold0 labels used:\nNO\n\nPublic labels used:\nNO\n\nGPU:\nNO\n\nModal:\nNO\n\nSafe next action:\nRECOVER_CANONICAL_SOURCE\n\nSTOP.")
        return

    # This directory was checked absent before this run.  If a rerun finds it,
    # copies are refreshed in place only after the source checks above pass.
    HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    identical_count = 0
    for rel_path, source in source_paths.items():
        destination = HANDOFF_ROOT / rel_path
        source_hash, handoff_hash, identical = copy_and_hash(source, destination)
        if identical:
            identical_count += 1
        record = {
            "canonical_relative_path": rel_path.as_posix(),
            "source_absolute_path": str(source.resolve()),
            "handoff_absolute_path": str(destination.resolve()),
            "source_SHA256": source_hash,
            "handoff_SHA256": handoff_hash,
            "byte_identical": identical,
            "file_size": source.stat().st_size,
            "row_or_query_count": infos[rel_path]["row_or_query_count"],
            "schema_summary": infos[rel_path]["schema_summary"],
            "canonical_evidence_reference": "docs/task1/workflow_c/workflow_c_evidence_map.md",
        }
        if rel_path == FOLDS_REL:
            record.update({f"fold{fold}_count": infos[rel_path]["fold_counts"][str(fold)] for fold in range(5)})
        records[rel_path.as_posix()] = record

    byte_error = identical_count != 4
    if byte_error:
        for rel_path, source in source_paths.items():
            destination = HANDOFF_ROOT / rel_path
            if destination.exists() and records[rel_path.as_posix()]["byte_identical"] is False:
                destination.unlink(missing_ok=True)
        print("TASK1 C1-I CANONICAL ARTIFACT HANDOFF PREPARATION")
        print("\nStatus:\nBLOCKED_IDENTITY_MISMATCH\n\nCanonical artifacts found:\n4/4\n\nfolds.json:\nFOUND")
        print(f"\nfolds.json SHA:\n{source_hashes[FOLDS_REL]}\n\nfolds structural contract:\nPASS\n\nbaseline predictions:\nFOUND\n\nbaseline SHA match:\nYES\n\nactions.jsonl:\nFOUND\n\nactions SHA match:\nYES\n\ncandidate_refs_full.jsonl:\nFOUND\n\ncandidate refs SHA match:\nYES")
        print(f"\nByte-identical handoff copies:\n{identical_count}/4\n\nHandoff directory:\n{HANDOFF_ROOT}\n\nHandoff manifest:\nN/A\n\nArtifacts regenerated:\nNO\n\nOriginal artifacts modified:\nNO\n\nFold0 labels used:\nNO\n\nPublic labels used:\nNO\n\nGPU:\nNO\n\nModal:\nNO\n\nSafe next action:\nRECOVER_CANONICAL_SOURCE\n\nSTOP.")
        return

    manifest = {
        "workflow": "TASK1 C1-I CANONICAL ARTIFACT HANDOFF PREPARATION",
        "status": "READY_FOR_TV4_TRANSFER",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_search_scope": {
            "canonical_repository_root": str(ROOT.resolve()),
            "evidence_map": str(EVIDENCE_MAP.resolve()),
            "candidate_sources_considered": [str(path.resolve()) for path in source_paths.values()],
            "unrelated_same_named_files_accepted": False,
        },
        "verification_scope": {
            "cpu_only": True,
            "gpu": False,
            "modal": False,
            "artifacts_regenerated": False,
            "original_artifacts_modified": False,
            "fold0_labels_used": False,
            "public_labels_used": False,
            "label_payloads_in_manifest": False,
            "label_bearing_payload_fields_decoded": False,
            "structural_identity_frozen_before_copy": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "artifacts": records,
        "fold_contract": {
            "total_query_ids": EXPECTED_QUERY_COUNT,
            "folds": [0, 1, 2, 3, 4],
            "per_fold_count": EXPECTED_FOLD_COUNT,
            "duplicate_query_id_count": infos[FOLDS_REL]["duplicate_query_id_count"],
        },
        "file_lifecycle_review": {
            "handoff_directory": HANDOFF_ROOT.as_posix(),
            "decision": "VERSIONED_NEW_FILE",
            "reason": "single temporary byte-identical C1-I transfer tree requested by the contract",
            "canonical_sources_modified": False,
            "files_deleted": [],
            "consumer": "TV4 C1-I structural identity audit",
        },
    }
    manifest_path = HANDOFF_ROOT / "c1i_canonical_handoff_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Final read-back and byte check includes the manifest's own creation.
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if loaded.get("status") != "READY_FOR_TV4_TRANSFER" or any(
        not record["byte_identical"] for record in loaded["artifacts"].values()
    ):
        raise RuntimeError("handoff manifest final read-back failed")

    baseline_sha_match = source_hashes[BASELINE_REL] == EXPECTED_SHA[BASELINE_REL]
    actions_sha_match = source_hashes[ACTIONS_REL] == EXPECTED_SHA[ACTIONS_REL]
    candidates_sha_match = source_hashes[CANDIDATES_REL] == EXPECTED_SHA[CANDIDATES_REL]
    print("TASK1 C1-I CANONICAL ARTIFACT HANDOFF PREPARATION")
    print("\nStatus:\nREADY_FOR_TV4_TRANSFER")
    print("\nCanonical artifacts found:\n4/4")
    print("\nfolds.json:\nFOUND")
    print(f"\nfolds.json SHA:\n{source_hashes[FOLDS_REL]}")
    print("\nfolds structural contract:\nPASS")
    print("\nbaseline predictions:\nFOUND")
    print(f"\nbaseline SHA match:\n{'YES' if baseline_sha_match else 'NO'}")
    print("\nactions.jsonl:\nFOUND")
    print(f"\nactions SHA match:\n{'YES' if actions_sha_match else 'NO'}")
    print("\ncandidate_refs_full.jsonl:\nFOUND")
    print(f"\ncandidate refs SHA match:\n{'YES' if candidates_sha_match else 'NO'}")
    print(f"\nByte-identical handoff copies:\n{identical_count}/4")
    print(f"\nHandoff directory:\n{HANDOFF_ROOT}")
    print(f"\nHandoff manifest:\n{manifest_path}")
    print("\nArtifacts regenerated:\nNO")
    print("\nOriginal artifacts modified:\nNO")
    print("\nFold0 labels used:\nNO")
    print("\nPublic labels used:\nNO")
    print("\nGPU:\nNO")
    print("\nModal:\nNO")
    print("\nSafe next action:\nTRANSFER_HANDOFF_DIRECTORY_TO_TV4\n\nSTOP.")


if __name__ == "__main__":
    main()
