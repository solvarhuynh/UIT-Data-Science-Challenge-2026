"""Workflow C1-I: label-free identifier, universe, provenance, and schema audit.

This CPU-only program intentionally does not import ML libraries.  Phase 1 reads
only structural fields from the label-bearing K20 action stream: values for
``label``, ``truth_delta``, and similarly named fields are skipped without
decoding.  Phase 2 is deliberately not implemented as a fallback: it is only
reported as unavailable unless a separately frozen, structurally valid input
bundle exists.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports/task1/workflow_c/tv4/c1i"
PATHS = {
    "folds": ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json",
    "baseline": ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
    "k20_actions": ROOT / "artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl",
    "full_candidates": ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl",
    "k77_contract": ROOT / "reports/task1/workflow_b/tv4/b1/contracts/workflow_b_b1_preregistered_contract.json",
}
EXPECTED = {
    "k20_actions": 215_422,
    "k20_features": 58,
    "k77_queries": 5_600,
    "k77_actions": 806_644,
    "k77_features": 36,
    "k77_global_row_hash": "d1a5695657105e56d0fbf467dd8b33091e7a61710680f38b4bff62ff946c6a80",
}
FORBIDDEN_ACTION_VALUE_FIELDS = {"label", "truth_delta", "utility", "target", "answer", "gold"}
TARGET_FOLDS = {1, 2, 3, 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name: str, payload: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def skip_json_value(text: str, index: int) -> int:
    """Return the end offset of one JSON value without decoding it."""
    index = whitespace(text, index)
    if text[index] == '"':
        return json.JSONDecoder().raw_decode(text, index)[1]
    if text[index] not in "[{":
        while index < len(text) and text[index] not in ",]}":
            index += 1
        return index
    expected = ["}" if text[index] == "{" else "]"]
    index += 1
    quoted = escaped = False
    while index < len(text) and expected:
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            expected.append("}")
        elif char == "[":
            expected.append("]")
        elif char == expected[-1]:
            expected.pop()
        index += 1
    return index


def object_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield object key and raw value span; values are not decoded here."""
    decoder = json.JSONDecoder()
    index = whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        raise ValueError("expected JSON object")
    index = whitespace(text, index + 1)
    while index < len(text) and text[index] != "}":
        key, index = decoder.raw_decode(text, index)
        index = whitespace(text, index)
        if text[index] != ":":
            raise ValueError("invalid JSON object")
        start = whitespace(text, index + 1)
        end = skip_json_value(text, start)
        yield str(key), start, end
        index = whitespace(text, end)
        if index < len(text) and text[index] == ",":
            index = whitespace(text, index + 1)


def structural_object(text: str, wanted: set[str]) -> tuple[dict[str, Any], list[str]]:
    decoder = json.JSONDecoder()
    values: dict[str, Any] = {}
    keys: list[str] = []
    for key, start, end in object_spans(text):
        keys.append(key)
        if key in wanted:
            values[key] = decoder.raw_decode(text, start)[0]
    return values, keys


def normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value)
    return result if result else None


def load_canonical_fold_map(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) != {
        "fold_count", "folds", "input", "input_sha256", "normalization", "schema_version", "seed"
    }:
        raise ValueError("unsupported canonical folds.json root schema")
    folds = raw["folds"]
    if not isinstance(folds, list):
        raise ValueError("canonical folds.json 'folds' must be a list")

    result: dict[str, int] = {}
    for row in folds:
        if not isinstance(row, dict) or set(row) != {"fold", "training_ids", "validation_ids"}:
            raise ValueError("unsupported canonical folds.json entry schema")
        fold, query_ids = row["fold"], row["validation_ids"]
        if not isinstance(fold, int) or fold not in range(raw["fold_count"]):
            raise ValueError("invalid canonical fold ID")
        if not isinstance(query_ids, list) or not all(isinstance(query_id, str) and query_id for query_id in query_ids):
            raise ValueError("invalid canonical validation query IDs")
        for query_id in query_ids:
            if query_id in result:
                raise ValueError("duplicate canonical validation query ID")
            result[query_id] = fold
    return result


def load_fold_map(path: Path) -> dict[str, int]:
    """Return the F1-F4 scientific target subset of canonical membership."""
    return {q: f for q, f in load_canonical_fold_map(path).items() if f in TARGET_FOLDS}


def audit_baseline(path: Path, fold_map: dict[str, int]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    baseline: dict[str, list[str]] = {}
    duplicate_query_ids = 0
    invalid_top5 = 0
    fold_mismatches = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row, _ = structural_object(line, {"query_id", "fold", "top5"})
        query_id = normalize_id(row.get("query_id"))
        if query_id not in fold_map:
            continue
        top5 = [normalize_id(value) for value in row.get("top5", [])]
        if query_id in baseline:
            duplicate_query_ids += 1
        baseline[query_id] = [value for value in top5 if value is not None]
        if len(top5) != 5 or len(set(top5)) != 5 or any(value is None for value in top5):
            invalid_top5 += 1
        if "fold" in row and int(row["fold"]) != fold_map[query_id]:
            fold_mismatches += 1
    return baseline, {
        "target_queries": len(baseline),
        "missing_target_queries": len(set(fold_map) - set(baseline)),
        "duplicate_query_ids": duplicate_query_ids,
        "invalid_top5": invalid_top5,
        "fold_mismatches": fold_mismatches,
    }


def action_record(line: str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Decode only identity, fold, and feature payload; skip labels/truth values."""
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {}
    keys: list[str] = []
    feature_order: list[str] = []
    for key, start, end in object_spans(line):
        keys.append(key)
        if key in {"query_id", "fold", "incoming_doc_id", "dropped_doc_id", "drop_rank", "incoming_union_rank"}:
            result[key] = decoder.raw_decode(line, start)[0]
        elif key == "features":
            features: dict[str, Any] = {}
            for feature, fstart, _ in object_spans(line[start:end]):
                feature_order.append(feature)
                features[feature] = decoder.raw_decode(line, start + fstart)[0]
            result["features"] = features
        # Explicitly skip all label-bearing values, including unknown fields.
    return result, keys, feature_order


def audit_k20(path: Path, full_fold_map: dict[str, int], fold_map: dict[str, int], baseline: dict[str, list[str]]) -> tuple[dict[str, Any], set[tuple[str, str]], set[tuple[str, str, str, int]]]:
    action_count = duplicate_actions = invalid_actions = fold_mismatches = 0
    target_action_count = fold0_excluded_action_count = unknown_query_action_count = 0
    missing_doc_ids = drop_baseline_mismatches = 0
    top_keys: Counter[str] = Counter()
    feature_order: list[str] | None = None
    feature_schema_variants = 0
    identities: set[tuple[str, str, str, int]] = set()
    incoming_members: set[tuple[str, str]] = set()
    structural_hash = hashlib.sha256()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row, keys, current_features = action_record(line)
        top_keys.update(keys)
        action_count += 1
        query_id = normalize_id(row.get("query_id"))
        incoming = normalize_id(row.get("incoming_doc_id"))
        dropped = normalize_id(row.get("dropped_doc_id"))
        rank = row.get("drop_rank")
        if query_id not in full_fold_map:
            unknown_query_action_count += 1
            continue
        assigned_fold = full_fold_map[query_id]
        if "fold" in row and int(row["fold"]) != assigned_fold:
            fold_mismatches += 1
        if assigned_fold == 0:
            fold0_excluded_action_count += 1
            continue
        if query_id not in fold_map:
            unknown_query_action_count += 1
            continue
        target_action_count += 1
        if feature_order is None:
            feature_order = current_features
        elif current_features != feature_order:
            feature_schema_variants += 1
        if incoming is None or dropped is None:
            missing_doc_ids += 1
            continue
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            rank = -1
        identity = (query_id, incoming, dropped, rank)
        if identity in identities:
            duplicate_actions += 1
        identities.add(identity)
        incoming_members.add((query_id, incoming))
        if rank not in {4, 5}:
            invalid_actions += 1
        elif query_id not in baseline or baseline[query_id][rank - 1] != dropped:
            drop_baseline_mismatches += 1
        structural_hash.update(json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode() + b"\n")
    features = feature_order or []
    return {
        "status": "PASS" if action_count == EXPECTED["k20_actions"] and len(features) == EXPECTED["k20_features"] and not any((duplicate_actions, invalid_actions, unknown_query_action_count, missing_doc_ids, drop_baseline_mismatches, fold_mismatches, feature_schema_variants)) else "BLOCKED",
        "actions": action_count,
        "k20_total_action_rows": action_count,
        "k20_target_f1_f4_action_rows": target_action_count,
        "k20_fold0_structurally_excluded_rows": fold0_excluded_action_count,
        "k20_unknown_query_action_rows": unknown_query_action_count,
        "queries": len({item[0] for item in identities}),
        "feature_count": len(features),
        "feature_names_ordered": features,
        "top_level_fields": sorted(top_keys),
        "forbidden_value_fields_skipped": sorted(FORBIDDEN_ACTION_VALUE_FIELDS & set(top_keys)),
        "duplicate_action_identities": duplicate_actions,
        "invalid_rank_actions": invalid_actions,
        "missing_query_ids": unknown_query_action_count,
        "missing_doc_ids": missing_doc_ids,
        "drop_baseline_mismatches": drop_baseline_mismatches,
        "fold_mismatches": fold_mismatches,
        "feature_schema_variants": feature_schema_variants,
        "structural_identity_sha256": structural_hash.hexdigest(),
        "file_sha256": sha256(path),
        "expected_actions_match": action_count == EXPECTED["k20_actions"],
        "expected_features_match": len(features) == EXPECTED["k20_features"],
    }, incoming_members, identities


def source_presence(candidate: dict[str, Any]) -> dict[str, bool]:
    ranks = candidate.get("source_ranks") or {}
    if not isinstance(ranks, dict):
        ranks = {}
    aliases = {
        "adaptive_dense": ("adaptive_k500", "adaptive_dense"),
        "bm25": ("bm25",),
        "word_knn": ("knn_word", "word_knn"),
        "char_knn": ("knn_char", "char_knn"),
    }
    return {
        name: any(ranks.get(alias) not in (None, "", 0) for alias in names)
        for name, names in aliases.items()
    }


def audit_full_and_k77(path: Path, fold_map: dict[str, int], baseline: dict[str, list[str]], k77_features: list[str]) -> tuple[dict[str, Any], dict[str, Any], set[tuple[str, str]], dict[str, int]]:
    full_queries: set[str] = set()
    target_queries: set[str] = set()
    full_members: set[tuple[str, str]] = set()
    k77_members: set[tuple[str, str]] = set()
    candidate_duplicates = query_fold_mismatches = missing_doc_ids = 0
    cap = 0
    source_counts = Counter()
    source_denominator = 0
    source_support_present = 0
    source_rank_keys = Counter()
    k77_hash = hashlib.sha256()
    k77_actions = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = normalize_id(row.get("query_id"))
        if query_id is None:
            continue
        full_queries.add(query_id)
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else [row]
        cap = max(cap, len(candidates))
        if query_id not in fold_map:
            continue
        target_queries.add(query_id)
        if "fold" in row and int(row["fold"]) != fold_map[query_id]:
            query_fold_mismatches += 1
        seen: set[str] = set()
        base = set(baseline.get(query_id, []))
        for candidate in candidates:
            doc_id = normalize_id(candidate.get("doc_id"))
            if doc_id is None:
                missing_doc_ids += 1
                continue
            if doc_id in seen:
                candidate_duplicates += 1
                continue
            seen.add(doc_id)
            full_members.add((query_id, doc_id))
            ranks = candidate.get("source_ranks") or {}
            if isinstance(ranks, dict):
                source_rank_keys.update(str(key) for key in ranks)
            presence = source_presence(candidate)
            source_denominator += 4
            source_counts.update(name for name, present in presence.items() if present)
            source_support_present += int(candidate.get("source_support") not in (None, ""))
            try:
                union_rank = int(candidate.get("union_rank"))
            except (TypeError, ValueError):
                union_rank = 10**9
            if union_rank <= 77 and doc_id not in base:
                k77_members.add((query_id, doc_id))
                for drop_rank in (4, 5):
                    dropped = baseline[query_id][drop_rank - 1]
                    identity = (query_id, doc_id, dropped, drop_rank)
                    k77_hash.update(json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode() + b"\n")
                    k77_actions += 1
    full_report = {
        "status": "PASS" if len(target_queries) == EXPECTED["k77_queries"] and not any((candidate_duplicates, query_fold_mismatches, missing_doc_ids)) else "BLOCKED",
        "all_file_queries": len(full_queries),
        "target_queries": len(target_queries),
        "missing_target_queries": len(set(fold_map) - target_queries),
        "candidate_cap": cap,
        "target_candidate_members": len(full_members),
        "duplicate_query_doc_memberships": candidate_duplicates,
        "query_fold_mismatches": query_fold_mismatches,
        "missing_doc_ids": missing_doc_ids,
        "file_sha256": sha256(path),
        "source_rank_key_names": sorted(source_rank_keys),
        "source_presence": dict(source_counts),
        "source_presence_denominator": source_denominator,
        "source_support_field_coverage": {"present": source_support_present, "total": len(full_members)},
    }
    k77_report = {
        "status": "PASS" if len(target_queries) == EXPECTED["k77_queries"] and k77_actions == EXPECTED["k77_actions"] and len(k77_features) == EXPECTED["k77_features"] else "BLOCKED",
        "queries": len(target_queries),
        "incoming_candidates": len(k77_members),
        "actions": k77_actions,
        "feature_count": len(k77_features),
        "feature_names_ordered": k77_features,
        "structural_identity_sha256": k77_hash.hexdigest(),
        "expected_actions_match": k77_actions == EXPECTED["k77_actions"],
        "expected_feature_count_match": len(k77_features) == EXPECTED["k77_features"],
        "expected_row_hash": EXPECTED["k77_global_row_hash"],
        "expected_row_hash_match": "N/A: expected hash serializes label and truth_delta; Phase 1 is label-free and no canonical NPZ is present",
    }
    return full_report, k77_report, full_members, dict(source_counts)


def classify_schemas(k20_features: list[str], k77_features: list[str]) -> dict[str, Any]:
    k20_set, k77_set = set(k20_features), set(k77_features)
    shared = [name for name in k20_features if name in k77_set]
    k20_only = [name for name in k20_features if name not in k77_set]
    k77_only = [name for name in k77_features if name not in k20_set]
    # Identical names are the only safe automatic identity assertion.
    return {
        "shared_identical": shared,
        "k20_only": k20_only,
        "k77_only": k77_only,
        "semantically_similar_not_identical": [],
        "unresolved": [],
        "ordering_shared_identical": [name for name in k77_features if name in k20_set] == shared,
        "automatic_merge_performed": False,
    }


def print_summary(status: str, universe: dict[str, Any], schema: dict[str, Any], paths: list[Path]) -> None:
    """The prescribed, side-effect-free terminal handoff."""
    k20 = universe.get("k20", {})
    k77 = universe.get("k77", {})
    full = universe.get("full_pool", {})
    joins = universe.get("joins", {})
    provenance = universe.get("source_provenance", {})
    mapping = schema.get("schema_mapping", {})
    coverage = provenance.get("coverage", {})
    print("WORKFLOW C1-I IDENTIFIER / UNIVERSE / SOURCE AUDIT")
    print()
    print(f"Status: {status}")
    print(f"K20 structural status: {k20.get('status', 'BLOCKED')}")
    print(f"K20 queries: {k20.get('queries', 0)}")
    print(f"K20 actions: {k20.get('actions', 0)}")
    print(f"K20 feature count: {k20.get('feature_count', 0)}")
    print(f"K77 structural status: {k77.get('status', 'BLOCKED')}")
    print(f"K77 queries: {k77.get('queries', 0)}")
    print(f"K77 actions: {k77.get('actions', 0)}")
    print(f"K77 feature count: {k77.get('feature_count', 0)}")
    hash_value = k77.get('expected_row_hash_match', 'N/A')
    print(f"K77 expected row hash match: {'YES' if hash_value is True else 'NO' if hash_value is False else 'N/A'}")
    print(f"Full-pool structural status: {full.get('status', 'BLOCKED')}")
    print(f"Full-pool queries: {full.get('target_queries', 0)}")
    print(f"Full-pool candidate cap: {full.get('candidate_cap', 0)}")
    print(f"K20-K77 shared feature count: {len(mapping.get('shared_identical', []))}")
    print(f"K20-only features: {len(mapping.get('k20_only', []))}")
    print(f"K77-only features: {len(mapping.get('k77_only', []))}")
    print(f"Schema unresolved features: {len(mapping.get('unresolved', []))}")
    print(f"Source provenance status: {provenance.get('status', 'BLOCKED')}")
    print(f"Source provenance coverage: {coverage.get('present', 0)}/{coverage.get('total', 0)}")
    print(f"Query ID mismatches: {joins.get('query_id_mismatches', 0)}")
    print(f"Document ID mismatches: {joins.get('document_id_mismatches', 0)}")
    print(f"Action ID mismatches: {joins.get('action_id_mismatches', 0)}")
    print(f"Duplicate identities: {joins.get('duplicate_identities', 0)}")
    print("Fold0 labels materialized: NO")
    print("Public labels used: NO")
    print("Historical V3A selected actions used: NO")
    print("Qwen used: NO")
    print("B1 resumed: NO")
    print("Model training: NO")
    print("Model inference: NO")
    print("GPU: NO")
    print("Modal: NO")
    print(f"Structural report: {paths[0].relative_to(ROOT)}")
    print(f"Universe report: {paths[1].relative_to(ROOT)}")
    print(f"Feature schema report: {paths[2].relative_to(ROOT)}")
    print("Safe next action: HANDOFF_C1I_TO_TV2_AFTER_REVIEW")
    print("STOP.")


def main() -> int:
    missing = [name for name, path in PATHS.items() if not path.is_file()]
    common = {
        "experiment": "C1-I",
        "owner": "TV4",
        "phase_1_label_free": True,
        "fold0_labels_materialized": False,
        "public_labels_used": False,
        "historical_v3a_selected_actions_used": False,
        "qwen_used": False,
        "b1_resumed": False,
        "model_training": False,
        "model_inference": False,
        "gpu": False,
        "modal": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in PATHS.items()},
        "missing_required_inputs": missing,
    }
    if missing:
        structural = {**common, "status": "BLOCKED", "reason": "required canonical structural inputs are absent", "phase_2_endpoint_verification": "UNVERIFIED: Phase 1 could not freeze identities"}
        universe = {**common, "status": "BLOCKED", "k20": {"status": "BLOCKED"}, "k77": {"status": "BLOCKED"}, "full_pool": {"status": "BLOCKED"}, "source_provenance": {"status": "BLOCKED"}}
        schema = {**common, "status": "BLOCKED", "reason": "K20 action and K77 contract inputs unavailable", "schema_mapping": {}}
        paths = [dump("c1i_structural_identity_report.json", structural), dump("c1i_action_universe_report.json", universe), dump("c1i_feature_schema_report.json", schema)]
        print_summary("BLOCKED", universe, schema, paths)
        return 2

    full_fold_map = load_canonical_fold_map(PATHS["folds"])
    fold_map = {q: f for q, f in full_fold_map.items() if f in TARGET_FOLDS}
    baseline, baseline_report = audit_baseline(PATHS["baseline"], fold_map)
    contract = json.loads(PATHS["k77_contract"].read_text(encoding="utf-8"))
    k77_features = [str(name) for name in contract.get("feature_columns", [])]
    k20, k20_incoming, _ = audit_k20(PATHS["k20_actions"], full_fold_map, fold_map, baseline)
    full, k77, full_members, _ = audit_full_and_k77(PATHS["full_candidates"], fold_map, baseline, k77_features)
    k20_not_full = len(k20_incoming - full_members)
    schema_mapping = classify_schemas(k20["feature_names_ordered"], k77_features)
    source_status = "PASS" if full["status"] == "PASS" and full["source_presence_denominator"] else "BLOCKED"
    universe = {**common, "status": "PASS" if all(item["status"] == "PASS" for item in (k20, k77, full)) else "PARTIAL", "baseline": baseline_report, "k20": k20, "k77": k77, "full_pool": full, "joins": {"k20_incoming_missing_from_full": k20_not_full, "k20_to_k77_exact_overlap": "UNRESOLVED: no canonical persisted K77 membership artifact; K77 was audited from full pool structurally", "query_id_mismatches": baseline_report["missing_target_queries"] + full["missing_target_queries"], "document_id_mismatches": k20_not_full + k20["missing_doc_ids"] + full["missing_doc_ids"], "action_id_mismatches": k20["drop_baseline_mismatches"] + k20["invalid_rank_actions"], "duplicate_identities": k20["duplicate_action_identities"] + full["duplicate_query_doc_memberships"]}, "source_provenance": {"status": source_status, "coverage": {"present": sum(full["source_presence"].values()), "total": full["source_presence_denominator"]}, "interpretation": "Counts represent explicit non-null source_ranks entries only; no source is inferred from file names or rank position."}}
    structural = {**common, "status": universe["status"], "normalization": {"query_id": "str(value), non-empty", "doc_id": "str(value), non-empty", "action_identity": "(query_id, incoming_doc_id, dropped_doc_id, drop_rank)", "drop_rank_domain": [4, 5], "duplicate_rule": "duplicate action identities and duplicate query/doc candidate memberships are invalid"}, "phase_1_freeze": {"input_file_sha256": {"baseline": sha256(PATHS["baseline"]), "k20_actions": sha256(PATHS["k20_actions"]), "full_candidates": sha256(PATHS["full_candidates"]), "k77_contract": sha256(PATHS["k77_contract"])}, "k20_structural_identity_sha256": k20["structural_identity_sha256"], "k77_structural_identity_sha256": k77["structural_identity_sha256"], "phase_2_endpoint_verification": "UNVERIFIED: optional endpoint verification intentionally deferred; it requires a separately frozen endpoint-prediction identity protocol"}}
    schema = {**common, "status": "PASS" if k20["status"] == "PASS" and len(k77_features) == EXPECTED["k77_features"] else "BLOCKED", "k20_feature_count": len(k20["feature_names_ordered"]), "k77_feature_count": len(k77_features), "schema_mapping": schema_mapping, "missing_value_contract": "K20: structural values not decoded beyond feature payload; K77: contract does not independently specify a per-feature missing-value sentinel", "silent_schema_recompute": "NOT PERFORMED"}
    paths = [dump("c1i_structural_identity_report.json", structural), dump("c1i_action_universe_report.json", universe), dump("c1i_feature_schema_report.json", schema)]
    print_summary(universe["status"], universe, schema, paths)
    return 0 if universe["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
