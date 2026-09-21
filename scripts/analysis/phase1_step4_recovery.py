"""CPU-only Phase 1 recovery/replay for the historical Step4 BGE policy.

This script is intentionally self-contained and conservative:

* it hashes existing inputs before using them and never edits them;
* it verifies the historical public Step4 calibration replay from its frozen
  anchor, report, and calibrated submission;
* it reconstructs the current PV1 F1-F4 BGE score map from the frozen delta
  scores plus the reusable historical score artifact;
* it evaluates the historical Step4 contract on the current K20 universe as
  ``STEP4_K20_PORT`` (not as an exact Top-50 reproduction);
* it only materializes a separate Private checkpoint when the preregistered
  F1-F4 gate passes;
* otherwise it runs one deterministic CPU fallback using the repository's
  existing LogisticRegression meta-ranker implementation, with no sweep.

No model is loaded.  No GPU, Modal, Qwen inference, Private answer, Fold0
label, or public label is used.  F1-F4 train labels are used only for the
authorized offline evaluation population.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "private_task1/experiments/sprint48_step4"

# Current PV1 validation inputs (F1-F4 only).
PV1_CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
PV1_WORKLIST = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
PV1_DELTA_SCORES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_bge_delta_scores.jsonl"
PV1_DELTA_METRICS = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_bge_delta_metrics.json"
PV1_DENSE = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_dense_predictions.jsonl"
PV1_BM25 = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_bm25.jsonl"
PV1_BM25_MANIFEST = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_bm25_manifest.json"
PV1_KNN_MANIFEST = ROOT / "private_task1/experiments/private_pv1/knn/private_pv1_word_tfidf_knn_manifest.json"
PV1_K20_CANDIDATES = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
PV1_K20_WORKLIST = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_bge_full_worklist.jsonl"
PV1_BGE_MERGED = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"
PV1_BGE_MANIFEST = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_merged_manifest.json"
PV1_SUBMISSION_JSON = ROOT / "private_task1/submissions/pv1_final/submission_private_pv1.json"
PV1_SUBMISSION_ZIP = ROOT / "private_task1/submissions/pv1_final/submission_private_pv1.zip"
PV1_SUBMISSION_MANIFEST = ROOT / "private_task1/submissions/pv1_final/pv1_final_manifest.json"

# Reusable current-V3 BGE rows.  Rows covered by the PV1 delta are always
# replaced by the PV1 delta row and are never silently taken from this file.
REUSABLE_SCORES = ROOT / "private_task1/experiments/private_safe_baseline/validation_download/results/production/scores.jsonl"

# Labels and fold mapping for the authorized F1-F4 evaluation only.
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/folds.json"

# Historical Step4 handoff.
STEP4_ROOT = ROOT / "artifacts/task1/handoff/step4_bge_handoff"
STEP4_MANIFEST = STEP4_ROOT / "manifest/step4_bge_handoff_manifest.json"
STEP4_ANCHOR_ZIP = STEP4_ROOT / "public/public_anchor_09391_submission.zip"
STEP4_FINAL_JSON = STEP4_ROOT / "public/submission_step4_calibrated.json"
STEP4_FINAL_ZIP = STEP4_ROOT / "public/submission_step4_calibrated.zip"
STEP4_REPORT = STEP4_ROOT / "fusion/step4_ensemble_report.json"
STEP4_CALIBRATION = STEP4_ROOT / "fusion/apply_step4_calibration.py"
STEP4_GENERATOR = STEP4_ROOT / "scripts/modal_step4_ensemble.py"
STEP4_UNION_F1F4 = STEP4_ROOT / "candidates/f1_f4_candidate_union.jsonl"
STEP4_HISTORY = STEP4_ROOT / "validation/experiment_history_09411.md"
STEP4_COMPLETION = STEP4_ROOT / "validation/step4_completion_report.md"

# Private current PV1 inputs.  These are read only and have answer=null in the
# input contract; no Private answer is loaded here.
PRIVATE_CANDIDATES = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
PRIVATE_SCORES = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"
PRIVATE_BASELINE = ROOT / "private_task1/submissions/pv1_final/submission_private_pv1.json"

EXPECTED_MODEL = "BAAI/bge-reranker-v2-m3"
EXPECTED_BASE_REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EXPECTED_SELECTOR = "true_s2_bm25_within_document_v2"
EXPECTED_AGGREGATION = "MAX"
EXPECTED_FT_SHA = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
EXPECTED_FT_CONFIG_SHA = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
TARGET_FOLDS = (1, 2, 3, 4)
EXPECTED_F1F4_QUERIES = 5600
EXPECTED_F1F4_QDOCS = 112000
STEP4_WEIGHTS = {"ft": 0.40, "base": 0.30, "rrf": 0.20, "support": 0.10}
STEP4_DELTA = 0.40
STEP4_MARGIN_RANK5 = 0.08
STEP4_MARGIN_RANK4 = 0.15


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_no
                rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "_line"}
            stream.write(json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def qkey(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def key_of(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["query_id"]), str(row["document_id"])


def load_fold_map() -> dict[str, int]:
    payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    result: dict[str, int] = {}
    for entry in payload["folds"]:
        for raw_qid in entry["validation_ids"]:
            qid = str(raw_qid)
            if qid in result:
                raise RuntimeError(f"duplicate fold assignment: {qid}")
            result[qid] = int(entry["fold"])
    return result


def load_gold() -> dict[str, dict[str, Any]]:
    payload = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
    return {str(qid): row for qid, row in payload.items() if isinstance(row, dict)}


def metric(predictions: dict[str, list[str]], gold: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    recalls: list[float] = []
    precisions: list[float] = []
    for qid, predicted in predictions.items():
        wanted = {str(value) for value in gold[qid].get("answer", [])}
        if not wanted:
            raise RuntimeError(f"empty gold answer in F1-F4 evaluation: {qid}")
        hits = len(wanted & set(predicted[:5]))
        recalls.append(hits / len(wanted))
        precisions.append(hits / 5.0)
    return {
        "queries": len(predictions),
        "recall": mean(recalls),
        "precision": mean(precisions),
    }


def freeze_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": rel(path), "exists": False}
    return {"path": rel(path), "exists": True, "bytes": path.stat().st_size, "sha256": sha256(path)}


def freeze_inputs() -> dict[str, Any]:
    fixed: dict[str, Path] = {
        "pv1_validation_candidate_union": PV1_CANDIDATES,
        "pv1_validation_worklist": PV1_WORKLIST,
        "pv1_validation_delta_scores": PV1_DELTA_SCORES,
        "pv1_validation_delta_metrics": PV1_DELTA_METRICS,
        "pv1_validation_dense": PV1_DENSE,
        "pv1_validation_bm25": PV1_BM25,
        "pv1_validation_bm25_manifest": PV1_BM25_MANIFEST,
        "pv1_knn_manifest": PV1_KNN_MANIFEST,
        "pv1_private_k20_candidates": PV1_K20_CANDIDATES,
        "pv1_private_k20_worklist": PV1_K20_WORKLIST,
        "pv1_private_bge_merged": PV1_BGE_MERGED,
        "pv1_private_bge_manifest": PV1_BGE_MANIFEST,
        "pv1_private_submission_json": PV1_SUBMISSION_JSON,
        "pv1_private_submission_zip": PV1_SUBMISSION_ZIP,
        "pv1_private_submission_manifest": PV1_SUBMISSION_MANIFEST,
        "reusable_v3_bge_scores": REUSABLE_SCORES,
        "train_f1_f4_source": TRAIN,
        "folds": FOLDS,
        "step4_handoff_manifest": STEP4_MANIFEST,
        "step4_anchor_zip": STEP4_ANCHOR_ZIP,
        "step4_final_json": STEP4_FINAL_JSON,
        "step4_final_zip": STEP4_FINAL_ZIP,
        "step4_report": STEP4_REPORT,
        "step4_calibration_code": STEP4_CALIBRATION,
        "step4_generator_code": STEP4_GENERATOR,
        "step4_f1_f4_union": STEP4_UNION_F1F4,
        "step4_history": STEP4_HISTORY,
        "step4_completion_report": STEP4_COMPLETION,
    }
    frozen = {name: freeze_file(path) for name, path in fixed.items()}
    previous: dict[str, dict[str, Any]] = {}
    submissions_root = ROOT / "private_task1/submissions"
    for path in sorted(submissions_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".zip"}:
            continue
        if any(part in {"sprint48_step4", "sprint48_fallback_direct_top5"} for part in path.parts):
            continue
        previous[rel(path)] = freeze_file(path)
    return {"fixed_inputs": frozen, "previous_private_submissions": previous}


def validate_score_row(row: dict[str, Any], label: str) -> None:
    if row.get("model_id") != EXPECTED_MODEL:
        raise RuntimeError(f"{label}: unexpected model at {key_of(row)}")
    if row.get("base_model_revision") != EXPECTED_BASE_REV:
        raise RuntimeError(f"{label}: unexpected base revision at {key_of(row)}")
    if row.get("selector") != EXPECTED_SELECTOR or row.get("aggregation") != EXPECTED_AGGREGATION:
        raise RuntimeError(f"{label}: scorer contract mismatch at {key_of(row)}")
    if row.get("ft_weight_sha256") != EXPECTED_FT_SHA or row.get("ft_config_sha256") != EXPECTED_FT_CONFIG_SHA:
        raise RuntimeError(f"{label}: fine-tuned provenance mismatch at {key_of(row)}")
    if not finite(row.get("bge_ft_score")) or not finite(row.get("bge_base_score")):
        raise RuntimeError(f"{label}: non-finite document score at {key_of(row)}")
    selected = row.get("selected_chunk_ids")
    ft_chunks = row.get("ft_chunk_scores")
    base_chunks = row.get("base_chunk_scores")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 3 or len(set(map(str, selected))) != len(selected):
        raise RuntimeError(f"{label}: invalid selected chunks at {key_of(row)}")
    if not isinstance(ft_chunks, list) or not isinstance(base_chunks, list) or len(ft_chunks) != len(selected) or len(base_chunks) != len(selected):
        raise RuntimeError(f"{label}: chunk-score length mismatch at {key_of(row)}")
    if any(not finite(value) for value in [*ft_chunks, *base_chunks]):
        raise RuntimeError(f"{label}: non-finite chunk score at {key_of(row)}")
    if abs(max(map(float, ft_chunks)) - float(row["bge_ft_score"])) > 1e-6:
        raise RuntimeError(f"{label}: FT MAX mismatch at {key_of(row)}")
    if abs(max(map(float, base_chunks)) - float(row["bge_base_score"])) > 1e-6:
        raise RuntimeError(f"{label}: Base MAX mismatch at {key_of(row)}")


def load_pv1_metadata() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    candidate_rows = read_jsonl(PV1_CANDIDATES)
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    docs_by_query: dict[str, list[dict[str, Any]]] = {}
    if len(candidate_rows) != EXPECTED_F1F4_QUERIES:
        raise RuntimeError(f"PV1 candidate query count: {len(candidate_rows)}")
    for row in candidate_rows:
        qid = str(row["query_id"])
        candidates = row.get("candidates", [])
        if len(candidates) != 20:
            raise RuntimeError(f"PV1 K20 shape mismatch at {qid}: {len(candidates)}")
        seen: set[str] = set()
        materialized: list[dict[str, Any]] = []
        for candidate in candidates:
            did = str(candidate["document_id"])
            if did in seen:
                raise RuntimeError(f"PV1 duplicate candidate {qid}/{did}")
            seen.add(did)
            source_ranks = {str(source): int(rank) for source, rank in (candidate.get("source_ranks") or {}).items()}
            rrf_score = float(candidate.get("rrf_score", 0.0))
            if not finite(rrf_score):
                raise RuntimeError(f"PV1 non-finite RRF at {qid}/{did}")
            rank = int(candidate["candidate_rank"])
            item = {
                "query_id": qid,
                "document_id": did,
                "candidate_rank": rank,
                "union_rank": rank,
                "rrf_score": rrf_score,
                "source_support": len(source_ranks),
                "source_ranks": source_ranks,
            }
            materialized.append(item)
            metadata[(qid, did)] = item
        docs_by_query[qid] = materialized
    if len(metadata) != EXPECTED_F1F4_QDOCS:
        raise RuntimeError(f"PV1 candidate q-doc count: {len(metadata)}")
    return candidate_rows, metadata, docs_by_query


def load_pv1_worklist() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows = read_jsonl(PV1_WORKLIST)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = key_of(row)
        if key in by_key:
            raise RuntimeError(f"PV1 worklist duplicate: {key}")
        selected = row.get("selected_chunk_ids")
        if not isinstance(selected, list) or not 1 <= len(selected) <= 3 or len(set(map(str, selected))) != len(selected):
            raise RuntimeError(f"PV1 worklist selected chunk failure: {key}")
        if int(row.get("expected_inference_units", len(selected))) != len(selected):
            raise RuntimeError(f"PV1 worklist expected unit failure: {key}")
        by_key[key] = row
    if len(rows) != EXPECTED_F1F4_QDOCS:
        raise RuntimeError(f"PV1 worklist q-doc count: {len(rows)}")
    if sum(int(row["expected_inference_units"]) for row in rows) <= 0:
        raise RuntimeError("PV1 worklist has no inference units")
    return rows, by_key


def load_and_merge_scores(worklist_by_key: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    delta_rows = read_jsonl(PV1_DELTA_SCORES)
    reusable_rows = read_jsonl(REUSABLE_SCORES)
    delta: dict[tuple[str, str], dict[str, Any]] = {}
    reusable: dict[tuple[str, str], dict[str, Any]] = {}
    for row in delta_rows:
        key = key_of(row)
        if key in delta:
            raise RuntimeError(f"delta score duplicate: {key}")
        validate_score_row(row, "PV1_DELTA")
        delta[key] = row
    for row in reusable_rows:
        key = key_of(row)
        if key in reusable:
            raise RuntimeError(f"reusable score duplicate: {key}")
        validate_score_row(row, "REUSABLE")
        reusable[key] = row
    if not set(delta).issubset(worklist_by_key):
        raise RuntimeError("delta score has identities outside PV1 worklist")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    source_counts = Counter()
    selected_chunk_mismatches: list[tuple[str, str]] = []
    for key, work_row in worklist_by_key.items():
        if key in delta:
            row = delta[key]
            source_counts["PV1_DELTA"] += 1
        elif key in reusable:
            row = reusable[key]
            source_counts["REUSED_HISTORICAL_PRODUCTION"] += 1
        else:
            raise RuntimeError(f"no score source for PV1 identity: {key}")
        if [str(x) for x in row["selected_chunk_ids"]] != [str(x) for x in work_row["selected_chunk_ids"]]:
            selected_chunk_mismatches.append(key)
        selected[key] = row
    if selected_chunk_mismatches:
        raise RuntimeError(f"selected chunk mismatches while reconstructing: {selected_chunk_mismatches[:5]}")
    if len(selected) != EXPECTED_F1F4_QDOCS:
        raise RuntimeError("reconstructed score count mismatch")
    reconstructed = [selected[key] for key in worklist_by_key]
    merge_info = {
        "delta_rows_available": len(delta),
        "reusable_rows_available": len(reusable),
        "reconstructed_rows": len(reconstructed),
        "source_counts": dict(source_counts),
        "reused_fraction": source_counts["REUSED_HISTORICAL_PRODUCTION"] / len(reconstructed),
        "delta_fraction": source_counts["PV1_DELTA"] / len(reconstructed),
        "selected_chunk_mismatches": 0,
        "old_score_extra_identities_not_in_pv1": len(set(reusable) - set(worklist_by_key)),
    }
    return reconstructed, merge_info


def bge_rank(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row["document_id"])
        for row in sorted(
            rows,
            key=lambda row: (
                -float(row["bge_ft_score"]),
                -float(row["bge_base_score"]),
                int(row["candidate_rank"]),
                qkey(str(row["document_id"])),
            ),
        )
    ]


def current_rrf_rank(rows: list[dict[str, Any]]) -> list[str]:
    """Reproduce frozen RETRIEVAL_RRF_NO_LABEL final ranking."""
    bge_order = bge_rank(rows)
    bge_rank_map = {did: index for index, did in enumerate(bge_order, 1)}
    weights = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}
    scores: dict[str, float] = {}
    ranks_by_doc: dict[str, dict[str, int]] = {}
    for row in rows:
        did = str(row["document_id"])
        source_ranks = {str(k): int(v) for k, v in (row.get("source_ranks") or {}).items()}
        source_ranks["bge"] = bge_rank_map[did]
        ranks_by_doc[did] = source_ranks
        scores[did] = sum(weight / (2 + source_ranks[source]) for source, weight in weights.items() if source in source_ranks)
    return sorted(
        scores,
        key=lambda did: (
            -scores[did],
            *(ranks_by_doc[did].get(source, 10**9) for source in weights),
            qkey(did),
        ),
    )


def minmax(values: dict[str, float]) -> dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    span = max(1e-6, high - low)
    return {key: (value - low) / span for key, value in values.items()}


def make_step4_features(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], list[str]]:
    docs = [str(row["document_id"]) for row in rows]
    ft = minmax({str(row["document_id"]): float(row["bge_ft_score"]) for row in rows})
    base = minmax({str(row["document_id"]): float(row["bge_base_score"]) for row in rows})
    rrf = minmax({str(row["document_id"]): float(row["rrf_score"]) for row in rows})
    ensemble: dict[str, float] = {}
    features: dict[str, dict[str, float]] = {}
    for row in rows:
        did = str(row["document_id"])
        support_ratio = float(row["source_support"]) / 4.0
        score = (
            STEP4_WEIGHTS["ft"] * ft[did]
            + STEP4_WEIGHTS["base"] * base[did]
            + STEP4_WEIGHTS["rrf"] * rrf[did]
            + STEP4_WEIGHTS["support"] * support_ratio
        )
        ensemble[did] = score
        features[did] = {
            "bge_ft_raw": float(row["bge_ft_score"]),
            "bge_base_raw": float(row["bge_base_score"]),
            "norm_ft": ft[did],
            "norm_base": base[did],
            "rrf_raw": float(row["rrf_score"]),
            "norm_rrf": rrf[did],
            "source_support": float(row["source_support"]),
            "support_ratio_denominator_4": support_ratio,
            "candidate_rank": float(row["candidate_rank"]),
            "ensemble_score": score,
        }
    ordered = sorted(docs, key=lambda did: -ensemble[did])
    return features, ordered


def historical_rank_safe(added: dict[str, Any], dropped: dict[str, Any]) -> bool:
    return not (
        int(added.get("union_rank", 999)) > int(dropped.get("union_rank", 999))
        and int(added.get("source_support", 1)) <= int(dropped.get("source_support", 1))
    )


def step4_query(rows: list[dict[str, Any]], baseline_top5: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Apply the recovered raw Step4 policy and then its calibration gate."""
    qid = str(rows[0]["query_id"])
    by_doc = {str(row["document_id"]): row for row in rows}
    features, ensemble_order = make_step4_features(rows)
    base = [str(value) for value in baseline_top5]
    if len(base) != 5 or len(set(base)) != 5:
        raise RuntimeError(f"invalid baseline top5: {qid}")
    if any(did not in by_doc for did in base):
        raise RuntimeError(f"baseline anchor absent from K20: {qid}")
    outside = [did for did in ensemble_order if did not in base]
    raw_swaps: list[dict[str, Any]] = []
    if outside:
        incoming = outside[0]
        dropped = base[4]
        inc_ft = features[incoming]["norm_ft"]
        inc_base = features[incoming]["norm_base"]
        drop_ft = features[dropped]["norm_ft"]
        drop_base = features[dropped]["norm_base"]
        delta = features[incoming]["ensemble_score"] - features[dropped]["ensemble_score"]
        models_agree = inc_ft >= drop_ft - 0.02 and inc_base >= drop_base - 0.02
        margin_ok = features[incoming]["ensemble_score"] > features[dropped]["ensemble_score"] + STEP4_MARGIN_RANK5
        retrieval_safe = (
            not (
                int(by_doc[dropped]["source_support"]) >= 4
                and int(by_doc[dropped]["union_rank"]) <= 10
            )
            or int(by_doc[incoming]["source_support"]) >= 2
            or int(by_doc[incoming]["union_rank"]) <= 20
        )
        raw_ok = margin_ok and models_agree and retrieval_safe
        if raw_ok:
            raw_swaps.append(
                {
                    "pos": 5,
                    "added": incoming,
                    "dropped": dropped,
                    "added_meta": by_doc[incoming],
                    "dropped_meta": by_doc[dropped],
                    "delta_ens_raw": delta,
                    # Historical report stored this value rounded to 4 places,
                    # and the calibration applier compared that persisted value.
                    "delta_ens": round(delta, 4),
                    "raw_margin_ok": margin_ok,
                    "raw_models_agree": models_agree,
                    "raw_retrieval_safe": retrieval_safe,
                }
            )
            if len(outside) >= 2:
                incoming2 = outside[1]
                dropped4 = base[3]
                delta4 = features[incoming2]["ensemble_score"] - features[dropped4]["ensemble_score"]
                margin4 = features[incoming2]["ensemble_score"] > features[dropped4]["ensemble_score"] + STEP4_MARGIN_RANK4
                models4 = (
                    features[incoming2]["norm_ft"] >= features[dropped4]["norm_ft"]
                    and features[incoming2]["norm_base"] >= features[dropped4]["norm_base"]
                )
                safe4 = (
                    not (
                        int(by_doc[dropped4]["source_support"]) >= 4
                        and int(by_doc[dropped4]["union_rank"]) <= 10
                    )
                    or int(by_doc[incoming2]["source_support"]) >= 2
                )
                if margin4 and models4 and safe4:
                    raw_swaps.append(
                        {
                            "pos": 4,
                            "added": incoming2,
                            "dropped": dropped4,
                            "added_meta": by_doc[incoming2],
                            "dropped_meta": by_doc[dropped4],
                            "delta_ens_raw": delta4,
                            "delta_ens": round(delta4, 4),
                            "raw_margin_ok": margin4,
                            "raw_models_agree": models4,
                            "raw_retrieval_safe": safe4,
                        }
                    )
    final = list(base)
    accepted: list[dict[str, Any]] = []
    for swap in raw_swaps:
        accepted_by_calibration = (
            float(swap["delta_ens"]) >= STEP4_DELTA
            and historical_rank_safe(swap["added_meta"], swap["dropped_meta"])
        )
        swap["calibration_delta_threshold_pass"] = float(swap["delta_ens"]) >= STEP4_DELTA
        swap["calibration_rank_safe_pass"] = historical_rank_safe(swap["added_meta"], swap["dropped_meta"])
        swap["accepted_by_calibration"] = accepted_by_calibration
        if accepted_by_calibration:
            final[int(swap["pos"]) - 1] = str(swap["added"])
            accepted.append(swap)
    if len(final) != 5 or len(set(final)) != 5:
        raise RuntimeError(f"Step4 produced invalid Top5 for {qid}: {final}")
    return final, {
        "query_id": qid,
        "baseline_top5": base,
        "step4_top5": final,
        "raw_swaps": raw_swaps,
        "accepted_swaps": accepted,
        "features": features,
        "ensemble_order": ensemble_order,
        "top1_top3_locked": final[:3] == base[:3],
    }


def evaluate_port(
    baseline: dict[str, list[str]],
    proposed: dict[str, list[str]],
    gold: dict[str, dict[str, Any]],
    folds: dict[str, int],
    traces: dict[str, dict[str, Any]],
    provenance_pass: bool,
) -> dict[str, Any]:
    baseline_metric = metric(baseline, gold)
    proposed_metric = metric(proposed, gold)
    baseline_by_fold: dict[str, dict[str, Any]] = {}
    proposed_by_fold: dict[str, dict[str, Any]] = {}
    fold_delta: dict[str, dict[str, float]] = {}
    for fold in TARGET_FOLDS:
        ids = [qid for qid in proposed if folds[qid] == fold]
        b = metric({qid: baseline[qid] for qid in ids}, gold)
        p = metric({qid: proposed[qid] for qid in ids}, gold)
        label = f"F{fold}"
        baseline_by_fold[label] = b
        proposed_by_fold[label] = p
        fold_delta[label] = {"recall": p["recall"] - b["recall"], "precision": p["precision"] - b["precision"]}
    changed = improved = harmed = neutral_changed = 0
    gained = lost = 0
    for qid, new_top in proposed.items():
        if new_top == baseline[qid]:
            continue
        changed += 1
        wanted = {str(value) for value in gold[qid].get("answer", [])}
        old_hits = len(wanted & set(baseline[qid]))
        new_hits = len(wanted & set(new_top))
        if new_hits > old_hits:
            improved += 1
            gained += new_hits - old_hits
        elif new_hits < old_hits:
            harmed += 1
            lost += old_hits - new_hits
        else:
            neutral_changed += 1
    swap_rank = Counter()
    raw_swap_rank = Counter()
    for trace in traces.values():
        for swap in trace["accepted_swaps"]:
            swap_rank[str(swap["pos"])] += 1
        for swap in trace["raw_swaps"]:
            raw_swap_rank[str(swap["pos"])] += 1
    delta_recall = proposed_metric["recall"] - baseline_metric["recall"]
    delta_precision = proposed_metric["precision"] - baseline_metric["precision"]
    strong = (
        delta_recall >= 0.0015
        and all(value["recall"] >= 0 for value in fold_delta.values())
        and delta_precision >= -0.001
        and provenance_pass
    )
    weak = (
        delta_recall > 0
        and all(value["recall"] >= 0 for value in fold_delta.values())
        and delta_precision >= -0.001
        and provenance_pass
    )
    return {
        "baseline": baseline_metric,
        "proposed": proposed_metric,
        "delta": {"recall": delta_recall, "precision": delta_precision},
        "baseline_by_fold": baseline_by_fold,
        "proposed_by_fold": proposed_by_fold,
        "delta_by_fold": fold_delta,
        "queries_changed": changed,
        "queries_improved": improved,
        "queries_harmed": harmed,
        "queries_neutral_changed": neutral_changed,
        "relevant_docs_gained": gained,
        "relevant_docs_lost": lost,
        "net_relevant_doc_change": gained - lost,
        "raw_swap_count": sum(raw_swap_rank.values()),
        "calibrated_swap_count": sum(swap_rank.values()),
        "raw_swaps_by_rank": dict(raw_swap_rank),
        "calibrated_swaps_by_rank": dict(swap_rank),
        "top1_top3_lock_pass": all(trace["top1_top3_locked"] for trace in traces.values()),
        "provenance_pass": provenance_pass,
        "gate": "STRONG_PASS" if strong else "WEAK_PASS" if weak else "FAIL",
    }


def verify_historical_step4() -> dict[str, Any]:
    manifest = json.loads(STEP4_MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    checked = []
    missing = []
    mismatches = []
    for entry in entries:
        path = STEP4_ROOT / str(entry["relative_path"])
        if not path.is_file():
            missing.append({"path": entry["relative_path"], "manifest_sha256": entry.get("sha256")})
            continue
        actual = sha256(path)
        record = {"path": rel(path), "manifest_sha256": entry.get("sha256"), "actual_sha256": actual, "match": actual == entry.get("sha256")}
        checked.append(record)
        if not record["match"]:
            mismatches.append(record)

    with zipfile.ZipFile(STEP4_ANCHOR_ZIP) as archive:
        anchor = json.loads(archive.read("submission.json").decode("utf-8"))
    report = json.loads(STEP4_REPORT.read_text(encoding="utf-8"))
    replay = {qid: {"answer": list(row["answer"])} for qid, row in anchor.items()}
    accepted_count = 0
    accepted_queries: set[str] = set()
    raw_count = 0
    for entry in report.get("swap_log", []):
        qid = str(entry["qid"])
        for swap in entry.get("swaps", []):
            raw_count += 1
            added_meta = swap.get("added_meta", {})
            dropped_meta = swap.get("dropped_meta", {})
            delta = float(swap.get("delta_ens", float("-inf")))
            accepted = delta >= STEP4_DELTA and not (
                int(added_meta.get("union_rank", 999)) > int(dropped_meta.get("union_rank", 999))
                and int(added_meta.get("source_support", 1)) <= int(dropped_meta.get("source_support", 1))
            )
            if accepted:
                replay[qid]["answer"][int(swap["pos"]) - 1] = str(swap["added"])
                accepted_count += 1
                accepted_queries.add(qid)
    frozen = json.loads(STEP4_FINAL_JSON.read_text(encoding="utf-8"))
    semantic_equal = replay == frozen
    report_counts = {
        "raw_swap_log_rows": raw_count,
        "raw_swapped_queries": len(report.get("swap_log", [])),
        "calibrated_swaps_replayed": accepted_count,
        "calibrated_queries_replayed": len(accepted_queries),
        "manifest_reported_total_swaps": report.get("total_swaps"),
        "manifest_reported_swapped_queries": report.get("swapped_queries_count"),
    }
    replay_gate = "PASS" if semantic_equal and not mismatches and accepted_count == 77 and len(accepted_queries) == 71 else "FAIL"
    return {
        "lineage_status": "FULLY_RECOVERED" if not missing and not mismatches else "PARTIALLY_RECOVERED",
        "deployment_evidence": "PARTIAL",
        "manifest_entries": len(entries),
        "checked_entries": checked,
        "missing_entries": missing,
        "hash_mismatches": mismatches,
        "historical_contract": {
            "shortlist_top_n": 50,
            "weights": STEP4_WEIGHTS,
            "minmax_per_query": True,
            "minmax_floor": 1e-6,
            "raw_margin_rank5": STEP4_MARGIN_RANK5,
            "raw_margin_rank4": STEP4_MARGIN_RANK4,
            "calibration_delta": STEP4_DELTA,
            "rank_safe": "added union rank <= dropped union rank OR added source support > dropped source support",
            "top1_top3_locked": True,
            "positions": [4, 5],
            "candidate_order_tiebreak": "stable sort by descending ensemble score over shortlist insertion order",
            "evidence_selector": "precomputed public evidence when available, otherwise lexical-overlap fallback in historical generator; exact public cache not included in handoff",
        },
        "frozen_public": {
            "anchor_zip_sha256": sha256(STEP4_ANCHOR_ZIP),
            "calibrated_json_sha256": sha256(STEP4_FINAL_JSON),
            "calibrated_zip_sha256": sha256(STEP4_FINAL_ZIP),
            "recorded_claim": "0.9391 -> 0.9411 in repository validation/history reports; no external leaderboard receipt is present",
        },
        "replay": {
            "semantic_prediction_equality": semantic_equal,
            "gate": replay_gate,
            "counts": report_counts,
        },
    }


def build_feature_matrix(
    docs_by_query: dict[str, list[dict[str, Any]]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
    folds: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in sorted(docs_by_query, key=qkey):
        joined = []
        for meta in docs_by_query[qid]:
            key = (qid, str(meta["document_id"]))
            row = {**meta, **score_by_key[key]}
            joined.append(row)
        features, _ = make_step4_features(joined)
        for meta in docs_by_query[qid]:
            did = str(meta["document_id"])
            rows.append({
                "query_id": qid,
                "fold": folds[qid],
                "document_id": did,
                "candidate_rank": int(meta["candidate_rank"]),
                "source_ranks": meta["source_ranks"],
                "selected_chunk_ids": [str(x) for x in score_by_key[(qid, did)]["selected_chunk_ids"]],
                **features[did],
            })
    return rows


def run_step4_port(
    docs_by_query: dict[str, list[dict[str, Any]]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
    baseline: dict[str, list[str]],
    folds: dict[str, int],
    gold: dict[str, dict[str, Any]],
    provenance_pass: bool,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, dict[str, Any]]]:
    proposed: dict[str, list[str]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for qid in sorted(docs_by_query, key=qkey):
        joined = [{**meta, **score_by_key[(qid, str(meta["document_id"]))]} for meta in docs_by_query[qid]]
        final, trace = step4_query(joined, baseline[qid])
        proposed[qid] = final
        traces[qid] = trace
    result = evaluate_port(baseline, proposed, gold, folds, traces, provenance_pass)
    return result, proposed, traces


def build_private_meta_and_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    candidate_rows = read_jsonl(PRIVATE_CANDIDATES)
    score_rows = read_jsonl(PRIVATE_SCORES)
    score_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in score_rows:
        key = key_of(row)
        if key in score_by_key:
            raise RuntimeError(f"Private score duplicate: {key}")
        validate_score_row(row, "PRIVATE")
        score_by_key[key] = row
    docs_by_query: dict[str, list[dict[str, Any]]] = {}
    expected_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        qid = str(row["query_id"])
        docs: list[dict[str, Any]] = []
        for candidate in row.get("candidates", []):
            did = str(candidate["document_id"])
            source_ranks = {str(source): int(rank) for source, rank in (candidate.get("source_ranks") or {}).items()}
            # The Private candidate artifact stores source ranks but not the
            # unweighted K=60 union RRF scalar.  Recompute that deterministic
            # retrieval signal from the persisted source ranks.
            rrf_score = sum(1.0 / (60 + rank) for rank in source_ranks.values())
            docs.append({
                "query_id": qid,
                "document_id": did,
                "candidate_rank": int(candidate["candidate_rank"]),
                "union_rank": int(candidate["candidate_rank"]),
                "rrf_score": rrf_score,
                "source_support": len(source_ranks),
                "source_ranks": source_ranks,
            })
            expected_keys.add((qid, did))
        docs_by_query[qid] = docs
    if len(docs_by_query) != 2080 or any(len(rows) != 20 for rows in docs_by_query.values()):
        raise RuntimeError("Private K20 candidate universe is not 2080 x 20")
    if expected_keys != set(score_by_key):
        raise RuntimeError("Private K20 candidate/score identity mismatch")
    return docs_by_query, score_by_key


def write_private_checkpoint(proposed: dict[str, list[str]], validation: dict[str, Any]) -> dict[str, Any]:
    output_dir = ROOT / "private_task1/submissions/sprint48_step4"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "submission_private_step4_pv1.json"
    zip_path = output_dir / "submission_private_step4_pv1.zip"
    manifest_path = output_dir / "submission_private_step4_pv1_manifest.json"
    payload = {qid: {"answer": docs} for qid, docs in sorted(proposed.items(), key=lambda item: qkey(item[0]))}
    if len(payload) != 2080 or any(len(row["answer"]) != 5 or len(set(row["answer"])) != 5 for row in payload.values()):
        raise RuntimeError("invalid Step4 Private checkpoint shape")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="submission.json")
    manifest = {
        "status": "CPU_ONLY_STEP4_K20_PORT_CHECKPOINT",
        "policy": "STEP4_K20_PORT",
        "qdocs": 2080,
        "top_k": 5,
        "source_submission": rel(PV1_SUBMISSION_JSON),
        "source_submission_sha256": sha256(PV1_SUBMISSION_JSON),
        "json": rel(json_path),
        "json_sha256": sha256(json_path),
        "zip": rel(zip_path),
        "zip_sha256": sha256(zip_path),
        "validation": validation,
        "fold0_used": False,
        "private_labels_used": False,
        "gpu_runs": 0,
        "modal_runs": 0,
    }
    write_json(manifest_path, manifest)
    return manifest


def run_direct_k20_fallback(
    docs_by_query: dict[str, list[dict[str, Any]]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
    baseline: dict[str, list[str]],
    folds: dict[str, int],
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """One fixed outer-OOF fallback using the repository meta-ranker.

    ``udsc2026.evaluation.legal_ir_meta_fusion.nested_oof_meta_ranker`` is the
    existing CPU LogisticRegression implementation.  It is used once with
    source depth 20 and fixed ``negatives_per_query=30`` (which means all
    available K20 non-gold candidates, never a sweep).  The fallback is
    diagnostic/validation only; it does not create a Private checkpoint.
    """
    from udsc2026.evaluation.legal_ir_meta_fusion import nested_oof_meta_ranker

    source_names = ("dense", "bm25", "knn_word")
    source_rankings: dict[str, dict[str, list[str]]] = {name: {} for name in source_names}
    external: dict[str, dict[str, dict[str, float]]] = {}
    for qid in sorted(docs_by_query, key=qkey):
        docs = docs_by_query[qid]
        for source in source_names:
            source_rankings[source][qid] = [
                did for did, _rank in sorted(
                    (
                        (str(meta["document_id"]), int(meta["source_ranks"].get(source, 10**9)))
                        for meta in docs
                        if source in meta["source_ranks"]
                    ),
                    key=lambda item: (item[1], qkey(item[0])),
                )
            ]
        ft_values = {str(meta["document_id"]): float(score_by_key[(qid, str(meta["document_id"]))]["bge_ft_score"]) for meta in docs}
        base_values = {str(meta["document_id"]): float(score_by_key[(qid, str(meta["document_id"]))]["bge_base_score"]) for meta in docs}
        ft_norm = minmax(ft_values)
        base_norm = minmax(base_values)
        max_rank = max(1, len(docs) - 1)
        external[qid] = {}
        for meta in docs:
            did = str(meta["document_id"])
            external[qid][did] = {
                "bge_ft_raw": ft_values[did],
                "bge_base_raw": base_values[did],
                "bge_ft_minmax": ft_norm[did],
                "bge_base_minmax": base_norm[did],
                "candidate_rank_norm": (len(docs) - int(meta["candidate_rank"])) / max_rank,
                "union_rrf": float(meta["rrf_score"]),
                "source_support": float(meta["source_support"]),
            }
    fold_gold = {qid: [str(value) for value in row.get("answer", [])] for qid, row in gold.items() if folds.get(qid) in TARGET_FOLDS}
    pred, fold_reports = nested_oof_meta_ranker(
        gold=fold_gold,
        folds={qid: folds[qid] for qid in fold_gold},
        sources=source_rankings,
        external_features=external,
        source_depth=20,
        negatives_per_query=30,
    )
    result = evaluate_port(baseline, pred, gold, folds, {}, provenance_pass=True)
    result.update({
        "policy": "DIRECT_K20_PLUS_ANCHORS_DOCUMENT_TOP5",
        "estimator": "udsc2026.evaluation.legal_ir_meta_fusion.nested_oof_meta_ranker",
        "model": "StandardScaler + LogisticRegression(C=0.2, class_weight=balanced, solver=liblinear, max_iter=500, random_state=2026)",
        "outer_oof": True,
        "source_depth": 20,
        "negatives_per_query": 30,
        "feature_names": sorted([
            "rr_dense", "rr_bm25", "rr_knn_word", "source_support_count",
            "bge_ft_raw", "bge_base_raw", "bge_ft_minmax", "bge_base_minmax",
            "candidate_rank_norm", "union_rrf", "source_support",
        ]),
        "fold_reports": fold_reports,
        "private_application": "NOT_RUN",
        "fallback_gate": (
            result["delta"]["recall"] >= 0.002
            and all(value["recall"] >= 0 for value in result["delta_by_fold"].values())
            and result["delta"]["precision"] >= -0.001
        ),
    })
    return result


def write_ft_v2_spec(
    merge_info: dict[str, Any],
    freeze: dict[str, Any],
    estimate: dict[str, Any],
) -> None:
    path = OUT / "same_bge_hard_negative_ft_v2_spec.md"
    text = f"""# SAME-BGE hard-negative fine-tuning V2 specification

Status: **READY_FOR_REVIEW — NOT TRAINED**.

This is a Phase 2 specification only. No training, checkpoint creation, GPU
run, Modal run, model load, or promotion occurred in Phase 1.

## Frozen backbone and inference contract

- Backbone: `BAAI/bge-reranker-v2-m3`
- Base revision: `{EXPECTED_BASE_REV}`
- Existing fine-tuned teacher artifact SHA256: `{EXPECTED_FT_SHA}`
- Existing fine-tuned config SHA256: `{EXPECTED_FT_CONFIG_SHA}`
- Max length: `512`
- Selector: `{EXPECTED_SELECTOR}`
- Aggregation: `MAX` over selected raw chunks
- Candidate/evidence unit: same raw-chunk construction used by the current BGE scorer
- Existing trainer to reuse after review: `scripts/training/finetune_task1_bge_reranker_v3.py`
- Existing group builder: `scripts/training/build_task1_chunk_training_v3.py`
- Existing negative miner: `scripts/training/mine_task1_negatives.py`

## Training population and leakage guard

- Use `data/raw/btc/LegalIR/train.json` labels only.
- Use strict folds from `artifacts/task1/evaluation/strict_cv_v2/folds.json`.
- For each held-out fold, exclude every query in that fold from training-group
  construction. Never read Private answers and never use Fold0 for model
  selection or promotion.
- Reuse the PV1 retrieval/candidate universe and current BGE scores as frozen
  teacher-side evidence; do not rebuild retrieval in this phase.
- Repaired PV1 corpus remains frozen; do not edit `data/processed_pv1`.

## Positive and hard-negative construction

For each training query, create positive examples from gold documents that are
present in the candidate/worklist and use the same selected raw chunks used at
inference. Prefer the gold chunk with the highest current teacher signal while
retaining up to three selected chunks under the current selector.

For each positive, sample deterministic non-gold negatives in this priority:

1. non-gold candidates with high current BGE score;
2. candidates that displaced or marginalized a gold document in current K20;
3. high-BM25 non-gold candidates;
4. high-Dense non-gold candidates;
5. high source-support non-gold candidates;
6. semantically/lexically similar legal documents with the wrong identity.

Use fixed per-query caps after deterministic priority sorting. Do not sample
from Private data and do not select the cap or hyperparameters by leaderboard
feedback.

## Training objective and fixed starting recipe

- Exact starting loss: the existing binary teacher-anchored loss in
  `finetune_task1_bge_reranker_v3.py`,
  `(1 - teacher_weight) * BCEWithLogits(student, target) + teacher_weight *
  MSE(student_logits, teacher_logits)`. Each selected positive/negative
  hard-negative pair produces one positive example (`1`) and one negative
  example (`0`). This is fixed before any training; no pairwise/listwise loss
  alternative is opened in V2.
- Teacher anchoring against the current frozen BGE teacher is retained to avoid
  catastrophic drift.
- Fixed starting values to preregister: learning rate `5e-7`, `1` epoch,
  batch size `4`, gradient accumulation `4`, max length `512`, freeze the
  first `18` encoder layers, teacher weight `0.5`, deterministic seed `2026`.
- No learning-rate, epoch, margin, negative-cap, model-family, or selector
  sweep in Phase 2. A failed gate closes V2.

## Validation and promotion

- Run strict outer OOF on F1-F4 only; compare against the frozen current PV1
  BGE baseline on the same candidate universe.
- Required gate to continue: pooled Recall delta at least `+0.0015`, every
  F1-F4 fold delta non-negative, Precision delta at least `-0.001`, all score
  finite, exact candidate/worklist provenance, and no Top1-3 guardrail change
  for any deployment policy that uses Step4-style residual swaps.
- Keep the existing BGE teacher and current PV1 submission untouched until a
  separately reviewed checkpoint passes the gate.

## CPU-only work-size estimate

- Negative cap: `{estimate['max_negatives_per_positive']}` per positive
  document, after deterministic priority ordering.
- F1-F4 reference work: `{estimate['all_f1_f4_reference_work']['estimated_pairs']}`
  estimated positive/negative pairs =
  `{estimate['all_f1_f4_reference_work']['estimated_training_examples']}`
  binary training examples, `{estimate['all_f1_f4_reference_work']['estimated_optimizer_steps']}`
  optimizer steps at batch `4` and gradient accumulation `4`.
- Strict outer-OOF training work by held-out fold:
  `{json.dumps(estimate['strict_outer_oof_training_work'], ensure_ascii=False, sort_keys=True)}`
- These are serialized pair/forward-unit counts only; wall time is not
  estimated because no comparable training benchmark was run.

## Current Phase 1 evidence

- Reconstructed current PV1 validation rows: `{merge_info['reconstructed_rows']}`.
- Reused score rows: `{merge_info['source_counts'].get('REUSED_HISTORICAL_PRODUCTION', 0)}`.
- PV1 delta score rows: `{merge_info['source_counts'].get('PV1_DELTA', 0)}`.
- Input hash ledger: `{rel(OUT / 'input_hashes.json')}`.

Training remains **NOT EXECUTED** here. The estimate is not training evidence
and must be revalidated against the final reviewed worklist before launch.
"""
    path.write_text(text, encoding="utf-8")


def estimate_ft_v2_work(
    docs_by_query: dict[str, list[dict[str, Any]]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
    baseline: dict[str, list[str]],
    folds: dict[str, int],
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Estimate fixed hard-negative work without loading a model or training."""
    max_negatives_per_positive = 6
    per_query: dict[str, dict[str, int]] = {}
    for qid in sorted(docs_by_query, key=qkey):
        gold_docs = {str(value) for value in gold[qid].get("answer", [])}
        rows = docs_by_query[qid]
        bge_order = bge_rank([{**meta, **score_by_key[(qid, str(meta["document_id"]))]} for meta in rows])
        bge_rank_map = {did: index for index, did in enumerate(bge_order, 1)}
        gold_in_candidate = [meta for meta in rows if str(meta["document_id"]) in gold_docs]
        negatives = [meta for meta in rows if str(meta["document_id"]) not in gold_docs]
        baseline_set = set(baseline[qid])

        def neg_key(meta: dict[str, Any]) -> tuple[Any, ...]:
            did = str(meta["document_id"])
            source = meta["source_ranks"]
            displaced_gold = bool(gold_docs - baseline_set) and did in baseline_set
            if bge_rank_map[did] <= 10:
                priority = 0  # high current BGE score
            elif displaced_gold:
                priority = 1  # current anchor marginalizes a gold document
            elif int(source.get("bm25", 10**9)) <= 10:
                priority = 2
            elif int(source.get("dense", 10**9)) <= 10:
                priority = 3
            elif int(meta["source_support"]) >= 2:
                priority = 4
            else:
                priority = 5
            return (
                priority,
                bge_rank_map[did],
                int(source.get("bm25", 10**9)),
                int(source.get("dense", 10**9)),
                -int(meta["source_support"]),
                qkey(did),
            )

        negatives.sort(key=neg_key)
        positive_count = len(gold_in_candidate)
        selected_negative_count = min(max_negatives_per_positive, len(negatives))
        per_query[qid] = {
            "positive_documents_in_k20": positive_count,
            "available_non_gold_k20": len(negatives),
            "negatives_per_positive": selected_negative_count,
            "estimated_pairs": positive_count * selected_negative_count,
        }

    def summarize(query_ids: Iterable[str]) -> dict[str, Any]:
        selected = [per_query[qid] for qid in query_ids]
        positives = sum(row["positive_documents_in_k20"] for row in selected)
        pairs = sum(row["estimated_pairs"] for row in selected)
        queries = len(selected)
        return {
            "queries": queries,
            "positive_document_examples": positives,
            "estimated_pairs": pairs,
            "estimated_training_examples": pairs * 2,
            "batch_size": 4,
            "gradient_accumulation": 4,
            "epochs": 1,
            "estimated_optimizer_steps": math.ceil((pairs * 2) / 16) if pairs else 0,
            "estimated_forward_pair_units": pairs,
            "estimated_forward_examples": pairs * 2,
        }

    outer = {
        str(held_out): summarize(qid for qid in docs_by_query if folds[qid] != held_out)
        for held_out in TARGET_FOLDS
    }
    full = summarize(docs_by_query.keys())
    return {
        "status": "CPU_ONLY_WORK_SIZE_ESTIMATE",
        "candidate_universe": "PV1 F1-F4 K20",
        "fold0_used": False,
        "private_labels_used": False,
        "max_negatives_per_positive": max_negatives_per_positive,
        "negative_priority": [
            "high current BGE score but non-gold",
            "candidate that displaces/marginalizes a gold document",
            "high BM25 non-gold",
            "high Dense non-gold",
            "high source-support non-gold",
            "similar wrong-identity residual candidate",
        ],
        "strict_outer_oof_training_work": outer,
        "all_f1_f4_reference_work": full,
        "runtime_seconds": "NOT_ESTIMATED — no comparable training throughput was run",
    }


def write_ledger(
    historical: dict[str, Any],
    step4: dict[str, Any],
    fallback: dict[str, Any] | None,
    merge_info: dict[str, Any],
) -> None:
    gate = step4.get("gate", "NOT_RUN")
    action = (
        "MANUAL_SUBMIT_STEP4_CHECKPOINT" if gate in {"STRONG_PASS", "WEAK_PASS"} else
        "RUN_FALLBACK_DIRECT_TOP5" if fallback is not None and fallback.get("fallback_gate") else
        "RUN_HARD_NEGATIVE_BGE_FT_V2"
    )
    lines = [
        "# Sprint 48-hour Phase 1 experiment ledger",
        "",
        "CPU-only recovery/replay ledger. No GPU, Modal, Qwen inference, Private answers, Fold0 labels, or submission upload.",
        "",
        "## Frozen facts",
        "",
        f"- Current PV1 F1-F4 baseline population: `{EXPECTED_F1F4_QUERIES}` queries / `{EXPECTED_F1F4_QDOCS}` q-docs.",
        f"- BGE score reconstruction: `{merge_info['source_counts']}`; selected chunk mismatches `{merge_info['selected_chunk_mismatches']}`.",
        "- Current corpus/data integrity and deterministic BGE gates were inherited as PASS from the prior forensic audit.",
        "- Historical Step4 artifacts are read-only; the public 0.9411 result is not treated as Private gold or as a leaderboard tuning target.",
        "",
        "## Experiment classifications",
        "",
        "| Experiment/state | Classification | Note |",
        "|---|---|---|",
        "| V3A / V3B / Workflow A / B1 LambdaMART | PROVENANCE_BLOCKED or CLOSED in prior ledger | Not reopened in Phase 1 |",
        "| Qwen standalone | SCIENTIFIC_FAIL / not a final policy | Candidate discovery only |",
        "| Qwen auxiliary / Workflow C C2 | INCONCLUSIVE or CLOSED | Not used for this replay |",
        "| direct relevance / direct neural / pairwise preference | INCONCLUSIVE | Not selected or swept |",
        "| full-document retrieval | ENGINEERING/SIGNAL evidence | Not rerun |",
        f"| Step4 calibrated historical | {historical.get('lineage_status', 'UNKNOWN')} / deployment evidence {historical.get('deployment_evidence', 'UNKNOWN')} | Public artifact replay gate `{historical.get('replay', {}).get('gate', 'UNKNOWN')}` |",
        "| Step5A / Step5B / Dual-Lever | CLOSED historical | No threshold/weight sweep |",
        "| current PV1 | SCIENTIFIC_BASELINE | Frozen `RETRIEVAL_RRF_NO_LABEL` |",
        "",
        "## Phase 1 result",
        "",
        f"- `STEP4_K20_PORT` gate: **{gate}**.",
        f"- Step4 K20 recall: `{step4.get('proposed', {}).get('recall', 'N/A')}`; delta: `{step4.get('delta', {}).get('recall', 'N/A')}`.",
        f"- Fold deltas: `{step4.get('delta_by_fold', {})}`.",
        f"- Private Step4 checkpoint: `{step4.get('private_checkpoint', {}).get('status', 'NOT_CREATED')}`.",
    ]
    if fallback is not None:
        lines.extend([
            "",
            "## Single fallback",
            "",
            f"- Policy: `DIRECT_K20_PLUS_ANCHORS_DOCUMENT_TOP5`; gate: **{'PASS' if fallback.get('fallback_gate') else 'FAIL'}**.",
            f"- Recall delta: `{fallback.get('delta', {}).get('recall')}`; fold deltas: `{fallback.get('delta_by_fold')}`.",
            "- No fallback sweep and no Private fallback application were run.",
        ])
    lines.extend([
        "",
        "## Phase 2 handoff",
        "",
        f"- SAME-BGE hard-negative FT V2 specification: `{rel(OUT / 'same_bge_hard_negative_ft_v2_spec.md')}`.",
        f"- SAME-BGE FT V2 CPU work-size estimate: `{rel(OUT / 'same_bge_hard_negative_ft_v2_estimate.json')}`.",
        "- Training remains unexecuted and requires separate review/authorization.",
        "",
        f"## Recommended final action: `{action}`",
        "",
    ])
    (OUT / "experiment_ledger.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = freeze_inputs()
    write_json(OUT / "input_hashes.json", freeze)

    historical = verify_historical_step4()
    write_json(OUT / "historical_step4_lineage.json", historical)
    if historical["replay"]["gate"] != "PASS":
        raise RuntimeError("historical Step4 CPU replay gate failed")

    folds = load_fold_map()
    gold_all = load_gold()
    target_qids = {qid for qid, fold in folds.items() if fold in TARGET_FOLDS}
    if len(target_qids) != EXPECTED_F1F4_QUERIES:
        raise RuntimeError(f"F1-F4 fold population mismatch: {len(target_qids)}")
    if any(qid not in gold_all for qid in target_qids):
        raise RuntimeError("F1-F4 gold coverage mismatch")
    gold = {qid: gold_all[qid] for qid in target_qids}

    _candidate_rows, _metadata, docs_by_query = load_pv1_metadata()
    worklist_rows, worklist_by_key = load_pv1_worklist()
    if set(docs_by_query) != target_qids or set(worklist_by_key) != {(str(row["query_id"]), str(row["document_id"])) for row in worklist_rows}:
        raise RuntimeError("PV1 query/worklist universe mismatch")
    if set(worklist_by_key) != set(_metadata):
        raise RuntimeError("PV1 candidate/worklist identity mismatch")
    reconstructed, merge_info = load_and_merge_scores(worklist_by_key)
    reconstructed_path = OUT / "reconstructed_pv1_bge_scores.jsonl"
    write_jsonl(reconstructed_path, reconstructed)
    merge_info["reconstructed_output"] = rel(reconstructed_path)
    merge_info["reconstructed_output_sha256"] = sha256(reconstructed_path)
    write_json(OUT / "score_reconstruction.json", merge_info)

    score_by_key = {key_of(row): row for row in reconstructed}
    baseline: dict[str, list[str]] = {}
    for qid in sorted(docs_by_query, key=qkey):
        rows = [{**meta, **score_by_key[(qid, str(meta["document_id"]))]} for meta in docs_by_query[qid]]
        baseline[qid] = current_rrf_rank(rows)[:5]
    baseline_metric = metric(baseline, gold)
    expected_baseline = {"recall": 0.9255863095238096, "precision": 0.19710714285714284}
    baseline_reproduction = {
        "metrics": baseline_metric,
        "expected_metrics": expected_baseline,
        "recall_match": abs(float(baseline_metric["recall"]) - expected_baseline["recall"]) <= 1e-15,
        "precision_match": abs(float(baseline_metric["precision"]) - expected_baseline["precision"]) <= 1e-15,
        "policy": "RETRIEVAL_RRF_NO_LABEL",
    }
    if not baseline_reproduction["recall_match"] or not baseline_reproduction["precision_match"]:
        raise RuntimeError(f"current PV1 baseline reproduction mismatch: {baseline_reproduction}")
    write_json(OUT / "pv1_baseline_reproduction.json", baseline_reproduction)

    ft_v2_estimate = estimate_ft_v2_work(docs_by_query, score_by_key, baseline, folds, gold)
    write_json(OUT / "same_bge_hard_negative_ft_v2_estimate.json", ft_v2_estimate)

    feature_rows = build_feature_matrix(docs_by_query, score_by_key, folds)
    feature_path = OUT / "step4_k20_feature_matrix.jsonl"
    write_jsonl(feature_path, feature_rows)

    step4_provenance_pass = (
        historical["replay"]["gate"] == "PASS"
        and len(reconstructed) == EXPECTED_F1F4_QDOCS
        and merge_info["selected_chunk_mismatches"] == 0
        and all(validate_score_row(row, "RECONSTRUCTED") is None for row in reconstructed)
    )
    step4_result, step4_predictions, step4_traces = run_step4_port(
        docs_by_query,
        score_by_key,
        baseline,
        folds,
        gold,
        step4_provenance_pass,
    )
    step4_result.update({
        "policy": "STEP4_K20_PORT",
        "port_status": "K20_COMPATIBLE_PORT",
        "historical_shortlist_top_n": 50,
        "current_candidate_depth": 20,
        "exact_historical_reproduction": False,
        "provenance": {
            "historical_lineage": historical,
            "score_reconstruction": merge_info,
            "current_worklist_sha256": sha256(PV1_WORKLIST),
            "current_candidate_sha256": sha256(PV1_CANDIDATES),
            "fold0_used": False,
            "public_labels_used": False,
            "private_labels_used": False,
            "gpu_runs": 0,
            "modal_runs": 0,
            "model_loaded": False,
            "feature_matrix_sha256": sha256(feature_path),
        },
    })
    write_jsonl(OUT / "step4_k20_port_predictions.jsonl", [
        {"query_id": qid, "fold": folds[qid], "baseline_top5": baseline[qid], "step4_top5": step4_predictions[qid]}
        for qid in sorted(step4_predictions, key=qkey)
    ])
    write_jsonl(OUT / "step4_k20_port_swap_log.jsonl", [
        {"query_id": qid, "fold": folds[qid], "raw_swaps": trace["raw_swaps"], "accepted_swaps": trace["accepted_swaps"]}
        for qid, trace in sorted(step4_traces.items(), key=lambda item: qkey(item[0]))
        if trace["raw_swaps"]
    ])

    fallback: dict[str, Any] | None = None
    private_checkpoint: dict[str, Any] = {"status": "NOT_CREATED", "reason": "Step4 gate did not pass"}
    if step4_result["gate"] in {"STRONG_PASS", "WEAK_PASS"}:
        private_docs, private_score_by_key = build_private_meta_and_rows()
        private_baseline_payload = json.loads(PRIVATE_BASELINE.read_text(encoding="utf-8"))
        private_baseline = {str(qid): [str(value) for value in row["answer"]] for qid, row in private_baseline_payload.items()}
        if len(private_baseline) != 2080:
            raise RuntimeError("Private baseline query count mismatch")
        private_proposed: dict[str, list[str]] = {}
        private_traces: dict[str, dict[str, Any]] = {}
        for qid in sorted(private_docs, key=qkey):
            joined = [{**meta, **private_score_by_key[(qid, str(meta["document_id"]))]} for meta in private_docs[qid]]
            private_proposed[qid], private_traces[qid] = step4_query(joined, private_baseline[qid])
        private_checkpoint = write_private_checkpoint(private_proposed, {
            "validation_gate": step4_result["gate"],
            "validation_recall_delta": step4_result["delta"]["recall"],
            "private_queries_changed": sum(private_proposed[qid] != private_baseline[qid] for qid in private_proposed),
            "private_top1_top3_locked": all(trace["top1_top3_locked"] for trace in private_traces.values()),
            "private_labels_used": False,
        })
    else:
        fallback = run_direct_k20_fallback(docs_by_query, score_by_key, baseline, folds, gold)
        write_json(OUT / "fallback_direct_k20_top5.json", fallback)

    step4_result["private_checkpoint"] = private_checkpoint
    final_payload = {
        "status": "PHASE1_STEP4_RECOVERY_COMPLETE",
        "current_pv1_already_submitted": "UNKNOWN",
        "historical": historical,
        "pv1_step4_port_status": "K20_COMPATIBLE_PORT",
        "baseline_reproduction": baseline_reproduction,
        "step4_k20_port": step4_result,
        "fallback_direct_top5": fallback,
        "private_checkpoint": private_checkpoint,
        "hard_negative_ft_v2_ready": True,
        "hard_negative_ft_v2_spec": rel(OUT / "same_bge_hard_negative_ft_v2_spec.md"),
        "hard_negative_ft_v2_estimate": rel(OUT / "same_bge_hard_negative_ft_v2_estimate.json"),
        "input_hashes": rel(OUT / "input_hashes.json"),
        "outputs": {
            "historical_lineage": rel(OUT / "historical_step4_lineage.json"),
            "score_reconstruction": rel(OUT / "score_reconstruction.json"),
            "baseline_reproduction": rel(OUT / "pv1_baseline_reproduction.json"),
            "feature_matrix": rel(feature_path),
            "step4_predictions": rel(OUT / "step4_k20_port_predictions.jsonl"),
            "step4_swap_log": rel(OUT / "step4_k20_port_swap_log.jsonl"),
        },
        "execution": {
            "gpu_runs": 0,
            "modal_runs": 0,
            "model_loads": 0,
            "qwen_inference_runs": 0,
            "private_labels_used": False,
            "fold0_used": False,
            "public_labels_used": False,
            "historical_artifacts_mutated": False,
            "old_pv1_submission_overwritten": False,
        },
    }
    write_ft_v2_spec(merge_info, freeze, ft_v2_estimate)
    write_ledger(historical, step4_result, fallback, merge_info)
    final_payload["outputs"]["experiment_ledger"] = rel(OUT / "experiment_ledger.md")
    final_payload["outputs"]["ft_v2_spec"] = rel(OUT / "same_bge_hard_negative_ft_v2_spec.md")
    final_payload["outputs"]["ft_v2_estimate"] = rel(OUT / "same_bge_hard_negative_ft_v2_estimate.json")
    write_json(OUT / "phase1_result.json", final_payload)

    report_lines = [
        "# Phase 1 — Step4 recovery and current PV1 port",
        "",
        "CPU-only. No GPU, Modal, Qwen inference, Private answers, Fold0 labels, public labels, or old submission overwrite.",
        "- External submission status of the existing PV1 artifact: **UNKNOWN**; the local incumbent remains preserved.",
        "",
        "## Historical lineage",
        "",
        f"- Lineage status: **{historical['lineage_status']}**.",
        f"- Deployment evidence: **{historical['deployment_evidence']}** (artifact identity and swap log are verified; no external leaderboard receipt is in the repo).",
        f"- Historical calibration replay: **{historical['replay']['gate']}**; semantic equality with frozen 0.9411 JSON: `{historical['replay']['semantic_prediction_equality']}`.",
        f"- Historical replay counts: raw `{historical['replay']['counts']['raw_swap_log_rows']}`, calibrated swaps `{historical['replay']['counts']['calibrated_swaps_replayed']}`, calibrated queries `{historical['replay']['counts']['calibrated_queries_replayed']}`.",
        "- Historical candidate depth: Top-50. Current PV1 candidate depth: K20; therefore the current result is labeled `STEP4_K20_PORT`.",
        "",
        "## Current PV1 score reconstruction",
        "",
        f"- Reconstructed q-docs: `{merge_info['reconstructed_rows']}`.",
        f"- Source counts: `{merge_info['source_counts']}`.",
        f"- Selected-chunk mismatches: `{merge_info['selected_chunk_mismatches']}`.",
        f"- Reconstructed score SHA256: `{merge_info['reconstructed_output_sha256']}`.",
        "- PV1 Step4 support feature: `len(explicit source_ranks)` from dense/BM25/word-KNN; normalized by the historical denominator `4.0`.",
        f"- Baseline `RETRIEVAL_RRF_NO_LABEL`: Recall `{baseline_metric['recall']:.15f}`, Precision `{baseline_metric['precision']:.15f}`; expected reproduction: **PASS**.",
        "",
        "## STEP4_K20_PORT",
        "",
        f"- Gate: **{step4_result['gate']}**.",
        f"- Recall: `{step4_result['proposed']['recall']:.15f}`; Precision: `{step4_result['proposed']['precision']:.15f}`.",
        f"- Delta vs PV1 baseline: Recall `{step4_result['delta']['recall']:+.15f}`; Precision `{step4_result['delta']['precision']:+.15f}`.",
        f"- Fold deltas: `{step4_result['delta_by_fold']}`.",
        f"- Queries changed/improved/harmed/neutral: `{step4_result['queries_changed']}/{step4_result['queries_improved']}/{step4_result['queries_harmed']}/{step4_result['queries_neutral_changed']}`.",
        f"- Swaps raw/calibrated: `{step4_result['raw_swap_count']}/{step4_result['calibrated_swap_count']}`; calibrated rank distribution `{step4_result['calibrated_swaps_by_rank']}`.",
        f"- Top1–3 lock: **{'PASS' if step4_result['top1_top3_lock_pass'] else 'FAIL'}**.",
        "",
        "## Private and fallback",
        "",
        f"- Private Step4 checkpoint: **{private_checkpoint.get('status')}**.",
        f"- Fallback `DIRECT_K20_PLUS_ANCHORS_DOCUMENT_TOP5`: **{'NOT_RUN' if fallback is None else ('PASS' if fallback.get('fallback_gate') else 'FAIL')}**.",
        f"- SAME-BGE hard-negative FT V2 spec: `{rel(OUT / 'same_bge_hard_negative_ft_v2_spec.md')}`; training: **NOT RUN**.",
        f"- FT V2 CPU work-size estimate: `{rel(OUT / 'same_bge_hard_negative_ft_v2_estimate.json')}`.",
        "",
        "## Frozen artifacts",
        "",
        f"- Input SHA ledger: `{rel(OUT / 'input_hashes.json')}`.",
        f"- Experiment ledger: `{rel(OUT / 'experiment_ledger.md')}`.",
        "",
    ]
    (OUT / "phase1_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({
        "status": final_payload["status"],
        "step4_gate": step4_result["gate"],
        "step4_recall": step4_result["proposed"]["recall"],
        "step4_delta": step4_result["delta"]["recall"],
        "private_checkpoint": private_checkpoint.get("status"),
        "fallback_gate": None if fallback is None else fallback.get("fallback_gate"),
        "gpu_runs": 0,
        "modal_runs": 0,
        "outputs": final_payload["outputs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
