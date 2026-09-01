"""CPU-only Step 1A recall-gap audit with structural target-only readers.

Aggregate JSONL inputs may physically contain Fold0.  This script reads only a
top-level query_id metadata field for every row; it fully decodes a row only
after the ID is known to be in folds 1-4.  No non-target payload is decoded,
stored, counted in statistics, or used by the oracle.
"""
from __future__ import annotations

import json
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
TARGET_FOLDS = {1, 2, 3, 4}
STEP0 = ROOT / "reports/task1/step0_metric_contract_report.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
CANDIDATES = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
REPORT = ROOT / "reports/task1/exp_1a_recall_gap_report.json"


def skip_ws(raw: str, pos: int) -> int:
    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    return pos


def scan_string(raw: str, pos: int) -> tuple[str, int]:
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(raw, pos)
    if not isinstance(value, str):
        raise ValueError("JSON object key must be a string")
    return value, end


def skip_json_value(raw: str, pos: int) -> int:
    """Return end index without constructing the JSON value."""
    pos = skip_ws(raw, pos)
    if pos >= len(raw):
        raise ValueError("missing JSON value")
    first = raw[pos]
    if first == '"':
        _, end = json.JSONDecoder().raw_decode(raw, pos)
        return end
    if first not in "[{":
        while pos < len(raw) and raw[pos] not in ",]}":
            pos += 1
        return pos
    closing = "}" if first == "{" else "]"
    stack = [closing]
    pos += 1
    in_string = False
    escaped = False
    while pos < len(raw) and stack:
        char = raw[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char == stack[-1]:
            stack.pop()
        pos += 1
    if stack:
        raise ValueError("unterminated JSON container")
    return pos


def top_level_string_field(raw: str, wanted: str) -> str:
    """Extract one top-level string field while structurally skipping values."""
    pos = skip_ws(raw, 0)
    if pos >= len(raw) or raw[pos] != "{":
        raise ValueError("JSONL row must be an object")
    pos = skip_ws(raw, pos + 1)
    while pos < len(raw) and raw[pos] != "}":
        key, pos = scan_string(raw, pos)
        pos = skip_ws(raw, pos)
        if pos >= len(raw) or raw[pos] != ":":
            raise ValueError("invalid JSON object")
        pos = skip_ws(raw, pos + 1)
        if key == wanted:
            value, end = json.JSONDecoder().raw_decode(raw, pos)
            if not isinstance(value, str):
                raise ValueError(f"{wanted} must be a string")
            return value
        pos = skip_ws(raw, skip_json_value(raw, pos))
        if pos < len(raw) and raw[pos] == ",":
            pos = skip_ws(raw, pos + 1)
        elif pos < len(raw) and raw[pos] != "}":
            raise ValueError("invalid JSON object separator")
    raise ValueError(f"top-level field missing: {wanted}")


def read_target_fold_map() -> dict[str, int]:
    """Decode fold metadata only; never decode Fold0 validation_ids."""
    helper_path = ROOT / "reports/task1/step0_metric_contract_audit.py"
    spec = importlib.util.spec_from_file_location("step0_structural_reader", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load approved Step 0 structural reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.read_target_fold_map(FOLDS)
    if len(result) != 5600 or set(result.values()) != TARGET_FOLDS:
        raise ValueError("target whitelist is not exactly 5,600 folds-1-4 IDs")
    return result


def read_target_train(target: set[str]) -> dict[str, set[str]]:
    """Parse only values for target object keys from the train JSON mapping."""
    raw = TRAIN.read_text(encoding="utf-8-sig")
    pos = skip_ws(raw, 0)
    if raw[pos] != "{":
        raise ValueError("train root must be object")
    pos = skip_ws(raw, pos + 1)
    gold: dict[str, set[str]] = {}
    decoder = json.JSONDecoder()
    while pos < len(raw) and raw[pos] != "}":
        qid, pos = scan_string(raw, pos)
        pos = skip_ws(raw, pos)
        if raw[pos] != ":":
            raise ValueError("invalid train object")
        pos = skip_ws(raw, pos + 1)
        if qid in target:
            record, pos = decoder.raw_decode(raw, pos)
            answer = record.get("answer") if isinstance(record, dict) else None
            if not isinstance(answer, list) or not answer:
                raise ValueError(f"invalid gold record: {qid}")
            docs = [str(doc) for doc in answer]
            if len(docs) != len(set(docs)):
                raise ValueError(f"duplicate gold IDs inconsistent with Step 0: {qid}")
            gold[qid] = set(docs)
        else:
            pos = skip_json_value(raw, pos)
        pos = skip_ws(raw, pos)
        if pos < len(raw) and raw[pos] == ",":
            pos = skip_ws(raw, pos + 1)
    if set(gold) != target:
        raise ValueError("target gold coverage differs from whitelist")
    return gold


def stream_target_rows(path: Path, target: set[str]) -> tuple[Iterator[dict[str, Any]], list[int]]:
    skipped = [0]
    def iterator() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                query_id = top_level_string_field(raw, "query_id")
                if query_id not in target:
                    skipped[0] += 1
                    continue
                row = json.loads(raw)
                if str(row.get("query_id")) != query_id:
                    raise ValueError(f"query_id metadata mismatch at {path}:{line_number}")
                yield row
    return iterator(), skipped


def read_baseline(target: set[str], folds: dict[str, int]) -> tuple[dict[str, list[str]], int]:
    rows, skipped = stream_target_rows(BASELINE, target)
    result: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row["query_id"])
        if int(row.get("fold", -1)) != folds[qid]:
            raise ValueError(f"baseline fold mismatch: {qid}")
        top5 = row.get("top5")
        if not isinstance(top5, list):
            raise ValueError(f"baseline top5 missing: {qid}")
        docs = [str(doc) for doc in top5]
        if not 1 <= len(docs) <= 5 or len(docs) != len(set(docs)):
            raise ValueError(f"invalid baseline top5: {qid}")
        if qid in result:
            raise ValueError(f"duplicate baseline query: {qid}")
        result[qid] = docs
    if set(result) != target:
        raise ValueError("baseline target coverage differs from whitelist")
    return result, skipped[0]


def read_candidates(target: set[str], folds: dict[str, int]) -> tuple[dict[str, set[str]], int]:
    rows, skipped = stream_target_rows(CANDIDATES, target)
    result: dict[str, set[str]] = defaultdict(set)
    target_seen: set[str] = set()
    for row in rows:
        qid = str(row["query_id"])
        if int(row.get("fold", -1)) != folds[qid]:
            raise ValueError(f"candidate fold mismatch: {qid}")
        target_seen.add(qid)
        if isinstance(row.get("candidates"), list):
            docs = [str(item["doc_id"]) for item in row["candidates"]]
        elif "doc_id" in row:
            docs = [str(row["doc_id"])]
        else:
            raise ValueError(f"candidate schema has neither candidates nor doc_id: {qid}")
        result[qid].update(docs)
    if target_seen != target:
        raise ValueError("candidate target coverage differs from whitelist")
    return dict(result), skipped[0]


def recall(gold: set[str], prediction: list[str] | set[str]) -> float:
    docs = set(prediction)
    if not 1 <= len(docs) <= 5:
        raise ValueError("oracle emitted invalid prediction length")
    return len(gold & docs) / len(gold)


def main() -> None:
    step0 = json.loads(STEP0.read_text(encoding="utf-8"))
    required = (step0.get("status") == "PASS" and step0.get("official_primary_metric") == "macro_recall"
                and step0.get("metric_semantics_consistent_on_v3_domain") is True
                and step0.get("total_queries") == 5600 and not step0.get("fold0_touched")
                and not step0.get("public_labels_used"))
    if not required:
        raise RuntimeError("Step 0 contract is not approved")
    folds = read_target_fold_map()
    target = set(folds)
    gold = read_target_train(target)
    baseline, baseline_skipped = read_baseline(target, folds)
    candidates, candidate_skipped = read_candidates(target, folds)

    print("STEP 1A CORRECTED PREFLIGHT")
    print(f"Step0 report: {STEP0.relative_to(ROOT)}\nStep0 status: PASS")
    print(f"Gold source: {TRAIN.relative_to(ROOT)}\nFold source: {FOLDS.relative_to(ROOT)}")
    print(f"Baseline source: {BASELINE.relative_to(ROOT)}\nFull candidate source: {CANDIDATES.relative_to(ROOT)}")
    print("Target folds: [1,2,3,4]\nTarget query whitelist count: 5600")
    print(f"Aggregate baseline contains non-target records: {baseline_skipped > 0}")
    print(f"Aggregate candidate pool contains non-target records: {candidate_skipped > 0}")
    print("Reader mode: STRUCTURAL TARGET-ONLY MATERIALIZATION")
    print(f"Baseline target query count: {len(baseline)}\nCandidate target query count: {len(candidates)}")
    print("Fold0 payload materialized: NO\nFold0 used in statistics: NO\nFold0 labels read: NO\nPublic labels read: NO\nGPU required: NO")
    print(f"Output: {REPORT.relative_to(ROOT)}")

    per_fold: dict[str, dict[str, Any]] = {str(fold): {"query_count": 0, "queries_below_capacity_ceiling": 0,
        "missing_gold_total": 0, "missing_gold_found_in_pool_total": 0, "queries_with_any_missing_gold_in_full_pool": 0,
        "queries_with_all_missing_gold_in_full_pool": 0, "queries_with_positive_one_swap_gain": 0,
        "sum_gain": 0.0, "macro_gain": 0.0, "sum_unconstrained_gain": 0.0, "unconstrained_macro_gain": 0.0} for fold in TARGET_FOLDS}
    below = missing_total = found_total = any_found = all_found = positive = 0
    sum_one = sum_unconstrained = 0.0
    samples: list[dict[str, Any]] = []
    one_nonnegative = unconstrained_ge = range_ok = True
    for qid in sorted(target, key=lambda value: int(value) if value.isdigit() else value):
        fold = folds[qid]; stats = per_fold[str(fold)]; stats["query_count"] += 1
        base = baseline[qid]; base_set = set(base); pool = candidates[qid]
        base_recall = recall(gold[qid], base)
        missing = gold[qid] - base_set
        found = missing & pool
        if base_recall < 1.0:
            below += 1; stats["queries_below_capacity_ceiling"] += 1
        missing_total += len(missing); stats["missing_gold_total"] += len(missing)
        found_total += len(found); stats["missing_gold_found_in_pool_total"] += len(found)
        if found:
            any_found += 1; stats["queries_with_any_missing_gold_in_full_pool"] += 1
        if missing and found == missing:
            all_found += 1; stats["queries_with_all_missing_gold_in_full_pool"] += 1
        best_one = 0.0
        for incoming in pool - base_set:
            for index in range(len(base)):
                swapped = base[:index] + [incoming] + base[index + 1:]
                if len(swapped) == len(set(swapped)):
                    best_one = max(best_one, recall(gold[qid], swapped) - base_recall)
        universe = base_set | pool
        best_unconstrained = min(len(gold[qid] & universe), 5) / len(gold[qid])
        unconstrained_gain = best_unconstrained - base_recall
        one_nonnegative &= best_one >= -1e-12
        unconstrained_ge &= unconstrained_gain + 1e-12 >= best_one
        range_ok &= all(-1e-12 <= value <= 1.0 + 1e-12 for value in (base_recall, best_unconstrained))
        if best_one > 0:
            positive += 1; stats["queries_with_positive_one_swap_gain"] += 1
            if len(samples) < 20:
                samples.append({"query_id": qid, "fold": fold, "relevant_count": len(gold[qid]), "baseline_recall": base_recall,
                    "missing_gold_count": len(missing), "missing_gold_found_in_pool_count": len(found),
                    "best_one_swap_gain": best_one, "best_unconstrained_gain": unconstrained_gain})
        sum_one += best_one; stats["sum_gain"] += best_one
        sum_unconstrained += unconstrained_gain; stats["sum_unconstrained_gain"] += unconstrained_gain
    for value in per_fold.values():
        value["macro_gain"] = value["sum_gain"] / value["query_count"]
        value["unconstrained_macro_gain"] = value["sum_unconstrained_gain"] / value["query_count"]
    one_gain = sum_one / 5600
    unc_gain = sum_unconstrained / 5600
    checks = {"all_queries_covered": set(gold) == set(baseline) == set(candidates) == target and len(target) == 5600,
              "one_swap_nonnegative": one_nonnegative, "unconstrained_ge_one_swap": unconstrained_ge,
              "recall_in_valid_range": range_ok, "fold0_payload_not_materialized": True}
    status = "CONTRACT_ERROR" if not all(checks.values()) else ("PASS" if one_gain > .003 else "FAIL" if one_gain < .001 else "INCONCLUSIVE")
    report = {"status": status, "official_metric": "macro_recall", "folds_used": [1,2,3,4], "fold0_touched": False,
        "public_labels_used": False, "aggregate_artifacts_contain_fold0": True, "fold0_payload_materialized": False,
        "fold0_used_in_statistics": False, "fold0_used_in_oracle": False, "fold0_labels_used": False,
        "baseline_non_target_records_skipped": baseline_skipped, "candidate_non_target_records_skipped": candidate_skipped,
        "total_queries": 5600, "audited_queries": 5600, "input_artifacts": {"step0_report": str(STEP0.relative_to(ROOT)), "train_labels": str(TRAIN.relative_to(ROOT)), "fold_mapping": str(FOLDS.relative_to(ROOT)), "baseline_top5": str(BASELINE.relative_to(ROOT)), "full_candidate_pool": str(CANDIDATES.relative_to(ROOT))},
        "input_provenance": {"baseline_trace": "scripts/beam/archive/build_public093_strict_oof.py → baseline_093_oof/predictions.jsonl → scripts/beam/beam_task1_v3_prepare_cpu.py → scripts/beam/task1_v3_residual/build_shortlist_features.py", "candidate_pool_trace": "scripts/beam/materialize_inference_matched_reranker_compact_direct.py → candidate_refs_full.jsonl → scripts/beam/beam_task1_v3_prepare_cpu.py → scripts/beam/task1_v3_residual/build_shortlist_features.py; shortlist is applied only afterwards."},
        "queries_below_capacity_ceiling": below, "missing_gold_total": missing_total, "missing_gold_found_in_pool_total": found_total,
        "queries_with_any_missing_gold_in_full_pool": any_found, "queries_with_all_missing_gold_in_full_pool": all_found,
        "queries_with_positive_one_swap_gain": positive, "sum_one_swap_gain": sum_one, "one_swap_candidate_ceiling_gain": one_gain,
        "sum_unconstrained_gain": sum_unconstrained, "unconstrained_top5_repair_ceiling_gain": unc_gain,
        "per_fold": per_fold, "sanity_checks": checks, "sample_query_ids": samples,
        "notes": ["Aggregate JSONL rows were scanned for top-level query_id metadata before payload decoding.", "Only target-fold payloads were decoded and materialized.", "The unconstrained ceiling is forensic only and is not the Step 1A gate."]}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
