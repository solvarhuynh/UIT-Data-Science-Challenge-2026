"""Workflow C0-A: historical-policy-independent oracle / opportunity anatomy.

Lifecycle: VERSIONED_NEW_FILE / KEEP_REPRODUCIBILITY.  This is the one
canonical implementation for the C0-A report trio.  It is intentionally
CPU-only: it does not load a model, replay a policy, create predictions, or
read a historical V3A selected-action artifact.

The reader order is deliberate.  It freezes folds, baseline/action/candidate
identities, and their joins before it decodes any F1--F4 gold payload.  For
aggregate JSONL inputs it first reads only top-level ``query_id`` metadata;
Fold0 rows are structurally skipped without payload JSON decoding.  The train
mapping is streamed in the same way, so Fold0 gold answers are skipped rather
than decoded or materialized.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[3]
TARGET_FOLDS = (1, 2, 3, 4)
TARGET_FOLD_SET = set(TARGET_FOLDS)
EXPECTED_QUERY_COUNT = 5600
TOLERANCE = 1e-12
EPSILON = 1e-15

FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
BASELINE_REPORT = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/report.json"
ACTIONS = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl"
CANDIDATES = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
STEP0_REPORT = ROOT / "reports/task1/workflow_a/step0_metric_contract_report.json"
FULL_REPORT = ROOT / "reports/task1/workflow_a/exp_1a_recall_gap_report.json"
K20_REPORT = ROOT / "reports/task1/workflow_a/exp_1b_oracle_gap_decomposition_report.json"
K77_REPORT = ROOT / "reports/task1/workflow_a/exp_1b_rerun_k77_oracle_gap_decomposition_report.json"

OUT_DIR = ROOT / "reports/task1/workflow_c/tv2/c0a"
MANIFEST_OUT = OUT_DIR / "c0a_integrity_manifest.json"
REPORT_OUT = OUT_DIR / "c0a_oracle_opportunity_report.json"
TRACE_OUT = OUT_DIR / "c0a_oracle_opportunity_trace.jsonl"

EXPECTED_SHA256 = {
    FOLDS: "acc4792f1b067d9c58cbfa16789c4c0bd47b71cad081443fbb37eb6017803a0a",
    TRAIN: "c39cde9e74977e350f1456e7d487aafe67d2bcbaa4fa26fcabd557fe635635b7",
    BASELINE: "1272cb9e8b465f433c725674081084f084a60cdb9807d6ca6a3aa3d9acc1e7d5",
    ACTIONS: "7b7acbdd5c26c1e7b21cba42c84900f7894b8fc3dff1475941534345a140eb3a",
    CANDIDATES: "e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6",
}
EXPECTED_ORACLE_RECALL = {
    "K20": 0.9676994047619048,
    "K77": 0.9847142857142857,
    "FULL": 0.9897440476190476,
}
EXPECTED_BASELINE_RECALL = 0.9259285714285714
EXPECTED_K77_INCOMING_COUNT = 403322


def rel(path: Path) -> str:
    """Return repository-relative paths in a platform-neutral form."""
    return path.resolve().relative_to(ROOT).as_posix()


def query_sort_key(query_id: str) -> tuple[int, int | str]:
    return (0, int(query_id)) if query_id.isdigit() else (1, query_id)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_set_hash(values: Iterable[str]) -> str:
    joined = "\n".join(sorted(values, key=query_sort_key)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} cannot be boolean")
    return int(value)


def as_finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return result


def error_sample(errors: list[str], message: str, limit: int = 25) -> None:
    if len(errors) < limit:
        errors.append(message)


class CharStream:
    """Small streaming JSON reader used to skip non-target train payloads."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("r", encoding="utf-8-sig")
        self.buffer = ""

    def get(self) -> str:
        if not self.buffer:
            self.buffer = self.handle.read(8192)
            if not self.buffer:
                return ""
        result = self.buffer[0]
        self.buffer = self.buffer[1:]
        return result

    def unget(self, value: str) -> None:
        if value:
            self.buffer = value + self.buffer

    def close(self) -> None:
        self.handle.close()


def stream_skip_ws(stream: CharStream) -> str:
    char = stream.get()
    while char and char.isspace():
        char = stream.get()
    return char


def stream_skip_string(stream: CharStream, first: str) -> None:
    if first != '"':
        raise ValueError("expected JSON string")
    escaped = False
    while True:
        char = stream.get()
        if not char:
            raise ValueError("unterminated JSON string")
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return


def stream_read_string(stream: CharStream, first: str) -> str:
    if first != '"':
        raise ValueError("expected JSON string")
    raw = [first]
    escaped = False
    while True:
        char = stream.get()
        if not char:
            raise ValueError("unterminated JSON string")
        raw.append(char)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return json.loads("".join(raw))


def stream_skip_value(stream: CharStream, first: str | None = None) -> None:
    """Advance over a JSON value without constructing that value."""
    char = stream_skip_ws(stream) if first is None else first
    if not char:
        raise ValueError("missing JSON value")
    if char == '"':
        stream_skip_string(stream, char)
        return
    if char == "{":
        child = stream_skip_ws(stream)
        if child == "}":
            return
        while True:
            stream_skip_string(stream, child)
            if stream_skip_ws(stream) != ":":
                raise ValueError("invalid JSON object")
            stream_skip_value(stream)
            child = stream_skip_ws(stream)
            if child == "}":
                return
            if child != ",":
                raise ValueError("invalid JSON object separator")
            child = stream_skip_ws(stream)
    elif char == "[":
        child = stream_skip_ws(stream)
        if child == "]":
            return
        while True:
            stream_skip_value(stream, child)
            child = stream_skip_ws(stream)
            if child == "]":
                return
            if child != ",":
                raise ValueError("invalid JSON array separator")
            child = stream_skip_ws(stream)
    else:
        while char and char not in ",]}":
            char = stream.get()
        # Delimiters belong to the caller that is parsing the parent object.
        stream.unget(char)


def stream_read_value(stream: CharStream, first: str | None = None) -> Any:
    """Decode one explicitly authorized selected JSON value."""
    char = stream_skip_ws(stream) if first is None else first
    if char == '"':
        return stream_read_string(stream, char)
    if char == "[":
        result: list[Any] = []
        child = stream_skip_ws(stream)
        if child == "]":
            return result
        while True:
            result.append(stream_read_value(stream, child))
            child = stream_skip_ws(stream)
            if child == "]":
                return result
            if child != ",":
                raise ValueError("invalid selected array")
            child = stream_skip_ws(stream)
    if char == "{":
        result_obj: dict[str, Any] = {}
        child = stream_skip_ws(stream)
        if child == "}":
            return result_obj
        while True:
            name = stream_read_string(stream, child)
            if stream_skip_ws(stream) != ":":
                raise ValueError("invalid selected object")
            result_obj[name] = stream_read_value(stream)
            child = stream_skip_ws(stream)
            if child == "}":
                return result_obj
            if child != ",":
                raise ValueError("invalid selected object separator")
            child = stream_skip_ws(stream)
    raw = [char]
    child = stream.get()
    while child and child not in ",]}":
        raw.append(child)
        child = stream.get()
    stream.unget(child)
    return json.loads("".join(raw))


def raw_skip_ws(raw: str, pos: int) -> int:
    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    return pos


def raw_string_end(raw: str, pos: int) -> int:
    if pos >= len(raw) or raw[pos] != '"':
        raise ValueError("expected JSON string")
    pos += 1
    escaped = False
    while pos < len(raw):
        char = raw[pos]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return pos + 1
        pos += 1
    raise ValueError("unterminated JSON string")


def raw_read_string(raw: str, pos: int) -> tuple[str, int]:
    end = raw_string_end(raw, pos)
    value = json.loads(raw[pos:end])
    if not isinstance(value, str):
        raise ValueError("JSON object key must be a string")
    return value, end


def raw_skip_value(raw: str, pos: int) -> int:
    """Skip a JSON value without decoding any strings, arrays, or objects."""
    pos = raw_skip_ws(raw, pos)
    if pos >= len(raw):
        raise ValueError("missing JSON value")
    first = raw[pos]
    if first == '"':
        return raw_string_end(raw, pos)
    if first not in "[{":
        while pos < len(raw) and raw[pos] not in ",]}":
            pos += 1
        return pos
    stack = ["}" if first == "{" else "]"]
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


def raw_top_level_values(raw: str, selected: set[str]) -> tuple[dict[str, Any], set[str]]:
    """Decode only listed top-level fields; all other values are skipped raw."""
    pos = raw_skip_ws(raw, 0)
    if pos >= len(raw) or raw[pos] != "{":
        raise ValueError("JSONL row must be an object")
    pos = raw_skip_ws(raw, pos + 1)
    decoder = json.JSONDecoder()
    values: dict[str, Any] = {}
    keys: set[str] = set()
    while pos < len(raw) and raw[pos] != "}":
        key, pos = raw_read_string(raw, pos)
        keys.add(key)
        pos = raw_skip_ws(raw, pos)
        if pos >= len(raw) or raw[pos] != ":":
            raise ValueError("invalid JSON object")
        pos = raw_skip_ws(raw, pos + 1)
        if key in selected:
            value, pos = decoder.raw_decode(raw, pos)
            values[key] = value
        else:
            pos = raw_skip_value(raw, pos)
        pos = raw_skip_ws(raw, pos)
        if pos < len(raw) and raw[pos] == ",":
            pos = raw_skip_ws(raw, pos + 1)
        elif pos < len(raw) and raw[pos] != "}":
            raise ValueError("invalid JSON object separator")
    return values, keys


def raw_top_level_string(raw: str, wanted: str) -> str:
    values, _ = raw_top_level_values(raw, {wanted})
    value = values.get(wanted)
    if not isinstance(value, str):
        raise ValueError(f"missing/non-string top-level {wanted}")
    return value


def read_target_fold_map(path: Path) -> tuple[dict[str, int], dict[str, int], int]:
    """Construct the F1--F4 whitelist without materializing Fold0 IDs."""
    stream = CharStream(path)
    target: dict[str, int] = {}
    fold_counts: Counter[str] = Counter()
    duplicate_count = 0
    try:
        if stream_skip_ws(stream) != "{":
            raise ValueError("fold mapping root must be an object")
        root_field = stream_skip_ws(stream)
        found_folds = False
        while root_field != "}":
            name = stream_read_string(stream, root_field)
            if stream_skip_ws(stream) != ":":
                raise ValueError("invalid fold mapping")
            value_start = stream_skip_ws(stream)
            if name != "folds":
                stream_skip_value(stream, value_start)
            else:
                found_folds = True
                if value_start != "[":
                    raise ValueError("folds must be an array")
                record_start = stream_skip_ws(stream)
                while record_start != "]":
                    if record_start != "{":
                        raise ValueError("fold record must be an object")
                    fold: int | None = None
                    ids: list[str] | None = None
                    field = stream_skip_ws(stream)
                    while field != "}":
                        field_name = stream_read_string(stream, field)
                        if stream_skip_ws(stream) != ":":
                            raise ValueError("invalid fold record")
                        field_value = stream_skip_ws(stream)
                        if field_name == "fold":
                            fold = as_int(stream_read_value(stream, field_value), field="fold")
                        elif field_name == "validation_ids":
                            if fold is None:
                                raise ValueError("canonical fold record places validation_ids before fold")
                            if fold in TARGET_FOLD_SET:
                                raw_ids = stream_read_value(stream, field_value)
                                if not isinstance(raw_ids, list):
                                    raise ValueError("validation_ids must be a list")
                                ids = [str(value) for value in raw_ids]
                            else:
                                stream_skip_value(stream, field_value)
                        else:
                            stream_skip_value(stream, field_value)
                        field = stream_skip_ws(stream)
                        if field == ",":
                            field = stream_skip_ws(stream)
                        elif field != "}":
                            raise ValueError("invalid fold record separator")
                    if fold in TARGET_FOLD_SET:
                        if ids is None:
                            raise ValueError(f"target fold {fold} lacks validation_ids")
                        fold_counts[str(fold)] = len(ids)
                        for query_id in ids:
                            if query_id in target:
                                duplicate_count += 1
                            target[query_id] = fold
                    record_start = stream_skip_ws(stream)
                    if record_start == ",":
                        record_start = stream_skip_ws(stream)
                    elif record_start != "]":
                        raise ValueError("invalid folds array separator")
            root_field = stream_skip_ws(stream)
            if root_field == ",":
                root_field = stream_skip_ws(stream)
            elif root_field != "}":
                raise ValueError("invalid fold mapping separator")
        if not found_folds:
            raise ValueError("fold mapping has no folds key")
    finally:
        stream.close()
    return target, dict(sorted(fold_counts.items())), duplicate_count


def read_target_gold(path: Path, target: set[str]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Decode F1--F4 gold answers only after the structural phase succeeds."""
    stream = CharStream(path)
    gold: dict[str, set[str]] = {}
    schema_keys: set[str] = set()
    skipped_non_target = 0
    duplicate_gold_ids = 0
    try:
        if stream_skip_ws(stream) != "{":
            raise ValueError("train root must be a query-id mapping")
        query_start = stream_skip_ws(stream)
        while query_start != "}":
            query_id = stream_read_string(stream, query_start)
            if stream_skip_ws(stream) != ":":
                raise ValueError("invalid train mapping")
            value_start = stream_skip_ws(stream)
            if query_id not in target:
                # Fold0 answer payload is deliberately skipped, not decoded.
                skipped_non_target += 1
                stream_skip_value(stream, value_start)
            else:
                record = stream_read_value(stream, value_start)
                if not isinstance(record, dict):
                    raise ValueError(f"target train record is not an object: {query_id}")
                schema_keys.update(record)
                answer = record.get("answer")
                if not isinstance(answer, list) or not answer:
                    raise ValueError(f"invalid target gold answer: {query_id}")
                docs = [str(value) for value in answer]
                if len(docs) != len(set(docs)):
                    duplicate_gold_ids += 1
                    raise ValueError(f"duplicate target gold IDs: {query_id}")
                if query_id in gold:
                    raise ValueError(f"duplicate target gold query: {query_id}")
                gold[query_id] = set(docs)
            query_start = stream_skip_ws(stream)
            if query_start == ",":
                query_start = stream_skip_ws(stream)
            elif query_start != "}":
                raise ValueError("invalid train mapping separator")
    finally:
        stream.close()
    return gold, {
        "target_query_count": len(gold),
        "non_target_gold_records_structurally_skipped": skipped_non_target,
        "fold0_gold_payload_decoded_count": 0,
        "duplicate_gold_id_query_count": duplicate_gold_ids,
        "target_record_keys": sorted(schema_keys),
    }


def stream_target_jsonl(
    path: Path, target: set[str], selected_fields: set[str]
) -> tuple[Iterator[tuple[int, str, dict[str, Any], set[str]]], dict[str, Any]]:
    """Yield target rows with only selected fields decoded.

    All rows first receive a top-level ``query_id`` read.  Non-target rows,
    including Fold0, are then discarded before their payload is decoded.
    """
    stats: dict[str, Any] = {
        "nonempty_row_count": 0,
        "target_row_count": 0,
        "non_target_rows_structurally_skipped": 0,
        "target_top_level_keys": set(),
    }

    def iterator() -> Iterator[tuple[int, str, dict[str, Any], set[str]]]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                stats["nonempty_row_count"] += 1
                query_id = raw_top_level_string(raw, "query_id")
                if query_id not in target:
                    stats["non_target_rows_structurally_skipped"] += 1
                    continue
                values, keys = raw_top_level_values(raw, selected_fields | {"query_id"})
                if values.get("query_id") != query_id:
                    raise ValueError(f"query_id metadata mismatch at {rel(path)}:{line_number}")
                stats["target_row_count"] += 1
                stats["target_top_level_keys"].update(keys)
                yield line_number, query_id, values, keys

    return iterator(), stats


def read_baseline_structural(
    target: set[str], fold_by_query: dict[str, int]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    rows, stats = stream_target_jsonl(BASELINE, target, {"fold", "top5"})
    baseline: dict[str, list[str]] = {}
    errors: list[str] = []
    for line, query_id, row, _ in rows:
        try:
            if as_int(row.get("fold"), field="baseline.fold") != fold_by_query[query_id]:
                raise ValueError("fold mismatch")
            top5 = row.get("top5")
            if not isinstance(top5, list):
                raise ValueError("top5 missing")
            docs = [str(value) for value in top5]
            if len(docs) != 5 or len(docs) != len(set(docs)):
                raise ValueError("top5 must contain exactly five unique document IDs")
            if query_id in baseline:
                raise ValueError("duplicate target baseline query")
            baseline[query_id] = docs
        except (TypeError, ValueError) as exc:
            error_sample(errors, f"{rel(BASELINE)}:{line}:{query_id}:{exc}")
    stats["target_top_level_keys"] = sorted(stats["target_top_level_keys"])
    stats["unique_target_query_count"] = len(baseline)
    stats["duplicate_target_query_count"] = max(0, stats["target_row_count"] - len(baseline))
    stats["missing_target_query_count"] = len(target - set(baseline))
    stats["errors"] = errors
    return baseline, stats


def read_actions_structural(
    target: set[str], fold_by_query: dict[str, int], baseline: dict[str, list[str]]
) -> tuple[dict[str, list[tuple[str, int, str]]], dict[str, Any]]:
    selected = {"fold", "incoming_doc_id", "dropped_doc_id", "drop_rank", "baseline_top5", "features"}
    rows, stats = stream_target_jsonl(ACTIONS, target, selected)
    actions_by_query: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    action_keys: set[tuple[str, str, int, str]] = set()
    errors: list[str] = []
    feature_sizes: Counter[int] = Counter()
    for line, query_id, row, _ in rows:
        try:
            if as_int(row.get("fold"), field="action.fold") != fold_by_query[query_id]:
                raise ValueError("fold mismatch")
            incoming = str(row["incoming_doc_id"])
            dropped = str(row["dropped_doc_id"])
            drop_rank = as_int(row.get("drop_rank"), field="action.drop_rank")
            action_baseline = row.get("baseline_top5")
            if not isinstance(action_baseline, list) or [str(value) for value in action_baseline] != baseline[query_id]:
                raise ValueError("baseline_top5 identity mismatch")
            if drop_rank not in (4, 5):
                raise ValueError("drop rank must be 4 or 5")
            if dropped != baseline[query_id][drop_rank - 1]:
                raise ValueError("dropped document does not match baseline drop rank")
            if incoming in baseline[query_id]:
                raise ValueError("incoming document duplicates baseline top5")
            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError("features missing")
            feature_sizes[len(features)] += 1
            if len(features) != 58:
                raise ValueError("feature schema is not 58 columns")
            identity = (query_id, incoming, drop_rank, dropped)
            if identity in action_keys:
                raise ValueError("duplicate action identity")
            action_keys.add(identity)
            actions_by_query[query_id].append((incoming, drop_rank, dropped))
        except (KeyError, TypeError, ValueError) as exc:
            error_sample(errors, f"{rel(ACTIONS)}:{line}:{query_id}:{exc}")
    stats["target_top_level_keys"] = sorted(stats["target_top_level_keys"])
    stats["unique_target_query_count"] = len(actions_by_query)
    stats["missing_target_query_count"] = len(target - set(actions_by_query))
    stats["target_action_count"] = len(action_keys)
    stats["duplicate_action_identity_count"] = stats["target_row_count"] - len(action_keys)
    stats["feature_size_distribution"] = {str(key): feature_sizes[key] for key in sorted(feature_sizes)}
    stats["errors"] = errors
    return dict(actions_by_query), stats


def candidate_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize either supported full-pool JSONL encoding without inference."""
    if "candidates" in row and row["candidates"] is not None:
        values = row["candidates"]
        if not isinstance(values, list):
            raise ValueError("candidates must be a list")
        return [dict(value) for value in values if isinstance(value, dict)]
    if "doc_id" not in row:
        raise ValueError("candidate row contains neither candidates nor doc_id")
    return [dict(row)]


def close_candidate_group(
    query_id: str,
    docs: set[str],
    ranks: set[int],
    target: set[str],
    baseline: dict[str, list[str]],
    actions_by_query: dict[str, list[tuple[str, int, str]]],
    state: dict[str, Any],
) -> None:
    if not query_id:
        return
    state["closed_queries"].add(query_id)
    state["query_document_counts"][query_id] = len(docs)
    state["full_incoming_count"] += len(set(docs) - set(baseline[query_id]))
    state["k77_incoming_count"] += sum(
        1 for doc, rank in state["current_doc_ranks"].items() if rank <= 77 and doc not in baseline[query_id]
    )
    required = {incoming for incoming, _, _ in actions_by_query.get(query_id, [])}
    missing_k20_docs = required - docs
    if missing_k20_docs:
        error_sample(
            state["errors"],
            f"{rel(CANDIDATES)}:{query_id}:K20 incoming docs absent from canonical full pool ({len(missing_k20_docs)})",
        )


def scan_candidates_structural(
    target: set[str], fold_by_query: dict[str, int], baseline: dict[str, list[str]],
    actions_by_query: dict[str, list[tuple[str, int, str]]],
) -> dict[str, Any]:
    selected = {"fold", "doc_id", "union_rank", "source_support", "source_ranks", "candidates"}
    rows, stats = stream_target_jsonl(CANDIDATES, target, selected)
    state: dict[str, Any] = {
        "errors": [], "closed_queries": set(), "query_document_counts": {}, "full_incoming_count": 0,
        "k77_incoming_count": 0, "candidate_doc_count": 0, "current_doc_ranks": {},
    }
    current_query = ""
    current_docs: set[str] = set()
    current_ranks: set[int] = set()
    for line, query_id, row, _ in rows:
        try:
            if as_int(row.get("fold"), field="candidate.fold") != fold_by_query[query_id]:
                raise ValueError("fold mismatch")
            if current_query and query_id != current_query:
                close_candidate_group(current_query, current_docs, current_ranks, target, baseline, actions_by_query, state)
                current_docs, current_ranks = set(), set()
                state["current_doc_ranks"] = {}
            if query_id in state["closed_queries"]:
                raise ValueError("query candidate rows are non-contiguous")
            current_query = query_id
            items = candidate_items(row)
            if not items:
                raise ValueError("candidate row has no candidate items")
            for item in items:
                doc = str(item.get("doc_id", ""))
                rank = as_int(item.get("union_rank"), field="candidate.union_rank")
                if not doc or rank < 1 or rank > 200:
                    raise ValueError("invalid document ID or union rank")
                if doc in current_docs:
                    raise ValueError("duplicate candidate document within query")
                if rank in current_ranks:
                    raise ValueError("duplicate candidate union rank within query")
                current_docs.add(doc)
                current_ranks.add(rank)
                state["current_doc_ranks"][doc] = rank
                state["candidate_doc_count"] += 1
        except (KeyError, TypeError, ValueError) as exc:
            error_sample(state["errors"], f"{rel(CANDIDATES)}:{line}:{query_id}:{exc}")
    close_candidate_group(current_query, current_docs, current_ranks, target, baseline, actions_by_query, state)
    stats["target_top_level_keys"] = sorted(stats["target_top_level_keys"])
    stats["unique_target_query_count"] = len(state["closed_queries"])
    stats["missing_target_query_count"] = len(target - state["closed_queries"])
    stats["target_candidate_document_count"] = state["candidate_doc_count"]
    stats["full_incoming_candidate_count"] = state["full_incoming_count"]
    stats["k77_incoming_candidate_count"] = state["k77_incoming_count"]
    stats["min_docs_per_target_query"] = min(state["query_document_counts"].values(), default=0)
    stats["max_docs_per_target_query"] = max(state["query_document_counts"].values(), default=0)
    stats["errors"] = state["errors"]
    return stats


def new_branch(name: str, allowed_ranks: tuple[int, ...]) -> dict[str, Any]:
    return {
        "name": name,
        "allowed_ranks": allowed_ranks,
        "best": {},
        "positive_action_utilities": [],
        "errors": [],
    }


def action_key(query_id: str, incoming: str, drop_rank: int) -> str:
    return f"{query_id}:{incoming}:{drop_rank}"


def blank_source_provenance() -> dict[str, Any]:
    return {
        "source_presence": {"adaptive_dense": None, "BM25": None, "word_KNN": None, "char_KNN": None},
        "source_rank": {"adaptive_dense": None, "BM25": None, "word_KNN": None, "char_KNN": None},
        "source_support_count": None,
        "provenance_available": False,
    }


def merge_provenance(current: dict[str, Any], incoming: dict[str, Any]) -> None:
    for source in current["source_presence"]:
        existing_presence = current["source_presence"][source]
        next_presence = incoming["source_presence"].get(source)
        if existing_presence is None and next_presence is not None:
            current["source_presence"][source] = next_presence
        existing_rank = current["source_rank"][source]
        next_rank = incoming["source_rank"].get(source)
        if existing_rank is None and next_rank is not None:
            current["source_rank"][source] = next_rank
    if current["source_support_count"] is None and incoming["source_support_count"] is not None:
        current["source_support_count"] = incoming["source_support_count"]
    current["provenance_available"] = any(
        value is not None for value in current["source_presence"].values()
    ) or current["source_support_count"] is not None


def provenance_from_action_features(features: dict[str, Any]) -> dict[str, Any]:
    source_fields = {
        "adaptive_dense": ("incoming_adaptive_k500_rank", "incoming_adaptive_k500_missing"),
        "BM25": ("incoming_bm25_rank", "incoming_bm25_missing"),
        "word_KNN": ("incoming_knn_word_rank", "incoming_knn_word_missing"),
        "char_KNN": ("incoming_knn_char_rank", "incoming_knn_char_missing"),
    }
    result = blank_source_provenance()
    for source, (rank_field, missing_field) in source_fields.items():
        if missing_field in features:
            missing = as_finite_float(features[missing_field], field=missing_field)
            result["source_presence"][source] = missing == 0.0
            if missing == 0.0 and rank_field in features:
                rank = as_finite_float(features[rank_field], field=rank_field)
                result["source_rank"][source] = int(rank) if rank > 0 else None
        elif rank_field in features:
            rank = as_finite_float(features[rank_field], field=rank_field)
            if rank > 0:
                result["source_presence"][source] = True
                result["source_rank"][source] = int(rank)
    if "incoming_source_support" in features:
        support = as_finite_float(features["incoming_source_support"], field="incoming_source_support")
        result["source_support_count"] = int(support)
    result["provenance_available"] = any(
        value is not None for value in result["source_presence"].values()
    ) or result["source_support_count"] is not None
    return result


def source_rank_from_mapping(mapping: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in mapping and mapping[name] is not None:
            rank = as_finite_float(mapping[name], field=f"source_ranks.{name}")
            if rank > 0:
                return int(rank)
    return None


def provenance_from_candidate(item: dict[str, Any]) -> dict[str, Any]:
    result = blank_source_provenance()
    source_ranks = item.get("source_ranks")
    if isinstance(source_ranks, dict):
        aliases = {
            "adaptive_dense": ("adaptive_dense", "adaptive_k500", "adaptive", "bge"),
            "BM25": ("bm25",),
            "word_KNN": ("word_knn", "knn_word"),
            "char_KNN": ("char_knn", "knn_char"),
        }
        for source, names in aliases.items():
            rank = source_rank_from_mapping(source_ranks, names)
            if rank is not None:
                result["source_presence"][source] = True
                result["source_rank"][source] = rank
    if "source_support" in item and item["source_support"] is not None:
        result["source_support_count"] = as_int(item["source_support"], field="candidate.source_support")
    result["provenance_available"] = any(
        value is not None for value in result["source_presence"].values()
    ) or result["source_support_count"] is not None
    return result


def add_beneficial_record(
    registry: dict[str, dict[str, Any]], *, query_id: str, incoming: str, drop_rank: int,
    utility: float, membership: str, provenance: dict[str, Any] | None = None,
) -> None:
    key = action_key(query_id, incoming, drop_rank)
    record = registry.setdefault(
        key,
        {
            "reference": key,
            "query_id": query_id,
            "incoming_doc_id": incoming,
            "drop_rank": drop_rank,
            "utility": utility,
            "membership": set(),
            **blank_source_provenance(),
        },
    )
    record["utility"] = max(record["utility"], utility)
    record["membership"].add(membership)
    if provenance is not None:
        merge_provenance(record, provenance)


def choose_best(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Choose max utility with a deterministic label-free tie breaker."""
    if candidate["utility"] > current["utility"] + EPSILON:
        return candidate
    if abs(candidate["utility"] - current["utility"]) <= EPSILON and candidate["utility"] > EPSILON:
        if candidate["tie_break"] < current["tie_break"]:
            return candidate
    return current


def compute_baseline_metrics(
    target: set[str], fold_by_query: dict[str, int], gold: dict[str, set[str]], baseline: dict[str, list[str]]
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    per_query: dict[str, dict[str, float]] = {}
    sums: dict[int, dict[str, float]] = {fold: {"recall": 0.0, "precision": 0.0} for fold in TARGET_FOLDS}
    for query_id in target:
        hits = len(gold[query_id] & set(baseline[query_id]))
        recall = hits / len(gold[query_id])
        precision = hits / len(baseline[query_id])
        per_query[query_id] = {"recall": recall, "precision": precision, "hits": float(hits)}
        sums[fold_by_query[query_id]]["recall"] += recall
        sums[fold_by_query[query_id]]["precision"] += precision
    per_fold = {
        str(fold): {
            "query_count": 1400,
            "macro_recall": sums[fold]["recall"] / 1400,
            "macro_precision": sums[fold]["precision"] / 1400,
        }
        for fold in TARGET_FOLDS
    }
    pooled = {
        "query_count": len(target),
        "macro_recall": sum(value["recall"] for value in per_query.values()) / len(target),
        "macro_precision": sum(value["precision"] for value in per_query.values()) / len(target),
        "per_fold": per_fold,
    }
    return per_query, pooled


def compute_k20_oracle(
    target: set[str], gold: dict[str, set[str]], baseline: dict[str, list[str]],
    actions_by_query: dict[str, list[tuple[str, int, str]]], baseline_metrics: dict[str, dict[str, float]],
    registry: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, set[int]]]:
    branch = new_branch("K20", (4, 5))
    rank_opportunities: dict[str, set[int]] = {query_id: set() for query_id in target}
    for query_id in sorted(target, key=query_sort_key):
        current = {"utility": 0.0, "action": None, "tie_break": (10**9, 10**9, "")}
        for incoming, drop_rank, dropped in actions_by_query[query_id]:
            swapped = list(baseline[query_id])
            swapped[drop_rank - 1] = incoming
            if len(swapped) != len(set(swapped)):
                error_sample(branch["errors"], f"{query_id}: duplicate post-swap K20 top5")
                continue
            after = len(gold[query_id] & set(swapped)) / len(gold[query_id])
            utility = after - baseline_metrics[query_id]["recall"]
            candidate = {
                "utility": utility,
                "action": {"incoming_doc_id": incoming, "dropped_doc_id": dropped, "drop_rank": drop_rank},
                "tie_break": (drop_rank, drop_rank, incoming),
            }
            current = choose_best(current, candidate)
            if utility > EPSILON:
                branch["positive_action_utilities"].append(utility)
                rank_opportunities[query_id].add(drop_rank)
                add_beneficial_record(
                    registry, query_id=query_id, incoming=incoming, drop_rank=drop_rank,
                    utility=utility, membership="K20",
                )
        branch["best"][query_id] = current
    return branch, rank_opportunities


def attach_k20_action_provenance(
    target: set[str], registry: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    wanted = set(registry)
    selected = {"incoming_doc_id", "drop_rank", "features"}
    rows, stats = stream_target_jsonl(ACTIONS, target, selected)
    found: set[str] = set()
    errors: list[str] = []
    for line, query_id, row, _ in rows:
        try:
            key = action_key(query_id, str(row["incoming_doc_id"]), as_int(row["drop_rank"], field="action.drop_rank"))
            if key not in wanted:
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError("beneficial action lacks features")
            merge_provenance(registry[key], provenance_from_action_features(features))
            found.add(key)
        except (KeyError, TypeError, ValueError) as exc:
            error_sample(errors, f"{rel(ACTIONS)}:{line}:{query_id}:{exc}")
    stats["target_top_level_keys"] = sorted(stats["target_top_level_keys"])
    stats["beneficial_action_provenance_found"] = len(found)
    stats["beneficial_action_provenance_missing"] = len(wanted - found)
    stats["errors"] = errors
    return stats


def compute_candidate_oracles(
    target: set[str], fold_by_query: dict[str, int], gold: dict[str, set[str]],
    baseline: dict[str, list[str]], baseline_metrics: dict[str, dict[str, float]],
    registry: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    k77 = new_branch("K77", (4, 5))
    full = new_branch("FULL", (1, 2, 3, 4, 5))
    for branch in (k77, full):
        branch["best"] = {
            query_id: {"utility": 0.0, "action": None, "tie_break": (10**9, 10**9, "")}
            for query_id in target
        }
    selected = {"fold", "doc_id", "union_rank", "source_support", "source_ranks", "candidates"}
    rows, stats = stream_target_jsonl(CANDIDATES, target, selected)
    errors: list[str] = []
    seen_by_query: dict[str, set[str]] = defaultdict(set)
    rank_by_query: dict[str, set[int]] = defaultdict(set)
    for line, query_id, row, _ in rows:
        try:
            if as_int(row.get("fold"), field="candidate.fold") != fold_by_query[query_id]:
                raise ValueError("fold mismatch")
            for item in candidate_items(row):
                incoming = str(item.get("doc_id", ""))
                union_rank = as_int(item.get("union_rank"), field="candidate.union_rank")
                if not incoming or union_rank < 1 or union_rank > 200:
                    raise ValueError("invalid candidate identity")
                if incoming in seen_by_query[query_id] or union_rank in rank_by_query[query_id]:
                    raise ValueError("duplicate candidate identity")
                seen_by_query[query_id].add(incoming)
                rank_by_query[query_id].add(union_rank)
                if incoming in baseline[query_id]:
                    continue
                provenance = provenance_from_candidate(item)
                for branch, enabled in ((k77, union_rank <= 77), (full, True)):
                    if not enabled:
                        continue
                    for drop_rank in branch["allowed_ranks"]:
                        swapped = list(baseline[query_id])
                        dropped = swapped[drop_rank - 1]
                        swapped[drop_rank - 1] = incoming
                        if len(swapped) != len(set(swapped)):
                            raise ValueError("candidate swap would duplicate a top5 document")
                        after = len(gold[query_id] & set(swapped)) / len(gold[query_id])
                        utility = after - baseline_metrics[query_id]["recall"]
                        candidate = {
                            "utility": utility,
                            "action": {"incoming_doc_id": incoming, "dropped_doc_id": dropped, "drop_rank": drop_rank},
                            "tie_break": (union_rank, drop_rank, incoming),
                        }
                        branch["best"][query_id] = choose_best(branch["best"][query_id], candidate)
                        if utility > EPSILON:
                            branch["positive_action_utilities"].append(utility)
                            add_beneficial_record(
                                registry, query_id=query_id, incoming=incoming, drop_rank=drop_rank,
                                utility=utility, membership=branch["name"], provenance=provenance,
                            )
        except (KeyError, TypeError, ValueError) as exc:
            error_sample(errors, f"{rel(CANDIDATES)}:{line}:{query_id}:{exc}")
    for branch in (k77, full):
        branch["errors"].extend(errors)
    stats["target_top_level_keys"] = sorted(stats["target_top_level_keys"])
    stats["unique_target_query_count"] = len(seen_by_query)
    stats["missing_target_query_count"] = len(target - set(seen_by_query))
    stats["errors"] = errors
    return k77, full, stats


def distribution(values: Iterable[float]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        counter[format(value, ".15g")] += 1
    return dict(sorted(counter.items(), key=lambda item: float(item[0])))


def summarize_branch(
    branch: dict[str, Any], target: set[str], fold_by_query: dict[str, int], gold: dict[str, set[str]],
    baseline: dict[str, list[str]], baseline_metrics: dict[str, dict[str, float]], expected_recall: float,
) -> dict[str, Any]:
    per_fold_sums: dict[int, dict[str, float]] = {
        fold: {"oracle_recall": 0.0, "oracle_precision": 0.0, "gain": 0.0, "opportunity_queries": 0}
        for fold in TARGET_FOLDS
    }
    best_utilities: list[float] = []
    oracle_recall_by_query: dict[str, float] = {}
    oracle_precision_by_query: dict[str, float] = {}
    for query_id in target:
        best = branch["best"][query_id]
        utility = max(0.0, best["utility"])
        action = best["action"] if utility > EPSILON else None
        predicted = list(baseline[query_id])
        if action is not None:
            predicted[action["drop_rank"] - 1] = action["incoming_doc_id"]
        hits = len(gold[query_id] & set(predicted))
        recall = hits / len(gold[query_id])
        precision = hits / len(predicted)
        oracle_recall_by_query[query_id] = recall
        oracle_precision_by_query[query_id] = precision
        best_utilities.append(utility)
        fold = fold_by_query[query_id]
        per_fold_sums[fold]["oracle_recall"] += recall
        per_fold_sums[fold]["oracle_precision"] += precision
        per_fold_sums[fold]["gain"] += utility
        if utility > EPSILON:
            per_fold_sums[fold]["opportunity_queries"] += 1
    per_fold = {
        str(fold): {
            "query_count": 1400,
            "oracle_macro_recall": per_fold_sums[fold]["oracle_recall"] / 1400,
            "oracle_macro_precision": per_fold_sums[fold]["oracle_precision"] / 1400,
            "gain_vs_baseline": per_fold_sums[fold]["gain"] / 1400,
            "opportunity_query_count": int(per_fold_sums[fold]["opportunity_queries"]),
        }
        for fold in TARGET_FOLDS
    }
    pooled_recall = sum(oracle_recall_by_query.values()) / len(target)
    pooled_precision = sum(oracle_precision_by_query.values()) / len(target)
    pooled_gain = pooled_recall - sum(metric["recall"] for metric in baseline_metrics.values()) / len(target)
    status = "PASS" if not branch["errors"] and abs(pooled_recall - expected_recall) <= TOLERANCE else "BLOCKED"
    return {
        "status": status,
        "expected_oracle_recall": expected_recall,
        "oracle_macro_recall": pooled_recall,
        "oracle_macro_precision": pooled_precision,
        "gain_vs_baseline": pooled_gain,
        "per_fold": per_fold,
        "opportunity_query_count": sum(value > EPSILON for value in best_utilities),
        "best_action_utility_distribution": distribution(best_utilities),
        "positive_best_gain_distribution": distribution(value for value in best_utilities if value > EPSILON),
        "beneficial_action_utility_distribution": distribution(branch["positive_action_utilities"]),
        "beneficial_action_count": len(branch["positive_action_utilities"]),
        "absolute_difference_from_expected": abs(pooled_recall - expected_recall),
        "errors": branch["errors"],
        "_recall_by_query": oracle_recall_by_query,
        "_precision_by_query": oracle_precision_by_query,
    }


def serialize_provenance(registry: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_combo: Counter[str] = Counter()
    support_kind: Counter[str] = Counter()
    missed_k20_combo: Counter[str] = Counter()
    coverage = 0
    for key in sorted(registry):
        value = registry[key]
        presence = value["source_presence"]
        known_sources = sorted(source for source, present in presence.items() if present is True)
        unknown_sources = sorted(source for source, present in presence.items() if present is None)
        combo = "+".join(known_sources) if known_sources else "SOURCE_UNAVAILABLE"
        if unknown_sources:
            combo += "|unknown=" + "+".join(unknown_sources)
        by_combo[combo] += 1
        if known_sources:
            coverage += 1
            support_kind["single_source" if len(known_sources) == 1 else "multi_source"] += 1
        else:
            support_kind["source_unavailable"] += 1
        membership = sorted(value["membership"])
        if "K20" not in membership and ("K77" in membership or "FULL" in membership):
            missed_k20_combo[combo] += 1
        records.append(
            {
                "reference": value["reference"],
                "query_id": value["query_id"],
                "incoming_doc_id": value["incoming_doc_id"],
                "drop_rank": value["drop_rank"],
                "utility": value["utility"],
                "membership": membership,
                "source_presence": presence,
                "source_rank": value["source_rank"],
                "source_support_count": value["source_support_count"],
                "source_combination": combo,
            }
        )
    summary = {
        "beneficial_incoming_action_count": len(records),
        "explicit_source_coverage": {"covered": coverage, "total": len(records)},
        "beneficial_incoming_by_source_combination": dict(sorted(by_combo.items())),
        "single_source_vs_multi_source_support": dict(sorted(support_kind.items())),
        "k20_missed_but_k77_or_full_found_by_source_combination": dict(sorted(missed_k20_combo.items())),
    }
    return records, summary


def taxonomy(
    target: set[str], k20: dict[str, Any], k77: dict[str, Any], full: dict[str, Any]
) -> tuple[dict[str, str], dict[str, int]]:
    result: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for query_id in target:
        k20_gain = k20["_recall_by_query"][query_id] - 0.0  # key presence guard only
        # Use the immutable best utility, not historical policy behavior.
        k20_positive = k20["_best"][query_id]["utility"] > EPSILON
        k77_positive = k77["_best"][query_id]["utility"] > EPSILON
        full_positive = full["_best"][query_id]["utility"] > EPSILON
        if k20_positive:
            category = "K20_REPAIR_AVAILABLE"
        elif k77_positive:
            category = "K77_ONLY_REPAIR_AVAILABLE"
        elif full_positive:
            category = "FULL_ONLY_REPAIR_AVAILABLE"
        else:
            category = "NO_REPAIR_FOUND_IN_FULL_POOL"
        result[query_id] = category
        counts[category] += 1
    return result, dict(sorted(counts.items()))


def metric_report_matches(
    baseline_summary: dict[str, Any], baseline_report: dict[str, Any]
) -> dict[str, Any]:
    report_metrics = baseline_report.get("baseline_093_oof", {})
    report_folds = report_metrics.get("per_fold", {}) if isinstance(report_metrics, dict) else {}
    matches: dict[str, Any] = {}
    for fold in TARGET_FOLDS:
        expected = report_folds.get(str(fold), {}) if isinstance(report_folds, dict) else {}
        actual = baseline_summary["per_fold"][str(fold)]
        matches[str(fold)] = {
            "recall_expected_from_canonical_report": expected.get("macro_recall"),
            "precision_expected_from_canonical_report": expected.get("macro_precision"),
            "recall_absolute_difference": abs(actual["macro_recall"] - float(expected.get("macro_recall", float("nan")))),
            "precision_absolute_difference": abs(actual["macro_precision"] - float(expected.get("macro_precision", float("nan")))),
        }
        matches[str(fold)]["match_within_tolerance"] = (
            matches[str(fold)]["recall_absolute_difference"] <= TOLERANCE
            and matches[str(fold)]["precision_absolute_difference"] <= TOLERANCE
        )
    return matches


def input_manifest_entry(path: Path, role: str, label_bearing: bool, *, expected_sha: str | None = None) -> dict[str, Any]:
    present = path.exists()
    digest = sha256(path) if present else None
    return {
        "path": rel(path),
        "exists": present,
        "sha256": digest,
        "expected_sha256": expected_sha,
        "sha256_match": digest == expected_sha if expected_sha else None,
        "bytes": path.stat().st_size if present else None,
        "scientific_role": role,
        "label_bearing": label_bearing,
        "row_or_query_count": None,
        "schema": None,
    }


def json_object_schema(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return sorted(value) if isinstance(value, dict) else [type(value).__name__]


def run_git_diff_check() -> str:
    result = subprocess.run(["git", "diff", "--check"], cwd=ROOT, text=True, capture_output=True, check=False)
    return "PASS" if result.returncode == 0 else "FAIL"


def terminal_summary(
    status: str, baseline_summary: dict[str, Any] | None, k20: dict[str, Any] | None,
    k77: dict[str, Any] | None, full: dict[str, Any] | None, rank4: int, rank5: int,
    source_summary: dict[str, Any], shared_mismatches: int, k20_mismatches: int,
    k77_mismatches: int, full_mismatches: int, git_check: str,
) -> str:
    def metric(branch: dict[str, Any] | None, field: str) -> str:
        return str(branch[field]) if branch is not None else "N/A"

    coverage = source_summary["explicit_source_coverage"]
    baseline = baseline_summary["macro_recall"] if baseline_summary else "N/A"
    baseline_match = "YES" if baseline_summary and abs(baseline - EXPECTED_BASELINE_RECALL) <= TOLERANCE else "NO"
    k20_delta_k77 = (
        k77["oracle_macro_recall"] - k20["oracle_macro_recall"] if k20 and k77 else "N/A"
    )
    k20_delta_full = (
        full["oracle_macro_recall"] - k20["oracle_macro_recall"] if k20 and full else "N/A"
    )
    return "\n".join(
        [
            "WORKFLOW C0-A ORACLE / OPPORTUNITY ANATOMY",
            "",
            "Status:", status,
            "",
            f"F1-F4 queries:\n{EXPECTED_QUERY_COUNT if baseline_summary else 0}/5600",
            "",
            "Fold0 labels materialized:\nNO",
            "",
            "Public labels used:\nNO",
            "",
            f"Baseline Recall:\n{baseline}",
            "",
            f"Expected baseline Recall match:\n{baseline_match}",
            "",
            f"K20 status:\n{metric(k20, 'status')}",
            "",
            f"K20 oracle Recall:\n{metric(k20, 'oracle_macro_recall')}",
            "",
            f"K20 opportunity queries:\n{metric(k20, 'opportunity_query_count')}",
            "",
            f"K77 status:\n{metric(k77, 'status')}",
            "",
            f"K77 oracle Recall:\n{metric(k77, 'oracle_macro_recall')}",
            "",
            f"K77 opportunity queries:\n{metric(k77, 'opportunity_query_count')}",
            "",
            f"K20 to K77 oracle delta:\n{k20_delta_k77}",
            "",
            f"Full-pool status:\n{metric(full, 'status')}",
            "",
            f"Full-pool oracle Recall:\n{metric(full, 'oracle_macro_recall')}",
            "",
            f"K20 to full oracle delta:\n{k20_delta_full}",
            "",
            f"Rank4 opportunity queries:\n{rank4}",
            "",
            f"Rank5 opportunity queries:\n{rank5}",
            "",
            f"Candidate/source provenance coverage:\n{coverage['covered']}/{coverage['total']}",
            "",
            f"Shared join mismatches:\n{shared_mismatches}",
            "",
            f"K20 join mismatches:\n{k20_mismatches}",
            "",
            f"K77 join mismatches:\n{k77_mismatches}",
            "",
            f"Full-pool join mismatches:\n{full_mismatches}",
            "",
            f"Integrity manifest:\n{rel(MANIFEST_OUT)}",
            "",
            f"Opportunity report:\n{rel(REPORT_OUT)}",
            "",
            f"Opportunity trace:\n{rel(TRACE_OUT)}",
            "",
            "Historical V3A choices used:\nNO",
            "",
            "Model training:\nNO",
            "",
            "Model inference:\nNO",
            "",
            "GPU:\nNO",
            "",
            "Modal:\nNO",
            "",
            f"git diff --check:\n{git_check}",
            "",
            "Safe next action:\nREVIEW_C0A_RESULT_BEFORE_C2",
            "",
            "STOP.",
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lifecycle = {
        "pre_create_duplicate_check": {
            "searched_paths": ["scripts/analysis/workflow_c", "reports/task1/workflow_c/tv2/c0a"],
            "equivalent_canonical_c0a_implementation_found": False,
            "equivalent_c0a_outputs_found": False,
        },
        "decisions": [
            {"path": rel(Path(__file__)), "decision": "VERSIONED_NEW_FILE", "classification": "KEEP_REPRODUCIBILITY", "reason": "canonical CPU-only C0-A implementation"},
            {"path": rel(MANIFEST_OUT), "decision": "VERSIONED_NEW_FILE", "reason": "canonical integrity manifest required by C0-A"},
            {"path": rel(REPORT_OUT), "decision": "VERSIONED_NEW_FILE", "reason": "canonical opportunity report required by C0-A"},
            {"path": rel(TRACE_OUT), "decision": "VERSIONED_NEW_FILE", "reason": "canonical 5,600-query opportunity trace required by C0-A"},
        ],
        "workflow_a_or_b_artifacts_modified": False,
        "deletions": [],
    }
    input_specs = [
        (FOLDS, "canonical strict-CV F1--F4 membership", False),
        (TRAIN, "F1--F4-only diagnostic gold labels after structural freeze", True),
        (BASELINE, "canonical baseline top5 identity", False),
        (BASELINE_REPORT, "canonical baseline Recall/Precision reference", True),
        (ACTIONS, "canonical K20 swap action universe", True),
        (CANDIDATES, "canonical full candidate pool and retrieval provenance", False),
        (STEP0_REPORT, "metric-contract equivalence evidence", False),
        (FULL_REPORT, "canonical full-pool oracle evidence", False),
        (K20_REPORT, "canonical K20 oracle evidence", False),
        (K77_REPORT, "canonical K77 oracle evidence", False),
    ]
    inputs = {
        rel(path): input_manifest_entry(path, role, label_bearing, expected_sha=EXPECTED_SHA256.get(path))
        for path, role, label_bearing in input_specs
    }
    missing_inputs = [path for path, _, _ in input_specs if not path.exists()]
    if missing_inputs:
        raise RuntimeError("required canonical input missing: " + ", ".join(rel(path) for path in missing_inputs))

    # Read report contracts before any gold label payload.  They contain no row-level labels.
    step0 = json.loads(STEP0_REPORT.read_text(encoding="utf-8"))
    full_evidence = json.loads(FULL_REPORT.read_text(encoding="utf-8"))
    k20_evidence = json.loads(K20_REPORT.read_text(encoding="utf-8"))
    k77_evidence = json.loads(K77_REPORT.read_text(encoding="utf-8"))
    baseline_reference = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    for path in (STEP0_REPORT, FULL_REPORT, K20_REPORT, K77_REPORT, BASELINE_REPORT):
        # These are scalar JSON reports, but each records the scientific
        # query population explicitly.  Bind that population rather than
        # pretending the report has one scientific row.
        report_query_counts = {
            STEP0_REPORT: step0.get("total_queries"),
            FULL_REPORT: full_evidence.get("total_queries"),
            K20_REPORT: k20_evidence.get("audited_queries", k20_evidence.get("total_queries")),
            K77_REPORT: k77_evidence.get("total_queries"),
            BASELINE_REPORT: baseline_reference.get("baseline_093_oof", {}).get("pooled", {}).get("query_count"),
        }
        inputs[rel(path)]["row_or_query_count"] = report_query_counts.get(path)
        inputs[rel(path)]["schema"] = {"top_level_keys": json_object_schema(path)}

    target_by_query, fold_counts, fold_duplicate_count = read_target_fold_map(FOLDS)
    target = set(target_by_query)
    inputs[rel(FOLDS)].update({
        "row_or_query_count": len(target),
        "schema": {"kind": "fold mapping", "target_folds": list(TARGET_FOLDS)},
    })
    baseline, baseline_stats = read_baseline_structural(target, target_by_query)
    inputs[rel(BASELINE)].update({
        "row_or_query_count": baseline_stats["nonempty_row_count"],
        "schema": {"target_top_level_keys": baseline_stats["target_top_level_keys"]},
    })
    actions_by_query, action_stats = read_actions_structural(target, target_by_query, baseline)
    inputs[rel(ACTIONS)].update({
        "row_or_query_count": action_stats["nonempty_row_count"],
        "schema": {"target_top_level_keys": action_stats["target_top_level_keys"], "feature_size_distribution": action_stats["feature_size_distribution"]},
    })
    candidate_stats = scan_candidates_structural(target, target_by_query, baseline, actions_by_query)
    inputs[rel(CANDIDATES)].update({
        "row_or_query_count": candidate_stats["nonempty_row_count"],
        "schema": {"target_top_level_keys": candidate_stats["target_top_level_keys"]},
    })

    shared_errors: list[str] = []
    if len(target) != EXPECTED_QUERY_COUNT:
        error_sample(shared_errors, f"target query count {len(target)} != {EXPECTED_QUERY_COUNT}")
    for fold in TARGET_FOLDS:
        if fold_counts.get(str(fold)) != 1400:
            error_sample(shared_errors, f"fold {fold} count {fold_counts.get(str(fold))} != 1400")
    if fold_duplicate_count:
        error_sample(shared_errors, f"duplicate target fold IDs: {fold_duplicate_count}")
    if baseline_stats["errors"]:
        shared_errors.extend(baseline_stats["errors"])
    if baseline_stats["missing_target_query_count"]:
        error_sample(shared_errors, f"baseline missing target queries: {baseline_stats['missing_target_query_count']}")
    if step0.get("status") != "PASS" or step0.get("total_queries") != EXPECTED_QUERY_COUNT:
        error_sample(shared_errors, "Step0 metric contract is not PASS for 5,600 target queries")
    if step0.get("fold0_touched") is not False or step0.get("public_labels_used") is not False:
        error_sample(shared_errors, "Step0 scope contract does not prove Fold0/public-label exclusion")
    if not all(entry["sha256_match"] is not False for entry in inputs.values()):
        error_sample(shared_errors, "one or more canonical input SHA256 checks failed")

    structural_freeze_passed = not shared_errors and not action_stats["errors"] and not candidate_stats["errors"]
    if not structural_freeze_passed:
        # The current canonical artifacts are expected to pass.  Fail closed rather than decode labels.
        raise RuntimeError("structural identities did not freeze: " + " | ".join((shared_errors + action_stats["errors"] + candidate_stats["errors"])[:5]))

    # The only gold-label decoding happens here, after all structural identities and joins are frozen.
    gold, gold_stats = read_target_gold(TRAIN, target)
    inputs[rel(TRAIN)].update({
        "row_or_query_count": gold_stats["target_query_count"],
        "schema": {"target_record_keys": gold_stats["target_record_keys"]},
    })
    if set(gold) != target:
        raise RuntimeError("target gold/query whitelist coverage mismatch")
    baseline_per_query, baseline_summary = compute_baseline_metrics(target, target_by_query, gold, baseline)
    baseline_report_match = metric_report_matches(baseline_summary, baseline_reference)
    baseline_reference_ok = all(value["match_within_tolerance"] for value in baseline_report_match.values())
    if abs(baseline_summary["macro_recall"] - EXPECTED_BASELINE_RECALL) > TOLERANCE:
        raise RuntimeError("baseline Recall does not match the C0-A contract")
    if not baseline_reference_ok:
        raise RuntimeError("per-fold baseline Recall/Precision does not match canonical baseline report")

    registry: dict[str, dict[str, Any]] = {}
    k20_branch, rank_opportunities = compute_k20_oracle(
        target, gold, baseline, actions_by_query, baseline_per_query, registry
    )
    k20_summary = summarize_branch(
        k20_branch, target, target_by_query, gold, baseline, baseline_per_query, EXPECTED_ORACLE_RECALL["K20"]
    )
    k20_summary["_best"] = k20_branch["best"]
    k20_provenance_stats = attach_k20_action_provenance(target, registry)

    k77_branch, full_branch, candidate_oracle_stats = compute_candidate_oracles(
        target, target_by_query, gold, baseline, baseline_per_query, registry
    )
    k77_summary = summarize_branch(
        k77_branch, target, target_by_query, gold, baseline, baseline_per_query, EXPECTED_ORACLE_RECALL["K77"]
    )
    full_summary = summarize_branch(
        full_branch, target, target_by_query, gold, baseline, baseline_per_query, EXPECTED_ORACLE_RECALL["FULL"]
    )
    k77_summary["_best"] = k77_branch["best"]
    full_summary["_best"] = full_branch["best"]

    # Evidence cross-checks have their own branch-local status.
    k20_evidence_gain = float(k20_evidence.get("pooled_C", float("nan")))
    k77_evidence_gain = float(k77_evidence.get("oracle_decomposition", {}).get("pooled_C77", float("nan")))
    full_evidence_gain = float(full_evidence.get("one_swap_candidate_ceiling_gain", float("nan")))
    k20_evidence_ok = abs(k20_summary["gain_vs_baseline"] - k20_evidence_gain) <= TOLERANCE
    k77_evidence_ok = abs(k77_summary["gain_vs_baseline"] - k77_evidence_gain) <= TOLERANCE
    full_evidence_ok = abs(full_summary["gain_vs_baseline"] - full_evidence_gain) <= TOLERANCE
    k77_identity_ok = candidate_stats["k77_incoming_candidate_count"] == EXPECTED_K77_INCOMING_COUNT
    if not k20_evidence_ok:
        k20_summary["status"] = "BLOCKED"
        error_sample(k20_summary["errors"], "K20 gain does not reproduce canonical K20 evidence")
    if not (k77_evidence_ok and k77_identity_ok):
        k77_summary["status"] = "BLOCKED"
        error_sample(k77_summary["errors"], "K77 evidence gain or exact K77 candidate count mismatch")
    if not full_evidence_ok:
        full_summary["status"] = "BLOCKED"
        error_sample(full_summary["errors"], "full-pool gain does not reproduce canonical full-pool evidence")

    tax_by_query, tax_counts = taxonomy(target, k20_summary, k77_summary, full_summary)
    provenance_records, provenance_summary = serialize_provenance(registry)
    rank4_queries = sum(4 in ranks for ranks in rank_opportunities.values())
    rank5_queries = sum(5 in ranks for ranks in rank_opportunities.values())
    rank_breakdown = {
        "rank4_only": sum(ranks == {4} for ranks in rank_opportunities.values()),
        "rank5_only": sum(ranks == {5} for ranks in rank_opportunities.values()),
        "both_rank4_rank5": sum(ranks == {4, 5} for ranks in rank_opportunities.values()),
        "rank4_any": rank4_queries,
        "rank5_any": rank5_queries,
    }

    shared_join_mismatches = (
        fold_duplicate_count + baseline_stats["missing_target_query_count"] + candidate_stats["missing_target_query_count"]
        + (0 if set(gold) == target else len(target ^ set(gold)))
    )
    k20_join_mismatches = action_stats["missing_target_query_count"] + action_stats["duplicate_action_identity_count"] + len(action_stats["errors"])
    k77_join_mismatches = candidate_oracle_stats["missing_target_query_count"] + len(candidate_oracle_stats["errors"])
    full_join_mismatches = candidate_oracle_stats["missing_target_query_count"] + len(candidate_oracle_stats["errors"])
    overall = "PASS" if all(summary["status"] == "PASS" for summary in (k20_summary, k77_summary, full_summary)) else (
        "PARTIAL" if k20_summary["status"] == "PASS" else "BLOCKED"
    )

    # Trace contains C0-A oracle facts only; it intentionally has no V3A-policy fields.
    with TRACE_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in sorted(target, key=query_sort_key):
            refs = [record["reference"] for record in provenance_records if record["query_id"] == query_id]
            trace = {
                "query_id": query_id,
                "fold": target_by_query[query_id],
                "baseline_top5": baseline[query_id],
                "baseline_relevant_count": len(gold[query_id]),
                "baseline_recall": baseline_per_query[query_id]["recall"],
                "K20_best_action": k20_branch["best"][query_id]["action"],
                "K20_best_utility": max(0.0, k20_branch["best"][query_id]["utility"]),
                "K20_oracle_recall": k20_summary["_recall_by_query"][query_id],
                "K77_best_action": k77_branch["best"][query_id]["action"],
                "K77_best_utility": max(0.0, k77_branch["best"][query_id]["utility"]),
                "K77_oracle_recall": k77_summary["_recall_by_query"][query_id],
                "full_best_action": full_branch["best"][query_id]["action"],
                "full_best_utility": max(0.0, full_branch["best"][query_id]["utility"]),
                "full_oracle_recall": full_summary["_recall_by_query"][query_id],
                "taxonomy": tax_by_query[query_id],
                "beneficial_incoming_provenance_references": refs,
            }
            handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")

    # Remove internal working keys before serializing reports.
    for summary in (k20_summary, k77_summary, full_summary):
        summary.pop("_recall_by_query", None)
        summary.pop("_precision_by_query", None)
        summary.pop("_best", None)

    report = {
        "workflow": "C0-A — HISTORICAL-POLICY-INDEPENDENT ORACLE / OPPORTUNITY ANATOMY",
        "status": overall,
        "scope": {
            "folds": list(TARGET_FOLDS), "query_count": len(target), "fold0_labels_materialized": False,
            "public_labels_used": False, "historical_v3a_choices_used": False,
            "model_training": False, "model_inference": False, "gpu": False, "modal": False,
        },
        "required_interpretation": "ORACLE RESULTS ARE LABEL-AWARE DIAGNOSTIC CEILINGS. THEY DO NOT ESTIMATE LEARNABLE OR PUBLIC PERFORMANCE.",
        "baseline": {**baseline_summary, "canonical_report_per_fold_crosscheck": baseline_report_match},
        "oracle_branches": {"K20": k20_summary, "K77": k77_summary, "FULL": full_summary},
        "K20_action_contract": {
            "action_rows_on_F1_F4": action_stats["target_action_count"],
            "feature_schema_distribution": action_stats["feature_size_distribution"],
            "drop_ranks": [4, 5],
            "drop_rank_violation_count": 0,
            "incoming_duplicate_of_baseline_count": 0,
            "protected_rank1_3": True,
            "one_swap_maximum": True,
        },
        "deltas": {
            "baseline_to_K20": k20_summary["gain_vs_baseline"],
            "K20_to_K77": k77_summary["oracle_macro_recall"] - k20_summary["oracle_macro_recall"],
            "K77_to_full": full_summary["oracle_macro_recall"] - k77_summary["oracle_macro_recall"],
            "K20_to_full": full_summary["oracle_macro_recall"] - k20_summary["oracle_macro_recall"],
        },
        "opportunity_counts": {
            "K20": k20_summary["opportunity_query_count"], "K77": k77_summary["opportunity_query_count"],
            "FULL": full_summary["opportunity_query_count"],
            "K77_incremental_over_K20": k77_summary["opportunity_query_count"] - k20_summary["opportunity_query_count"],
            "FULL_incremental_over_K20": full_summary["opportunity_query_count"] - k20_summary["opportunity_query_count"],
            "FULL_incremental_over_K77": full_summary["opportunity_query_count"] - k77_summary["opportunity_query_count"],
        },
        "k20_rank4_rank5_opportunity": rank_breakdown,
        "taxonomy_definition": {
            "K20_REPAIR_AVAILABLE": "at least one exact canonical K20 rank-4/rank-5 swap has positive Recall utility",
            "K77_ONLY_REPAIR_AVAILABLE": "no K20 repair; at least one exact union-rank<=77 rank-4/rank-5 swap has positive utility",
            "FULL_ONLY_REPAIR_AVAILABLE": "no K20/K77 repair; a full-pool one-swap has positive utility",
            "NO_REPAIR_FOUND_IN_FULL_POOL": "no permitted canonical full-pool one-swap has positive utility",
        },
        "taxonomy_counts": tax_counts,
        "source_provenance_summary": provenance_summary,
        "beneficial_incoming_provenance_records": provenance_records,
        "integrity_status": {
            "shared_core": "PASS" if shared_join_mismatches == 0 else "BLOCKED",
            "K20": k20_summary["status"], "K77": k77_summary["status"], "FULL": full_summary["status"],
            "join_mismatches": {"shared": shared_join_mismatches, "K20": k20_join_mismatches, "K77": k77_join_mismatches, "FULL": full_join_mismatches},
        },
    }

    git_check = run_git_diff_check()
    manifest = {
        "workflow": report["workflow"],
        "status": overall,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code": {"path": rel(Path(__file__)), "sha256": sha256(Path(__file__)), "classification": "KEEP_REPRODUCIBILITY"},
        "environment": {"python": sys.version, "platform": platform.platform(), "implementation": platform.python_implementation()},
        "metric_contract": {"official_primary": "macro_recall", "official_secondary": "macro_precision", "tolerance": TOLERANCE},
        "inputs": inputs,
        "fold_and_label_isolation": {
            "target_folds": list(TARGET_FOLDS), "target_query_count": len(target), "fold_counts": fold_counts,
            "target_query_id_sha256": sorted_set_hash(target), "duplicate_target_fold_ids": fold_duplicate_count,
            "missing_target_ids": 0, "fold0_validation_ids_materialized": False,
            "fold0_gold_payload_decoded_count": gold_stats["fold0_gold_payload_decoded_count"],
            "fold0_gold_records_structurally_skipped": gold_stats["non_target_gold_records_structurally_skipped"],
            "aggregate_baseline_non_target_rows_structurally_skipped": baseline_stats["non_target_rows_structurally_skipped"],
            "aggregate_actions_non_target_rows_structurally_skipped": action_stats["non_target_rows_structurally_skipped"],
            "aggregate_candidates_non_target_rows_structurally_skipped": candidate_stats["non_target_rows_structurally_skipped"],
            "public_labels_used": False,
        },
        "join_cardinalities": {
            "target": len(target), "gold": len(gold), "baseline": len(baseline), "K20_action_queries": len(actions_by_query),
            "candidate_queries": candidate_stats["unique_target_query_count"], "K20_action_rows": action_stats["target_action_count"],
            "K77_incoming_candidates": candidate_stats["k77_incoming_candidate_count"],
            "full_incoming_candidates": candidate_stats["full_incoming_candidate_count"],
        },
        "duplicate_and_missing_counts": {
            "fold_target_duplicates": fold_duplicate_count, "baseline_target_duplicate_rows": baseline_stats["duplicate_target_query_count"],
            "baseline_missing_target_queries": baseline_stats["missing_target_query_count"],
            "K20_duplicate_action_identities": action_stats["duplicate_action_identity_count"],
            "K20_missing_target_queries": action_stats["missing_target_query_count"],
            "candidate_missing_target_queries": candidate_stats["missing_target_query_count"],
        },
        "branch_statuses": {"K20": k20_summary["status"], "K77": k77_summary["status"], "FULL": full_summary["status"]},
        "evidence_crosschecks": {
            "baseline_recall_expected": EXPECTED_BASELINE_RECALL,
            "baseline_recall_actual": baseline_summary["macro_recall"],
            "baseline_per_fold_recall_precision_match": baseline_reference_ok,
            "K20_evidence_gain_match": k20_evidence_ok,
            "K77_evidence_gain_match": k77_evidence_ok,
            "K77_expected_incoming_count": EXPECTED_K77_INCOMING_COUNT,
            "K77_actual_incoming_count": candidate_stats["k77_incoming_candidate_count"],
            "FULL_evidence_gain_match": full_evidence_ok,
        },
        "structural_freeze_before_gold_decode": True,
        "file_lifecycle_review": lifecycle,
        "git_diff_check": git_check,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(terminal_summary(
        overall, baseline_summary, k20_summary, k77_summary, full_summary, rank4_queries, rank5_queries,
        provenance_summary, shared_join_mismatches, k20_join_mismatches, k77_join_mismatches,
        full_join_mismatches, git_check,
    ))


if __name__ == "__main__":
    main()
