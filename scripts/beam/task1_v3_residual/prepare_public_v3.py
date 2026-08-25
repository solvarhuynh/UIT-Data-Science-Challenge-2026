"""Build public V3 shortlist inputs from existing public ranking artifacts only.

This adapter deliberately does not read ``answer`` and does not run retrieval,
model inference, training, or GPU code.  ``fold=-1`` is an inference sentinel;
it is not a training fold and must not be consumed by policy training.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.beam.task1_v2.evidence import iter_payloads, prepare_document, select_true_s2_prepared

DEFAULT_QUESTIONS = ROOT / "data/raw/btc/LegalIR/public-official.json"
DEFAULT_UNION = ROOT / "artifacts/task1/recovery_096/final_public_v3/public_candidate_union.jsonl"
DEFAULT_UNION_MANIFEST = ROOT / "artifacts/task1/recovery_096/final_public_v3/public_candidate_union_manifest.json"
DEFAULT_BASELINE = ROOT / "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip"
DEFAULT_PAYLOADS = ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
DEFAULT_OUTPUT = ROOT / "artifacts/task1/recovery_096/final_public_v3"
DEFAULT_ADAPTIVE_REPORT = DEFAULT_OUTPUT / "public_adaptive_k500_report.json"
DEFAULT_ADAPTIVE_OUTPUT = DEFAULT_OUTPUT / "public_adaptive_k500.jsonl"
DEFAULT_ADAPTIVE_MANIFEST = ROOT / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_final_fit_manifest.json"
DEFAULT_ADAPTIVE_MODEL = ROOT / "artifacts/task1/recovery_096/adaptive_k500_v1/adaptive_k500_final_fit.joblib"
DEFAULT_V3A_MANIFEST = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit_manifest.json"
DEFAULT_V3A_MODEL = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib"
DEFAULT_TRAIN_FEATURE_REPORT = ROOT / "artifacts/task1/recovery_096/v3_residual/frozen_features_report.json"
DEFAULT_POLICY_REPORT = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/policy_training_report.json"
DEFAULT_BASELINE_MANIFEST = ROOT / "artifacts/task1/recovery_096/public_anchor_093/producer_manifest.json"
DEFAULT_TRAIN_BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"

TRAIN_ONLY_FIELDS = {"answer", "gold_documents", "gain", "label", "baseline_recall", "fold0"}


def refuse_unverified_execution() -> None:
    raise RuntimeError(
        "PUBLIC_PREP_BLOCKED_UNVERIFIED_CONTRACT: exact V3A source semantics "
        "and real train parity are not proven; no public artifact may be created"
    )


def verify_public_contract(
    questions_path: Path,
    union_path: Path,
    baseline_path: Path,
    payloads_path: Path,
    adaptive_report_path: Path = DEFAULT_ADAPTIVE_REPORT,
    adaptive_output_path: Path = DEFAULT_ADAPTIVE_OUTPUT,
    adaptive_manifest_path: Path = DEFAULT_ADAPTIVE_MANIFEST,
    adaptive_model_path: Path = DEFAULT_ADAPTIVE_MODEL,
    v3a_manifest_path: Path = DEFAULT_V3A_MANIFEST,
    v3a_model_path: Path = DEFAULT_V3A_MODEL,
    train_feature_report_path: Path = DEFAULT_TRAIN_FEATURE_REPORT,
    policy_report_path: Path = DEFAULT_POLICY_REPORT,
    baseline_manifest_path: Path = DEFAULT_BASELINE_MANIFEST,
    train_baseline_path: Path = DEFAULT_TRAIN_BASELINE,
    union_manifest_path: Path = DEFAULT_UNION_MANIFEST,
    public_raw_k500_path: Path = DEFAULT_OUTPUT / "public_raw_k500.jsonl",
) -> dict[str, Any]:
    """Compute the public contract; no status is accepted from a caller."""
    checks: dict[str, bool] = {}
    required = [questions_path, public_raw_k500_path, union_path, union_manifest_path, baseline_path, payloads_path, train_baseline_path,
                adaptive_output_path,
                adaptive_report_path, adaptive_manifest_path, adaptive_model_path,
                v3a_manifest_path, v3a_model_path,
                train_feature_report_path, policy_report_path, baseline_manifest_path]
    checks["required_files"] = all(path.is_file() for path in required)
    if not checks["required_files"]:
        return {"checks": checks, "missing": [str(path) for path in required if not path.is_file()], "production_adapter_verified": False}
    ids = load_public_ids(questions_path)
    try:
        union_manifest = json.loads(union_manifest_path.read_text(encoding="utf-8"))
        union = load_candidate_union(union_path, ids)
        source_sha256 = union_manifest.get("source_sha256", {})
        checks["union_contract"] = (
            len(union) == 1000
            and union_manifest.get("schema_version") == "public-candidate-union-v2"
            and union_manifest.get("status") == "PUBLIC_CANDIDATE_UNION_PASS"
            and union_manifest.get("source_keys") == ["adaptive_k500", "bm25", "knn_word", "knn_char"]
            and union_manifest.get("output_sha256") == sha256(union_path)
            and all(isinstance(value, str) and len(value) == 64 for value in source_sha256.values())
        )
    except (OSError, ValueError, TypeError):
        checks["union_contract"] = False
    try:
        baseline = load_baseline(baseline_path, ids)
        checks["baseline_shape"] = len(baseline) == 1000 and all(len(docs) == 5 and len(set(docs)) == 5 for docs in baseline.values())
    except (OSError, ValueError, TypeError, zipfile.BadZipFile):
        checks["baseline_shape"] = False
    try:
        adaptive = json.loads(adaptive_report_path.read_text(encoding="utf-8"))
        checks["adaptive_feature_parity"] = (
            adaptive.get("adaptive_feature_parity") is True
            and adaptive.get("public_bge_alignment", {}).get("status") == "PASS"
            and adaptive.get("public_bge_manifest", {}).get("status") == "PASS"
            and adaptive.get("public_raw_k500_sha256") == sha256(public_raw_k500_path)
        )
        checks["adaptive_gating_rule_parity"] = adaptive.get("adaptive_gating_rule_parity") is True and "selected_budget" in adaptive
        checks["adaptive_public_coverage"] = adaptive.get("status") == "PUBLIC_ADAPTIVE_K500_PASS" and int(adaptive.get("query_count", -1)) == 1000 and adaptive.get("no_public_labels_used") is True
    except (OSError, ValueError, TypeError):
        checks["adaptive_feature_parity"] = checks["adaptive_gating_rule_parity"] = checks["adaptive_public_coverage"] = False
    try:
        adaptive_rows = list(jsonl(adaptive_output_path))
        adaptive_ids = {str(row["query_id"]) for row in adaptive_rows}
        checks["adaptive_output_contract"] = len(adaptive_rows) == 1000 and adaptive_ids == set(ids) and all(
            not set(row) & TRAIN_ONLY_FIELDS and row.get("ranking") == list(row.get("k200_documents", [])) + list(row.get("k500_added_documents", []))
            for row in adaptive_rows
        )
    except (OSError, ValueError, TypeError, KeyError):
        checks["adaptive_output_contract"] = False
    try:
        adaptive_manifest = json.loads(adaptive_manifest_path.read_text(encoding="utf-8"))
        checks["adaptive_model_schema"] = bool(adaptive_manifest.get("feature_names")) and len(adaptive_manifest["feature_names"]) == 14 and adaptive_manifest.get("gating_rule", "").startswith("rank probabilities") and adaptive_manifest.get("model_sha256") == sha256(adaptive_model_path)
    except (OSError, ValueError, TypeError, KeyError):
        checks["adaptive_model_schema"] = False
    try:
        baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
        producer = baseline_manifest.get("producer", {})
        train_baseline = list(jsonl(train_baseline_path))
        train_baseline_contract = (
            len(train_baseline) == 7000
            and all(len(row.get("top5", [])) == 5 and len(set(map(str, row.get("top5", [])))) == 5 for row in train_baseline)
            and {int(row.get("fold", -1)) for row in train_baseline} == {0, 1, 2, 3, 4}
        )
        checks["baseline_semantic_parity"] = train_baseline_contract and (
            baseline_manifest.get("status") == "EXACT_BYTE_REPRODUCED"
            and producer.get("questions") == "data/raw/btc/LegalIR/public-official.json"
            and producer.get("writer") == "scripts/submission/write_legal_ir_submission.py"
            and int(baseline_manifest.get("anchor", {}).get("question_count", -1)) == 1000
            and int(baseline_manifest.get("anchor", {}).get("documents_per_question", -1)) == 5
            and baseline_manifest.get("constraints", {}).get("public_anchor_used_as_gold") is False
            and baseline_manifest.get("reproduction", {}).get("submission_sha256") == sha256(baseline_path)
        )
    except (OSError, ValueError, TypeError, KeyError):
        checks["baseline_semantic_parity"] = False
    try:
        train_features = json.loads(train_feature_report_path.read_text(encoding="utf-8"))
        policy = json.loads(policy_report_path.read_text(encoding="utf-8"))
        v3a = json.loads(v3a_manifest_path.read_text(encoding="utf-8"))
        from scripts.beam.task1_v3_residual.build_actions import BASE_FEATURES, DIFF_FEATURES
        # ``baseline_rank`` and ``is_baseline_top5`` are already members of
        # BASE_FEATURES.  The action builder's explicit assignments preserve
        # those same two prefixed names; appending them again here would create
        # a false 60-column expectation for the canonical 58-column model.
        expected_action_names = sorted(
            [f"{prefix}_{name}" for prefix in ("incoming", "dropped") for name in BASE_FEATURES]
            + [f"diff_{name}" for name in DIFF_FEATURES]
        )
        policy_names = list(policy.get("feature_names", []))
        v3a_names = list(v3a.get("feature_names", []))
        checks["train_public_schema_parity"] = (
            train_features.get("status") == "FROZEN_FEATURES_COMPLETE"
            and int(train_features.get("query_count", -1)) == 7000
            and train_features.get("all_chunk_logits_saved") is True
            and len(policy_names) == 58
            and len(v3a_names) == 58
            and policy_names == v3a_names == expected_action_names
            and v3a.get("model_sha256") == sha256(v3a_model_path)
        )
        replay = v3a.get("fold0_replay", {})
        checks["v3a_fold0_replay"] = replay.get("status") == "PASS" and int(replay.get("mismatch_count", -1)) == 0
    except (OSError, ValueError, TypeError):
        checks["train_public_schema_parity"] = checks["v3a_fold0_replay"] = False
    selector_source = Path(__file__).resolve().parents[1] / "task1_v2" / "evidence.py"
    checks["selector_identity"] = selector_source.is_file() and "select_true_s2_prepared" in selector_source.read_text(encoding="utf-8")
    checks["public_id_coverage"] = checks.get("union_contract", False) and checks.get("baseline_shape", False)
    checks["real_train_parity_pass"] = checks.get("train_public_schema_parity", False) and checks.get("v3a_fold0_replay", False)
    production = all(checks.values())
    return {"checks": checks, "missing": [], "production_adapter_verified": production}


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                yield value


def load_public_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        value = value.get("question_ids", value.get("ids"))
        if value is None:
            value = list(json.loads(path.read_text(encoding="utf-8-sig")))
    if not isinstance(value, list):
        raise ValueError("public ID manifest must be a list")
    ids = [str(item) for item in value]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("public ID manifest must contain 1000 unique IDs")
    return ids


def load_questions(path: Path, expected_ids: list[str]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("public questions must be an object keyed by question ID")
    if set(map(str, payload)) != set(expected_ids):
        raise ValueError("public question IDs do not match manifest")
    questions: dict[str, str] = {}
    for qid in expected_ids:
        row = payload[qid]
        if not isinstance(row, dict) or "question" not in row:
            raise ValueError(f"missing public question text: {qid}")
        # Intentionally never inspect row["answer"].  Public answers are null.
        questions[qid] = str(row["question"])
    return questions


def load_baseline(path: Path, expected_ids: list[str]) -> dict[str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError("baseline ZIP must contain exactly submission.json")
        payload = json.loads(archive.read("submission.json"))
    if set(map(str, payload)) != set(expected_ids):
        raise ValueError("baseline ZIP IDs do not match public manifest")
    result: dict[str, list[str]] = {}
    for qid in expected_ids:
        docs = payload[qid].get("answer") if isinstance(payload[qid], dict) else None
        if not isinstance(docs, list) or len(docs) != 5 or len(set(map(str, docs))) != 5:
            raise ValueError(f"baseline must contain five unique docs: {qid}")
        result[qid] = [str(doc) for doc in docs]
    return result


def source_rows(path: Path, expected_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in jsonl(path):
        qid = str(row["question_id"])
        if qid in rows:
            raise ValueError(f"duplicate public query ID: {qid} in {path}")
        rows[qid] = row
    if set(rows) != set(expected_ids) or len(rows) != 1000:
        raise ValueError(f"public source coverage mismatch: {path}")
    result: dict[str, list[dict[str, Any]]] = {}
    for qid in expected_ids:
        seen: set[str] = set()
        docs: list[dict[str, Any]] = []
        for rank, hit in enumerate(rows[qid].get("hits", []), 1):
            doc_id = str(hit.get("doc_id", ""))
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            docs.append({"doc_id": doc_id, "rank": rank})
        if not docs:
            raise ValueError(f"empty public candidate source: {qid}")
        result[qid] = docs
    return result


def load_candidate_union(path: Path, expected_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in jsonl(path):
        qid = str(row.get("query_id", ""))
        if not qid or qid in rows:
            raise ValueError(f"duplicate public union query ID: {qid}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates or len(candidates) > 200:
            raise ValueError(f"invalid public candidate union row: {qid}")
        normalized = []
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"invalid public candidate: {qid}")
            doc_id = str(candidate.get("doc_id", ""))
            if not doc_id or doc_id in seen:
                raise ValueError(f"duplicate/empty public candidate: {qid}")
            source_ranks = dict(candidate.get("source_ranks") or {})
            if set(source_ranks) - {"adaptive_k500", "bm25", "knn_word", "knn_char"}:
                raise ValueError(f"unexpected public source key: {qid}")
            seen.add(doc_id)
            normalized.append({**candidate, "doc_id": doc_id, "source_ranks": source_ranks})
        rows[qid] = normalized
    if set(rows) != set(expected_ids) or len(rows) != 1000:
        raise ValueError(f"public candidate union coverage mismatch: {path}")
    return rows


def rrf_union(sources: list[tuple[str, list[dict[str, Any]]]], k: int = 60, cap: int = 200) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, docs in sources:
        for rank, doc in enumerate(docs, 1):
            doc_id = str(doc["doc_id"])
            scores[doc_id] += 1.0 / (k + rank)
            ranks[doc_id][name] = rank
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:cap]
    return [
        {"doc_id": doc_id, "union_rank": rank, "source_support": len(ranks[doc_id]), "source_ranks": ranks[doc_id]}
        for rank, doc_id in enumerate(ordered, 1)
    ]


def make_shortlist(pool: list[dict[str, Any]], baseline: list[str]) -> list[dict[str, Any]]:
    by_id = {str(item["doc_id"]): dict(item) for item in pool}
    selected = {str(item["doc_id"]): dict(item) for item in pool if int(item["union_rank"]) <= 20}
    for doc_id in baseline:
        selected.setdefault(str(doc_id), by_id.get(str(doc_id), {
            "doc_id": str(doc_id), "union_rank": 10**9, "source_support": 0, "source_ranks": {}
        }))
    output = sorted(selected.values(), key=lambda item: (int(item["union_rank"]), str(item["doc_id"])))
    if not output or len(output) > 25 or len({str(item["doc_id"]) for item in output}) != len(output):
        raise ValueError("invalid public V3 shortlist")
    return output


def prepare_rows(questions: dict[str, str], ids: list[str], union_path: Path,
                 baseline_zip: Path, payloads: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    union_rows = load_candidate_union(union_path, ids)
    baseline = load_baseline(baseline_zip, ids)
    candidate_rows: list[dict[str, Any]] = []
    requested: set[str] = set()
    for qid in ids:
        candidate_rows.append({"query_id": qid, "candidates": union_rows[qid]})
        requested.update(str(item["doc_id"]) for item in union_rows[qid])

    chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in iter_payloads(payloads):
        doc_id = str(payload.get("doc_id", ""))
        if doc_id in requested:
            chunks[doc_id].append(payload)

    prepared = {doc_id: prepare_document(values) for doc_id, values in chunks.items()}
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        qid = candidate["query_id"]
        docs = make_shortlist(candidate["candidates"], baseline[qid])
        for doc in docs:
            evidence = select_true_s2_prepared(questions[qid], prepared.get(doc["doc_id"], prepare_document(())))
            if not evidence:
                raise ValueError(f"missing true-S2 evidence for {qid}:{doc['doc_id']}")
            doc["evidence"] = evidence
        row = {"query_id": qid, "question": questions[qid], "baseline_top5": baseline[qid], "docs": docs}
        if any(key in row or any(key in doc for doc in docs) for key in TRAIN_ONLY_FIELDS):
            raise AssertionError(f"train-only field leaked into public row: {qid}")
        rows.append(row)
    report = {
        "status": "CLEAN",
        "schema": "task1-v3-residual-public-shortlist-v1",
        "query_count": len(rows),
        "query_id_count": len({row["query_id"] for row in rows}),
        "candidate_union": {"sources": ["adaptive_k500", "bm25", "knn_word", "knn_char"], "rrf_k": 60, "cap": 200, "input": str(union_path)},
        "shortlist": {"union_rank_max": 20, "baseline_top5_union": True, "deduplicated": True, "max_docs": max(len(row["docs"]) for row in rows)},
        "fold_semantics": "-1 is public inference sentinel; no train fold or fold0 is used",
        "no_answers_read": True,
        "no_labels_or_gain": True,
        "missing_evidence_count": 0,
        "selector": {"identity": "scripts.beam.task1_v2.evidence.select_true_s2_prepared", "source": str(Path(__file__).resolve().parents[1] / "task1_v2" / "evidence.py"), "sha256": sha256(Path(__file__).resolve().parents[1] / "task1_v2" / "evidence.py")},
        "candidate_union_sha256": sha256(union_path),
        "baseline_sha256": sha256(baseline_zip),
        "payload_sha256": sha256(payloads),
    }
    return candidate_rows, rows, report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_test() -> dict[str, bool]:
    pool = rrf_union([("a", [{"doc_id": "B"}, {"doc_id": "A"}]), ("b", [{"doc_id": "A"}, {"doc_id": "C"}])])
    shortlist = make_shortlist(pool, ["Z", "Y", "X", "W", "V"])
    assert [x["doc_id"] for x in pool] == ["A", "B", "C"]
    assert [x["doc_id"] for x in shortlist] == ["A", "B", "C", "V", "W", "X", "Y", "Z"]
    assert len({x["doc_id"] for x in pool}) == len(pool)
    public_row = {"query_id": "q", "fold": -1, "question": "q", "baseline_top5": ["A"], "docs": shortlist}
    train_only_excluded = not any(key in json.dumps(public_row) for key in TRAIN_ONLY_FIELDS)
    try:
        make_shortlist([], [])
    except ValueError:
        empty_rejected = True
    else:
        empty_rejected = False
    duplicate_rejected = False
    try:
        load_public_ids_from_value(["1", "1"])
    except ValueError:
        duplicate_rejected = True
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    model_or_gpu_tokens_absent = not imported_modules.intersection({"torch", "transformers"})
    execution_blocked = False
    try:
        refuse_unverified_execution()
    except RuntimeError as exc:
        execution_blocked = "PUBLIC_PREP_BLOCKED_UNVERIFIED_CONTRACT" in str(exc)
    return {
        "candidate_dedup_order": [x["doc_id"] for x in pool] == ["A", "B", "C"] and len({x["doc_id"] for x in pool}) == 3,
        "empty_candidate_rejected": empty_rejected,
        "duplicate_query_id_rejected": duplicate_rejected,
        "train_only_fields_excluded": train_only_excluded,
        "fold0_not_referenced": public_row["fold"] == -1 and all(row.get("fold") != 0 for row in [public_row]),
        "no_model_or_gpu_path": model_or_gpu_tokens_absent,
        "public_null_answer_accepted_by_design": "answer" not in json.dumps(public_row),
        "unverified_execution_blocked": execution_blocked,
    }


def load_public_ids_from_value(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("IDs must be a list")
    ids = [str(item) for item in value]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query IDs")
    return ids


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--ids", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--union", type=Path, default=DEFAULT_UNION)
    p.add_argument("--union-manifest", type=Path, default=DEFAULT_UNION_MANIFEST)
    p.add_argument("--baseline-zip", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--adaptive-report", type=Path, default=DEFAULT_ADAPTIVE_REPORT)
    p.add_argument("--adaptive-output", type=Path, default=DEFAULT_ADAPTIVE_OUTPUT)
    p.add_argument("--adaptive-manifest", type=Path, default=DEFAULT_ADAPTIVE_MANIFEST)
    p.add_argument("--adaptive-model", type=Path, default=DEFAULT_ADAPTIVE_MODEL)
    p.add_argument("--v3a-manifest", type=Path, default=DEFAULT_V3A_MANIFEST)
    p.add_argument("--v3a-model", type=Path, default=DEFAULT_V3A_MODEL)
    p.add_argument("--train-feature-report", type=Path, default=DEFAULT_TRAIN_FEATURE_REPORT)
    p.add_argument("--policy-report", type=Path, default=DEFAULT_POLICY_REPORT)
    p.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    p.add_argument("--train-baseline", type=Path, default=DEFAULT_TRAIN_BASELINE)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--preflight", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()):
            raise AssertionError(checks)
        print(json.dumps({
            "status": "TOY_UNIT_TEST_PASS",
            "toy_unit_test_pass": True,
            "real_train_parity_pass": False,
            "production_adapter_verified": False,
            "public_v3a_feature_contract_reproducible": False,
            "parity_reason": "Toy tests do not substitute for the deterministic production contract check.",
            **checks,
            "gpu_launched": False,
        }, indent=2))
        return
    required = [args.questions, args.ids, args.union, args.union_manifest, args.baseline_zip, args.payloads]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing public preparation inputs:\n" + "\n".join(missing))
    ids = load_public_ids(args.ids)
    contract = verify_public_contract(
        args.questions, args.union, args.baseline_zip, args.payloads,
        args.adaptive_report, args.adaptive_output, args.adaptive_manifest,
        args.adaptive_model, args.v3a_manifest, args.v3a_model,
        args.train_feature_report, args.policy_report, args.baseline_manifest,
        args.train_baseline, args.union_manifest,
    )
    if args.preflight:
        load_questions(args.questions, ids)
        load_baseline(args.baseline_zip, ids)
        load_candidate_union(args.union, ids)
        print(json.dumps({
            "status": "PREFLIGHT_PASS" if contract["production_adapter_verified"] else "PREFLIGHT_BLOCKED_UNVERIFIED_CONTRACT",
            "query_count": 1000,
            "production_adapter_verified": contract["production_adapter_verified"],
            "public_v3a_feature_contract_reproducible": bool(contract["checks"].get("adaptive_feature_parity") and contract["checks"].get("train_public_schema_parity")),
            "real_train_parity_pass": contract["checks"].get("real_train_parity_pass", False),
            "remote_execution_allowed": bool(contract["production_adapter_verified"]),
            "contract_checks": contract["checks"],
            "missing": contract["missing"],
            "gpu_launched": False,
            "retrieval_rerun": False,
            "answers_read": False,
            "training": False,
        }, indent=2))
        return
    if not contract["production_adapter_verified"]:
        raise RuntimeError(json.dumps({"status": "PUBLIC_PREP_BLOCKED_UNVERIFIED_CONTRACT", **contract}, indent=2))
    questions = load_questions(args.questions, ids)
    candidate_rows, rows, report = prepare_rows(questions, ids, args.union, args.baseline_zip, args.payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    union_path = args.output_dir / "public_candidate_union.jsonl"
    shortlist_path = args.output_dir / "public_shortlist_evidence.jsonl"
    union_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_rows), encoding="utf-8")
    shortlist_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report["candidate_union_sha256"] = sha256(union_path)
    report["shortlist_evidence_sha256"] = sha256(shortlist_path)
    (args.output_dir / "public_shortlist_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PUBLIC_PREP_COMPLETE", "query_count": len(rows), "output_dir": str(args.output_dir), "gpu_launched": False}, indent=2))


if __name__ == "__main__":
    main()
