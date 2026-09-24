"""Recover the missing Team 09205 V2 anchor from frozen 4-source RRF inputs.

This is deliberately CPU-only.  It calls the canonical private RRF builder's
input verifier and ``rank_query`` helper, and never opens train/warmup labels.
The two supplied Team 09205 submissions are independent behavioural checksums;
the recovered anchor is promoted only when both checksums have identical Top5
membership.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
TEAM_DIR = ROOT / "private_task1" / "submissions" / "team_09205"
ANCHOR_DIR = TEAM_DIR / "anchors"
CANDIDATE = ANCHOR_DIR / "submission_v2_reconstructed_candidate.zip"
RECOVERED = ANCHOR_DIR / "submission_v2.zip"
MANIFEST = ANCHOR_DIR / "submission_v2_manifest.json"
REPORT = ROOT / "private_task1" / "reports" / "team_09205" / "v2_anchor_recovery_audit.json"
GUARDED = (
    ROOT
    / "private_task1"
    / "submissions"
    / "sprint48_guarded_direct"
    / "submission_private_guarded_direct.zip"
)
SAFEGUARD = TEAM_DIR / "submission_private_dual_anchor_guarded_09198.zip"
RRF_09205 = TEAM_DIR / "submission_private_constrained_dual_anchor_rrf_09205.zip"
RRF_09205_EXPECTED_SHA = "aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae"
CANONICAL_BUILDER = ROOT / "scripts" / "analysis" / "finalize_private_rrf_k20.py"
RRF_HANDOFF_BUILDER = (
    ROOT
    / "private_task1"
    / "scripts"
    / "analysis"
    / "team_09205"
    / "build_constrained_dual_anchor_rrf_submission.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_submission(path: Path) -> dict[str, dict[str, list[str]]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise RuntimeError(f"expected one JSON member in {path}, got {members}")
        payload = json.loads(archive.read(members[0]).decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"non-object submission payload: {path}")
    result: dict[str, dict[str, list[str]]] = {}
    for raw_qid, row in payload.items():
        answer = row.get("answer") if isinstance(row, dict) else None
        if not isinstance(answer, list) or len(answer) != 5:
            raise RuntimeError(f"invalid Top5 for {raw_qid} in {path}")
        documents = [str(value) for value in answer]
        if any(not value for value in documents) or len(documents) != len(set(documents)):
            raise RuntimeError(f"duplicate/null Top5 document for {raw_qid} in {path}")
        result[str(raw_qid)] = {"answer": documents}
    return result


def compare(left: dict[str, dict[str, list[str]]], right: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError(
            f"query universe mismatch: missing={len(set(right) - set(left))} extra={len(set(left) - set(right))}"
        )
    order_mismatches = [qid for qid in sorted(left, key=numeric_id) if left[qid]["answer"] != right[qid]["answer"]]
    membership_mismatches = [
        qid
        for qid in sorted(left, key=numeric_id)
        if set(left[qid]["answer"]) != set(right[qid]["answer"])
    ]
    return {
        "query_count": len(left),
        "exact_order_match_queries": len(left) - len(order_mismatches),
        "exact_top5_set_match_queries": len(left) - len(membership_mismatches),
        "order_mismatches": len(order_mismatches),
        "membership_mismatches": len(membership_mismatches),
        "first_20_order_mismatch_query_ids": order_mismatches[:20],
        "first_20_membership_mismatch_query_ids": membership_mismatches[:20],
        "canonical_json_payload_equal": left == right,
    }


def numeric_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def write_zip(payload: dict[str, dict[str, list[str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", encoded)


def structural_check(
    payload: dict[str, dict[str, list[str]]], expected_ids: set[str]
) -> dict[str, int]:
    missing = expected_ids - set(payload)
    extra = set(payload) - expected_ids
    duplicates = 0
    nulls = 0
    bad_length = 0
    for row in payload.values():
        answer = row["answer"]
        bad_length += int(len(answer) != 5)
        duplicates += len(answer) - len(set(answer))
        nulls += sum(not document for document in answer)
    return {
        "queries": len(payload),
        "missing": len(missing),
        "extra": len(extra),
        "duplicate_documents": duplicates,
        "null_documents": nulls,
        "bad_length_queries": bad_length,
    }


def write_report(report: dict[str, Any]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    for required in (GUARDED, SAFEGUARD, RRF_09205, CANONICAL_BUILDER, RRF_HANDOFF_BUILDER):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256(RRF_09205) != RRF_09205_EXPECTED_SHA:
        raise RuntimeError("09205 incumbent SHA mismatch")

    # This exact V2 artifact was absent from the supplied Team 09205 handoff.
    original_handoff_v2_file_present = RECOVERED.exists()
    canonical = load_module("team09205_canonical_rrf", CANONICAL_BUILDER)
    handoff = load_module("team09205_handoff_rrf", RRF_HANDOFF_BUILDER)

    # Reuse, rather than reimplement, the frozen production verifier and RRF
    # ranker. verify_inputs asserts the private answer fields are null.
    (
        production_manifest,
        score_by_key,
        _work_by_key,
        work_by_query,
        _corpus_ids,
        _addendum,
    ) = canonical.verify_inputs()
    private_ids = set(work_by_query)
    predictions: dict[str, dict[str, list[str]]] = {}
    for qid in sorted(work_by_query, key=canonical.numeric_id):
        ranked = canonical.rank_query(work_by_query[qid], score_by_key)
        predictions[qid] = {"answer": [document_id for document_id, _ in ranked[:5]]}
    structure = structural_check(predictions, private_ids)
    if structure != {
        "queries": 2080,
        "missing": 0,
        "extra": 0,
        "duplicate_documents": 0,
        "null_documents": 0,
        "bad_length_queries": 0,
    }:
        raise RuntimeError(f"canonical V2 structural gate failed: {structure}")
    write_zip(predictions, CANDIDATE)

    guarded = load_submission(GUARDED)
    candidate = load_submission(CANDIDATE)
    supplied_safeguard = load_submission(SAFEGUARD)
    supplied_rrf = load_submission(RRF_09205)
    if set(guarded) != private_ids:
        raise RuntimeError("guarded private query set mismatch")

    # Exact handoff safeguard: ranks 1-4 are immutable; only slot 5 can change.
    reconstructed_safeguard: dict[str, dict[str, list[str]]] = {}
    safeguard_injections = 0
    for qid in sorted(guarded, key=numeric_id):
        answer = list(guarded[qid]["answer"])
        v2_rank1 = candidate[qid]["answer"][0]
        if v2_rank1 not in answer:
            answer[4] = v2_rank1
            safeguard_injections += 1
        reconstructed_safeguard[qid] = {"answer": answer}
    safeguard_comparison = compare(reconstructed_safeguard, supplied_safeguard)

    report: dict[str, Any] = {
        "status": "RECOVERY_FAIL_SAFEGUARD_CHECKSUM",
        "original_handoff_v2_file_present_before_recovery": original_handoff_v2_file_present,
        "private_09168_reproduction_lineage": {
            "status": "PASS",
            "generator": str(CANONICAL_BUILDER.relative_to(ROOT)).replace("\\", "/"),
            "rank_helper": "finalize_private_rrf_k20.py::verify_inputs + rank_query",
            "formula": "sum(weight/(2+rank)); dense=.2,bge=.3,knn_word=.2,bm25=.3",
            "tie_break": "weighted RRF desc; dense/bge/knn_word/bm25 ranks; numeric document_id",
            "worklist": "private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl",
            "worklist_sha256": production_manifest["worklist_sha256"],
            "scores": "private_task1/experiments/private_rrf_k20/gpu_download/results/production/scores.jsonl",
            "scores_sha256": production_manifest["output_sha256"],
        },
        "reconstructed_v2": {
            "path": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(CANDIDATE),
            "structural_check": structure,
        },
        "safeguard_09198": {
            "injections": safeguard_injections,
            "comparison": safeguard_comparison,
        },
        "rrf_09205": {"comparison": None},
        "private_oracle_matching_used": False,
        "warmup_reads": 0,
        "train_answer_reads_for_private": 0,
        "private_labels_used": False,
    }
    if safeguard_comparison["membership_mismatches"] != 0:
        write_report(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    # Reuse the exact supplied constrained-RRF implementation and its ordering.
    reconstructed_rrf, guarded_r1_replaced, v2_r1_injected = handoff.build(guarded, candidate)
    rrf_comparison = compare(reconstructed_rrf, supplied_rrf)
    report["rrf_09205"] = {
        "guarded_r1_replaced": guarded_r1_replaced,
        "v2_r1_injected": v2_r1_injected,
        "comparison": rrf_comparison,
    }
    if rrf_comparison["membership_mismatches"] != 0:
        report["status"] = "RECOVERY_FAIL_RRF_CHECKSUM"
        write_report(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3

    # Both observed handoff artifacts independently validate functional identity.
    shutil.copyfile(CANDIDATE, RECOVERED)
    report["status"] = "V2_ANCHOR_LINEAGE_GATE_PASS"
    report["recovered_v2"] = {
        "path": str(RECOVERED.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(RECOVERED),
        "source_type": "RECONSTRUCTED_FROM_CANONICAL_4_SOURCE_RRF",
    }
    write_report(report)
    manifest = {
        "status": "V2_ANCHOR_LINEAGE_GATE_PASS",
        "source_type": "RECONSTRUCTED_FROM_CANONICAL_4_SOURCE_RRF",
        "formula": report["private_09168_reproduction_lineage"]["formula"],
        "tie_break_implementation_source": report["private_09168_reproduction_lineage"]["generator"],
        "canonical_source_artifact_hashes": {
            "worklist_sha256": production_manifest["worklist_sha256"],
            "scores_sha256": production_manifest["output_sha256"],
        },
        "structural_check": structure,
        "safeguard_checksum": safeguard_comparison,
        "rrf_09205_checksum": rrf_comparison,
        "reconstructed_v2_zip_sha256": sha256(RECOVERED),
        "warmup_reads": 0,
        "train_answer_reads_for_private": 0,
        "private_oracle_matching": 0,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
