"""Finalize the fixed V2B Slim choice from a completed no-GPU preflight report.

This does not recompute a frontier or inspect labels.  It only applies the
already-declared cost budget rule to the immutable frontier and selector audit.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from analyze_post_k500_bounded_union_v2 import RECOVERY


def atomic_write(path: Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RECOVERY / "v2b_slim_preflight")
    args = parser.parse_args()
    path = args.output_dir / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    chosen_name = "union_rank_le_50"
    chosen = report["frontier"][chosen_name]
    evidence = report["evidence_selector_sanity"]
    selector_ok = min(evidence["k200"]["any_overlap_at3_rate"] or 0, evidence["k500_v1"]["any_overlap_at3_rate"] or 0) >= 0.20
    cost_ok = chosen["relative_cost_vs_k500_v1_at_3"] <= 1.5
    status = "V2B_SLIM_READY_FOR_GPU" if selector_ok and cost_ok else ("EVIDENCE_SELECTOR_MISMATCH_BLOCKS_GPU" if not selector_ok else "V2B_SLIM_NOT_WORTH_GPU")
    report["status"] = status
    report["recommended_plan"] = {
        "policy_name": chosen_name,
        "selection_basis": "fixed union-rank cap=50; deepest hand-specified rank frontier under 1.0x K500 V1 cost, independent of gold",
        **chosen,
        "evidence_selector": "token_overlap_topk_v1",
        "expected_output_paths": {
            "beam_output_dir": "/workspace/p13/runtime/artifacts/task1/recovery_096/selective_bge_bounded_union_v2b_slim",
            "local_download_dir": "artifacts/task1/recovery_096/selective_bge_bounded_union_v2b_slim_from_beam",
        },
    }
    report["decision_rule"] = {"selector_any_overlap_at3_minimum": 0.20, "cost_maximum_relative_to_k500_v1": 1.5}
    atomic_write(path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# V2B Slim preflight", "", f"Status: `{status}`", "", "| Policy | docs | chunks@3 | cost vs K500 | rescued-gold capture |", "|---|---:|---:|---:|---:|"]
    for name, value in report["frontier"].items():
        diagnostic = value["missing_bge_rescued_gold_diagnostic"]
        lines.append(f"| {name} | {value['candidate_doc_occurrences']:,} | {value['chunks_to_score_max_chunks_per_doc_3']:,} | {value['relative_cost_vs_k500_v1_at_3']:.3f}x | {diagnostic['kept']}/{diagnostic['fixed_missing_bge_rescued_gold_cohort']} ({diagnostic['capture_rate']:.3f}) |")
    lines.extend(["", "## Selector sanity", "", json.dumps(evidence, ensure_ascii=False, indent=2), ""])
    atomic_write(args.output_dir / "report.md", "\n".join(lines))
    print(json.dumps({"status": status, "policy": chosen_name, "chunks": chosen["chunks_to_score_max_chunks_per_doc_3"], "cost": chosen["relative_cost_vs_k500_v1_at_3"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
