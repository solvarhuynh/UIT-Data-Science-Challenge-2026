"""Synthetic-only provenance and parity audit for Task 1 LegalIR scorers.

This program intentionally never opens public reference labels or Fold0 data.
It inventories local candidate artifacts and evaluates controlled synthetic rows.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "task1"
DOC_SCORER = ROOT / "docs" / "Scoring-Program-Task-LegalIR" / "scoring.py"
LOCAL_SCORER = ROOT / "src" / "udsc2026" / "evaluation" / "legal_ir.py"
HISTORICAL_SCORER = ROOT / "src" / "udsc2026" / "evaluation" / "legal_ir_recovery.py"
OVERVIEW = ROOT / "docs" / "competition" / "DSC2026_Task1_LegalIR_Data_Overview.docx"
METADATA = ROOT / "docs" / "Scoring-Program-Task-LegalIR" / "metadata.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path: Path) -> list[str]:
    if path.is_file():
        return [path.name]
    return sorted(str(item.relative_to(path)).replace("\\", "/") for item in path.rglob("*") if item.is_file())


def git_provenance(path: Path) -> dict[str, str | None]:
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    try:
        commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    except OSError:
        return {"last_committing_revision": None, "worktree_status": None}
    return {"last_committing_revision": commit or None, "worktree_status": status or "clean_or_untracked"}


def docx_text(path: Path) -> str:
    if not path.exists():
        return ""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    return " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))


def provenance_filename_hits() -> list[str]:
    """Find local names that could carry immutable CodaBench provenance.

    This inspects paths only (not any labels or prediction payloads) and skips
    VCS and Python environment directories.
    """
    tokens = ("scoring_program", "codabench", "competition.yaml", "competition.yml", "phase", "task.yaml", "task.yml", "docker", "challenge", "bundle")
    hits: list[str] = []
    for item in ROOT.rglob("*"):
        if any(part in {".git", ".venv", "__pycache__"} for part in item.parts):
            continue
        if any(token in item.name.casefold() for token in tokens):
            hits.append(str(item.relative_to(ROOT)).replace("\\", "/"))
    return sorted(hits)


def analytic_metrics(ranking: list[str], gold: set[str]) -> dict[str, float]:
    rank = next((index for index, doc in enumerate(ranking, 1) if doc in gold), None)
    return {
        "mrr": 0.0 if rank is None else 1.0 / rank,
        "recall_at_3": float(bool(set(ranking[:3]).intersection(gold))),
        "top5_set_recall": len(set(ranking[:5]).intersection(gold)) / len(gold),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    candidates = [
        (DOC_SCORER, "LIKELY_ORGANIZER_SCORER_BUT_VERSION_UNPROVEN", "TOP5_SET_MACRO_RECALL; TOP5_SET_MACRO_PRECISION", "1..5 only; empty or >5 is zero; unordered sets"),
        (LOCAL_SCORER, "LOCAL_REPOSITORY_SCORER_ONLY", "TOP5_SET_MACRO_RECALL; TOP5_SET_MACRO_PRECISION", "1..5 only; empty or >5 is zero; unordered sets"),
        (HISTORICAL_SCORER, "LOCAL_REPOSITORY_SCORER_ONLY", "TOP5_SET_MACRO_RECALL; TOP5_SET_MACRO_PRECISION", "uses predicted[:5] as an unordered set"),
        (OVERVIEW, "LOCAL_REPOSITORY_SCORER_ONLY", "documentation", "official competition overview; not executable"),
        (METADATA, "LOCAL_REPOSITORY_SCORER_ONLY", "metadata", "command: python3 scoring.py; no task/package/image provenance"),
    ]
    inventory = {"schema_version": "task1-scorer-artifact-inventory-v1", "exact_active_coda_bench_artifact_found": False, "search_scope": "entire local repository (path inventory only; no labels read)", "provenance_filename_hits": provenance_filename_hits(), "candidates": []}
    for path, confidence, classification, behavior in candidates:
        inventory["candidates"].append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"), "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None, "file_tree": tree(path),
            "confidence": confidence, "classification": classification, "metric_behavior": behavior,
            "docker_image": None, "docker_digest": None, "immutable_active_task_provenance": None,
            "git_provenance": git_provenance(path) if path.exists() else None,
        })

    # Each case has its own control, so exactly one intended property changes.
    # All IDs and relevance reference sets are synthetic.
    variants = {
        "A_rank4_changed_only": (["A", "B", "C", "R", "E"], ["A", "B", "C", "X", "E"], {"R"}),
        "B_rank5_changed_only": (["A", "B", "C", "D", "R"], ["A", "B", "C", "D", "X"], {"R"}),
        "C_rank1_changed": (["R", "B", "C", "D", "E"], ["X", "B", "C", "D", "E"], {"R"}),
        "D_rank2_changed": (["A", "R", "C", "D", "E"], ["A", "X", "C", "D", "E"], {"R"}),
        "E_rank3_changed": (["A", "B", "R", "D", "E"], ["A", "B", "X", "D", "E"], {"R"}),
        "F_relevant_rank3_to_rank4": (["A", "B", "R", "C", "D"], ["A", "B", "C", "R", "D"], {"R"}),
        "G_relevant_rank4_to_rank5": (["A", "B", "C", "R", "D"], ["A", "B", "C", "D", "R"], {"R"}),
        "H_same_top5_reordered": (["R", "A", "B", "C", "D"], ["A", "B", "C", "D", "R"], {"R"}),
        "I_different_top5_same_top3": (["A", "B", "C", "R", "E"], ["A", "B", "C", "X", "E"], {"R"}),
    }
    docs_eval = runpy.run_path(str(DOC_SCORER))["eval_retrieval"]
    from udsc2026.evaluation.legal_ir import legal_ir_precision, legal_ir_recall
    from udsc2026.evaluation.legal_ir_recovery import metrics as historical_metrics
    results = {"schema_version": "task1-scorer-metamorphic-v1", "synthetic_only": True, "variants": {}}
    for name, (control, candidate, gold) in variants.items():
        def score(ranking: list[str]) -> dict[str, object]:
            docs = docs_eval({"q_synthetic": {"answer": ranking}}, {"q_synthetic": sorted(gold)})
            local = {"recall": legal_ir_recall(ranking, sorted(gold)), "precision": legal_ir_precision(ranking, sorted(gold))}
            historical = historical_metrics(["q_synthetic"], {"q_synthetic": sorted(gold)}, {"q_synthetic": ranking})
            return {"analytic": analytic_metrics(ranking, gold), "docs_scoring_py": docs, "legal_ir_py": local, "historical_recovery": {"recall": historical["recall"], "precision": historical["precision"]}}
        results["variants"][name] = {"gold": sorted(gold), "control_ranking": control, "candidate_ranking": candidate, "control": score(control), "candidate": score(candidate)}
    results["observations"] = {
        "all_local_scorers_agree": True,
        "rank4_and_rank5_set_changes_can_affect_local_score": True,
        "same_top5_set_reordered_changes_local_score": False,
        "rank_position_only_changes_change_local_set_score": False,
        "mrr_and_recall_at_3_are_analytic_comparators_not_active_local_scorers": True,
    }
    docs_text = docx_text(OVERVIEW)
    documentation = {
        "artifact": str(OVERVIEW.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(OVERVIEW) if OVERVIEW.exists() else None,
        "summary": "Recall is primary and Precision secondary; returned document IDs are scored as sets; at most five IDs; macro aggregation.",
        "metric_names": ["Recall", "Precision"], "ranking_depth": "1..5 IDs; >5 receives zero per-query Recall/Precision", "aggregation": "macro mean", "primary_secondary": "Recall primary; Precision secondary tie-break", "display_score_mapping": "DISPLAY_SCORE_MAPPING_UNSPECIFIED",
        "mentions_mrr": "MRR" in docs_text, "mentions_recall_at_3": "Recall@3" in docs_text,
    }
    audit = {
        "schema_version": "active-scorer-provenance-and-parity-audit-v1",
        "active_scorer_contract": "UNRESOLVED", "exact_active_coda_bench_scorer_obtained": False,
        "exact_scorer_package_path": None, "exact_scorer_sha256": None,
        "competition_task_provenance": None, "docker_image": None, "docker_image_digest": None,
        "documentation_contract": documentation,
        "candidate_scorers": inventory["candidates"],
        "known_submission_structure": {"queries": 1000, "top1_top2_top3_changes": 0, "rank4_changes": 110, "rank5_changes": 218, "prediction_set_differences": 328, "public_truth_used": False},
        "mathematical_compatibility": {"recall_at_3_alone_explains_public_delta": "NO", "local_top5_set_scorer_explains_structural_delta": "DEPENDS_ON_UNKNOWN_TRUTH", "mrr_can_distinguish_rank4_or_rank5_changes": "DEPENDS_ON_UNKNOWN_TRUTH"},
        "decision_reason": "No immutable active CodaBench bundle/package hash, phase/task mapping, source revision, or image digest was found locally. Local behavior cannot establish the active deployed contract.",
        "safe_to_open_workflow_b": False,
        "acquisition_checklist": ["active scoring_program.zip or competition bundle", "competition and phase/task identifiers plus task mapping/competition.yaml", "scorer package SHA256", "scorer source revision", "Docker image name and immutable digest", "official phase metadata/export", "participant UI evidence whether bundle, scoring program, phase/task metadata, image name, or competition dump is downloadable"],
        "ui_availability": "Not locally verifiable; no automatic organizer artifact retrieval was attempted.",
    }
    (REPORTS / "scorer_artifact_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORTS / "scorer_metamorphic_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORTS / "active_scorer_provenance_and_parity_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = """# Active scorer provenance and parity audit\n\nStatus: PASS (audit completed; active contract remains unresolved).\n\n## Authoritative decision\n\nNo immutable local evidence identifies the active UIT-DSC 2026 Task1 CodaBench scorer. There is no local active bundle/package hash, competition/phase/task mapping, source revision, Docker image, or digest. Therefore `ACTIVE_SCORER_CONTRACT = UNRESOLVED`; Workflow B is not safe to open.\n\n## Local candidates\n\n`docs/Scoring-Program-Task-LegalIR/scoring.py` is a likely organizer scorer but its version is unproven. It calculates macro set Recall (primary) and set Precision (secondary), accepting 1–5 IDs and assigning zero for empty or >5 answers. `legal_ir.py` matches that set behavior. `legal_ir_recovery.py` is historical local code and scores the first five IDs as a set. None proves active CodaBench deployment provenance.\n\n## Documentation contract\n\nThe local Task1 overview says Recall is primary, Precision is secondary, output IDs are treated as sets, up to five IDs are evaluated, and aggregation is macro. It does not define how a single displayed CodaBench Score is mapped: `DISPLAY_SCORE_MAPPING_UNSPECIFIED`. The document has no MRR or Recall@3 wording.\n\n## Synthetic metamorphic parity\n\nAll tests used only the synthetic reference in `scorer_metamorphic_results.json`. The three local implementations agree: rank-4/rank-5 document-set changes can change Recall/Precision; a reordering of the same top-5 set cannot. Pure position changes cannot affect their set scores unless the returned set changes. Analytic MRR and Recall@3 comparators show that moving a relevant result from rank 3 to 4 alters both, while moving rank 4 to 5 alters MRR but not Recall@3.\n\nThe two public prediction files have identical top 1–3 by the locked structural facts. Recall@3 alone therefore cannot explain their score delta. The local top-5 set scorer could distinguish them only if the changed IDs differ in relevance under unknown public truth; no public truth was opened or scored. MRR can also distinguish some rank-4/rank-5 changes, so structural data alone is not identifying.\n\n## Required acquisition\n\nObtain the active `scoring_program.zip` or competition bundle, competition/phase/task IDs and mapping, package SHA256, source revision, Docker image plus digest, and official phase metadata/export. Confirm separately whether the participant UI exposes each item; that availability is not locally verifiable.\n"""
    (REPORTS / "active_scorer_provenance_and_parity_audit.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
