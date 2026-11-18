"""Estimate paired OOF uplift from validated LegalIR or LegalQA reports.

Bootstrap units are normalized-question groups. This is an internal decision
aid, not a calibrated probability of improving the private leaderboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from udsc2026.evaluation.legal_ir import (  # noqa: E402
    LegalIREvaluationReport,
    normalize_legal_ir_matching_question,
)
from udsc2026.evaluation.legal_qa import LegalQAEvaluationReport  # noqa: E402


def _read(path: Path) -> Any:
    """Load strict JSON from one input path."""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    """Fingerprint an input file for the audit report."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _questions(path: Path) -> dict[str, str]:
    """Read organizer question text without labels."""

    payload = _read(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty organizer mapping")
    result: dict[str, str] = {}
    for question_id, item in payload.items():
        if not isinstance(question_id, str) or not isinstance(item, dict):
            raise ValueError("invalid question record")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no text")
        result[question_id] = question
    return result


def _scores(path: Path, task: str) -> tuple[str, dict[str, float]]:
    """Validate one report and return per-query primary metric scores."""

    encoded = path.read_text(encoding="utf-8-sig")
    if task == "legal-ir":
        report = LegalIREvaluationReport.model_validate_json(encoded)
        return report.dataset_fingerprint, {
            item.id: item.recall for item in report.per_query
        }
    report = LegalQAEvaluationReport.model_validate_json(encoded)
    return report.dataset_fingerprint, {
        item.id: item.meteor.score for item in report.per_query
    }


def _quantile(values: list[float], fraction: float) -> float:
    """Return a deterministic linearly interpolated empirical quantile."""

    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def compare(
    baseline: Path,
    candidate: Path,
    questions: Path,
    *,
    task: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Compare matched query scores with a grouped paired bootstrap."""

    if draws < 100:
        raise ValueError("draws must be at least 100")
    base_fingerprint, base = _scores(baseline, task)
    candidate_fingerprint, new = _scores(candidate, task)
    if base_fingerprint != candidate_fingerprint:
        raise ValueError("reports use different reference datasets")
    text_by_id = _questions(questions)
    if set(base) != set(new) or set(base) != set(text_by_id):
        raise ValueError("report and question ID sets must match exactly")

    groups: dict[str, list[str]] = defaultdict(list)
    for question_id, question in text_by_id.items():
        group = normalize_legal_ir_matching_question(question)
        if not group:
            raise ValueError(f"empty normalized question: {question_id}")
        groups[group].append(question_id)
    units = [
        (sum(new[qid] - base[qid] for qid in ids), len(ids))
        for ids in groups.values()
    ]
    unit_count = len(units)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(draws):
        sample = [units[rng.randrange(unit_count)] for _ in range(unit_count)]
        deltas.append(sum(delta for delta, _ in sample) / sum(n for _, n in sample))
    deltas.sort()
    improvement_probability = sum(delta > 0 for delta in deltas) / draws
    mean_base = sum(base.values()) / len(base)
    mean_new = sum(new.values()) / len(new)
    minimum_groups = 100
    return {
        "task": task,
        "metric": "recall_at_5" if task == "legal-ir" else "local_meteor_diagnostic",
        "scope": "paired_grouped_bootstrap_internal",
        "private_leaderboard_probability": None,
        "reference_fingerprint": base_fingerprint,
        "input_sha256": {
            "baseline": _sha256(baseline),
            "candidate": _sha256(candidate),
            "questions": _sha256(questions),
        },
        "query_count": len(base),
        "group_count": unit_count,
        "minimum_groups_for_gate": minimum_groups,
        "draws": draws,
        "seed": seed,
        "baseline_mean": mean_base,
        "candidate_mean": mean_new,
        "observed_delta": mean_new - mean_base,
        "bootstrap_probability_delta_positive": improvement_probability,
        "bootstrap_delta_interval_95": [
            _quantile(deltas, 0.025),
            _quantile(deltas, 0.975),
        ],
        "passes_internal_70_percent_gate": (
            unit_count >= minimum_groups
            and mean_new > mean_base
            and improvement_probability >= 0.70
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report comparison CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("legal-ir", "legal-qa"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        report = compare(
            args.baseline,
            args.candidate,
            args.questions,
            task=args.task,
            draws=args.draws,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"comparison error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
