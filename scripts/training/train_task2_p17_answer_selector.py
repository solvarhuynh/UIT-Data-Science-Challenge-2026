"""Prepare, score, and select Task 2 answer candidates with a cross-encoder.

The script never exposes an evaluation reference to the model input.  Reference
scores are used only as listwise training targets, and the OOF command enforces
that every selected question was scored by a fold that did not train on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from udsc2026.evaluation.legal_qa_candidates import (  # noqa: E402
    build_answer_candidates,
    validate_candidate_bank,
)

SCHEMA_VERSION = "task2-p17-answer-selector-v1"
_SPACE_RE = re.compile(r"\s+")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--question-ids", type=Path, required=True)
    prepare.add_argument("--candidate-bank", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--folds", type=_positive_int, default=5)
    prepare.add_argument("--seed", type=int, default=2026)

    build_bank = commands.add_parser("build-bank")
    build_bank.add_argument("--questions", type=Path, required=True)
    build_bank.add_argument("--qwen", type=Path, required=True)
    build_bank.add_argument("--extractive", type=Path, required=True)
    build_bank.add_argument("--output", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--questions", type=Path, required=True)
    score.add_argument("--candidate-bank", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--question-ids", type=Path)
    score.add_argument("--fold", type=int)
    score.add_argument("--batch-size", type=_positive_int, default=128)
    score.add_argument("--max-length", type=_positive_int, default=256)
    score.add_argument("--device", default="cuda")
    score.add_argument("--amp", action="store_true")

    oof = commands.add_parser("select-oof")
    oof.add_argument("--candidate-bank", type=Path, required=True)
    oof.add_argument("--split-manifest", type=Path, required=True)
    oof.add_argument("--scores", type=Path, action="append", required=True)
    oof.add_argument("--output-dir", type=Path, required=True)

    public = commands.add_parser("select-public")
    public.add_argument("--candidate-bank", type=Path, required=True)
    public.add_argument("--scores", type=Path, required=True)
    public.add_argument("--selector-report", type=Path, required=True)
    public.add_argument("--output", type=Path, required=True)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"expected a non-empty object: {path}")
    return payload


def _read_ids(path: Path) -> list[str]:
    payload = _read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected a non-empty ID array: {path}")
    values = [str(value) for value in payload]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate question IDs: {path}")
    return values


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _atomic_write(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            if jsonl:
                for row in payload:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                    stream.write("\n")
            else:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _question_text(record: Any, question_id: str) -> str:
    value = record.get("question") if isinstance(record, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"question {question_id!r} has no text")
    return value.strip()


def _normalized_question(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _fold_for_question(question: str, folds: int, seed: int) -> int:
    normalized = _normalized_question(question)
    digest = hashlib.sha256(f"task2-p17|{seed}|{normalized}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def candidate_document(profile: str, answer: str) -> str:
    """Expose profile/length plus both answer ends within the 256-token limit."""

    words = answer.split()
    head = words[:80]
    tail = words[-64:] if len(words) > 80 else []
    profile_text = profile.replace("_", " ")
    parts = [
        f"Kiểu phương án: {profile_text}. Độ dài đầy đủ: {len(words)} từ.",
        "Phần đầu: " + " ".join(head),
    ]
    if tail:
        parts.append("Phần cuối: " + " ".join(tail))
    return "\n".join(parts)


def _validated_bank(path: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows = _read_jsonl(path)
    ids, profiles = validate_candidate_bank(rows)
    for index, row in enumerate(rows, 1):
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"candidate {index} has a blank answer")
    return rows, ids, profiles


def run_prepare(args: argparse.Namespace) -> list[Path]:
    if args.folds < 3:
        raise ValueError("folds must be at least 3")
    questions = _read_object(args.questions)
    requested_ids = _read_ids(args.question_ids)
    bank, bank_ids, profiles = _validated_bank(args.candidate_bank)
    if requested_ids != bank_ids:
        raise ValueError("question ID order and candidate-bank order differ")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bank:
        grouped[str(row["id"])].append(row)

    fold_by_id: dict[str, int] = {}
    prepared: list[dict[str, Any]] = []
    for question_id in requested_ids:
        question = _question_text(questions.get(question_id), question_id)
        fold_by_id[question_id] = _fold_for_question(
            question, args.folds, args.seed
        )
        candidates = grouped[question_id]
        for row in candidates:
            target = float(row.get("target", row.get("meteor", math.nan)))
            if not math.isfinite(target) or not 0 <= target <= 1:
                raise ValueError(f"invalid target for {question_id}/{row['profile']}")
        best = max(
            candidates,
            key=lambda row: (
                float(row.get("target", row.get("meteor", 0.0))),
                float(row.get("rouge_l", 0.0)),
                str(row["profile"]),
            ),
        )
        for row in candidates:
            profile = str(row["profile"])
            prepared.append(
                {
                    "schema_version": 1,
                    "question_id": question_id,
                    "question": question,
                    "doc_id": "task2-answer-candidates",
                    "parent_id": profile,
                    "parent_text": candidate_document(profile, str(row["answer"])),
                    "label": int(row is best),
                    "reference_overlap": float(
                        row.get("target", row.get("meteor", 0.0))
                    ),
                }
            )

    outputs: list[Path] = []
    all_path = args.output_dir / "all.jsonl"
    _atomic_write(all_path, prepared, jsonl=True)
    outputs.append(all_path)
    fold_counts: dict[str, int] = {}
    for fold in range(args.folds):
        train = [row for row in prepared if fold_by_id[row["question_id"]] != fold]
        valid = [row for row in prepared if fold_by_id[row["question_id"]] == fold]
        train_path = args.output_dir / f"fold_{fold}_train.jsonl"
        valid_path = args.output_dir / f"fold_{fold}_valid.jsonl"
        valid_ids_path = args.output_dir / f"fold_{fold}_valid_ids.json"
        _atomic_write(train_path, train, jsonl=True)
        _atomic_write(valid_path, valid, jsonl=True)
        valid_ids = list(
            dict.fromkeys(str(row["question_id"]) for row in valid)
        )
        _atomic_write(valid_ids_path, valid_ids)
        outputs.extend((train_path, valid_path, valid_ids_path))
        fold_counts[str(fold)] = len(valid_ids)
    if min(fold_counts.values()) < 1:
        raise ValueError(f"an empty fold was produced: {fold_counts}")
    manifest_path = args.output_dir / "split_manifest.json"
    _atomic_write(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "folds": args.folds,
            "seed": args.seed,
            "question_count": len(requested_ids),
            "candidate_count": len(bank),
            "profiles": profiles,
            "fold_counts": fold_counts,
            "fold_by_id": fold_by_id,
        },
    )
    outputs.append(manifest_path)
    print(
        json.dumps(
            {
                "event": "p17_prepare_complete",
                "question_count": len(requested_ids),
                "candidate_count": len(bank),
                "fold_counts": fold_counts,
            }
        )
    )
    return outputs


def _prediction_map(path: Path) -> dict[str, str]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"prediction file must be an array: {path}")
    mapped: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"invalid prediction row: {path}")
        question_id = str(row.get("id", ""))
        answer = row.get("answer")
        if not question_id or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"invalid prediction row: {path}")
        if question_id in mapped:
            raise ValueError(f"duplicate prediction ID {question_id!r}: {path}")
        mapped[question_id] = answer.strip()
    return mapped


def run_build_bank(args: argparse.Namespace) -> list[Path]:
    questions = _read_object(args.questions)
    qwen = _prediction_map(args.qwen)
    extractive = _prediction_map(args.extractive)
    if set(qwen) != set(questions) or set(extractive) != set(questions):
        raise ValueError("public question and prediction ID sets differ")
    rows: list[dict[str, str]] = []
    for question_id in questions:
        question = _question_text(questions[question_id], question_id)
        for candidate in build_answer_candidates(
            question, qwen[question_id], extractive[question_id]
        ):
            rows.append(
                {
                    "id": question_id,
                    "profile": candidate.profile,
                    "answer": candidate.answer,
                }
            )
    validate_candidate_bank(rows)
    _atomic_write(args.output, rows, jsonl=True)
    print(f"P17_PUBLIC_BANK_READY questions={len(questions)} candidates={len(rows)}")
    return [args.output]


def run_score(args: argparse.Namespace) -> list[Path]:
    if args.max_length > 256:
        raise ValueError("max-length exceeds the parent encoder contract")
    questions = _read_object(args.questions)
    bank, bank_ids, _ = _validated_bank(args.candidate_bank)
    requested = _read_ids(args.question_ids) if args.question_ids else bank_ids
    requested_set = set(requested)
    if not requested_set <= set(bank_ids):
        raise ValueError("score question IDs are absent from candidate bank")
    pairs: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for row in bank:
        question_id = str(row["id"])
        if question_id not in requested_set:
            continue
        question = _question_text(questions.get(question_id), question_id)
        profile = str(row["profile"])
        pairs.append(
            {
                "question": question,
                "parent_text": candidate_document(profile, str(row["answer"])),
            }
        )
        selected_rows.append(row)
    if len(pairs) != len(requested) * (len(bank) // len(bank_ids)):
        raise ValueError("score candidate matrix is incomplete")

    from scripts.training import train_legal_qa_parent_crossencoder as parent

    torch, tokenizer, model = parent._load_model_stack(args.checkpoint, args.device)
    device = next(model.parameters()).device
    amp_enabled = bool(args.amp and device.type == "cuda")
    scores = parent._score_pairs(
        torch,
        tokenizer,
        model,
        pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        amp_enabled=amp_enabled,
    )
    output_rows = [
        {
            "id": str(row["id"]),
            "profile": str(row["profile"]),
            "logit": float(score),
            **({"fold": args.fold} if args.fold is not None else {}),
        }
        for row, score in zip(selected_rows, scores)
    ]
    _atomic_write(args.output, output_rows, jsonl=True)
    print(f"P17_SCORE_COMPLETE candidates={len(output_rows)} output={args.output}")
    return [args.output]


def _score_index(paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            key = (str(row.get("id", "")), str(row.get("profile", "")))
            logit = float(row.get("logit", math.nan))
            if not all(key) or not math.isfinite(logit) or key in indexed:
                raise ValueError(f"invalid or duplicate score {key}: {path}")
            indexed[key] = row
    return indexed


def _zscore(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    if scale <= 1e-12:
        return [0.0] * len(values)
    return [(value - mean) / scale for value in values]


def _select(
    bank: Sequence[dict[str, Any]],
    scores: dict[tuple[str, str], dict[str, Any]],
    priors_by_fold: dict[int, dict[str, float]],
    fold_by_id: dict[str, int],
    alpha: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bank:
        grouped[str(row["id"])].append(row)
    selected: list[dict[str, Any]] = []
    for question_id, candidates in grouped.items():
        fold = fold_by_id[question_id]
        priors = priors_by_fold[fold]
        logits = [
            scores[(question_id, str(row["profile"]))]["logit"]
            for row in candidates
        ]
        normalized_logits = _zscore([float(value) for value in logits])
        normalized_priors = _zscore(
            [priors[str(row["profile"])] for row in candidates]
        )
        winner = max(
            range(len(candidates)),
            key=lambda index: (
                alpha * normalized_logits[index]
                + (1 - alpha) * normalized_priors[index],
                str(candidates[index]["profile"]),
            ),
        )
        row = candidates[winner]
        selected.append(
            {
                "id": question_id,
                "answer": str(row["answer"]),
                "profile": str(row["profile"]),
            }
        )
    return selected


def _candidate_metrics(
    predictions: Sequence[dict[str, Any]], bank: Sequence[dict[str, Any]]
) -> dict[str, float]:
    lookup = {
        (str(row["id"]), str(row["profile"])): row
        for row in bank
    }
    chosen = [lookup[(str(row["id"]), str(row["profile"]))] for row in predictions]
    return {
        "meteor": sum(float(row["meteor"]) for row in chosen) / len(chosen),
        "rouge_l": sum(float(row["rouge_l"]) for row in chosen) / len(chosen),
    }


def _profile_priors(
    bank: Sequence[dict[str, Any]],
    fold_by_id: dict[str, int],
    folds: int,
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    profiles = sorted({str(row["profile"]) for row in bank})
    for fold in range(folds):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in bank:
            if fold_by_id[str(row["id"])] != fold:
                grouped[str(row["profile"])].append(float(row["target"]))
        if set(grouped) != set(profiles):
            raise ValueError(f"profile priors incomplete for fold {fold}")
        result[fold] = {
            profile: sum(grouped[profile]) / len(grouped[profile])
            for profile in profiles
        }
    return result


def run_select_oof(args: argparse.Namespace) -> list[Path]:
    bank, bank_ids, _ = _validated_bank(args.candidate_bank)
    manifest = _read_object(args.split_manifest)
    fold_by_id = {str(key): int(value) for key, value in manifest["fold_by_id"].items()}
    if set(fold_by_id) != set(bank_ids):
        raise ValueError("split manifest and candidate bank IDs differ")
    folds = int(manifest["folds"])
    scores = _score_index(args.scores)
    expected = {(str(row["id"]), str(row["profile"])) for row in bank}
    if set(scores) != expected:
        raise ValueError("OOF scores do not cover the candidate matrix exactly")
    for (question_id, _), row in scores.items():
        if int(row.get("fold", -1)) != fold_by_id[question_id]:
            raise ValueError(f"score leakage or wrong fold for {question_id}")
    priors = _profile_priors(bank, fold_by_id, folds)
    trials: list[dict[str, Any]] = []
    for step in range(21):
        alpha = step / 20
        predictions = _select(bank, scores, priors, fold_by_id, alpha)
        trials.append(
            {"alpha": alpha, "metrics": _candidate_metrics(predictions, bank)}
        )
    best = max(
        trials,
        key=lambda row: (
            float(row["metrics"]["meteor"]),
            float(row["metrics"]["rouge_l"]),
            -abs(float(row["alpha"]) - 0.5),
        ),
    )
    predictions = _select(bank, scores, priors, fold_by_id, float(best["alpha"]))
    fold_metrics: dict[str, dict[str, float]] = {}
    for fold in range(folds):
        subset = [row for row in predictions if fold_by_id[str(row["id"])] == fold]
        fold_metrics[str(fold)] = _candidate_metrics(subset, bank)
    clean_predictions = [
        {"id": row["id"], "answer": row["answer"]} for row in predictions
    ]
    predictions_path = args.output_dir / "oof_predictions.json"
    report_path = args.output_dir / "selector_report.json"
    full_profile_values: dict[str, list[float]] = defaultdict(list)
    for row in bank:
        full_profile_values[str(row["profile"])].append(float(row["target"]))
    full_profile_priors = {
        profile: sum(values) / len(values)
        for profile, values in sorted(full_profile_values.items())
    }
    _atomic_write(predictions_path, clean_predictions)
    _atomic_write(
        report_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "OOF_COMPLETE",
            "question_count": len(bank_ids),
            "selected_alpha": best["alpha"],
            "oof_fast_metrics": best["metrics"],
            "fold_metrics": fold_metrics,
            "full_profile_priors": full_profile_priors,
            "alpha_trials": trials,
        },
    )
    print(json.dumps(_read_json(report_path), ensure_ascii=False, sort_keys=True))
    return [predictions_path, report_path]


def run_select_public(args: argparse.Namespace) -> list[Path]:
    bank, bank_ids, _ = _validated_bank(args.candidate_bank)
    scores = _score_index([args.scores])
    expected = {(str(row["id"]), str(row["profile"])) for row in bank}
    if set(scores) != expected:
        raise ValueError("public scores do not cover the candidate matrix exactly")
    report = _read_object(args.selector_report)
    alpha = float(report["selected_alpha"])
    # Fold -1 means a model fitted on all organizer-labelled questions.  The
    # priors stored in the report are not needed: public profile priors are
    # estimated from the complete labelled bank before this command is called.
    priors = report.get("full_profile_priors")
    if not isinstance(priors, dict) or not priors:
        raise ValueError("selector report lacks full_profile_priors")
    fold_by_id = {question_id: -1 for question_id in bank_ids}
    selected = _select(
        bank,
        scores,
        {-1: {str(key): float(value) for key, value in priors.items()}},
        fold_by_id,
        alpha,
    )
    _atomic_write(
        args.output,
        [{"id": row["id"], "answer": row["answer"]} for row in selected],
    )
    print(f"P17_PUBLIC_SELECTION_READY questions={len(selected)} alpha={alpha}")
    return [args.output]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runners = {
        "prepare": run_prepare,
        "build-bank": run_build_bank,
        "score": run_score,
        "select-oof": run_select_oof,
        "select-public": run_select_public,
    }
    try:
        runners[args.command](args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Task 2 P17 selector error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
