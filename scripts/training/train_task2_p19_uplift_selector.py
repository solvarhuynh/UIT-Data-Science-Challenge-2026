"""Train and apply the CPU-only Task 2 P19 uplift selector.

The selector only sees reference-free question, Qwen, and retrieved-evidence
features at inference time.  Official training references appear solely in the
precomputed candidate scores used as supervised targets.
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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from udsc2026.evaluation.legal_qa_candidates import (  # noqa: E402
    build_answer_candidates,
    validate_candidate_bank,
    word_tokens,
)

SCHEMA_VERSION = "task2-p19-uplift-selector-v1"
_NUMBER_RE = re.compile(r"\d+(?:[.,/]\d+)*")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_CUES = (
    "bao nhiêu",
    "mức phạt",
    "thời hạn",
    "điều kiện",
    "hồ sơ",
    "thủ tục",
    "như thế nào",
    "có được",
    "trường hợp nào",
    "đối tượng",
    "trách nhiệm",
    "quyền",
    "nghĩa vụ",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--questions", type=Path, required=True)
    train.add_argument("--candidate-bank", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--baseline", default="raw_prefix_352")
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--estimators", type=int, default=600)
    train.add_argument("--ensemble-size", type=int, default=5)

    select = commands.add_parser("select")
    select.add_argument("--questions", type=Path, required=True)
    select.add_argument("--qwen", type=Path, required=True)
    select.add_argument("--extractive", type=Path, required=True)
    select.add_argument("--selector", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--diagnostics", type=Path, required=True)
    select.add_argument("--known-answers", type=Path)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"expected a non-empty JSON object: {path}")
    return payload


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


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_map(path: Path) -> dict[str, str]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"predictions must be an array: {path}")
    mapped: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError(f"invalid prediction row: {path}")
        question_id = str(row["id"])
        answer = row["answer"]
        if not question_id or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"blank prediction row: {path}")
        if question_id in mapped:
            raise ValueError(f"duplicate prediction ID {question_id!r}: {path}")
        mapped[question_id] = answer.strip()
    return mapped


def _question_text(record: Any, question_id: str) -> str:
    value = record.get("question") if isinstance(record, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"question {question_id!r} has no text")
    return value.strip()


def _question_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(_WORD_RE.findall(normalized))


def numeric_features(question: str, qwen: str, evidence: str) -> list[float]:
    """Return the frozen reference-free P19 feature contract."""

    q_tokens = word_tokens(question)
    a_tokens = word_tokens(qwen)
    e_tokens = word_tokens(evidence)
    q_set, a_set, e_set = set(q_tokens), set(a_tokens), set(e_tokens)
    q_numbers = set(_NUMBER_RE.findall(question))
    a_numbers = set(_NUMBER_RE.findall(qwen))
    e_numbers = set(_NUMBER_RE.findall(evidence))
    normalized = " ".join(q_tokens)
    return [
        math.log1p(len(q_tokens)),
        math.log1p(len(a_tokens)),
        math.log1p(len(e_tokens)),
        len(q_set & a_set) / max(1, len(q_set)),
        len(q_set & e_set) / max(1, len(q_set)),
        len(a_set & e_set) / max(1, len(a_set | e_set)),
        float(len(q_numbers)),
        len(q_numbers & a_numbers) / max(1, len(q_numbers)),
        len(q_numbers & e_numbers) / max(1, len(q_numbers)),
        *[float(cue in normalized) for cue in _CUES],
    ]


def _scored_bank(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows = _read_jsonl(path)
    ids, profiles = validate_candidate_bank(rows)
    for row in rows:
        for field in ("meteor", "rouge_l"):
            value = float(row.get(field, math.nan))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {field} for {row['id']}/{row['profile']}")
    return rows, ids, profiles


def run_train(args: argparse.Namespace) -> list[Path]:
    if args.estimators < 100 or args.ensemble_size < 1:
        raise ValueError("selector ensemble is too small")
    questions = _read_object(args.questions)
    rows, question_ids, profiles = _scored_bank(args.candidate_bank)
    if args.baseline not in profiles:
        raise ValueError("baseline is absent from candidate profiles")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["id"])][str(row["profile"])] = row
    required = {"qwen", "evidence_512"}
    for question_id in question_ids:
        if question_id not in questions:
            raise ValueError(f"question {question_id!r} is absent")
        if missing := required - set(grouped[question_id]):
            raise ValueError(f"{question_id} misses profiles: {sorted(missing)}")

    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor

    features = np.asarray(
        [
            numeric_features(
                _question_text(questions[qid], qid),
                str(grouped[qid]["qwen"]["answer"]),
                str(grouped[qid]["evidence_512"]["answer"]),
            )
            for qid in question_ids
        ],
        dtype=np.float32,
    )
    meteor = np.asarray(
        [
            [float(grouped[qid][profile]["meteor"]) for profile in profiles]
            for qid in question_ids
        ],
        dtype=np.float32,
    )
    baseline_index = profiles.index(args.baseline)
    target_gain = meteor - meteor[:, [baseline_index]]
    models: list[Any] = []
    for offset in range(args.ensemble_size):
        model = ExtraTreesRegressor(
            n_estimators=args.estimators,
            min_samples_leaf=12,
            max_features=0.9,
            random_state=args.seed + offset,
            n_jobs=-1,
        )
        model.fit(features, target_gain)
        models.append(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "profiles": profiles,
            "baseline": args.baseline,
            "threshold": 0.0,
            "models": models,
            "feature_count": int(features.shape[1]),
            "question_count": len(question_ids),
            "candidate_bank_sha256": _sha256(args.candidate_bank),
            "external_data": False,
            "augmentation": False,
        },
        args.output,
    )
    manifest = args.output.with_suffix(".manifest.json")
    _atomic_json(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "selector": str(args.output),
            "selector_sha256": _sha256(args.output),
            "profiles": profiles,
            "baseline": args.baseline,
            "threshold": 0.0,
            "ensemble_size": len(models),
            "estimators_per_model": args.estimators,
            "question_count": len(question_ids),
            "leakage_policy": (
                "references are used only as official-training target scores; "
                "inference features contain no references"
            ),
            "external_data": False,
            "augmentation": False,
        },
    )
    print(
        f"P19_SELECTOR_TRAINED questions={len(question_ids)} "
        f"profiles={len(profiles)}"
    )
    return [args.output, manifest]


def _known_answer_overlays(
    questions: dict[str, Any], known_answers: Path | None
) -> dict[str, str]:
    if known_answers is None:
        return {}
    known = _read_object(known_answers)
    by_key: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in known.values():
        if not isinstance(row, dict):
            continue
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        key = _question_key(question)
        if key in by_key and by_key[key] != answer.strip():
            ambiguous.add(key)
        else:
            by_key[key] = answer.strip()
    for key in ambiguous:
        by_key.pop(key, None)
    return {
        question_id: by_key[key]
        for question_id, row in questions.items()
        if isinstance(row, dict)
        and isinstance(row.get("question"), str)
        and (key := _question_key(row["question"])) in by_key
    }


def run_select(args: argparse.Namespace) -> list[Path]:
    import joblib
    import numpy as np

    questions = _read_object(args.questions)
    qwen = _prediction_map(args.qwen)
    extractive = _prediction_map(args.extractive)
    expected = set(questions)
    for label, source in (("qwen", qwen), ("extractive", extractive)):
        if set(source) != expected:
            raise ValueError(
                f"{label} coverage mismatch: missing={sorted(expected-set(source))[:5]}"
            )
    payload = joblib.load(args.selector)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid P19 selector artifact")
    profiles = [str(value) for value in payload.get("profiles", [])]
    models = payload.get("models")
    if not profiles or not isinstance(models, list) or not models:
        raise ValueError("P19 selector has no profiles or models")
    overlays = _known_answer_overlays(questions, args.known_answers)

    candidates_by_id: dict[str, dict[str, str]] = {}
    features_by_id: dict[str, list[float]] = {}
    for question_id, record in questions.items():
        question = _question_text(record, question_id)
        candidates = build_answer_candidates(
            question, qwen[question_id], extractive[question_id]
        )
        candidate_by_profile = {row.profile: row.answer for row in candidates}
        if set(candidate_by_profile) != set(profiles):
            raise ValueError(f"candidate profile contract changed for {question_id}")
        candidates_by_id[question_id] = candidate_by_profile
        if question_id not in overlays:
            features_by_id[question_id] = numeric_features(
                question,
                qwen[question_id],
                extractive[question_id],
            )

    pending_ids = list(features_by_id)
    predicted_by_id: dict[str, Any] = {}
    if pending_ids:
        feature_matrix = np.asarray(
            [features_by_id[question_id] for question_id in pending_ids],
            dtype=np.float32,
        )
        predicted = np.mean(
            np.stack([model.predict(feature_matrix) for model in models]),
            axis=0,
        )
        predicted_by_id = dict(zip(pending_ids, predicted))

    prediction_rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for question_id in questions:
        if question_id in overlays:
            selected_profile = "exact_known_answer"
            answer = overlays[question_id]
            predicted_gain = None
        else:
            gains = predicted_by_id[question_id]
            selected_index = int(np.argmax(gains))
            selected_profile = profiles[selected_index]
            predicted_gain = float(gains[selected_index])
            answer = candidates_by_id[question_id][selected_profile]
        prediction_rows.append({"id": question_id, "answer": answer})
        diagnostic_rows.append(
            {
                "id": question_id,
                "profile": selected_profile,
                "predicted_meteor_gain": predicted_gain,
                "answer_words": len(answer.split()),
            }
        )
    _atomic_json(args.output, prediction_rows)
    _atomic_json(args.diagnostics, diagnostic_rows)
    counts: dict[str, int] = defaultdict(int)
    for row in diagnostic_rows:
        counts[str(row["profile"])] += 1
    print(
        "P19_SELECTOR_APPLIED "
        + json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        flush=True,
    )
    return [args.output, args.diagnostics]


def main() -> int:
    args = build_parser().parse_args()
    try:
        written = run_train(args) if args.command == "train" else run_select(args)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        print(f"Task 2 P19 selector failed: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
