"""Evaluate conservative, leakage-safe Task 2 answer-profile selectors.

This is a CPU-only research gate.  It consumes the already-scored P15
candidate bank, but every prediction for a question is produced by a fold
that did not train on that question.  Models predict candidate uplift over a
fixed baseline instead of absolute score; a non-positive prediction keeps the
baseline and avoids unnecessary profile switching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for _root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from udsc2026.evaluation.legal_qa_candidates import word_tokens  # noqa: E402

DEFAULT_PROFILES = (
    "qwen",
    "qwen_trim_256",
    "qwen_trim_320",
    "qwen_trim_384",
    "qwen_trim_448",
    "qwen_trim_512",
    "smart_prefix_48",
    "smart_suffix_48",
    "smart_prefix_80",
    "smart_suffix_80",
    "smart_prefix_112",
    "smart_suffix_112",
    "smart_prefix_160",
    "smart_suffix_160",
    "smart_prefix_224",
    "smart_suffix_224",
    "raw_prefix_96",
    "raw_prefix_160",
    "raw_prefix_224",
    "raw_prefix_288",
    "raw_prefix_352",
    "raw_prefix_416",
    "evidence_256",
    "evidence_384",
    "evidence_512",
)
_NUMBER_RE = re.compile(r"\d+(?:[.,/]\d+)*")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="raw_prefix_352")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profiles", nargs="*", default=list(DEFAULT_PROFILES))
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("candidate bank is empty")
    return rows


def _fold(question_id: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"task2-p19|{seed}|{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _document(question: str, qwen: str, evidence: str) -> str:
    return (
        f"QUESTION {question.strip()}\n"
        f"QWEN {' '.join(qwen.split()[:192])}\n"
        f"EVIDENCE {' '.join(evidence.split()[:224])}"
    )


def _numeric_features(question: str, qwen: str, evidence: str) -> list[float]:
    q_tokens = word_tokens(question)
    a_tokens = word_tokens(qwen)
    e_tokens = word_tokens(evidence)
    q_set, a_set, e_set = set(q_tokens), set(a_tokens), set(e_tokens)
    q_numbers = set(_NUMBER_RE.findall(question))
    a_numbers = set(_NUMBER_RE.findall(qwen))
    e_numbers = set(_NUMBER_RE.findall(evidence))
    normalized = " ".join(q_tokens)
    cues = (
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
    return [
        math.log1p(len(q_tokens)),
        math.log1p(len(a_tokens)),
        math.log1p(len(e_tokens)),
        len(q_set & a_set) / max(1, len(q_set)),
        len(q_set & e_set) / max(1, len(q_set)),
        len(a_set & e_set) / max(1, len(a_set | e_set)),
        len(q_numbers),
        len(q_numbers & a_numbers) / max(1, len(q_numbers)),
        len(q_numbers & e_numbers) / max(1, len(q_numbers)),
        *[float(cue in normalized) for cue in cues],
    ]


def _aggregate(
    selected: Sequence[int], meteor: Any, rouge: Any, profiles: Sequence[str]
) -> dict[str, Any]:
    import numpy as np

    indices = np.arange(len(selected))
    counts = Counter(profiles[index] for index in selected)
    return {
        "meteor": float(np.mean(meteor[indices, selected])),
        "rouge_l": float(np.mean(rouge[indices, selected])),
        "profile_counts": dict(sorted(counts.items())),
    }


def _select(predicted_gain: Any, baseline_index: int, threshold: float) -> Any:
    import numpy as np

    chosen = np.argmax(predicted_gain, axis=1)
    best = predicted_gain[np.arange(len(chosen)), chosen]
    return np.where(best > threshold, chosen, baseline_index)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.folds < 3:
        raise ValueError("folds must be at least 3")
    questions = _read_json(args.questions)
    if not isinstance(questions, dict):
        raise ValueError("questions must be a JSON object")
    rows = _read_jsonl(args.candidate_bank)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    ordered_ids: list[str] = []
    for row in rows:
        question_id = str(row.get("id", ""))
        profile = str(row.get("profile", ""))
        if not question_id or not profile:
            raise ValueError(f"invalid candidate key {question_id!r}/{profile!r}")
        if question_id not in grouped:
            ordered_ids.append(question_id)
        elif profile in grouped[question_id]:
            raise ValueError(f"duplicate candidate key {question_id!r}/{profile!r}")
        grouped[question_id][profile] = row

    profiles = list(args.profiles)
    if args.baseline not in profiles:
        raise ValueError("baseline must be included in profiles")
    required = set(profiles) | {"qwen", "evidence_512"}
    for question_id in ordered_ids:
        if missing := required - set(grouped[question_id]):
            raise ValueError(f"{question_id} misses profiles: {sorted(missing)}")
        if question_id not in questions:
            raise ValueError(f"question {question_id!r} is absent")

    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    meteor = np.asarray(
        [
            [float(grouped[qid][profile]["meteor"]) for profile in profiles]
            for qid in ordered_ids
        ],
        dtype=np.float32,
    )
    rouge = np.asarray(
        [
            [float(grouped[qid][profile]["rouge_l"]) for profile in profiles]
            for qid in ordered_ids
        ],
        dtype=np.float32,
    )
    baseline_index = profiles.index(args.baseline)
    target_gain = meteor - meteor[:, [baseline_index]]
    documents: list[str] = []
    numeric: list[list[float]] = []
    for question_id in ordered_ids:
        record = questions[question_id]
        question = str(record["question"])
        qwen = str(grouped[question_id]["qwen"]["answer"])
        evidence = str(grouped[question_id]["evidence_512"]["answer"])
        documents.append(_document(question, qwen, evidence))
        numeric.append(_numeric_features(question, qwen, evidence))
    numeric_matrix = np.asarray(numeric, dtype=np.float32)
    fold_ids = np.asarray(
        [_fold(qid, args.folds, args.seed) for qid in ordered_ids],
        dtype=np.int16,
    )

    char_alphas = (2.0, 5.0, 10.0, 20.0, 40.0)
    word_alphas = (5.0, 10.0, 20.0, 40.0)
    predictions = {
        **{
            f"char_a{alpha:g}": np.full_like(target_gain, np.nan)
            for alpha in char_alphas
        },
        **{
            f"word_a{alpha:g}": np.full_like(target_gain, np.nan)
            for alpha in word_alphas
        },
        "numeric_extra_trees": np.full_like(target_gain, np.nan),
    }
    for fold in range(args.folds):
        train = np.flatnonzero(fold_ids != fold)
        valid = np.flatnonzero(fold_ids == fold)
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=40_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        char_train = char_vectorizer.fit_transform([documents[i] for i in train])
        char_valid = char_vectorizer.transform([documents[i] for i in valid])
        for alpha in char_alphas:
            model = Ridge(alpha=alpha)
            model.fit(char_train, target_gain[train])
            predictions[f"char_a{alpha:g}"][valid] = model.predict(char_valid)

        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        word_train = word_vectorizer.fit_transform([documents[i] for i in train])
        word_valid = word_vectorizer.transform([documents[i] for i in valid])
        for alpha in word_alphas:
            model = Ridge(alpha=alpha)
            model.fit(word_train, target_gain[train])
            predictions[f"word_a{alpha:g}"][valid] = model.predict(word_valid)

        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(numeric_matrix[train])
        numeric_valid = scaler.transform(numeric_matrix[valid])
        tree = ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=12,
            max_features=0.9,
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        tree.fit(numeric_train, target_gain[train])
        predictions["numeric_extra_trees"][valid] = tree.predict(numeric_valid)
        print(
            f"p19_fold={fold + 1}/{args.folds} train={len(train)} "
            f"valid={len(valid)} char_features={char_train.shape[1]} "
            f"word_features={word_train.shape[1]}",
            flush=True,
        )

    for name, matrix in predictions.items():
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"OOF predictions are incomplete for {name}")

    # These blends are fixed before observing their aggregate metrics.  The
    # threshold sweep is diagnostic; promotion must later use nested tuning.
    predictions["blend_char10_word20"] = (
        0.65 * predictions["char_a10"] + 0.35 * predictions["word_a20"]
    )
    predictions["blend_text_tree"] = (
        0.75 * predictions["blend_char10_word20"]
        + 0.25 * predictions["numeric_extra_trees"]
    )
    thresholds = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
    methods: dict[str, Any] = {}
    for name, matrix in predictions.items():
        for threshold in thresholds:
            selected = _select(matrix, baseline_index, threshold)
            methods[f"{name}_t{threshold:g}"] = _aggregate(
                selected, meteor, rouge, profiles
            )
    baseline_selected = np.full(len(ordered_ids), baseline_index, dtype=np.int64)
    oracle_selected = np.argmax(meteor, axis=1)
    report = {
        "schema_version": "task2-p19-uplift-selector-research-v1",
        "question_count": len(ordered_ids),
        "folds": args.folds,
        "seed": args.seed,
        "profiles": profiles,
        "baseline_profile": args.baseline,
        "baseline": _aggregate(baseline_selected, meteor, rouge, profiles),
        "oracle": _aggregate(oracle_selected, meteor, rouge, profiles),
        "methods": dict(
            sorted(
                methods.items(),
                key=lambda item: (-item[1]["meteor"], -item[1]["rouge_l"], item[0]),
            )
        ),
        "selection_note": (
            "Every prediction is question-grouped OOF. Threshold sweep is "
            "research-only and requires nested validation before promotion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    best_name, best = next(iter(report["methods"].items()))
    print(
        f"P19_RESEARCH_COMPLETE best={best_name} "
        f"meteor={best['meteor']:.6f} rouge_l={best['rouge_l']:.6f}",
        flush=True,
    )
    return report


def main() -> int:
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print(f"Task 2 P19 selector research failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
