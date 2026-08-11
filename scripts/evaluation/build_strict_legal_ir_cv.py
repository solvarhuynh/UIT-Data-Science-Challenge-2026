"""Create deterministic leakage-safe grouped folds for LegalIR training data.

Every question sharing the ensemble's normalized text is assigned to exactly
one validation fold.  This prevents exact-question label/KNN leakage during
out-of-fold evaluation while retaining all organizer train labels for the
appropriate training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.legal_ir import (  # noqa: E402
    WarmupSample,
    load_warmup,
    normalize_legal_ir_matching_question,
)

DEFAULT_SEED = 2026
DEFAULT_FOLDS = 5


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    """Return the checked-out commit without making git a hard dependency."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unavailable"


def _canonical_corpus_hash() -> str:
    """Read the immutable canonical-corpus fingerprint when it is available."""

    manifest_path = PROJECT_ROOT / "data/processed_v3/metadata/manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unavailable"
    value = payload.get("corpus_hash") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else "unavailable"


def _group_question_ids(samples: Sequence[WarmupSample]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        groups[normalize_legal_ir_matching_question(sample.raw_question)].append(
            sample.id
        )
    return {key: sorted(ids) for key, ids in groups.items()}


def build_strict_grouped_folds(
    samples: Sequence[WarmupSample],
    *,
    fold_count: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[list[str]]:
    """Assign complete normalized-question groups to balanced folds."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not samples:
        raise ValueError("samples must not be empty")
    groups = _group_question_ids(samples)
    if len(groups) < fold_count:
        raise ValueError("fold_count cannot exceed normalized question group count")

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    # Large duplicate groups go first so the greedy balancing step cannot leave
    # one fold disproportionately large.  The seeded position breaks ties.
    ordered_keys = sorted(
        enumerate(keys), key=lambda item: (-len(groups[item[1]]), item[0])
    )
    fold_ids: list[list[str]] = [[] for _ in range(fold_count)]
    fold_group_counts = [0] * fold_count
    for _, key in ordered_keys:
        target = min(
            range(fold_count),
            key=lambda index: (len(fold_ids[index]), fold_group_counts[index], index),
        )
        fold_ids[target].extend(groups[key])
        fold_group_counts[target] += 1
    return [sorted(ids) for ids in fold_ids]


def validate_strict_grouped_folds(
    samples: Sequence[WarmupSample], folds: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Validate exact union/disjointness and normalized-question isolation."""

    if not samples:
        raise ValueError("samples must not be empty")
    if len(folds) < 2:
        raise ValueError("at least two folds are required")
    all_ids = {sample.id for sample in samples}
    question_key_by_id = {
        sample.id: normalize_legal_ir_matching_question(sample.raw_question)
        for sample in samples
    }
    fold_sets = [set(fold) for fold in folds]
    observed_ids = set().union(*fold_sets)
    duplicate_ids = sorted(
        {
            question_id
            for index, fold in enumerate(fold_sets)
            for later in fold_sets[index + 1 :]
            for question_id in fold.intersection(later)
        }
    )
    unknown_ids = sorted(observed_ids.difference(all_ids))
    missing_ids = sorted(all_ids.difference(observed_ids))
    if duplicate_ids or unknown_ids or missing_ids:
        raise ValueError(
            "invalid fold coverage "
            f"duplicate={duplicate_ids[:5]} unknown={unknown_ids[:5]} "
            f"missing={missing_ids[:5]}"
        )

    group_to_fold: dict[str, int] = {}
    normalized_overlap: list[str] = []
    for fold_index, ids in enumerate(fold_sets):
        for question_id in ids:
            key = question_key_by_id[question_id]
            previous = group_to_fold.setdefault(key, fold_index)
            if previous != fold_index:
                normalized_overlap.append(key)
    if normalized_overlap:
        raise ValueError(
            "normalized question groups cross validation folds: "
            + ", ".join(sorted(set(normalized_overlap))[:5])
        )
    return {
        "folds_disjoint": True,
        "validation_union_equals_all_train": True,
        "normalized_question_groups_disjoint": True,
        "question_count": len(all_ids),
        "normalized_question_group_count": len(group_to_fold),
    }


def _fold_payload(
    samples: Sequence[WarmupSample], folds: Sequence[Sequence[str]], *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = validate_strict_grouped_folds(samples, folds)
    all_ids = {sample.id for sample in samples}
    key_by_id = {
        sample.id: normalize_legal_ir_matching_question(sample.raw_question)
        for sample in samples
    }
    groups = _group_question_ids(samples)
    fold_rows: list[dict[str, Any]] = []
    fold_stats: list[dict[str, Any]] = []
    for index, ids in enumerate(folds):
        validation_ids = sorted(ids)
        validation_set = set(validation_ids)
        training_ids = sorted(all_ids.difference(validation_set))
        validation_keys = {key_by_id[item] for item in validation_ids}
        training_keys = {key_by_id[item] for item in training_ids}
        overlap = validation_keys.intersection(training_keys)
        if overlap:
            raise ValueError(
                f"fold {index} has normalized training/validation overlap: "
                f"{sorted(overlap)[:5]}"
            )
        fold_rows.append(
            {
                "fold": index,
                "validation_ids": validation_ids,
                "training_ids": training_ids,
            }
        )
        fold_stats.append(
            {
                "fold": index,
                "validation_question_count": len(validation_ids),
                "training_question_count": len(training_ids),
                "validation_normalized_question_group_count": len(validation_keys),
                "training_normalized_question_group_count": len(training_keys),
                "normalized_train_validation_overlap_count": len(overlap),
            }
        )
    duplicate_groups = {key: ids for key, ids in groups.items() if len(ids) > 1}
    folds_payload = {
        "schema_version": "legal-ir-strict-cv-v2",
        "seed": seed,
        "fold_count": len(folds),
        "normalization": "ensemble-nfkc-casefold-nonword-collapse-v1",
        "folds": fold_rows,
    }
    stats_payload = {
        "schema_version": "legal-ir-strict-cv-v2",
        "seed": seed,
        "fold_count": len(folds),
        "question_count": len(samples),
        "normalized_question_group_count": len(groups),
        "duplicate_normalized_group_count": len(duplicate_groups),
        "duplicate_normalized_question_count": sum(
            len(ids) for ids in duplicate_groups.values()
        ),
        "validation": validation,
        "folds": fold_stats,
    }
    return folds_payload, stats_payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _readme() -> str:
    return """# LegalIR strict grouped CV v2

This directory contains deterministic five-fold out-of-fold partitions of
`data/raw/btc/LegalIR/train.json`.

- Seed: `2026`
- Normalization: the exact NFKC/case-fold/punctuation-collapse function used by
  `scripts/submission/build_legal_ir_ensemble.py`.
- Every identical normalized question is placed in one validation fold only.
- For fold *n*, use `validation_ids` for scoring and `training_ids` for any
  KNN, exact-label, or other supervised source. Do not train on either the
  validation IDs or their normalized-question equivalents.

`fold_stats.json` proves fold disjointness, complete union coverage, and the
absence of normalized-question leakage. `manifest.json` records the input,
commit, parameters, and the canonical corpus fingerprint without modifying the
corpus.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/btc/LegalIR/train.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/task1/evaluation/strict_cv_v2"),
    )
    parser.add_argument("--folds", type=_positive_int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Optional deterministic first-N-question data smoke selection.",
    )
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    samples = load_warmup(args.input)
    samples = sorted(samples, key=lambda sample: sample.id)
    if args.limit is not None:
        samples = samples[: args.limit]
    folds = build_strict_grouped_folds(
        samples,
        fold_count=args.folds,
        seed=args.seed,
    )
    folds_payload, stats_payload = _fold_payload(samples, folds, seed=args.seed)
    folds_payload["input"] = str(args.input)
    folds_payload["input_sha256"] = _sha256(args.input)
    stats_payload["input"] = str(args.input)
    stats_payload["input_sha256"] = _sha256(args.input)
    if args.limit is not None:
        folds_payload["question_limit"] = args.limit
        stats_payload["question_limit"] = args.limit

    manifest_payload = {
        "schema_version": "task1-artifact-manifest-v1",
        "git_commit": _git_commit(),
        "config": {
            "script": "scripts/evaluation/build_strict_legal_ir_cv.py",
            "input": str(args.input),
            "input_sha256": folds_payload["input_sha256"],
            "normalization": folds_payload["normalization"],
        },
        "model": {"name": None, "used": False},
        "corpus": {
            "root": "data/processed_v3",
            "corpus_hash": _canonical_corpus_hash(),
            "used_for_fold_assignment": False,
        },
        "parameters": {
            "fold_count": args.folds,
            "seed": args.seed,
            "question_limit": args.limit,
        },
    }

    outputs = [
        args.output_dir / "folds.json",
        args.output_dir / "fold_stats.json",
        args.output_dir / "manifest.json",
        args.output_dir / "README.md",
    ]
    _write_json(outputs[0], folds_payload)
    _write_json(outputs[1], stats_payload)
    _write_json(outputs[2], manifest_payload)
    outputs[3].write_text(_readme(), encoding="utf-8", newline="\n")
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"strict LegalIR CV error: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
