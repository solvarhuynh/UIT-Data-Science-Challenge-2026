"""Build source-only LegalQA answers from parent cross-encoder rankings.

The input is the JSONL emitted by ``train_legal_qa_parent_crossencoder.py
score``.  Answers contain only bounded parent or child-source text.  This
builder has no known-answer overlay, reference-dependent selection, rewriting,
or submission packaging step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--question-ids",
        type=Path,
        help="Explicit ordered ID subset for strict held-out evaluation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-parents", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--max-parent-tokens", type=_positive_int, default=512)
    parser.add_argument("--max-total-tokens", type=_positive_int, default=1024)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-weight", type=float, default=0.0)
    parser.add_argument("--rrf-k", type=_non_negative_int, default=5)
    return parser


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc


def _clean_id(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip() or _CONTROL_RE.search(value):
        raise ValueError(f"{label} must be a clean non-empty string")
    return value


def load_question_ids(path: Path, subset_path: Path | None = None) -> list[str]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty JSON object")
    output: list[str] = []
    for raw_id, raw_record in payload.items():
        question_id = _clean_id(raw_id, label="question ID")
        if not isinstance(raw_record, dict):
            raise TypeError(f"question {question_id!r} must be an object")
        question = raw_record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no text")
        output.append(question_id)
    if subset_path is None:
        return output
    raw_subset = _load_json(subset_path)
    if not isinstance(raw_subset, list) or not raw_subset:
        raise ValueError("question-ids must be a non-empty JSON string array")
    subset = [
        _clean_id(value, label="question subset ID") for value in raw_subset
    ]
    if len(subset) != len(set(subset)):
        raise ValueError("question-ids contains duplicate IDs")
    unknown = [question_id for question_id in subset if question_id not in payload]
    if unknown:
        raise ValueError(f"question-ids are missing from questions: {unknown[:5]}")
    return subset


def _positive_rank(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_score(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{label} must be finite")
    return score


def _validated_parent(raw: object, *, question_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"parents for {question_id!r} must be objects")
    parent = dict(raw)
    parent["doc_id"] = _clean_id(parent.get("doc_id"), label="doc_id")
    parent["parent_id"] = _clean_id(
        parent.get("parent_id"), label="parent_id"
    )
    for name in ("parent_text", "anchor_text"):
        value = parent.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"parent {parent['parent_id']!r} for {question_id!r} "
                f"has no {name}"
            )
        parent[name] = value.strip()
    parent["rank"] = _positive_rank(parent.get("rank"), label="rank")
    parent["candidate_rank"] = _positive_rank(
        parent.get("candidate_rank"), label="candidate_rank"
    )
    parent["crossencoder_logit"] = _finite_score(
        parent.get("crossencoder_logit"), label="crossencoder_logit"
    )
    return parent


def load_parent_scores(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_json_constant,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must be an object")
            question_id = _clean_id(
                row.get("question_id"), label=f"question_id at line {line_number}"
            )
            if question_id in output:
                raise ValueError(f"duplicate scored question {question_id!r}")
            raw_parents = row.get("parents")
            if not isinstance(raw_parents, list) or not raw_parents:
                raise ValueError(f"question {question_id!r} has no scored parents")
            parents = [
                _validated_parent(parent, question_id=question_id)
                for parent in raw_parents
            ]
            keys = [(parent["doc_id"], parent["parent_id"]) for parent in parents]
            if len(keys) != len(set(keys)):
                raise ValueError(f"question {question_id!r} repeats a parent")
            parents.sort(key=lambda parent: parent["rank"])
            if [parent["rank"] for parent in parents] != list(
                range(1, len(parents) + 1)
            ):
                raise ValueError(
                    f"question {question_id!r} parent ranks must be contiguous"
                )
            for previous, current in zip(parents, parents[1:]):
                if previous["crossencoder_logit"] < current["crossencoder_logit"]:
                    raise ValueError(
                        f"question {question_id!r} ranks contradict CE logits"
                    )
            output[question_id] = parents
    if not output:
        raise ValueError(f"parent score file is empty: {path}")
    return output


def _find_subsequence(haystack: list[str], needle: list[str]) -> int | None:
    if not needle:
        return 0
    if len(needle) > len(haystack):
        return None
    prefix = [0] * len(needle)
    matched = 0
    for index in range(1, len(needle)):
        while matched and needle[index] != needle[matched]:
            matched = prefix[matched - 1]
        if needle[index] == needle[matched]:
            matched += 1
            prefix[index] = matched
    matched = 0
    for index, token in enumerate(haystack):
        while matched and token != needle[matched]:
            matched = prefix[matched - 1]
        if token == needle[matched]:
            matched += 1
            if matched == len(needle):
                return index - len(needle) + 1
    return None


def bounded_parent_window(parent_text: str, anchor_text: str, limit: int) -> str:
    """Match ParentContextExpander's single-anchor bounded-window behavior."""

    if limit < 1:
        raise ValueError("parent window limit must be positive")
    parent_tokens = parent_text.split()
    anchor_tokens = anchor_text.split()
    if not parent_tokens or not anchor_tokens:
        raise ValueError("parent and anchor text must be non-blank")
    if len(parent_tokens) <= limit:
        return parent_text.strip()
    start = _find_subsequence(
        [token.casefold() for token in parent_tokens],
        [token.casefold() for token in anchor_tokens],
    )
    if start is None:
        return " ".join(anchor_tokens[:limit])
    end = start + len(anchor_tokens)
    extra = max(limit - len(anchor_tokens), 0)
    left = max(0, start - extra // 2)
    right = min(len(parent_tokens), end + extra - extra // 2)
    return " ".join(parent_tokens[left : min(right, left + limit)])


def fuse_parent_ranks(
    parents: Sequence[dict[str, Any]],
    *,
    ce_weight: float,
    retrieval_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Fuse CE rank with the original retrieval parent rank using RRF."""

    weights = (ce_weight, retrieval_weight)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("fusion weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fusion weights must sum to one")
    if rrf_k < 0:
        raise ValueError("rrf-k must be non-negative")
    retrieval_order = sorted(
        parents,
        key=lambda parent: (
            parent["candidate_rank"],
            parent["doc_id"],
            parent["parent_id"],
        ),
    )
    retrieval_rank = {
        (parent["doc_id"], parent["parent_id"]): rank
        for rank, parent in enumerate(retrieval_order, 1)
    }
    return sorted(
        parents,
        key=lambda parent: (
            -(
                ce_weight / (rrf_k + parent["rank"])
                + retrieval_weight
                / (
                    rrf_k
                    + retrieval_rank[(parent["doc_id"], parent["parent_id"])]
                )
            ),
            parent["rank"],
            parent["candidate_rank"],
            parent["doc_id"],
            parent["parent_id"],
        ),
    )


def build_predictions(
    question_ids: Sequence[str],
    scores: dict[str, list[dict[str, Any]]],
    *,
    top_parents: int,
    max_parent_tokens: int,
    max_total_tokens: int,
    ce_weight: float = 1.0,
    retrieval_weight: float = 0.0,
    rrf_k: int = 5,
) -> tuple[list[dict[str, str]], dict[str, int | float]]:
    if top_parents not in (1, 2, 3, 4):
        raise ValueError("top-parents must be in [1, 4]")
    if max_parent_tokens < 1 or max_total_tokens < 1:
        raise ValueError("answer token limits must be positive")
    if max_parent_tokens > max_total_tokens:
        raise ValueError("max-parent-tokens must not exceed max-total-tokens")
    expected = set(question_ids)
    actual = set(scores)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            f"score/question coverage mismatch: missing={missing}, extra={extra}"
        )
    predictions: list[dict[str, str]] = []
    parents_used = 0
    truncated_parents = 0
    fallback_anchors = 0
    for question_id in question_ids:
        remaining = max_total_tokens
        spans: list[str] = []
        ranked = fuse_parent_ranks(
            scores[question_id],
            ce_weight=ce_weight,
            retrieval_weight=retrieval_weight,
            rrf_k=rrf_k,
        )
        for parent in ranked[:top_parents]:
            if remaining <= 0:
                break
            limit = min(max_parent_tokens, remaining)
            parent_text = parent["parent_text"]
            anchor_text = parent["anchor_text"]
            window = bounded_parent_window(parent_text, anchor_text, limit)
            if window == " ".join(anchor_text.split()[:limit]) and (
                " ".join(anchor_text.casefold().split())
                not in " ".join(parent_text.casefold().split())
            ):
                fallback_anchors += 1
            if len(window.split()) < len(parent_text.split()):
                truncated_parents += 1
            spans.append(window)
            parents_used += 1
            remaining -= len(window.split())
        answer = "\n".join(spans).strip()
        if not answer:
            raise ValueError(f"question {question_id!r} produced an empty answer")
        predictions.append({"id": question_id, "answer": answer})
    return predictions, {
        "question_count": len(predictions),
        "parents_used": parents_used,
        "truncated_parents": truncated_parents,
        "fallback_anchors": fallback_anchors,
        "ce_weight": ce_weight,
        "retrieval_weight": retrieval_weight,
        "rrf_k": rrf_k,
    }


def _paths_collide(left: Path, right: Path) -> bool:
    try:
        if left.resolve(strict=True) == right.resolve(strict=False):
            return True
    except OSError:
        pass
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            pass
    return False


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run(args: argparse.Namespace) -> dict[str, int | float]:
    if args.output.suffix.casefold() != ".json":
        raise ValueError("internal prediction output must use .json")
    if _paths_collide(args.output, args.scores) or _paths_collide(
        args.output, args.questions
    ):
        raise ValueError("output must not overwrite an input")
    subset_path = getattr(args, "question_ids", None)
    if subset_path is not None and _paths_collide(args.output, subset_path):
        raise ValueError("output must not overwrite an input")
    if args.max_parent_tokens > args.max_total_tokens:
        raise ValueError("max-parent-tokens must not exceed max-total-tokens")
    weights = (args.ce_weight, args.retrieval_weight)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("fusion weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fusion weights must sum to one")
    question_ids = load_question_ids(args.questions, subset_path)
    scores = load_parent_scores(args.scores)
    predictions, summary = build_predictions(
        question_ids,
        scores,
        top_parents=args.top_parents,
        max_parent_tokens=args.max_parent_tokens,
        max_total_tokens=args.max_total_tokens,
        ce_weight=args.ce_weight,
        retrieval_weight=args.retrieval_weight,
        rrf_k=args.rrf_k,
    )
    _write_json(args.output, predictions)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"parent-score answer build error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
