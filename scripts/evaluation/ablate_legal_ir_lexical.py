"""Strict-CV CPU ablation for Task 1 legal lexical retrieval.

This script only uses cached dense/BGE JSONL inputs.  It does not load a
neural model or construct an ANN index.  KNN labels are rebuilt per fold after
removing every validation ID and every normalized duplicate validation text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from udsc2026.evaluation.legal_ir import (  # noqa: E402
    LegalIRPrediction,
    LegalIRReference,
    evaluate_legal_ir,
    load_warmup,
    normalize_legal_ir_matching_question,
)
from udsc2026.evaluation.legal_ir_lexical import (  # noqa: E402
    LegalContext,
    build_bm25f_rankings,
    build_citation_rankings,
    build_knn_rankings,
    parse_legal_citations,
    weighted_rrf,
)

TITLE_WEIGHTS = (0, 1, 2, 4, 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_fingerprint(path: Path) -> str:
    """Stable input fingerprint for either one file or a context directory."""

    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(path.glob("*.json")):
        digest.update(item.name.encode("utf-8"))
        digest.update(_sha256(item).encode("ascii"))
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_contexts(path: Path) -> list[LegalContext]:
    result: list[LegalContext] = []
    for file_path in sorted(path.glob("*.json")):
        item = _read_json(file_path)
        title = item.get("title", item.get("name", ""))
        if title is None:
            title = ""
        result.append(LegalContext(str(item["id"]), str(item["passage"]), str(title)))
    if not result:
        raise ValueError(f"no context JSON files in {path}")
    return sorted(result, key=lambda item: (len(item.doc_id), item.doc_id))


def _load_rankings(path: Path, allowed_ids: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in result:
                raise ValueError(
                    f"duplicate question ID {question_id!r} in {path}:{line_number}"
                )
            ranking: list[str] = []
            seen: set[str] = set()
            for hit in row.get("hits", []):
                doc_id = str(hit["doc_id"])
                if doc_id not in allowed_ids:
                    raise ValueError(f"unknown doc {doc_id!r} in {path}:{line_number}")
                if doc_id not in seen:
                    seen.add(doc_id)
                    ranking.append(doc_id)
            result[question_id] = ranking
    return result


def _fold_map(path: Path) -> dict[str, int]:
    payload = _read_json(path)
    result: dict[str, int] = {}
    for item in payload["folds"]:
        for question_id in item["validation_ids"]:
            if question_id in result:
                raise ValueError(f"duplicate strict-CV question ID {question_id!r}")
            result[str(question_id)] = int(item["fold"])
    return result


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "gold": args.gold,
        "strict_folds": args.strict_folds,
        "dense": args.dense,
        "contexts_dir": args.contexts_dir,
    }
    if args.bge:
        paths["bge"] = args.bge
    return {
        "schema_version": "legal-ir-p3-component-cache-v1",
        "inputs": {name: _path_fingerprint(path) for name, path in paths.items()},
        "parameters": {
            "title_weights": list(TITLE_WEIGHTS),
            "word": ["word", 1, 2],
            "char": ["char_wb", 3, 5],
            "rrf_k": 60,
            "max_questions": args.max_questions,
        },
    }


def _metric(
    samples: Sequence[Any], rankings: dict[str, list[str]]
) -> dict[str, float | int]:
    references = [
        LegalIRReference(id=sample.id, gold_documents=sample.gold_documents)
        for sample in samples
    ]
    predictions = [
        LegalIRPrediction(id=sample.id, documents=rankings[sample.id][:5])
        for sample in samples
    ]
    report = evaluate_legal_ir(references, predictions)
    return {
        "recall": report.aggregate.recall,
        "precision": report.aggregate.precision,
        "count": len(samples),
    }


def _subset_metric(
    samples: Sequence[Any], rankings: dict[str, list[str]], predicate: Any
) -> dict[str, float | int] | None:
    subset = [sample for sample in samples if predicate(sample)]
    return _metric(subset, rankings) if subset else None


def _build_components(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    samples = load_warmup(args.gold)
    folds = _fold_map(args.strict_folds)
    by_id = {sample.id: sample for sample in samples}
    if set(folds) != set(by_id):
        raise ValueError("strict fold coverage must equal gold train coverage")
    dense = _load_rankings(
        args.dense, {item.doc_id for item in _load_contexts(args.contexts_dir)}
    )
    selected = [sample for sample in samples if sample.id in dense]
    if args.max_questions is not None:
        selected = selected[: args.max_questions]
    if not selected:
        raise ValueError("no gold questions overlap cached dense rankings")
    bge = (
        _load_rankings(
            args.bge, {item.doc_id for item in _load_contexts(args.contexts_dir)}
        )
        if args.bge
        else {}
    )
    if bge and set(bge) != set(dense):
        raise ValueError("dense and BGE cached prediction coverage mismatch")
    contexts = _load_contexts(args.contexts_dir)
    # These components are unsupervised, so compute them once across selected
    # queries instead of re-vectorizing the corpus separately for each fold.
    all_questions = {item.id: item.raw_question for item in selected}
    bm25_all = {
        str(weight): build_bm25f_rankings(all_questions, contexts, title_weight=weight)
        for weight in TITLE_WEIGHTS
    }
    citation_all = build_citation_rankings(all_questions, contexts)
    records: list[dict[str, Any]] = []
    for fold in sorted({folds[item.id] for item in selected}):
        validation = [item for item in selected if folds[item.id] == fold]
        excluded_texts = {
            normalize_legal_ir_matching_question(item.raw_question)
            for item in validation
        }
        pool = [
            (item.id, item.raw_question, item.gold_documents)
            for item in samples
            if item.id not in {value.id for value in validation}
            and normalize_legal_ir_matching_question(item.raw_question)
            not in excluded_texts
        ]
        questions = {item.id: item.raw_question for item in validation}
        word = build_knn_rankings(questions, pool, analyzer="word", ngram_range=(1, 2))
        char = build_knn_rankings(
            questions, pool, analyzer="char_wb", ngram_range=(3, 5)
        )
        for item in validation:
            records.append(
                {
                    "id": item.id,
                    "fold": fold,
                    "dense": dense[item.id],
                    "bge": bge.get(item.id, []),
                    "knn_word": word[item.id],
                    "knn_char": char[item.id],
                    "bm25_body": bm25_all["0"][item.id],
                    "bm25_title_body": {
                        weight: bm25_all[weight][item.id]
                        for weight in bm25_all
                        if weight != "0"
                    },
                    "citation": citation_all[item.id],
                }
            )
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache, cache_manifest = (
        args.output_dir / "component_cache.json",
        args.output_dir / "component_cache_manifest.json",
    )
    wanted_manifest = _manifest(args)
    if (
        cache.is_file()
        and cache_manifest.is_file()
        and _read_json(cache_manifest) == wanted_manifest
    ):
        records = _read_json(cache)
        cache_status = "reused"
    else:
        records = _build_components(args, wanted_manifest)
        cache.write_text(
            json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        cache_manifest.write_text(
            json.dumps(wanted_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        cache_status = "built"
    samples = {item.id: item for item in load_warmup(args.gold)}
    selected = [samples[item["id"]] for item in records]
    by_id = {item["id"]: item for item in records}
    variants = {
        "baseline_lexical": lambda item: [item["knn_word"], item["bm25_body"]],
        "char_knn": lambda item: [
            item["knn_word"],
            item["knn_char"],
            item["bm25_body"],
        ],
        "citation": lambda item: [
            item["knn_word"],
            item["bm25_body"],
            item["citation"],
        ],
        "all_lexical": lambda item: [
            item["knn_word"],
            item["knn_char"],
            item["bm25_title_body"]["1"],
            item["citation"],
        ],
    }
    variants.update(
        {
            f"bm25f_title_weight_{weight}": lambda item, weight=str(weight): [
                item["knn_word"],
                item["bm25_title_body"].get(weight, item["bm25_body"]),
            ]
            for weight in TITLE_WEIGHTS
        }
    )
    report: dict[str, Any] = {
        "schema_version": "legal-ir-p3-ablation-v1",
        "cache_status": cache_status,
        "question_count": len(selected),
        "candidate_depth": max(len(item["dense"]) for item in records),
        "variants": {},
    }
    for name, sources in variants.items():
        rankings = {item.id: weighted_rrf(sources(by_id[item.id])) for item in selected}
        per_fold = {
            str(fold): _metric(
                [item for item in selected if by_id[item.id]["fold"] == fold], rankings
            )
            for fold in sorted({by_id[item.id]["fold"] for item in selected})
        }
        report["variants"][name] = {
            "overall": _metric(selected, rankings),
            "explicit_citation": _subset_metric(
                selected,
                rankings,
                lambda item: parse_legal_citations(item.raw_question).explicit,
            ),
            "non_citation": _subset_metric(
                selected,
                rankings,
                lambda item: not parse_legal_citations(item.raw_question).explicit,
            ),
            "multi_gold": _subset_metric(
                selected, rankings, lambda item: len(item.gold_documents) > 1
            ),
            "per_fold": per_fold,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P3 lexical strict-CV ablation",
        "",
        f"- cached components: {cache_status}",
        f"- questions: {len(selected)}",
        "",
        "| variant | Recall | Precision |",
        "|---|---:|---:|",
    ]
    lines += [
        "| "
        f"{name} | {value['overall']['recall']:.6f} | "
        f"{value['overall']['precision']:.6f} |"
        for name, value in report["variants"].items()
    ]
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--strict-folds", type=Path, required=True)
    parser.add_argument("--contexts-dir", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--bge", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_questions is not None and args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
