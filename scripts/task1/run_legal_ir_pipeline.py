"""Unified, resumable operational runner for Task 1 LegalIR.

The runner owns orchestration and manifests only. Retrieval, lexical scoring,
reranking, and submission validation remain implemented by existing scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

STAGES = (
    "preflight",
    "retrieve",
    "lexical",
    "candidate-union",
    "rerank",
    "ensemble",
    "submission",
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/task1_baseline.yaml")
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="Question source; required for non-dry-run stages.",
    )
    parser.add_argument("--train", type=Path)
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed_v3")
    )
    parser.add_argument(
        "--contexts-dir",
        type=Path,
        default=Path("data/raw/btc/LegalIR/selected-contexts"),
    )
    parser.add_argument(
        "--vector-store-root", type=Path, default=Path("data/vector_store")
    )
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/task1/pipeline")
    )
    parser.add_argument("--stage", choices=STAGES + ("all",), default="all")
    parser.add_argument("--max-questions", type=_positive)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Task 1 runner config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "task1-baseline-v1"
    ):
        raise ValueError("config must be a validated task1-baseline-v1 mapping")
    return payload


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_path(path: Path) -> str | None:
    if path.is_file():
        return _hash_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for item in sorted(path.rglob("*")):
            if item.is_file():
                digest.update(str(item.relative_to(path)).encode("utf-8"))
                digest.update(_hash_file(item).encode("ascii"))
        return digest.hexdigest()
    return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "config": args.config,
        "questions": args.questions,
        "train": args.train,
        "processed_root": args.processed_root,
        "contexts_dir": args.contexts_dir,
        "vector_store_root": args.vector_store_root,
        "artifacts_root": args.artifacts_root,
    }
    return {
        "paths": {
            name: (str(path.resolve()) if path is not None else None)
            for name, path in paths.items()
        },
        "config": config,
        "stage": args.stage,
        "max_questions": args.max_questions,
        "resume": args.resume,
        "dry_run": args.dry_run,
    }


def _stage_list(stage: str) -> list[str]:
    return list(STAGES) if stage == "all" else ["preflight", stage]


def _corpus_ids(processed_root: Path, output: Path) -> Path:
    ids: list[str] = []
    for path in sorted((processed_root / "documents").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        document_id = payload.get("id", path.stem)
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"invalid document ID in {path}")
        ids.append(document_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("corpus document IDs must be non-empty and unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def _manifest(
    args: argparse.Namespace, config: dict[str, Any], stages: list[str]
) -> dict[str, Any]:
    resolved = _resolve(args, config)
    input_hashes = {
        name: _hash_path(path)
        for name, path in (
            (name, Path(value))
            for name, value in resolved["paths"].items()
            if value and name != "artifacts_root"
        )
    }
    return {
        "schema_version": "legal-ir-operational-run-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "inputs": input_hashes,
        "resolved": resolved,
        "stages": stages,
        "model_revisions": {"reranker": config["reranker"]["name"]},
        "representation_versions": config["representation_versions"],
        "candidate_depths": config["candidate"]["depths"],
        "reranker": config["reranker"],
        "rrf_weights": config["ensemble"],
        "outputs": {},
    }


def _print_plan(resolved: dict[str, Any], stages: list[str]) -> None:
    print(
        json.dumps(
            {"resolved": resolved, "stages": stages}, ensure_ascii=False, indent=2
        )
    )


def _run_submission(
    args: argparse.Namespace, corpus_manifest: Path, ensemble: Path
) -> list[Path]:
    output = args.artifacts_root / "submission.zip"
    writer = ROOT / "scripts/submission/write_legal_ir_submission.py"
    validator = ROOT / "scripts/submission/validate_legal_ir_submission.py"
    subprocess.run(
        [
            sys.executable,
            str(writer),
            "--input",
            str(ensemble),
            "--questions",
            str(args.questions),
            "--corpus-manifest",
            str(corpus_manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--input",
            str(output),
            "--questions",
            str(args.questions),
            "--corpus-manifest",
            str(corpus_manifest),
        ],
        cwd=ROOT,
        check=True,
    )
    return [output]


def _cached_stage_artifact(args: argparse.Namespace, stage: str) -> Path | None:
    """Find known cached outputs without loading neural components."""

    candidates = {
        "retrieve": [
            args.artifacts_root / "retrieve" / "dense_predictions.jsonl",
            ROOT / "artifacts/task1/train500_dense200_predictions.jsonl",
            ROOT / "artifacts/task1/train500b_dense200_predictions.jsonl",
        ],
        "lexical": [
            args.artifacts_root / "lexical" / "summary.json",
            ROOT / "artifacts/task1/evaluation/p3_lexical/summary.json",
        ],
        "candidate-union": [
            args.artifacts_root / "candidate-union" / "predictions.json"
        ],
        "rerank": [args.artifacts_root / "rerank" / "predictions_after.jsonl"],
        "ensemble": [args.artifacts_root / "ensemble" / "predictions.json"],
    }
    return next((path for path in candidates.get(stage, []) if path.is_file()), None)


def run(args: argparse.Namespace) -> int:
    config = _load_yaml(args.config)
    stages = _stage_list(args.stage)
    resolved = _resolve(args, config)
    if args.dry_run:
        _print_plan(resolved, stages)
        return 0
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.artifacts_root / "run_manifest.json"
    manifest = _manifest(args, config, stages)
    if args.resume and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_outputs = old.get("outputs", {})
        outputs_exist = all(
            (args.artifacts_root / name).is_file()
            for name in old_outputs
            if name != "corpus_document_ids"
        )
        if (
            old.get("inputs") == manifest["inputs"]
            and old.get("resolved", {}).get("config") == manifest["resolved"]["config"]
            and outputs_exist
        ):
            print(f"RESUME manifest matched: {manifest_path}")
            return 0
    if "preflight" in stages:
        if args.questions is None:
            raise ValueError("--questions is required unless --dry-run is used")
        for path in (
            args.config,
            args.questions,
            args.processed_root,
            args.contexts_dir,
        ):
            if not path.exists():
                raise FileNotFoundError(f"preflight input missing: {path}")
        corpus_manifest = _corpus_ids(
            args.processed_root, args.artifacts_root / "corpus_document_ids.json"
        )
        manifest["outputs"]["corpus_document_ids"] = _hash_file(corpus_manifest)
    else:
        corpus_manifest = args.artifacts_root / "corpus_document_ids.json"
    for stage in stages:
        if stage in {"preflight", "submission"}:
            continue
        cached = _cached_stage_artifact(args, stage)
        status = "reused" if cached is not None else "not-run-neural-or-missing-cache"
        manifest.setdefault("stage_status", {})[stage] = {
            "status": status,
            "artifact": str(cached) if cached is not None else None,
            "artifact_sha256": _hash_file(cached) if cached is not None else None,
        }
        print(f"stage={stage} status={status}")
    if "submission" in stages:
        ensemble = args.artifacts_root / "ensemble" / "predictions.json"
        if not ensemble.is_file():
            raise FileNotFoundError(
                f"submission stage needs existing ensemble artifact: {ensemble}"
            )
        for output in _run_submission(args, corpus_manifest, ensemble):
            manifest["outputs"][str(output.relative_to(args.artifacts_root))] = (
                _hash_file(output)
            )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"completed stages={stages} manifest={manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Task 1 pipeline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
