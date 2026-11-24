"""Build leakage-safe grouped training data with mined semantic confusers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-groups", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hard", type=int, default=2)
    parser.add_argument("--max-semi-hard", type=int, default=2)
    return parser


def _mine(args: argparse.Namespace, negative_dir: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("mine_hard_negatives.py")),
        "--train", str(args.train),
        "--folds", str(args.folds),
        "--rankings", f"dense={args.dense}",
        "--output-dir", str(negative_dir),
        "--processed-root", str(args.processed_root),
        "--max-hard-per-query", str(max(args.max_hard, 1)),
        "--max-semi-hard-per-query", str(max(args.max_semi_hard, 1)),
    ]
    subprocess.run(command, check=True)


def _load_rows(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[str(row["query_id"])].append(row)
    return rows


def run(args: argparse.Namespace) -> dict:
    if args.max_hard < 0 or args.max_semi_hard < 0:
        raise ValueError("negative limits must be non-negative")
    for path in (args.train, args.folds, args.dense, args.base_groups, args.base_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    negative_dir = output / "mined"
    _mine(args, negative_dir)

    mined: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(negative_dir.glob("fold_*.jsonl")):
        for query_id, rows in _load_rows(path).items():
            mined[query_id].extend(rows)

    destination = output / "groups.jsonl"
    counts = {"queries": 0, "added_hard": 0, "added_semi_hard": 0, "skipped": 0}
    with args.base_groups.open(encoding="utf-8") as source, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            group = json.loads(line)
            query_id = str(group["query_id"])
            existing = {
                str(hit.get("doc_id"))
                for values in group.get("negative_chunks", {}).values()
                for hit in values
            }
            selected: dict[str, list[dict]] = {"hard": [], "semi_hard": []}
            seen: set[str] = set()
            rank = {"hard": 0, "semi_hard": 1}
            for row in sorted(
                mined.get(query_id, []),
                key=lambda item: (rank.get(str(item.get("strength")), 2), int(item.get("source_rank", 10**9)), str(item.get("negative_doc", ""))),
            ):
                kind = str(row.get("strength", ""))
                if kind not in selected:
                    continue
                doc_id = str(row.get("negative_doc", "")).strip()
                text = str(row.get("negative_evidence", "")).strip()
                if not doc_id or not text or doc_id in existing or doc_id in seen:
                    continue
                if len(selected[kind]) >= (args.max_hard if kind == "hard" else args.max_semi_hard):
                    continue
                selected[kind].append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"hard_negative_{hashlib.sha1((query_id + doc_id).encode()).hexdigest()[:16]}",
                        "text": text,
                    }
                )
                seen.add(doc_id)
                counts["added_hard" if kind == "hard" else "added_semi_hard"] += 1
            group.setdefault("negative_chunks", {})
            for kind, values in selected.items():
                group["negative_chunks"].setdefault(kind, []).extend(values)
            target.write(json.dumps(group, ensure_ascii=False) + "\n")
            counts["queries"] += 1

    base_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    manifest = {
        **base_manifest,
        "schema_version": "retrieval-training-groups-v2",
        "groups_sha256": _sha256(destination),
        "base_groups_sha256": _sha256(args.base_groups),
        "hard_negative_mining_manifest_sha256": _sha256(negative_dir / "mining_manifest.json"),
        "hard_negative_limits": {"hard": args.max_hard, "semi_hard": args.max_semi_hard},
        "hard_negative_counts": counts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": str(destination), "manifest": str(manifest_path), **counts}, indent=2))
    return manifest


if __name__ == "__main__":
    run(build_parser().parse_args())
