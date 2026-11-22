"""Input provenance and grouped train/validation partitions."""

import hashlib
import json
from pathlib import Path

from udsc2026.evaluation.retrieval_metrics import normalize_legal_ir_matching_question


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checkpoint_files(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    if not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError(f"missing model weights in {directory}")
    return {path.name: sha256(path) for path in paths}


def verify_checkpoint(directory: Path, report: dict) -> None:
    if report.get("complete") is not True or report.get("files") != checkpoint_files(
        directory
    ):
        raise ValueError(f"checkpoint is incomplete or has changed: {directory}")


def commit_training_checkpoint(volume_name: str | None) -> None:
    if volume_name:
        from udsc2026.infrastructure.modal_volume import commit_mounted_volume_sync

        commit_mounted_volume_sync(volume_name)


def identity_inputs(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256(path) for name, path in paths.items()}


def question_mapping_bytes(questions: dict[str, str]) -> bytes:
    mapping = {qid: {"question": question} for qid, question in questions.items()}
    return (json.dumps(mapping, ensure_ascii=False, indent=2) + "\n").encode()


def validate_folds(path: Path, questions: dict[str, str]) -> dict[int, set[str]]:
    records = read_json(path)["folds"]
    assignments = {}
    validation = {}
    groups = {}
    for record in records:
        fold = record["fold"]
        ids = record["validation_ids"]
        if fold in validation or not ids:
            raise ValueError(f"empty or duplicate validation fold {fold}")
        validation[fold] = set(ids)
        for qid in ids:
            if qid in assignments or qid not in questions:
                raise ValueError(f"invalid or overlapping validation ID {qid}")
            assignments[qid] = fold
            group = normalize_legal_ir_matching_question(questions[qid])
            if groups.setdefault(group, fold) != fold:
                raise ValueError(f"normalized question group crosses folds at ID {qid}")
        if set(record["training_ids"]) != questions.keys() - validation[fold]:
            raise ValueError(f"training IDs in fold {fold} differ from the complement")
    if set(validation) != set(range(5)) or assignments.keys() != questions.keys():
        raise ValueError("the five validation folds must partition all training IDs")
    return validation
