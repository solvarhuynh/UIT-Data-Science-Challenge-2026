"""Content-addressed recovery of the missing Team 09205 V2 anchor.

This CPU-only forensic tool derives V2's rank-1 fingerprint from the supplied
safeguard artifact, searches bounded prediction locations by payload shape, and
uses the two supplied handoff submissions as checksums.  It never loads labels,
models, or private oracle data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
TEAM_DIR = ROOT / "private_task1" / "submissions" / "team_09205"
ANCHOR_DIR = TEAM_DIR / "anchors"
REPORT_DIR = ROOT / "private_task1" / "reports" / "team_09205"
FINGERPRINT_PATH = REPORT_DIR / "v2_rank1_fingerprint_189.jsonl"
SEARCH_REPORT_PATH = REPORT_DIR / "v2_candidate_fingerprint_search.json"
INVERSE_REPORT_PATH = REPORT_DIR / "v2_inverse_identifiability_audit.json"
MANIFEST_PATH = ANCHOR_DIR / "submission_v2_manifest.json"
RECOVERED_PATH = ANCHOR_DIR / "submission_v2.zip"

GUARDED_PATH = (
    ROOT
    / "private_task1"
    / "submissions"
    / "sprint48_guarded_direct"
    / "submission_private_guarded_direct.zip"
)
SAFEGUARD_PATH = TEAM_DIR / "submission_private_dual_anchor_guarded_09198.zip"
RRF_PATH = TEAM_DIR / "submission_private_constrained_dual_anchor_rrf_09205.zip"
PRIVATE_INPUT = ROOT / "private_task1" / "input" / "private-official.json"
CURRENT_K20 = ROOT / "private_task1" / "rerank" / "worklists" / "private_rrf_k20_bge_worklist.jsonl"
RRF_BUILDER_PATH = (
    ROOT
    / "private_task1"
    / "scripts"
    / "analysis"
    / "team_09205"
    / "build_constrained_dual_anchor_rrf_submission.py"
)

SEARCH_ROOTS = (
    ROOT / "private_task1" / "submissions",
    ROOT / "submissions",
    ROOT / "outputs" / "task1",
    ROOT / "outputs" / "task1" / "private",
    ROOT / "artifacts" / "task1",
    ROOT / "reports" / "task1",
)
ALLOWED_SUFFIXES = {".zip", ".json", ".jsonl"}
# A complete 2080 x Top5 submission is roughly 0.1--0.4 MB.  The bounds
# intentionally exclude score/q-doc dumps and model metadata, which cannot be
# an answer-list artifact and must not be parsed during this forensic search.
MIN_CANDIDATE_BYTES = 40 * 1024
MAX_CANDIDATE_BYTES = 1024 * 1024
INVERSE_SOLUTION_CAP = 2
INVERSE_ATTEMPT_CAP = 20_000
EXPECTED_RRF_SHA = "aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_submission_mapping(raw: Any) -> dict[str, dict[str, list[str]]] | None:
    """Return a strict 2080-query Top5 mapping, or None for another schema."""

    if isinstance(raw, dict) and isinstance(raw.get("predictions"), dict):
        raw = raw["predictions"]
    if not isinstance(raw, dict) or len(raw) != 2080:
        return None
    result: dict[str, dict[str, list[str]]] = {}
    for qid, row in raw.items():
        answer = row.get("answer") if isinstance(row, dict) else None
        if not isinstance(answer, list) or len(answer) != 5:
            return None
        documents = [str(value) for value in answer]
        if any(not value for value in documents) or len(documents) != len(set(documents)):
            return None
        result[str(qid)] = {"answer": documents}
    return result


def load_zip_submission(path: Path) -> dict[str, dict[str, list[str]]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise RuntimeError(f"expected one JSON member in {path}, got {members}")
        raw = json.loads(archive.read(members[0]).decode("utf-8-sig"))
    result = normalize_submission_mapping(raw)
    if result is None:
        raise RuntimeError(f"invalid submission shape: {path}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_submission_zip(payload: dict[str, dict[str, list[str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", encoded)


def derive_fingerprint(
    guarded: dict[str, dict[str, list[str]]], safeguard: dict[str, dict[str, list[str]]]
) -> dict[str, str]:
    if set(guarded) != set(safeguard):
        raise RuntimeError("guarded/safeguard query universe mismatch")
    fingerprint: dict[str, str] = {}
    for qid in sorted(guarded, key=numeric_id):
        before, after = guarded[qid]["answer"], safeguard[qid]["answer"]
        if before[:4] != after[:4]:
            raise RuntimeError(f"safeguard violated immutable guarded ranks 1-4: {qid}")
        changed = set(before) != set(after)
        slot_change = before[4] != after[4]
        if changed != slot_change:
            raise RuntimeError(f"safeguard change shape is not rank-5-only: {qid}")
        if changed:
            inferred = after[4]
            if inferred in before:
                raise RuntimeError(f"safeguard injection is already in guarded Top5: {qid}")
            fingerprint[qid] = inferred
    if len(fingerprint) != 189:
        raise RuntimeError(f"expected 189 safeguard injections, got {len(fingerprint)}")
    return fingerprint


def source_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def extract_jsonl_submission(path: Path) -> dict[str, dict[str, list[str]]] | None:
    result: dict[str, dict[str, list[str]]] = {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return None
                qid = row.get("query_id", row.get("question_id", row.get("id")))
                answer = row.get("answer", row.get("top5", row.get("documents")))
                if qid is None or not isinstance(answer, list) or len(answer) != 5:
                    return None
                docs = [str(value) for value in answer]
                if any(not value for value in docs) or len(docs) != len(set(docs)):
                    return None
                qid = str(qid)
                if qid in result:
                    return None
                result[qid] = {"answer": docs}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return result if len(result) == 2080 else None


def extract_candidate(path: Path) -> tuple[dict[str, dict[str, list[str]]] | None, str | None]:
    """Read only compact files that can actually store a 2080-query Top5."""

    try:
        if path.stat().st_size > MAX_CANDIDATE_BYTES:
            return None, None
        if path.suffix.casefold() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.endswith(".json")]
                if len(members) != 1 or archive.getinfo(members[0]).file_size > MAX_CANDIDATE_BYTES:
                    return None, None
                raw = json.loads(archive.read(members[0]).decode("utf-8-sig"))
            return normalize_submission_mapping(raw), "zip_json_submission"
        if path.suffix.casefold() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            return normalize_submission_mapping(raw), "json_submission"
        if path.suffix.casefold() == ".jsonl":
            return extract_jsonl_submission(path), "jsonl_submission"
    except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile):
        return None, None
    return None, None


def fingerprint_stats(
    candidate: dict[str, dict[str, list[str]]],
    guarded: dict[str, dict[str, list[str]]],
    fingerprint: dict[str, str],
) -> dict[str, Any]:
    coverage = set(candidate) == set(guarded)
    injection_set = {
        qid for qid in guarded if coverage and candidate[qid]["answer"][0] not in guarded[qid]["answer"]
    }
    expected = set(fingerprint)
    doc_match = sum(
        candidate[qid]["answer"][0] == fingerprint[qid]
        for qid in expected
        if qid in candidate
    )
    return {
        "query_coverage": len(candidate),
        "coverage_exact": coverage,
        "injection_set_size": len(injection_set),
        "injection_query_id_match": len(injection_set & expected),
        "injection_query_id_extra": len(injection_set - expected),
        "injection_query_id_missing": len(expected - injection_set),
        "inferred_rank1_doc_match": doc_match,
        "rank1_exact": coverage
        and len(injection_set) == 189
        and injection_set == expected
        and doc_match == 189,
    }


def iterate_search_files() -> Iterable[Path]:
    seen: set[Path] = set()
    excluded_directories = {
        ".cache", "__pycache__", ".worker_jobs", "cache", "checkpoint",
        "checkpoints", "chunks", "documents", "parents", "dense_faiss",
        "models", "model", "results", "tokenizer", "volume",
        "archive", "evaluation", "large_outputs", "full_doc_qwen_optimized_outputs",
        "full_doc_qwen_optimized_merged", "modal_stage_bge_slot45",
        "modal_stage_bge_top5", "qwen_teacher_bge_student", "qwen_to_bge_minimal",
        "qwen_to_bge_salvage", "training", "workflow_a", "workflow_b", "workflow_c",
        "data", "raw", "processed", "corpus",
    }
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        try:
            command = ["rg", "--files", "--no-ignore", str(root)]
            for suffix in ALLOWED_SUFFIXES:
                command.extend(["-g", f"*{suffix}"])
            for directory in excluded_directories:
                command.extend(["-g", f"!**/{directory}/**"])
            command.extend(["-g", "!**/.tmp_pv1_bge_ft*/**"])
            discovered = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, check=False
            )
            for raw_path in discovered.stdout.splitlines():
                path = Path(raw_path)
                try:
                    resolved = path.absolute()
                    size = path.stat().st_size
                    if (
                        resolved in seen
                        or (path.suffix.casefold() != ".zip" and size < MIN_CANDIDATE_BYTES)
                        or size > MAX_CANDIDATE_BYTES
                    ):
                        continue
                    seen.add(resolved)
                    yield resolved
                except OSError:
                    continue
        except OSError:
            continue


def git_history_candidates() -> tuple[list[dict[str, Any]], int]:
    """Read Git metadata only; no checkout/restoration is permitted."""

    try:
        completed = subprocess.run(
            [
                "git", "log", "--all", "--name-only", "--pretty=format:", "--",
                "private_task1/submissions", "submissions", "outputs/task1",
                "artifacts/task1", "reports/task1",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return [], 0
    if completed.returncode != 0:
        return [], 0
    paths = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    relevant = [
        path
        for path in paths
        if path.startswith((
            "private_task1/submissions/", "submissions/", "outputs/task1/", "artifacts/task1/", "reports/task1/"
        ))
        and Path(path).suffix.casefold() in ALLOWED_SUFFIXES
        # Git search is explicitly term-routed in the recovery contract.  The
        # content-addressed scan above remains filename-agnostic.
        and any(term in path.casefold() for term in ("submission", "private", "rrf", "anchor", "v2"))
    ]
    # A path name is not a candidate by itself.  Resolve each named path at the
    # oldest/newest commits that touched it; only small blobs are materialized.
    candidates: list[dict[str, Any]] = []
    for path in relevant:
        commits = subprocess.run(
            ["git", "log", "--all", "--format=%H", "--", path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        for commit in sorted(set(commits)):
            spec = f"{commit}:{path}"
            size = subprocess.run(
                ["git", "cat-file", "-s", spec], cwd=ROOT, capture_output=True, text=True, check=False
            )
            if size.returncode != 0 or not size.stdout.strip().isdigit() or int(size.stdout) > MAX_CANDIDATE_BYTES:
                continue
            content = subprocess.run(
                ["git", "show", spec], cwd=ROOT, capture_output=True, check=False
            )
            if content.returncode != 0:
                continue
            # Git-tracked ZIP blobs are intentionally not decoded here: no such
            # submission blob exists in this repository's history. JSON/JSONL
            # are parsed in an isolated in-memory buffer.
            if Path(path).suffix.casefold() == ".zip":
                continue
            try:
                if Path(path).suffix.casefold() == ".json":
                    payload = normalize_submission_mapping(json.loads(content.stdout.decode("utf-8-sig")))
                else:
                    temp = REPORT_DIR / ".git_candidate_probe.jsonl"
                    # Never write a historical blob to the working tree; use a
                    # line parser in memory instead.
                    rows: dict[str, dict[str, list[str]]] = {}
                    for line in content.stdout.decode("utf-8-sig").splitlines():
                        row = json.loads(line)
                        qid = row.get("query_id", row.get("question_id", row.get("id")))
                        answer = row.get("answer", row.get("top5", row.get("documents")))
                        if qid is None or not isinstance(answer, list) or len(answer) != 5:
                            rows = {}
                            break
                        docs = [str(value) for value in answer]
                        if any(not value for value in docs) or len(docs) != len(set(docs)):
                            rows = {}
                            break
                        rows[str(qid)] = {"answer": docs}
                    payload = rows if len(rows) == 2080 else None
            except (UnicodeError, json.JSONDecodeError):
                payload = None
            if payload is not None:
                candidates.append({"path": f"git:{spec}", "kind": "git_blob", "payload": payload, "sha256": hashlib.sha256(content.stdout).hexdigest()})
    return candidates, len(relevant)


def rebuild_safeguard(
    guarded: dict[str, dict[str, list[str]]], candidate: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for qid in sorted(guarded, key=numeric_id):
        answer = list(guarded[qid]["answer"])
        if candidate[qid]["answer"][0] not in answer:
            answer[4] = candidate[qid]["answer"][0]
        result[qid] = {"answer": answer}
    return result


def compare(left: dict[str, dict[str, list[str]]], right: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("checksum query universe mismatch")
    order = [qid for qid in sorted(left, key=numeric_id) if left[qid]["answer"] != right[qid]["answer"]]
    membership = [
        qid for qid in sorted(left, key=numeric_id)
        if set(left[qid]["answer"]) != set(right[qid]["answer"])
    ]
    return {
        "queries": len(left),
        "set_mismatches": len(membership),
        "order_mismatches": len(order),
        "payload_equal": left == right,
        "first_20_set_mismatch_query_ids": membership[:20],
        "first_20_order_mismatch_query_ids": order[:20],
    }


def load_current_k20() -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with CURRENT_K20.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["query_id"])].add(str(row["document_id"]))
    if len(result) != 2080 or any(len(docs) != 20 for docs in result.values()):
        raise RuntimeError("current K20 shape is not 2080 x 20")
    return result


def constrained_rrf_one(
    builder: Any, qid: str, guarded: dict[str, list[str]], v2: list[str]
) -> list[str] | None:
    try:
        result, _g_replaced, _v_injected = builder.build(
            {qid: {"answer": guarded}}, {qid: {"answer": v2}}
        )
    except RuntimeError:
        return None
    return result[qid]["answer"]


def inverse_identifiability(
    builder: Any,
    guarded: dict[str, dict[str, list[str]]],
    safeguard: dict[str, dict[str, list[str]]],
    target_rrf: dict[str, dict[str, list[str]]],
    fingerprint: dict[str, str],
) -> tuple[dict[str, dict[str, list[str]]] | None, dict[str, Any]]:
    """Bounded exact search in the declared K20∪anchor-output universe.

    Results marked ambiguous include a second exact witness or an exhausted
    bounded search budget.  The latter is deliberately conservative: it is
    never promoted as a unique recovered V2 list.
    """

    k20 = load_current_k20()
    details: list[dict[str, Any]] = []
    unique_payload: dict[str, dict[str, list[str]]] = {}
    all_unique = True
    for qid in sorted(guarded, key=numeric_id):
        g = guarded[qid]["answer"]
        target = target_rrf[qid]["answer"]
        pool = set(k20[qid]) | set(g) | set(safeguard[qid]["answer"]) | set(target)
        rank1_options = [fingerprint[qid]] if qid in fingerprint else list(g)
        solutions: list[list[str]] = []
        attempts = 0
        capped = False
        for rank1 in rank1_options:
            required = set(target) - set(g)
            if rank1 not in target or rank1 not in pool:
                continue
            required.discard(rank1)
            if len(required) > 4:
                continue
            free = 4 - len(required)
            available = sorted(pool - required - {rank1}, key=numeric_id)
            for extras in itertools.combinations(available, free):
                for tail in itertools.permutations((*sorted(required, key=numeric_id), *extras)):
                    attempts += 1
                    if attempts > INVERSE_ATTEMPT_CAP:
                        capped = True
                        break
                    v2 = [rank1, *tail]
                    output = constrained_rrf_one(builder, qid, g, v2)
                    if output == target:
                        solutions.append(v2)
                        if len(solutions) >= INVERSE_SOLUTION_CAP:
                            break
                if capped or len(solutions) >= INVERSE_SOLUTION_CAP:
                    break
            if capped or len(solutions) >= INVERSE_SOLUTION_CAP:
                break
        if len(solutions) >= 2:
            classification = "MULTIPLE_SOLUTIONS"
        elif capped:
            classification = "AMBIGUOUS_SEARCH_CAP"
        elif len(solutions) == 1:
            classification = "UNIQUE_SOLUTION"
            unique_payload[qid] = {"answer": solutions[0]}
        else:
            classification = "NO_SOLUTION"
        if classification != "UNIQUE_SOLUTION":
            all_unique = False
        details.append({
            "query_id": qid,
            "classification": classification,
            "solution_count_capped_at": len(solutions),
            "attempts": attempts,
            "candidate_universe_size": len(pool),
        })
    summary = {
        "status": "ALL_UNIQUE" if all_unique else "NOT_IDENTIFIABLE",
        "universe": "current_private_K20 UNION guarded_top5 UNION safeguard_top5 UNION constrained_rrf_top5",
        "solution_count_cap": INVERSE_SOLUTION_CAP,
        "attempt_cap_per_query": INVERSE_ATTEMPT_CAP,
        "queries": len(details),
        "unique_solution_queries": sum(item["classification"] == "UNIQUE_SOLUTION" for item in details),
        "ambiguous_queries": sum(item["classification"] in {"MULTIPLE_SOLUTIONS", "AMBIGUOUS_SEARCH_CAP"} for item in details),
        "no_solution_queries": sum(item["classification"] == "NO_SOLUTION" for item in details),
        "details": details,
    }
    return (unique_payload if all_unique else None), summary


def main() -> int:
    for required in (GUARDED_PATH, SAFEGUARD_PATH, RRF_PATH, PRIVATE_INPUT, CURRENT_K20, RRF_BUILDER_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256(RRF_PATH) != EXPECTED_RRF_SHA:
        raise RuntimeError("supplied 0.92054 artifact SHA mismatch")
    guarded = load_zip_submission(GUARDED_PATH)
    safeguard = load_zip_submission(SAFEGUARD_PATH)
    target_rrf = load_zip_submission(RRF_PATH)
    private_ids = set(json.loads(PRIVATE_INPUT.read_text(encoding="utf-8-sig")))
    if set(guarded) != private_ids:
        raise RuntimeError("guarded/private query universe mismatch")

    fingerprint = derive_fingerprint(guarded, safeguard)
    write_jsonl(
        FINGERPRINT_PATH,
        ({"query_id": qid, "guarded_top5": guarded[qid]["answer"], "inferred_v2_rank1": fingerprint[qid]} for qid in sorted(fingerprint, key=numeric_id)),
    )
    fingerprint_sha = sha256(FINGERPRINT_PATH)

    scanned = 0
    candidates: list[dict[str, Any]] = []
    seen_payloads: set[str] = set()
    for path in iterate_search_files():
        scanned += 1
        payload, kind = extract_candidate(path)
        if payload is None:
            continue
        signature = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        stats = fingerprint_stats(payload, guarded, fingerprint)
        candidate = {
            "path": source_rel(path),
            "kind": kind,
            "source_sha256": sha256(path),
            "payload_signature_sha256": signature,
            "duplicate_payload_of_prior_candidate": signature in seen_payloads,
            "fingerprint": stats,
            "payload": payload,
        }
        seen_payloads.add(signature)
        candidates.append(candidate)
    git_candidates, git_paths_examined = git_history_candidates()
    for candidate in git_candidates:
        candidate["payload_signature_sha256"] = hashlib.sha256(
            json.dumps(candidate["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        candidate["fingerprint"] = fingerprint_stats(candidate["payload"], guarded, fingerprint)
        candidate["duplicate_payload_of_prior_candidate"] = candidate["payload_signature_sha256"] in seen_payloads
        seen_payloads.add(candidate["payload_signature_sha256"])
        candidates.append(candidate)

    rank1_exact = [candidate for candidate in candidates if candidate["fingerprint"]["rank1_exact"]]
    handoff_builder = load_module("team09205_fingerprint_rrf", RRF_BUILDER_PATH)
    survivors: list[dict[str, Any]] = []
    for candidate in rank1_exact:
        safeguard_replay = rebuild_safeguard(guarded, candidate["payload"])
        safeguard_check = compare(safeguard_replay, safeguard)
        candidate["safeguard_checksum"] = safeguard_check
        if safeguard_check["set_mismatches"] != 0:
            continue
        rrf_replay, _guarded_replaced, _v2_injected = handoff_builder.build(guarded, candidate["payload"])
        rrf_check = compare(rrf_replay, target_rrf)
        candidate["rrf_09205_checksum"] = rrf_check
        if rrf_check["set_mismatches"] == 0:
            survivors.append(candidate)

    search_report = {
        "status": "EXACT_CANDIDATE_FOUND" if len(survivors) == 1 else "NO_UNIQUE_EXACT_FILE_CANDIDATE",
        "fingerprint": {"queries": len(fingerprint), "sha256": fingerprint_sha, "path": source_rel(FINGERPRINT_PATH)},
        "filesystem_candidate_files_scanned": scanned,
        "structural_2080_candidates": len(candidates),
        "git_historical_candidate_paths_examined": git_paths_examined,
        "git_historical_candidates": len(git_candidates),
        "rank1_exact_candidates": len(rank1_exact),
        "candidates": [{key: value for key, value in candidate.items() if key != "payload"} for candidate in candidates],
        "exact_checksum_survivor_count": len(survivors),
    }
    write_json(SEARCH_REPORT_PATH, search_report)

    recovered: dict[str, Any] | None = None
    inverse_summary: dict[str, Any] | None = None
    if len(survivors) == 1:
        winner = survivors[0]
        if winner["kind"] == "zip_json_submission" and not winner["path"].startswith("git:"):
            shutil.copyfile(ROOT / winner["path"], RECOVERED_PATH)
        else:
            write_submission_zip(winner["payload"], RECOVERED_PATH)
        recovered = winner
    elif not survivors:
        inverse_payload, inverse_summary = inverse_identifiability(
            handoff_builder, guarded, safeguard, target_rrf, fingerprint
        )
        write_json(INVERSE_REPORT_PATH, inverse_summary)
        if inverse_payload is not None:
            # This branch is intentionally conditional: only a proof of one
            # V2 Top5 per query may be materialized.
            write_submission_zip(inverse_payload, RECOVERED_PATH)
            safe_check = compare(rebuild_safeguard(guarded, inverse_payload), safeguard)
            rrf_replay, _g, _v = handoff_builder.build(guarded, inverse_payload)
            rrf_check = compare(rrf_replay, target_rrf)
            if safe_check["set_mismatches"] == 0 and rrf_check["set_mismatches"] == 0:
                recovered = {
                    "path": source_rel(INVERSE_REPORT_PATH),
                    "kind": "uniquely_identified_inverse",
                    "source_sha256": None,
                    "payload": inverse_payload,
                    "fingerprint": fingerprint_stats(inverse_payload, guarded, fingerprint),
                    "safeguard_checksum": safe_check,
                    "rrf_09205_checksum": rrf_check,
                }
            else:
                RECOVERED_PATH.unlink(missing_ok=True)

    if recovered is not None:
        final_payload = load_zip_submission(RECOVERED_PATH)
        manifest = {
            "status": "V2_ANCHOR_LINEAGE_GATE_PASS",
            "source_type": "RECOVERED_BY_CONTENT_FINGERPRINT_AND_DUAL_CHECKSUM",
            "original_recovered_source_path": recovered["path"],
            "source_sha256": recovered["source_sha256"],
            "recovered_v2_sha256": sha256(RECOVERED_PATH),
            "fingerprint_sha256": fingerprint_sha,
            "rank1_fingerprint_match": "189/189",
            "extra_injections": 0,
            "missing_injections": 0,
            "safeguard_09198_checksum": recovered["safeguard_checksum"],
            "rrf_09205_checksum": recovered["rrf_09205_checksum"],
            "private_oracle_matching": 0,
            "private_labels_used": False,
            "query_count": len(final_payload),
        }
        write_json(MANIFEST_PATH, manifest)

    final = {
        "v2_fingerprint_queries": len(fingerprint),
        "v2_fingerprint_sha256": fingerprint_sha,
        "filesystem_candidates_scanned": scanned,
        "structural_2080_candidates": len(candidates),
        "rank1_exact_candidates": len(rank1_exact),
        "best_candidate_path": survivors[0]["path"] if len(survivors) == 1 else "NONE",
        "best_candidate_injection_set": (
            f"{survivors[0]['fingerprint']['injection_set_size']}/189" if len(survivors) == 1 else "N/A"
        ),
        "best_candidate_rank1_doc_match": (
            f"{survivors[0]['fingerprint']['inferred_rank1_doc_match']}/189" if len(survivors) == 1 else "N/A"
        ),
        "safeguard_set_mismatches": survivors[0]["safeguard_checksum"]["set_mismatches"] if len(survivors) == 1 else "N/A",
        "rrf_09205_set_mismatches": survivors[0]["rrf_09205_checksum"]["set_mismatches"] if len(survivors) == 1 else "N/A",
        "rrf_09205_order_mismatches": survivors[0]["rrf_09205_checksum"]["order_mismatches"] if len(survivors) == 1 else "N/A",
        "git_historical_candidates": len(git_candidates),
        "inverse_identifiability_run": inverse_summary is not None,
        "unique_inverse_queries": inverse_summary["unique_solution_queries"] if inverse_summary else "N/A",
        "ambiguous_inverse_queries": inverse_summary["ambiguous_queries"] if inverse_summary else "N/A",
        "no_solution_queries": inverse_summary["no_solution_queries"] if inverse_summary else "N/A",
        "v2_anchor_lineage_gate": "PASS" if recovered is not None else "FAIL",
        "recovered_v2_path": source_rel(RECOVERED_PATH) if recovered is not None else "N/A",
        "recovered_v2_sha256": sha256(RECOVERED_PATH) if recovered is not None else "N/A",
        "f1_f4_generator_provenance": "NOT_REACHED" if recovered is None else "PENDING_TRACE",
        "fusion_oof_recall": "N/A",
        "selector_oof_recall": "N/A",
        "current_private_incumbent": 0.920544597,
        "auto_submit": False,
        "final_status": "READY_TO_RESUME_SELECTOR" if recovered is not None else "BLOCKED_V2_NOT_IDENTIFIABLE",
        "next_action": "RESUME_STAGE_3_F1_F4_SELECTOR" if recovered is not None else "KEEP_09205_INCUMBENT",
    }
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if recovered is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
