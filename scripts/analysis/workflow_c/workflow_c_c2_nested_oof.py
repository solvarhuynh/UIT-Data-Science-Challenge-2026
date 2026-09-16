"""Frozen C2 nested-OOF harness for Task1 Workflow C.

Lifecycle: REPLACE_CANONICAL / KEEP_ACTIVE.  This module implements the
approved C2 mechanics. The preflight contract remains PREFLIGHT_ONLY forever;
--execute is reachable only with an explicitly supplied, manually issued
governance artifact bound to this exact script and frozen inputs. The artifact
enforces workflow sequencing and auditability, not cryptographic authenticity
against a malicious same-user local operator (out of scope).
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_REL = Path("reports/task1/workflow_c/shared/c2/c2_preflight_contract.json")
PROVENANCE_REL = Path("reports/task1/workflow_c/shared/c2/c2_input_provenance_manifest.json")
EXECUTION_ROOT_REL = Path("reports/task1/workflow_c/shared/c2/execution")
PINNED_PYTHON_REL = Path(".venv/Scripts/python.exe")

# Raw-byte identities, intentionally not merely JSON status checks.
EXPECTED_CONTRACT_SHA256 = "9c49acbb3c90630861b9fadd759a66bf964f8d8466fe74d6e6c2038bc303265f"
EXPECTED_PROVENANCE_SHA256 = "ec131ea6e3ef2ab9665814abd9d5b867b4bf873140106f24f05be31b049a6943"
AUTHORIZATION_SCHEMA = "workflow-c-c2-manual-execution-authorization-v1"
AUTHORIZATION_RECEIPTS_REL = EXECUTION_ROOT_REL / "authorization_receipts"
AUTHORIZATION_SESSIONS_REL = EXECUTION_ROOT_REL / "authorization_sessions"
_LIVE_AUTHORIZATION_IDS: set[str] = set()
_LIVE_SESSION_IDS: set[str] = set()

REQUIRED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}
REQUIRED_VERSIONS = {
    "python": "3.12.6",
    "numpy": "1.26.4",
    "scikit_learn": "1.7.2",
    "threadpoolctl": "3.6.0",
}
TARGET_FOLDS = (1, 2, 3, 4)
EXPECTED_QUERY_COUNT = 5600
EXPECTED_ACTION_COUNT = 172338
EXPECTED_FEATURE_COUNT = 58
MAXIMUM_PLANNED_FITS = 32
PER_FIT_TIMEOUT_SECONDS = 15 * 60
TOTAL_RUN_TIMEOUT_SECONDS = 8 * 60 * 60
RSS_HARD_CEILING_BYTES = 8 * 1024**3
MINIMUM_AVAILABLE_MEMORY_BYTES = 4 * 1024**3
EXACT_DELTA_FIELD = "gain"
SERIALIZATION_CONTRACT_ID = "workflow-c-c2-jsonl-v1-f1-f4-stored-validation-order"
FORBIDDEN_SCORE_FIELDS = frozenset({
    "gold_documents", "gold", "relevance", "relevance_labels", "label",
    "exact_delta_recall", "gain", "BENEFIT", "HARM", "NEUTRAL",
    "target", "truth_delta", "recall", "precision", "baseline_recall",
    "policy_recall", "baseline_precision", "policy_precision",
})


class C2HarnessError(RuntimeError):
    """A hard, non-recoverable C2 provenance or contract violation."""


def fail(code: str, detail: str = "") -> None:
    message = code if not detail else f"{code}: {detail}"
    raise C2HarnessError(message)


def require_scientific_fold(fold: int, boundary: str) -> None:
    """Fail closed after the one approved structural Fold0 exclusion point."""
    if fold == 0:
        fail("FOLD0_SCIENTIFIC_LEAKAGE", boundary)
    if fold not in TARGET_FOLDS:
        fail("UNKNOWN_SCIENTIFIC_FOLD", f"{boundary}: fold={fold}")


def structural_filter_f1_f4(rows: Iterable[Mapping[str, Any]], boundary: str) -> list[Mapping[str, Any]]:
    """The only structural Fold0 filter; every downstream builder must assert."""
    filtered: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as exc:
            fail("UNKNOWN_SCIENTIFIC_FOLD", f"{boundary}: {exc}")
        if fold == 0:
            continue
        require_scientific_fold(fold, boundary)
        filtered.append(row)
    return filtered


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def running_script_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def canonical_json_bytes(value: Any) -> bytes:
    """The only future metadata serialization allowed by this harness."""
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read_verified_json(relative_path: Path, expected_sha256: str, mismatch_code: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        fail(mismatch_code, f"missing {relative_path.as_posix()}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        fail(mismatch_code, f"expected {expected_sha256}, got {actual}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(mismatch_code, f"invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(mismatch_code, "root must be a JSON object")
    return value


def load_frozen_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_verified_json(CONTRACT_REL, EXPECTED_CONTRACT_SHA256, "C2_CONTRACT_HASH_MISMATCH")
    provenance = read_verified_json(PROVENANCE_REL, EXPECTED_PROVENANCE_SHA256, "C2_PROVENANCE_MANIFEST_HASH_MISMATCH")
    validate_contract_shape(contract)
    return contract, provenance


def validate_contract_shape(contract: Mapping[str, Any]) -> None:
    if contract.get("status") != "PASS" or contract.get("contract_governance") != "PREFLIGHT_CONTRACT_RESOLVED":
        fail("C2_CONTRACT_INVALID", "preflight contract is not resolved PASS")
    if contract.get("authorization") != "PREFLIGHT_ONLY":
        fail("C2_CONTRACT_INVALID", "frozen preflight contract must remain PREFLIGHT_ONLY")
    population = contract.get("scientific_population")
    if not isinstance(population, Mapping) or population.get("core_universe") != "K20_ONLY":
        fail("C2_CONTRACT_INVALID", "K20_ONLY core universe required")
    if population.get("included_folds") != list(TARGET_FOLDS) or population.get("excluded_folds") != [0]:
        fail("C2_CONTRACT_INVALID", "F1-F4 only population required")
    if population.get("query_count") != EXPECTED_QUERY_COUNT or population.get("k20_action_count") != EXPECTED_ACTION_COUNT:
        fail("C2_CONTRACT_INVALID", "frozen K20 population count mismatch")
    features = contract.get("feature_order")
    if not isinstance(features, Mapping) or features.get("feature_count") != EXPECTED_FEATURE_COUNT:
        fail("C2_CONTRACT_INVALID", "58-feature contract required")
    names = features.get("ordered_names")
    if not isinstance(names, list) or len(names) != EXPECTED_FEATURE_COUNT or len(set(names)) != EXPECTED_FEATURE_COUNT:
        fail("C2_CONTRACT_INVALID", "frozen feature order invalid")
    if tuple(names) != frozen_feature_names(contract):
        fail("C2_CONTRACT_INVALID", "feature order is not internally stable")
    expected_outer = {
        "F1": {"outer_validation_fold": 1, "outer_training_folds": [2, 3, 4]},
        "F2": {"outer_validation_fold": 2, "outer_training_folds": [1, 3, 4]},
        "F3": {"outer_validation_fold": 3, "outer_training_folds": [1, 2, 4]},
        "F4": {"outer_validation_fold": 4, "outer_training_folds": [1, 2, 3]},
    }
    nested = contract.get("nested_split_contract")
    if not isinstance(nested, Mapping) or nested.get("outer_splits") != expected_outer:
        fail("C2_CONTRACT_INVALID", "outer split definition mismatch")
    if contract.get("c2_v_control", {}).get("name") != "C2V_FRESH_HGBC_1_7_2_K20":
        fail("C2_CONTRACT_INVALID", "fresh C2-V identity mismatch")
    if contract.get("c2_r_challenger", {}).get("formulation") != "SWAP_ONLY_EXACT_DELTA_RECALL_REGRESSION":
        fail("C2_CONTRACT_INVALID", "C2-R formulation mismatch")


def strict_json_object(raw: bytes, code: str) -> dict[str, Any]:
    """Parse a governance artifact without accepting duplicate JSON keys."""
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(code, f"duplicate key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(code, f"invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(code, "root must be an object")
    return value


@dataclass(frozen=True)
class ManualAuthorization:
    authorization_id: str
    session_id: str
    artifact_sha256: str
    execution_root: Path


def _authorization_timestamp(value: Any, name: str) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        fail("C2_AUTHORIZATION_INVALID", f"{name} must be an RFC3339 UTC Z timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        fail("C2_AUTHORIZATION_INVALID", f"invalid {name}: {exc}")
    if parsed.tzinfo != dt.timezone.utc:
        fail("C2_AUTHORIZATION_INVALID", f"{name} must be UTC")
    return parsed


def _consumption_marker_path(state_root: Path, directory: str, identity: str, identity_name: str) -> Path:
    """Construct an internal, traversal-safe durable-consumption marker path."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", identity):
        fail("C2_AUTHORIZATION_INVALID", f"{identity_name} format invalid")
    root = state_root.resolve()
    path = (root / directory / f"{identity}.json").resolve()
    if root not in path.parents:
        fail("C2_AUTHORIZATION_INVALID", f"{identity_name} marker escapes authorization state root")
    return path


def authorization_receipt_path(execution_root: Path, authorization_id: str) -> Path:
    return _consumption_marker_path(execution_root, "authorization_receipts", authorization_id, "authorization_id")


def session_receipt_path(execution_root: Path, session_id: str) -> Path:
    return _consumption_marker_path(execution_root, "authorization_sessions", session_id, "session_id")


def _assert_consumption_unused(execution_root: Path, authorization_id: str, session_id: str) -> None:
    authorization_receipt = authorization_receipt_path(execution_root, authorization_id)
    session_receipt = session_receipt_path(execution_root, session_id)
    if (authorization_id in _LIVE_AUTHORIZATION_IDS or session_id in _LIVE_SESSION_IDS
            or authorization_receipt.exists() or session_receipt.exists()):
        fail("C2_AUTHORIZATION_REUSED", "authorization/session has already been consumed")


def validate_manual_authorization(path_text: str) -> ManualAuthorization:
    """Validate an explicitly supplied manual artifact before any Task1 load."""
    if not isinstance(path_text, str) or not path_text:
        fail("C2_AUTHORIZATION_REQUIRED", "--execute requires --authorization-artifact PATH")
    path = Path(path_text).resolve()
    if not path.is_file():
        fail("C2_AUTHORIZATION_INVALID", "authorization artifact is missing")
    raw = path.read_bytes()
    artifact = strict_json_object(raw, "C2_AUTHORIZATION_INVALID")
    required = {
        "schema", "decision", "authorization_id", "session_id", "script_sha256", "contract_sha256",
        "provenance_sha256", "mode", "execution_root", "issued_at_utc", "expires_at_utc", "c2_v_fit_max",
        "c2_r_fit_max", "aggregate_fit_max", "concurrency_max", "retry_allowed", "fallback_allowed",
        "gpu_allowed", "modal_allowed", "fold0_scientific_use_allowed", "public_labels_allowed", "single_run",
    }
    if set(artifact) != required:
        fail("C2_AUTHORIZATION_INVALID", "authorization schema keys do not match exactly")
    if artifact.get("decision") != "AUTHORIZE_C2_EXECUTION" or artifact.get("mode") != "C2":
        fail("C2_AUTHORIZATION_INVALID", "decision or mode is invalid")
    # These raw-byte checks precede all execution/orchestration and bind the manual decision to frozen inputs.
    if sha256_file(ROOT / CONTRACT_REL) != EXPECTED_CONTRACT_SHA256:
        fail("C2_CONTRACT_HASH_MISMATCH", "frozen contract differs from its reviewed identity")
    if sha256_file(ROOT / PROVENANCE_REL) != EXPECTED_PROVENANCE_SHA256:
        fail("C2_PROVENANCE_MANIFEST_HASH_MISMATCH", "frozen provenance differs from its reviewed identity")
    expected_values = {
        "schema": AUTHORIZATION_SCHEMA, "decision": "AUTHORIZE_C2_EXECUTION", "mode": "C2",
        "script_sha256": running_script_sha256(), "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "provenance_sha256": EXPECTED_PROVENANCE_SHA256, "c2_v_fit_max": 16, "c2_r_fit_max": 16,
        "aggregate_fit_max": 32, "concurrency_max": 1, "retry_allowed": False, "fallback_allowed": False,
        "gpu_allowed": False, "modal_allowed": False, "fold0_scientific_use_allowed": False,
        "public_labels_allowed": False, "single_run": True,
    }
    for key, expected in expected_values.items():
        if artifact.get(key) != expected or type(artifact.get(key)) is not type(expected):
            fail("C2_AUTHORIZATION_INVALID", f"{key} does not match the frozen execution policy")
    authorization_id, session_id = artifact.get("authorization_id"), artifact.get("session_id")
    if (not isinstance(authorization_id, str) or not isinstance(session_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", session_id)):
        fail("C2_AUTHORIZATION_INVALID", "authorization/session identity format invalid")
    execution_root = (ROOT / EXECUTION_ROOT_REL).resolve()
    if artifact.get("execution_root") != str(execution_root):
        fail("C2_AUTHORIZATION_INVALID", "execution_root is not canonical")
    issued, expires = _authorization_timestamp(artifact.get("issued_at_utc"), "issued_at_utc"), _authorization_timestamp(artifact.get("expires_at_utc"), "expires_at_utc")
    now = dt.datetime.now(dt.timezone.utc)
    if issued > now or expires <= issued or expires <= now:
        fail("C2_AUTHORIZATION_INVALID", "authorization is stale, expired, or not yet valid")
    _assert_consumption_unused(execution_root, authorization_id, session_id)
    return ManualAuthorization(authorization_id, session_id, hashlib.sha256(raw).hexdigest(), execution_root)


def record_authorization_consumption(authorization: ManualAuthorization) -> None:
    """A successful real execution consumes its one manual authorization before data access."""
    _assert_consumption_unused(authorization.execution_root, authorization.authorization_id, authorization.session_id)
    authorization_receipt = authorization_receipt_path(authorization.execution_root, authorization.authorization_id)
    session_receipt = session_receipt_path(authorization.execution_root, authorization.session_id)
    authorization_receipt.parent.mkdir(parents=True, exist_ok=True)
    authorization_payload = canonical_json_bytes({"authorization_id": authorization.authorization_id,
                                                  "artifact_sha256": authorization.artifact_sha256,
                                                  "script_sha256": running_script_sha256(), "session_id": authorization.session_id,
                                                  "status": "CONSUMED"})
    try:
        with authorization_receipt.open("xb") as handle:
            handle.write(authorization_payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Deliberately after authorization consumption: later failure leaves a durable fail-closed partial state.
        session_receipt.parent.mkdir(parents=True, exist_ok=True)
        session_payload = canonical_json_bytes({"authorization_id": authorization.authorization_id,
                                                "session_id": authorization.session_id,
                                                "script_sha256": running_script_sha256(),
                                                "contract_sha256": EXPECTED_CONTRACT_SHA256,
                                                "provenance_sha256": EXPECTED_PROVENANCE_SHA256,
                                                "consumed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                                                "status": "CONSUMED"})
        with session_receipt.open("xb") as handle:
            handle.write(session_payload)
            handle.flush()
            os.fsync(handle.fileno())
    except (FileExistsError, OSError) as exc:
        fail("C2_AUTHORIZATION_CONSUMPTION_FAILURE", f"durable consumption marker failure: {exc}")
    _LIVE_AUTHORIZATION_IDS.add(authorization.authorization_id)
    _LIVE_SESSION_IDS.add(authorization.session_id)


def manual_authorization_self_checks() -> bool:
    """Synthetic-only validation coverage; it never consumes an authorization or loads Task1."""
    now = dt.datetime.now(dt.timezone.utc)
    fixture = {"schema": AUTHORIZATION_SCHEMA, "decision": "AUTHORIZE_C2_EXECUTION", "authorization_id": "synthetic-auth-0001",
               "session_id": "synthetic-session-0001", "script_sha256": running_script_sha256(),
               "contract_sha256": EXPECTED_CONTRACT_SHA256, "provenance_sha256": EXPECTED_PROVENANCE_SHA256,
               "mode": "C2", "execution_root": str((ROOT / EXECUTION_ROOT_REL).resolve()),
               "issued_at_utc": (now - dt.timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
               "expires_at_utc": (now + dt.timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
               "c2_v_fit_max": 16, "c2_r_fit_max": 16, "aggregate_fit_max": 32, "concurrency_max": 1,
               "retry_allowed": False, "fallback_allowed": False, "gpu_allowed": False, "modal_allowed": False,
               "fold0_scientific_use_allowed": False, "public_labels_allowed": False, "single_run": True}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic-authorization.json"
        path.write_bytes(canonical_json_bytes(fixture))
        try:
            validate_manual_authorization(str(path))
        except C2HarnessError:
            return False
        mutations = (("schema", "wrong"), ("decision", "wrong"), ("mode", "wrong"), ("script_sha256", "0" * 64),
                     ("contract_sha256", "0" * 64), ("provenance_sha256", "0" * 64), ("execution_root", "wrong"),
                     ("c2_v_fit_max", 15), ("expires_at_utc", "2000-01-01T00:00:00Z"))
        for key, value in mutations:
            bad = dict(fixture)
            bad[key] = value
            path.write_bytes(canonical_json_bytes(bad))
            try:
                validate_manual_authorization(str(path))
            except C2HarnessError:
                continue
            return False
        path.write_bytes(b'{"schema":"x","schema":"y"}\n')
        try:
            validate_manual_authorization(str(path))
        except C2HarnessError:
            return True
    return False


def durable_session_consumption_self_checks() -> bool:
    """Synthetic restart/partial-failure coverage; it creates no governed artifact or Task1 object."""
    _LIVE_AUTHORIZATION_IDS.clear()
    _LIVE_SESSION_IDS.clear()
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ManualAuthorization("synthetic-auth-A001", "synthetic-session-S001", "a" * 64, root)
            record_authorization_consumption(first)
            if not (authorization_receipt_path(root, first.authorization_id).is_file()
                    and session_receipt_path(root, first.session_id).is_file()):
                return False
            # Simulate restart: only volatile state disappears; durable S marker must still reject B/S.
            _LIVE_AUTHORIZATION_IDS.clear()
            _LIVE_SESSION_IDS.clear()
            restart_session_rejected = False
            reused_authorization_rejected = False
            try:
                _assert_consumption_unused(root, "synthetic-auth-B001", first.session_id)
            except C2HarnessError:
                restart_session_rejected = True
            try:
                _assert_consumption_unused(root, first.authorization_id, "synthetic-session-T001")
            except C2HarnessError:
                reused_authorization_rejected = True
            _assert_consumption_unused(root, "synthetic-auth-B001", "synthetic-session-T001")
            fresh = ManualAuthorization("synthetic-auth-B001", "synthetic-session-T001", "b" * 64, root)
            record_authorization_consumption(fresh)
            duplicate_session_rejected = False
            try:
                _assert_consumption_unused(root, "synthetic-auth-C001", fresh.session_id)
            except C2HarnessError:
                duplicate_session_rejected = True
            malformed_session_rejected = False
            try:
                session_receipt_path(root, "../traversal")
            except C2HarnessError:
                malformed_session_rejected = True
            duplicate_exclusive_rejected = False
            try:
                with session_receipt_path(root, fresh.session_id).open("xb"):
                    pass
            except FileExistsError:
                duplicate_exclusive_rejected = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = ManualAuthorization("synthetic-auth-P001", "synthetic-session-P001", "c" * 64, root)
            (root / "authorization_sessions").write_text("block session marker directory", encoding="utf-8")
            partial_failed = False
            try:
                record_authorization_consumption(partial)
            except C2HarnessError:
                partial_failed = True
            partial_marker_retained = authorization_receipt_path(root, partial.authorization_id).exists()
            return (restart_session_rejected and reused_authorization_rejected and duplicate_session_rejected
                    and malformed_session_rejected and duplicate_exclusive_rejected and partial_failed
                    and partial_marker_retained)
    except (C2HarnessError, OSError):
        return False
    finally:
        _LIVE_AUTHORIZATION_IDS.clear()
        _LIVE_SESSION_IDS.clear()


def frozen_feature_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    names = contract["feature_order"]["ordered_names"]
    return tuple(str(name) for name in names)


def worker_environment() -> dict[str, str]:
    """Return the closed, explicit numerical environment for every fit worker."""
    environment = dict(os.environ)
    environment.update(REQUIRED_ENVIRONMENT)
    # Do not allow a user site package to change the frozen worker identity.
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def worker_capability_message(*, session_id: str, parent_pid: int, worker_pid: int, launch_nonce: str,
                              job_id: str, job_sha256: str, job_scope: str, job_path: Path, family: str,
                              family_ordinal: int, aggregate_ordinal: int) -> str:
    """Build a per-launch capability only after parent authorization and PID proof."""
    payload = {"kind": "C2_WORKER_CAPABILITY", "session_id": session_id, "parent_pid": parent_pid,
               "worker_pid": worker_pid, "launch_nonce": launch_nonce, "job_id": job_id,
               "job_sha256": job_sha256, "job_scope": job_scope, "job_path": str(job_path.resolve()), "family": family, "family_ordinal": family_ordinal,
               "aggregate_ordinal": aggregate_ordinal, "script_sha256": running_script_sha256(),
               "contract_sha256": EXPECTED_CONTRACT_SHA256, "provenance_sha256": EXPECTED_PROVENANCE_SHA256}
    return _protocol_line(payload)


def worker_launch_context(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Expected launch facts are inherited from the live parent process, never from stdin capability."""
    required = {"C2_EXPECTED_SESSION_ID", "C2_EXPECTED_JOB_ID", "C2_EXPECTED_JOB_SHA256",
                "C2_EXPECTED_JOB_SCOPE", "C2_EXPECTED_JOB_PATH", "C2_EXPECTED_FAMILY", "C2_EXPECTED_FAMILY_ORDINAL",
                "C2_EXPECTED_AGGREGATE_ORDINAL", "C2_EXPECTED_SCRIPT_SHA256", "C2_EXPECTED_CONTRACT_SHA256",
                "C2_EXPECTED_PROVENANCE_SHA256"}
    values = {key: os.environ.get(key, "") for key in required}
    if any(not value for value in values.values()):
        fail("WORKER_AUTHORIZATION_FAILURE", "live parent launch context is absent")
    try:
        context = {"session_id": values["C2_EXPECTED_SESSION_ID"], "parent_pid": os.getppid(),
                   "worker_pid": identity["pid"], "launch_nonce": identity["launch_nonce"], "job_id": values["C2_EXPECTED_JOB_ID"],
                   "job_sha256": values["C2_EXPECTED_JOB_SHA256"], "job_scope": values["C2_EXPECTED_JOB_SCOPE"],
                   "job_path": str(Path(values["C2_EXPECTED_JOB_PATH"]).resolve()), "family": values["C2_EXPECTED_FAMILY"],
                   "family_ordinal": int(values["C2_EXPECTED_FAMILY_ORDINAL"]),
                   "aggregate_ordinal": int(values["C2_EXPECTED_AGGREGATE_ORDINAL"]),
                   "script_sha256": values["C2_EXPECTED_SCRIPT_SHA256"], "contract_sha256": values["C2_EXPECTED_CONTRACT_SHA256"],
                   "provenance_sha256": values["C2_EXPECTED_PROVENANCE_SHA256"]}
    except (KeyError, ValueError) as exc:
        fail("WORKER_AUTHORIZATION_FAILURE", f"malformed launch context: {exc}")
    if context["parent_pid"] != identity.get("parent_pid") or context["script_sha256"] != running_script_sha256():
        fail("WORKER_AUTHORIZATION_FAILURE", "launch context parent/script identity mismatch")
    if context["contract_sha256"] != EXPECTED_CONTRACT_SHA256 or context["provenance_sha256"] != EXPECTED_PROVENANCE_SHA256:
        fail("WORKER_AUTHORIZATION_FAILURE", "launch context frozen identity mismatch")
    return context


def validate_worker_capability(raw_message: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    message = strict_json_object(raw_message.encode("utf-8"), "WORKER_AUTHORIZATION_FAILURE")
    required = {"kind", "session_id", "parent_pid", "worker_pid", "launch_nonce", "job_id", "job_sha256", "job_scope", "job_path",
                "family", "family_ordinal", "aggregate_ordinal", "script_sha256", "contract_sha256",
                "provenance_sha256"}
    if set(message) != required or message.get("kind") != "C2_WORKER_CAPABILITY":
        fail("WORKER_AUTHORIZATION_FAILURE", "capability schema invalid")
    for key in required - {"kind"}:
        if message.get(key) != expected.get(key):
            fail("WORKER_AUTHORIZATION_FAILURE", f"capability {key} does not match live launch context")
    return message


def pinned_python() -> Path:
    executable = (ROOT / PINNED_PYTHON_REL).resolve()
    if not executable.is_file():
        fail("C2_ENVIRONMENT_MISMATCH", f"pinned Python missing: {executable}")
    return executable


def require_environment_before_sklearn(require_pinned_executable: bool = False) -> tuple[Any, Any, Any]:
    """Gate every sklearn import behind exact environment and version identity.

    The worker calls this before importing NumPy/sklearn.  The parent calls it
    only for constructor/review validation; fit workers additionally prove the
    executable identity themselves.
    """
    for name, expected in REQUIRED_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            fail("C2_ENVIRONMENT_MISMATCH", f"{name} must equal {expected!r}")
    if require_pinned_executable and Path(sys.executable).resolve() != pinned_python():
        fail("C2_ENVIRONMENT_MISMATCH", "worker Python executable is not pinned .venv Python")
    if ".".join(str(value) for value in sys.version_info[:3]) != REQUIRED_VERSIONS["python"]:
        fail("C2_ENVIRONMENT_MISMATCH", f"Python {REQUIRED_VERSIONS['python']} required")
    import numpy as np
    import sklearn
    import threadpoolctl
    if np.__version__ != REQUIRED_VERSIONS["numpy"]:
        fail("C2_ENVIRONMENT_MISMATCH", f"NumPy {REQUIRED_VERSIONS['numpy']} required")
    if sklearn.__version__ != REQUIRED_VERSIONS["scikit_learn"]:
        fail("C2_ENVIRONMENT_MISMATCH", f"scikit-learn {REQUIRED_VERSIONS['scikit_learn']} required")
    if threadpoolctl.__version__ != REQUIRED_VERSIONS["threadpoolctl"]:
        fail("C2_ENVIRONMENT_MISMATCH", f"threadpoolctl {REQUIRED_VERSIONS['threadpoolctl']} required")
    return np, sklearn, threadpoolctl


def worker_identity_envelope() -> dict[str, Any]:
    """Worker-side attestation, emitted before it may receive fit approval."""
    np, sklearn, threadpoolctl = require_environment_before_sklearn(require_pinned_executable=True)
    return {
        "kind": "C2_WORKER_IDENTITY",
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "launch_nonce": secrets.token_urlsafe(32),
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "environment": {name: os.environ.get(name) for name in REQUIRED_ENVIRONMENT},
    }


def validate_worker_identity(identity: Mapping[str, Any], expected_pid: int | None = None) -> None:
    if identity.get("kind") != "C2_WORKER_IDENTITY" or not isinstance(identity.get("pid"), int) or not isinstance(identity.get("parent_pid"), int):
        fail("C2_WORKER_IDENTITY_MISMATCH", "worker identity envelope malformed")
    if expected_pid is not None and identity.get("pid") != expected_pid:
        fail("WORKER_PID_MISMATCH", f"reported={identity.get('pid')}, spawned={expected_pid}")
    if not isinstance(identity.get("launch_nonce"), str) or len(identity["launch_nonce"]) < 32:
        fail("C2_WORKER_IDENTITY_MISMATCH", "worker launch nonce malformed")
    expected = {
        "python": REQUIRED_VERSIONS["python"],
        "numpy": REQUIRED_VERSIONS["numpy"],
        "scikit_learn": REQUIRED_VERSIONS["scikit_learn"],
        "threadpoolctl": REQUIRED_VERSIONS["threadpoolctl"],
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        fail("C2_WORKER_IDENTITY_MISMATCH", "worker package versions differ from the frozen environment")
    environment = identity.get("environment")
    if not isinstance(environment, Mapping) or dict(environment) != REQUIRED_ENVIRONMENT:
        fail("C2_WORKER_IDENTITY_MISMATCH", "worker numerical environment differs from frozen values")


def verify_manifest_inputs(provenance: Mapping[str, Any]) -> dict[str, str]:
    entries = provenance.get("inputs")
    if not isinstance(entries, list) or not entries:
        fail("C2_PROVENANCE_INVALID", "input list missing")
    root = ROOT.resolve()
    verified: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            fail("C2_PROVENANCE_INVALID", "input entry is not an object")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            fail("C2_PROVENANCE_INVALID", "input path/hash missing")
        path = (ROOT / relative).resolve()
        if root not in path.parents and path != root:
            fail("C2_PROVENANCE_INVALID", f"path escapes repository: {relative}")
        if not path.is_file():
            fail("C2_INPUT_HASH_MISMATCH", f"missing {relative}")
        actual = sha256_file(path)
        if actual != expected:
            fail("C2_INPUT_HASH_MISMATCH", f"{relative}: expected {expected}, got {actual}")
        verified[relative] = actual
    required = {
        "artifacts/task1/evaluation/strict_cv_v2/folds.json",
        "data/raw/btc/LegalIR/train.json",
        "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
        "artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl",
        "reports/task1/workflow_c/tv2/c0a/c0a_integrity_manifest.json",
        "reports/task1/workflow_c/tv4/c1i/c1i_action_universe_report.json",
        "reports/task1/workflow_c/tv4/c1i/c1i_feature_schema_report.json",
        "artifacts/task1/recovery_096/v3_residual/policy/policy_training_report.json",
    }
    missing = sorted(required - set(verified))
    if missing:
        fail("C2_PROVENANCE_INVALID", f"required input provenance omitted: {missing}")
    return verified


def outer_splits(contract: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    splits = contract["nested_split_contract"]["outer_splits"]
    result: dict[int, tuple[int, ...]] = {}
    for spec in splits.values():
        validation = int(spec["outer_validation_fold"])
        training = tuple(int(fold) for fold in spec["outer_training_folds"])
        if validation not in TARGET_FOLDS or len(training) != 3 or validation in training or set(training) | {validation} != set(TARGET_FOLDS):
            fail("C2_SPLIT_CONTRACT_FAILURE", "invalid outer split")
        result[validation] = training
    if set(result) != set(TARGET_FOLDS):
        fail("C2_SPLIT_CONTRACT_FAILURE", "outer folds incomplete")
    return result


def inner_rotations(outer_training_folds: Sequence[int]) -> tuple[tuple[int, tuple[int, int]], ...]:
    if len(outer_training_folds) != 3 or len(set(outer_training_folds)) != 3 or not set(outer_training_folds).issubset(TARGET_FOLDS):
        fail("C2_SPLIT_CONTRACT_FAILURE", "inner rotations require three distinct F1-F4 folds")
    ordered = tuple(int(fold) for fold in outer_training_folds)
    return tuple((validation, tuple(fold for fold in ordered if fold != validation)) for validation in ordered)


@dataclass(frozen=True)
class ScoringAction:
    """Label-free view: deliberately contains no target, label, or gold payload."""

    query_id: str
    fold: int
    drop_rank: int
    incoming_doc_id: str
    features: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingAction(ScoringAction):
    exact_delta_recall: float


@dataclass(frozen=True)
class BaselineIdentity:
    """The only baseline fields permitted into future top5 construction."""

    query_id: str
    fold: int
    top5: tuple[str, str, str, str, str]


def assert_score_view_label_free(value: Mapping[str, Any]) -> None:
    forbidden = sorted(set(value).intersection(FORBIDDEN_SCORE_FIELDS))
    if forbidden:
        fail("VALIDATION_SCORING_LABEL_LEAKAGE", ",".join(forbidden))


def action_score_payload(action: ScoringAction, score: float) -> dict[str, Any]:
    """Only this label-free record is eligible for worker/parent score exchange."""
    require_scientific_fold(action.fold, "label-free score payload")
    numeric_score = float(score)
    if not math.isfinite(numeric_score):
        fail("NONFINITE_PREDICTION_FAILURE", action.query_id)
    payload = {
        "fold": action.fold,
        "query_id": action.query_id,
        "drop_rank": action.drop_rank,
        "incoming_doc_id": action.incoming_doc_id,
        "score": numeric_score,
    }
    assert_score_view_label_free(payload)
    return payload


def raw_skip_whitespace(raw: str, position: int) -> int:
    while position < len(raw) and raw[position].isspace():
        position += 1
    return position


def raw_string_end(raw: str, position: int) -> int:
    if position >= len(raw) or raw[position] != '"':
        fail("BASELINE_SCHEMA_FAILURE", "expected JSON string")
    position += 1
    escaped = False
    while position < len(raw):
        character = raw[position]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            return position + 1
        position += 1
    fail("BASELINE_SCHEMA_FAILURE", "unterminated JSON string")
    return position  # Unreachable; retained for static type checkers.


def raw_skip_value(raw: str, position: int) -> int:
    """Skip an arbitrary JSON value without decoding its payload."""
    position = raw_skip_whitespace(raw, position)
    if position >= len(raw):
        fail("BASELINE_SCHEMA_FAILURE", "missing JSON value")
    first = raw[position]
    if first == '"':
        return raw_string_end(raw, position)
    if first not in "[{":
        while position < len(raw) and raw[position] not in ",]}":
            position += 1
        return position
    stack = ["}" if first == "{" else "]"]
    position += 1
    in_string = False
    escaped = False
    while position < len(raw) and stack:
        character = raw[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            stack.append("}")
        elif character == "[":
            stack.append("]")
        elif character == stack[-1]:
            stack.pop()
        position += 1
    if stack:
        fail("BASELINE_SCHEMA_FAILURE", "unterminated JSON container")
    return position


def selected_top_level_values(raw: str, selected: set[str]) -> dict[str, Any]:
    """Decode selected fields only; e.g., baseline gold_documents is raw-skipped."""
    position = raw_skip_whitespace(raw, 0)
    if position >= len(raw) or raw[position] != "{":
        fail("BASELINE_SCHEMA_FAILURE", "JSONL row must be an object")
    position = raw_skip_whitespace(raw, position + 1)
    decoder = json.JSONDecoder()
    values: dict[str, Any] = {}
    while position < len(raw) and raw[position] != "}":
        key_end = raw_string_end(raw, position)
        key = json.loads(raw[position:key_end])
        position = raw_skip_whitespace(raw, key_end)
        if position >= len(raw) or raw[position] != ":":
            fail("BASELINE_SCHEMA_FAILURE", "invalid object separator")
        position = raw_skip_whitespace(raw, position + 1)
        if key in selected:
            value, position = decoder.raw_decode(raw, position)
            values[key] = value
        else:
            position = raw_skip_value(raw, position)
        position = raw_skip_whitespace(raw, position)
        if position < len(raw) and raw[position] == ",":
            position = raw_skip_whitespace(raw, position + 1)
        elif position < len(raw) and raw[position] != "}":
            fail("BASELINE_SCHEMA_FAILURE", "invalid object terminator")
    return values


def baseline_identity_from_raw_jsonl(raw: str) -> BaselineIdentity:
    """Future baseline loader: gold_documents and every other payload are skipped raw."""
    values = selected_top_level_values(raw, {"query_id", "fold", "top5"})
    try:
        query_id = values["query_id"]
        fold = int(values["fold"])
        top5_values = tuple(values["top5"])
    except (KeyError, TypeError, ValueError) as exc:
        fail("BASELINE_SCHEMA_FAILURE", str(exc))
    if not isinstance(query_id, str) or not all(isinstance(value, str) for value in top5_values):
        fail("BASELINE_SCHEMA_FAILURE", "canonical IDs must already be strings; coercion is forbidden")
    require_scientific_fold(fold, "baseline identity")
    if len(top5_values) != 5 or len(set(top5_values)) != 5:
        fail("BASELINE_SCHEMA_FAILURE", "top5 must contain five unique document IDs")
    return BaselineIdentity(query_id=query_id, fold=fold, top5=top5_values)


def scoring_view(raw: Mapping[str, Any]) -> ScoringAction:
    """Whitelist only fields allowed to reach a validation scorer."""
    try:
        query_id = raw["query_id"]
        incoming_doc_id = raw["incoming_doc_id"]
        if not isinstance(query_id, str) or not isinstance(incoming_doc_id, str):
            fail("C2_SCORING_VIEW_FAILURE", "canonical IDs must be strings; coercion is forbidden")
        result = ScoringAction(
            query_id=query_id,
            fold=int(raw["fold"]),
            drop_rank=int(raw["drop_rank"]),
            incoming_doc_id=incoming_doc_id,
            features=raw["features"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        fail("C2_SCORING_VIEW_FAILURE", str(exc))
    require_scientific_fold(result.fold, "scoring view")
    if result.drop_rank not in (4, 5) or not result.query_id or not result.incoming_doc_id:
        fail("C2_SCORING_VIEW_FAILURE", "invalid identity")
    if not isinstance(result.features, Mapping):
        fail("C2_SCORING_VIEW_FAILURE", "features must be an object")
    # The dataclass is physically separate from the raw labelled row.  Its
    # serialized representation is checked again before worker exchange.
    assert_score_view_label_free({
        "query_id": result.query_id, "fold": result.fold,
        "drop_rank": result.drop_rank, "incoming_doc_id": result.incoming_doc_id,
        "features": result.features,
    })
    return result


def exact_delta_recall_from_training_row(raw: Mapping[str, Any]) -> float:
    """Use canonical ``gain`` only; it is the recorded single-swap Recall delta."""
    try:
        require_scientific_fold(int(raw.get("fold", -1)), "training target")
    except (TypeError, ValueError) as exc:
        fail("UNKNOWN_SCIENTIFIC_FOLD", str(exc))
    if EXACT_DELTA_FIELD not in raw:
        fail("EXACT_DELTA_TARGET_SCHEMA_UNRESOLVED", f"missing canonical {EXACT_DELTA_FIELD!r} field")
    try:
        delta = float(raw[EXACT_DELTA_FIELD])
    except (TypeError, ValueError) as exc:
        fail("EXACT_DELTA_TARGET_SCHEMA_UNRESOLVED", str(exc))
    if not math.isfinite(delta):
        fail("EXACT_DELTA_TARGET_SCHEMA_UNRESOLVED", "non-finite gain")
    label = raw.get("label")
    expected_label = "BENEFIT" if delta > 0 else "HARM" if delta < 0 else "NEUTRAL"
    if label is not None and label != expected_label:
        fail("EXACT_DELTA_TARGET_SCHEMA_UNRESOLVED", "gain/label sign mismatch")
    return delta


def training_view(raw: Mapping[str, Any]) -> TrainingAction:
    score = scoring_view(raw)
    return TrainingAction(**score.__dict__, exact_delta_recall=exact_delta_recall_from_training_row(raw))


def build_feature_matrix(actions: Sequence[ScoringAction], names: Sequence[str], np: Any) -> Any:
    """Future-only matrix construction: no imputation, reordering, or extra features."""
    frozen = tuple(names)
    if len(frozen) != EXPECTED_FEATURE_COUNT or len(set(frozen)) != EXPECTED_FEATURE_COUNT:
        fail("FEATURE_ORDER_MISMATCH", "invalid frozen feature sequence")
    rows: list[list[float]] = []
    for action in actions:
        require_scientific_fold(action.fold, "matrix construction")
        feature_keys = set(action.features)
        if feature_keys != set(frozen):
            missing = sorted(set(frozen) - feature_keys)
            extra = sorted(feature_keys - set(frozen))
            fail("FEATURE_ORDER_MISMATCH", f"missing={missing}; extra={extra}")
        try:
            row = [float(action.features[name]) for name in frozen]
        except (TypeError, ValueError) as exc:
            fail("FEATURE_ORDER_MISMATCH", str(exc))
        if not all(math.isfinite(value) for value in row):
            fail("NONFINITE_FEATURE_FAILURE", action.query_id)
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_FEATURE_COUNT or not bool(np.isfinite(matrix).all()):
        fail("NONFINITE_FEATURE_FAILURE", "matrix invariant")
    return matrix


def query_balanced_weights(actions: Sequence[TrainingAction]) -> list[float]:
    for action in actions:
        require_scientific_fold(action.fold, "query weighting")
    counts = Counter(action.query_id for action in actions)
    if not counts or any(not query_id for query_id in counts):
        fail("QUERY_WEIGHT_CONTRACT_FAILURE", "missing query IDs")
    weights = [1.0 / counts[action.query_id] for action in actions]
    totals: dict[str, float] = defaultdict(float)
    for action, weight in zip(actions, weights):
        totals[action.query_id] += weight
    if any(abs(total - 1.0) > 1e-12 for total in totals.values()):
        fail("QUERY_WEIGHT_CONTRACT_FAILURE", "query total is not 1.0")
    return weights


def classifier_target(delta: float) -> int:
    if not math.isfinite(delta):
        fail("EXACT_DELTA_TARGET_SCHEMA_UNRESOLVED", "non-finite delta")
    return 1 if delta > 0 else 2 if delta < 0 else 0


def margin_candidates(best_scores: Sequence[float], np: Any) -> tuple[float, ...]:
    values = [float(value) for value in best_scores]
    if not all(math.isfinite(value) for value in values):
        fail("MARGIN_CANDIDATE_FAILURE", "non-finite score")
    positive = [value for value in values if value > 0.0]
    if not positive:
        return (0.0,)
    quantiles = np.quantile(np.asarray(positive, dtype=np.float64), [0.75, 0.90, 0.95], method="linear")
    candidates = {0.0}
    candidates.update(float(value) for value in quantiles)
    result = tuple(sorted(candidates))
    if len(result) > 4 or any(value < 0.0 or not math.isfinite(value) for value in result):
        fail("MARGIN_CANDIDATE_FAILURE", "candidate contract violated")
    return result


def action_identity(action: ScoringAction) -> tuple[str, int, str]:
    return (action.query_id, action.drop_rank, action.incoming_doc_id)


def validate_unique_action_identities(actions: Iterable[ScoringAction]) -> None:
    seen: set[tuple[str, int, str]] = set()
    for action in actions:
        require_scientific_fold(action.fold, "action identity")
        identity = action_identity(action)
        if identity in seen:
            fail("PROVENANCE_FAILURE", f"duplicate action identity {identity}")
        seen.add(identity)


def choose_best_action(scored_actions: Sequence[tuple[ScoringAction, float]], selected_margin: float) -> ScoringAction | None:
    """Strict KEEP and explicit bytewise action ordering; input order is irrelevant."""
    if math.isnan(selected_margin) or selected_margin < 0.0:
        fail("MARGIN_CONTRACT_FAILURE", "margin must be nonnegative or +Infinity")
    if not scored_actions:
        return None
    actions = [item[0] for item in scored_actions]
    validate_unique_action_identities(actions)
    query_ids = {action.query_id for action in actions}
    if len(query_ids) != 1:
        fail("PROVENANCE_FAILURE", "best-action selection is per query")
    normalized: list[tuple[ScoringAction, float]] = []
    for action, score in scored_actions:
        require_scientific_fold(action.fold, "action choice")
        numeric = float(score)
        if not math.isfinite(numeric):
            fail("NONFINITE_PREDICTION_FAILURE", action.query_id)
        normalized.append((action, numeric))
    best, best_score = sorted(
        normalized,
        key=lambda item: (-item[1], -item[0].drop_rank, item[0].incoming_doc_id.encode("utf-8")),
    )[0]
    return best if best_score > max(0.0, selected_margin) else None


def one_swap_top5(baseline_top5: Sequence[str], selected: ScoringAction | None) -> list[str]:
    baseline = list(baseline_top5)
    if not all(isinstance(document, str) for document in baseline):
        fail("TOP5_CONTRACT_FAILURE", "canonical document identities must not be reinterpreted")
    if len(baseline) != 5 or len(set(baseline)) != 5:
        fail("TOP5_CONTRACT_FAILURE", "baseline must be five unique document IDs")
    if selected is None:
        # New list, exact existing identities and order: no normalization/rerank.
        return list(baseline)
    if selected.drop_rank not in (4, 5):
        fail("TOP5_CONTRACT_FAILURE", "only ranks 4 and 5 may be replaced")
    require_scientific_fold(selected.fold, "top5 construction")
    if not isinstance(selected.incoming_doc_id, str):
        fail("TOP5_CONTRACT_FAILURE", "incoming canonical identity must be a string")
    output = list(baseline)
    output[selected.drop_rank - 1] = selected.incoming_doc_id
    unchanged_rank = 5 if selected.drop_rank == 4 else 4
    if (output[:3] != baseline[:3] or output[unchanged_rank - 1] != baseline[unchanged_rank - 1]
            or len(output) != 5 or len(set(output)) != 5):
        fail("TOP5_CONTRACT_FAILURE", "protected ranks or uniqueness violated")
    return output


def c2v_parameters(contract: Mapping[str, Any]) -> dict[str, Any]:
    return dict(contract["c2_v_control"]["estimator_parameters"])


def c2r_parameters(contract: Mapping[str, Any]) -> dict[str, Any]:
    return dict(contract["c2_r_challenger"]["estimator_parameters"])


def construct_c2v_estimator(contract: Mapping[str, Any], sklearn: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    params = c2v_parameters(contract)
    estimator = HistGradientBoostingClassifier(**params)
    if estimator.get_params(deep=False) != params:
        fail("C2_ESTIMATOR_CONTRACT_FAILURE", "C2-V parameters did not round-trip")
    return estimator


def construct_c2r_estimator(contract: Mapping[str, Any], sklearn: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = c2r_parameters(contract)
    estimator = HistGradientBoostingRegressor(**params)
    if estimator.get_params(deep=False) != params:
        fail("C2_ESTIMATOR_CONTRACT_FAILURE", "C2-R parameters did not round-trip")
    return estimator


def c2v_utility_from_probabilities(estimator: Any, probabilities: Any, np: Any) -> Any:
    """Future-only utility mapping; never assumes probability-column order."""
    classes = [int(value) for value in estimator.classes_]
    if set(classes) != {0, 1, 2} or len(classes) != 3:
        fail("C2V_CLASS_AVAILABILITY_FAILURE", f"classes={classes}")
    positions = {label: index for index, label in enumerate(classes)}
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not bool(np.isfinite(values).all()):
        fail("NONFINITE_PREDICTION_FAILURE", "C2-V probabilities")
    return values[:, positions[1]] - 1.5 * values[:, positions[2]]


def future_c2v_scores(estimator: Any, x: Any, np: Any) -> Any:
    """Future-only scoring call; class-aware utility is kept separate from fitting."""
    probabilities = getattr(estimator, "predict_proba")(x)
    utilities = c2v_utility_from_probabilities(estimator, probabilities, np)
    if not bool(np.isfinite(utilities).all()):
        fail("NONFINITE_PREDICTION_FAILURE", "C2-V utility")
    return utilities


def future_c2r_scores(estimator: Any, x: Any, np: Any) -> Any:
    """Future-only C2-R score call; no label-derived field reaches this view."""
    values = np.asarray(getattr(estimator, "predict")(x), dtype=np.float64)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        fail("NONFINITE_PREDICTION_FAILURE", "C2-R utility")
    return values


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_ulong), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_ulong),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_ulong), ("SchedulingClass", ctypes.c_ulong)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                          "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
TH32CS_SNAPPROCESS = 0x00000002


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong), ("th32ProcessID", ctypes.c_ulong),
        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
        ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long), ("dwFlags", ctypes.c_ulong),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


@dataclass
class WorkerContainment:
    """Windows Job Object containment for exactly one launcher and all governed descendants."""

    job_handle: int
    launcher_pid: int
    worker_pid: int | None = None

    def terminate(self) -> bool:
        """Use the Job Object as the bounded tree-level fallback before releasing its handle."""
        if not self.job_handle:
            return True
        return bool(ctypes.windll.kernel32.TerminateJobObject(ctypes.c_void_p(self.job_handle), 1))

    def close(self) -> None:
        if self.job_handle:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self.job_handle))
            self.job_handle = 0


def create_worker_containment(launcher: subprocess.Popen[str]) -> WorkerContainment:
    if os.name != "nt":
        fail("RESOURCE_MONITOR_IMPLEMENTATION_BLOCKER", "C2 containment requires Windows")
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        fail("WORKER_CONTAINMENT_FAILURE", "CreateJobObjectW failed")
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(ctypes.c_void_p(job), JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                            ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(ctypes.c_void_p(job))
        fail("WORKER_CONTAINMENT_FAILURE", "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(launcher._handle)):
        kernel32.CloseHandle(ctypes.c_void_p(job))
        fail("WORKER_CONTAINMENT_FAILURE", "AssignProcessToJobObject launcher failed")
    return WorkerContainment(int(job), launcher.pid)


def windows_parent_pid(process_id: int) -> int:
    """Read a process's actual parent from the Windows process snapshot, never from worker JSON."""
    if os.name != "nt" or process_id <= 0:
        fail("WORKER_PID_MISMATCH", "Windows parent PID query unavailable")
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or snapshot == invalid_handle:
        fail("WORKER_PID_MISMATCH", "CreateToolhelp32Snapshot failed")
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        found = bool(kernel32.Process32FirstW(ctypes.c_void_p(snapshot), ctypes.byref(entry)))
        while found:
            if int(entry.th32ProcessID) == process_id:
                return int(entry.th32ParentProcessID)
            entry.dwSize = ctypes.sizeof(entry)
            found = bool(kernel32.Process32NextW(ctypes.c_void_p(snapshot), ctypes.byref(entry)))
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(snapshot))
    fail("WORKER_PID_MISMATCH", f"PID {process_id} absent from Windows process snapshot")
    return 0


def verify_contained_worker(containment: WorkerContainment, worker_pid: int, reported_parent_pid: int) -> None:
    """Verify actual Python worker is the launcher's direct child and a Job Object member."""
    if worker_pid <= 0 or reported_parent_pid != containment.launcher_pid:
        fail("WORKER_PID_MISMATCH", "worker is not a direct child of the governed launcher")
    if windows_parent_pid(worker_pid) != containment.launcher_pid:
        fail("WORKER_PID_MISMATCH", "worker OS parent is not the governed launcher")
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, worker_pid)
    if not handle:
        fail("WORKER_PID_MISMATCH", "cannot inspect reported worker PID")
    try:
        in_job = ctypes.c_int(0)
        kernel32 = ctypes.windll.kernel32
        if not kernel32.IsProcessInJob(ctypes.c_void_p(handle), ctypes.c_void_p(containment.job_handle), ctypes.byref(in_job)):
            fail("WORKER_PID_MISMATCH", "cannot inspect Job Object membership")
        if not in_job.value:
            fail("WORKER_PID_MISMATCH", "reported worker is outside governed containment")
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    containment.worker_pid = worker_pid


def available_physical_memory_bytes() -> int:
    """Windows stdlib-only memory gate; no unfrozen monitoring dependency."""
    if os.name != "nt":
        fail("RESOURCE_MONITOR_IMPLEMENTATION_BLOCKER", "frozen environment is Windows-only")
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        fail("RESOURCE_MONITOR_IMPLEMENTATION_BLOCKER", "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def require_fit_memory_headroom() -> None:
    if available_physical_memory_bytes() < MINIMUM_AVAILABLE_MEMORY_BYTES:
        fail("AVAILABLE_MEMORY_FAILURE", "available physical memory is below 4 GiB")


def process_rss_bytes(process_id: int) -> int:
    """Read process RSS via Windows psapi without installing a dependency."""
    if os.name != "nt":
        fail("RESOURCE_MONITOR_IMPLEMENTATION_BLOCKER", "frozen environment is Windows-only")

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    process_query_information = 0x0400
    process_vm_read = 0x0010
    handle = ctypes.windll.kernel32.OpenProcess(process_query_information | process_vm_read, False, process_id)
    if not handle:
        fail("RESOURCE_MONITOR_IMPLEMENTATION_BLOCKER", f"cannot open PID {process_id}")
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            fail("RESOURCE_MONITOR_IMPLEMENTATION_BLOCKER", f"GetProcessMemoryInfo failed for PID {process_id}")
        return int(counters.WorkingSetSize)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


@dataclass
class C2RunBudget:
    """Future-only parent-side limits: one worker, 32 fits, eight hours total."""

    started_monotonic: float
    session_id: str
    fits_started: int = 0
    c2v_fit_launches: int = 0
    c2r_fit_launches: int = 0
    worker_nonces: set[str] = field(default_factory=set)
    job_ids: set[str] = field(default_factory=set)

    @classmethod
    def start(cls, session_id: str | None = None) -> "C2RunBudget":
        return cls(started_monotonic=time.monotonic(), session_id=session_id or secrets.token_urlsafe(32))

    def reserve_fit(self, family: str) -> tuple[int, int]:
        if family not in ("c2_v", "c2_r"):
            fail("FIT_BUDGET_EXCEEDED", f"unknown model family {family}")
        if self.fits_started >= MAXIMUM_PLANNED_FITS:
            fail("FIT_BUDGET_EXCEEDED", f"maximum planned fits is {MAXIMUM_PLANNED_FITS}")
        if family == "c2_v" and self.c2v_fit_launches >= 16:
            fail("FIT_BUDGET_EXCEEDED", "C2-V maximum is 16 fits")
        if family == "c2_r" and self.c2r_fit_launches >= 16:
            fail("FIT_BUDGET_EXCEEDED", "C2-R maximum is 16 fits")
        if time.monotonic() - self.started_monotonic > TOTAL_RUN_TIMEOUT_SECONDS:
            fail("TOTAL_C2_TIMEOUT", "eight-hour total hard wall exceeded")
        self.fits_started += 1
        if family == "c2_v":
            self.c2v_fit_launches += 1
        else:
            self.c2r_fit_launches += 1
        return (self.c2v_fit_launches if family == "c2_v" else self.c2r_fit_launches, self.fits_started)


@dataclass(frozen=True)
class FitDeadline:
    """One non-resettable deadline covering every operation of one worker lifecycle."""

    started_monotonic: float
    expires_monotonic: float

    @classmethod
    def start(cls, monotonic: Callable[[], float] = time.monotonic) -> "FitDeadline":
        started = monotonic()
        return cls(started, started + PER_FIT_TIMEOUT_SECONDS)

    def remaining_seconds(self, monotonic: Callable[[], float] = time.monotonic) -> float:
        return self.expires_monotonic - monotonic()


class BoundedDiagnosticTail:
    """Drain worker stderr continuously without letting diagnostics consume unbounded memory."""

    def __init__(self, byte_limit: int = 64 * 1024) -> None:
        self.byte_limit = byte_limit
        self._payload = bytearray()
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        encoded = text.encode("utf-8", errors="replace")
        with self._lock:
            self._payload.extend(encoded)
            if len(self._payload) > self.byte_limit:
                del self._payload[:-self.byte_limit]

    def text(self) -> str:
        with self._lock:
            return bytes(self._payload).decode("utf-8", errors="replace")


def terminate_worker(process: subprocess.Popen[str]) -> None:
    """Terminate only the launched worker tree, with bounded Windows cleanup confirmation."""
    containment = getattr(process, "_c2_containment", None)
    if isinstance(containment, WorkerContainment):
        containment.terminate()  # Direct Job Object termination covers descendants even if taskkill is unavailable.
        containment.close()  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE kills governed launcher/worker/descendants.
    if process.poll() is None and os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except subprocess.TimeoutExpired:
            # taskkill is one bounded cleanup mechanism, never the only one.
            # Continue to the exact-Popen fallback and final survivor check.
            pass
        if process.poll() is None:
            # taskkill may not report a just-created Python launcher tree promptly; kill the exact Popen root as fallback.
            process.kill()
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        fail("C2_WORKER_SURVIVED_TERMINATION", f"PID {process.pid}")
    if process.poll() is None:
        fail("C2_WORKER_SURVIVED_TERMINATION", f"PID {process.pid}")


def _reader_thread(stream: Any, out: Queue[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            out.put(line)
    finally:
        out.put("")


def _stderr_drain_thread(stream: Any, tail: BoundedDiagnosticTail) -> None:
    try:
        for chunk in iter(lambda: stream.read(4096), ""):
            tail.append(chunk)
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _watch_worker(process: subprocess.Popen[str], budget: C2RunBudget, deadline: FitDeadline, phase: str) -> None:
    """Watchdog used by every parent lifecycle wait, independent of worker cooperation."""
    if time.monotonic() - budget.started_monotonic > TOTAL_RUN_TIMEOUT_SECONDS:
        terminate_worker(process)
        fail("TOTAL_C2_TIMEOUT", phase)
    if deadline.remaining_seconds() <= 0:
        terminate_worker(process)
        fail("FIT_TIMEOUT", phase)
    monitored_pid = getattr(process, "_c2_worker_pid", process.pid)
    if process.poll() is None and process_rss_bytes(monitored_pid) > RSS_HARD_CEILING_BYTES:
        terminate_worker(process)
        fail("RSS_LIMIT_EXCEEDED", phase)


def _wait_worker_line(process: subprocess.Popen[str], lines: Queue[str], budget: C2RunBudget,
                      deadline: FitDeadline, phase: str) -> str:
    """Bounded stdout wait sharing the pre-launch absolute deadline."""
    while True:
        _watch_worker(process, budget, deadline, phase)
        try:
            line = lines.get(timeout=min(0.25, max(0.001, deadline.remaining_seconds())))
        except Empty:
            if process.poll() is not None:
                fail("WORKER_NONZERO_EXIT", f"{phase}: exit={process.returncode}")
            continue
        if line:
            return line
        if process.poll() is not None:
            fail("WORKER_NONZERO_EXIT", f"{phase}: exit={process.returncode}")


def resource_monitor_capability() -> bool:
    """Review-only capability probe: no child, fit, data, or scientific output."""
    try:
        return available_physical_memory_bytes() > 0 and process_rss_bytes(os.getpid()) > 0
    except (AttributeError, OSError, C2HarnessError):
        return False


def _capability_writer(stream: Any, payload: str, completed: Queue[BaseException | None]) -> None:
    try:
        stream.write(payload)
        stream.flush()
        completed.put(None)
    except BaseException as exc:  # Main watchdog converts this to a closed protocol failure.
        completed.put(exc)


def _wait_capability_delivery(process: subprocess.Popen[str], completed: Queue[BaseException | None],
                              budget: C2RunBudget, deadline: FitDeadline) -> None:
    while True:
        _watch_worker(process, budget, deadline, "capability-delivery")
        try:
            outcome = completed.get(timeout=min(0.25, max(0.001, deadline.remaining_seconds())))
        except Empty:
            continue
        if outcome is not None:
            terminate_worker(process)
            fail("WORKER_PROTOCOL_FAILURE", f"capability delivery failed: {outcome}")
        return


def pinned_worker_identity_self_checks() -> dict[str, bool]:
    """Exercise the exact pinned launcher and prove its reported Python child is contained, not merely trusted."""
    code = "import json,os,time,secrets; x=dict(pid=os.getpid(),parent_pid=os.getppid(),nonce=secrets.token_urlsafe(24),exe=os.path.abspath(os.sys.executable)); print(json.dumps(x),flush=True); time.sleep(30)"
    process = subprocess.Popen([str(pinned_python()), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", env=worker_environment(), cwd=str(ROOT))
    containment: WorkerContainment | None = None
    try:
        containment = create_worker_containment(process)
        if process.stdout is None:
            return {"pinned_worker_identity": False, "unrelated_worker_pid_rejection": False,
                    "actual_worker_rss_target": False, "os_parent_verified_worker_identity": False}
        line = process.stdout.readline().strip()
        identity = strict_json_object(line.encode("utf-8"), "SYNTHETIC_IDENTITY")
        worker_pid, parent_pid = identity.get("pid"), identity.get("parent_pid")
        if not isinstance(worker_pid, int) or not isinstance(parent_pid, int) or worker_pid == process.pid:
            return {"pinned_worker_identity": False, "unrelated_worker_pid_rejection": False,
                    "actual_worker_rss_target": False, "os_parent_verified_worker_identity": False}
        os_parent_verified = windows_parent_pid(worker_pid) == process.pid
        verify_contained_worker(containment, worker_pid, parent_pid)
        process._c2_worker_pid = worker_pid
        monitor_targets_worker = getattr(process, "_c2_worker_pid") == worker_pid != process.pid
        unrelated = subprocess.Popen([str(pinned_python()), "-c", "import time; time.sleep(30)"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, env=worker_environment(), cwd=str(ROOT))
        try:
            rejected = False
            try:
                verify_contained_worker(containment, unrelated.pid, process.pid)
            except C2HarnessError:
                rejected = True
            return {"pinned_worker_identity": os_parent_verified and monitor_targets_worker and rejected,
                    "unrelated_worker_pid_rejection": rejected,
                    "actual_worker_rss_target": monitor_targets_worker,
                    "os_parent_verified_worker_identity": os_parent_verified}
        finally:
            unrelated.kill()
            unrelated.wait(timeout=5)
    except (C2HarnessError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return {"pinned_worker_identity": False, "unrelated_worker_pid_rejection": False,
                "actual_worker_rss_target": False, "os_parent_verified_worker_identity": False}
    finally:
        if containment is not None:
            containment.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def pinned_worker_identity_self_check() -> bool:
    return all(pinned_worker_identity_self_checks().values())


def synthetic_timeout_supervision_cases() -> dict[str, str]:
    """Label-free child-process tests prove watchdog termination without Task1 paths or model calls."""
    def run_case(kind: str) -> str:
        code = ("import sys,time; time.sleep(60)" if kind != "stderr" else
                "import sys,time; sys.stderr.write('x'*262144); sys.stderr.flush(); time.sleep(60)")
        process = subprocess.Popen([str(pinned_python()), "-c", code], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", env=worker_environment(), cwd=str(ROOT))
        if process.stdout is None or process.stderr is None:
            terminate_worker(process)
            return "PIPE_UNAVAILABLE"
        lines: Queue[str] = Queue()
        tail = BoundedDiagnosticTail(1024)
        threading.Thread(target=_reader_thread, args=(process.stdout, lines), daemon=True).start()
        threading.Thread(target=_stderr_drain_thread, args=(process.stderr, tail), daemon=True).start()
        budget = C2RunBudget.start("synthetic-session-timeout")
        deadline = FitDeadline(time.monotonic() - PER_FIT_TIMEOUT_SECONDS - 1, time.monotonic() - 1)
        try:
            if kind == "capability":
                _wait_capability_delivery(process, Queue(), budget, deadline)
            else:
                _wait_worker_line(process, lines, budget, deadline, f"synthetic-{kind}")
        except C2HarnessError as exc:
            if not str(exc).startswith("FIT_TIMEOUT"):
                return f"UNEXPECTED:{exc}"
            if process.poll() is None:
                return "CHILD_SURVIVED"
            if kind == "stderr" and len(tail.text().encode("utf-8")) > 1024:
                return "STDERR_TAIL_UNBOUNDED"
            return "PASS"
        finally:
            if process.poll() is None:
                terminate_worker(process)
        return "NO_TIMEOUT"
    try:
        return {kind: run_case(kind) for kind in ("identity", "capability", "result", "stderr")}
    except (C2HarnessError, OSError) as exc:
        return {"harness": f"HARNESS_EXCEPTION:{type(exc).__name__}:{exc}"}


def synthetic_timeout_supervision_self_checks() -> bool:
    return all(value == "PASS" for value in synthetic_timeout_supervision_cases().values())


def process_is_alive(process_id: int) -> bool:
    """Query an exact PID without relying on a launcher Popen handle."""
    if os.name != "nt" or process_id <= 0:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong(0)
        return bool(kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)) and code.value == STILL_ACTIVE)
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def process_is_in_containment(process_id: int, containment: WorkerContainment) -> bool:
    if os.name != "nt" or process_id <= 0:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return False
    try:
        in_job = ctypes.c_int(0)
        return bool(kernel32.IsProcessInJob(ctypes.c_void_p(handle), ctypes.c_void_p(containment.job_handle),
                                            ctypes.byref(in_job)) and in_job.value)
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def descendant_cleanup_self_check() -> bool:
    """Use the real pinned launcher and Job Object to prove a sleeping descendant is killed."""
    child_code = (
        "import json,os,subprocess,sys,time; "
        "inner='import os,time; print(os.getpid(),flush=True); time.sleep(60)'; "
        "d=subprocess.Popen([sys.executable,'-c',inner],stdout=subprocess.PIPE,text=True); "
        "descendant_pid=int(d.stdout.readline()); "
        "print(json.dumps(dict(pid=os.getpid(),parent_pid=os.getppid(),descendant_pid=descendant_pid)),flush=True); time.sleep(60)"
    )
    process = subprocess.Popen([str(pinned_python()), "-c", child_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", env=worker_environment(), cwd=str(ROOT))
    containment: WorkerContainment | None = None
    unrelated: subprocess.Popen[str] | None = None
    try:
        containment = create_worker_containment(process)
        if process.stdout is None:
            return False
        lines: Queue[str] = Queue(maxsize=2)
        threading.Thread(target=_reader_thread, args=(process.stdout, lines), daemon=True).start()
        try:
            identity = strict_json_object(lines.get(timeout=10).encode("utf-8"), "SYNTHETIC_TREE")
        except (Empty, C2HarnessError):
            return False
        worker_pid, parent_pid, descendant_pid = identity.get("pid"), identity.get("parent_pid"), identity.get("descendant_pid")
        if not all(isinstance(value, int) and value > 0 for value in (worker_pid, parent_pid, descendant_pid)):
            return False
        verify_contained_worker(containment, worker_pid, parent_pid)
        process._c2_containment = containment
        process._c2_worker_pid = worker_pid
        unrelated = subprocess.Popen([str(pinned_python()), "-c", "import time; time.sleep(60)"], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, env=worker_environment(), cwd=str(ROOT))
        governed_before = (process_is_in_containment(process.pid, containment)
                            and process_is_in_containment(worker_pid, containment)
                            and process_is_in_containment(descendant_pid, containment))
        unrelated_before = process_is_alive(unrelated.pid)
        started = time.monotonic()
        terminate_worker(process)
        elapsed = time.monotonic() - started
        gone = not any(process_is_alive(pid) for pid in (process.pid, worker_pid, descendant_pid))
        return governed_before and unrelated_before and gone and process_is_alive(unrelated.pid) and elapsed <= 15.0
    except (C2HarnessError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return False
    finally:
        if containment is not None:
            containment.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if unrelated is not None:
            unrelated.kill()
            unrelated.wait(timeout=5)


def contained_grandchild_rejection_self_check() -> bool:
    """A governed descendant cannot forge the launcher's parent PID and become the worker."""
    child_code = (
        "import json,os,subprocess,sys,time; "
        "inner='import os,time; print(os.getpid(),flush=True); time.sleep(60)'; "
        "d=subprocess.Popen([sys.executable,'-c',inner],stdout=subprocess.PIPE,text=True); "
        "grandchild_pid=int(d.stdout.readline()); "
        "print(json.dumps(dict(pid=os.getpid(),parent_pid=os.getppid(),grandchild_pid=grandchild_pid)),flush=True); time.sleep(60)"
    )
    process = subprocess.Popen([str(pinned_python()), "-c", child_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", env=worker_environment(), cwd=str(ROOT))
    containment: WorkerContainment | None = None
    try:
        containment = create_worker_containment(process)
        if process.stdout is None:
            return False
        line = process.stdout.readline()
        identity = strict_json_object(line.encode("utf-8"), "SYNTHETIC_GRANDCHILD")
        worker_pid, parent_pid, grandchild_pid = identity.get("pid"), identity.get("parent_pid"), identity.get("grandchild_pid")
        if not all(isinstance(value, int) and value > 0 for value in (worker_pid, parent_pid, grandchild_pid)):
            return False
        verify_contained_worker(containment, worker_pid, parent_pid)
        if not process_is_in_containment(grandchild_pid, containment):
            return False
        try:
            verify_contained_worker(containment, grandchild_pid, process.pid)
        except C2HarnessError:
            return windows_parent_pid(grandchild_pid) != process.pid
        return False
    except (C2HarnessError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return False
    finally:
        if containment is not None:
            containment.terminate()
            containment.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def taskkill_fallback_self_check() -> bool:
    """A taskkill timeout must still invoke bounded direct cleanup of the exact Popen root."""
    process = subprocess.Popen([str(pinned_python()), "-c", "import time; time.sleep(60)"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, env=worker_environment(), cwd=str(ROOT))
    original_run, original_kill = subprocess.run, process.kill
    fallback_called = False

    def failed_taskkill(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("taskkill", 10)

    def direct_fallback() -> None:
        nonlocal fallback_called
        fallback_called = True
        original_kill()

    try:
        subprocess.run = failed_taskkill  # type: ignore[assignment]
        process.kill = direct_fallback  # type: ignore[method-assign]
        started = time.monotonic()
        terminate_worker(process)
        return fallback_called and process.poll() is not None and time.monotonic() - started <= 15.0
    except (C2HarnessError, OSError, subprocess.TimeoutExpired):
        return False
    finally:
        subprocess.run = original_run  # type: ignore[assignment]
        if process.poll() is None:
            original_kill()
            process.wait(timeout=5)


def _runtime_journal_path() -> tuple[Path, str, str]:
    session_id = f"runtimecheck-{secrets.token_urlsafe(12)}"
    job_id = f"runtime-check-{secrets.token_urlsafe(12)}"
    path = worker_diagnostic_path(session_id, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, session_id, job_id


def terminal_journal_self_checks() -> dict[str, bool]:
    """Durably exercise label-free timeout/protocol/resource/cleanup terminal evidence."""
    try:
        families = "c2_v"
        cases = {
            "timeout_terminal_journal": ("FIT_TIMEOUT", "FIT_TIMEOUT"),
            "protocol_terminal_journal": ("PROTOCOL_ABORT", "WORKER_PROTOCOL_FAILURE"),
            "resource_terminal_journal": ("RESOURCE_ABORT", "RSS_LIMIT_EXCEEDED"),
        }
        outcomes: dict[str, bool] = {}
        for key, (stage, category) in cases.items():
            path, session_id, job_id = _runtime_journal_path()
            process = subprocess.Popen([str(pinned_python()), "-c", "import time; time.sleep(60)"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       env=worker_environment(), cwd=str(ROOT))
            containment = create_worker_containment(process)
            process._c2_containment = containment
            primary = C2HarnessError(f"{category}: synthetic-runtime-acceptance")
            try:
                written = append_worker_diagnostic(path, stage, session_id=session_id, job_id=job_id, family=families,
                                                   process=process, failure_category=category)
                cleanup_ok = cleanup_worker_after_primary_failure(process, path, session_id=session_id, job_id=job_id,
                                                                   family=families, primary=primary)
                records = [strict_json_object(line.encode("utf-8"), "RUNTIME_JOURNAL")
                           for line in path.read_text(encoding="utf-8").splitlines()]
                stages = [record.get("stage") for record in records]
                outcomes[key] = (written and cleanup_ok and not process_is_alive(process.pid)
                                 and stages == [stage, "CLEANUP_STARTED", "CLEANUP_COMPLETED", "WORKER_EXIT"])
            finally:
                containment.close()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        path, session_id, job_id = _runtime_journal_path()
        process = subprocess.Popen([str(pinned_python()), "-c", "import time; time.sleep(60)"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=worker_environment(), cwd=str(ROOT))
        containment = create_worker_containment(process)
        process._c2_containment = containment
        original_terminate = terminate_worker
        try:
            def simulated_cleanup_failure(_process: subprocess.Popen[str]) -> None:
                raise C2HarnessError("C2_WORKER_SURVIVED_TERMINATION: synthetic")
            globals()["terminate_worker"] = simulated_cleanup_failure
            cleanup_ok = cleanup_worker_after_primary_failure(
                process, path, session_id=session_id, job_id=job_id, family=families,
                primary=C2HarnessError("FIT_TIMEOUT: synthetic"))
        finally:
            globals()["terminate_worker"] = original_terminate
            original_terminate(process)
            containment.close()
        records = [strict_json_object(line.encode("utf-8"), "RUNTIME_JOURNAL") for line in path.read_text(encoding="utf-8").splitlines()]
        stages = [record.get("stage") for record in records]
        forbidden = set().union(*(set(record) for record in records)).intersection(FORBIDDEN_SCORE_FIELDS)
        outcomes["terminal_failure_journaling"] = not forbidden and all(stage in _TERMINAL_DIAGNOSTIC_STAGES for stage in stages)
        outcomes["cleanup_journal"] = (not cleanup_ok and stages == ["CLEANUP_STARTED", "CLEANUP_FAILED", "WORKER_EXIT_UNCONFIRMED"])
        outcomes["cleanup_failure_no_false_worker_exit"] = ("WORKER_EXIT" not in stages
                                                              and stages[-1] == "WORKER_EXIT_UNCONFIRMED")
        with tempfile.TemporaryDirectory() as directory:
            primary = C2HarnessError("FIT_TIMEOUT: synthetic")
            outcomes["journal_failure_preserves_primary"] = (
                append_worker_diagnostic(Path(directory), "FIT_TIMEOUT", session_id=session_id, job_id=job_id,
                                         family=families, failure_category="FIT_TIMEOUT") is False
                and str(primary).startswith("FIT_TIMEOUT"))
        return outcomes
    except (C2HarnessError, OSError, json.JSONDecodeError):
        return {key: False for key in ("timeout_terminal_journal", "protocol_terminal_journal", "resource_terminal_journal",
                                       "terminal_failure_journaling", "cleanup_journal", "cleanup_failure_no_false_worker_exit",
                                       "journal_failure_preserves_primary")}


def launch_stage_terminal_journal_self_check() -> bool:
    """Launch failures with no usable Popen still journal primary-before-cleanup ordering."""
    try:
        cases = (
            (FitDeadline(time.monotonic() - 1.0, time.monotonic() - 0.001), "FIT_TIMEOUT"),
            (FitDeadline.start(), "PROTOCOL_ABORT"),
        )
        outcomes: list[bool] = []
        for deadline, expected_stage in cases:
            path, session_id, job_id = _runtime_journal_path()
            try:
                _launch_worker_bounded([str(ROOT / "missing-runtime-launcher.exe")], {}, C2RunBudget.start(), deadline)
                outcomes.append(False)
                continue
            except C2HarnessError as primary:
                journal_primary_failure(None, path, session_id=session_id, job_id=job_id, family="c2_v", primary=primary)
            records = [strict_json_object(line.encode("utf-8"), "LAUNCH_JOURNAL")
                       for line in path.read_text(encoding="utf-8").splitlines()]
            outcomes.append([record.get("stage") for record in records]
                            == [expected_stage, "CLEANUP_STARTED", "CLEANUP_COMPLETED"])
        return all(outcomes)
    except (C2HarnessError, OSError, json.JSONDecodeError):
        return False


def containment_handle_lifecycle_self_checks() -> dict[str, bool]:
    """Exercise close-after-exit and close-after-failure without fitting or Task1 data."""
    outcomes = {"success_path_containment_closed": False, "failure_path_containment_closed": False}
    process: subprocess.Popen[str] | None = None
    containment: WorkerContainment | None = None
    try:
        process = subprocess.Popen([str(pinned_python()), "-c", "import time; time.sleep(0.1)"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=worker_environment(), cwd=str(ROOT))
        containment = create_worker_containment(process)
        live_before_exit = containment.job_handle != 0 and process.poll() is None
        process.wait(timeout=5)
        containment.close()
        outcomes["success_path_containment_closed"] = live_before_exit and process.poll() is not None and containment.job_handle == 0
    except (C2HarnessError, OSError, subprocess.TimeoutExpired):
        pass
    finally:
        if containment is not None:
            containment.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    process = None
    containment = None
    try:
        process = subprocess.Popen([str(pinned_python()), "-c", "import time; time.sleep(60)"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=worker_environment(), cwd=str(ROOT))
        containment = create_worker_containment(process)
        process._c2_containment = containment
        path, session_id, job_id = _runtime_journal_path()
        cleanup_worker_after_primary_failure(process, path, session_id=session_id, job_id=job_id, family="c2_v",
                                             primary=C2HarnessError("FIT_TIMEOUT: synthetic"))
        outcomes["failure_path_containment_closed"] = process.poll() is not None and containment.job_handle == 0
    except (C2HarnessError, OSError, subprocess.TimeoutExpired):
        pass
    finally:
        if containment is not None:
            containment.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    return outcomes


def deadline_race_self_checks() -> dict[str, bool]:
    """Deterministically exercise the one absolute deadline at every acceptance boundary."""
    clock = [100.0]
    deadline = FitDeadline.start(lambda: clock[0])
    same_deadline = deadline.started_monotonic == 100.0 and deadline.expires_monotonic == 100.0 + PER_FIT_TIMEOUT_SECONDS

    class AlreadyExited:
        pid = 0
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            return 0

    def rejected_at_boundary(phase: str) -> bool:
        clock[0] = deadline.expires_monotonic
        try:
            # This calls the exact production watchdog branch used after each named stage.
            _watch_worker(AlreadyExited(), C2RunBudget.start("synthetic-deadline-race"), deadline, phase)  # type: ignore[arg-type]
        except C2HarnessError as exc:
            return str(exc).startswith("FIT_TIMEOUT") and deadline.expires_monotonic == 100.0 + PER_FIT_TIMEOUT_SECONDS
        return False

    return {
        "post_launch_deadline": rejected_at_boundary("post-launch"),
        "post_identity_deadline": rejected_at_boundary("post-identity-receipt"),
        "post_identity_validation_deadline": rejected_at_boundary("post-identity-validation"),
        "post_capability_deadline": rejected_at_boundary("post-capability-delivery"),
        "post_result_deadline": rejected_at_boundary("post-result-receipt"),
        "post_parse_deadline": rejected_at_boundary("post-result-parse"),
        "final_acceptance_deadline": rejected_at_boundary("pre-fit-result-return"),
        "absolute_fit_deadline_not_reset": same_deadline,
    }


def _launch_worker_bounded(command: Sequence[str], launch_kwargs: Mapping[str, Any], budget: C2RunBudget,
                           deadline: FitDeadline) -> subprocess.Popen[str]:
    """Keep the pre-launch absolute deadline enforceable even if Windows process creation stalls."""
    outcome: Queue[subprocess.Popen[str] | BaseException] = Queue()
    cancelled = threading.Event()

    def launch() -> None:
        try:
            process = subprocess.Popen(command, **dict(launch_kwargs))
            if cancelled.is_set():
                terminate_worker(process)
            outcome.put(process)
        except BaseException as exc:
            outcome.put(exc)

    threading.Thread(target=launch, daemon=True).start()
    while True:
        if time.monotonic() - budget.started_monotonic > TOTAL_RUN_TIMEOUT_SECONDS:
            cancelled.set()
            fail("TOTAL_C2_TIMEOUT", "worker-launch")
        if deadline.remaining_seconds() <= 0:
            cancelled.set()
            fail("FIT_TIMEOUT", "worker-launch")
        try:
            result = outcome.get(timeout=min(0.25, max(0.001, deadline.remaining_seconds())))
        except Empty:
            continue
        if isinstance(result, BaseException):
            fail("WORKER_PROTOCOL_FAILURE", f"worker launch failed: {result}")
        if cancelled.is_set():
            terminate_worker(result)
            fail("FIT_TIMEOUT", "worker-launch")
        return result


def launch_fit_worker(job_path: Path, job_id: str, job_sha256: str, job_scope: str,
                      budget: C2RunBudget, family: str) -> dict[str, Any]:
    """Parent-controlled handshake plus wall/RSS/failure enforcement for one fit."""
    require_fit_memory_headroom()
    deadline = FitDeadline.start()  # Established before every Popen/IPC operation; never reset.
    family_ordinal, aggregate_ordinal = budget.reserve_fit(family)  # A launch attempt consumes budget before process creation.
    diagnostic_path = worker_diagnostic_path(budget.session_id, job_id)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostic_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_protocol_line({"stage": "WORKER_PROCESS_LAUNCHING",
                                     "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                                     "session_id": budget.session_id, "job_id": job_id, "family": family}))
    command = [str(pinned_python()), str(Path(__file__).resolve()), "--fit-worker", str(job_path)]
    launch_env = worker_environment()
    launch_env.update({"C2_EXPECTED_SESSION_ID": budget.session_id,
                       "C2_EXPECTED_JOB_ID": job_id, "C2_EXPECTED_JOB_SHA256": job_sha256,
                       "C2_EXPECTED_JOB_SCOPE": job_scope, "C2_EXPECTED_JOB_PATH": str(job_path.resolve()),
                       "C2_EXPECTED_FAMILY": family, "C2_EXPECTED_FAMILY_ORDINAL": str(family_ordinal),
                       "C2_EXPECTED_AGGREGATE_ORDINAL": str(aggregate_ordinal),
                       "C2_EXPECTED_SCRIPT_SHA256": running_script_sha256(),
                       "C2_EXPECTED_CONTRACT_SHA256": EXPECTED_CONTRACT_SHA256,
                       "C2_EXPECTED_PROVENANCE_SHA256": EXPECTED_PROVENANCE_SHA256,
                       "C2_WORKER_DIAGNOSTIC_PATH": str(diagnostic_path)})
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process: subprocess.Popen[str] | None = None
    try:
        process = _launch_worker_bounded(command, {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
                                          "stderr": subprocess.PIPE, "text": True, "encoding": "utf-8",
                                          "env": launch_env, "cwd": str(ROOT), "creationflags": creationflags}, budget, deadline)
        process._c2_containment = create_worker_containment(process)
        _watch_worker(process, budget, deadline, "post-launch")
        if process.stdin is None or process.stdout is None or process.stderr is None:
            fail("WORKER_PROTOCOL_FAILURE", "worker pipes unavailable")
    except C2HarnessError as exc:
        journal_primary_failure(process, diagnostic_path, session_id=budget.session_id, job_id=job_id,
                                family=family, primary=exc)
        raise
    assert process is not None
    with diagnostic_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_protocol_line({"stage": "WORKER_PROCESS_LAUNCHED",
                                     "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                                     "pid": process.pid, "session_id": budget.session_id, "job_id": job_id, "family": family}))
    lines: Queue[str] = Queue()
    threading.Thread(target=_reader_thread, args=(process.stdout, lines), daemon=True).start()
    stderr_tail = BoundedDiagnosticTail()
    threading.Thread(target=_stderr_drain_thread, args=(process.stderr, stderr_tail), daemon=True).start()
    try:
        identity_line = _wait_worker_line(process, lines, budget, deadline, "identity")
        _watch_worker(process, budget, deadline, "post-identity-receipt")
        identity = json.loads(identity_line)
        _watch_worker(process, budget, deadline, "post-identity-parse")
        if not isinstance(identity, Mapping):
            fail("C2_WORKER_IDENTITY_MISMATCH", "identity is not an object")
        validate_worker_identity(identity)
        verify_contained_worker(process._c2_containment, identity["pid"], identity["parent_pid"])
        process._c2_worker_pid = identity["pid"]
        _watch_worker(process, budget, deadline, "post-identity-validation")
        with diagnostic_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_protocol_line({"stage": "WORKER_IDENTITY_RECEIVED", "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"), "pid": process.pid, "session_id": budget.session_id, "job_id": job_id, "family": family}))
        if identity["launch_nonce"] in budget.worker_nonces or job_id in budget.job_ids:
            fail("WORKER_AUTHORIZATION_FAILURE", "worker nonce or job identity replayed")
        budget.worker_nonces.add(identity["launch_nonce"])
        budget.job_ids.add(job_id)
        delivery: Queue[BaseException | None] = Queue(maxsize=1)
        payload = worker_capability_message(session_id=budget.session_id, parent_pid=process.pid,
                                            worker_pid=identity["pid"], launch_nonce=identity["launch_nonce"],
                                            job_id=job_id, job_sha256=job_sha256, job_scope=job_scope, job_path=job_path, family=family,
                                            family_ordinal=family_ordinal, aggregate_ordinal=aggregate_ordinal)
        threading.Thread(target=_capability_writer, args=(process.stdin, payload, delivery), daemon=True).start()
        _wait_capability_delivery(process, delivery, budget, deadline)
        _watch_worker(process, budget, deadline, "post-capability-delivery")
        with diagnostic_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_protocol_line({"stage": "CAPABILITY_SENT", "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"), "pid": process.pid, "session_id": budget.session_id, "job_id": job_id, "family": family}))
        result_line = _wait_worker_line(process, lines, budget, deadline, "fit-result")
        _watch_worker(process, budget, deadline, "post-result-receipt")
        append_worker_diagnostic(diagnostic_path, "RESULT_RECEIVED", session_id=budget.session_id, job_id=job_id,
                                 family=family, process=process)
        result = json.loads(result_line)
        _watch_worker(process, budget, deadline, "post-result-parse")
        if not isinstance(result, dict) or result.get("kind") != "C2_WORKER_RESULT":
            fail("WORKER_RESULT_CONTRACT_FAILURE", "missing label-free worker result")
        forbidden = set(result).intersection(FORBIDDEN_SCORE_FIELDS)
        if forbidden:
            fail("WORKER_RESULT_CONTRACT_FAILURE", f"forbidden worker fields: {sorted(forbidden)}")
        _watch_worker(process, budget, deadline, "post-result-validation")
        append_worker_diagnostic(diagnostic_path, "RESULT_VALIDATED", session_id=budget.session_id, job_id=job_id,
                                 family=family, process=process)
        while process.poll() is None:
            _watch_worker(process, budget, deadline, "worker-exit")
            time.sleep(min(0.05, max(0.001, deadline.remaining_seconds())))
        if process.returncode != 0:
            fail("WORKER_NONZERO_EXIT", str(process.returncode))
        _watch_worker(process, budget, deadline, "pre-fit-result-return")
        process._c2_containment.close()
        return result
    except (json.JSONDecodeError, BrokenPipeError) as exc:
        primary = C2HarnessError(f"WORKER_PROTOCOL_FAILURE: {exc}; stderr_tail={stderr_tail.text()[-1024:]}")
        journal_primary_failure(process, diagnostic_path, session_id=budget.session_id, job_id=job_id,
                                family=family, primary=primary)
        raise primary
    except C2HarnessError as exc:
        journal_primary_failure(process, diagnostic_path, session_id=budget.session_id, job_id=job_id,
                                family=family, primary=exc)
        raise
    except Exception as exc:
        primary = C2HarnessError(f"WORKER_PROTOCOL_FAILURE: {type(exc).__name__}: {exc}")
        journal_primary_failure(process, diagnostic_path, session_id=budget.session_id, job_id=job_id,
                                family=family, primary=primary)
        raise primary from exc


def integrated_launch_journal_io_self_check() -> bool:
    """Run the real parent launch lifecycle against a label-free synthetic worker, never a Task1 fit."""
    session_id = f"integrated-{secrets.token_urlsafe(12)}"
    job_id = f"integrated-job-{secrets.token_urlsafe(12)}"
    job_scope = "inner_o1_i2"
    diagnostic_path = worker_diagnostic_path(session_id, job_id)
    job_path = worker_job_path(job_scope, "c2_v", job_id)
    original_launcher = _launch_worker_bounded
    try:
        synthetic_code = (
            "import json,os,sys; "
            "e=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONHASHSEED='0'); "
            "i=dict(kind='C2_WORKER_IDENTITY',pid=os.getpid(),parent_pid=os.getppid(),launch_nonce='n'*48,python='3.12.6',numpy='1.26.4',scikit_learn='1.7.2',threadpoolctl='3.6.0',environment=e); "
            "print(json.dumps(i),flush=True); c=json.loads(sys.stdin.readline()); "
            "print(json.dumps(dict(kind='C2_WORKER_RESULT',family=c['family'],scores=[])),flush=True)"
        )

        def synthetic_launcher(_command: Sequence[str], _kwargs: Mapping[str, Any], _budget: C2RunBudget,
                               _deadline: FitDeadline) -> subprocess.Popen[str]:
            return subprocess.Popen([str(pinned_python()), "-c", synthetic_code], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    env=worker_environment(), cwd=str(ROOT))

        globals()["_launch_worker_bounded"] = synthetic_launcher
        result = launch_fit_worker(job_path, job_id, "a" * 64, job_scope, C2RunBudget.start(session_id), "c2_v")
        records = [strict_json_object(line.encode("utf-8"), "INTEGRATED_JOURNAL")
                   for line in diagnostic_path.read_text(encoding="utf-8").splitlines()]
        stages = [record.get("stage") for record in records]
        success_ok = (result == {"kind": "C2_WORKER_RESULT", "family": "c2_v", "scores": []}
                      and diagnostic_path.is_file() and diagnostic_path.stat().st_size > 0
                      and stages[:2] == ["WORKER_PROCESS_LAUNCHING", "WORKER_PROCESS_LAUNCHED"]
                      and "WORKER_IDENTITY_RECEIVED" in stages and "RESULT_RECEIVED" in stages
                      and "RESULT_VALIDATED" in stages)
        failure_session = f"integrated-{secrets.token_urlsafe(12)}"
        failure_job = f"integrated-job-{secrets.token_urlsafe(12)}"
        failure_path = worker_diagnostic_path(failure_session, failure_job)

        def failed_launcher(_command: Sequence[str], _kwargs: Mapping[str, Any], _budget: C2RunBudget,
                            _deadline: FitDeadline) -> subprocess.Popen[str]:
            fail("WORKER_PROTOCOL_FAILURE", "synthetic integrated launch failure")
            raise AssertionError("unreachable")

        globals()["_launch_worker_bounded"] = failed_launcher
        try:
            launch_fit_worker(worker_job_path(job_scope, "c2_v", failure_job), failure_job, "a" * 64, job_scope,
                              C2RunBudget.start(failure_session), "c2_v")
            early_failure_ok = False
        except C2HarnessError as exc:
            failure_records = [strict_json_object(line.encode("utf-8"), "INTEGRATED_LAUNCH_FAILURE")
                               for line in failure_path.read_text(encoding="utf-8").splitlines()]
            early_failure_ok = (str(exc).startswith("WORKER_PROTOCOL_FAILURE")
                                and [record.get("stage") for record in failure_records]
                                == ["WORKER_PROCESS_LAUNCHING", "PROTOCOL_ABORT", "CLEANUP_STARTED", "CLEANUP_COMPLETED"])
        return success_ok and early_failure_ok
    except (C2HarnessError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired, TypeError):
        return False
    finally:
        globals()["_launch_worker_bounded"] = original_launcher


def future_fit_only_after_separate_authorization(estimator: Any, x: Any, y: Any, weights: Any, threadpoolctl: Any) -> Any:
    """Future-only call site, deliberately unreachable while contract is PREFLIGHT_ONLY."""
    with threadpoolctl.threadpool_limits(limits=1):
        return getattr(estimator, "fit")(x, y, sample_weight=weights)


@dataclass(frozen=True)
class FrozenArtifact:
    """An immutable byte identity. Label APIs require this, never mutable rows."""

    path: Path
    sha256: str
    payload: bytes
    kind: str
    state: str = "FROZEN"

    def require_frozen(self) -> None:
        if self.state != "FROZEN" or hashlib.sha256(self.payload).hexdigest() != self.sha256:
            fail("PREDICTION_FREEZE_BOUNDARY_FAILURE", self.kind)


@dataclass(frozen=True)
class InnerMetric:
    recall_delta: float
    precision_delta: float


@dataclass(frozen=True)
class MarginDecision:
    selected_margin: float
    challenger_inner_gate_possible: bool
    candidates: tuple[float, ...]
    candidate_metrics: tuple[tuple[float, InnerMetric, float], ...]


@dataclass(frozen=True)
class MarginCandidateArtifacts:
    outer_validation_fold: int
    margin: float
    candidate_index: int
    score_artifact_sha256: Mapping[int, str]
    selected_artifacts: Mapping[int, FrozenArtifact]
    top5_artifacts: Mapping[int, FrozenArtifact]
    input_identity: str


@dataclass(frozen=True)
class C2InnerGateDecision:
    outer_validation_fold: int
    pooled_recall_delta: float
    per_inner_fold_recall_delta: Mapping[int, float]
    pooled_precision_delta: float
    prediction_and_provenance_pass: bool
    passed: bool


def execution_output_root(candidate: Path | None = None) -> Path:
    """The single governed scientific root; caller substitution is forbidden."""
    approved = (ROOT / EXECUTION_ROOT_REL).resolve()
    actual = approved if candidate is None else candidate.resolve()
    if actual != approved:
        fail("OUTPUT_ROOT_REJECTED", f"only {EXECUTION_ROOT_REL.as_posix()}/ is approved")
    return approved


def worker_job_path(scope: str, family: str, job_id: str, root: Path | None = None) -> Path:
    """Derive a non-substitutable, immutable path below the governed worker-job root."""
    approved = execution_output_root() if root is None else root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,95}", scope):
        fail("WORKER_PROTOCOL_FAILURE", "job scope format invalid")
    if family not in ("c2_v", "c2_r") or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", job_id):
        fail("WORKER_PROTOCOL_FAILURE", "job family or identity format invalid")
    path = (approved / ".worker_jobs" / scope / family / f"{job_id}.json").resolve()
    if approved not in path.parents:
        fail("WORKER_PROTOCOL_FAILURE", "derived worker job path escaped execution root")
    return path


def worker_diagnostic_path(session_id: str, job_id: str) -> Path:
    root = execution_output_root()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", session_id):
        fail("WORKER_PROTOCOL_FAILURE", "diagnostic session identity invalid")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", job_id):
        fail("WORKER_PROTOCOL_FAILURE", "diagnostic job identity invalid")
    path = (root / "worker_diagnostics" / session_id / f"{job_id}.jsonl").resolve()
    if root not in path.parents:
        fail("WORKER_PROTOCOL_FAILURE", "diagnostic path escaped execution root")
    return path


_TERMINAL_DIAGNOSTIC_STAGES = frozenset({
    "FIT_TIMEOUT", "PROTOCOL_ABORT", "RESOURCE_ABORT", "RESULT_RECEIVED", "RESULT_VALIDATED",
    "CLEANUP_STARTED", "CLEANUP_COMPLETED", "CLEANUP_FAILED", "WORKER_EXIT", "WORKER_EXIT_UNCONFIRMED",
})


def append_worker_diagnostic(path: Path, stage: str, *, session_id: str, job_id: str, family: str,
                             process: subprocess.Popen[str] | None = None,
                             failure_category: str | None = None) -> bool:
    """Best-effort, label-free parent journal; observability never masks worker failure."""
    if stage not in _TERMINAL_DIAGNOSTIC_STAGES and not stage.startswith("WORKER_") and stage != "CAPABILITY_SENT":
        fail("WORKER_PROTOCOL_FAILURE", f"unknown diagnostic stage {stage}")
    payload: dict[str, Any] = {
        "stage": stage,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": session_id,
        "job_id": job_id,
        "family": family,
    }
    if process is not None:
        payload["launcher_pid"] = process.pid
        worker_pid = getattr(process, "_c2_worker_pid", None)
        if isinstance(worker_pid, int):
            payload["worker_pid"] = worker_pid
        if process.poll() is not None:
            payload["exit_code"] = process.returncode
    if failure_category is not None:
        payload["failure_category"] = failure_category
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_protocol_line(payload))
            handle.flush()
        return True
    except OSError:
        return False


def terminal_stage_for_failure(exc: C2HarnessError) -> str:
    category = str(exc).split(":", 1)[0]
    if category in {"FIT_TIMEOUT", "TOTAL_C2_TIMEOUT"}:
        return "FIT_TIMEOUT"
    if category == "RSS_LIMIT_EXCEEDED":
        return "RESOURCE_ABORT"
    return "PROTOCOL_ABORT"


def cleanup_worker_after_primary_failure(process: subprocess.Popen[str] | None, diagnostic_path: Path, *,
                                         session_id: str, job_id: str, family: str,
                                         primary: C2HarnessError) -> bool:
    """Attempt all bounded cleanup without replacing the original lifecycle failure."""
    append_worker_diagnostic(diagnostic_path, "CLEANUP_STARTED", session_id=session_id, job_id=job_id,
                             family=family, process=process, failure_category=str(primary).split(":", 1)[0])
    cleanup_ok = True
    if process is not None:
        try:
            terminate_worker(process)
        except C2HarnessError:
            cleanup_ok = False
    exit_verified = process is not None and process.poll() is not None
    append_worker_diagnostic(diagnostic_path, "CLEANUP_COMPLETED" if cleanup_ok else "CLEANUP_FAILED",
                             session_id=session_id, job_id=job_id, family=family, process=process,
                             failure_category=None if cleanup_ok else "C2_WORKER_SURVIVED_TERMINATION")
    if process is not None:
        append_worker_diagnostic(diagnostic_path, "WORKER_EXIT" if exit_verified else "WORKER_EXIT_UNCONFIRMED",
                                 session_id=session_id, job_id=job_id, family=family, process=process,
                                 failure_category=None if exit_verified else "C2_WORKER_SURVIVED_TERMINATION")
    return cleanup_ok and (process is None or exit_verified)


def journal_primary_failure(process: subprocess.Popen[str] | None, diagnostic_path: Path, *, session_id: str,
                            job_id: str, family: str, primary: C2HarnessError) -> None:
    """Persist the primary failure before any cleanup attempt, even before Popen exists."""
    category = str(primary).split(":", 1)[0]
    append_worker_diagnostic(diagnostic_path, terminal_stage_for_failure(primary), session_id=session_id, job_id=job_id,
                             family=family, process=process, failure_category=category)
    cleanup_worker_after_primary_failure(process, diagnostic_path, session_id=session_id, job_id=job_id,
                                         family=family, primary=primary)


def bundle_directory(outer_validation_fold: int, family: str, inner_validation_fold: int | None = None) -> Path:
    require_scientific_fold(outer_validation_fold, "outer bundle")
    if family not in ("c2_v", "c2_r"):
        fail("OUTPUT_IDENTITY_FAILURE", f"unknown family {family}")
    root = execution_output_root()
    directory = root / f"outer_f{outer_validation_fold}"
    if inner_validation_fold is not None:
        require_scientific_fold(inner_validation_fold, "inner bundle")
        directory = directory / f"inner_validation_f{inner_validation_fold}"
    return directory / family


def canonical_jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Exact governed bytes: sorted keys, ASCII, compact JSON, one LF per row."""
    payload = bytearray()
    for record in records:
        if not isinstance(record, Mapping):
            fail("OUTPUT_SERIALIZATION_FAILURE", "JSONL record must be an object")
        payload.extend(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        payload.extend(b"\n")
    return bytes(payload)


def stored_validation_order(folds_document: Mapping[str, Any], fold: int) -> tuple[str, ...]:
    """Read the stored validation_ids sequence without sorting or rediscovery."""
    require_scientific_fold(fold, "stored validation IDs")
    entries = folds_document.get("folds")
    if not isinstance(entries, list):
        fail("FOLD_SCHEMA_FAILURE", "folds array missing")
    matches = [entry for entry in entries if isinstance(entry, Mapping) and int(entry.get("fold", -1)) == fold]
    if len(matches) != 1 or not isinstance(matches[0].get("validation_ids"), list):
        fail("FOLD_SCHEMA_FAILURE", f"missing unique validation_ids for F{fold}")
    values = tuple(matches[0]["validation_ids"])
    if len(values) != 1400 or len(set(values)) != len(values) or not all(isinstance(value, str) for value in values):
        fail("FOLD_SCHEMA_FAILURE", f"invalid stored validation_ids for F{fold}")
    return values


def order_records_by_validation_ids(records: Iterable[Mapping[str, Any]], validation_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    """Query order is the frozen sequence; only same-query action rows are sorted."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        query_id = record.get("query_id")
        if not isinstance(query_id, str):
            fail("OUTPUT_SERIALIZATION_FAILURE", "record query_id missing")
        grouped[query_id].append(record)
    if set(grouped) != set(validation_ids) or len(validation_ids) != len(set(validation_ids)):
        fail("OUTPUT_SERIALIZATION_FAILURE", "query population does not exactly match stored validation IDs")
    ordered: list[Mapping[str, Any]] = []
    for query_id in validation_ids:
        rows = grouped[query_id]
        # This key is meaningful for action rows; fixed top5 rows have one row.
        rows.sort(key=lambda row: (-int(row.get("drop_rank", 0)), str(row.get("incoming_doc_id", "")).encode("utf-8")))
        ordered.extend(rows)
    return ordered


def validate_selected_action_records(records: Sequence[Mapping[str, Any]], expected_fold: int,
                                     validation_ids: Sequence[str]) -> None:
    require_scientific_fold(expected_fold, "selected-action artifact")
    ordered = order_records_by_validation_ids(records, validation_ids)
    if list(records) != ordered:
        fail("OUTPUT_SERIALIZATION_FAILURE", "selected actions are not in frozen row order")
    for record in records:
        if int(record.get("fold", -1)) != expected_fold:
            fail("FOLD0_SCIENTIFIC_LEAKAGE", "selected-action artifact fold mismatch")
        assert_score_view_label_free(record)
        selected = record.get("selected_action")
        if selected is not None:
            if not isinstance(selected, Mapping):
                fail("OUTPUT_STRUCTURAL_FAILURE", "selected_action must be null or object")
            if int(selected.get("drop_rank", -1)) not in (4, 5) or not isinstance(selected.get("incoming_doc_id"), str):
                fail("OUTPUT_STRUCTURAL_FAILURE", "invalid selected action")


def validate_top5_records(records: Sequence[Mapping[str, Any]], expected_fold: int,
                           validation_ids: Sequence[str], baseline: Mapping[str, BaselineIdentity]) -> None:
    require_scientific_fold(expected_fold, "top5 artifact")
    ordered = order_records_by_validation_ids(records, validation_ids)
    if list(records) != ordered:
        fail("OUTPUT_SERIALIZATION_FAILURE", "top5 rows are not in frozen row order")
    for record in records:
        query_id = record.get("query_id")
        if int(record.get("fold", -1)) != expected_fold or query_id not in baseline:
            fail("OUTPUT_STRUCTURAL_FAILURE", "top5 identity mismatch")
        top5 = record.get("top5")
        if not isinstance(top5, list) or len(top5) != 5 or len(set(top5)) != 5:
            fail("TOP5_CONTRACT_FAILURE", "final top5 invalid")
        if not all(isinstance(document, str) for document in top5):
            fail("TOP5_CONTRACT_FAILURE", "top5 identity type changed")
        selected = record.get("selected_action")
        action = None if selected is None else ScoringAction(query_id, expected_fold, int(selected["drop_rank"]),
                                                              selected["incoming_doc_id"], MappingProxyType({}))
        if one_swap_top5(baseline[query_id].top5, action) != top5:
            fail("TOP5_CONTRACT_FAILURE", "final top5 does not equal canonical one-swap result")


def freeze_artifact(path: Path, payload: bytes, kind: str, validator: Callable[[], None],
                    resume_expected_sha256: str | None = None) -> FrozenArtifact:
    """Validate then atomically promote bytes; frozen files are never overwritten."""
    root = execution_output_root()
    resolved = path.resolve()
    if root not in resolved.parents:
        fail("OUTPUT_ROOT_REJECTED", str(path))
    validator()
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if resume_expected_sha256 != digest or sha256_file(path) != digest:
            fail("FROZEN_ARTIFACT_OVERWRITE_REJECTED", str(path))
        return FrozenArtifact(path, digest, payload, kind)
    staging = root / ".staging" / path.relative_to(root)
    staging = staging.with_suffix(staging.suffix + ".pending")
    staging.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if sha256_file(staging) != digest:
        fail("ARTIFACT_SHA_PROVENANCE_FAILURE", str(path))
    os.replace(staging, path)
    if sha256_file(path) != digest:
        fail("ARTIFACT_SHA_PROVENANCE_FAILURE", str(path))
    return FrozenArtifact(path, digest, payload, kind)


def require_frozen_for_label_loading(artifact: FrozenArtifact, expected_sha256: str) -> None:
    artifact.require_frozen()
    if artifact.sha256 != expected_sha256:
        fail("PREDICTION_FREEZE_BOUNDARY_FAILURE", "artifact identity not supplied/mismatched")


def evaluate_after_freeze(artifact: FrozenArtifact, expected_sha256: str,
                          label_loader: Callable[[FrozenArtifact], Any],
                          evaluator: Callable[[FrozenArtifact, Any], InnerMetric]) -> InnerMetric:
    """The sole label boundary. A loader cannot be invoked until a SHA is frozen."""
    require_frozen_for_label_loading(artifact, expected_sha256)
    labels = label_loader(artifact)
    result = evaluator(artifact, labels)
    if not all(math.isfinite(value) for value in (result.recall_delta, result.precision_delta)):
        fail("INNER_METRIC_FAILURE", "non-finite post-freeze metric")
    return result


def select_c2r_margin(frozen_inner_artifacts: Mapping[int, FrozenArtifact],
                      best_scores: Mapping[str, float],
                      candidate_factory: Callable[[float, int], MarginCandidateArtifacts],
                      label_loader: Callable[[FrozenArtifact], Any],
                      evaluator: Callable[[FrozenArtifact, Any, float], InnerMetric], np: Any) -> MarginDecision:
    """Inner-only margin selection with candidate action/top5 freeze before labels."""
    if len(frozen_inner_artifacts) != 3:
        fail("MARGIN_CANDIDATE_FAILURE", "one outer complement requires exactly three frozen inner artifacts")
    candidates = margin_candidates(tuple(best_scores.values()), np)
    evaluated: list[tuple[float, InnerMetric, float]] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_artifacts = candidate_factory(candidate, candidate_index)
        if candidate_artifacts.margin != candidate or candidate_artifacts.candidate_index != candidate_index:
            fail("MARGIN_CANDIDATE_FAILURE", "candidate identity mismatch")
        if set(candidate_artifacts.score_artifact_sha256) != set(frozen_inner_artifacts):
            fail("MARGIN_CANDIDATE_FAILURE", "candidate score identity incomplete")
        if (not candidate_artifacts.input_identity or
                set(candidate_artifacts.top5_artifacts) != set(candidate_artifacts.selected_artifacts)):
            fail("MARGIN_CANDIDATE_FAILURE", "candidate output identity incomplete")
        per_fold: list[InnerMetric] = []
        for fold, artifact in candidate_artifacts.selected_artifacts.items():
            require_scientific_fold(fold, "margin artifact")
            artifact.require_frozen()
            candidate_artifacts.top5_artifacts[fold].require_frozen()
            if candidate_artifacts.score_artifact_sha256[fold] != frozen_inner_artifacts[fold].sha256:
                fail("MARGIN_CANDIDATE_FAILURE", "candidate score SHA changed")
            # Labels can only be requested after candidate selected-action and
            # top5 bytes were validated and frozen.  The SHA is mandatory.
            labels = label_loader(artifact)
            metric = evaluator(artifact, labels, candidate)
            if not all(math.isfinite(value) for value in (metric.recall_delta, metric.precision_delta)):
                fail("INNER_METRIC_FAILURE", "non-finite margin metric")
            per_fold.append(metric)
        pooled = InnerMetric(
            recall_delta=sum(item.recall_delta for item in per_fold) / len(per_fold),
            precision_delta=sum(item.precision_delta for item in per_fold) / len(per_fold),
        )
        evaluated.append((candidate, pooled, min(item.recall_delta for item in per_fold)))
    admissible = [item for item in evaluated if item[1].precision_delta >= -0.001]
    if not admissible:
        return MarginDecision(math.inf, False, candidates, tuple(evaluated))
    winner = max(admissible, key=lambda item: (item[1].recall_delta, item[1].precision_delta, item[2], item[0]))
    return MarginDecision(winner[0], True, candidates, tuple(evaluated))


def c2_inner_gate(outer_validation_fold: int, c2v: Mapping[int, InnerMetric], c2r: Mapping[int, InnerMetric],
                  prediction_and_provenance_pass: bool) -> C2InnerGateDecision:
    require_scientific_fold(outer_validation_fold, "C2 inner gate")
    expected_inner = set(TARGET_FOLDS) - {outer_validation_fold}
    if set(c2v) != expected_inner or set(c2r) != expected_inner:
        fail("C2_INNER_GATE_FAILURE", "each outer complement requires exactly its three inner folds")
    recall = {fold: c2r[fold].recall_delta - c2v[fold].recall_delta for fold in expected_inner}
    precision = [c2r[fold].precision_delta - c2v[fold].precision_delta for fold in expected_inner]
    pooled_recall = sum(recall.values()) / len(recall)
    pooled_precision = sum(precision) / len(precision)
    passed = (prediction_and_provenance_pass and pooled_recall >= 0.002
              and all(value >= 0.0 for value in recall.values()) and pooled_precision >= -0.001)
    return C2InnerGateDecision(outer_validation_fold, pooled_recall, MappingProxyType(recall), pooled_precision,
                               prediction_and_provenance_pass, passed)


def scientific_metadata(*, input_hashes: Mapping[str, str], family: str, estimator_params: Mapping[str, Any],
                        model_seed: int, outer_fold: int, query_count: int, action_count: int,
                        artifact_hashes: Mapping[str, str], selected_margin: float | None,
                        inner_train_folds: Sequence[int] | None = None,
                        inner_validation_fold: int | None = None) -> dict[str, Any]:
    """Label-free durable provenance for each frozen future artifact bundle."""
    require_scientific_fold(outer_fold, "metadata")
    if inner_validation_fold is not None:
        require_scientific_fold(inner_validation_fold, "metadata")
    if inner_train_folds is not None:
        for fold in inner_train_folds:
            require_scientific_fold(int(fold), "metadata")
    schema_hash = input_hashes.get("reports/task1/workflow_c/tv4/c1i/c1i_feature_schema_report.json")
    if not isinstance(schema_hash, str):
        fail("ARTIFACT_SHA_PROVENANCE_FAILURE", "frozen feature schema hash missing")
    return {
        "artifact_sha256": dict(artifact_hashes), "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "feature_schema_identity": f"c1i_feature_schema_report_sha256:{schema_hash}", "fold0_excluded": True,
        "gpu": False, "inner_train_folds": None if inner_train_folds is None else list(inner_train_folds),
        "inner_validation_fold": inner_validation_fold, "input_hashes": dict(input_hashes),
        "modal": False, "model_family": family, "model_seed": model_seed,
        "outer_fold": outer_fold, "public_labels_used": False, "python_version": REQUIRED_VERSIONS["python"],
        "numpy_version": REQUIRED_VERSIONS["numpy"], "scikit_learn_version": REQUIRED_VERSIONS["scikit_learn"],
        "threadpoolctl_version": REQUIRED_VERSIONS["threadpoolctl"], "estimator_params": dict(estimator_params),
        "query_count": query_count, "action_count": action_count,
        "selected_margin": "+Infinity" if selected_margin is not None and math.isinf(selected_margin) else selected_margin,
        "script_sha256": running_script_sha256(), "serialization_contract": SERIALIZATION_CONTRACT_ID,
        "provenance_manifest_sha256": EXPECTED_PROVENANCE_SHA256,
    }


def frozen_estimator_params(family: str) -> dict[str, Any]:
    contract, _provenance = load_frozen_contracts()
    if family == "c2_v":
        return c2v_parameters(contract)
    if family == "c2_r":
        return c2r_parameters(contract)
    fail("C2_ESTIMATOR_CONTRACT_FAILURE", f"unknown family {family}")
    return {}


def _protocol_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def emit_worker_stage(stage: str) -> None:
    """Append only label-free worker lifecycle facts to the parent-derived diagnostic journal."""
    path_text = os.environ.get("C2_WORKER_DIAGNOSTIC_PATH")
    if not path_text:
        return
    path = Path(path_text).resolve()
    root = execution_output_root()
    if root not in path.parents:
        fail("WORKER_PROTOCOL_FAILURE", "diagnostic path escapes execution root")
    payload = {"stage": stage, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
               "pid": os.getpid(), "session_id": os.environ.get("C2_EXPECTED_SESSION_ID", ""),
               "job_id": os.environ.get("C2_EXPECTED_JOB_ID", ""), "family": os.environ.get("C2_EXPECTED_FAMILY", "")}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_protocol_line(payload))
        handle.flush()


def fit_worker(job_path_text: str) -> int:
    """Dedicated fit process entry point; it cannot fit until parent approval."""
    try:
        identity = worker_identity_envelope()  # imports/sklearn identity gate happens here, first.
        emit_worker_stage("WORKER_IDENTITY_READY")
        sys.stdout.write(_protocol_line(identity))
        sys.stdout.flush()
        state = "UNAUTHORIZED"
        expected = worker_launch_context(identity)
        capability = validate_worker_capability(sys.stdin.readline(), expected)
        emit_worker_stage("CAPABILITY_VALIDATED")
        state = "AUTHORIZED_ONCE"
        job_path = Path(job_path_text).resolve()
        root = execution_output_root()
        if (job_path != Path(expected["job_path"]).resolve()
                or job_path != worker_job_path(expected["job_scope"], expected["family"], expected["job_id"])):
            fail("WORKER_AUTHORIZATION_FAILURE", "job path does not match live launch context")
        emit_worker_stage("JOB_READ_STARTED")
        job_bytes = job_path.read_bytes()
        emit_worker_stage("JOB_READ_COMPLETE")
        if hashlib.sha256(job_bytes).hexdigest() != capability["job_sha256"]:
            fail("WORKER_AUTHORIZATION_FAILURE", "job SHA256 mismatch")
        emit_worker_stage("JOB_HASH_VERIFIED")
        job = strict_json_object(job_bytes, "WORKER_PROTOCOL_FAILURE")
        emit_worker_stage("JOB_PARSE_COMPLETE")
        state = "CONSUMED"
        if not isinstance(job, Mapping):
            fail("WORKER_PROTOCOL_FAILURE", "job must be an object")
        family = job.get("family")
        if family not in ("c2_v", "c2_r"):
            fail("WORKER_PROTOCOL_FAILURE", "unknown family")
        if family != capability["family"] or job.get("job_id") != capability["job_id"] or job.get("job_scope") != capability["job_scope"]:
            fail("WORKER_AUTHORIZATION_FAILURE", "job/capability mismatch")
        score_identities = job.get("score_identities")
        if not isinstance(score_identities, list):
            fail("WORKER_RESULT_CONTRACT_FAILURE", "label-free score identities absent")
        for item in score_identities:
            if not isinstance(item, Mapping):
                fail("WORKER_RESULT_CONTRACT_FAILURE", "invalid score identity")
            assert_score_view_label_free(item)
        contract, _provenance = load_frozen_contracts()
        np, sklearn, threadpoolctl = require_environment_before_sklearn(require_pinned_executable=True)
        x_train = np.asarray(job.get("train_matrix"), dtype=np.float32)
        y_train = np.asarray(job.get("train_targets"))
        weights = np.asarray(job.get("sample_weights"), dtype=np.float64)
        x_score = np.asarray(job.get("score_matrix"), dtype=np.float32)
        emit_worker_stage("MATRICES_READY")
        if (x_train.ndim != 2 or x_train.shape[1] != EXPECTED_FEATURE_COUNT or x_score.ndim != 2
                or x_score.shape[1] != EXPECTED_FEATURE_COUNT or len(x_train) != len(y_train)
                or len(weights) != len(y_train) or len(x_score) != len(score_identities)):
            fail("WORKER_PROTOCOL_FAILURE", "matrix/target dimensions violate the frozen schema")
        if not bool(np.isfinite(x_train).all()) or not bool(np.isfinite(x_score).all()) or not bool(np.isfinite(weights).all()):
            fail("NONFINITE_FEATURE_FAILURE", "worker matrix")
        estimator = construct_c2v_estimator(contract, sklearn) if family == "c2_v" else construct_c2r_estimator(contract, sklearn)
        emit_worker_stage("ESTIMATOR_CONSTRUCTED")
        emit_worker_stage("FIT_STARTED")
        future_fit_only_after_separate_authorization(estimator, x_train, y_train, weights, threadpoolctl)
        emit_worker_stage("FIT_COMPLETED")
        emit_worker_stage("PREDICTION_STARTED")
        scores = future_c2v_scores(estimator, x_score, np) if family == "c2_v" else future_c2r_scores(estimator, x_score, np)
        emit_worker_stage("PREDICTION_COMPLETED")
        result_scores: list[dict[str, Any]] = []
        for identity_row, score in zip(score_identities, scores):
            result_scores.append(action_score_payload(
                ScoringAction(identity_row["query_id"], int(identity_row["fold"]), int(identity_row["drop_rank"]),
                              identity_row["incoming_doc_id"], MappingProxyType({})), float(score)))
        result = {"kind": "C2_WORKER_RESULT", "family": family, "scores": result_scores}
        sys.stdout.write(_protocol_line(result))
        sys.stdout.flush()
        emit_worker_stage("RESULT_EMITTED")
        return 0
    except C2HarnessError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # The parent treats every worker error as fatal; no retry.
        print(f"WORKER_UNHANDLED_FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def write_worker_job(path: Path, job: Mapping[str, Any], approved_root: Path | None = None) -> None:
    """Only future execution uses this pending job file, beneath the governed root."""
    root = execution_output_root() if approved_root is None else approved_root.resolve()
    if root not in path.resolve().parents:
        fail("OUTPUT_ROOT_REJECTED", str(path))
    for identity in job.get("score_identities", []):
        if not isinstance(identity, Mapping):
            fail("WORKER_PROTOCOL_FAILURE", "invalid score identity")
        assert_score_view_label_free(identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(job))
        handle.flush()
        os.fsync(handle.fileno())


def score_with_worker(family: str, train: Sequence[TrainingAction], score: Sequence[ScoringAction],
                      feature_names: Sequence[str], np: Any, budget: C2RunBudget,
                      job_scope: str) -> list[dict[str, Any]]:
    """Create one label-separated job and return only parent-visible label-free scores."""
    if family not in ("c2_v", "c2_r") or not train or not score:
        fail("POPULATION_MISMATCH", "empty/unknown worker job")
    for action in train:
        require_scientific_fold(action.fold, "worker training bundle")
    for action in score:
        require_scientific_fold(action.fold, "worker scoring bundle")
    targets = [classifier_target(action.exact_delta_recall) if family == "c2_v" else action.exact_delta_recall for action in train]
    job_id = secrets.token_urlsafe(24)
    job_path = worker_job_path(job_scope, family, job_id)
    job = {
        "job_id": job_id,
        "job_scope": job_scope,
        "family": family,
        "train_matrix": build_feature_matrix(train, feature_names, np).tolist(),
        "train_targets": targets,
        "sample_weights": query_balanced_weights(train),
        "score_matrix": build_feature_matrix(score, feature_names, np).tolist(),
        "score_identities": [
            {"query_id": action.query_id, "fold": action.fold, "drop_rank": action.drop_rank,
             "incoming_doc_id": action.incoming_doc_id}
            for action in score
        ],
    }
    write_worker_job(job_path, job)
    result = launch_fit_worker(job_path, job_id, sha256_file(job_path), job_scope, budget, family)
    scores = result.get("scores")
    if not isinstance(scores, list) or len(scores) != len(score):
        fail("WORKER_RESULT_CONTRACT_FAILURE", "worker score cardinality mismatch")
    for item in scores:
        if not isinstance(item, Mapping):
            fail("WORKER_RESULT_CONTRACT_FAILURE", "worker score row malformed")
        assert_score_view_label_free(item)
    return [dict(item) for item in scores]


def selected_and_top5_records(scores: Sequence[Mapping[str, Any]], actions: Mapping[tuple[str, int, str], ScoringAction],
                              baseline: Mapping[str, BaselineIdentity], margin: float,
                              validation_ids: Sequence[str], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parent-side deterministic action choice and exactly-one-swap reconstruction."""
    require_scientific_fold(fold, "selected/top5 records")
    grouped: dict[str, list[tuple[ScoringAction, float]]] = defaultdict(list)
    for row in scores:
        identity = (row.get("query_id"), int(row.get("drop_rank", -1)), row.get("incoming_doc_id"))
        action = actions.get(identity)
        if action is None or action.fold != fold:
            fail("WORKER_RESULT_CONTRACT_FAILURE", "worker score identity was not requested")
        grouped[action.query_id].append((action, float(row.get("score"))))
    if set(grouped) != set(validation_ids) or set(baseline) != set(validation_ids):
        fail("POPULATION_MISMATCH", "score/baseline query IDs mismatch frozen validation sequence")
    selected_records: list[dict[str, Any]] = []
    top5_records: list[dict[str, Any]] = []
    for query_id in validation_ids:  # Never sort; this is the stored folds.json sequence.
        chosen = choose_best_action(grouped[query_id], margin)
        selected = None if chosen is None else {"drop_rank": chosen.drop_rank, "incoming_doc_id": chosen.incoming_doc_id}
        selected_records.append({"fold": fold, "query_id": query_id, "selected_action": selected})
        top5_records.append({"fold": fold, "query_id": query_id, "selected_action": selected,
                             "top5": one_swap_top5(baseline[query_id].top5, chosen)})
    return selected_records, top5_records


def freeze_prediction_bundle(directory: Path, fold: int, validation_ids: Sequence[str],
                             selected_records: Sequence[Mapping[str, Any]], top5_records: Sequence[Mapping[str, Any]],
                             baseline: Mapping[str, BaselineIdentity]) -> tuple[FrozenArtifact, FrozenArtifact]:
    selected_path = directory / "selected_actions.jsonl"
    top5_path = directory / "final_top5.jsonl"
    selected = freeze_artifact(selected_path, canonical_jsonl_bytes(selected_records), "selected-actions",
                               lambda: validate_selected_action_records(selected_records, fold, validation_ids))
    top5 = freeze_artifact(top5_path, canonical_jsonl_bytes(top5_records), "final-top5",
                           lambda: validate_top5_records(top5_records, fold, validation_ids, baseline))
    return selected, top5


def freeze_label_free_scores(directory: Path, fold: int, validation_ids: Sequence[str],
                             scores: Sequence[Mapping[str, Any]]) -> FrozenArtifact:
    ordered = order_records_by_validation_ids(scores, validation_ids)
    if list(scores) != ordered:
        fail("OUTPUT_SERIALIZATION_FAILURE", "raw scores are not in frozen query/action order")
    for record in scores:
        if int(record.get("fold", -1)) != fold:
            fail("FOLD0_SCIENTIFIC_LEAKAGE", "score artifact fold mismatch")
        assert_score_view_label_free(record)
        if not math.isfinite(float(record.get("score"))):
            fail("NONFINITE_PREDICTION_FAILURE", str(record.get("query_id")))
    return freeze_artifact(directory / "label_free_scores.jsonl", canonical_jsonl_bytes(scores), "label-free-scores",
                           lambda: None)


def score_only(action: TrainingAction) -> ScoringAction:
    """Explicit physical separation: the score object has no target attribute."""
    return ScoringAction(action.query_id, action.fold, action.drop_rank, action.incoming_doc_id, action.features)


def best_scores_by_query(scores: Sequence[Mapping[str, Any]],
                         actions: Mapping[tuple[str, int, str], ScoringAction]) -> dict[str, float]:
    grouped: dict[str, list[tuple[ScoringAction, float]]] = defaultdict(list)
    for row in scores:
        identity = (row.get("query_id"), int(row.get("drop_rank", -1)), row.get("incoming_doc_id"))
        action = actions.get(identity)
        if action is None:
            fail("WORKER_RESULT_CONTRACT_FAILURE", "unknown score action")
        grouped[action.query_id].append((action, float(row["score"])))
    output: dict[str, float] = {}
    for query_id, candidates in grouped.items():
        validate_unique_action_identities(item[0] for item in candidates)
        output[query_id] = sorted(candidates, key=lambda item: (-item[1], -item[0].drop_rank,
                                                                 item[0].incoming_doc_id.encode("utf-8")))[0][1]
    return output


@dataclass(frozen=True)
class FutureExecutionData:
    """Execution-only data container; review-only never constructs one."""

    actions: tuple[TrainingAction, ...]
    baseline_by_query: Mapping[str, BaselineIdentity]
    validation_ids_by_fold: Mapping[int, tuple[str, ...]]


@dataclass(frozen=True)
class InnerFamilyResult:
    family: str
    score_artifacts: Mapping[int, FrozenArtifact]
    selected_artifacts: Mapping[int, FrozenArtifact]
    top5_artifacts: Mapping[int, FrozenArtifact]
    metrics: Mapping[int, InnerMetric]
    selected_margin: float


def _subset_baseline(baselines: Mapping[str, BaselineIdentity], query_ids: Sequence[str], fold: int) -> dict[str, BaselineIdentity]:
    result = {query_id: baselines[query_id] for query_id in query_ids if query_id in baselines}
    if set(result) != set(query_ids) or any(value.fold != fold for value in result.values()):
        fail("POPULATION_MISMATCH", "baseline does not exactly match validation population")
    return result


def _score_one_inner_fold(family: str, outer_fold: int, inner_fold: int, inner_train_folds: Sequence[int],
                          data: FutureExecutionData, feature_names: Sequence[str], np: Any, budget: C2RunBudget) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], ScoringAction]]:
    require_scientific_fold(outer_fold, "inner orchestration")
    require_scientific_fold(inner_fold, "inner orchestration")
    if tuple(inner_train_folds) != tuple(fold for fold in inner_train_folds if fold != inner_fold) or set(inner_train_folds) != set(TARGET_FOLDS) - {outer_fold, inner_fold}:
        fail("C2_SPLIT_CONTRACT_FAILURE", "inner training folds mismatch")
    train = [action for action in data.actions if action.fold in set(inner_train_folds)]
    score = [score_only(action) for action in data.actions if action.fold == inner_fold]
    if any(action.fold == 0 for action in train) or any(action.fold == 0 for action in score):
        fail("FOLD0_SCIENTIFIC_LEAKAGE", "inner worker bundle")
    identities = {action_identity(action): action for action in score}
    if len(identities) != len(score):
        fail("PROVENANCE_FAILURE", "duplicate inner score action")
    job_scope = f"inner_o{outer_fold}_i{inner_fold}"
    return score_with_worker(family, train, score, feature_names, np, budget, job_scope), identities


def run_inner_family(family: str, outer_fold: int, outer_training_folds: Sequence[int], data: FutureExecutionData,
                     feature_names: Sequence[str], input_hashes: Mapping[str, str], np: Any, budget: C2RunBudget,
                     label_loader: Callable[[FrozenArtifact], Any],
                     evaluator: Callable[[FrozenArtifact, Any, float], InnerMetric]) -> InnerFamilyResult:
    """Future complete inner lifecycle; labels are inaccessible until each score SHA freezes."""
    if family not in ("c2_v", "c2_r"):
        fail("C2_SPLIT_CONTRACT_FAILURE", "unknown model family")
    score_artifacts: dict[int, FrozenArtifact] = {}
    score_rows: dict[int, list[dict[str, Any]]] = {}
    score_actions: dict[int, dict[tuple[str, int, str], ScoringAction]] = {}
    for inner_fold, inner_train_folds in inner_rotations(outer_training_folds):
        scores, actions = _score_one_inner_fold(family, outer_fold, inner_fold, inner_train_folds, data, feature_names, np, budget)
        validation_ids = data.validation_ids_by_fold[inner_fold]
        ordered_scores = order_records_by_validation_ids(scores, validation_ids)
        directory = bundle_directory(outer_fold, family, inner_fold)
        score_artifacts[inner_fold] = freeze_label_free_scores(directory, inner_fold, validation_ids, ordered_scores)
        score_rows[inner_fold], score_actions[inner_fold] = ordered_scores, actions
    margin = 0.0
    if family == "c2_r":
        best = {f"{fold}:{query}": score for fold in outer_training_folds
                for query, score in best_scores_by_query(score_rows[fold], score_actions[fold]).items()}
        def candidate_factory(candidate: float, candidate_index: int) -> MarginCandidateArtifacts:
            selected_candidates: dict[int, FrozenArtifact] = {}
            top5_candidates: dict[int, FrozenArtifact] = {}
            for inner_fold in outer_training_folds:
                validation_ids = data.validation_ids_by_fold[inner_fold]
                baseline = _subset_baseline(data.baseline_by_query, validation_ids, inner_fold)
                selected_rows, top5_rows = selected_and_top5_records(score_rows[inner_fold], score_actions[inner_fold], baseline,
                                                                       candidate, validation_ids, inner_fold)
                candidate_dir = bundle_directory(outer_fold, family, inner_fold) / f"margin_candidate_{candidate_index:02d}"
                selected, top5 = freeze_prediction_bundle(candidate_dir, inner_fold, validation_ids,
                                                           selected_rows, top5_rows, baseline)
                selected_candidates[inner_fold], top5_candidates[inner_fold] = selected, top5
            identity_payload = {"outer_fold": outer_fold, "margin": candidate, "candidate_index": candidate_index,
                                "score_artifact_sha256": {str(fold): artifact.sha256 for fold, artifact in score_artifacts.items()}}
            identity = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
            return MarginCandidateArtifacts(outer_fold, candidate, candidate_index,
                                            MappingProxyType({fold: artifact.sha256 for fold, artifact in score_artifacts.items()}),
                                            MappingProxyType(selected_candidates), MappingProxyType(top5_candidates), identity)
        decision = select_c2r_margin(score_artifacts, best, candidate_factory, label_loader, evaluator, np)
        margin = decision.selected_margin
    selected_artifacts: dict[int, FrozenArtifact] = {}
    top5_artifacts: dict[int, FrozenArtifact] = {}
    metrics: dict[int, InnerMetric] = {}
    for inner_fold in outer_training_folds:
        validation_ids = data.validation_ids_by_fold[inner_fold]
        baseline = _subset_baseline(data.baseline_by_query, validation_ids, inner_fold)
        selected_rows, top5_rows = selected_and_top5_records(score_rows[inner_fold], score_actions[inner_fold], baseline,
                                                               margin, validation_ids, inner_fold)
        directory = bundle_directory(outer_fold, family, inner_fold)
        selected, top5 = freeze_prediction_bundle(directory, inner_fold, validation_ids, selected_rows, top5_rows, baseline)
        selected_artifacts[inner_fold], top5_artifacts[inner_fold] = selected, top5
        # The label loader receives an immutable selected-action identity only after both outputs freeze.
        metrics[inner_fold] = evaluate_after_freeze(selected, selected.sha256, label_loader,
                                                    lambda artifact, labels: evaluator(artifact, labels, margin))
        metadata = scientific_metadata(input_hashes=input_hashes, family=family,
                                       estimator_params=frozen_estimator_params(family), model_seed=2026 if family == "c2_v" else 2027,
                                       outer_fold=outer_fold, inner_train_folds=tuple(fold for fold in outer_training_folds if fold != inner_fold),
                                       inner_validation_fold=inner_fold, query_count=len(validation_ids),
                                       action_count=len(score_rows[inner_fold]), selected_margin=margin,
                                       artifact_hashes={"label_free_scores": score_artifacts[inner_fold].sha256,
                                                        "selected_actions": selected.sha256, "final_top5": top5.sha256})
        freeze_artifact(directory / "metadata.json", canonical_json_bytes(metadata), "metadata", lambda: None)
    return InnerFamilyResult(family, MappingProxyType(score_artifacts), MappingProxyType(selected_artifacts),
                             MappingProxyType(top5_artifacts), MappingProxyType(metrics), margin)


def outer_c2_stop_boundary(all_outer_bundles_frozen: bool) -> str:
    """C2 has no outer truth join, outer metrics, bootstrap, or C3 invocation."""
    if not all_outer_bundles_frozen:
        fail("OUTER_BUNDLE_FREEZE_FAILURE", "all F1-F4 bundles must freeze before C2 completion")
    return "READY_FOR_SEPARATE_C3_AUTHORIZATION"


def freeze_pooled_outer_artifact(family: str, artifact_name: str,
                                 by_outer_fold: Mapping[int, FrozenArtifact]) -> FrozenArtifact:
    """Pooled JSONL identity is the literal F1, F2, F3, F4 byte concatenation."""
    if set(by_outer_fold) != set(TARGET_FOLDS):
        fail("OUTPUT_SERIALIZATION_FAILURE", "pooled output requires exactly F1-F4")
    for fold in TARGET_FOLDS:
        by_outer_fold[fold].require_frozen()
    payload = b"".join(by_outer_fold[fold].payload for fold in TARGET_FOLDS)
    path = execution_output_root() / "pooled" / family / artifact_name
    return freeze_artifact(path, payload, f"pooled-{artifact_name}", lambda: None)


def run_outer_family(family: str, outer_fold: int, outer_training_folds: Sequence[int], data: FutureExecutionData,
                     feature_names: Sequence[str], input_hashes: Mapping[str, str], np: Any, budget: C2RunBudget,
                     selected_margin: float) -> tuple[FrozenArtifact, FrozenArtifact]:
    """Future outer C2 path ends with frozen label-free outputs and deliberately stops."""
    require_scientific_fold(outer_fold, "outer execution")
    if set(outer_training_folds) != set(TARGET_FOLDS) - {outer_fold}:
        fail("C2_SPLIT_CONTRACT_FAILURE", "outer training folds mismatch")
    train = [action for action in data.actions if action.fold in set(outer_training_folds)]
    score = [score_only(action) for action in data.actions if action.fold == outer_fold]
    validation_ids = data.validation_ids_by_fold[outer_fold]
    baseline = _subset_baseline(data.baseline_by_query, validation_ids, outer_fold)
    identities = {action_identity(action): action for action in score}
    if len(identities) != len(score):
        fail("PROVENANCE_FAILURE", "duplicate outer score action")
    directory = bundle_directory(outer_fold, family)
    scores = score_with_worker(family, train, score, feature_names, np, budget, f"outer_o{outer_fold}")
    ordered_scores = order_records_by_validation_ids(scores, validation_ids)
    score_artifact = freeze_label_free_scores(directory, outer_fold, validation_ids, ordered_scores)
    selected_rows, top5_rows = selected_and_top5_records(ordered_scores, identities, baseline, selected_margin,
                                                           validation_ids, outer_fold)
    selected, top5 = freeze_prediction_bundle(directory, outer_fold, validation_ids, selected_rows, top5_rows, baseline)
    metadata = scientific_metadata(input_hashes=input_hashes, family=family, estimator_params=frozen_estimator_params(family),
                                   model_seed=2026 if family == "c2_v" else 2027, outer_fold=outer_fold,
                                   query_count=len(validation_ids), action_count=len(ordered_scores),
                                   selected_margin=selected_margin,
                                   artifact_hashes={"label_free_scores": score_artifact.sha256,
                                                    "selected_actions": selected.sha256, "final_top5": top5.sha256})
    freeze_artifact(directory / "metadata.json", canonical_json_bytes(metadata), "metadata", lambda: None)
    # No outer label loader/evaluator exists in this call graph. C3 needs new authority.
    return selected, top5


def run_future_c2_execution(contract: Mapping[str, Any], provenance: Mapping[str, Any], data: FutureExecutionData,
                            label_loader: Callable[[FrozenArtifact], Any],
                            evaluator: Callable[[FrozenArtifact, Any, float], InnerMetric], session_id: str) -> str:
    """The complete future C2 schedule: exactly 16 C2-V + 16 C2-R fit attempts maximum.

    Current CLI authorization never reaches this function.  It is intentionally
    parameterized by a future, reviewed, structural data loader so review-only
    cannot parse Task1 labels or matrices.
    """
    input_hashes = verify_manifest_inputs(provenance)
    np, _sklearn, _threadpoolctl = require_environment_before_sklearn()
    feature_names = frozen_feature_names(contract)
    splits = outer_splits(contract)
    budget = C2RunBudget.start(session_id)
    all_outer_frozen = True
    pooled_selected: dict[str, dict[int, FrozenArtifact]] = {"c2_v": {}, "c2_r": {}}
    pooled_top5: dict[str, dict[int, FrozenArtifact]] = {"c2_v": {}, "c2_r": {}}
    for outer_fold in TARGET_FOLDS:  # F1, F2, F3, F4; never dictionary iteration order.
        training_folds = splits[outer_fold]
        c2v = run_inner_family("c2_v", outer_fold, training_folds, data, feature_names, input_hashes, np, budget,
                               label_loader, evaluator)
        c2r = run_inner_family("c2_r", outer_fold, training_folds, data, feature_names, input_hashes, np, budget,
                               label_loader, evaluator)
        gate = c2_inner_gate(outer_fold, c2v.metrics, c2r.metrics,
                             prediction_and_provenance_pass=(c2r.selected_margin != math.inf))
        if not gate.passed:
            fail("C2_INNER_GATE_FAILED", json.dumps({"outer_fold": outer_fold, "decision": gate.__dict__}, default=dict))
        v_selected, v_top5 = run_outer_family("c2_v", outer_fold, training_folds, data, feature_names, input_hashes,
                                               np, budget, 0.0)
        r_selected, r_top5 = run_outer_family("c2_r", outer_fold, training_folds, data, feature_names, input_hashes,
                                               np, budget, c2r.selected_margin)
        pooled_selected["c2_v"][outer_fold], pooled_top5["c2_v"][outer_fold] = v_selected, v_top5
        pooled_selected["c2_r"][outer_fold], pooled_top5["c2_r"][outer_fold] = r_selected, r_top5
    if budget.fits_started != MAXIMUM_PLANNED_FITS:
        fail("FIT_BUDGET_EXCEEDED", f"planned schedule launched {budget.fits_started}, expected {MAXIMUM_PLANNED_FITS}")
    for family in ("c2_v", "c2_r"):
        freeze_pooled_outer_artifact(family, "selected_actions.jsonl", pooled_selected[family])
        freeze_pooled_outer_artifact(family, "final_top5.jsonl", pooled_top5[family])
    return outer_c2_stop_boundary(all_outer_frozen)


_FOLD_IDENTITY_RE = re.compile(r'(?:(?:^|[,{])\s*)"fold"\s*:\s*(-?\d+)')


def structural_fold_from_raw_line(raw_line: str) -> int:
    """Extract only the fold scalar; does not JSON-decode any action payload."""
    match = _FOLD_IDENTITY_RE.search(raw_line)
    if match is None:
        fail("UNKNOWN_SCIENTIFIC_FOLD", "structural fold identity missing")
    try:
        return int(match.group(1))
    except ValueError as exc:
        fail("UNKNOWN_SCIENTIFIC_FOLD", str(exc))
    return -1


def read_actions_after_structural_fold_filter(path: Path) -> list[dict[str, Any]]:
    """Exclude Fold0 before decoding its complete label-bearing action object."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return decode_action_lines_after_structural_filter(handle, str(path))


def decode_action_lines_after_structural_filter(lines: Iterable[str], source: str = "synthetic") -> list[dict[str, Any]]:
    """Testable ingestion boundary: Fold0 lines are never passed to json.loads."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.endswith("\n"):
            fail("POPULATION_MISMATCH", f"non-LF action input at {source}:{number}")
        fold = structural_fold_from_raw_line(line)
        if fold == 0:
            continue
        require_scientific_fold(fold, "action structural boundary")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail("POPULATION_MISMATCH", f"{source}:{number}: {exc}")
        if not isinstance(row, dict):
            fail("POPULATION_MISMATCH", f"{source}:{number}: row is not an object")
        rows.append(row)
    return rows


def load_future_execution_data(contract: Mapping[str, Any]) -> FutureExecutionData:
    """Future-only structural loader. It applies the sole Fold0 filter before views exist."""
    names = frozen_feature_names(contract)
    actions_path = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl"
    raw_actions = read_actions_after_structural_fold_filter(actions_path)
    actions: list[TrainingAction] = []
    for raw in raw_actions:
        features = raw.get("features")
        if not isinstance(features, Mapping):
            features = {name: raw.get(name) for name in names}
        copied = dict(raw)
        copied["features"] = dict(features)
        action = training_view(copied)
        require_scientific_fold(action.fold, "loaded training action")
        actions.append(action)
    validate_unique_action_identities(actions)
    if len(actions) != EXPECTED_ACTION_COUNT:
        fail("POPULATION_MISMATCH", f"expected {EXPECTED_ACTION_COUNT} F1-F4 actions, got {len(actions)}")
    folds_doc = json.loads((ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json").read_text(encoding="utf-8"))
    if not isinstance(folds_doc, Mapping):
        fail("FOLD_SCHEMA_FAILURE", "folds document is not an object")
    validation_ids = {fold: stored_validation_order(folds_doc, fold) for fold in TARGET_FOLDS}
    if sum(len(values) for values in validation_ids.values()) != EXPECTED_QUERY_COUNT:
        fail("POPULATION_MISMATCH", "stored F1-F4 query count mismatch")
    baseline: dict[str, BaselineIdentity] = {}
    baseline_path = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
    with baseline_path.open("r", encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, start=1):
            fields = selected_top_level_values(line, {"query_id", "fold", "top5"})
            try:
                fold = int(fields["fold"])
            except (KeyError, TypeError, ValueError) as exc:
                fail("BASELINE_SCHEMA_FAILURE", f"{number}: {exc}")
            if fold == 0:  # The sole baseline structural exclusion; gold payload remains raw-skipped.
                continue
            require_scientific_fold(fold, "approved baseline structural boundary")
            identity = baseline_identity_from_raw_jsonl(line)
            if identity.query_id in baseline:
                fail("POPULATION_MISMATCH", f"duplicate baseline query {identity.query_id}")
            baseline[identity.query_id] = identity
    expected_queries = set().union(*[set(values) for values in validation_ids.values()])
    if set(baseline) != expected_queries:
        fail("POPULATION_MISMATCH", "baseline F1-F4 population mismatch")
    if any(action.query_id not in expected_queries for action in actions):
        fail("POPULATION_MISMATCH", "action query outside F1-F4 validation universe")
    return FutureExecutionData(tuple(actions), MappingProxyType(baseline), MappingProxyType(validation_ids))


def canonical_inner_label_loader(artifact: FrozenArtifact) -> Mapping[str, frozenset[str]]:
    """Concrete post-freeze loader for the canonical LegalIR mapping (inner only)."""
    artifact.require_frozen()
    query_ids: set[str] = set()
    for line in artifact.payload.splitlines():
        row = json.loads(line)
        if isinstance(row, Mapping) and isinstance(row.get("query_id"), str):
            query_ids.add(row["query_id"])
    train_path = ROOT / "data/raw/btc/LegalIR/train.json"
    source = json.loads(train_path.read_text(encoding="utf-8-sig"))
    if not isinstance(source, Mapping):
        fail("LABEL_SCHEMA_FAILURE", "LegalIR train root must be a mapping")
    labels: dict[str, frozenset[str]] = {}
    for query_id in query_ids:
        record = source.get(query_id)
        answer = record.get("answer") if isinstance(record, Mapping) else None
        if not isinstance(answer, list) or not answer or not all(isinstance(doc, str) for doc in answer):
            fail("LABEL_SCHEMA_FAILURE", f"missing canonical answer for {query_id}")
        labels[query_id] = frozenset(answer)
    if set(labels) != query_ids:
        fail("LABEL_SCHEMA_FAILURE", "post-freeze label coverage mismatch")
    return MappingProxyType(labels)


def canonical_inner_metric_evaluator(artifact: FrozenArtifact, labels: Mapping[str, frozenset[str]],
                                     _margin: float) -> InnerMetric:
    """Macro set Recall/Precision delta; this is callable only after artifact freeze."""
    require_frozen_for_label_loading(artifact, artifact.sha256)
    top5_path = artifact.path.parent / "final_top5.jsonl"
    if not top5_path.is_file():
        fail("ARTIFACT_SHA_PROVENANCE_FAILURE", "final top5 missing beside frozen selected actions")
    with top5_path.open("r", encoding="utf-8", newline="") as handle:
        predictions = [json.loads(line) for line in handle if line]
    baseline_path = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
    baselines: dict[str, tuple[str, ...]] = {}
    with baseline_path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            fields = selected_top_level_values(line, {"query_id", "fold", "top5"})
            query_id = fields.get("query_id")
            if query_id in labels:
                values = fields.get("top5")
                if not isinstance(values, list) or len(values) != 5 or not all(isinstance(doc, str) for doc in values):
                    fail("BASELINE_SCHEMA_FAILURE", "canonical baseline top5 malformed")
                baselines[query_id] = tuple(values)
    if set(baselines) != set(labels):
        fail("LABEL_SCHEMA_FAILURE", "baseline coverage mismatch after freeze")
    recall_deltas: list[float] = []
    precision_deltas: list[float] = []
    for row in predictions:
        query_id, top5 = row.get("query_id"), row.get("top5")
        if query_id not in labels or not isinstance(top5, list) or len(top5) != 5:
            fail("INNER_METRIC_FAILURE", "top5 label evaluation row malformed")
        gold = labels[query_id]
        baseline_set, prediction_set = set(baselines[query_id]), set(top5)
        recall_deltas.append(len(gold & prediction_set) / len(gold) - len(gold & baseline_set) / len(gold))
        precision_deltas.append(len(gold & prediction_set) / 5.0 - len(gold & baseline_set) / 5.0)
    if not recall_deltas:
        fail("INNER_METRIC_FAILURE", "empty evaluated query set")
    return InnerMetric(sum(recall_deltas) / len(recall_deltas), sum(precision_deltas) / len(precision_deltas))


def execute_authorized_session(authorization: ManualAuthorization, contract: Mapping[str, Any], provenance: Mapping[str, Any]) -> str:
    """Reachable separately-authorized entry point; C3 is intentionally not called."""
    record_authorization_consumption(authorization)
    data = load_future_execution_data(contract)
    return run_future_c2_execution(contract, provenance, data, canonical_inner_label_loader,
                                   canonical_inner_metric_evaluator, authorization.session_id)


def pure_helper_self_checks(np: Any) -> dict[str, bool]:
    features = {name: 0.0 for name in ("a",)}
    left = TrainingAction("q", 1, 4, "b", features, 0.0)
    right = TrainingAction("q", 1, 5, "a", features, 0.5)
    weights = query_balanced_weights([left, right])
    weight_ok = abs(sum(weights) - 1.0) <= 1e-12
    candidates = margin_candidates([0.1, 0.2, 0.3, -0.5], np)
    candidates_ok = candidates == (0.0, 0.25, 0.28, 0.29)
    duplicate_margin_ok = margin_candidates([0.4, 0.4, 0.4], np) == (0.0, 0.4)
    selected = choose_best_action([(left, 0.1), (right, 0.1)], 0.0)
    tie_ok = selected is right
    keep_ok = choose_best_action([(right, 0.0)], 0.0) is None
    top5_rank5_ok = one_swap_top5(["1", "2", "3", "4", "5"], right) == ["1", "2", "3", "4", "a"]
    rank4 = ScoringAction("q", 1, 4, "z", features)
    top5_rank4_ok = one_swap_top5(["1", "2", "3", "4", "5"], rank4) == ["1", "2", "3", "z", "5"]
    baseline_ok = baseline_identity_from_raw_jsonl(
        '{"query_id":"q","fold":1,"top5":["1","2","3","4","5"],"gold_documents":["not decoded"]}'
    ).top5 == ("1", "2", "3", "4", "5")
    duplicate_ok = False
    try:
        validate_unique_action_identities([right, right])
    except C2HarnessError as exc:
        duplicate_ok = str(exc).startswith("PROVENANCE_FAILURE")
    serialization_ok = (canonical_json_bytes({"b": 1, "a": [2]}) == b'{"a":[2],"b":1}\n'
                        and canonical_jsonl_bytes([{"b": 1}, {"a": 2}]) == b'{"b":1}\n{"a":2}\n'
                        and b"".join((b"f1\n", b"f2\n", b"f3\n", b"f4\n")) == b"f1\nf2\nf3\nf4\n")
    fold0_ok = False
    try:
        require_scientific_fold(0, "synthetic")
    except C2HarnessError as exc:
        fold0_ok = str(exc).startswith("FOLD0_SCIENTIFIC_LEAKAGE")
    fold0_predecode_ok = decode_action_lines_after_structural_filter(
        ['{"fold":0,"features":{"unterminated":,"label":"SECRET"}\n',
         '{"fold":1,"query_id":"q","drop_rank":4,"incoming_doc_id":"d","features":{}}\n']) == [
             {"fold": 1, "query_id": "q", "drop_rank": 4, "incoming_doc_id": "d", "features": {}}]
    malformed_f1_rejected = False
    try:
        decode_action_lines_after_structural_filter(['{"fold":1,"features":{"unterminated":}\n'])
    except C2HarnessError as exc:
        malformed_f1_rejected = str(exc).startswith("POPULATION_MISMATCH")
    forbidden_score_ok = False
    try:
        assert_score_view_label_free({"query_id": "q", "gold_documents": []})
    except C2HarnessError as exc:
        forbidden_score_ok = str(exc).startswith("VALIDATION_SCORING_LABEL_LEAKAGE")
    duplicate_top5_ok = False
    try:
        one_swap_top5(["1", "2", "3", "4", "5"], ScoringAction("q", 1, 4, "5", features))
    except C2HarnessError as exc:
        duplicate_top5_ok = str(exc).startswith("TOP5_CONTRACT_FAILURE")
    frozen = FrozenArtifact(Path("synthetic.jsonl"), hashlib.sha256(b"synthetic\n").hexdigest(), b"synthetic\n", "synthetic")
    unfrozen = FrozenArtifact(Path("synthetic.jsonl"), frozen.sha256, frozen.payload, "synthetic", "UNFROZEN")
    lifecycle_rejects_unfrozen = False
    try:
        evaluate_after_freeze(unfrozen, unfrozen.sha256, lambda _artifact: None,
                              lambda _artifact, _labels: InnerMetric(0.0, 0.0))
    except C2HarnessError as exc:
        lifecycle_rejects_unfrozen = str(exc).startswith("PREDICTION_FREEZE_BOUNDARY_FAILURE")
    lifecycle_accepts_frozen = evaluate_after_freeze(frozen, frozen.sha256, lambda _artifact: None,
                                                      lambda _artifact, _labels: InnerMetric(0.0, 0.0)) == InnerMetric(0.0, 0.0)
    mutation_rejected = False
    try:
        mutated = FrozenArtifact(frozen.path, frozen.sha256, b"mutated\n", "synthetic")
        evaluate_after_freeze(mutated, frozen.sha256, lambda _artifact: None,
                              lambda _artifact, _labels: InnerMetric(0.0, 0.0))
    except C2HarnessError:
        mutation_rejected = True
    gate = c2_inner_gate(1, {2: InnerMetric(0.0, 0.0), 3: InnerMetric(0.0, 0.0), 4: InnerMetric(0.0, 0.0)},
                         {2: InnerMetric(0.002, -0.001), 3: InnerMetric(0.002, -0.001), 4: InnerMetric(0.002, -0.001)}, True)
    gate_ok = gate.passed and c2_inner_gate(1, {2: InnerMetric(0.0, 0.0), 3: InnerMetric(0.0, 0.0), 4: InnerMetric(0.0, 0.0)},
                                            {2: InnerMetric(0.002, -0.001), 3: InnerMetric(-0.001, -0.001), 4: InnerMetric(0.002, -0.001)}, True).passed is False
    def synthetic_candidate_factory(margin: float, candidate_index: int) -> MarginCandidateArtifacts:
        return MarginCandidateArtifacts(1, margin, candidate_index,
                                        MappingProxyType({2: frozen.sha256, 3: frozen.sha256, 4: frozen.sha256}),
                                        MappingProxyType({2: frozen, 3: frozen, 4: frozen}),
                                        MappingProxyType({2: frozen, 3: frozen, 4: frozen}), "synthetic")
    inf_margin = select_c2r_margin({2: frozen, 3: frozen, 4: frozen}, {"q": 0.4}, synthetic_candidate_factory,
                                   lambda _artifact: None,
                                   lambda _artifact, _labels, _margin: InnerMetric(0.0, -0.01), np)
    infinity_ok = math.isinf(inf_margin.selected_margin) and not inf_margin.challenger_inner_gate_possible
    root_reject_ok = False
    try:
        execution_output_root(ROOT / "not_the_approved_execution_root")
    except C2HarnessError as exc:
        root_reject_ok = str(exc).startswith("OUTPUT_ROOT_REJECTED")
    budget_v = C2RunBudget.start()
    for _ in range(16):
        budget_v.reserve_fit("c2_v")
    c2v_budget_ok = False
    try:
        budget_v.reserve_fit("c2_v")
    except C2HarnessError as exc:
        c2v_budget_ok = str(exc).startswith("FIT_BUDGET_EXCEEDED")
    budget_r = C2RunBudget.start()
    for _ in range(16):
        budget_r.reserve_fit("c2_r")
    c2r_budget_ok = False
    try:
        budget_r.reserve_fit("c2_r")
    except C2HarnessError as exc:
        c2r_budget_ok = str(exc).startswith("FIT_BUDGET_EXCEEDED")
    budget_total = C2RunBudget.start()
    for _ in range(16):
        budget_total.reserve_fit("c2_v")
        budget_total.reserve_fit("c2_r")
    aggregate_budget_ok = False
    try:
        budget_total.reserve_fit("c2_v")
    except C2HarnessError as exc:
        aggregate_budget_ok = str(exc).startswith("FIT_BUDGET_EXCEEDED")
    deadline = FitDeadline.start(lambda: 100.0)
    absolute_deadline_ok = (deadline.started_monotonic == 100.0 and deadline.expires_monotonic == 100.0 + PER_FIT_TIMEOUT_SECONDS
                            and abs(deadline.remaining_seconds(lambda: 300.0) - 700.0) <= 1e-12)
    stderr_tail = BoundedDiagnosticTail(32)
    stderr_tail.append("x" * 128)
    stderr_tail_ok = len(stderr_tail.text().encode("utf-8")) == 32
    worker_capability_ok = True
    try:
        nonce = "n" * 48
        job_scope = "inner_o1_i2"
        job_id = "job-0001-abcdefghijkl"
        job_path = worker_job_path(job_scope, "c2_v", job_id)
        capability = json.loads(worker_capability_message(session_id="session-0001", parent_pid=os.getppid(),
                                                           worker_pid=os.getpid(), launch_nonce=nonce, job_id=job_id,
                                                           job_sha256="a" * 64, job_scope=job_scope, job_path=job_path, family="c2_v", family_ordinal=1,
                                                           aggregate_ordinal=1))
        expected_capability = {key: value for key, value in capability.items() if key != "kind"}
        validate_worker_capability(json.dumps(capability), expected_capability)
        for field, replacement in (("parent_pid", os.getppid() + 1), ("worker_pid", os.getpid() + 1), ("launch_nonce", "wrong"),
                                   ("job_id", "wrong"), ("job_path", "wrong"), ("job_sha256", "b" * 64), ("session_id", "wrong"),
                                   ("family", "c2_r"), ("family_ordinal", 2), ("aggregate_ordinal", 2)):
            bad = dict(capability)
            bad[field] = replacement
            try:
                validate_worker_capability(json.dumps(bad), expected_capability)
            except C2HarnessError:
                continue
            worker_capability_ok = False
    except C2HarnessError:
        worker_capability_ok = False
    job_path_ok = False
    try:
        with tempfile.TemporaryDirectory() as directory:
            synthetic_root = Path(directory)
            v_path = worker_job_path("inner_o1_i2", "c2_v", "job-v-0001-abcdefghijkl", synthetic_root)
            r_path = worker_job_path("inner_o1_i2", "c2_r", "job-r-0001-abcdefghijkl", synthetic_root)
            fixture_job = {"score_identities": []}
            write_worker_job(v_path, fixture_job, synthetic_root)
            write_worker_job(r_path, fixture_job, synthetic_root)
            duplicate_rejected = False
            try:
                write_worker_job(v_path, fixture_job, synthetic_root)
            except FileExistsError:
                duplicate_rejected = True
            traversal_rejected = False
            try:
                worker_job_path("../escape", "c2_v", "job-v-0001-abcdefghijkl", synthetic_root)
            except C2HarnessError:
                traversal_rejected = True
            job_path_ok = v_path != r_path and v_path.is_file() and r_path.is_file() and duplicate_rejected and traversal_rejected
    except (C2HarnessError, OSError):
        job_path_ok = False
    pid_identity = {"kind": "C2_WORKER_IDENTITY", "pid": 12344, "parent_pid": 99, "launch_nonce": "n" * 48, "python": "3.12.6", "numpy": "1.26.4",
                    "scikit_learn": "1.7.2", "threadpoolctl": "3.6.0", "environment": dict(REQUIRED_ENVIRONMENT)}
    pid_binding_ok = False
    try:
        validate_worker_identity(pid_identity, expected_pid=12345)
    except C2HarnessError as exc:
        pid_binding_ok = str(exc).startswith("WORKER_PID_MISMATCH")
    pid_identity["pid"] = 12345
    try:
        validate_worker_identity(pid_identity, expected_pid=12345)
        pid_binding_ok = pid_binding_ok
    except C2HarnessError:
        pid_binding_ok = False
    identity_ok = True
    try:
        validate_worker_identity({"kind": "C2_WORKER_IDENTITY", "pid": 1, "parent_pid": 99, "launch_nonce": "n" * 48, "python": "3.12.6", "numpy": "1.26.4",
                                  "scikit_learn": "1.7.2", "threadpoolctl": "3.6.0", "environment": dict(REQUIRED_ENVIRONMENT)})
        validate_worker_identity({"kind": "C2_WORKER_IDENTITY", "pid": 1, "parent_pid": 99, "launch_nonce": "n" * 48, "python": "bad", "numpy": "1.26.4",
                                  "scikit_learn": "1.7.2", "threadpoolctl": "3.6.0", "environment": dict(REQUIRED_ENVIRONMENT)})
        identity_ok = False
    except C2HarnessError:
        pass
    pinned_runtime = pinned_worker_identity_self_checks()
    terminal_runtime = terminal_journal_self_checks()
    deadline_runtime = deadline_race_self_checks()
    containment_runtime = containment_handle_lifecycle_self_checks()
    return {
        "query_weight": weight_ok,
        "margin_candidates": candidates_ok,
        "margin_duplicate_removal": duplicate_margin_ok,
        "margin_infinity_failure": infinity_ok,
        "action_tie_break": tie_ok,
        "strict_keep": keep_ok,
        "top5_rank4_swap": top5_rank4_ok,
        "top5_rank5_swap": top5_rank5_ok,
        "top5_duplicate_rejection": duplicate_top5_ok,
        "baseline_payload_isolation": baseline_ok,
        "duplicate_identity": duplicate_ok,
        "deterministic_serialization": serialization_ok,
        "fold0_rejection": fold0_ok,
        "fold0_predecode_exclusion": fold0_predecode_ok and malformed_f1_rejected,
        "score_view_forbidden_column_rejection": forbidden_score_ok,
        "prediction_freeze_lifecycle": lifecycle_rejects_unfrozen and lifecycle_accepts_frozen and mutation_rejected,
        "c2_inner_gate_thresholds": gate_ok,
        "canonical_output_root_rejection": root_reject_ok,
        "fit_budget_per_family": c2v_budget_ok and c2r_budget_ok,
        "fit_budget_32_33": aggregate_budget_ok,
        "absolute_fit_deadline_not_reset": absolute_deadline_ok and deadline_runtime["absolute_fit_deadline_not_reset"],
        "bounded_stderr_drain_tail": stderr_tail_ok,
        "synthetic_timeout_supervision": synthetic_timeout_supervision_self_checks(),
        **pinned_runtime,
        "contained_grandchild_rejected": contained_grandchild_rejection_self_check(),
        "descendant_cleanup": descendant_cleanup_self_check(),
        "taskkill_fallback_bounded": taskkill_fallback_self_check(),
        "launch_stage_terminal_journal": launch_stage_terminal_journal_self_check(),
        "integrated_launch_journal_io": integrated_launch_journal_io_self_check(),
        **terminal_runtime,
        **containment_runtime,
        **deadline_runtime,
        "manual_authorization_validation": manual_authorization_self_checks(),
        "durable_session_consumption": durable_session_consumption_self_checks(),
        "worker_capability": worker_capability_ok,
        "worker_job_path_uniqueness": job_path_ok,
        "worker_pid_binding": pid_binding_ok,
        "worker_identity_handshake": identity_ok,
        "resource_monitor_capability": resource_monitor_capability(),
    }


def review_only() -> dict[str, Any]:
    contract, provenance = load_frozen_contracts()
    np, sklearn, _threadpoolctl = require_environment_before_sklearn()
    verified_inputs = verify_manifest_inputs(provenance)
    split_map = outer_splits(contract)
    rotations = {str(fold): inner_rotations(training) for fold, training in split_map.items()}
    c2v = construct_c2v_estimator(contract, sklearn)
    c2r = construct_c2r_estimator(contract, sklearn)
    checks = pure_helper_self_checks(np)
    if not all(checks.values()):
        fail("C2_REVIEW_SELF_CHECK_FAILURE", str(checks))
    return {
        "status": "REVIEW_ONLY_PASS",
        "mode": "review-only",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "provenance_manifest_sha256": EXPECTED_PROVENANCE_SHA256,
        "verified_input_count": len(verified_inputs),
        "outer_splits": {str(key): list(value) for key, value in split_map.items()},
        "inner_rotation_count_per_outer": {key: len(value) for key, value in rotations.items()},
        "c2v_constructor": type(c2v).__name__,
        "c2r_constructor": type(c2r).__name__,
        "helper_checks": checks,
        "script_sha256": running_script_sha256(),
        "task1_data_parsed": False,
        "fit_called": False,
        "predict_called": False,
        "predict_proba_called": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--review-only", action="store_true", help="verify frozen identity and run constructor/helper checks only")
    mode.add_argument("--execute", action="store_true", help="run only with an explicit manual C2 authorization artifact")
    parser.add_argument("--authorization-artifact", metavar="PATH", help="manual governance artifact required with --execute")
    mode.add_argument("--fit-worker", metavar="JOB", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.authorization_artifact and not args.execute:
            fail("C2_AUTHORIZATION_INVALID", "--authorization-artifact is valid only with --execute")
        if args.fit_worker:
            return fit_worker(args.fit_worker)
        if args.execute:
            authorization = validate_manual_authorization(args.authorization_artifact)
            contract, provenance = load_frozen_contracts()
            status = execute_authorized_session(authorization, contract, provenance)
            print(json.dumps({"status": status}, ensure_ascii=True, sort_keys=True))
            return 0
        result = review_only()
    except C2HarnessError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
