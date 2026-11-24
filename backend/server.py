"""Small local submission grader.

This server is intentionally independent from the UDSC application package. It
only uses the Python standard library at runtime. Optional NLTK and
rouge-score installations improve LegalQA scorer parity when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import sqlite3
import sys
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = ROOT.parent / "frontend"
DATA_ROOT = ROOT / "data"
REFERENCE_ROOT = Path(
    os.environ.get("UDSC_GRADER_REFERENCE_DIR", str(DATA_ROOT / "references"))
).expanduser().resolve()
UPLOAD_ROOT = DATA_ROOT / "uploads"
DB_PATH = DATA_ROOT / "grader.sqlite3"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

STAGES = ("warmup", "public", "private", "holdout")
STAGE_LABELS = {
    "warmup": "Warm-up",
    "public": "Public test",
    "private": "Private test",
    "holdout": "Local holdout",
}
DEFAULT_STAGE = os.environ.get("UDSC_GRADER_STAGE", "warmup").strip().casefold()
if DEFAULT_STAGE not in STAGES:
    DEFAULT_STAGE = "warmup"
TASK_LABELS = {"ir": "LegalIR", "qa": "LegalQA"}
_DB_LOCK = threading.Lock()
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


class GraderError(ValueError):
    """Expected user-facing validation error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_task_stage(task: str, stage: str) -> None:
    if task not in TASK_LABELS:
        raise GraderError("task phải là ir hoặc qa")
    if stage not in STAGES:
        raise GraderError(f"stage không hợp lệ: {stage}")


def _validate_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GraderError(f"{label} phải là chuỗi không rỗng")
    if any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in value):
        raise GraderError(f"{label} chứa ký tự điều khiển")
    return value


def _read_json_bytes(payload: bytes, label: str) -> object:
    try:
        return json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraderError(f"{label} không phải JSON UTF-8 hợp lệ: {exc}") from exc


def _load_submission_payload(filename: str, payload: bytes) -> object:
    suffix = Path(filename).suffix.casefold()
    if suffix == ".json":
        return _read_json_bytes(payload, "submission")
    if suffix != ".zip":
        raise GraderError("chỉ nhận file .json hoặc .zip")
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1 or members[0].filename != "submission.json":
                raise GraderError("ZIP phải chứa đúng một file tên submission.json")
            info = members[0]
            if info.flag_bits & 0x1:
                raise GraderError("ZIP mã hóa không được hỗ trợ")
            if info.file_size > MAX_UPLOAD_BYTES:
                raise GraderError("submission.json trong ZIP quá lớn")
            return _read_json_bytes(archive.read(info), "submission.json")
    except zipfile.BadZipFile as exc:
        raise GraderError(f"ZIP không hợp lệ: {exc}") from exc


def _iter_object_entries(payload: object) -> Iterable[tuple[str, object]]:
    if isinstance(payload, dict):
        return payload.items()
    if isinstance(payload, list):
        entries: list[tuple[str, object]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise GraderError(f"submission[{index}] phải là object")
            question_id = item.get("id")
            if question_id is None:
                question_id = item.get("question_id")
            if question_id is None:
                raise GraderError(f"submission[{index}] thiếu id")
            entries.append((str(question_id), item))
        return entries
    raise GraderError("submission phải là object theo question ID hoặc một array record")


def normalize_submission(task: str, payload: object) -> dict[str, Any]:
    """Normalize official object JSON and the repo's internal array JSON."""

    result: dict[str, Any] = {}
    for raw_id, record in _iter_object_entries(payload):
        question_id = _validate_id(raw_id, "question ID")
        if question_id in result:
            raise GraderError(f"trùng question ID: {question_id}")
        if not isinstance(record, dict):
            raise GraderError(f"record của {question_id} phải là object")
        answer = record.get("answer")
        if answer is None and task == "ir":
            answer = record.get("documents")
        if task == "ir":
            if not isinstance(answer, list):
                raise GraderError(f"IR answer của {question_id} phải là array")
            docs = [_validate_id(value, f"IR answer của {question_id}") for value in answer]
            if len(docs) > 5:
                raise GraderError(f"IR answer của {question_id} vượt quá 5 document_id")
            if len(docs) != len(set(docs)):
                raise GraderError(f"IR answer của {question_id} có document_id trùng")
            result[question_id] = docs
        else:
            if not isinstance(answer, str):
                raise GraderError(f"QA answer của {question_id} phải là chuỗi")
            result[question_id] = answer
    if not result:
        raise GraderError("submission không có bản ghi nào")
    return result


def _reference_path(task: str, stage: str) -> Path:
    return REFERENCE_ROOT / task / f"{stage}.json"


def load_reference(task: str, stage: str) -> tuple[dict[str, Any] | None, str | None]:
    path = _reference_path(task, stage)
    if not path.is_file():
        return None, "Chưa có reference độc lập cho stage này."
    try:
        payload = _read_json_bytes(path.read_bytes(), f"reference {path.name}")
        reference = normalize_submission(task, payload)
    except (OSError, GraderError) as exc:
        return None, f"Reference không hợp lệ: {exc}"
    return reference, None


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        short, long = left, right
    else:
        short, long = right, left
    previous = [0] * (len(short) + 1)
    for token in long:
        current = [0]
        for index, short_token in enumerate(short, start=1):
            if token == short_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _fallback_rouge_l(prediction: str, reference: str) -> float:
    predicted = _tokenize(prediction)
    expected = _tokenize(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    lcs = _lcs_length(predicted, expected)
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def _fallback_meteor(prediction: str, reference: str) -> float:
    """Approximate METEOR used only when NLTK WordNet data is unavailable."""

    predicted = _tokenize(prediction)
    expected = _tokenize(reference)
    if not predicted or not expected:
        return 0.0
    remaining = list(expected)
    matches = 0
    positions: list[int] = []
    for token in predicted:
        try:
            position = remaining.index(token)
        except ValueError:
            continue
        matches += 1
        positions.append(position)
        remaining[position] = "\0"
    if not matches:
        return 0.0
    precision = matches / len(predicted)
    recall = matches / len(expected)
    harmonic = 10 * precision * recall / (recall + 9 * precision)
    chunks = 1
    for previous, current in zip(positions, positions[1:]):
        if current != previous + 1:
            chunks += 1
    return harmonic * (1 - 0.5 * (chunks / matches) ** 3)


def _qa_scorers() -> tuple[Any, Any, str]:
    meteor = None
    rouge = None
    try:
        from nltk.translate.meteor_score import meteor_score

        import nltk

        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/omw-1.4")
        meteor = meteor_score
    except Exception:
        meteor = None
    try:
        from rouge_score import rouge_scorer

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    except Exception:
        rouge = None
    if meteor is not None and rouge is not None:
        return meteor, rouge, "official-compatible"
    if meteor is not None:
        return meteor, None, "meteor-official-rouge-fallback"
    if rouge is not None:
        return None, rouge, "meteor-fallback-rouge-official"
    return None, None, "approximate-fallback"


def score_ir(prediction: dict[str, Any], reference: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    recalls: list[float] = []
    precisions: list[float] = []
    missing: list[str] = []
    extra: list[str] = []
    for question_id, gold in reference.items():
        predicted = prediction.get(question_id)
        if predicted is None:
            missing.append(question_id)
            predicted = []
        if not 0 < len(predicted) <= 5:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        relevant = set(gold)
        found = len(relevant.intersection(predicted))
        recalls.append(found / len(relevant) if relevant else 0.0)
        precisions.append(found / len(predicted))
    extra = [question_id for question_id in prediction if question_id not in reference]
    metrics = {
        "primary": sum(recalls) / len(recalls),
        "recall": sum(recalls) / len(recalls),
        "precision": sum(precisions) / len(precisions),
        "evaluated_queries": len(reference),
        "missing_queries": len(missing),
        "extra_queries": len(extra),
        "scorer": "BTC-LegalIR-recall-precision-v1",
    }
    details = {
        "missing_query_ids": missing[:50],
        "extra_query_ids": extra[:50],
        "coverage": len(reference) - len(missing),
    }
    return metrics, details


def score_qa(prediction: dict[str, Any], reference: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    meteor_fn, rouge_fn, scorer_name = _qa_scorers()
    meteors: list[float] = []
    rouges: list[float] = []
    missing: list[str] = []
    for question_id, expected in reference.items():
        actual = prediction.get(question_id)
        if actual is None:
            missing.append(question_id)
            actual = ""
        if meteor_fn is not None:
            try:
                meteor_value = float(meteor_fn([_tokenize(expected)], _tokenize(actual)))
            except Exception:
                meteor_value = _fallback_meteor(actual, expected)
        else:
            meteor_value = _fallback_meteor(actual, expected)
        if rouge_fn is not None:
            try:
                rouge_value = float(rouge_fn.score(expected, actual)["rougeL"].fmeasure)
            except Exception:
                rouge_value = _fallback_rouge_l(actual, expected)
        else:
            rouge_value = _fallback_rouge_l(actual, expected)
        meteors.append(meteor_value)
        rouges.append(rouge_value)
    extra = [question_id for question_id in prediction if question_id not in reference]
    metrics = {
        "primary": sum(meteors) / len(meteors),
        "meteor": sum(meteors) / len(meteors),
        "rouge_l": sum(rouges) / len(rouges),
        "evaluated_queries": len(reference),
        "missing_queries": len(missing),
        "extra_queries": len(extra),
        "scorer": scorer_name,
    }
    details = {
        "missing_query_ids": missing[:50],
        "extra_query_ids": extra[:50],
        "coverage": len(reference) - len(missing),
        "warning": (
            None
            if scorer_name == "official-compatible"
            else "Thiếu một hoặc nhiều gói scorer/corpus NLTK; kết quả có thể là xấp xỉ."
        ),
    }
    return metrics, details


def connect_db() -> sqlite3.Connection:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            stage TEXT NOT NULL,
            label TEXT NOT NULL,
            filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            metrics_json TEXT,
            details_json TEXT
        )
        """
    )
    connection.commit()
    return connection


def _row_to_submission(row: sqlite3.Row) -> dict[str, Any]:
    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else None
    details = json.loads(row["details_json"]) if row["details_json"] else None
    primary = metrics.get("primary") if metrics else None
    secondary = None
    if metrics:
        secondary = metrics.get("precision", metrics.get("rouge_l"))
    return {
        "id": row["id"],
        "task": row["task"],
        "stage": row["stage"],
        "stage_label": STAGE_LABELS.get(row["stage"], row["stage"]),
        "label": row["label"],
        "filename": row["filename"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "status": row["status"],
        "uploaded_at": row["uploaded_at"],
        "metrics": metrics,
        "primary": primary,
        "secondary": secondary,
        "details": details,
    }


def list_submissions(task: str, stage: str) -> list[dict[str, Any]]:
    _rescore_pending(task, stage)
    with _DB_LOCK:
        connection = connect_db()
        rows = connection.execute(
            "SELECT * FROM submissions WHERE task = ? AND stage = ? ORDER BY uploaded_at DESC",
            (task, stage),
        ).fetchall()
        connection.close()
    values = [_row_to_submission(row) for row in rows]
    values.sort(
        key=lambda item: (
            item["primary"] is None,
            -(item["primary"] or 0.0),
            -(item["secondary"] or 0.0),
            item["uploaded_at"],
        )
    )
    for index, item in enumerate(values, start=1):
        item["rank"] = index
    return values


def _rescore_pending(task: str, stage: str) -> None:
    """Revisit uploads saved before a reference file was installed."""

    reference, _ = load_reference(task, stage)
    if reference is None:
        return
    with _DB_LOCK:
        connection = connect_db()
        rows = connection.execute(
            "SELECT * FROM submissions WHERE task = ? AND stage = ? AND status = ?",
            (task, stage, "awaiting_reference"),
        ).fetchall()
        for row in rows:
            stored_path = UPLOAD_ROOT / row["stored_filename"]
            try:
                payload = stored_path.read_bytes()
                parsed = normalize_submission(task, _load_submission_payload(row["filename"], payload))
                if task == "ir":
                    metrics, score_details = score_ir(parsed, reference)
                else:
                    metrics, score_details = score_qa(parsed, reference)
                details = json.loads(row["details_json"] or "{}")
                details.update(score_details)
                details["reference_error"] = None
                connection.execute(
                    "UPDATE submissions SET status = ?, metrics_json = ?, details_json = ? WHERE id = ?",
                    (
                        "scored",
                        json.dumps(metrics, ensure_ascii=False),
                        json.dumps(details, ensure_ascii=False),
                        row["id"],
                    ),
                )
            except (OSError, GraderError, json.JSONDecodeError):
                # Keep the original pending record visible; the next refresh can
                # retry after the user fixes the reference or file.
                continue
        connection.commit()
        connection.close()


def save_submission(
    *, task: str, stage: str, label: str, filename: str, payload: bytes
) -> dict[str, Any]:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise GraderError("file vượt quá giới hạn 100 MB")
    label = label.strip() or Path(filename).stem or "submission"
    label = label[:120]
    parsed = normalize_submission(task, _load_submission_payload(filename, payload))
    reference, reference_error = load_reference(task, stage)
    status = "awaiting_reference"
    metrics: dict[str, Any] | None = None
    details: dict[str, Any] = {
        "submission_queries": len(parsed),
        "reference_error": reference_error,
    }
    if reference is not None:
        if task == "ir":
            metrics, score_details = score_ir(parsed, reference)
        else:
            metrics, score_details = score_qa(parsed, reference)
        details.update(score_details)
        status = "scored"
    submission_id = uuid.uuid4().hex[:12]
    digest = sha256_bytes(payload)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)[:120] or "submission.bin"
    stored_filename = f"{submission_id}_{safe_name}"
    (UPLOAD_ROOT / stored_filename).write_bytes(payload)
    with _DB_LOCK:
        connection = connect_db()
        connection.execute(
            """
            INSERT INTO submissions
            (id, task, stage, label, filename, stored_filename, size_bytes, sha256,
             status, uploaded_at, metrics_json, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission_id,
                task,
                stage,
                label,
                Path(filename).name,
                stored_filename,
                len(payload),
                digest,
                status,
                utc_now(),
                json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
                json.dumps(details, ensure_ascii=False),
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
        connection.close()
    if row is None:
        raise RuntimeError("không đọc được bản ghi vừa lưu")
    return _row_to_submission(row)


def delete_submission(submission_id: str) -> None:
    with _DB_LOCK:
        connection = connect_db()
        row = connection.execute(
            "SELECT stored_filename FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
        if row is None:
            connection.close()
            raise GraderError("không tìm thấy submission")
        connection.execute("DELETE FROM submissions WHERE id = ?", (submission_id,))
        connection.commit()
        connection.close()
    try:
        (UPLOAD_ROOT / row["stored_filename"]).unlink()
    except FileNotFoundError:
        pass


def config_payload() -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    for task in TASK_LABELS:
        stages[task] = {}
        for stage in STAGES:
            path = _reference_path(task, stage)
            reference, error = load_reference(task, stage)
            stages[task][stage] = {
                "label": STAGE_LABELS[stage],
                "reference_available": reference is not None,
                "reference_count": len(reference) if reference is not None else 0,
                "reference_error": error,
                "path_hint": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None,
            }
    return {
        "tasks": TASK_LABELS,
        "stages": stages,
        "default_stage": DEFAULT_STAGE,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "reference_root": str(REFERENCE_ROOT),
    }


def _parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str, bytes]]:
    match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type, re.IGNORECASE)
    if not match:
        raise GraderError("multipart request thiếu boundary")
    boundary = (match.group(1) or match.group(2)).encode("utf-8")
    result: dict[str, tuple[str, bytes]] = {}
    for raw_part in body.split(b"--" + boundary):
        part = raw_part.strip(b"\r\n-")
        if not part or b"\r\n\r\n" not in part:
            continue
        header_bytes, value = part.split(b"\r\n\r\n", 1)
        headers = header_bytes.decode("utf-8", errors="replace")
        disposition = next(
            (line for line in headers.split("\r\n") if line.lower().startswith("content-disposition:")),
            "",
        )
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        filename = filename_match.group(1) if filename_match else ""
        result[name_match.group(1)] = (filename, value.rstrip(b"\r\n"))
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "UDSCLocalGrader/1.0"

    def log_message(self, format: str, *args: object) -> None:
        sys.stdout.write(f"[{utc_now()}] {format % args}\n")

    def _send(self, status: int, payload: object, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"ok": False, "error": message})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._send(HTTPStatus.OK, {"ok": True, "service": "local-grader"})
                return
            if parsed.path == "/api/config":
                self._send(HTTPStatus.OK, {"ok": True, **config_payload()})
                return
            if parsed.path == "/api/submissions":
                query = parse_qs(parsed.query)
                task = query.get("task", ["ir"])[0]
                stage = query.get("stage", [DEFAULT_STAGE])[0]
                _validate_task_stage(task, stage)
                self._send(
                    HTTPStatus.OK,
                    {"ok": True, "submissions": list_submissions(task, stage)},
                )
                return
            if parsed.path.startswith("/api/submissions/"):
                submission_id = unquote(parsed.path.rsplit("/", 1)[-1])
                with _DB_LOCK:
                    connection = connect_db()
                    row = connection.execute(
                        "SELECT * FROM submissions WHERE id = ?", (submission_id,)
                    ).fetchone()
                    connection.close()
                if row is None:
                    self._error(HTTPStatus.NOT_FOUND, "không tìm thấy submission")
                else:
                    self._send(HTTPStatus.OK, {"ok": True, "submission": _row_to_submission(row)})
                return
            self._serve_static(parsed.path)
        except GraderError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - last-resort local server guard
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"lỗi máy chủ: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/submissions":
            self._error(HTTPStatus.NOT_FOUND, "endpoint không tồn tại")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise GraderError("request rỗng hoặc vượt giới hạn 100 MB")
            body = self.rfile.read(content_length)
            fields = _parse_multipart(self.headers.get("Content-Type", ""), body)
            task = fields.get("task", ("", b""))[1].decode("utf-8").strip()
            stage = fields.get("stage", ("", b""))[1].decode("utf-8").strip() or DEFAULT_STAGE
            label = fields.get("label", ("", b""))[1].decode("utf-8", errors="replace")
            filename, payload = fields.get("file", ("", b""))
            if not filename or not payload:
                raise GraderError("cần chọn một file submission")
            _validate_task_stage(task, stage)
            result = save_submission(
                task=task,
                stage=stage,
                label=label,
                filename=filename,
                payload=payload,
            )
            self._send(HTTPStatus.CREATED, {"ok": True, "submission": result})
        except GraderError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - last-resort local server guard
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"lỗi máy chủ: {exc}")

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/submissions/"):
            self._error(HTTPStatus.NOT_FOUND, "endpoint không tồn tại")
            return
        try:
            delete_submission(unquote(parsed.path.rsplit("/", 1)[-1]))
            self._send(HTTPStatus.OK, {"ok": True})
        except GraderError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))

    def _serve_static(self, request_path: str) -> None:
        relative = unquote(request_path.lstrip("/")) or "index.html"
        candidate = (FRONTEND_ROOT / relative).resolve()
        if not candidate.is_file() or FRONTEND_ROOT.resolve() not in candidate.parents:
            candidate = FRONTEND_ROOT / "index.html"
        try:
            body = candidate.read_bytes()
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "không tìm thấy tài nguyên")
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, body, content_type)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated local UDSC submission grader")
    parser.add_argument("--host", default=os.environ.get("UDSC_GRADER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("UDSC_GRADER_PORT", "8765")))
    args = parser.parse_args()
    if not FRONTEND_ROOT.is_dir():
        raise SystemExit(f"Không tìm thấy frontend/: {FRONTEND_ROOT}")
    connect_db().close()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local grader: http://{args.host}:{args.port}", flush=True)
    print(f"Reference root: {REFERENCE_ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng local grader.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
