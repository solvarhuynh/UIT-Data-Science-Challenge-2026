"""Run Beam CLI with POSIX remote paths from native Windows.

Beam/beta9 0.2.207 builds remote volume paths with ``os.path.join``. On
Windows this sends backslashes to Beam's Linux file service and breaks
multipart downloads. This narrow launcher patches only ``RemotePath`` joining
for its own process, then delegates to the official Beam CLI.
"""

from __future__ import annotations

import posixpath
import re
from importlib import metadata
from typing import Any

_URL_QUERY_RE = re.compile(r"(https?://[^\s?'\"<>]+)\?[^\s'\"<>]+", re.IGNORECASE)
_SIGNED_FIELD_RE = re.compile(
    r"(?i)(X-Amz-(?:Signature|Credential|Security-Token)|Signature|token)="
    r"([^&\s,'\")]+)"
)
SUPPORTED_VERSIONS = {"beam-client": "0.2.207", "beta9": "0.1.265"}


def _clean_volume_path(value: Any) -> str:
    raw = str(value).replace("\\", "/").lstrip("/")
    normalized = posixpath.normpath(raw) if raw else ""
    if normalized in {"", "."}:
        return ""
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("remote volume path cannot escape its volume")
    return normalized


def _posix_remote_path(remote: Any) -> str:
    volume_path = _clean_volume_path(remote.volume_path)
    if not volume_path:
        return f"{remote.volume_name}/"
    return posixpath.join(str(remote.volume_name), volume_path)


def _posix_remote_divide(remote: Any, other: Any) -> Any:
    path = other if isinstance(other, str) else other.volume_path
    joined = _clean_volume_path(
        posixpath.join(
            _clean_volume_path(remote.volume_path),
            str(path).replace("\\", "/"),
        )
    )
    return type(remote)(
        remote.scheme,
        remote.volume_name,
        joined,
        other.is_dir if not isinstance(other, str) else remote.is_dir,
    )


def _redact_download_error(message: str) -> str:
    sanitized = _URL_QUERY_RE.sub(r"\1?<redacted>", message)
    return _SIGNED_FIELD_RE.sub(r"\1=<redacted>", sanitized)


def _require_supported_versions() -> None:
    observed = {name: metadata.version(name) for name in SUPPORTED_VERSIONS}
    if observed != SUPPORTED_VERSIONS:
        raise RuntimeError(
            f"unsupported Beam SDK versions: expected {SUPPORTED_VERSIONS}, "
            f"got {observed}"
        )


def main() -> None:
    import beta9.multipart as multipart
    from beam.cli.main import cli

    _require_supported_versions()
    original_download = multipart.beta9_download
    original_upload = multipart.beta9_upload

    def sanitized_download(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_download(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(_redact_download_error(str(exc))) from None

    def sanitized_upload(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_upload(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(_redact_download_error(str(exc))) from None

    multipart.RemotePath.path = property(_posix_remote_path)
    multipart.RemotePath.__truediv__ = _posix_remote_divide
    multipart.beta9_download = sanitized_download
    multipart.beta9_upload = sanitized_upload
    cli()


if __name__ == "__main__":
    main()
