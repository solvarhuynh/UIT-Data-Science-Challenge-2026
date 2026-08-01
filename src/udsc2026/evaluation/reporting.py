"""Stable JSON and Markdown rendering for evaluation reports."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Sequence

from udsc2026.evaluation.models import EvaluationComparison, EvaluationReport

Report = EvaluationReport | EvaluationComparison
ReportDestination = tuple[Report, str | Path, str | Path]


def _format_metric(value: float) -> str:
    """Format metrics consistently without locale-sensitive output."""

    return f"{value:.6f}"


def _render_single_report(report: EvaluationReport, heading_level: int = 1) -> str:
    """Render one report as a compact Markdown section."""

    prefix = "#" * heading_level
    lines = [
        f"{prefix} Evaluation: {report.name}",
        "",
        f"- Samples: {report.sample_count}",
        f"- Dataset SHA-256: `{report.dataset_fingerprint}`",
        f"- MRR: {_format_metric(report.retrieval.mrr)}",
    ]
    for cutoff in report.k_values:
        lines.append(
            f"- Recall@{cutoff}: {_format_metric(report.retrieval.recall_at_k[cutoff])}"
        )
    if report.qa is not None:
        lines.extend(
            [
                f"- ROUGE-L: {_format_metric(report.qa.rouge_l)}",
                f"- QA samples: {report.qa.sample_count}",
            ]
        )
    if report.latency is not None:
        lines.extend(
            [
                "",
                f"{prefix}# Latency (ms)",
                "",
                "| Mean | Median | P95 | P99 | Min | Max |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    _format_metric(value)
                    for value in (
                        report.latency.mean_ms,
                        report.latency.median_ms,
                        report.latency.p95_ms,
                        report.latency.p99_ms,
                        report.latency.min_ms,
                        report.latency.max_ms,
                    )
                )
                + " |",
            ]
        )
    return "\n".join(lines)


def render_report_markdown(report: Report) -> str:
    """Render an evaluation or before/after comparison to deterministic Markdown."""

    if isinstance(report, EvaluationReport):
        return _render_single_report(report) + "\n"

    delta_lines: List[str] = [
        "# Evaluation comparison",
        "",
        "| Metric | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
        "| MRR | "
        f"{_format_metric(report.before.retrieval.mrr)} | "
        f"{_format_metric(report.after.retrieval.mrr)} | "
        f"{_format_metric(report.delta.mrr)} |",
    ]
    for cutoff in report.before.k_values:
        delta_lines.append(
            f"| Recall@{cutoff} | "
            f"{_format_metric(report.before.retrieval.recall_at_k[cutoff])} | "
            f"{_format_metric(report.after.retrieval.recall_at_k[cutoff])} | "
            f"{_format_metric(report.delta.recall_at_k[cutoff])} |"
        )
    if (
        report.before.qa is not None
        and report.after.qa is not None
        and report.delta.rouge_l is not None
    ):
        delta_lines.append(
            "| ROUGE-L | "
            f"{_format_metric(report.before.qa.rouge_l)} | "
            f"{_format_metric(report.after.qa.rouge_l)} | "
            f"{_format_metric(report.delta.rouge_l)} |"
        )
    if (
        report.before.latency is not None
        and report.after.latency is not None
        and report.delta.mean_latency_ms is not None
        and report.delta.p95_latency_ms is not None
    ):
        delta_lines.extend(
            [
                "| Mean latency (ms) | "
                f"{_format_metric(report.before.latency.mean_ms)} | "
                f"{_format_metric(report.after.latency.mean_ms)} | "
                f"{_format_metric(report.delta.mean_latency_ms)} |",
                "| P95 latency (ms) | "
                f"{_format_metric(report.before.latency.p95_ms)} | "
                f"{_format_metric(report.after.latency.p95_ms)} | "
                f"{_format_metric(report.delta.p95_latency_ms)} |",
            ]
        )
    delta_lines.extend(
        [
            "",
            _render_single_report(report.before, heading_level=2),
            "",
            _render_single_report(report.after, heading_level=2),
            "",
        ]
    )
    return "\n".join(delta_lines)


def _stage_text(path: Path, content: str, *, encoding: str) -> Path:
    """Write complete content beside its destination without replacing it yet."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        stream = os.fdopen(descriptor, "w", encoding=encoding, newline="\n")
        descriptor = -1
        with stream:
            stream.write(content)
        return temporary_path
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _best_effort_unlink(temporary_path)
        raise


def _backup_target(path: Path) -> Path:
    """Copy an existing regular file beside itself for transactional rollback."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".backup", dir=str(path.parent)
    )
    backup_path = Path(temporary_name)
    try:
        destination = os.fdopen(descriptor, "wb")
        descriptor = -1
        with destination:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination)
        shutil.copystat(path, backup_path)
        return backup_path
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _best_effort_unlink(backup_path)
        raise


def _best_effort_unlink(path: Path) -> None:
    """Remove a staging artifact without masking the primary operation result."""

    try:
        path.unlink()
    except OSError:
        pass


def _render_report_files(
    destinations: Sequence[ReportDestination],
) -> list[tuple[Path, str]]:
    """Render every report before any destination is touched."""

    if not destinations:
        raise ValueError("at least one report destination is required")

    rendered: list[tuple[Path, str]] = []
    for report, json_path, markdown_path in destinations:
        if not isinstance(report, (EvaluationReport, EvaluationComparison)):
            raise TypeError("report must be EvaluationReport or EvaluationComparison")
        json_content = (
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        rendered.extend(
            [
                (Path(json_path), json_content),
                (Path(markdown_path), render_report_markdown(report)),
            ]
        )
    return rendered


def _prepare_output_targets(paths: Sequence[Path]) -> None:
    """Validate all targets and create their parent directories before staging."""

    resolved_paths: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in resolved_paths:
            raise ValueError("report output paths must be different")
        resolved_paths.add(resolved)
        if path.is_symlink():
            raise ValueError(
                f"report output target must not be a symbolic link: {path}"
            )
        if path.is_dir():
            raise ValueError(f"report output target must not be a directory: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"report output target must be a regular file: {path}")

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Creating one parent can reveal an ancestor collision with another target.
    for path in paths:
        if path.is_dir():
            raise ValueError(f"report output target must not be a directory: {path}")


def _restore_attempted_targets(
    attempted: Sequence[Path],
    backups: dict[Path, Path | None],
) -> list[tuple[Path, BaseException]]:
    """Best-effort rollback of every destination whose replace was attempted."""

    errors: list[tuple[Path, BaseException]] = []
    for target in reversed(attempted):
        backup = backups[target]
        try:
            if backup is None:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            else:
                os.replace(backup, target)
        except BaseException as exc:
            errors.append((target, exc))
    return errors


def write_report_bundle(destinations: Sequence[ReportDestination]) -> None:
    """Transactionally replace one or more JSON/Markdown report pairs.

    Every file is rendered and staged before commit. If any replacement fails,
    all destinations already attempted are restored to their exact prior bytes.
    """

    destination_list = list(destinations)
    rendered = _render_report_files(destination_list)
    targets = [path for path, _ in rendered]
    _prepare_output_targets(targets)

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    attempted: list[Path] = []
    preserve_backups = False
    try:
        for target, content in rendered:
            staged[target] = _stage_text(target, content, encoding="utf-8")
        for target in targets:
            backups[target] = _backup_target(target) if target.exists() else None

        try:
            for target in targets:
                attempted.append(target)
                os.replace(staged[target], target)
        except BaseException as commit_error:
            rollback_errors = _restore_attempted_targets(attempted, backups)
            if rollback_errors:
                preserve_backups = True
                details = "; ".join(
                    f"{target}: {type(error).__name__}: {error}"
                    for target, error in rollback_errors
                )
                raise RuntimeError(
                    "report bundle commit failed and rollback was incomplete; "
                    f"backup files were retained ({details})"
                ) from commit_error
            raise
    finally:
        for temporary_path in staged.values():
            _best_effort_unlink(temporary_path)
        if not preserve_backups:
            for backup_path in backups.values():
                if backup_path is not None:
                    _best_effort_unlink(backup_path)


def write_report(
    report: Report,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Transactionally write matching JSON and Markdown for one report."""

    write_report_bundle([(report, json_path, markdown_path)])
