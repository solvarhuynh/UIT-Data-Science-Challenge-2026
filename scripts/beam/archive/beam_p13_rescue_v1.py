from beam import function, Image, Volume

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"

BASE_OUT = (
    RUNTIME
    / "artifacts/task1/models/bge_reranker_finetune/full_oof"
)

RESCUE_OUT = (
    RUNTIME
    / "artifacts/task1/models/bge_reranker_rescue_v1/"
      "fold0_pairwise_sym_bal4_lr2e6"
)

RUNTIME_TRAINER = (
    RUNTIME
    / "scripts/training/finetune_task1_bge_reranker.py"
)

PATCHED_TRAINER = Path("/workspace/p13/runtime/scripts/training/p13_rescue_trainer.py")

image = Image(
    python_version="python3.11",
    python_packages=[
        "torch",
        "transformers==5.0.0",
        "sentence-transformers==5.4.1",
        "accelerate>=1.1,<2",
        "huggingface-hub>=1.3,<2",
        "PyYAML>=6",
        "tqdm>=4.66,<5",
    ],
)


def _run_short(cmd, *, cwd=None):
    cmd = [str(x) for x in cmd]
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=True,
    )


def _stage_marker_path(stage: str) -> Path:
    return RESCUE_OUT / "stage_markers" / f"{stage}.json"


def _read_stage_marker(stage: str) -> dict | None:
    path = _stage_marker_path(stage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("status") == "done" else None


def _write_stage_marker(stage: str, **paths) -> None:
    """Publish a rescue stage marker atomically after its work is complete."""

    target = _stage_marker_path(stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = (
        json.dumps(
            {
                "status": "done",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "paths": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
        ) + "\n"
    )

    # Beam volume can transiently lock the atomic marker temp file.
    # Retry instead of killing a long rescue job after successful work.
    last_error = None
    for attempt in range(10):
        try:
            temporary.write_text(
                payload,
                encoding="utf-8",
            )
            os.replace(temporary, target)
            return
        except BlockingIOError as exc:
            last_error = exc
            time.sleep(min(5, attempt + 1))

    if last_error is not None:
        raise last_error


def _stage_done(stage: str, *, required_paths=()) -> bool:
    marker = _read_stage_marker(stage)
    return bool(marker and all(Path(path).exists() for path in required_paths))


def _rescue_result_from_decision(decision: dict) -> dict:
    return {
        "status": decision["status"],
        "baseline_recall": decision["rescue_baseline"]["official_macro_recall"],
        "rescue_recall": decision["rescue_tuned"]["official_macro_recall"],
        "delta": decision["recall_delta"],
        "output_dir": str(RESCUE_OUT),
    }


def _run_long(cmd, *, cwd, log_path):
    cmd = [str(x) for x in cmd]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    local_log = Path("/tmp") / (log_path.name + ".running")
    tail_path = log_path.with_suffix(log_path.suffix + ".tail")

    print("\n============================================================", flush=True)
    print("RESCUE RUN:", flush=True)
    print(" ".join(cmd), flush=True)
    print("Persistent log:", log_path, flush=True)
    print("============================================================", flush=True)

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TQDM_MININTERVAL": "60",
        "TQDM_MAXINTERVAL": "120",
    }

    with local_log.open("wb") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

        heartbeat = 0
        last = time.monotonic()
        while proc.poll() is None:
            time.sleep(5)
            now = time.monotonic()
            if now - last < 60:
                continue
            last = now
            heartbeat += 1
            f.flush()

            try:
                raw = local_log.read_bytes()[-65536:]
                text = raw.decode("utf-8", errors="replace").replace("\r", "\n")
                lines = [x.strip() for x in text.splitlines() if x.strip()]
                last_line = lines[-1] if lines else "(no output yet)"
                print(f"[HEARTBEAT {heartbeat}] {last_line}", flush=True)
                tail_path.write_text(
                    "\n".join(lines[-200:]) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                print("Heartbeat warning:", repr(exc), flush=True)

        rc = proc.wait()

    shutil.copy2(local_log, log_path)
    print(f"Trainer return code: {rc}", flush=True)

    tail = local_log.read_bytes()[-120000:].decode(
        "utf-8", errors="replace"
    ).replace("\r", "\n")
    print("\n--- FINAL TRAINER TAIL ---", flush=True)
    print("\n".join(tail.splitlines()[-120:]), flush=True)
    print("--- END TRAINER TAIL ---\n", flush=True)

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _ensure_runtime_bundle() -> None:
    """Ensure the persistent Beam runtime contains the P13 code/data bundle.

    Rescue jobs may start on a volume that still has the uploaded archives but
    not the extracted runtime tree.  Mirror beam_p13_run_all.py: extract only
    when required runtime inputs are actually missing, then continue normally.
    Existing artifacts/checkpoints are not deleted.
    """

    required_inputs = (
        RUNTIME / "data/raw/btc/LegalIR/train.json",
        RUNTIME / "artifacts/task1/evaluation/strict_cv_v2/folds.json",
        RUNTIME / "artifacts/task1/candidates/train7000_document_candidates.jsonl",
        RUNTIME / "artifacts/task1/training/negatives/fold_0.jsonl",
    )
    missing = [path for path in required_inputs if not path.exists()]
    if not missing:
        print("[RUNTIME] required P13 inputs already extracted", flush=True)
        return

    print("[RUNTIME] missing extracted inputs:", flush=True)
    for path in missing:
        print(f"  - {path}", flush=True)

    # These are the same persistent bundles used by beam_p13_run_all.py.
    archives = (
        VOLUME_ROOT / "udsc_p13_code.zip",
        VOLUME_ROOT / "udsc_p13_data.zip",
    )
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(
                f"Runtime input is missing and required archive is unavailable: {archive}"
            )

    RUNTIME.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f"[RUNTIME] extracting {archive.name} -> {RUNTIME}", flush=True)
        _run_short([
            "/usr/bin/python3.11",
            "-m",
            "zipfile",
            "-e",
            archive,
            RUNTIME,
        ])

    still_missing = [path for path in required_inputs if not path.exists()]
    if still_missing:
        raise FileNotFoundError(
            "P13 runtime bundle extraction completed but inputs are still missing: "
            + ", ".join(str(path) for path in still_missing)
        )

    print("[RUNTIME] P13 code/data bundle ready", flush=True)


def _stage_base_model():
    local_stage = Path("/tmp/p13_rescue_model_stage")
    local_zip = Path("/tmp/udsc_p13_reranker_rescue.zip")
    local_model = local_stage / "models/reranker"

    shutil.rmtree(local_stage, ignore_errors=True)
    if local_zip.exists():
        local_zip.unlink()

    src_zip = VOLUME_ROOT / "udsc_p13_reranker.zip"
    if not src_zip.is_file():
        raise FileNotFoundError(src_zip)

    shutil.copy2(src_zip, local_zip)
    local_stage.mkdir(parents=True, exist_ok=True)

    _run_short([
        "/usr/bin/python3.11",
        "-m",
        "zipfile",
        "-e",
        local_zip,
        local_stage,
    ])

    if not (local_model / "config.json").is_file():
        raise FileNotFoundError(
            f"Base model staging failed: {local_model}"
        )

    print("Base reranker staged:", local_model, flush=True)
    return local_model


def _patch_trainer():
    if not RUNTIME_TRAINER.is_file():
        raise FileNotFoundError(RUNTIME_TRAINER)

    text = RUNTIME_TRAINER.read_text(encoding="utf-8")

    old_negative = """negative = _training_document(
            negative_doc,
            record.get("negative_evidence"),
            str(title) if title else None,
        )"""
    new_negative = """negative = _training_document(
            negative_doc,
            record.get("negative_evidence"),
        )"""

    occurrences = text.count(old_negative)
    if occurrences != 2:
        raise RuntimeError(
            "Expected exactly 2 asymmetric negative formatter blocks; "
            f"found {occurrences}. Refusing unsafe patch."
        )
    text = text.replace(old_negative, new_negative)

    old_select = """    selected.sort(
        key=lambda row: (
            str(row["query_id"]),
            str(row["positive_doc"]),
            str(row["negative_doc"]),
            int(row.get("_copy_index", 0)),
        )
    )
    if max_training_pairs:
        selected = selected[:max_training_pairs]
"""

    new_select = """    selected.sort(
        key=lambda row: (
            str(row["query_id"]),
            str(row["positive_doc"]),
            str(row["negative_doc"]),
            int(row.get("_copy_index", 0)),
        )
    )

    # P13 rescue v1: query-balanced hard-negative cap.
    rescue_cap_per_query = 4
    rescue_kind_order = (
        "hard_false_positive",
        "semantic_confuser",
        "semi_hard",
        "lexical_confuser",
    )
    rescue_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        rescue_by_query.setdefault(str(row["query_id"]), []).append(row)

    rescue_balanced: list[dict[str, Any]] = []
    for rescue_query_id in sorted(rescue_by_query):
        rescue_rows = rescue_by_query[rescue_query_id]
        rescue_chosen: list[dict[str, Any]] = []
        rescue_used: set[int] = set()

        for rescue_kind in rescue_kind_order:
            for rescue_index, rescue_row in enumerate(rescue_rows):
                if rescue_index in rescue_used:
                    continue
                if str(rescue_row.get("negative_type", "")) == rescue_kind:
                    rescue_chosen.append(rescue_row)
                    rescue_used.add(rescue_index)
                    break

        if len(rescue_chosen) < rescue_cap_per_query:
            for rescue_index, rescue_row in enumerate(rescue_rows):
                if rescue_index in rescue_used:
                    continue
                rescue_chosen.append(rescue_row)
                rescue_used.add(rescue_index)
                if len(rescue_chosen) == rescue_cap_per_query:
                    break

        rescue_balanced.extend(rescue_chosen[:rescue_cap_per_query])

    selected = rescue_balanced
    if max_training_pairs:
        selected = selected[:max_training_pairs]
"""

    occurrences = text.count(old_select)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one selected.sort/max_training_pairs block; "
            f"found {occurrences}. Refusing unsafe patch."
        )
    text = text.replace(old_select, new_select)

    cls = text.index("class TorchBGERerankerBackend:")
    fit_start = text.index("    def fit_epoch(", cls)
    score_start = text.index("    def score(", fit_start)

    new_fit = """    def fit_epoch(self, examples: Sequence[TrainingExample]) -> float:
        \"\"\"Train one epoch with pairwise logistic ranking loss.\"\"\"
        if self._optimizer is None or self._scheduler is None:
            raise RuntimeError("prepare_training must run before fit_epoch")
        if len(examples) % 2:
            raise RuntimeError("pairwise rescue requires an even example count")

        torch = self._torch
        pairs: list[tuple[TrainingExample, TrainingExample]] = []
        for index in range(0, len(examples), 2):
            positive = examples[index]
            negative = examples[index + 1]
            if positive.label != 1.0 or negative.label != 0.0:
                raise RuntimeError(
                    "pairwise rescue expected adjacent positive/negative examples"
                )
            if positive.query_id != negative.query_id:
                raise RuntimeError(
                    "pairwise rescue pair has mismatched query IDs"
                )
            pairs.append((positive, negative))

        random.shuffle(pairs)
        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)

        pair_batch_size = max(1, self._batch_size // 2)
        batch_starts = range(0, len(pairs), pair_batch_size)
        losses: list[float] = []

        for start in tqdm(
            batch_starts,
            total=math.ceil(len(pairs) / pair_batch_size),
            desc="Pairwise training batches",
            unit="batch",
            leave=False,
        ):
            pair_batch = pairs[start : start + pair_batch_size]
            positives = [pair[0] for pair in pair_batch]
            negatives = [pair[1] for pair in pair_batch]
            combined = positives + negatives

            encoded, _ = self._batch(combined)
            autocast = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if self._fp16
                else nullcontext()
            )
            with autocast:
                logits = self._logits(encoded)
                n = len(pair_batch)
                positive_logits = logits[:n]
                negative_logits = logits[n:]
                loss = torch.nn.functional.softplus(
                    -(positive_logits - negative_logits)
                ).mean()
                scaled_loss = loss / self._gradient_accumulation

            self._scaler.scale(scaled_loss).backward()

            batch_index = start // pair_batch_size + 1
            is_last = start + pair_batch_size >= len(pairs)
            if batch_index % self._gradient_accumulation == 0 or is_last:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._scheduler.step()
                self._optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))

        if not losses:
            raise RuntimeError("pairwise rescue produced zero training batches")
        return sum(losses) / len(losses)

"""

    text = text[:fit_start] + new_fit + text[score_start:]

    PATCHED_TRAINER.parent.mkdir(parents=True, exist_ok=True)
    PATCHED_TRAINER.write_text(text, encoding="utf-8")

    _run_short([
        "/usr/bin/python3.11",
        "-m",
        "py_compile",
        PATCHED_TRAINER,
    ])

    patched = PATCHED_TRAINER.read_text(encoding="utf-8")
    gates = {
        "pairwise_loss": "Pairwise training batches" in patched,
        "balanced_cap": "rescue_cap_per_query = 4" in patched,
        "asymmetry_removed": patched.count(old_negative) == 0,
        "full_oof_subset_fix_present": "full_oof_requested" in patched,
    }
    print("PATCH GATES:", json.dumps(gates, indent=2), flush=True)
    print(
        "[PATCH] patched_trainer_sha256="
        + hashlib.sha256(PATCHED_TRAINER.read_bytes()).hexdigest(),
        flush=True,
    )
    if not all(gates.values()):
        raise RuntimeError(f"Patch gates failed: {gates}")


def _metric_value(d, key):
    value = d.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _evaluate_rescue() -> dict:
    """Evaluate the completed rescue fold without invoking training."""

    metrics_path = RESCUE_OUT / "fold_0/metrics.json"
    completion = RESCUE_OUT / "fold_0/completion.json"
    if not metrics_path.is_file() or not completion.is_file():
        raise RuntimeError("Rescue fold did not complete")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = metrics["baseline"]
    tuned = metrics["fine_tuned"]
    br = _metric_value(baseline, "official_macro_recall")
    tr = _metric_value(tuned, "official_macro_recall")
    bp = _metric_value(baseline, "official_macro_precision")
    tp = _metric_value(tuned, "official_macro_precision")
    bm = _metric_value(baseline, "multi_gold_recall")
    tm = _metric_value(tuned, "multi_gold_recall")
    if br is None or tr is None:
        raise RuntimeError("Rescue metrics are missing official_macro_recall")

    delta = tr - br
    precision_ok = bp is None or tp is None or tp >= bp - 0.005
    multi_ok = bm is None or tm is None or tm >= bm - 0.01
    pass_gate = delta >= 0.005 and precision_ok and multi_ok

    original_fold0 = json.loads(
        (BASE_OUT / "fold_0/metrics.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": "p13-rescue-v1-decision",
        "status": "PASS_SINGLE_FOLD_GATE" if pass_gate else "REJECT_SINGLE_FOLD_GATE",
        "fold": 0,
        "rescue_baseline": baseline,
        "rescue_tuned": tuned,
        "recall_delta": delta,
        "precision_ok": precision_ok,
        "multi_gold_ok": multi_ok,
        "original_p13_fold0_tuned": original_fold0.get("fine_tuned", {}),
        "next_action": (
            "run_cross_fold_rescue_validation"
            if pass_gate
            else "do_not_expand_this_training_variant"
        ),
    }


def _print_decision(decision: dict) -> None:
    baseline = decision["rescue_baseline"]
    tuned = decision["rescue_tuned"]
    print("\n============================================================", flush=True)
    print("P13 RESCUE V1 RESULT", flush=True)
    print("============================================================", flush=True)
    print(f"BASELINE_RECALL={baseline['official_macro_recall']:.6f}", flush=True)
    print(f"RESCUE_RECALL={tuned['official_macro_recall']:.6f}", flush=True)
    print(f"RESCUE_DELTA={decision['recall_delta']:+.6f}", flush=True)
    print(f"DECISION={decision['status']}", flush=True)
    print(f"NEXT={decision['next_action']}", flush=True)
    print(f"ARTIFACT={RESCUE_OUT / 'RESCUE_DECISION.json'}", flush=True)
    print("============================================================", flush=True)


@function(
    name="udsc-p13-rescue-v1-fold0",
    cpu=4,
    memory="64Gi",
    gpu="RTX4090",
    image=image,
    volumes=[
        Volume(
            name="udsc-p13",
            mount_path=str(VOLUME_ROOT),
        )
    ],
    timeout=-1,
    retries=2,
    headless=True,
)
def run_rescue():
    print("=== P13 RESCUE V1: FOLD 0 FAST GATE ===", flush=True)
    print(
        "Variant: symmetric formatting + balanced 4 negatives/query "
        "+ pairwise loss + LR 2e-6 + 1 epoch",
        flush=True,
    )

    RESCUE_OUT.mkdir(parents=True, exist_ok=True)
    decision_path = RESCUE_OUT / "RESCUE_DECISION.json"

    # A rescue launch can land on a volume where the P13 archives exist but the
    # runtime tree was never extracted (or was partially lost). Restore only
    # the code/data inputs; do not delete checkpoints or rescue artifacts.
    _ensure_runtime_bundle()
    if _stage_done("05_evaluation_done", required_paths=(decision_path,)):
        print("[SKIP] evaluation already completed", flush=True)
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        _print_decision(decision)
        return _rescue_result_from_decision(decision)
    
    RUNTIME_TRAINER.parent.mkdir(parents=True, exist_ok=True)

    source_trainer = Path(
    "/mnt/code/scripts/training/finetune_task1_bge_reranker.py"
)

    if not source_trainer.is_file():
        raise FileNotFoundError(source_trainer)

    RUNTIME_TRAINER.parent.mkdir(parents=True, exist_ok=True)

    tmp_trainer = RUNTIME_TRAINER.with_suffix(".py.tmp")
    shutil.copy2(source_trainer, tmp_trainer)
    os.replace(tmp_trainer, RUNTIME_TRAINER)

    source_sha256 = hashlib.sha256(source_trainer.read_bytes()).hexdigest()
    runtime_sha256 = hashlib.sha256(RUNTIME_TRAINER.read_bytes()).hexdigest()
    print(
        f"[SYNC] fresh trainer copied: "
        f"{source_trainer} -> {RUNTIME_TRAINER}",
        flush=True,
    )
    print(f"[SYNC] source_sha256={source_sha256}", flush=True)
    print(f"[SYNC] runtime_sha256={runtime_sha256}", flush=True)
    if source_sha256 != runtime_sha256:
        raise RuntimeError("Fresh trainer copy checksum mismatch")
    required = [
        RUNTIME_TRAINER,
        RUNTIME / "data/raw/btc/LegalIR/train.json",
        RUNTIME / "artifacts/task1/evaluation/strict_cv_v2/folds.json",
        RUNTIME / "artifacts/task1/candidates/train7000_document_candidates.jsonl",
        RUNTIME / "artifacts/task1/training/negatives/fold_0.jsonl",
        BASE_OUT / "BEAM_P13_FULL_5FOLD_SUCCESS.txt",
        BASE_OUT / "fold_0/metrics.json",
    ]
    if _stage_done("01_inputs_checked", required_paths=required):
        print("[SKIP] inputs already checked", flush=True)
    else:
        print("[STAGE] inputs check", flush=True)
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)
        _write_stage_marker(
            "01_inputs_checked",
            runtime=RUNTIME,
            runtime_trainer=RUNTIME_TRAINER,
            train=RUNTIME / "data/raw/btc/LegalIR/train.json",
            candidates=RUNTIME / "artifacts/task1/candidates/train7000_document_candidates.jsonl",
            negatives=RUNTIME / "artifacts/task1/training/negatives/fold_0.jsonl",
        )

    local_model = Path("/tmp/p13_rescue_model_stage/models/reranker")
    if _stage_done("02_base_model_staged", required_paths=(local_model,)):
        print("[SKIP] base model already staged", flush=True)
        model = local_model
    else:
        print("[STAGE] base model staging", flush=True)
        model = _stage_base_model()
        _write_stage_marker(
            "02_base_model_staged",
            source_archive=VOLUME_ROOT / "udsc_p13_reranker.zip",
            staged_model=model,
        )

    print("[STAGE] trainer patch - always rebuild from fresh source", flush=True)

    if PATCHED_TRAINER.exists():
        PATCHED_TRAINER.unlink()

    _patch_trainer()

    _write_stage_marker(
        "03_trainer_patched",
        runtime_trainer=RUNTIME_TRAINER,
        patched_trainer=PATCHED_TRAINER,
    )

    (RESCUE_OUT / "RESCUE_CONFIG.json").write_text(
        json.dumps(
            {
                "schema_version": "p13-rescue-v1",
                "fold": 0,
                "changes": [
                    "symmetric_positive_negative_formatting",
                    "max_4_negatives_per_query_stratified",
                    "pairwise_logistic_ranking_loss",
                    "learning_rate_2e-6",
                    "one_epoch",
                ],
                "base_model": str(model),
                "candidate_depth": 200,
                "evidence_limit": 2,
                "seed": 2026,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    cmd = [
        "/usr/bin/python3.11",
        "-u",
        PATCHED_TRAINER,
        "--train", "data/raw/btc/LegalIR/train.json",
        "--folds", "artifacts/task1/evaluation/strict_cv_v2/folds.json",
        "--negatives-dir", "artifacts/task1/training/negatives",
        "--candidates", "artifacts/task1/candidates/train7000_document_candidates.jsonl",
        "--base-model", model,
        "--output-dir", RESCUE_OUT,
        "--folds-to-run", "0",
        "--selection-policy", "fixed_epochs",
        "--ablation", "semi-hard-plus-hard",
        "--candidate-depth", "200",
        "--evidence-limit", "2",
        "--learning-rate", "2e-6",
        "--epochs", "1",
        "--batch-size", "4",
        "--gradient-accumulation", "4",
        "--max-length", "512",
        "--warmup-ratio", "0.1",
        "--seed", "2026",
        "--device", "cuda",
    ]

    training_artifacts = (
        RESCUE_OUT / "fold_0/metrics.json",
        RESCUE_OUT / "fold_0/completion.json",
        RESCUE_OUT / "fold_0/checkpoint",
    )
    if _stage_done("04_training_done", required_paths=training_artifacts):
        print("[SKIP] training already completed", flush=True)
    else:
        print("[STAGE] pairwise training", flush=True)
        _run_long(
            cmd,
            cwd=RUNTIME,
            log_path=RESCUE_OUT / "beam_logs/rescue_fold0.log",
        )
        if not all(path.exists() for path in training_artifacts):
            raise RuntimeError("Rescue trainer returned success without completed artifacts")
        _write_stage_marker(
            "04_training_done",
            output_dir=RESCUE_OUT,
            model_checkpoint=RESCUE_OUT / "fold_0/checkpoint",
            command=" ".join(str(part) for part in cmd),
        )

    print("[STAGE] evaluation", flush=True)
    decision = _evaluate_rescue()
    _write_stage_marker(
        "05_evaluation_done",
        output_dir=RESCUE_OUT,
        metrics=RESCUE_OUT / "fold_0/metrics.json",
        completion=RESCUE_OUT / "fold_0/completion.json",
    )

    print("[STAGE] decision", flush=True)
    decision_path.write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_decision(decision)
    return _rescue_result_from_decision(decision)


if __name__ == "__main__":
    print(
        "Enqueuing P13 rescue v1 single-fold gate on Beam RTX4090...",
        flush=True,
    )
    result = run_rescue.remote()
    print("REMOTE RESULT:", result, flush=True)
