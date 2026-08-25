from beam import function, Image, Volume


import json
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"

OUT = (
    RUNTIME
    / "artifacts/task1/models/"
      "bge_reranker_finetune/full_oof"
)

TRAINER = (
    RUNTIME
    / "scripts/training/"
      "finetune_task1_bge_reranker.py"
)


def synced_trainer_source() -> Path:
    """Locate the patched trainer shipped with the Beam source bundle."""
    here = Path(__file__).resolve().parent

    candidates = [
        # Production root launcher on Beam:
        # /mnt/code/beam_p13_run_all.py
        here / "finetune_task1_bge_reranker.py",

        # Local/canonical nested layout.
        here / "scripts" / "training" / "finetune_task1_bge_reranker.py",
        here.parent / "training" / "finetune_task1_bge_reranker.py",
    ]

    for candidate in candidates:
        if candidate.is_file():
            print(
                f"Patched trainer source: {candidate}",
                flush=True,
            )
            return candidate

    raise FileNotFoundError(
        "Beam source bundle is missing patched "
        "finetune_task1_bge_reranker.py"
    )

def sync_patched_trainer() -> None:
    """Overlay the uploaded trainer onto the persistent extracted runtime."""

    source = synced_trainer_source()
    TRAINER.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRAINER.with_name(TRAINER.name + ".sync.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, TRAINER)
    print(f"Synced patched trainer: {source} -> {TRAINER}", flush=True)


# ============================================================
# BEAM IMAGE
# ============================================================

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


# ============================================================
# HELPERS
# ============================================================

def run(cmd, cwd=None, log_path=None):
    cmd = [str(x) for x in cmd]

    print("\n============================================================")
    print("RUN:")
    print(" ".join(cmd))
    print("============================================================\n", flush=True)

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Trainer stdout is captured locally; this keeps the local log compact.
        "TQDM_MININTERVAL": "60",
        "TQDM_MAXINTERVAL": "120",
    }

    cwd_str = str(cwd) if cwd is not None else None

    # Short commands can stream directly to Beam Dashboard.
    if log_path is None:
        subprocess.run(
            cmd,
            cwd=cwd_str,
            env=env,
            check=True,
        )
        return

    # Long-running trainer output goes to local ephemeral disk.
    # Beam Dashboard receives only one heartbeat/minute, avoiding log-rate-limit
    # spam.  A small tail snapshot is persisted every heartbeat, and the full
    # log is copied to Beam Volume when the subprocess exits.
    persistent_log = Path(log_path)
    persistent_log.parent.mkdir(parents=True, exist_ok=True)

    persistent_tail = persistent_log.with_suffix(
        persistent_log.suffix + ".tail"
    )
    local_log = Path("/tmp") / (persistent_log.name + ".running")

    print(f"Full subprocess log -> {local_log}", flush=True)
    print(f"Persistent full log -> {persistent_log}", flush=True)
    print(f"Persistent tail     -> {persistent_tail}", flush=True)

    with local_log.open("wb") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd_str,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

        heartbeat_no = 0
        last_heartbeat = time.monotonic()

        while proc.poll() is None:
            time.sleep(5)

            now = time.monotonic()
            if now - last_heartbeat < 60:
                continue

            last_heartbeat = now
            heartbeat_no += 1

            try:
                f.flush()

                with local_log.open("rb") as rf:
                    rf.seek(0, 2)
                    size = rf.tell()
                    rf.seek(max(0, size - 65536))
                    tail = (
                        rf.read()
                        .decode("utf-8", errors="replace")
                        .replace("\r", "\n")
                    )

                lines = [x.strip() for x in tail.splitlines() if x.strip()]
                last_line = lines[-1] if lines else "(no output yet)"

                # Persist only a small tail once/minute.  This is intentionally
                # low-write so the distributed volume is not used as stdout.
                try:
                    persistent_tail.write_text(
                        "\n".join(lines[-200:]) + "\n",
                        encoding="utf-8",
                    )
                except Exception as e:
                    print(
                        f"[HEARTBEAT {heartbeat_no}] "
                        f"tail snapshot warning: {e}",
                        flush=True,
                    )

            except Exception as e:
                last_line = f"(heartbeat read error: {e})"

            print(
                f"[HEARTBEAT {heartbeat_no}] {last_line}",
                flush=True,
            )

        rc = proc.returncode

    # The child has exited. Persist the complete log with a few retries because
    # Beam Volume can occasionally have transient distributed-filesystem errors.
    copy_error = None
    for attempt in range(1, 6):
        try:
            shutil.copy2(local_log, persistent_log)
            copy_error = None
            print(f"Full log persisted: {persistent_log}", flush=True)
            break
        except Exception as e:
            copy_error = e
            print(
                f"WARNING: full-log persist attempt {attempt}/5 failed: {e}",
                flush=True,
            )
            time.sleep(3)

    if copy_error is not None:
        print(
            f"WARNING: full log remains only at {local_log}: {copy_error}",
            flush=True,
        )

    if rc != 0:
        print(
            "\n"
            "============================================================\n"
            f"SUBPROCESS FAILED WITH EXIT CODE {rc}\n"
            "LAST 200 LOG LINES:\n"
            "============================================================",
            flush=True,
        )

        try:
            text = local_log.read_text(errors="replace")
            lines = text.replace("\r", "\n").splitlines()
            print("\n".join(lines[-200:]), flush=True)
        except Exception as e:
            print(f"Could not read failure log: {e}", flush=True)

        raise subprocess.CalledProcessError(rc, cmd)



def prepare_volume_fold_dir(fold_dir: Path):
    """
    Beam Volume filesystem guard.

    Create the active fold directory and verify a real write/read/delete
    succeeds before giving the path to the trainer.

    Errno 11 / EAGAIN is treated as transient here only.
    """
    probe = fold_dir / ".beam_volume_write_probe"

    last_error = None

    for attempt in range(1, 13):
        try:
            fold_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            token = (
                f"beam-volume-probe "
                f"{fold_dir.name} "
                f"{time.time_ns()}\n"
            )

            probe.write_text(
                token,
                encoding="utf-8",
            )

            got = probe.read_text(
                encoding="utf-8",
            )

            if got != token:
                raise RuntimeError(
                    "Beam Volume probe readback mismatch"
                )

            probe.unlink(missing_ok=True)

            # Exercise directory metadata as well.
            list(fold_dir.iterdir())

            print(
                f"Beam Volume fold-dir PASS: {fold_dir}",
                flush=True,
            )

            return

        except OSError as e:
            probe.unlink(missing_ok=True)

            if e.errno != 11:
                raise

            last_error = e

            print(
                f"WARNING: Beam Volume EAGAIN while preparing "
                f"{fold_dir.name}; attempt {attempt}/12: {e}",
                flush=True,
            )

            time.sleep(10)

        except Exception:
            probe.unlink(missing_ok=True)
            raise

    raise RuntimeError(
        f"Beam Volume did not stabilize for {fold_dir} "
        f"after 12 attempts. Last error: {last_error}"
    )


def trainer_failed_early_with_eagain(log_path: Path):
    """
    Retry only the safe case:

      * subprocess failed with Errno 11 / EAGAIN
      * actual model training never started

    Once 'Training batches:' appears, return False so that we never
    automatically burn another multi-hour fold.
    """
    local_log = Path("/tmp") / (
        log_path.name + ".running"
    )

    try:
        text = local_log.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        print(
            f"WARNING: cannot inspect local trainer log: {e}",
            flush=True,
        )
        return False

    has_eagain = (
        "[Errno 11]" in text
        or "Resource temporarily unavailable" in text
    )

    training_started = (
        "Training batches:" in text
    )

    return (
        has_eagain
        and not training_started
    )

def count_lines(path: Path):
    with path.open("rb") as f:
        return sum(1 for _ in f)


def has_valid_completion(completion: Path) -> bool:
    """Legacy completed folds remain valid without new per-stage markers."""

    if not completion.is_file():
        return False
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "COMPLETE"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_query_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            query_id = str(row.get("query_id", "")).strip()
            if not query_id:
                raise RuntimeError(f"missing query_id at {path}:{line_number}")
            ids.append(query_id)
    return ids


def validate_fold_for_oof(fold_dir: Path, fold: int) -> dict:
    """Validate legacy/new completed folds by artifacts, not stage-marker age.

    Fold 0-3 may predate per-stage READY markers.  They are accepted only when
    completion/metrics/dataset hashes and both exact validation prediction files
    agree.  New stage-tracked folds must have all four READY markers.
    """

    required = {
        "completion": fold_dir / "completion.json",
        "metrics": fold_dir / "metrics.json",
        "dataset": fold_dir / "dataset_manifest.json",
        "validation_ids": fold_dir / "validation_ids.json",
        "baseline": fold_dir / "baseline_predictions.jsonl",
        "finetuned": fold_dir / "finetuned_predictions.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"fold_{fold}: missing OOF artifacts: {', '.join(missing)}")

    completion = _read_json(required["completion"])
    metrics = _read_json(required["metrics"])
    dataset = _read_json(required["dataset"])
    validation_ids = [str(x) for x in _read_json(required["validation_ids"])]

    if completion.get("status") != "COMPLETE":
        raise RuntimeError(f"fold_{fold}: completion status is not COMPLETE")
    if completion.get("fold", fold) != fold:
        raise RuntimeError(f"fold_{fold}: completion fold mismatch")
    if metrics.get("status") != "COMPLETE" or metrics.get("fold") != fold:
        raise RuntimeError(f"fold_{fold}: metrics status/fold mismatch")
    if len(validation_ids) != 1400 or len(set(validation_ids)) != 1400:
        raise RuntimeError(f"fold_{fold}: validation_ids must contain 1400 unique IDs")

    dataset_sha = dataset.get("dataset_sha256")
    if not dataset_sha:
        raise RuntimeError(f"fold_{fold}: dataset_manifest missing dataset_sha256")
    if completion.get("dataset_sha256") != dataset_sha:
        raise RuntimeError(f"fold_{fold}: completion/dataset hash mismatch")
    if metrics.get("dataset_sha256") != dataset_sha:
        raise RuntimeError(f"fold_{fold}: metrics/dataset hash mismatch")
    if metrics.get("validation_query_count") != 1400:
        raise RuntimeError(f"fold_{fold}: metrics validation_query_count != 1400")
    if metrics.get("selected_epoch") != 2:
        raise RuntimeError(f"fold_{fold}: expected selected_epoch=2")

    history = metrics.get("training_history")
    history_epochs = [item.get("epoch") for item in history] if isinstance(history, list) else []
    if history_epochs != [1, 2]:
        raise RuntimeError(
            f"fold_{fold}: training history must be exactly epochs [1, 2], got {history_epochs}"
        )

    baseline_ids = _jsonl_query_ids(required["baseline"])
    finetuned_ids = _jsonl_query_ids(required["finetuned"])
    if baseline_ids != validation_ids:
        raise RuntimeError(f"fold_{fold}: baseline predictions do not match validation_ids exactly")
    if finetuned_ids != validation_ids:
        raise RuntimeError(f"fold_{fold}: finetuned predictions do not match validation_ids exactly")

    stages = ["valid1", "epoch1", "epoch2", "valid2"]
    marker_paths = {stage: fold_dir / "stages" / stage / "READY.json" for stage in stages}
    marker_exists = {stage: path.is_file() for stage, path in marker_paths.items()}
    if all(marker_exists.values()):
        mode = "stage_tracked_v1"
        for stage, marker_path in marker_paths.items():
            marker = _read_json(marker_path)
            if (
                marker.get("schema_version") != "task1-p13-stage-ready-v1"
                or marker.get("fold") != fold
                or marker.get("stage") != stage
                or marker.get("dataset_sha256") != dataset_sha
            ):
                raise RuntimeError(f"fold_{fold}: invalid {stage} READY marker")
    elif not any(marker_exists.values()):
        mode = "legacy_complete"
    else:
        raise RuntimeError(
            f"fold_{fold}: COMPLETE fold has a partial stage-marker set: {marker_exists}"
        )

    return {
        "fold": fold,
        "compatibility_mode": mode,
        "dataset_sha256": dataset_sha,
        "validation_query_count": 1400,
        "selected_epoch": 2,
        "training_history_epochs": history_epochs,
        "baseline_rows": len(baseline_ids),
        "finetuned_rows": len(finetuned_ids),
        "stage_markers": marker_exists,
    }


# ============================================================
# REMOTE P13 JOB
# ============================================================

@function(
    name="udsc-p13-full-5fold",
    cpu=4,
    memory="64Gi",
    gpu="RTX5090",
    image=image,
    volumes=[
        Volume(
            name="udsc-p13",
            mount_path=str(VOLUME_ROOT),
        )
    ],

    # P13 có thể chạy nhiều giờ
    timeout=-1,

    # Epoch-boundary resume makes container-crash retries safe.
    retries=2,

    # Continue cloud training if local client disconnects
    headless=True,

)
def train_all_p13():

    print("\n")
    print("############################################################")
    print("# UDSC 2026 TASK1 - P13 FULL 5-FOLD")
    print("# Beam Serverless Function RTX5090")
    print("############################################################")
    print(flush=True)

    # --------------------------------------------------------
    # A. GPU PREFLIGHT
    # --------------------------------------------------------

    print("\n=== [A] NVIDIA GPU ===", flush=True)

    run(["nvidia-smi"])

    print("\n=== [A2] TORCH CUDA ===", flush=True)

    run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print('torch =', torch.__version__); "
                "print('cuda_available =', torch.cuda.is_available()); "
                "print('cuda_version =', torch.version.cuda); "
                "assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; "
                "print('gpu =', torch.cuda.get_device_name(0)); "
                "print('vram_GB =', "
                "round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))"
            ),
        ]
    )

    print("\n=== [A3] CONTAINER MEMORY ===", flush=True)

    run([
        "bash",
        "-lc",
        (
            "free -h; "
            "echo -n 'cgroup memory.max = '; "
            "cat /sys/fs/cgroup/memory.max 2>/dev/null || true; "
            "echo -n 'cgroup memory.current = '; "
            "cat /sys/fs/cgroup/memory.current 2>/dev/null || true"
        ),
    ])

    # --------------------------------------------------------
    # B. EXTRACT BUNDLE ONCE
    # --------------------------------------------------------

    print("\n=== [B] PREPARE PERSISTENT RUNTIME ===", flush=True)

    RUNTIME.mkdir(parents=True, exist_ok=True)

    extract_marker = RUNTIME / ".p13_bundle_extracted"

    if not extract_marker.exists():

        print("First run -> extracting P13 bundle...", flush=True)

        archives = [
            VOLUME_ROOT / "udsc_p13_code.zip",
            VOLUME_ROOT / "udsc_p13_data.zip",
            VOLUME_ROOT / "udsc_p13_reranker.zip",
        ]

        for archive in archives:

            if not archive.exists():
                raise FileNotFoundError(
                    f"Missing archive: {archive}"
                )

            print(
                f"\nExtracting {archive.name} ...",
                flush=True,
            )

            run(
                [
                    sys.executable,
                    "-m",
                    "zipfile",
                    "-e",
                    str(archive),
                    str(RUNTIME),
                ]
            )

        extract_marker.write_text(
            "P13 bundle extracted successfully\n"
        )

        print("\nBundle extraction COMPLETE.", flush=True)

    else:

        print(
            "Bundle already extracted -> SKIP extraction.",
            flush=True,
        )

    # The runtime bundle is deliberately immutable between jobs, but this small
    # patched trainer is synced with the Beam function source on every launch.
    sync_patched_trainer()

    # --------------------------------------------------------
    # C. REQUIRED FILE CHECK
    # --------------------------------------------------------

    print("\n=== [C] REQUIRED FILE CHECK ===", flush=True)

    required = [
        RUNTIME / "pyproject.toml",

        TRAINER,

        RUNTIME
        / "data/raw/btc/LegalIR/train.json",

        RUNTIME
        / "artifacts/task1/candidates/"
          "train7000_document_candidates.jsonl",

        RUNTIME
        / "artifacts/task1/evaluation/"
          "strict_cv_v2/folds.json",

        RUNTIME
        / "models/reranker",
    ]

    for fold in range(5):
        required.append(
            RUNTIME
            / "artifacts/task1/training/negatives/"
              f"fold_{fold}.jsonl"
        )

    missing = []

    for path in required:

        ok = path.exists()

        print(
            "[OK]     " if ok else "[MISSING]",
            path,
            flush=True,
        )

        if not ok:
            missing.append(str(path))

    if missing:
        raise RuntimeError(
            "P13 bundle incomplete:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # D. INSTALL LOCAL PROJECT
    # --------------------------------------------------------

    print("\n=== [D] INSTALL PROJECT ===", flush=True)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            ".",
        ],
        cwd=RUNTIME,
    )

    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(TRAINER),
        ],
        cwd=RUNTIME,
    )

    # --------------------------------------------------------
    # D2. STAGE RERANKER ZIP TO LOCAL EPHEMERAL DISK
    # Avoid loading HuggingFace model directly from Beam Volume.
    # --------------------------------------------------------

    print(
        "\n=== [D2] STAGE RERANKER TO LOCAL DISK ===",
        flush=True,
    )

    local_stage = Path("/tmp/p13_model_stage")
    local_zip = Path("/tmp/udsc_p13_reranker.zip")
    local_model = local_stage / "models/reranker"

    run([
        "rm",
        "-rf",
        str(local_stage),
        str(local_zip),
    ])

    run([
        "cp",
        str(VOLUME_ROOT / "udsc_p13_reranker.zip"),
        str(local_zip),
    ])

    local_stage.mkdir(
        parents=True,
        exist_ok=True,
    )

    run([
        "/usr/bin/python3.11",
        "-m",
        "zipfile",
        "-e",
        str(local_zip),
        str(local_stage),
    ])

    if not (local_model / "config.json").exists():
        raise FileNotFoundError(
            f"Local reranker staging failed: {local_model}"
        )

    print(
        f"Local reranker ready: {local_model}",
        flush=True,
    )

    run([
        "/usr/bin/python3.11",
        "-c",
        (
            "from transformers import AutoTokenizer; "
            "t=AutoTokenizer.from_pretrained("
            "'/tmp/p13_model_stage/models/reranker', "
            "local_files_only=True); "
            "print('LOCAL TOKENIZER LOAD PASS:', "
            "type(t).__name__)"
        ),
    ])

    # --------------------------------------------------------
    # E. INITIALIZE A FRESH FULL-OFF RUN ONCE
    # --------------------------------------------------------

    print("\n=== [E] INITIALIZE FULL 5-FOLD RUN ===", flush=True)

    fresh_marker = (
        RUNTIME / ".beam_p13_full5_initialized_v1"
    )

    if not fresh_marker.exists():

        print(
            "Fresh Beam P13 run requested.",
            flush=True,
        )

        if OUT.exists():

            print(
                "Existing full_oof state retained; completed folds and any "
                "resumable Fold 4 state are never deleted.",
                flush=True,
            )

        OUT.mkdir(
            parents=True,
            exist_ok=True,
        )

        fresh_marker.write_text(
            "Fresh Beam P13 5-fold run initialized\n"
        )

        print(
            "Fresh output initialized.",
            flush=True,
        )

    else:

        OUT.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "Existing Beam P13 run detected.",
            flush=True,
        )

        print(
            "Completed folds will be preserved/skipped.",
            flush=True,
        )

    # --------------------------------------------------------
    # F. CURRENT STATE
    # --------------------------------------------------------

    print("\n=== [F] CURRENT FOLD STATE ===", flush=True)

    for fold in range(5):

        completion = (
            OUT
            / f"fold_{fold}"
            / "completion.json"
        )

        print(
            f"fold_{fold}: "
            f"{'COMPLETE' if has_valid_completion(completion) else 'NOT COMPLETE'}",
            flush=True,
        )

    # --------------------------------------------------------
    # G. FULL 5-FOLD TRAINING
    # --------------------------------------------------------

    print("\n=== [G] START FULL P13 TRAINING ===", flush=True)

    for fold in range(5):

        fold_dir = OUT / f"fold_{fold}"

        completion = (
            fold_dir / "completion.json"
        )

        if has_valid_completion(completion):

            print(
                f"\n#############################################\n"
                f"FOLD {fold} ALREADY COMPLETE -> SKIP\n"
                f"#############################################",
                flush=True,
            )

            continue

        print(
            f"\n#############################################\n"
            f"STARTING FOLD {fold}/4\n"
            f"#############################################",
            flush=True,
        )

        cmd = [
            sys.executable,
            "-u",

            "scripts/training/"
            "finetune_task1_bge_reranker.py",

            "--train",
            "data/raw/btc/LegalIR/train.json",

            "--folds",
            "artifacts/task1/evaluation/"
            "strict_cv_v2/folds.json",

            "--negatives-dir",
            "artifacts/task1/training/negatives",

            "--candidates",
            "artifacts/task1/candidates/"
            "train7000_document_candidates.jsonl",

            "--base-model",
            "/tmp/p13_model_stage/models/reranker",

            "--output-dir",
            str(OUT),

            "--folds-to-run",
            str(fold),

            "--selection-policy",
            "fixed_epochs",

            "--ablation",
            "semi-hard-plus-hard",

            "--candidate-depth",
            "200",

            "--evidence-limit",
            "2",

            "--learning-rate",
            "2e-5",

            "--epochs",
            "2",

            "--batch-size",
            "4",

            "--gradient-accumulation",
            "4",

            "--max-length",
            "512",

            "--warmup-ratio",
            "0.1",

            "--seed",
            "2026",

            "--device",
            "cuda",
        ]

        # Preserve incomplete fold state.  The trainer validates and resumes a
        # READY epoch checkpoint, or starts epoch 1 if none is valid.

        # For folds 1..4, --resume preserves the already-completed earlier folds
        # in the shared full_oof output directory.
        if fold > 0:
            cmd.append("--resume")

        # ----------------------------------------------------
        # Beam distributed-volume guard.
        #
        # Fold 0 already proved that the trainer itself works.
        # Before each new fold, make sure its directory can really
        # be created/read/written on the mounted Beam Volume.
        # ----------------------------------------------------

        prepare_volume_fold_dir(
            fold_dir,
        )

        fold_log_path = (
            OUT
            / "beam_logs"
            / f"fold_{fold}.log"
        )

        # ----------------------------------------------------
        # Safe automatic retry for EARLY Beam Volume EAGAIN only.
        #
        # If actual "Training batches:" has appeared, DO NOT retry
        # automatically because this trainer has no mid-fold
        # checkpoint and that could waste hours of GPU compute.
        # ----------------------------------------------------

        for launch_attempt in range(1, 4):

            try:
                run(
                    cmd,
                    cwd=RUNTIME,
                    log_path=fold_log_path,
                )

                break

            except subprocess.CalledProcessError:

                safe_eagain_retry = (
                    trainer_failed_early_with_eagain(
                        fold_log_path
                    )
                )

                if (
                    not safe_eagain_retry
                    or launch_attempt >= 3
                ):
                    raise

                print(
                    "\n"
                    "============================================================\n"
                    f"EARLY BEAM VOLUME EAGAIN ON FOLD {fold}\n"
                    f"Safe relaunch attempt {launch_attempt + 1}/3 "
                    "after 30 seconds.\n"
                    "No Training batches were started, so no training "
                    "progress is being discarded.\n"
                    "============================================================",
                    flush=True,
                )

                time.sleep(30)

                prepare_volume_fold_dir(
                    fold_dir,
                )

        if not has_valid_completion(completion):

            raise RuntimeError(
                f"Fold {fold} process exited successfully "
                f"but completion.json was not created."
            )

        print(
            f"\n=============================================\n"
            f"FOLD {fold} COMPLETE ✅\n"
            f"=============================================",
            flush=True,
        )

    # --------------------------------------------------------
    # H. FINAL VALIDATION
    # --------------------------------------------------------

    print("\n=== [H] FINAL 5-FOLD VALIDATION ===", flush=True)

    # Fold 0-3 may have produced a stale, partial root aggregate before Fold 4
    # existed.  Re-run only the trainer's aggregation path: completed folds are
    # loaded through --resume and no model training is repeated.
    print("Rebuilding strict OOF aggregate from all five completed folds...", flush=True)
    run(
        [
            sys.executable,
            "-u",
            "scripts/training/finetune_task1_bge_reranker.py",
            "--train", "data/raw/btc/LegalIR/train.json",
            "--folds", "artifacts/task1/evaluation/strict_cv_v2/folds.json",
            "--negatives-dir", "artifacts/task1/training/negatives",
            "--candidates", "artifacts/task1/candidates/train7000_document_candidates.jsonl",
            "--base-model", "/tmp/p13_model_stage/models/reranker",
            "--output-dir", str(OUT),
            "--folds-to-run", "0", "1", "2", "3", "4",
            "--selection-policy", "fixed_epochs",
            "--ablation", "semi-hard-plus-hard",
            "--candidate-depth", "200",
            "--evidence-limit", "2",
            "--learning-rate", "2e-5",
            "--epochs", "2",
            "--batch-size", "4",
            "--gradient-accumulation", "4",
            "--max-length", "512",
            "--warmup-ratio", "0.1",
            "--seed", "2026",
            "--device", "cuda",
            "--resume",
        ],
        cwd=RUNTIME,
        log_path=OUT / "beam_logs" / "aggregate.log",
    )

    compatibility_rows = []
    validation_union: set[str] = set()

    for fold in range(5):
        fold_dir = OUT / f"fold_{fold}"
        audit = validate_fold_for_oof(fold_dir, fold)
        compatibility_rows.append(audit)
        fold_validation_ids = [
            str(x) for x in _read_json(fold_dir / "validation_ids.json")
        ]
        overlap = validation_union.intersection(fold_validation_ids)
        if overlap:
            raise RuntimeError(
                f"fold_{fold}: validation IDs overlap earlier folds: {sorted(overlap)[:5]}"
            )
        validation_union.update(fold_validation_ids)
        print(
            f"fold_{fold}: OOF COMPAT PASS; mode={audit['compatibility_mode']}; "
            "Valid1=1400, Epochs=[1,2], Valid2=1400",
            flush=True,
        )

    if len(validation_union) != 7000:
        raise RuntimeError(
            f"strict validation union must be 7000 unique IDs, got {len(validation_union)}"
        )

    compatibility_manifest = OUT / "OOF_FOLD_COMPATIBILITY.json"
    compatibility_manifest.write_text(
        json.dumps(
            {
                "schema_version": "task1-p13-oof-fold-compatibility-v1",
                "status": "PASS",
                "note": (
                    "legacy_complete folds predate per-stage READY markers; "
                    "stage markers are crash-recovery metadata and do not change OOF scoring"
                ),
                "expected_fold_count": 5,
                "expected_validation_queries_per_fold": 1400,
                "expected_total_oof_queries": 7000,
                "folds": compatibility_rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    aggregate_oof = OUT / "oof_predictions.jsonl"
    aggregate_required = (
        aggregate_oof,
        OUT / "comparison.json",
        OUT / "comparison.md",
        OUT / "final_decision.json",
    )
    missing_aggregate = [str(path) for path in aggregate_required if not path.is_file()]
    if missing_aggregate:
        raise RuntimeError("P13 aggregate outputs missing:\n" + "\n".join(missing_aggregate))
    aggregate_rows = count_lines(aggregate_oof)
    print(f"Aggregate OOF predictions = {aggregate_rows}", flush=True)
    if aggregate_rows != 7000:
        raise RuntimeError(
            "P13 aggregate is not a complete strict OOF result: "
            f"expected 7000 rows, got {aggregate_rows}"
        )
    aggregate_ids = _jsonl_query_ids(aggregate_oof)
    if len(set(aggregate_ids)) != 7000:
        raise RuntimeError("aggregate OOF contains duplicate query IDs")
    if set(aggregate_ids) != validation_union:
        raise RuntimeError("aggregate OOF query IDs do not equal the strict 5-fold validation union")
    comparison_payload = _read_json(OUT / "comparison.json")
    if comparison_payload.get("fold_count") != 5:
        raise RuntimeError("comparison.json fold_count is not 5")
    if comparison_payload.get("partial_oof") is not False:
        raise RuntimeError("comparison.json says OOF is partial")
    print("Aggregate OOF identity/fold compatibility PASS.", flush=True)

    # --------------------------------------------------------
    # I. SUCCESS MARKER ON PERSISTENT VOLUME
    # --------------------------------------------------------

    success_marker = (
        OUT / "BEAM_P13_FULL_5FOLD_SUCCESS.txt"
    )

    success_marker.write_text(
        "P13 Beam full 5-fold training COMPLETE\n"
        "5 folds x 1400 OOF predictions = 7000 rows\n"
    )

    print("\n")
    print("############################################################")
    print("# P13 FULL 5-FOLD COMPLETE ✅")
    print("#")
    print(f"# Persistent output:")
    print(f"# {OUT}")
    print("#")
    print("# 5 x 1400 OOF rows = 7000")
    print("############################################################")
    print("\n", flush=True)

    return {
        "status": "complete",
        "folds": 5,
        "oof_rows": 7000,
        "output_dir": str(OUT),
    }


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    print(
        "Enqueuing P13 full 5-fold training "
        "on Beam serverless RTX5090...",
        flush=True,
    )

    result = train_all_p13.remote()
    print("REMOTE RESULT:", result, flush=True)
