from beam import function, Image, Volume


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


def remove_incomplete_fold(fold_dir: Path, completion: Path):
    """Remove a partial fold safely.

    This trainer does not persist model/optimizer checkpoints mid-fold.
    Therefore an incomplete fold cannot resume at batch/epoch level and must
    restart cleanly. Completed folds are never removed.
    """
    if completion.exists() or not fold_dir.exists():
        return

    try:
        has_anything = any(fold_dir.iterdir())
    except FileNotFoundError:
        return

    if not has_anything:
        return

    print(
        f"Incomplete {fold_dir.name} detected. "
        "No mid-fold checkpoint exists -> restarting this fold cleanly.",
        flush=True,
    )

    last_error = None
    for attempt in range(1, 6):
        try:
            shutil.rmtree(fold_dir)
            last_error = None
            break
        except FileNotFoundError:
            last_error = None
            break
        except Exception as e:
            last_error = e
            print(
                f"WARNING: remove attempt {attempt}/5 failed: {e}",
                flush=True,
            )
            time.sleep(3)

    if last_error is not None or fold_dir.exists():
        raise RuntimeError(
            f"Could not clean incomplete fold directory: {fold_dir}"
        )

    print(f"{fold_dir.name}: partial state removed.", flush=True)


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

    # Không tự chạy lại cả training khi có lỗi
    retries=0,

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
                "Removing stale full_oof directory...",
                flush=True,
            )

            shutil.rmtree(OUT)

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
            f"{'COMPLETE' if completion.exists() else 'NOT COMPLETE'}",
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

        if completion.exists():

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

        # Completed folds are skipped above. A partial fold cannot be resumed
        # at batch/epoch level because this trainer does not persist model /
        # optimizer checkpoints mid-fold, so restart only that fold cleanly.
        remove_incomplete_fold(
            fold_dir=fold_dir,
            completion=completion,
        )

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

        if not completion.exists():

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

    all_complete = True

    for fold in range(5):

        fold_dir = OUT / f"fold_{fold}"

        completion = (
            fold_dir / "completion.json"
        )

        predictions = (
            fold_dir
            / "finetuned_predictions.jsonl"
        )

        if not completion.exists():

            print(
                f"fold_{fold}: MISSING completion.json",
                flush=True,
            )

            all_complete = False
            continue

        if not predictions.exists():

            print(
                f"fold_{fold}: MISSING finetuned_predictions.jsonl",
                flush=True,
            )

            all_complete = False
            continue

        n = count_lines(predictions)

        print(
            f"fold_{fold}: COMPLETE, "
            f"OOF predictions = {n}",
            flush=True,
        )

        if n != 1400:

            print(
                f"WARNING: expected 1400 rows, got {n}",
                flush=True,
            )

            all_complete = False

    if not all_complete:

        raise RuntimeError(
            "P13 5-fold validation FAILED."
        )

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
