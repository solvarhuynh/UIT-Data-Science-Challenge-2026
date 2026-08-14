import time

from beam import Sandbox, Image, PythonVersion, Volume


sb = None

try:
    sb = Sandbox(
        name="udsc-p13-gpu-test",
        cpu=2,
        memory="8Gi",
        gpu="RTX5090",
        image=Image(
            python_version=PythonVersion.Python311
        ),
        volumes=[
            Volume(
                name="udsc-p13",
                mount_path="/workspace/p13",
            )
        ],
    ).create()

    print("GPU sandbox created; waiting for readiness...", flush=True)
    time.sleep(5)

    process = None
    last_error = None

    for attempt in range(1, 9):
        try:
            process = sb.process.exec(
                "bash",
                "-lc",
                (
                    "nvidia-smi; "
                    "echo; "
                    "echo '=== VOLUME ==='; "
                    "ls -lh /workspace/p13"
                ),
            )
            break
        except Exception as e:
            last_error = e
            print(
                f"Sandbox exec not ready ({attempt}/8): {e}",
                flush=True,
            )
            time.sleep(5)

    if process is None:
        raise RuntimeError(
            f"GPU sandbox never became ready: {last_error}"
        )

    process.wait()

    for line in process.logs:
        print(line, end="")

finally:
    if sb is not None:
        try:
            sb.terminate()
            print("\nGPU sandbox terminated.", flush=True)
        except Exception as e:
            print(
                "\nWARNING: GPU sandbox termination failed: "
                f"{e}",
                flush=True,
            )
