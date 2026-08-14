import time

from beam import Sandbox, Image, PythonVersion, Volume


sb = None

try:
    sb = Sandbox(
        name="udsc-p13-read-failure-tail",
        cpu=2,
        memory="8Gi",
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

    print("Sandbox created; waiting for readiness...", flush=True)
    time.sleep(8)

    cmd = r'''
set -e

LOG=/workspace/p13/runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/beam_logs/fold_0.log.tail

echo "============================================================"
echo "P13 FOLD 0 FAILURE TAIL"
echo "============================================================"

if [ ! -f "$LOG" ]; then
    echo "ERROR: tail log not found:"
    echo "$LOG"
    exit 2
fi

echo
echo "=== FILE ==="
ls -lh "$LOG"

echo
echo "=== LAST 300 LINES ==="
tail -n 300 "$LOG"

echo
echo "============================================================"
echo "TAIL READ COMPLETE"
echo "============================================================"
'''

    p = sb.process.exec(
        "bash",
        "-lc",
        cmd,
    )

    p.wait()

    for line in p.logs:
        print(line, end="")

finally:
    if sb is not None:
        try:
            sb.terminate()
            print("\nSandbox terminated.", flush=True)
        except Exception as e:
            print(
                "\nWARNING: sandbox termination:",
                repr(e),
                flush=True,
            )
