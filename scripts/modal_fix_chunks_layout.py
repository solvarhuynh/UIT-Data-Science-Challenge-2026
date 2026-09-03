from pathlib import Path
import shutil
import modal

app = modal.App("fix-udsc-p13-chunks-layout")
volume = modal.Volume.from_name("udsc-p13")

MOUNT = Path("/workspace/p13")
SRC = MOUNT / "runtime/data/processed_v3/chunks/chunks"
DST = MOUNT / "runtime/data/processed_v3/chunks"

@app.function(volumes={str(MOUNT): volume}, timeout=1800)
def fix():
    if not SRC.exists():
        raise RuntimeError(f"Source not found: {SRC}")

    DST.mkdir(parents=True, exist_ok=True)

    files = [p for p in SRC.iterdir() if p.is_file()]
    print(f"Moving {len(files)} files")

    for p in files:
        target = DST / p.name

        if target.exists():
            if target.stat().st_size != p.stat().st_size:
                raise RuntimeError(f"Conflict with different size: {p.name}")
            p.unlink()
        else:
            shutil.move(str(p), str(target))

    try:
        SRC.rmdir()
    except OSError:
        pass

    volume.commit()

    remaining = list(SRC.iterdir()) if SRC.exists() else []
    final_files = [p for p in DST.iterdir() if p.is_file()]

    print(f"Final files in destination: {len(final_files)}")
    print(f"Remaining in nested folder: {len(remaining)}")


@app.local_entrypoint()
def main():
    fix.remote()