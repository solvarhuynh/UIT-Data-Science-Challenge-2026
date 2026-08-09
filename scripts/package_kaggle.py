"""Package code and assets into a zip bundle for Kaggle notebook execution.

Creates 'tv3_kaggle_bundle.zip' containing all necessary code:
- src/ (core package with contracts, qa, llm)
- prompts/ (system prompts & rag templates)
- experiments/tv3/ (mock retriever & realistic benchmark scripts)
- docs/Scoring-Program-Task-LegalQA/ (BTC scoring code)
"""

import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = BASE_DIR / "tv3_kaggle_bundle.zip"

DIRS_TO_INCLUDE = [
    "src",
    "prompts",
    "experiments/tv3",
    "docs/Scoring-Program-Task-LegalQA",
]

def main():
    print(f"📦 Packaging TV3 Kaggle Bundle into '{OUTPUT_ZIP.name}'...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_dir in DIRS_TO_INCLUDE:
            target_dir = BASE_DIR / rel_dir
            if not target_dir.exists():
                print(f"⚠️ Warning: Directory '{rel_dir}' not found, skipping.")
                continue
            for file_path in target_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(".pyc") and "__pycache__" not in str(file_path):
                    arcname = file_path.relative_to(BASE_DIR)
                    zf.write(file_path, arcname)
                    print(f"  + Added: {arcname}")

    print("\n" + "=" * 60)
    print(f"🎉 SUCCESS: Bundle created at '{OUTPUT_ZIP}' ({OUTPUT_ZIP.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
