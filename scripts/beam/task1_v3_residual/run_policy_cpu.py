"""Local CPU orchestration for V3 actions, inner-CV policy, and fold0 evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def call(command: list[str]) -> None:
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode:
        raise RuntimeError(f"V3 CPU policy stage failed: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/task1/recovery_096/v3_residual/frozen_features.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/task1/recovery_096/v3_residual/policy")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--public-model", type=Path, default=ROOT / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib")
    parser.add_argument("--public-baseline", type=Path, default=ROOT / "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip")
    parser.add_argument("--public-output", type=Path, default=ROOT / "artifacts/task1/recovery_096/final_public_v3/public_v3a_predictions.json")
    parser.add_argument("--public-report", type=Path, default=ROOT / "artifacts/task1/recovery_096/final_public_v3/public_v3a_policy_report.json")
    parser.add_argument("--public-questions", type=Path, default=ROOT / "data/raw/btc/LegalIR/public-official.json")
    args = parser.parse_args()
    if args.public:
        if args.preflight:
            required = [Path(__file__).resolve().with_name("train_residual_policy.py"), Path(__file__).resolve().with_name("build_actions.py")]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("public V3A policy source missing:\n" + "\n".join(missing))
            manifest = args.public_model.with_name("v3a_final_fit_manifest.json")
            ready = False
            if manifest.is_file() and args.public_model.is_file():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                replay = payload.get("validation_replay", payload.get("fold0_replay", {}))
                production = payload.get("production_final_fit", {})
                ready = (
                    replay.get("status") == "PASS"
                    and int(replay.get("mismatch_count", -1)) == 0
                    and production.get("training_folds") == [0, 1, 2, 3, 4]
                    and production.get("no_model_selection_performed") is True
                    and production.get("no_public_training") is True
                )
            print(json.dumps({"status": "PREFLIGHT_PASS" if ready else "PREFLIGHT_BLOCKED", "mode": "public", "labels_required": False, "folds_required": False, "final_fit_manifest": str(manifest), "gpu_launched": False}))
            return
        call([sys.executable, "scripts/beam/task1_v3_residual/train_residual_policy.py", "--public", "--actions", str(args.features), "--output-dir", str(args.public_report.parent), "--public-model", str(args.public_model), "--public-baseline", str(args.public_baseline), "--public-output", str(args.public_output), "--public-report", str(args.public_report), "--public-questions", str(args.public_questions)])
        return
    if args.preflight:
        required = [Path(__file__).resolve().with_name("build_actions.py"), Path(__file__).resolve().with_name("train_residual_policy.py"), Path(__file__).resolve().with_name("evaluate_v3_fold0.py")]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError("V3 CPU policy source missing")
        print('{"status":"PREFLIGHT_PASS","gpu_launched":false,"remote_submitted":false}')
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = args.output_dir / "actions.jsonl"
    call([sys.executable, "scripts/beam/task1_v3_residual/build_actions.py", "--features", str(args.features), "--questions", "data/raw/btc/LegalIR/train.json", "--output", str(actions), "--report", str(args.output_dir / "actions_report.json")])
    call([sys.executable, "scripts/beam/task1_v3_residual/train_residual_policy.py", "--actions", str(actions), "--output-dir", str(args.output_dir)])
    call([sys.executable, "scripts/beam/task1_v3_residual/evaluate_v3_fold0.py", "--features", str(args.features), "--policy-predictions", str(args.output_dir / "fold0_v3_policy_predictions.jsonl"), "--baseline", "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl", "--questions", "data/raw/btc/LegalIR/train.json", "--report", str(args.output_dir / "fold0_evaluation.json")])


if __name__ == "__main__":
    main()
