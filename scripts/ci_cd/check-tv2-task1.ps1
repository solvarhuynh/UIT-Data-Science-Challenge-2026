param(
    [ValidateSet("unit", "data", "model")]
    [string]$Tier = "unit"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot
$RepoPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
# A fixed pytest runtime can be left owned by a different Windows account.
# Use a fresh workspace-local directory per invocation so pytest never tries
# to delete an inaccessible prior runtime (and never needs C:\Temp).
$PytestBaseTemp = Join-Path $RepoRoot (".pytest_runtime_" + [guid]::NewGuid().ToString("N"))
$Task1Python = if (Test-Path -LiteralPath $RepoPython -PathType Leaf) {
    $RepoPython
} else {
    "python"
}

function Invoke-CheckedPython {
    param([string[]]$PythonArgs)
    & $Task1Python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $Task1Python $($PythonArgs -join ' ')"
    }
}

function Invoke-CheckedTool {
    param([string[]]$Command)
    & $Command[0] $Command[1..($Command.Count - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $($Command -join ' ')"
    }
}

function Invoke-UnitTier {
    Invoke-CheckedPython -PythonArgs @("-m", "compileall", "-q", "src", "scripts")

    $TaskOneLintTargets = @(
        "src/udsc2026/evaluation/legal_ir.py",
        "src/udsc2026/evaluation/legal_ir_candidates.py",
        "src/udsc2026/evaluation/legal_ir_diagnostics.py",
        "src/udsc2026/evaluation/legal_ir_lexical.py",
        "src/udsc2026/evaluation/legal_ir_document_candidates.py",
        "src/udsc2026/retrieval/multigranularity.py",
        "src/udsc2026/evaluation/legal_ir_submission.py",
        "scripts/evaluation/evaluate_legal_ir.py",
        "scripts/evaluation/benchmark_reranker.py",
        "scripts/evaluation/analyze_legal_ir_candidates.py",
        "scripts/evaluation/generate_dense_candidates.py",
        "scripts/evaluation/tune_legal_ir_document_aggregation.py",
        "scripts/evaluation/build_strict_legal_ir_cv.py",
        "scripts/evaluation/check_task1_p13_prerequisites.py",
        "scripts/package_kaggle_task1.py",
        "scripts/evaluation/ablate_legal_ir_lexical.py",
        "scripts/training/mine_task1_negatives.py",
        "scripts/training/finetune_task1_bge_reranker.py",
        "scripts/data_prep/plan_legal_multigranularity_index.py",
        "scripts/task1/run_legal_ir_pipeline.py",
        "scripts/submission/build_legal_ir_ensemble.py",
        "scripts/submission/write_legal_ir_submission.py",
        "scripts/submission/validate_legal_ir_submission.py",
        "tests/unit/test_evaluation/test_legal_ir.py",
        "tests/unit/test_evaluation/test_legal_ir_candidates.py",
        "tests/unit/test_evaluation/test_legal_ir_diagnostics.py",
        "tests/unit/test_evaluation/test_legal_ir_strict_cv.py",
        "tests/unit/test_task1_p13_prerequisites.py",
        "tests/unit/test_package_kaggle_task1.py",
        "tests/unit/test_evaluation/test_legal_ir_lexical.py",
        "tests/unit/test_evaluation/test_legal_ir_document_candidates.py",
        "tests/unit/test_evaluation/test_task1_pipeline_runner.py",
        "tests/unit/test_retrieval/test_multigranularity.py",
        "tests/unit/test_evaluation/test_reranker_batch_cli.py",
        "tests/unit/test_mine_task1_negatives.py",
        "tests/unit/test_finetune_task1_bge_reranker.py"
    )
    $RuffArgs = @("-m", "ruff", "check") + $TaskOneLintTargets
    Invoke-CheckedPython -PythonArgs $RuffArgs

    Invoke-CheckedPython -PythonArgs @(
        "-m", "pytest", "-q",
        "--basetemp", $PytestBaseTemp,
        "tests/unit/test_evaluation/test_legal_ir.py",
        "tests/unit/test_evaluation/test_legal_ir_candidates.py",
        "tests/unit/test_evaluation/test_legal_ir_cli.py",
        "tests/unit/test_evaluation/test_legal_ir_submission.py",
        "tests/unit/test_evaluation/test_build_legal_ir_ensemble.py",
        "tests/unit/test_evaluation/test_legal_ir_diagnostics.py",
        "tests/unit/test_evaluation/test_legal_ir_strict_cv.py",
        "tests/unit/test_task1_p13_prerequisites.py",
        "tests/unit/test_package_kaggle_task1.py",
        "tests/unit/test_evaluation/test_legal_ir_lexical.py",
        "tests/unit/test_evaluation/test_legal_ir_document_candidates.py",
        "tests/unit/test_evaluation/test_task1_pipeline_runner.py",
        "tests/unit/test_retrieval/test_multigranularity.py",
        "tests/unit/test_evaluation/test_reranker_batch_cli.py",
        "tests/unit/test_mine_task1_negatives.py",
        "tests/unit/test_finetune_task1_bge_reranker.py",
        "tests/retrieval/test_bm25_tokenizer.py",
        "tests/retrieval/test_score_fusion.py"
    )
}

function Invoke-DataTier {
    Invoke-UnitTier
    $TrainPath = Join-Path $RepoRoot "data\raw\btc\LegalIR\train.json"
    if (-not (Test-Path -LiteralPath $TrainPath -PathType Leaf)) {
        Write-Host "SKIPPED data tier: LegalIR train.json is unavailable."
        return
    }

    $DataSmokeOutput = Join-Path $RepoRoot "artifacts\task1\evaluation\test_gate_data_cv"
    Invoke-CheckedPython -PythonArgs @(
        "scripts/evaluation/build_strict_legal_ir_cv.py",
        "--input", $TrainPath,
        "--output-dir", $DataSmokeOutput,
        "--folds", "5",
        "--seed", "2026",
        "--limit", "100"
    )
    $P2Predictions = Join-Path $RepoRoot "artifacts\task1\train500b_dense200_predictions.jsonl"
    $P2Manifest = Join-Path $RepoRoot "artifacts\task1\train500b_dense200_manifest.json"
    $P2StrictFolds = Join-Path $RepoRoot "artifacts\task1\evaluation\strict_cv_v2\folds.json"
    if (
        (Test-Path -LiteralPath $P2Predictions -PathType Leaf) -and
        (Test-Path -LiteralPath $P2Manifest -PathType Leaf) -and
        (Test-Path -LiteralPath $P2StrictFolds -PathType Leaf)
    ) {
        Invoke-CheckedPython -PythonArgs @(
            "scripts/evaluation/analyze_legal_ir_candidates.py",
            "--gold", $TrainPath,
            "--predictions", $P2Predictions,
            "--manifest", $P2Manifest,
            "--strict-folds", $P2StrictFolds,
            "--max-questions", "100",
            "--output-dir", "artifacts/task1/evaluation/test_gate_p2_candidate"
        )
    } else {
        Write-Host "SKIPPED P2 candidate smoke: cached train500b dense200 inputs are unavailable."
    }
    # The targeted fixture tests above exercise evaluator, TF-IDF and BM25
    # without loading neural weights. The commands above additionally validate
    # the real train schema, strict grouping, and cached candidate diagnostics.
    Write-Host "PASS data tier: train schema/grouping smoke completed (max 100 questions)."
}

function Invoke-ModelTier {
    Invoke-DataTier
    $RerankerModel = Join-Path $RepoRoot "models\cross-encoder\bge-reranker-v2-m3"
    $SmokeBenchmark = Join-Path $RepoRoot "artifacts\task1\model_smoke_benchmark.jsonl"
    $SmokeCandidates = Join-Path $RepoRoot "artifacts\task1\model_smoke_candidates.jsonl"
    if (
        -not (Test-Path -LiteralPath $RerankerModel -PathType Container) -or
        -not (Test-Path -LiteralPath $SmokeBenchmark -PathType Leaf) -or
        -not (Test-Path -LiteralPath $SmokeCandidates -PathType Leaf)
    ) {
        Write-Host "SKIPPED model tier: requires a local reranker and prebuilt <=10-question model_smoke benchmark/candidates; no model was downloaded."
        return
    }

    Invoke-CheckedPython -PythonArgs @(
        "scripts/evaluation/benchmark_reranker.py",
        "--benchmark", $SmokeBenchmark,
        "--candidates", $SmokeCandidates,
        "--model", $RerankerModel,
        "--candidate-k", "50",
        "--top-n", "5",
        "--output-dir", "artifacts/task1/evaluation/test_gate_model"
    )
    Write-Host "PASS model tier: local model smoke completed without remote download."
}

switch ($Tier) {
    "unit" { Invoke-UnitTier }
    "data" { Invoke-DataTier }
    "model" { Invoke-ModelTier }
}

Write-Host "PASS Task 1 gate tier=$Tier"
