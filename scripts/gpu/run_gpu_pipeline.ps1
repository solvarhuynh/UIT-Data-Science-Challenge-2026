param(
    [switch]$SkipInstall,
    [switch]$SkipDownload,
    [switch]$SkipLLMSmoke,
    [string]$ProcessedRoot = "data/processed_v3",
    [int]$SmokeChunks = 10000,
    [int]$SmokeQuestions = 10
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

# Retrieval and QA configs intentionally bind parent expansion to this canonical
# corpus. Reject an alternate root instead of silently mixing its children with
# data/processed_v3 parents at runtime.
$CanonicalProcessedRoot = (Resolve-Path (Join-Path $RepoRoot "data\processed_v3")).Path
$RequestedProcessedRoot = (Resolve-Path $ProcessedRoot).Path
if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
    $CanonicalProcessedRoot.TrimEnd('\'),
    $RequestedProcessedRoot.TrimEnd('\')
)) {
    throw "ProcessedRoot must resolve to $CanonicalProcessedRoot so chunks, benchmark, and parent context use one corpus."
}

function Invoke-CheckedPython {
    param([string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: python $($PythonArgs -join ' ')"
    }
}

if (-not $SkipInstall) {
    Invoke-CheckedPython -PythonArgs @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-CheckedPython -PythonArgs @(
        "-m", "pip", "install", "torch>=2.7,<3.0",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    )
    Invoke-CheckedPython -PythonArgs @("-m", "pip", "install", "-e", ".[gpu]")
}

if (-not $SkipDownload) {
    Invoke-CheckedPython -PythonArgs @("download_models.py")
}

Invoke-CheckedPython -PythonArgs @(
    "scripts/gpu/preflight.py",
    "--processed-root", $ProcessedRoot
)

if (-not $SkipLLMSmoke) {
    Invoke-CheckedPython -PythonArgs @("scripts/gpu/smoke_llm.py")
}

# A bounded end-to-end smoke build catches CUDA/model/FAISS incompatibilities
# before committing several hours to the complete corpus.
Invoke-CheckedPython -PythonArgs @(
    "scripts/data_prep/index_chunks.py",
    "--chunks-dir", (Join-Path $ProcessedRoot "chunks"),
    "--config-env", "gpu",
    "--skip-bm25",
    "--max-chunks", "$SmokeChunks",
    "--force"
)

Invoke-CheckedPython -PythonArgs @(
    "scripts/evaluation/generate_dense_candidates.py",
    "--benchmark", (Join-Path $ProcessedRoot "benchmarks/synthetic_qa.jsonl"),
    "--config-env", "gpu",
    "--limit", "$SmokeQuestions",
    "--benchmark-subset", "artifacts/tv2/smoke_benchmark.jsonl",
    "--output", "artifacts/tv2/smoke_dense_predictions.jsonl",
    "--manifest", "artifacts/tv2/smoke_dense_manifest.json"
)

Invoke-CheckedPython -PythonArgs @(
    "scripts/evaluation/benchmark_reranker.py",
    "--benchmark", "artifacts/tv2/smoke_benchmark.jsonl",
    "--candidates", "artifacts/tv2/smoke_dense_predictions.jsonl",
    "--model", "models/reranker",
    "--device", "cuda",
    "--batch-size", "8",
    "--max-length", "1024",
    "--candidate-k", "50",
    "--top-n", "5",
    "--fp16",
    "--output-dir", "artifacts/tv5/smoke-bge-reranker-v2-m3"
)

# Full dense index and the 100-question internal benchmark.
Invoke-CheckedPython -PythonArgs @(
    "scripts/data_prep/index_chunks.py",
    "--chunks-dir", (Join-Path $ProcessedRoot "chunks"),
    "--config-env", "gpu",
    "--skip-bm25",
    "--force"
)

Invoke-CheckedPython -PythonArgs @(
    "scripts/evaluation/generate_dense_candidates.py",
    "--benchmark", (Join-Path $ProcessedRoot "benchmarks/synthetic_qa.jsonl"),
    "--config-env", "gpu",
    "--output", "artifacts/tv2/dense_predictions.jsonl",
    "--manifest", "artifacts/tv2/dense_run_manifest.json"
)

Invoke-CheckedPython -PythonArgs @(
    "scripts/evaluation/benchmark_reranker.py",
    "--benchmark", (Join-Path $ProcessedRoot "benchmarks/synthetic_qa.jsonl"),
    "--candidates", "artifacts/tv2/dense_predictions.jsonl",
    "--model", "models/reranker",
    "--device", "cuda",
    "--batch-size", "8",
    "--max-length", "1024",
    "--candidate-k", "50",
    "--top-n", "5",
    "--fp16",
    "--output-dir", "artifacts/tv5/bge-reranker-v2-m3"
)

Write-Host "GPU pipeline completed. Review artifacts/tv5/bge-reranker-v2-m3/evaluation/comparison.md"
