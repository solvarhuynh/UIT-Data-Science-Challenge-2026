[CmdletBinding()]
param(
    [switch]$SkipFrontend,
    [switch]$SkipCompose
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironmentPythons = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\bin\python")
)
$ProjectPython = $null
foreach ($Candidate in $VirtualEnvironmentPythons) {
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        $ProjectPython = $Candidate
        break
    }
}
if ($null -eq $ProjectPython) {
    $ProjectPython = (Get-Command python -ErrorAction Stop).Source
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "== $Name =="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    Invoke-CheckedCommand "Ruff format" {
        & $ProjectPython -m ruff format --check .
    }
    Invoke-CheckedCommand "Ruff lint" {
        & $ProjectPython -m ruff check .
    }
    Invoke-CheckedCommand "Mypy" {
        & $ProjectPython -m mypy src --no-warn-unused-configs
    }
    Invoke-CheckedCommand "Docstring style" {
        & $ProjectPython -m pydocstyle src
    }
    Invoke-CheckedCommand "Pytest sharded for native FAISS isolation" {
        $PreviousOmpThreads = $env:OMP_NUM_THREADS
        $PreviousOpenBlasThreads = $env:OPENBLAS_NUM_THREADS
        try {
            $env:OMP_NUM_THREADS = "1"
            $env:OPENBLAS_NUM_THREADS = "1"
            & $ProjectPython -m pytest -q -rs `
                --ignore=tests/unit/test_vector_db `
                --ignore=tests/retrieval/test_vector_db_adapters.py `
                --cov=src/udsc2026 `
                --cov-report=
            if ($LASTEXITCODE -ne 0) {
                throw "non-VectorDB pytest shard failed with exit code $LASTEXITCODE"
            }
            & $ProjectPython -m pytest -q -rs `
                tests/unit/test_vector_db `
                tests/retrieval/test_vector_db_adapters.py `
                --cov=src/udsc2026 `
                --cov-append `
                --cov-report=term-missing `
                --cov-fail-under=70
        }
        finally {
            $env:OMP_NUM_THREADS = $PreviousOmpThreads
            $env:OPENBLAS_NUM_THREADS = $PreviousOpenBlasThreads
        }
    }
    Invoke-CheckedCommand "Bandit" {
        & $ProjectPython -m bandit -q -r src
    }
    Invoke-CheckedCommand "TV5 smoke" {
        & $ProjectPython scripts/smoke_test.py --mode host
    }

    if (-not $SkipFrontend) {
        if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
            throw "npm is required for frontend checks; use -SkipFrontend to omit them"
        }
        Push-Location "frontend/giao dien"
        try {
            Invoke-CheckedCommand "Frontend check" {
                npm run check
            }
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipCompose) {
        if (Get-Command docker -ErrorAction SilentlyContinue) {
            Invoke-CheckedCommand "Docker Compose config" {
                docker compose config --quiet
            }
        }
        else {
            Write-Warning "Docker CLI is unavailable; Compose validation was skipped"
        }
    }

    Write-Host "All available project checks passed"
}
finally {
    Pop-Location
}
