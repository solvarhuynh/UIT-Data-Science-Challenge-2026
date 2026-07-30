[CmdletBinding()]
param(
    [switch]$BuildDocker,
    [switch]$RunDockerSmoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironmentPythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv/bin/python")
)
$ProjectPython = $VirtualEnvironmentPythonCandidates |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1

if (-not $ProjectPython) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        $PythonCommand = Get-Command python3 -ErrorAction Stop
    }
    $ProjectPython = $PythonCommand.Source
}

Push-Location $ProjectRoot
try {
    Write-Host "== TV5 offline smoke checks =="
    & $ProjectPython ".\scripts\smoke_test.py" "--mode" "host"
    if ($LASTEXITCODE -ne 0) {
        throw "TV5 host smoke checks failed with exit code $LASTEXITCODE"
    }

    $Docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $Docker) {
        if ($BuildDocker -or $RunDockerSmoke) {
            throw "Docker was requested but the docker command is unavailable"
        }
        Write-Warning "Docker is unavailable; skipped Compose validation"
        return
    }

    Write-Host "== Docker Compose static validation =="
    & docker compose config --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose config failed with exit code $LASTEXITCODE"
    }

    if ($BuildDocker -or $RunDockerSmoke) {
        Write-Host "== Backend image build =="
        & docker compose build backend
        if ($LASTEXITCODE -ne 0) {
            throw "backend image build failed with exit code $LASTEXITCODE"
        }
    }

    if ($RunDockerSmoke) {
        Write-Host "== Packaged backend smoke checks =="
        & docker compose run --rm --no-deps backend-smoke
        if ($LASTEXITCODE -ne 0) {
            throw "container smoke checks failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
