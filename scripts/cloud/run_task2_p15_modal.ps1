[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Setup", "Check", "Prepare", "Smoke", "Run", "Status", "Logs", "Download", "Stop")]
    [string]$Stage,

    [string]$P14Archive = "artifacts\task2\task2_p14_beam_input.tar.zst",

    [string]$P15Bundle = "artifacts\task2\task2_p15_modal_input.zip",

    [string]$DownloadDir = "artifacts\task2\modal_p15_download"
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Runner = Join-Path $ProjectRoot "scripts\cloud\modal_task2_p15.py"

function Resolve-PythonExecutable {
    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    throw "Python was not found. Create the repository virtual environment first."
}

function Resolve-ModalExecutable {
    param([Parameter(Mandatory = $true)][string]$PythonExe)

    $pythonDirectory = Split-Path -Parent $PythonExe
    $candidates = @(
        (Join-Path $pythonDirectory "modal.exe"),
        (Join-Path $ProjectRoot ".venv\Scripts\modal.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\modal.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    $command = Get-Command modal -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    throw (
        "Modal CLI is not installed. Run: " +
        ".\scripts\cloud\run_task2_p15_modal.ps1 -Stage Setup"
    )
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $Executable $($Arguments -join ' ')"
    }
}

function Assert-ModalServerReady {
    param([Parameter(Mandatory = $true)][string]$ModalExe)

    $lastOutput = @()
    $lastExitCode = -1
    $lastJsonValid = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $previous = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $lastOutput = @(& $ModalExe app list --json 2>&1)
            $lastExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previous
        }

        # Windows PowerShell 5.1 can surface a successful native CLI response
        # as NativeCommandError when stderr is merged into stdout. A valid
        # `modal app list --json` payload proves that the server was reached and
        # the current profile was authenticated, even if LASTEXITCODE is stale.
        $probeText = (($lastOutput | ForEach-Object { $_.ToString() }) -join "`n").Trim()
        $lastJsonValid = $false
        if (-not [string]::IsNullOrWhiteSpace($probeText)) {
            try {
                $null = $probeText | ConvertFrom-Json -ErrorAction Stop
                $lastJsonValid = $true
            } catch {
                $lastJsonValid = $false
            }
        }

        if (($lastExitCode -eq 0) -or $lastJsonValid) {
            if ($lastExitCode -ne 0) {
                Write-Warning (
                    "Modal returned valid authenticated JSON despite native exit code " +
                    "$lastExitCode; accepting the successful preflight response."
                )
            }
            Write-Host "MODAL SERVER PREFLIGHT PASS"
            return
        }
        if ($attempt -lt 3) {
            Start-Sleep -Seconds (5 * $attempt)
        }
    }
    throw (
        "Cannot reach Modal after 3 attempts (exit code $lastExitCode; " +
        "valid JSON: $lastJsonValid). No GPU job was launched. Last output:`n" +
        (($lastOutput | ForEach-Object { $_.ToString() }) -join "`n")
    )
}

function Resolve-ActiveModalAppId {
    param(
        [Parameter(Mandatory = $true)][string]$ModalExe,
        [Parameter(Mandatory = $true)][string]$AppName
    )

    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& $ModalExe app list --json 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    $text = (($output | ForEach-Object { $_.ToString() }) -join "`n").Trim()
    if ($exitCode -ne 0) {
        throw "Cannot list Modal apps (exit code $exitCode):`n$text"
    }
    try {
        $apps = @($text | ConvertFrom-Json -ErrorAction Stop)
    } catch {
        throw "Modal app list returned invalid JSON:`n$text"
    }

    $active = @(
        $apps | Where-Object {
            $_.description -eq $AppName -and
            $_.state -ne "stopped" -and
            [int]$_.tasks -gt 0
        } | Sort-Object -Property created_at -Descending
    )
    if ($active.Count -eq 0) {
        throw (
            "No active detached Modal job named '$AppName' was found. " +
            "Run -Stage Run first, or inspect the latest run client log."
        )
    }
    return [string]$active[0].app_id
}

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Modal P15 runner is missing: $Runner"
}

$Python = Resolve-PythonExecutable
Set-Location $ProjectRoot

if ($Stage -eq "Setup") {
    Invoke-NativeChecked -Executable $Python -Arguments @(
        "-m", "pip", "install", "--upgrade", "modal"
    )
    $Modal = Resolve-ModalExecutable -PythonExe $Python
    Invoke-NativeChecked -Executable $Modal -Arguments @("setup")
    Write-Host "MODAL P15 SETUP COMPLETE"
    exit 0
}

$Modal = Resolve-ModalExecutable -PythonExe $Python

if ($Stage -eq "Logs") {
    $AppId = Resolve-ActiveModalAppId `
        -ModalExe $Modal `
        -AppName "udsc-task2-p15-modal"
    Write-Host "FOLLOWING MODAL APP: $AppId"
    Invoke-NativeChecked -Executable $Modal -Arguments @(
        "app", "logs", $AppId, "--follow", "--timestamps"
    )
    exit 0
}

if ($Stage -eq "Stop") {
    Invoke-NativeChecked -Executable $Modal -Arguments @(
        "app", "stop", "udsc-task2-p15-modal", "--yes"
    )
    Write-Host "TASK 2 P15 MODAL APP STOPPED"
    exit 0
}

$ResolvedP14Archive = Resolve-ProjectPath -PathValue $P14Archive
$ResolvedP15Bundle = Resolve-ProjectPath -PathValue $P15Bundle
$ResolvedDownloadDir = Resolve-ProjectPath -PathValue $DownloadDir

if ($Stage -in @("Check", "Prepare", "Smoke", "Run")) {
    $P14Manifest = [System.IO.Path]::ChangeExtension(
        $ResolvedP14Archive,
        "manifest.json"
    )
    $P15Manifest = [System.IO.Path]::ChangeExtension(
        $ResolvedP15Bundle,
        "manifest.json"
    )
    foreach ($path in @(
        $ResolvedP14Archive,
        $P14Manifest,
        $ResolvedP15Bundle,
        $P15Manifest
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required P15 input is missing: $path"
        }
    }
}

if ($Stage -eq "Check") {
    Invoke-NativeChecked -Executable $Python -Arguments @(
        "-m", "pytest",
        "tests/unit/test_evaluation/test_finetune_task2_qwen_lora.py",
        "tests/unit/test_evaluation/test_legal_qa_candidates.py",
        "tests/unit/test_evaluation/test_legal_qa_scoring.py",
        "tests/unit/test_evaluation/test_prepare_task2_p15_adaptation.py",
        "tests/unit/test_evaluation/test_build_legal_qa_rankings_from_extractive.py",
        "tests/unit/test_evaluation/test_select_task2_p15_profile.py",
        "tests/unit/test_package_task2_p15_modal.py",
        "-q"
    )
}

if ($Stage -in @("Prepare", "Smoke", "Run")) {
    Assert-ModalServerReady -ModalExe $Modal
}

$RemoteStage = $Stage.ToLowerInvariant()
$ModalArguments = @("run")
if ($Stage -eq "Run") {
    $ModalArguments += "--detach"
}
$ModalArguments += @(
    $Runner,
    "--stage", $RemoteStage,
    "--p14-archive", $ResolvedP14Archive,
    "--p15-bundle", $ResolvedP15Bundle,
    "--download-dir", $ResolvedDownloadDir
)

$LogDirectory = Join-Path $ProjectRoot "artifacts\task2\modal_p15_logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ClientLog = Join-Path $LogDirectory "modal_p15_${RemoteStage}_${Timestamp}.log"
$previous = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $Modal @ModalArguments 2>&1 | Tee-Object -FilePath $ClientLog
    $ModalExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previous
}
if ($ModalExitCode -ne 0) {
    throw (
        "Modal P15 $RemoteStage failed with exit code $ModalExitCode. " +
        "Log: $ClientLog"
    )
}

if ($Stage -in @("Run", "Download")) {
    $Result = Join-Path $ResolvedDownloadDir "task2_p15_modal_result.zip"
    if (-not (Test-Path -LiteralPath $Result -PathType Leaf)) {
        throw "Modal P15 finished without a downloaded result: $Result"
    }
    Write-Host "TASK 2 P15 MODAL RESULT: $Result"
    $Submission = Join-Path $ResolvedDownloadDir "submission.zip"
    if (Test-Path -LiteralPath $Submission -PathType Leaf) {
        Write-Host "TASK 2 P15 SUBMISSION: $Submission"
    }
}
