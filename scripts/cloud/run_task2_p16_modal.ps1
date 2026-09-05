[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Setup", "Check", "Prepare", "Dev", "Run", "Status", "Logs", "Download", "Stop")]
    [string]$Stage,

    [string]$P14Archive = "artifacts\task2\task2_p14_beam_input.tar.zst",
    [string]$P15Bundle = "artifacts\task2\task2_p15_modal_input.zip",
    [string]$P16Bundle = "artifacts\task2\task2_p16_modal_input.zip",
    [string]$DownloadDir = "artifacts\task2\modal_p16_download"
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Runner = Join-Path $ProjectRoot "scripts\cloud\modal_task2_p16.py"
$Packager = Join-Path $ProjectRoot "scripts\cloud\package_task2_p16_modal.py"
$AppName = "udsc-task2-p16-modal"

function Resolve-PythonExecutable {
    foreach ($candidate in @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\python.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    throw "Python was not found. Create the repository virtual environment first."
}

function Resolve-ModalExecutable {
    param([Parameter(Mandatory = $true)][string]$PythonExe)
    foreach ($candidate in @(
        (Join-Path (Split-Path -Parent $PythonExe) "modal.exe"),
        (Join-Path $ProjectRoot ".venv\Scripts\modal.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\modal.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    $command = Get-Command modal -ErrorAction SilentlyContinue
    if ($null -ne $command) { return $command.Source }
    throw "Modal CLI is missing. Run this script with -Stage Setup."
}

function Resolve-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
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
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $previous = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $lastOutput = @(& $ModalExe app list --json 2>&1)
            $lastExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previous
        }
        $text = (($lastOutput | ForEach-Object { $_.ToString() }) -join "`n").Trim()
        $jsonValid = $false
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            try {
                $null = $text | ConvertFrom-Json -ErrorAction Stop
                $jsonValid = $true
            } catch { $jsonValid = $false }
        }
        if (($lastExitCode -eq 0) -or $jsonValid) {
            Write-Host "MODAL SERVER PREFLIGHT PASS"
            return
        }
        if ($attempt -lt 3) { Start-Sleep -Seconds (5 * $attempt) }
    }
    throw "Cannot reach Modal after 3 attempts. No GPU job was launched. Last output:`n$($lastOutput -join "`n")"
}

function Resolve-ActiveModalAppId {
    param(
        [Parameter(Mandatory = $true)][string]$ModalExe,
        [Parameter(Mandatory = $true)][string]$Name
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
    try { $apps = @($text | ConvertFrom-Json -ErrorAction Stop) }
    catch { throw "Modal app list returned invalid JSON:`n$text" }
    $active = @(
        $apps | Where-Object {
            $_.description -eq $Name -and $_.state -ne "stopped" -and [int]$_.tasks -gt 0
        } | Sort-Object -Property created_at -Descending
    )
    if ($active.Count -eq 0) {
        throw "No active Modal P16 job was found. Run -Stage Dev or -Stage Run first."
    }
    return [string]$active[0].app_id
}

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Modal P16 runner is missing: $Runner"
}
$Python = Resolve-PythonExecutable
Set-Location $ProjectRoot

if ($Stage -eq "Setup") {
    Invoke-NativeChecked -Executable $Python -Arguments @("-m", "pip", "install", "--upgrade", "modal")
    $Modal = Resolve-ModalExecutable -PythonExe $Python
    Invoke-NativeChecked -Executable $Modal -Arguments @("setup")
    Write-Host "MODAL P16 SETUP COMPLETE"
    exit 0
}

$Modal = Resolve-ModalExecutable -PythonExe $Python
if ($Stage -eq "Logs") {
    $AppId = Resolve-ActiveModalAppId -ModalExe $Modal -Name $AppName
    Write-Host "FOLLOWING MODAL APP: $AppId"
    Invoke-NativeChecked -Executable $Modal -Arguments @("app", "logs", $AppId, "--follow", "--timestamps")
    exit 0
}
if ($Stage -eq "Stop") {
    Invoke-NativeChecked -Executable $Modal -Arguments @("app", "stop", $AppName, "--yes")
    Write-Host "TASK 2 P16 MODAL APP STOPPED"
    exit 0
}

$ResolvedP14 = Resolve-ProjectPath -PathValue $P14Archive
$ResolvedP15 = Resolve-ProjectPath -PathValue $P15Bundle
$ResolvedP16 = Resolve-ProjectPath -PathValue $P16Bundle
$ResolvedDownload = Resolve-ProjectPath -PathValue $DownloadDir

if ($Stage -in @("Check", "Prepare", "Dev", "Run")) {
    Invoke-NativeChecked -Executable $Python -Arguments @(
        $Packager,
        "--p14-result", (Join-Path $ProjectRoot "artifacts\task2\modal_download\task2_p14_modal_result.zip"),
        "--output", $ResolvedP16
    )
    foreach ($path in @(
        $ResolvedP14,
        [System.IO.Path]::ChangeExtension($ResolvedP14, "manifest.json"),
        $ResolvedP15,
        [System.IO.Path]::ChangeExtension($ResolvedP15, "manifest.json"),
        $ResolvedP16,
        [System.IO.Path]::ChangeExtension($ResolvedP16, "manifest.json")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required P16 input is missing: $path"
        }
    }
}

if ($Stage -eq "Check") {
    Invoke-NativeChecked -Executable $Python -Arguments @(
        "-m", "pytest",
        "tests/unit/test_package_task2_p16_modal.py",
        "tests/unit/test_modal_task2_p16.py",
        "tests/unit/test_evaluation/test_finetune_task2_qwen_lora.py",
        "tests/unit/test_evaluation/test_select_task2_p15_profile.py",
        "-q"
    )
}
if ($Stage -in @("Prepare", "Dev", "Run")) {
    Assert-ModalServerReady -ModalExe $Modal
}

$RemoteStage = $Stage.ToLowerInvariant()
$ModalArguments = @("run")
if ($Stage -in @("Dev", "Run")) { $ModalArguments += "--detach" }
$ModalArguments += @(
    $Runner,
    "--stage", $RemoteStage,
    "--p14-archive", $ResolvedP14,
    "--p15-bundle", $ResolvedP15,
    "--p16-bundle", $ResolvedP16,
    "--download-dir", $ResolvedDownload
)
$LogDirectory = Join-Path $ProjectRoot "artifacts\task2\modal_p16_logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$ClientLog = Join-Path $LogDirectory ("modal_p16_{0}_{1}.log" -f $RemoteStage, (Get-Date -Format "yyyyMMdd_HHmmss"))
$previous = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $Modal @ModalArguments 2>&1 | Tee-Object -FilePath $ClientLog
    $exitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previous
}
if ($exitCode -ne 0) {
    throw "Modal P16 $RemoteStage failed with exit code $exitCode. Log: $ClientLog"
}
if ($Stage -in @("Dev", "Run", "Download")) {
    $Result = Join-Path $ResolvedDownload "task2_p16_modal_result.zip"
    if (-not (Test-Path -LiteralPath $Result -PathType Leaf)) {
        throw "Modal P16 finished without a downloaded result: $Result"
    }
    Write-Host "TASK 2 P16 MODAL RESULT: $Result"
    $Submission = Join-Path $ResolvedDownload "submission.zip"
    if (Test-Path -LiteralPath $Submission -PathType Leaf) {
        Write-Host "TASK 2 P16 SUBMISSION: $Submission"
    }
}
