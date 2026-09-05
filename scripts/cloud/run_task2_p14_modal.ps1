[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Setup", "Check", "Prepare", "Smoke", "Full", "Run", "Public", "Status", "Logs", "Download", "Stop")]
    [string]$Stage,

    [string]$Archive = "artifacts\task2\task2_p14_beam_input.tar.zst",

    [string]$DownloadDir = "artifacts\task2\modal_download"
)

$ErrorActionPreference = "Stop"
# Modal's Rich CLI prints Unicode status glyphs (for example U+2713). Windows
# PowerShell 5.1 switches native stdout to the legacy ANSI code page when it is
# piped through Tee-Object, which previously crashed Modal with a charmap error.
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Runner = Join-Path $ProjectRoot "scripts\cloud\modal_task2_p14.py"

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
    throw "Python was not found. Create/activate the repository virtual environment first."
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
        "Modal CLI is not installed in this environment. Run: " +
        ".\scripts\cloud\run_task2_p14_modal.ps1 -Stage Setup"
    )
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    # Windows PowerShell 5.1 wraps every native stderr line in a
    # NativeCommandError. With the script-wide Stop preference that used to
    # abort before LASTEXITCODE and the useful Modal error body were captured.
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments
        $NativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($NativeExitCode -ne 0) {
        throw "Command failed with exit code ${NativeExitCode}: $Executable $($Arguments -join ' ')"
    }
}

function Assert-ModalServerReady {
    param(
        [Parameter(Mandatory = $true)][string]$ModalExe,
        [int]$Attempts = 3
    )

    $LastProbeOutput = @()
    $LastProbeExitCode = -1
    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt++) {
        $PreviousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $LastProbeOutput = @(& $ModalExe app list --json 2>&1)
            $LastProbeExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($LastProbeExitCode -eq 0) {
            Write-Host "MODAL SERVER PREFLIGHT PASS"
            return
        }
        if ($Attempt -lt $Attempts) {
            $DelaySeconds = 5 * $Attempt
            Write-Warning (
                "Modal server preflight failed ($Attempt/$Attempts); " +
                "retrying in $DelaySeconds seconds..."
            )
            Start-Sleep -Seconds $DelaySeconds
        }
    }

    $ProbeText = ($LastProbeOutput | ForEach-Object { $_.ToString() }) -join "`n"
    throw (
        "Cannot reach the Modal server after $Attempts attempts (exit code " +
        "$LastProbeExitCode). No GPU job was launched and no GPU credit was " +
        "spent. Last Modal output:`n$ProbeText"
    )
}

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Modal Task 2 runner is missing: $Runner"
}

$Python = Resolve-PythonExecutable
Set-Location $ProjectRoot

if ($Stage -eq "Setup") {
    Invoke-NativeChecked -Executable $Python -Arguments @(
        "-m", "pip", "install", "--upgrade", "modal"
    )
    $Modal = Resolve-ModalExecutable -PythonExe $Python
    Write-Host "Modal SDK installed. Complete the browser login if Modal asks for it."
    Invoke-NativeChecked -Executable $Modal -Arguments @("setup")
    Write-Host "MODAL SETUP COMPLETE"
    exit 0
}

$Modal = Resolve-ModalExecutable -PythonExe $Python

$PythonStdoutEncoding = (& $Python -c "import sys; print(sys.stdout.encoding)").Trim()
if ($LASTEXITCODE -ne 0 -or $PythonStdoutEncoding -notmatch '^(?i:utf-?8)$') {
    throw (
        "Python UTF-8 preflight failed: stdout encoding is " +
        "'$PythonStdoutEncoding'. Refusing to invoke Modal."
    )
}
Write-Host "PYTHON UTF8 PREFLIGHT PASS: $PythonStdoutEncoding"

if ($Stage -eq "Logs") {
    Invoke-NativeChecked -Executable $Modal -Arguments @(
        "app", "logs", "udsc-task2-p14-modal", "--follow", "--timestamps"
    )
    exit 0
}

if ($Stage -eq "Stop") {
    Invoke-NativeChecked -Executable $Modal -Arguments @(
        "app", "stop", "udsc-task2-p14-modal", "--yes"
    )
    Write-Host "TASK 2 MODAL APP STOPPED"
    exit 0
}

if ($Stage -in @("Smoke", "Full", "Run", "Public")) {
    Assert-ModalServerReady -ModalExe $Modal
}

$ResolvedArchive = if ([System.IO.Path]::IsPathRooted($Archive)) {
    [System.IO.Path]::GetFullPath($Archive)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Archive))
}
$Manifest = [System.IO.Path]::ChangeExtension($ResolvedArchive, "manifest.json")
$ResolvedDownloadDir = if ([System.IO.Path]::IsPathRooted($DownloadDir)) {
    [System.IO.Path]::GetFullPath($DownloadDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $DownloadDir))
}

if ($Stage -in @("Check", "Prepare", "Smoke", "Run", "Public")) {
    if (-not (Test-Path -LiteralPath $ResolvedArchive -PathType Leaf)) {
        throw "Task 2 archive is missing: $ResolvedArchive"
    }
    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
        throw "Task 2 archive manifest is missing: $Manifest"
    }
}

if ($Stage -eq "Check") {
    Invoke-NativeChecked -Executable $Python -Arguments @(
        "-m", "pytest",
        "tests/unit/test_modal_task2_p14.py",
        "tests/unit/test_beam_task2_p14.py",
        "tests/unit/test_evaluation/test_train_legal_qa_parent_crossencoder.py",
        "-q"
    )
}

$RemoteStage = $Stage.ToLowerInvariant()
$ModalArguments = @("run")
if ($Stage -eq "Run" -or $Stage -eq "Public") {
    # Once the GPU function has been submitted, it survives a local terminal
    # disconnect. Re-run Download later if the automatic download was missed.
    $ModalArguments += "--detach"
}
$ModalArguments += @(
    $Runner,
    "--stage", $RemoteStage,
    "--archive", $ResolvedArchive,
    "--download-dir", $ResolvedDownloadDir
)
$LogDirectory = Join-Path $ProjectRoot "artifacts\task2\modal_logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ClientLog = Join-Path $LogDirectory "modal_${RemoteStage}_${Timestamp}.log"

$PreviousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $Modal @ModalArguments 2>&1 | Tee-Object -FilePath $ClientLog
    $ModalExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}
if ($ModalExitCode -ne 0) {
    $ClientOutput = Get-Content -LiteralPath $ClientLog -Raw
    $ImageIds = @(
        [regex]::Matches($ClientOutput, 'im-[A-Za-z0-9]+') |
            ForEach-Object { $_.Value } |
            Select-Object -Unique
    )
    foreach ($ImageId in $ImageIds) {
        $ImageLog = Join-Path $LogDirectory "modal_image_${ImageId}.log"
        Write-Host "Fetching failed Modal image log: $ImageId"
        $PreviousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $Modal image logs $ImageId 2>&1 |
                Tee-Object -FilePath $ImageLog
            $ImageLogExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($ImageLogExitCode -ne 0) {
            Write-Warning (
                "Unable to fetch Modal image log $ImageId " +
                "(exit code $ImageLogExitCode)."
            )
        }
    }
    throw (
        "Modal $RemoteStage failed with exit code $ModalExitCode. " +
        "Client/build logs: $LogDirectory"
    )
}

if ($Stage -eq "Run" -or $Stage -eq "Full" -or $Stage -eq "Public" -or $Stage -eq "Download") {
    $Result = Join-Path $ResolvedDownloadDir "task2_p14_modal_result.zip"
    if (-not (Test-Path -LiteralPath $Result -PathType Leaf)) {
        throw "Modal finished without a downloaded Task 2 result: $Result"
    }
    Write-Host "TASK 2 MODAL RESULT: $Result"
}
