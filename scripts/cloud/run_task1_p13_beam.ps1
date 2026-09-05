[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Check", "Prepare", "Smoke", "Full", "Status", "Logs", "Download")]
    [string]$Stage,

    [Parameter(Mandatory = $false)]
    [string]$TaskId = "",

    [Parameter(Mandatory = $false)]
    [ValidateSet("RTX4090", "RTX5090")]
    [string]$GpuType = "RTX4090"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Archive = Join-Path $ProjectRoot "artifacts\task1\task1_p13_kaggle_upload.tar.zst"
$VolumeName = "udsc-task1-p13"
$RemoteArchivePath = "$VolumeName/input/task1_p13_kaggle_upload.tar.zst"
$ExpectedArchiveBytes = 1763448793L
$ExpectedSha256 = "21DDA96ECAB20686EF81B85D1F6813D8C6FA0D80F892E43C61604A3F90469921"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:UDSC_BEAM_GPU = $GpuType
$env:MULTIPART_REQUEST_TIMEOUT = "300"
$env:MULTIPART_MAX_WORKERS = "8"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

function Resolve-BeamExecutable {
    $command = Get-Command beam -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    $fallback = Join-Path $env:USERPROFILE ".local\bin\beam.exe"
    if (Test-Path -LiteralPath $fallback -PathType Leaf) {
        return $fallback
    }
    throw "Beam CLI not found. Install/configure beam-client first."
}

function Resolve-BeamPython {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($script:BeamExe)) {
        $beamDirectory = Split-Path -Parent $script:BeamExe
        $candidates += (Join-Path $beamDirectory "python.exe")
    }
    else {
        $beamCommand = Get-Command beam -ErrorAction SilentlyContinue
        if ($null -ne $beamCommand) {
            $beamDirectory = Split-Path -Parent $beamCommand.Source
            $candidates += (Join-Path $beamDirectory "python.exe")
        }
    }
    $candidates += @(
        (Join-Path $env:APPDATA "uv\tools\beam-client\Scripts\python.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\tools\beam-client\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "Cannot find the Python environment owned by beam-client. Reinstall Beam with: uv tool install beam-client"
}

function Invoke-Beam {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    if ([string]::IsNullOrWhiteSpace($script:BeamExe)) {
        $script:BeamExe = Resolve-BeamExecutable
    }
    & $script:BeamExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Beam command failed with exit code ${LASTEXITCODE}: $Arguments"
    }
}

function Invoke-NativeLogged {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    $nativeExitCode = 1
    $writer = New-Object System.IO.StreamWriter($LogPath, $false, $script:utf8NoBom)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 wraps redirected native stderr as ErrorRecord.
        # With the script-wide Stop preference that aborts on the first log line.
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            $writer.WriteLine($line)
            $writer.Flush()
            Write-Host $line
        }
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        $writer.Dispose()
    }
    return $nativeExitCode
}

function Invoke-NativeInteractive {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Keep stdout attached to the real console so Rich can render multipart
        # progress. The compatibility launcher redacts signed URLs on failure.
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Invoke-BeamFunction {
    param([Parameter(Mandatory = $true)][ValidateSet("check", "smoke", "full")][string]$Name)
    $beamPython = Resolve-BeamPython
    $launcher = Join-Path $ProjectRoot "scripts\cloud\beam_task1_p13.py"
    $logDirectory = Join-Path $ProjectRoot "artifacts\task1"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $clientLog = Join-Path $logDirectory "beam_${Name}_client.log"
    $beamExitCode = Invoke-NativeLogged `
        -Executable $beamPython `
        -Arguments @($launcher, $Name) `
        -LogPath $clientLog
    if ($beamExitCode -ne 0) {
        throw "Beam $Name function failed with exit code ${beamExitCode}. Client log: $clientLog"
    }
}

function Get-BeamMountedArchiveProbe {
    $beamPython = Resolve-BeamPython
    $launcher = Join-Path $ProjectRoot "scripts\cloud\beam_task1_p13.py"
    $logDirectory = Join-Path $ProjectRoot "artifacts\task1"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $probeLog = Join-Path $logDirectory "beam_input_probe_client.log"
    $probeExitCode = Invoke-NativeLogged `
        -Executable $beamPython `
        -Arguments @($launcher, "input") `
        -LogPath $probeLog
    if ($probeExitCode -notin @(0, 3)) {
        throw "Beam mounted-input probe failed with exit code ${probeExitCode}. Log: $probeLog"
    }
    $prefix = "BEAM_INPUT_PROBE_JSON="
    $record = Get-Content -LiteralPath $probeLog | `
        Where-Object { $_.StartsWith($prefix) } | `
        Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($record)) {
        throw "Beam mounted-input probe returned no machine-readable result. Log: $probeLog"
    }
    return ($record.Substring($prefix.Length) | ConvertFrom-Json)
}

function Test-BeamMountedArchiveProbe {
    param([Parameter(Mandatory = $true)]$Probe)
    return (
        $Probe.status -eq "INPUT_READY" -and
        $Probe.is_file -eq $true -and
        [long]$Probe.size -eq $ExpectedArchiveBytes -and
        $Probe.sha256 -eq $ExpectedSha256
    )
}

function Ensure-BeamInputArchive {
    # GetOrCreate is idempotent and never replaces an existing volume.
    Invoke-Beam volume create $VolumeName
    # stat_path can cache the size of an overwritten V1 object. Probe the file
    # through the actual /mnt/task1 mount and verify its SHA before deciding.
    $before = Get-BeamMountedArchiveProbe
    if (Test-BeamMountedArchiveProbe -Probe $before) {
        Write-Host "Beam input already verified: $RemoteArchivePath ($ExpectedArchiveBytes bytes)" -ForegroundColor Green
        return
    }

    if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
        throw "Beam input is absent or invalid, and the local archive is missing: $Archive"
    }
    $localLength = (Get-Item -LiteralPath $Archive).Length
    if ($localLength -ne $ExpectedArchiveBytes) {
        throw "Local archive size mismatch. Expected $ExpectedArchiveBytes bytes, got $localLength"
    }
    $observed = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash
    if ($observed -ne $ExpectedSha256) {
        throw "Archive SHA-256 mismatch. Expected $ExpectedSha256, got $observed"
    }

    # A partial file written by Beam's non-resumable V1 streamer can prevent a
    # multipart completion from replacing the target. Remove only the exact
    # invalid file after the mounted probe has proved it is not usable.
    if ($before.is_file -eq $true) {
        Write-Host "Removing invalid partial Beam input ($($before.size) bytes)..." -ForegroundColor Yellow
        Invoke-Beam rm $RemoteArchivePath
    }

    # The stock multipart CLI uses os.path.join() and sends backslashes from
    # Windows. Our compatibility launcher keeps the remote path POSIX while
    # retaining Beam's parallel multipart upload, which is much faster than
    # the V1 streaming fallback for this 1.76 GB archive.
    $beamPython = Resolve-BeamPython
    $compatibilityCli = Join-Path $ProjectRoot "scripts\cloud\beam_windows_cli.py"
    $uploadExitCode = Invoke-NativeInteractive `
        -Executable $beamPython `
        -Arguments @(
            $compatibilityCli,
            "cp",
            $Archive,
            "beam://$RemoteArchivePath"
        )
    if ($uploadExitCode -ne 0) {
        throw "Beam multipart upload failed with exit code ${uploadExitCode}."
    }

    $after = Get-BeamMountedArchiveProbe
    if (-not (Test-BeamMountedArchiveProbe -Probe $after)) {
        $observedProbe = $after | ConvertTo-Json -Compress
        throw "Beam upload finished but mounted archive verification failed: $observedProbe"
    }
    Write-Host "Beam input uploaded and verified: $RemoteArchivePath ($ExpectedArchiveBytes bytes)" -ForegroundColor Green
}

$BeamExe = $null
Set-Location -LiteralPath $ProjectRoot

switch ($Stage) {
    "Check" {
        Invoke-BeamFunction check
    }
    "Prepare" {
        Ensure-BeamInputArchive
        Write-Host "BEAM PREPARE PASS" -ForegroundColor Green
    }
    "Smoke" {
        Ensure-BeamInputArchive
        Invoke-BeamFunction smoke
    }
    "Full" {
        Ensure-BeamInputArchive
        Invoke-BeamFunction full
    }
    "Status" {
        Invoke-Beam task list --limit 20
        # Listing the volume root is valid before the first job creates results/.
        Invoke-Beam ls $VolumeName
    }
    "Logs" {
        if ([string]::IsNullOrWhiteSpace($TaskId)) {
            throw "-TaskId is required for -Stage Logs. Get it from Stage Status or the failed function output."
        }
        $logDirectory = Join-Path $ProjectRoot "artifacts\task1"
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        $remoteLog = Join-Path $logDirectory "beam_task_${TaskId}.log"
        if ([string]::IsNullOrWhiteSpace($script:BeamExe)) {
            $script:BeamExe = Resolve-BeamExecutable
        }
        $beamExitCode = Invoke-NativeLogged `
            -Executable $script:BeamExe `
            -Arguments @("logs", "--task-id", $TaskId, "-n", "300", "--show-timestamp") `
            -LogPath $remoteLog
        if ($beamExitCode -ne 0) {
            throw "Beam realtime logs are unavailable from this client (Beam native Windows may fail TLS). Client log: $remoteLog. Smoke/Full now return guarded tracebacks directly."
        }
        Write-Host "Saved Beam task log: $remoteLog" -ForegroundColor Green
    }
    "Download" {
        $destination = Join-Path $ProjectRoot "artifacts\task1\beam_results"
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        $result = Join-Path $destination "task1_p13_beam_result.zip"
        $partialResult = "${result}.${PID}.partial"
        if (Test-Path -LiteralPath $partialResult) {
            Remove-Item -LiteralPath $partialResult -Force
        }
        $beamPython = Resolve-BeamPython
        $compatibilityCli = Join-Path $ProjectRoot "scripts\cloud\beam_windows_cli.py"
        $validator = Join-Path $ProjectRoot "scripts\cloud\validate_beam_result.py"
        $downloadLog = Join-Path $destination "beam_download_client.log"
        $beamExitCode = Invoke-NativeLogged `
            -Executable $beamPython `
            -Arguments @(
                $compatibilityCli,
                "cp",
                "beam://$VolumeName/results/task1_p13_beam_result.zip",
                $partialResult
            ) `
            -LogPath $downloadLog
        if ($beamExitCode -ne 0) {
            Remove-Item -LiteralPath $partialResult -Force -ErrorAction SilentlyContinue
            throw "Beam result download failed with exit code ${beamExitCode}. Log: $downloadLog"
        }

        $validationLog = Join-Path $destination "beam_download_validation.log"
        $validationExitCode = Invoke-NativeLogged `
            -Executable $beamPython `
            -Arguments @($validator, $partialResult) `
            -LogPath $validationLog
        if ($validationExitCode -ne 0) {
            Remove-Item -LiteralPath $partialResult -Force -ErrorAction SilentlyContinue
            throw "Downloaded Beam result failed validation. Log: $validationLog"
        }
        try {
            if (Test-Path -LiteralPath $result -PathType Leaf) {
                [System.IO.File]::Replace($partialResult, $result, $null)
            }
            else {
                [System.IO.File]::Move($partialResult, $result)
            }
        }
        catch {
            Remove-Item -LiteralPath $partialResult -Force -ErrorAction SilentlyContinue
            throw
        }
        Write-Host "Downloaded: $result" -ForegroundColor Green
    }
}
