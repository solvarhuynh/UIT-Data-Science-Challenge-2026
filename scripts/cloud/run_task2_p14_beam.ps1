[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Check", "Capacity", "Guard", "ReplacePending", "Prepare", "Smoke", "Full", "Run", "Status", "Logs", "Stop", "Download")]
    [string]$Stage,

    [Parameter(Mandatory = $false)]
    [ValidateSet("FastServerless", "A10G", "RTX4090", "RTX5090", "H100")]
    [string]$GpuType = "FastServerless",

    [Parameter(Mandatory = $false)]
    [string]$PoolName = "",

    [Parameter(Mandatory = $false)]
    [string]$TaskId = "",

    [Parameter(Mandatory = $false)]
    [switch]$Rebuild,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 6)]
    [int]$GpuAttempts = 6
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Archive = Join-Path $ProjectRoot "artifacts\task2\task2_p14_beam_input.tar.zst"
$Manifest = Join-Path $ProjectRoot "artifacts\task2\task2_p14_beam_input.tar.manifest.json"
$LaunchManifest = Join-Path $ProjectRoot "artifacts\task2\beam_launch_manifest.json"
$VolumeName = "udsc-task2-p14"
$RemoteArchive = "$VolumeName/input/task2_p14_beam_input.tar.zst"
$RemoteManifest = "$VolumeName/input/task2_p14_beam_input.tar.manifest.json"
$ExpectedRunContractVersion = "p14-fast-v4"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$FastServerlessGpuOrder = @("RTX5090", "RTX4090", "A10G")
$GpuRequest = if ($GpuType -eq "FastServerless") {
    $FastServerlessGpuOrder -join ","
}
else {
    $GpuType
}
if (
    $GpuType -eq "H100" -and
    $Stage -in @("Smoke", "Full", "Run") -and
    [string]::IsNullOrWhiteSpace($PoolName)
) {
    throw (
        "H100 is Beam on-demand hardware, not a serverless GPU. " +
        "Reserve a named pool first, for example: " +
        "beam machine reserve --gpu H100 --ttl 6h --name udsc-task2-h100 --yes; " +
        "then rerun with -GpuType H100 -PoolName udsc-task2-h100."
    )
}
$env:UDSC_BEAM_GPU = $GpuRequest
if ([string]::IsNullOrWhiteSpace($PoolName)) {
    Remove-Item Env:UDSC_BEAM_POOL -ErrorAction SilentlyContinue
}
else {
    $env:UDSC_BEAM_POOL = $PoolName.Trim()
}
$env:MULTIPART_REQUEST_TIMEOUT = "300"
$env:MULTIPART_MAX_WORKERS = "8"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom

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
    $beamExecutable = Resolve-BeamExecutable
    $candidates = @(
        (Join-Path (Split-Path -Parent $beamExecutable) "python.exe"),
        (Join-Path $env:APPDATA "uv\tools\beam-client\Scripts\python.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\tools\beam-client\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "Cannot find Beam Python. Reinstall with: uv tool install beam-client"
}

function Invoke-Beam {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $beamExecutable = Resolve-BeamExecutable
    $priorPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        # Windows PowerShell 5.1 can turn a harmless native stderr warning into
        # a terminating NativeCommandError when the global preference is Stop.
        $ErrorActionPreference = "Continue"
        & $beamExecutable @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorPreference
    }
    if ($exitCode -ne 0) {
        throw "Beam command failed with exit code ${exitCode}: $Arguments"
    }
}

function Get-TerminalTaskIdsFromContent {
    param([Parameter(Mandatory = $true)][string]$Content)
    $ansiEscape = ([char]27) + '\[[0-?]*[ -/]*[@-~]'
    $normalized = [regex]::Replace($Content, $ansiEscape, "")
    return @(
        [regex]::Matches(
            $normalized,
            '(?im)^\s*(?:(?:=>)|(?:[^\x00-\x7f]{1,4}))?\s*Function\s+(?:failed|complete)\s+<([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})>\s*$'
        ) | ForEach-Object { $_.Groups[1].Value.ToLowerInvariant() }
    )
}

function Get-CanonicalBeamTaskId {
    param([AllowNull()][object]$Value)
    $match = [regex]::Match(
        [string]$Value,
        '(?i)(?<![0-9a-f])([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?![0-9a-f])'
    )
    if (-not $match.Success) {
        return $null
    }
    return $match.Groups[1].Value.ToLowerInvariant()
}

function Resolve-ExactBeamTaskId {
    param([AllowNull()][object]$Value)
    $trimmed = ([string]$Value).Trim()
    $canonical = Get-CanonicalBeamTaskId -Value $trimmed
    if ($null -eq $canonical -or $trimmed.ToLowerInvariant() -ne $canonical) {
        throw "Task ID must be one exact UUID copied from Beam, without < > or extra text."
    }
    return $canonical
}

function Get-LocallyConfirmedTerminalTaskMap {
    $logRoot = Join-Path $ProjectRoot "artifacts\task2\beam_logs"
    if (-not (Test-Path -LiteralPath $logRoot -PathType Container)) {
        return ,@{}
    }
    $terminalIds = @{}
    Get-ChildItem -LiteralPath $logRoot -Filter "*.log" -File | ForEach-Object {
        try {
            $content = Get-Content -LiteralPath $_.FullName -Raw -ErrorAction Stop
        }
        catch {
            # Missing/unreadable evidence must fail closed: the corresponding
            # task remains blocking if Beam still reports it as active.
            return
        }
        foreach ($taskId in @(Get-TerminalTaskIdsFromContent -Content $content)) {
            $terminalIds[$taskId] = $true
        }
    }
    # The unary comma prevents PowerShell from unrolling the dictionary into
    # pipeline entries. Callers can then use exact ContainsKey lookups.
    return ,$terminalIds
}

function Get-LocallyConfirmedTerminalTaskIds {
    $terminalTaskMap = Get-LocallyConfirmedTerminalTaskMap
    return @($terminalTaskMap.Keys)
}

function ConvertFrom-BeamTaskListJson {
    param([Parameter(Mandatory = $true)][string]$Json)
    try {
        $parsedTasks = $Json | ConvertFrom-Json
    }
    catch {
        throw "Beam returned invalid task-list JSON; refusing to risk a duplicate job."
    }
    # Windows PowerShell 5.1 can return the top-level JSON array as one nested
    # System.Object[] pipeline object. Pipe it once more to flatten jobs.
    return @($parsedTasks | ForEach-Object { $_ })
}

function Get-Task2BeamJobState {
    $beamExecutable = Resolve-BeamExecutable
    $stderrPath = [System.IO.Path]::GetTempFileName()
    $priorPreference = $ErrorActionPreference
    $exitCode = 1
    $stdout = @()
    try {
        # Keep stderr out of the JSON stream. Beam may print a capacity/version
        # warning while still returning valid JSON and exit code zero.
        $ErrorActionPreference = "Continue"
        $stdout = @(
            & $beamExecutable task list `
                --limit 1000 `
                --filter "status=running,pending,retry" `
                --format json 2> $stderrPath
        )
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorPreference
    }
    $stderr = ""
    try {
        if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
            $stderr = Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue
        }
    }
    finally {
        Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    }
    if ($exitCode -ne 0) {
        throw "Cannot verify active Beam tasks before launch (exit=$exitCode): $stderr"
    }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        Write-Warning ("Beam task-list warning: " + $stderr.Trim())
    }
    $tasks = @(
        ConvertFrom-BeamTaskListJson -Json ($stdout -join [Environment]::NewLine)
    )
    $task2Jobs = @(
        $tasks | Where-Object {
            $_.stub_name -match "scripts\.cloud\.beam_task2_p14:(smoke|full)"
        }
    )
    $terminalTaskMap = Get-LocallyConfirmedTerminalTaskMap
    $stale = @(
        $task2Jobs | Where-Object {
            $candidateId = Get-CanonicalBeamTaskId -Value $_.id
            $null -ne $candidateId -and
            $terminalTaskMap.ContainsKey($candidateId)
        }
    )
    if ($stale.Count -gt 0) {
        $staleIds = ($stale | ForEach-Object { $_.id }) -join ", "
        Write-Warning (
            "Beam still reports locally confirmed terminal task(s) as active " +
            "($staleIds); ignoring the stale task-list rows."
        )
    }
    $active = @(
        $task2Jobs | Where-Object {
            $candidateId = Get-CanonicalBeamTaskId -Value $_.id
            -not (
                $null -ne $candidateId -and
                $terminalTaskMap.ContainsKey($candidateId)
            )
        }
    )
    Write-Host (
        "Task 2 duplicate guard: reported=$($task2Jobs.Count), " +
        "locally_terminal=$($stale.Count), blocking=$($active.Count)"
    ) -ForegroundColor DarkGray
    return [pscustomobject]@{
        Reported = $task2Jobs
        Stale = $stale
        Active = $active
    }
}

function Assert-NoActiveTask2Job {
    $state = Get-Task2BeamJobState
    $active = @($state.Active)
    if ($active.Count -gt 0) {
        $ids = ($active | ForEach-Object { $_.id }) -join ", "
        throw (
            "A Task 2 Beam job is already pending/running ($ids). " +
            "Do not launch a duplicate. Use -Stage Status, then inspect/stop the " +
            "existing task deliberately if needed."
        )
    }
}

function Stop-PendingTask2Jobs {
    $state = Get-Task2BeamJobState
    $active = @($state.Active)
    $running = @(
        $active | Where-Object { ([string]$_.status).Trim().ToLowerInvariant() -eq "running" }
    )
    $unknown = @(
        $active | Where-Object {
            ([string]$_.status).Trim().ToLowerInvariant() -notin @("pending", "running")
        }
    )
    if ($running.Count -gt 0 -or $unknown.Count -gt 0) {
        $protected = @($running + $unknown)
        $ids = ($protected | ForEach-Object { $_.id }) -join ", "
        throw (
            "Refusing to replace a Task 2 job that is already running or has an " +
            "unknown state ($ids). Stop it deliberately with -Stage Stop -TaskId <real-id>."
        )
    }

    $pending = @(
        $active | Where-Object { ([string]$_.status).Trim().ToLowerInvariant() -eq "pending" }
    )
    if ($pending.Count -eq 0) {
        Write-Host "No genuine pending Task 2 Beam job needs replacement." -ForegroundColor Green
        return
    }

    foreach ($job in $pending) {
        $canonicalId = Get-CanonicalBeamTaskId -Value $job.id
        if ($null -eq $canonicalId) {
            throw "Beam returned a pending Task 2 row with an invalid task ID; refusing unsafe cleanup."
        }
        Invoke-Beam task stop $canonicalId
        Write-Host "Requested stop for pending Task 2 job: $canonicalId" -ForegroundColor Yellow
    }

    for ($poll = 1; $poll -le 12; $poll++) {
        Start-Sleep -Seconds 5
        $remaining = @((Get-Task2BeamJobState).Active)
        if ($remaining.Count -eq 0) {
            Write-Host "Pending Task 2 job replacement complete." -ForegroundColor Green
            return
        }
        Write-Host (
            "Waiting for Beam to finalize stopped Task 2 job(s) ($poll/12)..."
        ) -ForegroundColor DarkGray
    }
    $remainingIds = (@((Get-Task2BeamJobState).Active) | ForEach-Object { $_.id }) -join ", "
    throw (
        "Beam did not finalize the stopped pending Task 2 job(s) within 60 seconds " +
        "($remainingIds). No paid H100 should be reserved yet."
    )
}

function Invoke-NativeLogged {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $writer = New-Object System.IO.StreamWriter($LogPath, $false, $script:Utf8NoBom)
    $priorPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            $writer.WriteLine($line)
            $writer.Flush()
            Write-Host $line
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorPreference
        $writer.Dispose()
    }
    return $exitCode
}

function Invoke-BeamFunction {
    param([Parameter(Mandatory = $true)][ValidateSet("check", "source", "input", "model", "smoke", "full")][string]$Name)
    $beamPython = Resolve-BeamPython
    $launcher = Join-Path $ProjectRoot "scripts\cloud\beam_task2_p14.py"
    $logRoot = Join-Path $ProjectRoot "artifacts\task2\beam_logs"
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $logPath = Join-Path $logRoot "beam_${Name}_client.log"
    $exitCode = Invoke-NativeLogged `
        -Executable $beamPython `
        -Arguments @($launcher, $Name) `
        -LogPath $logPath
    return [pscustomobject]@{ ExitCode = $exitCode; LogPath = $logPath }
}

function Test-UnknownDeviceAllocationFailure {
    param([Parameter(Mandatory = $true)][string]$Content)
    # Beam/Rich may hard-wrap "setting up nvproxy" across physical lines.
    # Match the stable container-runtime signature across newlines instead.
    return (
        $Content -match '(?is)(creating container|setting\s+up\s+nvproxy)' -and
        $Content -match '(?is)nvidia-container-cli(?:\.real)?.{0,2048}unknown device'
    )
}

function Get-BeamFailureHint {
    param([Parameter(Mandatory = $true)][string]$LogPath)
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        return "Beam client log is missing."
    }
    $content = Get-Content -LiteralPath $LogPath -Raw
    if (Test-UnknownDeviceAllocationFailure -Content $content) {
        return "Beam GPU host allocation failed before Task 2 started. The input and code are not at fault; retry this stage once to request another GPU node."
    }
    if ($content -match "Beam Task 2 invocation failed: No module named 'torch'") {
        return "Beam completed the GPU function but the Windows client could not deserialize a non-portable Torch version value. Update the Task 2 runner before retrying; the Qwen Volume cache can be reused."
    }
    if ($content -match "Beam GPU is already occupied before Task 2 starts") {
        return "The allocated Beam GPU was already occupied before Task 2 started. Retry on another GPU type/node; reducing the Task 2 batch size cannot recover VRAM held by another process."
    }
    if ($content -match "CUDA out of memory") {
        return "Task 2 itself exhausted GPU memory after preflight. Use a larger GPU profile or lower the stage batch; this is different from a dirty Beam host."
    }
    if (
        $content -match "Function complete" -and
        $content -match "Beam returned no completed Task 2 result" -and
        $content -notmatch '"exception_type"'
    ) {
        return "The remote Task 2 function was interrupted or cancelled before it could return a result. This commonly follows Stop/cancel or a lost Beam worker; it is not a model traceback. Run the patched Smoke stage before starting a fresh Full task."
    }
    if ($content -match "Beam returned no completed Task 2 result") {
        return "Beam did not start or complete the remote function; inspect the final lines of the client log."
    }
    return "Inspect the final lines of the Beam client log for the remote exception."
}

function Test-RetryableGpuAllocationFailure {
    param([Parameter(Mandatory = $true)][string]$LogPath)
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        return $false
    }
    $content = Get-Content -LiteralPath $LogPath -Raw
    return Test-RetryableGpuAllocationFailureContent -Content $content
}

function Test-RetryableGpuAllocationFailureContent {
    param([Parameter(Mandatory = $true)][string]$Content)
    if ($Content -match "Beam GPU is already occupied before Task 2 starts") {
        return $true
    }
    if (Test-UnknownDeviceAllocationFailure -Content $Content) {
        return $true
    }
    # Beam can fail a low-capacity serverless request before the Python handler
    # starts (sometimes before container creation, sometimes just after the
    # cached image loads). There is then no remote marker/JSON/traceback: only
    # the capacity warning, terminal Function failed ID, and no result.
    if (
        $Content -match '(?i)GPU capacity for .+ is currently low' -and
        $Content -match '(?i)Function failed\s+<[0-9a-f-]{36}>' -and
        $Content -match '(?i)Beam returned no completed Task 2 result' -and
        $Content -notmatch '(?i)REMOTE_STAGE_START|Qwen cache ready|GPU_PREFLIGHT_JSON' -and
        $Content -notmatch '(?i)Traceback|"exception_type"|PROCESS EXIT|CUDA out of memory'
    ) {
        return $true
    }
    # Code/model/data failures, CUDA OOM, and generic client/worker interruption
    # are not proven allocation failures. Fail closed instead of changing the
    # user's GPU request or risking a duplicate headless job.
    return $false
}

function Get-GpuAttemptPlan {
    # Named pools have a fixed hardware contract. Do not silently send a pool
    # request to a different GPU class, especially for an explicitly reserved
    # H100. Serverless requests can rotate classes after an infrastructure
    # allocation failure so a retry does not keep landing on the same broken
    # nvproxy/nvidia-container runtime path.
    if (
        $GpuType -eq "H100" -or
        -not [string]::IsNullOrWhiteSpace($PoolName)
    ) {
        return @($GpuRequest)
    }
    if ($GpuType -eq "FastServerless") {
        # Give Beam every supported serverless class on every launch.  The SDK
        # treats a comma-separated request as an ordered priority list and can
        # therefore schedule 4090/A10G immediately when 5090 has no capacity.
        # Retrying rotates the priority only after a proven dirty-host/nvproxy
        # failure; it never waits on one GPU class before exposing the others.
        $rotations = @()
        for ($offset = 0; $offset -lt $FastServerlessGpuOrder.Count; $offset++) {
            $ordered = @()
            for ($index = 0; $index -lt $FastServerlessGpuOrder.Count; $index++) {
                $ordered += $FastServerlessGpuOrder[
                    ($offset + $index) % $FastServerlessGpuOrder.Count
                ]
            }
            $rotations += ,($ordered -join ",")
        }
        return $rotations
    }
    # An explicit GPU selection is a contract, not merely a preference. Only
    # FastServerless is authorized to rotate hardware classes.
    return @($GpuType)
}

function Invoke-GpuStage {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("smoke", "full")]
        [string]$Name
    )
    $attemptPlan = @(Get-GpuAttemptPlan)
    if ($attemptPlan.Count -lt 1) {
        throw "Task 2 Beam GPU attempt plan is empty"
    }
    for ($attempt = 1; $attempt -le $GpuAttempts; $attempt++) {
        $attemptGpuRequest = $attemptPlan[($attempt - 1) % $attemptPlan.Count]
        $env:UDSC_BEAM_GPU = $attemptGpuRequest
        $nonce = [Guid]::NewGuid().ToString("N").Substring(0, 12)
        $env:UDSC_BEAM_ALLOCATION_NONCE = $nonce
        Write-Host "Requesting fresh Beam GPU for $Name ($attempt/$GpuAttempts, profile=$attemptGpuRequest, nonce=$nonce)..." -ForegroundColor Cyan
        $result = Invoke-BeamFunction $Name
        $attemptLog = Join-Path `
            (Split-Path -Parent $result.LogPath) `
            "beam_${Name}_attempt_${attempt}_${nonce}.log"
        Copy-Item -LiteralPath $result.LogPath -Destination $attemptLog -Force
        if ($result.ExitCode -eq 0) {
            return $result
        }
        $retryable = Test-RetryableGpuAllocationFailure -LogPath $result.LogPath
        if ($retryable -and $attempt -lt $GpuAttempts) {
            $nextGpuRequest = $attemptPlan[$attempt % $attemptPlan.Count]
            Write-Host "Beam returned a broken/occupied GPU allocation; rotating $attemptGpuRequest -> $nextGpuRequest for the next fresh node..." -ForegroundColor Yellow
            if ($nextGpuRequest -ne $attemptGpuRequest) {
                # A different GPU class uses a different scheduler pool; a long
                # same-host cooldown only wastes time here.
                $backoff = 8 + (Get-Random -Minimum 0 -Maximum 5)
                Write-Host "Waiting ${backoff}s before switching Beam GPU pools..." -ForegroundColor Yellow
            }
            else {
                $backoff = [Math]::Min(90, (30 * [Math]::Pow(2, $attempt - 1)))
                $backoff = [int]$backoff + (Get-Random -Minimum 0 -Maximum 6)
                Write-Host "Waiting ${backoff}s so Beam does not return the same host..." -ForegroundColor Yellow
            }
            Start-Sleep -Seconds $backoff
            continue
        }
        $hint = Get-BeamFailureHint -LogPath $result.LogPath
        throw "Task 2 Beam $Name failed after attempt $attempt/$GpuAttempts. $hint Log: $($result.LogPath)"
    }
    throw "Task 2 Beam $Name exhausted all $GpuAttempts GPU attempts"
}

function Build-InputArchive {
    if (
        -not $Rebuild -and
        (Test-Path -LiteralPath $Archive -PathType Leaf) -and
        (Test-Path -LiteralPath $Manifest -PathType Leaf)
    ) {
        Write-Host "Using existing Task 2 archive: $Archive" -ForegroundColor Green
        return
    }
    $python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $python = "python"
    }
    $arguments = @(
        "scripts/cloud/package_task2_p14_beam.py",
        "--output", $Archive,
        "--compression-level", "10"
    )
    if ($Rebuild -or (Test-Path -LiteralPath $Archive -PathType Leaf)) {
        $arguments += "--force"
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Task 2 input packaging failed with exit code $LASTEXITCODE"
    }
}

function Get-LocalManifest {
    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
        throw "Task 2 input manifest is missing: $Manifest"
    }
    $record = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
    $observedSize = (Get-Item -LiteralPath $Archive).Length
    $observedHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([long]$record.archive_bytes -ne $observedSize) {
        throw "Local Task 2 archive size does not match its manifest"
    }
    if ($record.archive_sha256 -ne $observedHash) {
        throw "Local Task 2 archive SHA-256 does not match its manifest"
    }
    return $record
}

function Get-LocalSourceHash {
    $result = Invoke-BeamFunction source
    if ($result.ExitCode -ne 0) {
        throw "Cannot calculate the local Task 2 source hash. Log: $($result.LogPath)"
    }
    $prefix = "LOCAL_SOURCE_SHA256="
    $record = Get-Content -LiteralPath $result.LogPath | `
        Where-Object { $_.StartsWith($prefix) } | `
        Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($record)) {
        throw "Task 2 source-hash output is missing"
    }
    return $record.Substring($prefix.Length)
}

function Write-BeamLaunchManifest {
    $local = Get-LocalManifest
    $record = [ordered]@{
        schema_version = "task2-p14-beam-launch-v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        archive_sha256 = $local.archive_sha256
        source_sha256 = Get-LocalSourceHash
        contract_version = $ExpectedRunContractVersion
        gpu_request = $GpuRequest
        gpu_attempt_plan = @(Get-GpuAttemptPlan)
        pool_name = if ([string]::IsNullOrWhiteSpace($PoolName)) { $null } else { $PoolName }
    }
    $parent = Split-Path -Parent $LaunchManifest
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = "${LaunchManifest}.${PID}.partial"
    [System.IO.File]::WriteAllText(
        $temporary,
        (($record | ConvertTo-Json -Depth 5) + [Environment]::NewLine),
        $Utf8NoBom
    )
    Move-Item -LiteralPath $temporary -Destination $LaunchManifest -Force
    Write-Host "Recorded exact Beam launch contract: $LaunchManifest" -ForegroundColor Green
}

function Get-BeamLaunchManifest {
    if (-not (Test-Path -LiteralPath $LaunchManifest -PathType Leaf)) {
        throw (
            "Beam launch manifest is missing: $LaunchManifest. " +
            "Run Smoke/Full/Run with this wrapper before downloading."
        )
    }
    $record = Get-Content -LiteralPath $LaunchManifest -Raw | ConvertFrom-Json
    if ($record.schema_version -ne "task2-p14-beam-launch-v1") {
        throw "Beam launch manifest has an unsupported schema"
    }
    return $record
}

function Copy-ToBeam {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$RemotePath
    )
    $beamPython = Resolve-BeamPython
    $compatibilityCli = Join-Path $ProjectRoot "scripts\cloud\beam_windows_cli.py"
    & $beamPython $compatibilityCli cp $LocalPath "beam://$RemotePath"
    if ($LASTEXITCODE -ne 0) {
        throw "Beam multipart upload failed for $LocalPath"
    }
}

function Get-RemoteInputState {
    $result = Invoke-BeamFunction input
    $prefix = "BEAM_INPUT_PROBE_JSON="
    $record = Get-Content -LiteralPath $result.LogPath | `
        Where-Object { $_.StartsWith($prefix) } | `
        Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($record)) {
        return [pscustomobject]@{ status = "INPUT_INVALID"; reason = "probe_missing" }
    }
    return ($record.Substring($prefix.Length) | ConvertFrom-Json)
}

function Ensure-BeamInput {
    Build-InputArchive
    $local = Get-LocalManifest
    Invoke-Beam volume create $VolumeName
    $remote = Get-RemoteInputState
    if (
        $remote.status -eq "INPUT_READY" -and
        [long]$remote.size -eq [long]$local.archive_bytes -and
        $remote.sha256 -eq $local.archive_sha256
    ) {
        Write-Host "Beam Task 2 input already verified; upload skipped." -ForegroundColor Green
        return
    }
    Write-Host "Uploading compressed Task 2 archive once..." -ForegroundColor Cyan
    Copy-ToBeam -LocalPath $Archive -RemotePath $RemoteArchive
    Copy-ToBeam -LocalPath $Manifest -RemotePath $RemoteManifest
    for ($attempt = 1; $attempt -le 6; $attempt++) {
        $remote = Get-RemoteInputState
        if (
            $remote.status -eq "INPUT_READY" -and
            [long]$remote.size -eq [long]$local.archive_bytes -and
            $remote.sha256 -eq $local.archive_sha256
        ) {
            Write-Host "BEAM TASK 2 INPUT READY" -ForegroundColor Green
            return
        }
        Write-Host "Waiting for Beam Volume propagation ($attempt/6)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
    throw "Beam Task 2 upload finished but mounted hash verification did not pass"
}

function Ensure-BeamModel {
    Write-Host "Verifying pinned Qwen cache on a CPU worker..." -ForegroundColor Cyan
    $result = Invoke-BeamFunction model
    if ($result.ExitCode -ne 0) {
        throw "Task 2 pinned Qwen cache preparation failed. Log: $($result.LogPath)"
    }
    Write-Host "BEAM TASK 2 MODEL READY" -ForegroundColor Green
}

function Receive-BeamResult {
    $local = Get-LocalManifest
    $expected = Get-BeamLaunchManifest
    if ($expected.archive_sha256 -ne $local.archive_sha256) {
        throw (
            "Local Task 2 input changed after the recorded Beam launch. " +
            "Run the pipeline again instead of downloading a stale result."
        )
    }
    $destination = Join-Path $ProjectRoot "artifacts\task2\beam_results"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    $resultPath = Join-Path $destination "task2_p14_beam_result.zip"
    $partial = "${resultPath}.${PID}.partial"
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    $beamPython = Resolve-BeamPython
    $compatibilityCli = Join-Path $ProjectRoot "scripts\cloud\beam_windows_cli.py"
    & $beamPython $compatibilityCli cp `
        "beam://$VolumeName/results/task2_p14_beam_result.zip" `
        $partial
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        throw "Task 2 Beam result download failed"
    }
    $summary = $null
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [System.IO.Compression.ZipFile]::OpenRead($partial)
        try {
            $summaryEntry = $zip.Entries | `
                Where-Object { $_.FullName -match "(^|/)full_run_summary\.json$" } | `
                Select-Object -First 1
            if ($null -eq $summaryEntry) {
                throw "Downloaded Task 2 archive has no full_run_summary.json"
            }
            $reader = [System.IO.StreamReader]::new(
                $summaryEntry.Open(),
                [System.Text.Encoding]::UTF8,
                $true
            )
            try {
                $summary = $reader.ReadToEnd() | ConvertFrom-Json
            }
            finally {
                $reader.Dispose()
            }
            if ($summary.status -notin @("PUBLIC_CANDIDATE_READY", "HELDOUT_REJECTED")) {
                throw "Downloaded Task 2 summary has invalid status: $($summary.status)"
            }
            if ($summary.archive_sha256 -ne $expected.archive_sha256) {
                throw (
                    "Downloaded Task 2 result belongs to a different input archive " +
                    "($($summary.archive_sha256)); expected $($expected.archive_sha256)."
                )
            }
            if ($summary.contract_version -ne $expected.contract_version) {
                throw (
                    "Downloaded Task 2 result uses stale contract " +
                    "$($summary.contract_version); expected $($expected.contract_version)."
                )
            }
            if ($summary.source_sha256 -ne $expected.source_sha256) {
                throw (
                    "Downloaded Task 2 result belongs to different source code " +
                    "($($summary.source_sha256)); expected $($expected.source_sha256)."
                )
            }
        }
        finally {
            $zip.Dispose()
        }
    }
    catch {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        throw
    }
    Move-Item -LiteralPath $partial -Destination $resultPath -Force
    Write-Host "Downloaded and validated: $resultPath" -ForegroundColor Green
    return [pscustomobject]@{
        Path = $resultPath
        Status = $summary.status
        HeldoutMeteor = $summary.heldout.meteor
    }
}

Set-Location -LiteralPath $ProjectRoot

switch ($Stage) {
    "Check" {
        $wrappedUnknownDeviceFixture = @"
creating container: cannot create gofer process: setting up
nvproxy for gofer: nvidia-container-cli configure failed
stderr: nvidia-container-cli.real: device error: 7: unknown device
"@
        if (-not (Test-UnknownDeviceAllocationFailure -Content $wrappedUnknownDeviceFixture)) {
            throw "Task 2 Beam allocation-error classifier self-test failed"
        }
        if (-not (Test-RetryableGpuAllocationFailureContent -Content $wrappedUnknownDeviceFixture)) {
            throw "Task 2 Beam retryable-allocation classifier self-test failed"
        }
        $lowCapacityBeforeContainerFixture = @"
! GPU capacity for A10G is currently low.
=> Running function: <scripts.cloud.beam_task2_p14:smoke>
Loading image <d7f5395222bdfa75>...
Loaded image <d7f5395222bdfa75>, took: 2.68us
Function failed <3619d360-d903-44af-b5bb-2f1584dcf2ab>
Beam returned no completed Task 2 result
"@
        if (-not (Test-RetryableGpuAllocationFailureContent -Content $lowCapacityBeforeContainerFixture)) {
            throw "Task 2 Beam pre-container capacity classifier self-test failed"
        }
        $capacityWarningWithApplicationFailureFixture = @"
! GPU capacity for A10G is currently low.
Loading image <abc>
Loaded image <abc>
Traceback: application failed
Function failed <3619d360-d903-44af-b5bb-2f1584dcf2ab>
Beam returned no completed Task 2 result
"@
        if (Test-RetryableGpuAllocationFailureContent -Content $capacityWarningWithApplicationFailureFixture) {
            throw "Task 2 Beam capacity classifier accepted an application failure"
        }
        $expectedTerminalTaskId = "5d73a8fd-5a1d-4bc0-b17d-4d4dec898692"
        $terminalTaskFixtures = @(
            "Function failed <$expectedTerminalTaskId>",
            "=> Function complete <$($expectedTerminalTaskId.ToUpperInvariant())>",
            (([char]27) + "[31m" + ([char]0x2717) + " Function failed <$expectedTerminalTaskId>" + ([char]27) + "[0m")
        )
        foreach ($terminalTaskFixture in $terminalTaskFixtures) {
            $terminalTaskIds = @(Get-TerminalTaskIdsFromContent -Content $terminalTaskFixture)
            if (
                $terminalTaskIds.Count -ne 1 -or
                $terminalTaskIds[0] -ne $expectedTerminalTaskId
            ) {
                throw "Task 2 Beam terminal-task classifier self-test failed"
            }
        }
        foreach ($nonTerminalTaskFixture in @(
            "Traceback mentions Function failed <$expectedTerminalTaskId>",
            '"Function complete <5d73a8fd-5a1d-4bc0-b17d-4d4dec898692>"',
            "Function failed <not-a-uuid>"
        )) {
            if (@(Get-TerminalTaskIdsFromContent -Content $nonTerminalTaskFixture).Count -ne 0) {
                throw "Task 2 Beam terminal-task classifier accepted non-terminal text"
            }
        }
        foreach ($taskIdValue in @(
            $expectedTerminalTaskId,
            (([char]27) + "[31m" + $expectedTerminalTaskId + ([char]27) + "[0m"),
            " task-id=[$expectedTerminalTaskId] "
        )) {
            if ((Get-CanonicalBeamTaskId -Value $taskIdValue) -ne $expectedTerminalTaskId) {
                throw "Task 2 Beam task-ID canonicalizer self-test failed"
            }
        }
        if ($null -ne (Get-CanonicalBeamTaskId -Value "not-a-task-id")) {
            throw "Task 2 Beam task-ID canonicalizer accepted invalid input"
        }
        if ((Resolve-ExactBeamTaskId -Value $expectedTerminalTaskId) -ne $expectedTerminalTaskId) {
            throw "Task 2 Beam exact task-ID validator rejected a valid UUID"
        }
        foreach ($invalidExactTaskId in @(
            "<$expectedTerminalTaskId>",
            "task-id=$expectedTerminalTaskId",
            "not-a-task-id"
        )) {
            $invalidAccepted = $false
            try {
                Resolve-ExactBeamTaskId -Value $invalidExactTaskId | Out-Null
                $invalidAccepted = $true
            }
            catch {
                $invalidAccepted = $false
            }
            if ($invalidAccepted) {
                throw "Task 2 Beam exact task-ID validator accepted decorated input"
            }
        }
        $taskListFixture = @"
[
  {"id":"5d73a8fd-5a1d-4bc0-b17d-4d4dec898692","stub_name":"function/scripts.cloud.beam_task2_p14:smoke"},
  {"id":"845140eb-e87e-4859-8af4-031d4c33d9ca","stub_name":"function/scripts.cloud.beam_task2_p14:smoke"}
]
"@
        $parsedTaskListFixture = @(ConvertFrom-BeamTaskListJson -Json $taskListFixture)
        if (
            $parsedTaskListFixture.Count -ne 2 -or
            $parsedTaskListFixture[0].id -ne "5d73a8fd-5a1d-4bc0-b17d-4d4dec898692" -or
            $parsedTaskListFixture[1].id -ne "845140eb-e87e-4859-8af4-031d4c33d9ca"
        ) {
            throw "Task 2 Beam task-list flattening self-test failed"
        }
        if (-not (Test-RetryableGpuAllocationFailureContent -Content "Beam GPU is already occupied before Task 2 starts")) {
            throw "Task 2 Beam occupied-host classifier self-test failed"
        }
        foreach ($applicationFailure in @(
            "FileNotFoundError: missing source",
            "subprocess.CalledProcessError: training failed",
            "CUDA out of memory",
            "Function complete`nBeam returned no completed Task 2 result"
        )) {
            if (Test-RetryableGpuAllocationFailureContent -Content $applicationFailure) {
                throw "Task 2 Beam allocation classifier accepted an application failure"
            }
        }
        $attemptPlan = @(Get-GpuAttemptPlan)
        if (
            $GpuType -eq "FastServerless" -and
            [string]::IsNullOrWhiteSpace($PoolName) -and
            ($attemptPlan -join "|") -ne (
                "RTX5090,RTX4090,A10G|" +
                "RTX4090,A10G,RTX5090|" +
                "A10G,RTX5090,RTX4090"
            )
        ) {
            throw "Task 2 Beam FastServerless rotation self-test failed"
        }
        if (
            $GpuType -ne "FastServerless" -and
            ($attemptPlan.Count -ne 1 -or $attemptPlan[0] -ne $GpuType)
        ) {
            throw "Task 2 Beam explicit GPU preference self-test failed"
        }
        $result = Invoke-BeamFunction check
        if ($result.ExitCode -ne 0) {
            throw "Task 2 Beam local contract failed. Log: $($result.LogPath)"
        }
    }
    "Capacity" {
        Write-Host "Live Beam serverless inventory (read-only; no GPU is allocated):" -ForegroundColor Cyan
        Invoke-Beam machine list --no-offers
        Write-Host (
            "Inventory is only a scheduler snapshot. A Smoke request with the " +
            "RTX5090,RTX4090,A10G priority list is the authoritative allocation test."
        ) -ForegroundColor DarkGray
    }
    "Guard" {
        Assert-NoActiveTask2Job
        Write-Host "TASK 2 BEAM DUPLICATE GUARD PASS" -ForegroundColor Green
    }
    "ReplacePending" {
        Stop-PendingTask2Jobs
        Assert-NoActiveTask2Job
        Write-Host "TASK 2 BEAM PENDING REPLACEMENT PASS" -ForegroundColor Green
    }
    "Prepare" {
        Ensure-BeamInput
        Ensure-BeamModel
        Write-Host "BEAM TASK 2 PREPARE PASS" -ForegroundColor Green
    }
    "Smoke" {
        Assert-NoActiveTask2Job
        Ensure-BeamInput
        Ensure-BeamModel
        Write-BeamLaunchManifest
        try {
            Invoke-GpuStage smoke | Out-Null
        }
        finally {
            Remove-Item Env:UDSC_BEAM_ALLOCATION_NONCE -ErrorAction SilentlyContinue
        }
    }
    "Full" {
        Assert-NoActiveTask2Job
        Ensure-BeamInput
        Ensure-BeamModel
        Write-BeamLaunchManifest
        try {
            Invoke-GpuStage full | Out-Null
        }
        finally {
            Remove-Item Env:UDSC_BEAM_ALLOCATION_NONCE -ErrorAction SilentlyContinue
        }
    }
    "Run" {
        Assert-NoActiveTask2Job
        Ensure-BeamInput
        Ensure-BeamModel
        Write-BeamLaunchManifest
        try {
            Invoke-GpuStage smoke | Out-Null
            Invoke-GpuStage full | Out-Null
            $download = Receive-BeamResult
            if ($download.Status -eq "PUBLIC_CANDIDATE_READY") {
                Write-Host (
                    "TASK 2 BEAM CANDIDATE READY: $($download.Path) " +
                    "(held-out METEOR=$($download.HeldoutMeteor))"
                ) -ForegroundColor Green
            }
            else {
                Write-Warning (
                    "TASK 2 PIPELINE FINISHED BUT HELD-OUT REJECTED " +
                    "(METEOR=$($download.HeldoutMeteor)). Do not submit this run. " +
                    "Diagnostics were downloaded to $($download.Path)."
                )
            }
        }
        finally {
            Remove-Item Env:UDSC_BEAM_ALLOCATION_NONCE -ErrorAction SilentlyContinue
        }
    }
    "Status" {
        Invoke-Beam task list --limit 20
        Invoke-Beam ls $VolumeName
    }
    "Logs" {
        if ([string]::IsNullOrWhiteSpace($TaskId)) {
            throw "-TaskId is required. Copy the real ID from Stage Status; do not type <TASK_ID>."
        }
        $TaskId = Resolve-ExactBeamTaskId -Value $TaskId
        $logRoot = Join-Path $ProjectRoot "artifacts\task2\beam_logs"
        New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
        $logPath = Join-Path $logRoot "beam_task_${TaskId}.log"
        $beamExecutable = Resolve-BeamExecutable
        $exitCode = Invoke-NativeLogged `
            -Executable $beamExecutable `
            -Arguments @("logs", "--task-id", $TaskId, "-n", "300", "--show-timestamp") `
            -LogPath $logPath
        if ($exitCode -ne 0) {
            throw "Beam logs command failed. Log: $logPath"
        }
    }
    "Stop" {
        if ([string]::IsNullOrWhiteSpace($TaskId)) {
            throw "-TaskId is required for Stage Stop."
        }
        $TaskId = Resolve-ExactBeamTaskId -Value $TaskId
        Invoke-Beam task stop $TaskId
        Write-Host "Requested Beam task stop: $TaskId" -ForegroundColor Green
    }
    "Download" {
        Receive-BeamResult | Out-Null
    }
}
