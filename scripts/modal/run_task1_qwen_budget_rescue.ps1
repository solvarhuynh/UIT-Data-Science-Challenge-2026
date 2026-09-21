[CmdletBinding()]
param(
    [ValidateSet("controller", "A", "B")]
    [string]$Worker = "controller"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$modalPath = Join-Path $repoRoot ".venv\Scripts\modal.exe"
$runnerPath = Join-Path $repoRoot "scripts\modal\task1_full_doc_top200_qwen3vl2b_optimized.py"
$logRoot = Join-Path $repoRoot "logs\task1_qwen_budget_rescue"

if (-not (Test-Path -LiteralPath $modalPath -PathType Leaf)) {
    throw "Modal executable not found: $modalPath"
}
if (-not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
    throw "Optimized runner not found: $runnerPath"
}
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$workerA = @("09", "11", "13", "15", "17", "19", "21", "23", "25", "27", "29", "31")
$workerB = @("10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30")

function Invoke-ModalStage {
    param(
        [string]$LogPath,
        [string[]]$Arguments
    )
    & $modalPath run --profile nghiadethuong3107 $runnerPath @Arguments 2>&1 |
        Out-File -LiteralPath $LogPath -Encoding utf8 -Append
    return [int]$LASTEXITCODE
}

function Invoke-Worker {
    param(
        [string]$Name,
        [string[]]$Shards
    )
    foreach ($shard in $Shards) {
        $logPath = Join-Path $logRoot "shard_$shard.log"
        "$(Get-Date -Format o) worker=$Name shard=$shard PRESELECT_START" |
            Out-File -LiteralPath $logPath -Encoding utf8 -Append
        $code = Invoke-ModalStage -LogPath $logPath -Arguments @(
            "--mode", "prepare-optimized-production-shard-selection",
            "--shard-id", $shard
        )
        if ($code -ne 0) {
            "$(Get-Date -Format o) worker=$Name shard=$shard PRESELECT_FAILED exit=$code; worker stopped" |
                Out-File -LiteralPath $logPath -Encoding utf8 -Append
            break
        }
        "$(Get-Date -Format o) worker=$Name shard=$shard PRESELECT_COMPLETE" |
            Out-File -LiteralPath $logPath -Encoding utf8 -Append

        "$(Get-Date -Format o) worker=$Name shard=$shard GPU_SCORE_START" |
            Out-File -LiteralPath $logPath -Encoding utf8 -Append
        $code = Invoke-ModalStage -LogPath $logPath -Arguments @(
            "--mode", "optimized-production-shard-preselected",
            "--shard-id", $shard,
            "--inference-batch-size", "1",
            "--durability-qdocs", "4096"
        )
        if ($code -ne 0) {
            "$(Get-Date -Format o) worker=$Name shard=$shard GPU_SCORE_FAILED exit=$code; worker stopped; checkpoint preserved" |
                Out-File -LiteralPath $logPath -Encoding utf8 -Append
            break
        }
        "$(Get-Date -Format o) worker=$Name shard=$shard COMPLETE" |
            Out-File -LiteralPath $logPath -Encoding utf8 -Append
    }
}

if ($Worker -eq "A") {
    Invoke-Worker -Name "A" -Shards $workerA
    exit 0
}
if ($Worker -eq "B") {
    Invoke-Worker -Name "B" -Shards $workerB
    exit 0
}

$controllerArgsA = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Worker A"
$controllerArgsB = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Worker B"
$processA = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList $controllerArgsA -PassThru
$processB = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList $controllerArgsB -PassThru
Write-Output "Started budget-rescue workers A/B with PIDs $($processA.Id)/$($processB.Id)."
Write-Output "Maximum GPU concurrency: 2. Logs: $logRoot"
