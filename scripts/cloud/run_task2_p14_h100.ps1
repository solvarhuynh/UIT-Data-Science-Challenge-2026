[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidatePattern('^[1-9][0-9]*[mhd]$')]
    [string]$Ttl = "4h",

    [Parameter(Mandatory = $false)]
    [ValidateRange(1.0, 1000.0)]
    [double]$MaxSpendUsd = 16.0,

    [Parameter(Mandatory = $false)]
    [ValidatePattern('^[a-z0-9][a-z0-9-]{2,62}$')]
    [string]$PoolName = "udsc-task2-p14-h100",

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 3)]
    [int]$GpuAttempts = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Runner = Join-Path $ProjectRoot "scripts\cloud\run_task2_p14_beam.ps1"
$BeamCli = Join-Path $env:USERPROFILE ".local\bin\beam.exe"
$Reserved = $false
$PipelineError = $null
$ReleaseError = $null

if (-not (Test-Path -LiteralPath $BeamCli -PathType Leaf)) {
    throw "Beam CLI not found: $BeamCli"
}
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Task 2 Beam runner not found: $Runner"
}

$MaxSpendText = $MaxSpendUsd.ToString(
    "0.00",
    [System.Globalization.CultureInfo]::InvariantCulture
)

Set-Location -LiteralPath $ProjectRoot
Write-Warning (
    "This reserves one paid H100 on-demand machine. " +
    "TTL=$Ttl, max-spend=$MaxSpendText USD, pool=$PoolName. " +
    "Serverless-only credits may not cover this reservation."
)

& $BeamCli machine reserve `
    --gpu H100 `
    --nodes 1 `
    --ttl $Ttl `
    --name $PoolName `
    --max-spend $MaxSpendText `
    --yes
if ($LASTEXITCODE -ne 0) {
    throw "Beam could not reserve an H100 within the $MaxSpendText USD cap."
}
$Reserved = $true

try {
    Write-Host "H100 reservation created; verifying pool $PoolName..." -ForegroundColor Cyan
    & $BeamCli machine list --pool $PoolName --format table
    if ($LASTEXITCODE -ne 0) {
        throw "Beam could not verify H100 pool $PoolName."
    }

    & $Runner `
        -Stage Run `
        -GpuType H100 `
        -PoolName $PoolName `
        -GpuAttempts $GpuAttempts
}
catch {
    $PipelineError = $_
}
finally {
    if ($Reserved) {
        Write-Host "Releasing H100 pool $PoolName to stop billing..." -ForegroundColor Yellow
        & $BeamCli machine release --pool $PoolName --yes
        if ($LASTEXITCODE -ne 0) {
            $ReleaseError = "Beam failed to release H100 pool $PoolName. Run: beam machine release --pool $PoolName --yes"
        }
        else {
            Write-Host "H100 pool released; billing stopped." -ForegroundColor Green
        }
    }
}

if ($null -ne $PipelineError) {
    if ($null -ne $ReleaseError) {
        throw "$($PipelineError.Exception.Message) ALSO: $ReleaseError"
    }
    throw $PipelineError
}
if ($null -ne $ReleaseError) {
    throw $ReleaseError
}
