$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$legacy = @(
    'backend',
    'data_pipeline',
    'embed_serving',
    'llm_finetuning_serving',
    'database',
    'monitoring',
    'workflows'
)

foreach ($name in $legacy) {
    $path = Join-Path $root $name
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

$models = Join-Path $root 'models'
Get-ChildItem -LiteralPath $models -Force -Directory |
    Where-Object { $_.Name -notin @('bkai-bi-encoder', 'qwen3-legal') } |
    Remove-Item -Recurse -Force

New-Item -ItemType Directory -Force `
    (Join-Path $models 'bkai-bi-encoder'), `
    (Join-Path $models 'qwen3-legal') | Out-Null

Write-Host 'Legacy cleanup completed.'

