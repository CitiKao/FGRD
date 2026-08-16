param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $Root "runs\paper_exact_v_frozen_$Stamp"
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Set-Location $Root

Write-Host "Applying frozen validation-selected V weights to the packaged test cache..." -ForegroundColor Cyan
& $Python -u "tools\verify_paper_v_from_cache.py" `
    --cache-dir "paper_cache\v" `
    --weights "paper_weights\v\target_horizon_bridge_grid_per_step_weights.npy" `
    --expected-json "reference_results\paper_v_expected.json" `
    --output-dir $OutputDir `
    --tolerance 0.00001
if ($LASTEXITCODE -ne 0) {
    throw "V paper-reference verification failed."
}

Write-Host "V paper-exact frozen evaluation complete: $OutputDir" -ForegroundColor Green
