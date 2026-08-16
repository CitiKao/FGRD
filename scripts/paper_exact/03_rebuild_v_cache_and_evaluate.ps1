param(
    [string]$Python = "python",
    [string]$Device = "cuda",
    [string]$Precision = "auto",
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $Root "runs\paper_exact_v_rebuilt_$Stamp"
$CacheRunDir = Join-Path $OutputDir "cache_build"
$FinalDir = Join-Path $OutputDir "frozen_evaluation"
New-Item -ItemType Directory -Path $CacheRunDir -Force | Out-Null
New-Item -ItemType Directory -Path $FinalDir -Force | Out-Null
Set-Location $Root

$RunDirs = @(
    (Join-Path $Root "paper_checkpoints\v\stream_01"),
    (Join-Path $Root "paper_checkpoints\v\stream_02"),
    (Join-Path $Root "paper_checkpoints\v\stream_03"),
    (Join-Path $Root "paper_checkpoints\v\stream_04")
)

Write-Host "[V 1/2] Rebuilding four checkpoint streams and six causal candidates..." -ForegroundColor Cyan
& $Python -u "tools\ensemble_nyc_stgat_speed_predictions.py" `
    --run-dirs $RunDirs `
    --data-dir "data\nyc_dc" `
    --output-dir $CacheRunDir `
    --batch-size $BatchSize `
    --device $Device `
    --precision $Precision `
    --num-workers 0 `
    --calibration-fit-split val `
    --save-prediction-cache `
    --cache-only
if ($LASTEXITCODE -ne 0) {
    throw "V prediction-cache rebuild failed."
}

Write-Host "[V 2/2] Applying the frozen paper weights to the rebuilt cache..." -ForegroundColor Cyan
& $Python -u "tools\verify_paper_v_from_cache.py" `
    --cache-dir (Join-Path $CacheRunDir "prediction_cache") `
    --weights "paper_weights\v\target_horizon_bridge_grid_per_step_weights.npy" `
    --expected-json "reference_results\paper_v_expected.json" `
    --output-dir $FinalDir `
    --tolerance 0.02
if ($LASTEXITCODE -ne 0) {
    throw "Rebuilt V cache differs from the paper reference beyond tolerance."
}

Write-Host "V cache rebuild and evaluation complete: $OutputDir" -ForegroundColor Green
