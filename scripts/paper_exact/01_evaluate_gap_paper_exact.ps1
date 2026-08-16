param(
    [string]$Python = "python",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $Root "runs\paper_exact_gap_$Stamp"
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
Set-Location $Root

$LowLr = Join-Path $Root "paper_checkpoints\gap\low_lr_epoch003.pt"
$HighLr = Join-Path $Root "paper_checkpoints\gap\high_lr_epoch003.pt"
$CheckpointList = "$LowLr,$HighLr"

Write-Host "[GAP 1/2] Evaluating both paper-selected checkpoint candidates..." -ForegroundColor Cyan
& $Python -u "tools\chengdu_gap_arch_smoke.py" `
    --data-dir "data\nyc_dc" `
    --hist-len 14 `
    --pred-horizon 4 `
    --report-horizons-minutes "15,30,60" `
    --max-train-samples 0 `
    --max-val-samples 0 `
    --max-test-samples 0 `
    --batch-size 16 `
    --hidden-dim 16 `
    --num-heads 4 `
    --num-st-blocks 2 `
    --num-gtcn-layers 2 `
    --kernel-size 3 `
    --adaptive-emb 10 `
    --adaptive-topk 16 `
    --score-mape-weight 0.05 `
    --gap-target-transform signed_log `
    --gap-log-scale 10 `
    --calibration-mode stepwise_shrink `
    --calibration-mape-weight 0.05 `
    --variants gap_v_to_gap `
    --eval-checkpoints $CheckpointList `
    --seed 42 `
    --device $Device `
    --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "GAP checkpoint evaluation failed."
}

Write-Host "[GAP 2/2] Selecting 15/30/60-minute checkpoints using validation only..." -ForegroundColor Cyan
& $Python -u "tools\compose_paper_gap_results.py" `
    --evaluation-json (Join-Path $OutputDir "checkpoint_eval_results.json") `
    --output-dir $OutputDir `
    --expected-json "reference_results\paper_gap_expected.json" `
    --mape-weight 0.05 `
    --tolerance 0.0001
if ($LASTEXITCODE -ne 0) {
    throw "GAP paper-reference verification failed."
}

Write-Host "GAP paper-exact evaluation complete: $OutputDir" -ForegroundColor Green
