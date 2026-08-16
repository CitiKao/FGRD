param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$Required = @(
    "data\nyc_dc",
    "paper_checkpoints\gap\low_lr_epoch003.pt",
    "paper_checkpoints\gap\high_lr_epoch003.pt",
    "paper_checkpoints\v\stream_01\stgat_best.pt",
    "paper_checkpoints\v\stream_02\stgat_best.pt",
    "paper_checkpoints\v\stream_03\stgat_best.pt",
    "paper_checkpoints\v\stream_04\stgat_best.pt",
    "paper_cache\v\shared_features.npz",
    "paper_weights\v\target_horizon_bridge_grid_per_step_weights.npy",
    "reference_results\paper_gap_expected.json",
    "reference_results\paper_v_expected.json"
)

$Missing = @()
foreach ($RelativePath in $Required) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $RelativePath))) {
        $Missing += $RelativePath
    }
}
if ($Missing.Count -gt 0) {
    throw "Missing required package files:`n$($Missing -join "`n")"
}

Write-Host "Package files: OK" -ForegroundColor Green
& $Python -c "import numpy, torch; print('Python dependencies: OK'); print('torch=' + torch.__version__); print('CUDA available=' + str(torch.cuda.is_available())); print('GPU=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency check failed."
}

Write-Host "Preflight complete." -ForegroundColor Cyan
