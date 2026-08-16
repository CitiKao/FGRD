param(
    [string]$Python = "python",
    [int]$GapEpochs = 200,
    [int]$VEpochs = 610,
    [int]$Patience = 20,
    [int]$CheckpointEvery = 25,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$Only = "gap_no_adaptive,gap_single_st_block,gap_no_fixed_graph,v_no_adaptive,v_single_st_block,v_no_fixed_graph"
$ArgsList = @(
    "-u", "tools\run_nyc_gap_v_controlled_ablation_suite.py",
    "--only", $Only,
    "--gap-epochs", "$GapEpochs",
    "--v-epochs", "$VEpochs",
    "--patience", "$Patience",
    "--checkpoint-every", "$CheckpointEvery",
    "--python-exe", "$Python"
)
if ($OutputRoot) {
    $ArgsList += @("--output-root", $OutputRoot)
}

& $Python @ArgsList
exit $LASTEXITCODE
