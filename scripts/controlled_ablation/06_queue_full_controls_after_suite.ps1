param(
    [Parameter(Mandatory = $true)]
    [int]$CurrentSuitePid,
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [int]$GapEpochs = 200,
    [int]$VEpochs = 610,
    [int]$Patience = 20,
    [int]$CheckpointEvery = 25
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ResolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$SuiteStatusPath = Join-Path $ResolvedOutputRoot "suite_status.json"
$QueueStatusPath = Join-Path $ResolvedOutputRoot "full_controls_queue_status.json"
$QueueLogPath = Join-Path $ResolvedOutputRoot "full_controls_queue.log"
$FullStdoutPath = Join-Path $ResolvedOutputRoot "full_controls_stdout.log"
$FullStderrPath = Join-Path $ResolvedOutputRoot "full_controls_stderr.log"

function Get-LocalTimestamp {
    return [DateTimeOffset]::Now.ToString("o")
}

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $QueueLogPath -Encoding UTF8 -Value "$(Get-LocalTimestamp) $Message"
}

function Write-QueueStatus {
    param(
        [string]$State,
        [string]$Message,
        [Nullable[int]]$ExitCode = $null
    )
    $Payload = [ordered]@{
        state = $State
        updated_at = Get-LocalTimestamp
        message = $Message
        waited_suite_pid = $CurrentSuitePid
        output_root = $ResolvedOutputRoot
        gap_epochs = $GapEpochs
        v_epochs = $VEpochs
        patience = $Patience
        checkpoint_every = $CheckpointEvery
    }
    if ($null -ne $ExitCode) {
        $Payload.exit_code = $ExitCode
    }
    $TempPath = "$QueueStatusPath.tmp"
    $Payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $TempPath -Encoding UTF8
    Move-Item -LiteralPath $TempPath -Destination $QueueStatusPath -Force
}

try {
    Write-QueueLog "Queued Full GAP then Full V after six-ablation suite PID $CurrentSuitePid."
    Write-QueueStatus -State "waiting_for_six_ablations" -Message "Waiting for the active six-ablation suite to complete successfully."

    $CurrentProcess = Get-Process -Id $CurrentSuitePid -ErrorAction SilentlyContinue
    if ($null -ne $CurrentProcess) {
        Wait-Process -Id $CurrentSuitePid
    }

    Start-Sleep -Seconds 2
    if (-not (Test-Path -LiteralPath $SuiteStatusPath)) {
        throw "Missing suite status file after PID $CurrentSuitePid exited: $SuiteStatusPath"
    }
    $SuiteStatus = Get-Content -LiteralPath $SuiteStatusPath -Raw | ConvertFrom-Json
    if ($SuiteStatus.state -ne "complete") {
        $Message = "Six-ablation suite did not complete successfully (state=$($SuiteStatus.state)); Full controls were not started."
        Write-QueueLog $Message
        Write-QueueStatus -State "blocked" -Message $Message
        exit 2
    }

    $Message = "Six ablations completed successfully; starting same-protocol GAP Full followed by V Full."
    Write-QueueLog $Message
    Write-QueueStatus -State "running_full_controls" -Message $Message

    Set-Location $Root
    & (Join-Path $Root "scripts\controlled_ablation\03_run_full_controls_only.ps1") `
        -Python $Python `
        -GapEpochs $GapEpochs `
        -VEpochs $VEpochs `
        -Patience $Patience `
        -CheckpointEvery $CheckpointEvery `
        -OutputRoot $ResolvedOutputRoot `
        1>> $FullStdoutPath `
        2>> $FullStderrPath
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        $Message = "Full-control suite exited with code $ExitCode."
        Write-QueueLog $Message
        Write-QueueStatus -State "failed" -Message $Message -ExitCode $ExitCode
        exit $ExitCode
    }

    $Message = "GAP Full and V Full completed successfully."
    Write-QueueLog $Message
    Write-QueueStatus -State "complete" -Message $Message -ExitCode 0
    exit 0
}
catch {
    $Message = "Queue failure: $($_.Exception.Message)"
    Write-QueueLog $Message
    Write-QueueStatus -State "failed" -Message $Message -ExitCode 1
    exit 1
}
