# Registers the daily fund run as a Windows Scheduled Task.
#
# 07:10 local time = 00:10 UTC on this machine (SE Asia Standard Time, UTC+7),
# ten minutes after the daily candle closes. StartWhenAvailable makes a missed
# run (PC off or asleep) fire at the next chance; the run is idempotent per
# UTC day, so a late or repeated start never trades twice.
#
# Paper mode only. Live mode is never scheduled by this script.
#
# Usage: powershell -ExecutionPolicy Bypass -File fund\setup_task.ps1
#        powershell -ExecutionPolicy Bypass -File fund\setup_task.ps1 -Remove

param([switch]$Remove)
$ErrorActionPreference = "Stop"
$taskName = "TraderStuffFundDaily"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed $taskName"
    exit 0
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$python = (Get-Command python.exe).Source
$pythonw = Join-Path (Split-Path $python -Parent) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found next to $python" }

$action = New-ScheduledTaskAction -Execute $pythonw -Argument "-m fund.run_daily --mode paper" -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -Daily -At "07:10"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "TraderStuff fund: daily C4 paper run (fund/run_daily.py)" -Force | Out-Null
Write-Host "Registered $taskName : $pythonw -m fund.run_daily --mode paper (daily 07:10, cwd $repoRoot)"
