# Registers the fund as Windows Scheduled Tasks, so it runs without a terminal open.
#
#   TraderStuffFundDaily   paper run at 07:10 local (= 00:10 UTC on this machine,
#                          SE Asia Standard Time, UTC+7), then hourly retries for
#                          23 hours. The run is idempotent per UTC day and takes a
#                          lock, so retries never trade twice; they only catch a run
#                          that failed (late data) or was missed (PC off/asleep).
#   TraderStuffFundBot     Telegram commands (/status, /stop, /lanjut), at logon.
#   TraderStuffFundDash    dashboard on http://127.0.0.1:8090, at logon.
#
# Paper mode only. Live mode is never scheduled by this script.
#
# Usage: powershell -ExecutionPolicy Bypass -File fund\setup_task.ps1
#        powershell -ExecutionPolicy Bypass -File fund\setup_task.ps1 -Remove

param([switch]$Remove)
$ErrorActionPreference = "Stop"
$names = @("TraderStuffFundDaily", "TraderStuffFundBot", "TraderStuffFundDash")

if ($Remove) {
    foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Host "Removed $($names -join ', ')"
    exit 0
}

$repoRoot = Split-Path $PSScriptRoot -Parent
$python = (Get-Command python.exe).Source
$pythonw = Join-Path (Split-Path $python -Parent) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found next to $python" }

# Daily run with hourly retries
$daily = New-ScheduledTaskTrigger -Daily -At "07:10"
$daily.Repetition = (New-ScheduledTaskTrigger -Once -At "07:10" -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Hours 23)).Repetition
$runSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "TraderStuffFundDaily" -Trigger $daily -Settings $runSettings -Force `
    -Action (New-ScheduledTaskAction -Execute $pythonw -Argument "-m fund.run_daily --mode paper" -WorkingDirectory $repoRoot) `
    -Description "TraderStuff fund: daily C4 paper run (fund/run_daily.py)" | Out-Null

# Long-running services at logon, restarted if they die
$svcSettings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
Register-ScheduledTask -TaskName "TraderStuffFundBot" -Trigger $logon -Settings $svcSettings -Force `
    -Action (New-ScheduledTaskAction -Execute $pythonw -Argument "-m fund.telegram_bot" -WorkingDirectory $repoRoot) `
    -Description "TraderStuff fund: Telegram commands (fund/telegram_bot.py)" | Out-Null
Register-ScheduledTask -TaskName "TraderStuffFundDash" -Trigger $logon -Settings $svcSettings -Force `
    -Action (New-ScheduledTaskAction -Execute $pythonw -Argument "-m fund.dashboard" -WorkingDirectory $repoRoot) `
    -Description "TraderStuff fund: dashboard on 127.0.0.1:8090 (fund/dashboard.py)" | Out-Null

Write-Host "Registered: $($names -join ', ') (python: $pythonw, cwd: $repoRoot)"
Write-Host "Start the bot and dashboard now without logging off:"
Write-Host "  Start-ScheduledTask TraderStuffFundBot; Start-ScheduledTask TraderStuffFundDash"
