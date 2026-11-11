# Registers the fund as Windows Scheduled Tasks, so it runs without a terminal open.
#
#   TraderStuffFundDaemon  fund/daemon.py: every minute it values the book, writes a
#                          heartbeat and enforces the drawdown halt; every hour (hh:10)
#                          it makes the daily C4 run (idempotent per UTC day, replays
#                          missed days). Started at logon and re-triggered every 15
#                          minutes; a second copy exits at once, so a dead daemon is
#                          back within 15 minutes.
#   TraderStuffFundBot     Telegram commands (/status, /stop, /lanjut), at logon.
#   TraderStuffFundDash    dashboard on http://127.0.0.1:8090, at logon.
#
# Paper mode only. Live mode is never scheduled by this script.
#
# Usage: powershell -ExecutionPolicy Bypass -File fund\setup_task.ps1
#        powershell -ExecutionPolicy Bypass -File fund\setup_task.ps1 -Remove

param([switch]$Remove)
$ErrorActionPreference = "Stop"
$names = @("TraderStuffFundDaemon", "TraderStuffFundBot", "TraderStuffFundDash")
$retired = @("TraderStuffFundDaily")   # replaced by the daemon 2026-11-11
foreach ($n in $retired) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }

if ($Remove) {
    foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Host "Removed $($names -join ', ')"
    exit 0
}

$repoRoot = Split-Path $PSScriptRoot -Parent
# The project's venv first (the libraries the tests ran with), else python on PATH.
$venvw = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"
if (Test-Path $venvw) { $python = Join-Path $repoRoot ".venv\Scripts\python.exe" } else { $python = (Get-Command python.exe).Source }
$pythonw = Join-Path (Split-Path $python -Parent) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found next to $python" }

# The daemon: at logon, plus a 15-minute re-trigger that only matters if it died
$every15 = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 15)
$daemonSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "TraderStuffFundDaemon" -Settings $daemonSettings -Force `
    -Trigger @((New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME), $every15) `
    -Action (New-ScheduledTaskAction -Execute $pythonw -Argument "-m fund.daemon --mode paper" -WorkingDirectory $repoRoot) `
    -Description "TraderStuff fund: minute watch + hourly/daily C4 paper run (fund/daemon.py)" | Out-Null

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
Write-Host "Start everything now without logging off:"
Write-Host "  Start-ScheduledTask TraderStuffFundDaemon; Start-ScheduledTask TraderStuffFundBot; Start-ScheduledTask TraderStuffFundDash"
