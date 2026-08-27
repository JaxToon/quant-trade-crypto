#!/usr/bin/env bash
# Starts both TraderStuff processes (crypto default + stock/yfinance) the
# same way they've been launched manually all session: nohup'd, output
# appended to backtrack_logs/main_run{,_stock_yfinance}.log, disowned so
# they survive this shell exiting.
#
# Runs reconcile_downtime.py first (only meaningful if a ledger file
# already exists with open positions from before a shutdown -- see
# reconcile_downtime.py's own docstring) so any position that genuinely
# hit its SL/TP while the system was down gets closed for real before
# main.py resumes live monitoring, instead of sitting stale until someone
# notices. Refuses to start if either process already appears to be
# running, to avoid a double-launch (duplicate trading, port conflicts).
#
# Detects an already-running instance via PowerShell/WMI, not `ps` -- `ps`'s
# PID numbering is scoped to the specific MSYS/Git-Bash runtime that spawned
# a process, so a different bash session (or even a different Git Bash
# window) can't reliably see a process another session started, and not
# every MSYS ps build even supports the Windows-wide `-W` flag (found live,
# 2026-08-27, see stop_system.sh's header for the full story). PowerShell's
# WMI query returns the real Windows PID and works identically regardless
# of which shell calls it.
#
# Usage: bash start_system.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

find_pids() {
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*TraderStuff*main.py*' } | Select-Object -ExpandProperty ProcessId" | tr -d '\r' | grep -v '^$' || true
}

existing=$(find_pids)
if [ -n "$existing" ]; then
  echo "Refusing to start: a TraderStuff process already appears to be running (PID(s): $(echo $existing | tr '\n' ' '))."
  echo "Run stop_system.sh first if you want to restart."
  exit 1
fi

echo "=== Reconciling downtime (if any open positions exist from before) ==="
.venv/Scripts/python reconcile_downtime.py --asset-type both || echo "(reconcile_downtime.py exited non-zero -- check output above; continuing to start anyway)"

echo ""
echo "=== Starting crypto process (PORTFOLIO_MAX_CORRELATED_RISK_PCT=0.12, see RESEARCH_NOTES.md) ==="
# Raised from the 0.08 default for crypto only -- backtest evidence (2026-08-27,
# RESEARCH_NOTES.md "Directional sizing" + "Correlation cap vs short sizing"):
# 2yr/6-regime backtest showed +65% return with max drawdown slightly BETTER
# than default (26.7% vs 28.1%). Stock's own backtest showed the same cap
# increase is NOT a free improvement there (drawdown +24% relative worse) --
# deliberately NOT set on the stock process below, which falls through to
# .env's 0.08 default instead.
PORTFOLIO_MAX_CORRELATED_RISK_PCT=0.12 nohup .venv/Scripts/python -u main.py >> backtrack_logs/main_run.log 2>&1 &
disown
crypto_pid=$!

echo "=== Starting stock process (ASSET_TYPE=stock MARKET_SOURCE=yfinance, default risk cap) ==="
ASSET_TYPE=stock MARKET_SOURCE=yfinance nohup .venv/Scripts/python -u main.py >> backtrack_logs/main_run_stock_yfinance.log 2>&1 &
disown
stock_pid=$!

sleep 5
echo ""
echo "=== Process check ==="
new_pids=$(find_pids)
if [ -n "$new_pids" ]; then
  echo "Running (PID(s): $(echo $new_pids | tr '\n' ' '))"
else
  echo "WARNING: no processes found -- something failed to start, check the log files"
fi

echo ""
echo "Launched (shell job PIDs $crypto_pid / $stock_pid -- these are this shell's"
echo "own bookkeeping and will differ from the real Windows PIDs shown above)."
echo "Use stop_system.sh to stop -- it finds the real processes the same way"
echo "regardless of which terminal/session you run it from."
echo "Both processes take 1-2 minutes to finish backfilling before they're fully live."
echo "Tail the logs to watch startup: tail -f backtrack_logs/main_run.log"
echo "                                 tail -f backtrack_logs/main_run_stock_yfinance.log"
