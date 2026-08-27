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
# Usage: bash start_system.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

existing=$(ps aux | grep -i "\.venv/Scripts/python" | grep -v grep || true)
if [ -n "$existing" ]; then
  echo "Refusing to start: a TraderStuff process already appears to be running:"
  echo "$existing"
  echo "Run stop_system.sh first if you want to restart."
  exit 1
fi

echo "=== Reconciling downtime (if any open positions exist from before) ==="
.venv/Scripts/python reconcile_downtime.py --asset-type both || echo "(reconcile_downtime.py exited non-zero -- check output above; continuing to start anyway)"

echo ""
echo "=== Starting crypto process (default env) ==="
nohup .venv/Scripts/python -u main.py >> backtrack_logs/main_run.log 2>&1 &
disown
crypto_pid=$!

echo "=== Starting stock process (ASSET_TYPE=stock MARKET_SOURCE=yfinance) ==="
ASSET_TYPE=stock MARKET_SOURCE=yfinance nohup .venv/Scripts/python -u main.py >> backtrack_logs/main_run_stock_yfinance.log 2>&1 &
disown
stock_pid=$!

sleep 5
echo ""
echo "=== Process check ==="
ps aux | grep -i "\.venv/Scripts/python" | grep -v grep || echo "WARNING: no processes found -- something failed to start, check the log files"

echo ""
echo "Launched (shell job PIDs $crypto_pid / $stock_pid -- these will differ from"
echo "the actual python process PIDs shown above under a git-bash/MSYS shell)."
echo "Both processes take 1-2 minutes to finish backfilling before they're fully live."
echo "Tail the logs to watch startup: tail -f backtrack_logs/main_run.log"
echo "                                 tail -f backtrack_logs/main_run_stock_yfinance.log"
