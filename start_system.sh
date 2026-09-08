#!/usr/bin/env bash
# Starts both TraderStuff processes (crypto default + stock/yfinance):
# nohup'd, output appended to backtrack_logs/main_run{,_stock_yfinance}.log,
# disowned so they survive this shell exiting -- nohup/disown genuinely
# matters here, not just cosmetic: a process started WITHOUT it (a
# same-session workaround used a few times this project when nohup itself
# seemed to hang) was confirmed to die silently once its launching shell
# session ended, which real unattended operation can't tolerate.
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
# Self-healing launch (2026-09-03): a `nohup`'d launch intermittently
# starts a process that sits at ~0% CPU producing no log output for 90s+
# before either recovering on its own or never recovering at all -- root
# cause not pinned down (suspected MSYS nohup + the venv python.exe's own
# child-process re-exec interacting badly, but never reproduced on demand
# enough to confirm). The OLD version of this script only checked that the
# PID existed after a flat 5s sleep, which happily reported "Running" for
# a process that was actually stuck -- a real false-positive this session
# hit more than once. Now each process is launched, then the actual
# dashboard HTTP endpoint is polled (real proof of life, not just PID
# existence) for up to LAUNCH_TIMEOUT_SECONDS; a launch that never answers
# gets killed and retried once before this script gives up and reports a
# real failure instead of a misleading "Running".
#
# Usage: bash start_system.sh
set -uo pipefail  # NOT -e: retry logic below needs to survive failed steps
cd "$(dirname "${BASH_SOURCE[0]}")"

LAUNCH_TIMEOUT_SECONDS=150
POLL_INTERVAL_SECONDS=5
CRYPTO_PORT=8080
STOCK_PORT=8081

find_pids() {
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*TraderStuff*main.py*' } | Select-Object -ExpandProperty ProcessId" | tr -d '\r' | grep -v '^$' || true
}

find_pids_for() {
  # $1: the PID set (newline-separated) captured via find_pids() BEFORE this
  # launch -- env vars (ASSET_TYPE=stock etc.) don't show up in WMI's
  # CommandLine, so "which PID is the new one" is found by diffing against
  # that baseline instead of matching on command-line content.
  comm -13 <(printf '%s\n' "$1" | sort) <(find_pids | sort)
}

port_is_up() {
  local port="$1"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/" --max-time 3 2>/dev/null)
  [ "$code" = "200" ]
}

kill_pids() {
  for pid in $1; do
    powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1 || true
  done
}

# launch_and_verify NAME PORT ENV_PREFIX LOGFILE
# ENV_PREFIX is passed through eval'd word-splitting -- caller controls it, not user input.
launch_and_verify() {
  local name="$1" port="$2" env_prefix="$3" logfile="$4"
  local attempt baseline_pids new_pids waited

  for attempt in 1 2; do
    echo "--- Starting $name process (attempt $attempt/2) ---"
    baseline_pids=$(find_pids)
    env $env_prefix nohup .venv/Scripts/python -u main.py >> "$logfile" 2>&1 &
    disown

    waited=0
    while [ "$waited" -lt "$LAUNCH_TIMEOUT_SECONDS" ]; do
      if port_is_up "$port"; then
        new_pids=$(find_pids_for "$baseline_pids")
        echo "$name UP after ${waited}s (PID(s): $(echo $new_pids | tr '\n' ' '))"
        return 0
      fi
      sleep "$POLL_INTERVAL_SECONDS"
      waited=$((waited + POLL_INTERVAL_SECONDS))
    done

    new_pids=$(find_pids_for "$baseline_pids")
    echo "$name did not answer on port $port within ${LAUNCH_TIMEOUT_SECONDS}s -- killing and $([ "$attempt" = 1 ] && echo retrying || echo giving up)."
    kill_pids "$new_pids"
    sleep 2
  done
  return 1
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

# Path 2 (LLM debate) disabled by default since 2026-08-30 -- 3,050+
# recorded debate decisions across this project's history, zero ever
# executed a trade, but each cycle still burned 3 OpenAI calls/symbol.
# Path 1 (SMC/FVG paper trading) is untouched by this. See settings.py
# and RESEARCH_NOTES.md. Set ENABLE_LLM_CONSENSUS=true below to re-enable.
#
# Correlation cap: default 0.08 (not the backtest-favored 0.12/CAP_1.5X --
# see RESEARCH_NOTES.md "Live evidence vs backtest" for why that's still
# pending, not deployed).
# SPOT-VIABLE CONFIGURATION, live from 2026-09-08.
#
# The production sizing (2% risk, no notional cap) is not deployable: it
# needs up to 17x the account in total open notional, and spot cannot borrow.
# Measured over 730d/6 segments, the combination below is 100%
# spot-executable, profitable in 6/6 segments, and cuts worst-segment
# drawdown from 69.1% to 9.7%. It costs trade count and absolute return
# (+394% over 730d instead of an unattainable +2,244%).
#
# Neither half works alone: capping risk leaves total notional unchanged
# because the freed budget admits more positions, and capping notional at
# 1.0x blocks everything because one production-sized position already
# exceeds the account. See RESEARCH_NOTES.md and
# experiments/notional_cap_spot_viability.py.
#
# NOTE this changes what the live paper ledger records. Ledger history splits
# at this timestamp: everything before is the leveraged configuration.
SPOT_CFG="PAPER_TRADING_RISK_PCT=0.0025 PORTFOLIO_MAX_NOTIONAL_MULTIPLE=1.0"

# LiveOrderMirror: mirrors each paper decision onto Binance TESTNET so the
# order path runs against a real exchange with fake money. Real money needs
# BINANCE_TESTNET=false AND LIVE_TRADING_CONFIRM_REAL_MONEY=true as well --
# neither is set here, and the mirror refuses to arm without both.
MIRROR_CFG="LIVE_TRADING_ENABLED=true"

crypto_ok=true
launch_and_verify "crypto" "$CRYPTO_PORT" "ENABLE_LLM_CONSENSUS=false $SPOT_CFG $MIRROR_CFG" "backtrack_logs/main_run.log" || crypto_ok=false

# Stock runs the same sizing for consistency, but NOT the mirror: the
# exchange is Binance, which has no NASDAQ symbols to mirror to.
stock_ok=true
launch_and_verify "stock" "$STOCK_PORT" "ENABLE_LLM_CONSENSUS=false $SPOT_CFG ASSET_TYPE=stock MARKET_SOURCE=yfinance" "backtrack_logs/main_run_stock_yfinance.log" || stock_ok=false

echo ""
echo "=== Final status ==="
final_pids=$(find_pids)
echo "Running PID(s): $(echo $final_pids | tr '\n' ' ')"
if [ "$crypto_ok" = true ] && [ "$stock_ok" = true ]; then
  echo "Both dashboards confirmed up: http://localhost:${CRYPTO_PORT} (crypto), http://localhost:${STOCK_PORT} (stock)"
  echo "Use stop_system.sh to stop -- it finds the real processes the same way regardless of which terminal/session you run it from."
  exit 0
else
  echo "FAILED to confirm: $([ "$crypto_ok" = false ] && echo 'crypto ')$([ "$stock_ok" = false ] && echo 'stock')"
  echo "Check the log files above for what happened -- do NOT assume the system is running."
  exit 1
fi
