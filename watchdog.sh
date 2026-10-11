#!/usr/bin/env bash
# Health check + self-heal for the two live TraderStuff processes (crypto,
# stock). Meant to be run periodically and unattended (via Windows Task
# Scheduler -- see setup_watchdog_task.ps1) so an outage gets noticed and
# recovered from without a human checking the dashboard first. Before this
# script, the ONLY way this project detected an outage was someone manually
# checking -- real outages this session (Neo4j drops, sleep/hibernate on
# battery, an unhandled Binance WebSocket disconnect, an intermittent nohup
# launch hang) sat undetected for hours at a time more than once.
#
# Unlike start_system.sh (all-or-nothing: refuses to start if ANYTHING is
# already running), this checks crypto and stock INDEPENDENTLY and restarts
# only the one that's actually down, leaving a healthy sibling untouched.
#
# One run = one check. Exits 0 whether or not a restart was needed; exits 1
# only if a restart was attempted and still failed to come up (worth
# surfacing in Task Scheduler's history / an alert).
#
# Usage: bash watchdog.sh   (called on a schedule, not meant to be a
#                             persistent loop itself)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LAUNCH_TIMEOUT_SECONDS=150
POLL_INTERVAL_SECONDS=5
LOG=backtrack_logs/watchdog.log

# Same launch environment as start_system.sh. This file used to hard-code a
# copy without the spot sizing or the testnet mirror (fixed 2026-10-11).
. ./system_env.sh

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts) $1" | tee -a "$LOG"; }

find_pids() {
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*TraderStuff*main.py*' } | Select-Object -ExpandProperty ProcessId" | tr -d '\r' | grep -v '^$' || true
}

find_pids_for() {
  comm -13 <(printf '%s\n' "$1" | sort) <(find_pids | sort)
}

port_is_up() {
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$1/" --max-time 3 2>/dev/null)
  [ "$code" = "200" ]
}

kill_pids() {
  for pid in $1; do
    powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1 || true
  done
}

neo4j_is_up() {
  # Reuses the project's own Neo4jClient (loads .env the same way main.py
  # does) rather than reading raw NEO4J_* from this shell's environment --
  # those aren't set here at all (.env isn't sourced into bash), which made
  # an earlier version of this check always report Neo4j down.
  .venv/Scripts/python -c "
import sys
from graph.neo4j_client import Neo4jClient
try:
    Neo4jClient.get().run_read('RETURN 1')
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
}

# heal NAME PORT ENV_PREFIX LOGFILE ASSET_TYPE_ARG
# Duplicates start_system.sh's launch_and_verify on purpose -- that script's
# version is all-or-nothing (refuses if ANY TraderStuff process is running),
# this one restarts a single asset type without disturbing a healthy sibling.
# Keep both in sync by hand if the launch sequence ever changes.
heal() {
  local name="$1" port="$2" env_prefix="$3" logfile="$4" asset_arg="$5"
  local baseline_pids new_pids waited attempt

  log "$name: down (port $port not responding). Reconciling downtime and restarting..."
  .venv/Scripts/python reconcile_downtime.py --asset-type "$asset_arg" >> "$LOG" 2>&1 || true

  for attempt in 1 2; do
    baseline_pids=$(find_pids)
    env $env_prefix nohup .venv/Scripts/python -u main.py >> "$logfile" 2>&1 &
    disown

    waited=0
    while [ "$waited" -lt "$LAUNCH_TIMEOUT_SECONDS" ]; do
      if port_is_up "$port"; then
        new_pids=$(find_pids_for "$baseline_pids")
        log "$name: recovered after ${waited}s (PID(s): $(echo $new_pids | tr '\n' ' '))"
        return 0
      fi
      sleep "$POLL_INTERVAL_SECONDS"
      waited=$((waited + POLL_INTERVAL_SECONDS))
    done

    new_pids=$(find_pids_for "$baseline_pids")
    log "$name: restart attempt $attempt did not answer within ${LAUNCH_TIMEOUT_SECONDS}s -- killing and $([ "$attempt" = 1 ] && echo retrying || echo giving up)."
    kill_pids "$new_pids"
    sleep 2
  done
  log "$name: FAILED to recover after 2 attempts. Needs manual attention."
  return 1
}

overall_ok=true

if ! neo4j_is_up; then
  log "Neo4j: unreachable. No CLI way to start it (Neo4j Desktop is a GUI app) -- both processes will fail to (re)start until it's opened manually. Skipping process checks this run."
  exit 1
fi

if port_is_up 8080; then
  : # crypto healthy, nothing to log every run
else
  heal "crypto" 8080 "$CRYPTO_ENV" "backtrack_logs/main_run.log" "crypto" || overall_ok=false
fi

if port_is_up 8081; then
  : # stock healthy, nothing to log every run
else
  heal "stock" 8081 "$STOCK_ENV" "backtrack_logs/main_run_stock_yfinance.log" "stock" || overall_ok=false
fi

[ "$overall_ok" = true ]
