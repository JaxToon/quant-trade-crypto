#!/usr/bin/env bash
# Stops both TraderStuff processes.
#
# Detection uses PowerShell/WMI (Get-CimInstance Win32_Process) instead of
# `ps`, because `ps`'s PID numbering is scoped to the specific MSYS/Git-Bash
# runtime that spawned a process -- a DIFFERENT bash.exe (even a second
# window of "the same" Git Bash install) can't see or signal a process
# another MSYS session started, and `ps -W`'s Windows-wide flag isn't even
# supported by every MSYS ps build (found live, 2026-08-27: one session's
# ps rejected -W outright with "unsupported SysV option"). PowerShell's WMI
# query returns the real Windows PID and works identically no matter which
# shell/session calls it -- confirmed live across two different bash
# sessions on the same machine.
#
# Termination is forceful (PowerShell Stop-Process, no graceful SIGTERM) --
# confirmed live that neither `kill -TERM` on a cross-session PID nor a
# non-forced `taskkill` can gracefully signal these processes from a
# session that didn't spawn them (Windows has no cross-session POSIX-signal
# equivalent for a detached console app; taskkill without /F explicitly
# refused with "This process can only be terminated forcefully"). This is
# safe: PaperTradingLedger saves on every tick, not just on clean shutdown
# (see reconcile_downtime.py's docstring), and start_system.sh always runs
# reconcile_downtime.py on the next start specifically to close out
# whatever changed during the gap.
#
# Usage: bash stop_system.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

find_pids() {
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*TraderStuff*main.py*' } | Select-Object -ExpandProperty ProcessId" | tr -d '\r' | grep -v '^$' || true
}

pids=$(find_pids)

if [ -z "$pids" ]; then
  echo "No TraderStuff processes found running -- nothing to stop."
  exit 0
fi

count=$(echo "$pids" | wc -l)
if [ "$count" -ne 2 ]; then
  echo "WARNING: expected 2 processes (crypto + stock), found $count (PID(s): $(echo $pids | tr '\n' ' '))."
  echo "Proceeding to stop whatever was found anyway."
fi

for pid in $pids; do
  echo "Stopping PID $pid..."
  powershell.exe -NoProfile -Command "Stop-Process -Id $pid -ErrorAction SilentlyContinue"
done

echo "Waiting for processes to exit..."
for i in $(seq 1 10); do
  remaining=$(find_pids)
  if [ -z "$remaining" ]; then
    echo "All processes exited after ${i}s."
    break
  fi
  sleep 1
done

remaining=$(find_pids)
if [ -n "$remaining" ]; then
  echo "WARNING: still running, did not exit (PID(s): $(echo $remaining | tr '\n' ' ')):"
  echo "Not force-killing further -- investigate before doing so."
  exit 1
fi

echo ""
echo "=== Final ledger state ==="
.venv/Scripts/python -c "
import json
for asset in ['crypto', 'stock']:
    try:
        d = json.load(open(f'backtrack_logs/paper_ledger_{asset}.json'))
        print(f'{asset}: balance=\${d[\"balance\"]:.2f}, open={len(d[\"open_positions\"])}, closed={len(d[\"closed_trades\"])}')
    except FileNotFoundError:
        pass
"
echo ""
echo "Stopped (forceful -- see script header for why). Run start_system.sh to"
echo "resume -- it reconciles any downtime-window position changes automatically."
