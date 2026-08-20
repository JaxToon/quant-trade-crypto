#!/usr/bin/env bash
# Gracefully stops both TraderStuff processes (SIGTERM, same as done
# manually all session) and waits for them to actually exit before
# reporting done, so the ledger files are guaranteed to hold their final
# saved state (main.py saves on SIGTERM) before anything else touches them.
#
# Usage: bash stop_system.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pids=$(ps aux | grep -i "\.venv/Scripts/python" | grep -v grep | awk '{print $1}')

if [ -z "$pids" ]; then
  echo "No TraderStuff processes found running -- nothing to stop."
  exit 0
fi

count=$(echo "$pids" | wc -l)
if [ "$count" -ne 2 ]; then
  echo "WARNING: expected 2 processes (crypto + stock), found $count:"
  ps aux | grep -i "\.venv/Scripts/python" | grep -v grep
  echo "Proceeding to stop whatever was found anyway."
fi

for pid in $pids; do
  echo "Sending SIGTERM to PID $pid..."
  kill -TERM "$pid" 2>/dev/null || echo "  (already gone)"
done

echo "Waiting for graceful shutdown..."
for i in $(seq 1 20); do
  alive=""
  for pid in $pids; do
    kill -0 "$pid" 2>/dev/null && alive="$alive $pid"
  done
  if [ -z "$alive" ]; then
    echo "All processes exited after ${i}s."
    break
  fi
  sleep 1
done

remaining=$(ps aux | grep -i "\.venv/Scripts/python" | grep -v grep || true)
if [ -n "$remaining" ]; then
  echo "WARNING: still running after 20s, did not exit gracefully:"
  echo "$remaining"
  echo "Not force-killing -- investigate before doing so."
  exit 1
fi

echo ""
echo "=== Final ledger state ==="
python -c "
import json
for asset in ['crypto', 'stock']:
    try:
        d = json.load(open(f'backtrack_logs/paper_ledger_{asset}.json'))
        print(f'{asset}: balance=\${d[\"balance\"]:.2f}, open={len(d[\"open_positions\"])}, closed={len(d[\"closed_trades\"])}')
    except FileNotFoundError:
        pass
"
echo ""
echo "Stopped cleanly. Run start_system.sh to resume -- it will reconcile"
echo "downtime automatically before restarting."
