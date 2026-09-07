"""
Reconstructs the stop slippage our live system ACTUALLY paid, from trades
already in the ledger — and simulates what the 1s SL/TP monitor would have
paid on those same trades.

Why this exists. The live instrumentation added 2026-09-07 will answer the
question properly, but only at ~2.22 stop exits/day: n=10 around 11 Sep,
n=30 around 20 Sep. Meanwhile the ledger already holds 31 live-detected
stop exits going back to 2026-08-14 whose slippage was computed and thrown
away (`exit_price = effective_stop`, the theoretical level).

It is recoverable. We know each trade's stop and the instant we detected
the breach, and Binance serves 1-SECOND klines back through that window. So
the actual price at detection can be looked up, and the discarded number
reconstructed.

The same data answers the more useful question. Scanning 1s bars backwards
from detection finds when the stop was FIRST breached, which is what the new
monitor would have seen within ~1s. That gives a PAIRED before/after on
identical trades — far stronger than comparing two different time periods.

  OLD (60s decision_loop) : fill at the price observed at detection
  NEW (1s monitor)        : fill ~1s after the stop was first breached

WHAT THIS IS NOT. It models the CHECK-INTERVAL component only. Real fills
also pay order-book depth, market impact and exchange latency, and the NEW
arm additionally assumes the tick reaches us and is acted on within a
second — WebSocket delivery, thread scheduling and the ledger lock are not
modelled. Both arms are therefore optimistic, but they are optimistic in
the SAME way, so the DIFFERENCE between them is the honest output. Treat
the absolute NEW figure as a floor, not a forecast.

Usage: python -m experiments.reconstruct_historical_slippage
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone

import requests

from config import settings

BASE = settings.BINANCE_REST_BASE
LOOKBACK_S = 180   # how far back of 1s bars to scan for the first breach


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _klines_1s(symbol: str, start_ms: int, end_ms: int) -> list:
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": "1s",
                                     "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                             timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return []


def _adverse(direction: str, stop: float, fill: float) -> float:
    """Positive = filled worse than the stop, in both directions."""
    gap = (stop - fill) if direction == "long" else (fill - stop)
    return gap / stop * 100


def main() -> None:
    ledger = json.load(open("backtrack_logs/paper_ledger_crypto.json"))
    trades = ledger["closed_trades"]

    # Stop exits detected LIVE. Downtime-reconciled rows close on a bar
    # boundary (whole seconds, no microseconds) and were resolved from OHLC
    # after the fact, so they never had a detection instant to reconstruct.
    stops = [
        t for t in trades
        if ((t.get("exit_type") == "stop") or (t.get("exit_type") is None and t.get("outcome") == "loss"))
        and "." in t.get("closed_at", "")
    ]
    print(f"Reconstructing {len(stops)} live-detected stop exits "
          f"({stops[0]['closed_at'][:10]} to {stops[-1]['closed_at'][:10]})\n", flush=True)

    rows = []
    for t in stops:
        sym, direction = t["symbol"], t["direction"]
        stop = t["exit_price"]          # recorded AT the theoretical stop
        det = _ms(t["closed_at"])
        bars = _klines_1s(sym, det - LOOKBACK_S * 1000, det + 2000)
        if not bars:
            print(f"  {t['closed_at'][:19]} {sym:<9} no data", flush=True)
            continue

        # price at detection = close of the 1s bar covering that instant
        at_det = None
        for b in bars:
            if b[0] <= det < b[0] + 1000:
                at_det = float(b[4])
                break
        if at_det is None:
            at_det = float(bars[-1][4])

        # first breach: earliest 1s bar whose adverse extreme crosses the stop
        first_breach_fill = None
        for b in bars:
            if b[0] > det:
                break
            adverse_extreme = float(b[3]) if direction == "long" else float(b[2])
            crossed = adverse_extreme <= stop if direction == "long" else adverse_extreme >= stop
            if crossed:
                # what a monitor reacting within ~1s would fill at
                first_breach_fill = float(b[4])
                break
        if first_breach_fill is None:
            first_breach_fill = at_det

        old = _adverse(direction, stop, at_det)
        new = _adverse(direction, stop, first_breach_fill)
        rows.append({"symbol": sym, "direction": direction, "closed_at": t["closed_at"],
                     "old": old, "new": new, "pnl": t.get("pnl_dollars", 0.0)})
        print(f"  {t['closed_at'][:19]} {sym:<9}{direction:<6} "
              f"old={old:+7.3f}%  new={new:+7.3f}%", flush=True)
        time.sleep(0.15)

    if not rows:
        print("\nNo rows reconstructed.")
        return

    olds = [r["old"] for r in rows]
    news = [r["new"] for r in rows]
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(len(v) * p))]

    print(f"\n{'=' * 74}")
    print(f"RECONSTRUCTED STOP SLIPPAGE — n={len(rows)} paired live trades")
    print(f"{'=' * 74}")
    print(f"{'Arm':<26}{'mean':<10}{'median':<10}{'p90':<10}{'max':<10}")
    print("-" * 74)
    print(f"{'OLD (60s decision_loop)':<26}{statistics.mean(olds):<10.3f}"
          f"{statistics.median(olds):<10.3f}{q(olds,.90):<10.3f}{max(olds):<10.3f}")
    print(f"{'NEW (1s monitor)':<26}{statistics.mean(news):<10.3f}"
          f"{statistics.median(news):<10.3f}{q(news,.90):<10.3f}{max(news):<10.3f}")
    print("-" * 74)

    if statistics.mean(news) != 0:
        print(f"\nMean slippage reduction: {statistics.mean(olds)/statistics.mean(news):.1f}x")
    print(f"Break-even threshold: 0.51%")
    print(f"  OLD mean {statistics.mean(olds):.3f}% — "
          f"{'ABOVE' if statistics.mean(olds) >= 0.51 else 'below'} break-even")
    print(f"  NEW mean {statistics.mean(news):.3f}% — "
          f"{'ABOVE' if statistics.mean(news) >= 0.51 else 'below'} break-even")

    worse = sum(1 for r in rows if r["new"] > r["old"])
    print(f"\nTrades where the 1s monitor would have done WORSE: {worse}/{len(rows)}")
    print("(Expected to be small but non-zero — reacting faster means sometimes")
    print(" catching a spike that the slower check slept through.)")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
