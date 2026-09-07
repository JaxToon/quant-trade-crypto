"""
Reports the live execution slippage recorded by the paper ledgers.

Why this exists. `experiments/breakeven_under_slippage.py` put a number on
something the project had never priced: this system's backtested edge dies
somewhere around 0.51% stop slippage (baseline PnL +$288,902 at 0.00%,
+$63,640 at 0.40%, -$48,991 at 0.60%). The only real measurement available
at the time -- 4 live SHORT stop-outs, reconstructed by hand -- put actual
fills 0.25-0.60% past their trigger. That band straddles the break-even
line, so "is this system profitable" is currently an open question rather
than a settled one, and no amount of further backtesting can close it.

`check_and_close` was instrumented on 2026-09-07 to record, on every live
exit, the trigger level, the observed price, and the signed gap between
them. This script turns those rows into the answer.

READ THE CAVEAT BEFORE ACTING ON THE OUTPUT. What is measured here is the
price move between a stop being breached and the next poll observing it.
That is a LOWER BOUND on real execution slippage: a real fill also pays
order-book depth, market impact and exchange latency, none of which paper
trading can observe. So the true cost is this number OR WORSE -- never
better. Which is the useful direction: if the lower bound alone is already
near 0.5%, the verdict is in without needing the rest.

Rows from OHLC backtests carry no observed price and are excluded rather
than filled in from a bar extreme.

Usage: python slippage_report.py [--asset-type crypto|stock|both]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

BREAKEVEN_PCT = 0.51  # where backtested baseline PnL crosses zero


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))]


def _load(asset_type: str) -> list[dict]:
    path = os.path.join("backtrack_logs", f"paper_ledger_{asset_type}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("closed_trades", [])


def report(asset_type: str) -> None:
    trades = _load(asset_type)
    measured = [t for t in trades if t.get("observed_slippage_pct") is not None]
    stops = [t for t in measured if t.get("exit_type") in ("stop", "liquidation")]

    print(f"\n{'=' * 78}")
    print(f"LIVE SLIPPAGE — {asset_type.upper()}")
    print(f"{'=' * 78}")
    print(f"closed trades: {len(trades)}   with measurement: {len(measured)}   stop-type: {len(stops)}")

    if not stops:
        print("\nNo instrumented stop exits recorded yet.")
        print("Only exits closed AFTER the 2026-09-07 instrumentation carry this data;")
        print("everything before it was booked at the trigger price and cannot be recovered.")
        print(f"{'=' * 78}")
        return

    vals = [t["observed_slippage_pct"] for t in stops]
    mean = statistics.mean(vals)
    print(f"\nAdverse slippage on stop exits (positive = filled worse than trigger):")
    print(f"  mean {mean:+.3f}%   median {statistics.median(vals):+.3f}%   "
          f"p95 {_pct(vals, 0.95):+.3f}%   max {max(vals):+.3f}%")
    adverse = sum(1 for v in vals if v > 0)
    print(f"  adverse on {adverse}/{len(vals)} ({adverse / len(vals) * 100:.0f}%)")

    for direction in ("long", "short"):
        d = [t["observed_slippage_pct"] for t in stops if t.get("direction") == direction]
        if d:
            print(f"  {direction:<6}: n={len(d):<4} mean {statistics.mean(d):+.3f}%   median {statistics.median(d):+.3f}%")

    print(f"\nVERDICT vs the ~{BREAKEVEN_PCT:.2f}% break-even threshold:")
    if len(vals) < 30:
        print(f"  n={len(vals)} is too small to conclude anything. Target ~100 stop exits.")
        print(f"  (Running estimate {mean:+.3f}% — do not act on it yet.)")
    elif mean >= BREAKEVEN_PCT:
        print(f"  mean {mean:.3f}% is AT OR ABOVE break-even, and this is a LOWER BOUND.")
        print("  The strategy does not survive its own execution cost. Real money is off")
        print("  the table until either the cost or the strategy changes.")
    elif mean >= BREAKEVEN_PCT * 0.6:
        print(f"  mean {mean:.3f}% leaves little headroom below {BREAKEVEN_PCT:.2f}%, and the true")
        print("  cost is this or worse. Treat the edge as unproven.")
    else:
        print(f"  mean {mean:.3f}% sits meaningfully below {BREAKEVEN_PCT:.2f}%. The edge survives")
        print("  the measurable component of cost — necessary, not yet sufficient.")
    print(f"{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-type", default="both", choices=["crypto", "stock", "both"])
    args = ap.parse_args()
    for at in (["crypto", "stock"] if args.asset_type == "both" else [args.asset_type]):
        report(at)


if __name__ == "__main__":
    main()
