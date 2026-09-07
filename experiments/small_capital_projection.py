"""
What would $20 actually do in one month — and is it even tradeable?

Two separate questions, and the second one usually kills the first.

1. STATISTICAL: given the live per-trade PnL distribution we have actually
   observed, what is the spread of one-month outcomes on a small stake?
   Block bootstrap (contiguous runs of trades) so the serial correlation
   between overlapping positions is preserved — a plain resample would make
   the spread look far tighter than it is.

2. MECHANICAL: can the exchange even accept the orders? Position sizing
   risks PAPER_TRADING_RISK_PCT of capital per trade, and quantity is
   risk / stop_distance. On a tiny stake that produces order notionals that
   may fall under the venue's MIN_NOTIONAL, and fees become a large
   fraction of the risk budget. The ledger already enforces
   MIN_STOP_DISTANCE_PCT for exactly this reason (fees must stay under
   MAX_FEE_TO_RISK_RATIO of the risk unit), so some setups get rejected
   outright rather than traded.

Uses live crypto trades only — real observed results, not backtest, since
the backtest is known not to describe live SHORT behaviour.

Usage: python -m experiments.small_capital_projection [--capital 20]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

from config import settings
from backtrack.paper_trading import MAX_FEE_TO_RISK_RATIO, MIN_STOP_DISTANCE_PCT

random.seed(11)
BOOT = 20_000
BLOCK = 5
LIVE_START = 10_000.0      # the balance the live ledger's PnL figures are sized against
TYPICAL_MIN_NOTIONAL = 5.0  # Binance/Tokocrypto spot MIN_NOTIONAL, USD, order of magnitude


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    cap = args.capital

    d = json.load(open("backtrack_logs/paper_ledger_crypto.json"))
    tr = [t for t in d["closed_trades"] if t.get("closed_at")]
    tr.sort(key=lambda t: t["closed_at"])
    if len(tr) < BLOCK:
        print("not enough live trades"); return

    from datetime import datetime
    P = lambda s: datetime.fromisoformat(s).replace(tzinfo=None)
    span = (P(tr[-1]["closed_at"]) - P(tr[0]["closed_at"])).days or 1
    per_day = len(tr) / span
    n_month = max(1, round(per_day * args.days))

    # Each trade's PnL as a fraction of the account it was sized against.
    fracs = [t["pnl_dollars"] / LIVE_START for t in tr]

    print(f"{'=' * 84}")
    print(f"${cap:,.0f} FOR {args.days} DAYS — projected from {len(tr)} real live trades")
    print(f"{'=' * 84}")
    print(f"live history: {len(tr)} trades over {span} days = {per_day:.1f}/day")
    print(f"  -> ~{n_month} trades in {args.days} days")
    print(f"observed mean per trade: {statistics.mean(fracs)*100:+.3f}% of capital")

    # ---- 1. statistical spread ----
    finals = []
    nblocks = max(1, n_month // BLOCK)
    for _ in range(BOOT):
        eq = cap
        low = cap
        for _ in range(nblocks):
            s = random.randrange(0, len(fracs) - BLOCK + 1)
            for f in fracs[s:s + BLOCK]:
                eq += cap * f          # fixed-fraction of STARTING capital, as the system sizes
                low = min(low, eq)
        finals.append((eq, low))
    ends = sorted(e for e, _ in finals)
    lows = sorted(l for _, l in finals)
    q = lambda v, p: v[min(len(v) - 1, int(len(v) * p))]

    print(f"\nOUTCOME SPREAD after {args.days} days ({BOOT:,} block-bootstrap paths)")
    print(f"  {'worst 5%':<12}${q(ends,0.05):>8,.2f}   ({(q(ends,0.05)/cap-1)*100:+.0f}%)")
    print(f"  {'25th pct':<12}${q(ends,0.25):>8,.2f}   ({(q(ends,0.25)/cap-1)*100:+.0f}%)")
    print(f"  {'MEDIAN':<12}${q(ends,0.50):>8,.2f}   ({(q(ends,0.50)/cap-1)*100:+.0f}%)")
    print(f"  {'75th pct':<12}${q(ends,0.75):>8,.2f}   ({(q(ends,0.75)/cap-1)*100:+.0f}%)")
    print(f"  {'best 5%':<12}${q(ends,0.95):>8,.2f}   ({(q(ends,0.95)/cap-1)*100:+.0f}%)")
    losing = sum(1 for e in ends if e < cap) / len(ends) * 100
    print(f"\n  chance of ending BELOW ${cap:,.0f}: {losing:.0f}%")
    print(f"  typical worst point along the way (median of path minimums): ${statistics.median(lows):,.2f}")

    # ---- 2. mechanical feasibility ----
    risk = cap * settings.PAPER_TRADING_RISK_PCT
    print(f"\n{'=' * 84}")
    print("CAN THE EXCHANGE EVEN TAKE THESE ORDERS?")
    print(f"{'=' * 84}")
    print(f"  risk per trade  = {settings.PAPER_TRADING_RISK_PCT*100:.0f}% of ${cap:,.0f} = ${risk:.3f}")
    print(f"  minimum stop distance enforced = {MIN_STOP_DISTANCE_PCT*100:.2f}% "
          f"(keeps fees under {MAX_FEE_TO_RISK_RATIO*100:.0f}% of risk)")
    for stop_pct in (0.5, 1.0, 2.0):
        notional = risk / (stop_pct / 100)
        fee = notional * settings.PAPER_TRADING_FEE_PCT * 2
        ok = "OK" if notional >= TYPICAL_MIN_NOTIONAL else f"REJECTED (< ${TYPICAL_MIN_NOTIONAL:.0f} min)"
        print(f"    stop {stop_pct:>4.1f}% -> notional ${notional:>7.2f}  round-trip fee ${fee:.4f} "
              f"({fee/risk*100:>4.1f}% of risk)  {ok}")
    print(f"\n  Note the system runs ~3 positions concurrently, so each needs its own")
    print(f"  notional out of the same ${cap:,.0f}.")
    print(f"{'=' * 84}")


if __name__ == "__main__":
    main()
