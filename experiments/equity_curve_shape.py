"""
What does the equity curve actually LOOK like — and does a 68% drawdown
mean losing your capital, or giving back profits?

This exists to correct a claim I made carelessly: that a 68.5% drawdown
means "$1,000 becomes ~$320". That is only true if the drawdown happens
BEFORE the account has grown. Max drawdown is measured from the running
PEAK, so a 68% drawdown on an account that had already tripled means
falling from $30,000 to $9,600 — painful, but not the loss of the original
stake. The two readings are wildly different propositions for someone
deciding whether to fund this, and the difference is entirely about WHEN
the drawdown lands.

So: replay the baseline and record the actual curve per segment — starting
capital, running peak, the lowest point ever reached, the final balance,
and critically whether equity ever fell BELOW the initial deposit.

  "worst dip vs START"  — did the original stake ever go underwater, and
                          by how much? This is the number that answers
                          "could I lose my money?"
  "worst dip vs PEAK"   — the conventional max drawdown, which answers
                          "how much of my gains could evaporate?"

A CAVEAT THAT MATTERS MORE THAN THE HEADLINE. This backtest risks a FIXED
$200 per trade sized off the STARTING balance, never off current equity
(see `_sizing_base`). So as the account grows the risk fraction shrinks
toward nothing, and the returns below are what you would get while
deliberately refusing to compound. Real deployment would either compound
(much larger returns AND much larger drawdowns) or hold the fraction fixed
(different curve again). Treat the SHAPE and the ratio as the signal here,
not the headline percentage.

Usage: python -m experiments.equity_curve_shape
"""
from __future__ import annotations

import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SLIP = 0.00119
START = settings.PAPER_TRADING_STARTING_BALANCE


def run_segment(candles, tf, label):
    led = PaperTradingLedger(path=f"backtrack_logs/_ecs_{label}.json",
                             starting_balance=START, stop_slippage_pct=SLIP)
    led.balance = led.starting_balance
    led.open_positions.clear(); led.closed_trades.clear()
    led.max_closed_trades = 1_000_000

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in candles}

    n = min(len(c) for c in candles.values())
    for t in range(n):
        for sym, cs in candles.items():
            c = cs[t]
            sig = smc[sym].update(c)
            hist = list(smc[sym].candles)
            if len(hist) >= 3:
                fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2,
                                    current_price=c.close, volume=c.volume)
            fvg[sym].update_all(c.close)
            ctx = {**sig, **fvg[sym].get_trading_signal(
                current_price=c.close, trend=sig.get("market_structure", "unknown")),
                "timestamp": c.timestamp.isoformat()}
            ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)
            at = c.timestamp.isoformat()
            led.check_and_close_ohlc(sym, c.low, c.high, closed_at=at, timeframe=tf)
            led.maybe_open(sym, "spot", ins["spot"], timeframe=tf, opened_at=at)
            led.maybe_open(sym, "derivative", ins["derivative"], timeframe=tf, opened_at=at)
    return sorted(led.closed_trades, key=lambda t: t.get("closed_at", ""))


def main():
    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    print(f"Fetching {days}d of {tf}...\n", flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    if not allc:
        print("no data"); return

    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]

    rows = []
    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        tr = run_segment(part, tf, f"s{i}")
        eq = peak = low = START
        dd_peak = 0.0
        for t in tr:
            eq += t["pnl_dollars"]
            peak = max(peak, eq)
            low = min(low, eq)
            if peak > 0:
                dd_peak = max(dd_peak, (peak - eq) / peak * 100)
        rows.append({"seg": i, "n": len(tr), "peak": peak, "low": low, "final": eq,
                     "dd_peak": dd_peak, "dip_vs_start": (START - low) / START * 100})
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 96}")
    print(f"EQUITY CURVE SHAPE — {days}d, {n_seg} segments, each starting at ${START:,.0f}")
    print(f"{'=' * 96}")
    print(f"{'seg':<5}{'trades':<8}{'peak':>12}{'lowest':>12}{'final':>12}"
          f"{'return':>9}{'DD vs peak':>12}{'dip vs start':>14}")
    print("-" * 96)
    for r in rows:
        dip = f"{r['dip_vs_start']:.1f}%" if r["dip_vs_start"] > 0 else "never"
        print(f"{r['seg']:<5}{r['n']:<8}${r['peak']:>11,.0f}${r['low']:>11,.0f}${r['final']:>11,.0f}"
              f"{(r['final']-START)/START*100:>8.0f}%{r['dd_peak']:>11.1f}%{dip:>14}")
    print("-" * 96)

    rets = [(r["final"] - START) / START * 100 for r in rows]
    dds = [r["dd_peak"] for r in rows]
    dips = [r["dip_vs_start"] for r in rows]
    print(f"median return {statistics.median(rets):.0f}%   "
          f"median DD-vs-peak {statistics.median(dds):.1f}%   "
          f"worst dip below starting capital {max(dips):.1f}%")
    print(f"return / drawdown ratio (median): {statistics.median(rets)/statistics.median(dds):.1f}x")

    underwater = [r for r in rows if r["dip_vs_start"] > 0]
    print(f"\nSegments where the ORIGINAL stake ever went underwater: {len(underwater)}/{n_seg}")
    if underwater:
        print(f"  worst: seg{max(underwater, key=lambda r: r['dip_vs_start'])['seg']} "
              f"fell {max(dips):.1f}% below the starting deposit")
    else:
        print("  None. Every big drawdown was giving back PROFITS, not losing the stake —")
        print("  the account never dropped below what was originally put in.")

    print(f"\n{'=' * 96}")
    print("CAVEAT: fixed $200 risk sized off the STARTING balance, never compounded.")
    print("As equity grows the risk fraction shrinks toward zero, so these returns are")
    print("what you get while deliberately NOT compounding. Read the shape and the")
    print("return/drawdown ratio, not the headline percentage.")
    print(f"{'=' * 96}")


if __name__ == "__main__":
    main()
