"""
Is every drawdown number this project has reported understated?

`_max_dd` — used by every drawdown result here, including the headline
"spot config: worst-segment drawdown 9.7%" — walks the list of CLOSED
trades and accumulates realised PnL:

    for t in trades: equity += t["pnl_dollars"]

Two things are wrong with that as a measure of what an account experiences.

  1. UNREALISED LOSSES ARE INVISIBLE. A position 40% underwater contributes
     nothing until it closes. A real account marks to market continuously,
     and its equity dips while the position is still open. The pain that
     makes someone pull their money out happens before the trade closes.

  2. THE ORDERING IS BY CLOSE TIME, NOT CALENDAR TIME. Positions overlap, so
     the sequence of closes is not the sequence the account lived through.
     A trade opened early and closed late lands at the end of the curve.

Both push the same way: the reported figure is a floor, not the real one.

This replays the same window and computes drawdown three ways:

  CLOSED_ONLY  what every result so far used
  MARK_TO_MKT  equity = realised + unrealised, sampled every bar — what an
               account actually shows
  PEAK_INTRABAR same, but using each bar's adverse extreme, i.e. the worst
               the screen ever showed

If MARK_TO_MKT is materially worse, every drawdown conclusion in
RESEARCH_NOTES needs revising — including the one that made the spot
configuration look safe.

Usage: python -m experiments.drawdown_methodology_check
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
# The spot configuration whose 9.7% drawdown this is checking.
RISK, CAP = 0.0025, 1.0


def _dd_from_curve(curve):
    peak = curve[0] if curve else 0.0
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak * 100)
    return worst


def run_segment(candles, tf, label):
    settings.PAPER_TRADING_RISK_PCT = RISK
    settings.PORTFOLIO_MAX_NOTIONAL_MULTIPLE = CAP
    start = settings.PAPER_TRADING_STARTING_BALANCE
    led = PaperTradingLedger(path=f"backtrack_logs/_ddm_{label}.json",
                             starting_balance=start, stop_slippage_pct=SLIP)
    led.balance = led.starting_balance
    led.open_positions.clear(); led.closed_trades.clear()
    led.max_closed_trades = 1_000_000

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in candles}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in candles}
    last_price: dict[str, float] = {}
    mtm_curve, intrabar_curve = [], []

    n = min(len(c) for c in candles.values())
    for t in range(n):
        adverse = {}
        for sym, cs in candles.items():
            c = cs[t]
            last_price[sym] = c.close
            adverse[sym] = (c.low, c.high)
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

        # Mark every open position to market, and to its worst intrabar level.
        unreal = unreal_bad = 0.0
        for p in led.open_positions.values():
            px = last_price.get(p.symbol, p.entry_price)
            lo, hi = adverse.get(p.symbol, (px, px))
            sign = 1 if p.direction == "long" else -1
            unreal += p.quantity * (px - p.entry_price) * sign
            worst_px = lo if p.direction == "long" else hi
            unreal_bad += p.quantity * (worst_px - p.entry_price) * sign
        mtm_curve.append(led.balance + unreal)
        intrabar_curve.append(led.balance + unreal_bad)

    closed_curve = []
    eq = start
    for t_ in led.closed_trades:
        eq += t_["pnl_dollars"]
        closed_curve.append(eq)
    return (_dd_from_curve([start] + closed_curve),
            _dd_from_curve([start] + mtm_curve),
            _dd_from_curve([start] + intrabar_curve),
            len(led.closed_trades))


def main():
    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    print(f"Fetching {days}d of {tf} (spot config: risk {RISK*100:.2f}%, cap {CAP:.1f}x)...\n",
          flush=True)
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
        rows.append((i,) + run_segment(part, tf, f"s{i}"))
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 78}")
    print("DRAWDOWN METHODOLOGY — is the reported number understated?")
    print(f"{'=' * 78}")
    print(f"{'seg':<5}{'trades':<9}{'CLOSED_ONLY':>14}{'MARK_TO_MKT':>14}{'PEAK_INTRABAR':>16}")
    print("-" * 78)
    for i, c_, m, ib, n_ in rows:
        print(f"{i:<5}{n_:<9}{c_:>13.1f}%{m:>13.1f}%{ib:>15.1f}%")
    print("-" * 78)
    cw, mw, iw = (max(r[1] for r in rows), max(r[2] for r in rows), max(r[3] for r in rows))
    cm, mm, im = (statistics.median([r[1] for r in rows]),
                  statistics.median([r[2] for r in rows]),
                  statistics.median([r[3] for r in rows]))
    print(f"{'worst':<14}{cw:>13.1f}%{mw:>13.1f}%{iw:>15.1f}%")
    print(f"{'median':<14}{cm:>13.1f}%{mm:>13.1f}%{im:>15.1f}%")

    print(f"\n{'=' * 78}")
    print("VERDICT")
    if mw > cw * 1.2:
        print(f"  Mark-to-market drawdown ({mw:.1f}%) is materially worse than the")
        print(f"  closed-trade figure ({cw:.1f}%) this project has been reporting.")
        print("  Every drawdown conclusion in RESEARCH_NOTES understates what an")
        print("  account would actually have shown, and needs revising.")
    elif mw > cw:
        print(f"  Mark-to-market ({mw:.1f}%) is somewhat worse than closed-only ({cw:.1f}%),")
        print("  so reported figures are mildly optimistic but not misleading.")
    else:
        print(f"  Closed-only ({cw:.1f}%) is not optimistic here — mark-to-market is")
        print(f"  {mw:.1f}%. The reported drawdowns stand.")
    print(f"  Worst the screen ever showed (intrabar): {iw:.1f}%")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
