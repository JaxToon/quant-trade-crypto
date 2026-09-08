"""
Is the live/backtest SHORT gap just REGIME?

Six mechanistic hypotheses were tested and rejected for the gap between
backtest SHORT (55.8% WR) and live SHORT (36.8% WR): the GNN signal,
forming-candle entry, a cold analyzer on restart, testnet contamination, FVG
memory truncation, and intra-candle re-evaluation. None explained it.

A seventh explanation was never tested, and it is the most obvious one.
Measured 2026-09-08:

  - the live window (2026-08-13 to 09-08) rose a median 28.74%, 8/8 symbols
  - the backtest's 6 segments are 4 falling, 2 rising

Shorts lose in rallies. If backtest SHORT performance in the BULL segments
resembles live SHORT, the gap needs no mechanism at all — it is the market,
and six hypotheses were chasing something that was never there.

This splits backtest SHORT results by the segment's own direction and
compares each against the live figure.

Usage: python -m experiments.short_by_regime
"""
from __future__ import annotations

import statistics

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

SLIP = 0.00119
LIVE_SHORT_WR = 36.8      # measured, n=38
LIVE_MOVE = 28.74         # median symbol move over the live window


def run_segment(candles, tf, label):
    led = PaperTradingLedger(path=f"backtrack_logs/_sbr_{label}.json",
                             starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
                             stop_slippage_pct=SLIP)
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
    return list(led.closed_trades)


def main():
    tf, n_seg = "1h", 6
    days = YF_MAX_DAYS.get(tf, 730)
    print(f"Fetching {days}d of {tf}...\n", flush=True)
    allc = {}
    for s in settings.WATCHLIST:
        c = fetch_candles(s, days, tf)
        if c:
            allc[s] = c
    n = min(len(c) for c in allc.values())
    b = [round(n * i / n_seg) for i in range(n_seg + 1)]

    rows = []
    for i in range(n_seg):
        part = {s: c[b[i]:b[i + 1]] for s, c in allc.items()}
        moves = [(v[-1].close - v[0].open) / v[0].open * 100 for v in part.values() if len(v) > 1]
        move = statistics.median(moves)
        tr = run_segment(part, tf, f"s{i}")
        sh = [t for t in tr if t.get("direction") == "short"]
        lo = [t for t in tr if t.get("direction") == "long"]
        rows.append((i, move, sh, lo))
        print(f"  seg{i} done (move {move:+.1f}%, {len(sh)} shorts)", flush=True)

    print(f"\n{'=' * 84}")
    print("SHORT PERFORMANCE BY REGIME — does the live gap need a mechanism at all?")
    print(f"{'=' * 84}")
    print(f"{'seg':<5}{'market':>10}{'shorts':>9}{'SHORT WR':>10}{'SHORT PnL':>13}"
          f"{'longs':>8}{'LONG WR':>10}")
    print("-" * 84)
    for i, move, sh, lo in rows:
        swr = sum(1 for t in sh if t["pnl_dollars"] > 0) / len(sh) * 100 if sh else 0
        lwr = sum(1 for t in lo if t["pnl_dollars"] > 0) / len(lo) * 100 if lo else 0
        spnl = sum(t["pnl_dollars"] for t in sh)
        print(f"{i:<5}{move:>9.1f}%{len(sh):>9}{swr:>9.1f}%${spnl:>+12,.0f}{len(lo):>8}{lwr:>9.1f}%")
    print("-" * 84)

    bull = [r for r in rows if r[1] > 10]      # comparable to the live window's +28.7%
    bear = [r for r in rows if r[1] < -10]
    for label, group in (("RISING segments (>+10%)", bull), ("FALLING segments (<-10%)", bear)):
        sh = [t for r in group for t in r[2]]
        if not sh:
            continue
        wr = sum(1 for t in sh if t["pnl_dollars"] > 0) / len(sh) * 100
        print(f"{label:<28}n={len(sh):<5} SHORT WR {wr:>5.1f}%  "
              f"PnL ${sum(t['pnl_dollars'] for t in sh):>+10,.0f}")

    print(f"\n{'LIVE (market +28.7%)':<28}n=38    SHORT WR {LIVE_SHORT_WR:>5.1f}%")
    print(f"\n{'=' * 84}")
    print("VERDICT")
    if bull:
        sh = [t for r in bull for t in r[2]]
        wr = sum(1 for t in sh if t["pnl_dollars"] > 0) / len(sh) * 100
        gap = abs(wr - LIVE_SHORT_WR)
        if gap < 10:
            print(f"  Backtest SHORT in rising markets is {wr:.1f}%, against live's "
                  f"{LIVE_SHORT_WR:.1f}%.")
            print("  The gap collapses once regime is controlled for. Six mechanistic")
            print("  hypotheses were chasing an artefact — the explanation was the market.")
        else:
            print(f"  Backtest SHORT in rising markets is {wr:.1f}% vs live {LIVE_SHORT_WR:.1f}% "
                  f"({gap:.1f}pp apart).")
            print("  Regime explains part of the gap but not all of it.")
    print(f"{'=' * 84}")


if __name__ == "__main__":
    main()
