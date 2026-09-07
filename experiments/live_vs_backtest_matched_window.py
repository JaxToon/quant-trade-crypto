"""
Does the backtest actually describe the live system? Matched-window,
per-direction comparison.

This is the question everything else rests on. Every real-money plan in
this project is justified by backtest numbers, and the ledger now shows a
direct contradiction with them:

  LIVE crypto (n=84, pre-fix)   LONG  49 trades, 75.5% WR, +$10,273
                                SHORT 35 trades, 37.1% WR,  -$6,697

SHORT burns two thirds of what LONG earns. Yet RESEARCH_NOTES records a
decision from 2026-08-30 to KEEP short because "LONG_ONLY robustly loses"
in backtest, even with a real GNN signal. Both cannot be true.

The obvious confound is that live covers ~25 days of a single regime (the
logs show HIGH_STRESS_TURBULENT with the bear veto active throughout),
while the backtest spans 730 days and 6 regimes. So this replays the
backtest over the EXACT live window, with the same symbols and the same
three concurrent timeframe layers, and compares per direction.

  If backtest ALSO shows SHORT losing here -> the system is internally
  consistent and this was simply a bad stretch for shorts. The 730d
  decision stands, and live/backtest agreement is evidence the backtest
  can be trusted.

  If backtest shows SHORT PROFITABLE while live lost -> the backtest does
  not describe the live system, and no real-money plan built on backtest
  numbers is trustworthy until that is explained.

Data comes from Binance directly (settings.BINANCE_REST_BASE) rather than
yfinance: we need an exact date window at 5m/15m/1h for crypto pairs, and
yfinance caps intraday history at 60 days with coarser coverage.

Candles are fetched with WARMUP_BARS of extra history before the window so
SmartMoneyAnalyzer is fully primed; trades opened during warmup are
excluded from the comparison.

Usage: python -m experiments.live_vs_backtest_matched_window
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

TIMEFRAMES = ["5m", "15m", "1h"]
WARMUP_BARS = 150          # > SMC_LOOKBACK_CANDLES so the analyzer is primed
TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
LIVE_LEDGER = "backtrack_logs/paper_ledger_crypto.json"


def _fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Candle]:
    """Paginated Binance klines -> Candle list."""
    out, cursor = [], start_ms
    while cursor < end_ms:
        try:
            r = requests.get(f"{settings.BINANCE_REST_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": interval,
                                     "startTime": cursor, "endTime": end_ms, "limit": 1000},
                             timeout=20)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            for b in batch:
                out.append(Candle(
                    timestamp=datetime.utcfromtimestamp(b[0] / 1000),
                    open=float(b[1]), high=float(b[2]), low=float(b[3]),
                    close=float(b[4]), volume=float(b[5]),
                ))
            cursor = batch[-1][0] + TF_MS[interval]
            time.sleep(0.12)
        except Exception:
            break
    return out


def _live_window() -> tuple[int, int, list[dict]]:
    d = json.load(open(LIVE_LEDGER))
    trades = [t for t in d["closed_trades"] if t.get("opened_at") and t.get("closed_at")]
    lo = min(t["opened_at"] for t in trades)
    hi = max(t["closed_at"] for t in trades)
    to_ms = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)
    return to_ms(lo), to_ms(hi), trades


def _summarise(trades, label):
    if not trades:
        return f"{label:<26} (no trades)"
    w = sum(1 for t in trades if t["pnl_dollars"] > 0)
    pnl = sum(t["pnl_dollars"] for t in trades)
    wins = [t["pnl_dollars"] for t in trades if t["pnl_dollars"] > 0]
    loss = [t["pnl_dollars"] for t in trades if t["pnl_dollars"] <= 0]
    pf = (sum(wins) / abs(sum(loss))) if loss and sum(loss) != 0 else float("inf")
    return (f"{label:<26}{len(trades):<8}{w/len(trades)*100:>6.1f}%  "
            f"${pnl:>+11,.0f}  ${pnl/len(trades):>+9,.2f}  {pf:>6.2f}")


def main() -> None:
    start_ms, end_ms, live_trades = _live_window()
    print(f"Live window: {datetime.utcfromtimestamp(start_ms/1000)} -> "
          f"{datetime.utcfromtimestamp(end_ms/1000)}  ({(end_ms-start_ms)/86400000:.1f} days)")
    print(f"Live trades in ledger: {len(live_trades)}\n", flush=True)

    # Replay each timeframe layer into ONE ledger, as live does.
    led = PaperTradingLedger(path="backtrack_logs/_lvb_matched.json",
                             starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    led.balance = led.starting_balance
    led.open_positions.clear(); led.closed_trades.clear()
    led.max_closed_trades = 1_000_000

    series: dict[tuple[str, str], list[Candle]] = {}
    for tf in TIMEFRAMES:
        warm = start_ms - WARMUP_BARS * TF_MS[tf]
        for sym in settings.WATCHLIST:
            c = _fetch(sym, tf, warm, end_ms)
            if c:
                series[(sym, tf)] = c
                print(f"  {sym:<9}{tf:<5}{len(c):>6} candles", flush=True)

    if not series:
        print("No data fetched — aborting.")
        return

    # Merge every (symbol, timeframe) stream into one chronological pass so
    # concurrent layers compete for the same portfolio risk cap, as in live.
    events = []
    for (sym, tf), candles in series.items():
        for c in candles:
            events.append((c.timestamp, sym, tf, c))
    events.sort(key=lambda e: e[0])

    smc = {k: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for k in series}
    fvg = {k: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for k in series}
    window_start = datetime.utcfromtimestamp(start_ms / 1000)

    print(f"\nReplaying {len(events):,} candle events...", flush=True)
    for ts, sym, tf, c in events:
        k = (sym, tf)
        sig = smc[k].update(c)
        hist = list(smc[k].candles)
        if len(hist) >= 3:
            fvg[k].detect_fvg(candles=hist, index=len(hist) - 2, current_price=c.close, volume=c.volume)
        fvg[k].update_all(c.close)
        ctx = {**sig, **fvg[k].get_trading_signal(current_price=c.close,
               trend=sig.get("market_structure", "unknown")), "timestamp": ts.isoformat()}

        led.check_and_close_ohlc(sym, c.low, c.high, closed_at=ts.isoformat(), timeframe=tf)
        if ts < window_start:
            continue                      # warmup: prime analyzers, take no trades
        ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)
        led.maybe_open(sym, "spot", ins["spot"], timeframe=tf, opened_at=ts.isoformat())
        led.maybe_open(sym, "derivative", ins["derivative"], timeframe=tf, opened_at=ts.isoformat())

    bt = list(led.closed_trades)

    print(f"\n{'=' * 84}")
    print("LIVE vs BACKTEST — same window, same symbols, same 3 timeframe layers")
    print(f"{'=' * 84}")
    print(f"{'Arm':<26}{'n':<8}{'WR':>7}  {'PnL':>12}  {'exp/trade':>10}  {'PF':>6}")
    print("-" * 84)
    print(_summarise(live_trades, "LIVE  all"))
    print(_summarise(bt, "BACKTEST  all"))
    print("-" * 84)
    for dirn in ("long", "short"):
        print(_summarise([t for t in live_trades if t.get("direction") == dirn], f"LIVE  {dirn}"))
        print(_summarise([t for t in bt if t.get("direction") == dirn], f"BACKTEST  {dirn}"))
        print("-" * 84)

    lv_s = [t for t in live_trades if t.get("direction") == "short"]
    bt_s = [t for t in bt if t.get("direction") == "short"]
    if lv_s and bt_s:
        lp = sum(t["pnl_dollars"] for t in lv_s)
        bp = sum(t["pnl_dollars"] for t in bt_s)
        print("\nVERDICT on SHORT over this window:")
        if lp < 0 and bp < 0:
            print("  Both LOSE -> internally consistent. This was simply a bad stretch for")
            print("  shorts; the 730d/6-regime decision to keep SHORT is not contradicted.")
        elif lp < 0 <= bp:
            print("  Backtest PROFITS while live LOSES -> the backtest does not describe the")
            print("  live system. No real-money plan resting on backtest numbers is")
            print("  trustworthy until this is explained. THIS IS THE BLOCKER.")
        else:
            print(f"  live ${lp:+,.0f} vs backtest ${bp:+,.0f} — read the table above.")
    print(f"{'=' * 84}")
    print("\nCaveats, so this isn't over-read: live entered intra-candle on ticks for")
    print("part of this window (the closed-candle gating fix landed 2026-09-03), live")
    print("suffered outages that this replay does not, and live trade COUNT reflects")
    print("whatever the process was actually up for. Direction of the edge is the")
    print("signal here, not an exact trade-for-trade match.")


if __name__ == "__main__":
    main()
