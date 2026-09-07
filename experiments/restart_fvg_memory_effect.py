"""
Does restarting the process truncate FVG memory enough to change WHICH
setups get taken — and does that hit shorts hardest?

Established: the live/backtest SHORT gap is a SELECTION problem, not an
execution one (short_setup_attribution.py). On the 9 shorts both systems
took, outcomes agreed 8/9 and entry prices differed by a median 0.047%. But
only 24% of live shorts matched at all: live took 28 shorts the backtest
never proposed (-$4,227, 39% WR) and missed 43 the backtest did (+$4,093,
60% WR). The two systems are simply not looking at the same setups.

Ruled out along the way: the GNN (gated out in both arms), forming-candle
entry (simulated; if anything it helps shorts), a cold analyzer on restart
(`_backfill_history` correctly rebuilds BOTH the SMC and FVG analyzers),
and testnet contamination (the backfill client returns production data —
verified against a live production tick to the cent).

What survives is subtler. `_backfill_history` restores exactly
SMC_LOOKBACK_CANDLES (100) bars. That is complete for SmartMoneyAnalyzer,
which only keeps 100 anyway — but NOT for FairValueGapAnalyzer, whose gaps
persist until filled or invalidated and can therefore stay active far longer
than 100 candles. Every restart silently discards every FVG that cannot be
re-derived from the last 100 bars.

The live process restarted 39 times during the compared window (counted from
"SMC backfilled BTCUSDT [1h]" in the logs) — roughly every 15 hours. So live
ran with a permanently shallower FVG memory than any continuous backtest,
and FVG signals are a primary driver of entry_insight.

This tests exactly that: identical replay, but one arm periodically drops its
analyzer state and re-warms from only the last 100 candles, the way a restart
does.

  CONTINUOUS — one unbroken run (what every backtest in this repo does)
  RESTART_15 — reset + 100-candle re-warm every 15 candles (live's observed
               cadence at 1h)
  RESTART_50 — a gentler cadence, to see whether the effect scales with
               restart frequency rather than being an artifact of one number

PRE-REGISTERED: the hypothesis is supported only if RESTART arms degrade
SHORT materially more than LONG (>2x the win-rate drop, and at least 5pp).
Equal damage across directions means restarts matter but do not explain the
ASYMMETRY, and the search continues elsewhere. Improvement, or noise-level
differences, falsifies it outright.

Usage: python -m experiments.restart_fvg_memory_effect [--days 90] [--segments 3]
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import requests

from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

TF, TF_MS = "1h", 3_600_000
WARMUP = 150
BACKFILL_BARS = 100          # == settings.SMC_LOOKBACK_CANDLES, what a restart restores
ARMS = {"CONTINUOUS": None, "RESTART_15": 15, "RESTART_50": 50}


def _fetch(symbol, start_ms, end_ms):
    out, cursor = [], start_ms
    while cursor < end_ms:
        try:
            r = requests.get(f"{settings.BINANCE_REST_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": TF,
                                     "startTime": cursor, "endTime": end_ms, "limit": 1000},
                             timeout=20)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            for b in batch:
                out.append(Candle(timestamp=datetime.utcfromtimestamp(b[0] / 1000),
                                  open=float(b[1]), high=float(b[2]), low=float(b[3]),
                                  close=float(b[4]), volume=float(b[5])))
            cursor = batch[-1][0] + TF_MS
            time.sleep(0.1)
        except Exception:
            break
    return out


def _fresh(sym_history):
    """A restarted analyzer pair: rebuilt from the last BACKFILL_BARS only,
    exactly as _backfill_history does."""
    smc = SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES)
    fvg = FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02)
    for c in sym_history[-BACKFILL_BARS:]:
        smc.update(c)
        cs = list(smc.candles)
        if len(cs) >= 3:
            fvg.detect_fvg(candles=cs, index=len(cs) - 2, current_price=c.close, volume=c.volume)
    if smc.candles:
        fvg.update_all(smc.candles[-1].close)
    return smc, fvg


def _summarise(trades, label):
    if not trades:
        return f"{label:<22} (no trades)"
    w = sum(1 for t in trades if t["pnl_dollars"] > 0)
    pnl = sum(t["pnl_dollars"] for t in trades)
    return f"{label:<22}{len(trades):<7}{w/len(trades)*100:>6.1f}%  ${pnl:>+11,.0f}  ${pnl/len(trades):>+9,.2f}"


def run_segment(candles, label):
    out = {}
    for name, cadence in ARMS.items():
        led = PaperTradingLedger(path=f"backtrack_logs/_rfm_{label}_{name}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000

        smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in candles}
        fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in candles}
        seen = {s: [] for s in candles}

        n = min(len(c) for c in candles.values())
        for t in range(n):
            if cadence and t > 0 and t % cadence == 0:
                for s in candles:
                    smc[s], fvg[s] = _fresh(seen[s])
            for s, cs in candles.items():
                c = cs[t]
                seen[s].append(c)
                sig = smc[s].update(c)
                hist = list(smc[s].candles)
                if len(hist) >= 3:
                    fvg[s].detect_fvg(candles=hist, index=len(hist) - 2,
                                      current_price=c.close, volume=c.volume)
                fvg[s].update_all(c.close)
                ctx = {**sig, **fvg[s].get_trading_signal(
                    current_price=c.close, trend=sig.get("market_structure", "unknown")),
                    "timestamp": c.timestamp.isoformat()}
                at = c.timestamp.isoformat()
                led.check_and_close_ohlc(s, c.low, c.high, closed_at=at, timeframe=TF)
                if t < WARMUP:
                    continue
                ins = build_entry_insight(s, c.close, ctx, gnn_signal=0.0)
                led.maybe_open(s, "spot", ins["spot"], timeframe=TF, opened_at=at)
                led.maybe_open(s, "derivative", ins["derivative"], timeframe=TF, opened_at=at)
        out[name] = list(led.closed_trades)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--segments", type=int, default=3)
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"Fetching {args.days}d of {TF}...\n", flush=True)
    cand = {}
    for s in settings.WATCHLIST:
        c = _fetch(s, int(start.timestamp() * 1000), int(end.timestamp() * 1000))
        if c:
            cand[s] = c
            print(f"  {s:<9}{len(c):>6} candles", flush=True)
    if not cand:
        print("no data"); return

    n = min(len(c) for c in cand.values())
    b = [round(n * i / args.segments) for i in range(args.segments + 1)]
    agg = {k: [] for k in ARMS}
    for i in range(args.segments):
        part = {s: c[b[i]:b[i + 1]] for s, c in cand.items()}
        for name, tr in run_segment(part, f"s{i}").items():
            agg[name].extend(tr)
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 78}")
    print(f"RESTART / FVG-MEMORY EFFECT — {args.days}d, {args.segments} segments, {TF}")
    print(f"{'=' * 78}")
    print(f"{'Arm':<22}{'n':<7}{'WR':>7}  {'PnL':>12}  {'exp/trade':>10}")
    print("-" * 78)
    for name in ARMS:
        print(_summarise(agg[name], f"{name}  all"))
    print("-" * 78)
    wr = {}
    for dirn in ("long", "short"):
        for name in ARMS:
            g = [t for t in agg[name] if t.get("direction") == dirn]
            print(_summarise(g, f"{name}  {dirn}"))
            if g:
                wr[(name, dirn)] = sum(1 for t in g if t["pnl_dollars"] > 0) / len(g) * 100
        print("-" * 78)

    print("\nVERDICT:")
    for name in ("RESTART_15", "RESTART_50"):
        dl = wr.get(("CONTINUOUS", "long"), 0) - wr.get((name, "long"), 0)
        ds = wr.get(("CONTINUOUS", "short"), 0) - wr.get((name, "short"), 0)
        verdict = ("SUPPORTS the hypothesis" if ds > max(2 * dl, 5)
                   else "equal damage — restarts matter but do NOT explain the asymmetry"
                   if abs(ds - dl) < 5 else "does not match the hypothesis cleanly")
        print(f"  {name}: LONG loses {dl:+.1f}pp, SHORT loses {ds:+.1f}pp -> {verdict}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
