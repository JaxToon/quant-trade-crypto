"""
Does entering off a FORMING candle specifically damage shorts?

`live_vs_backtest_matched_window.py` found a sign flip that only affects
one direction, on the identical window:

  LONG    live 73.7% WR +$10,879   backtest 61.1% WR +$9,014
  SHORT   live 35.1% WR  -$7,975   backtest 55.8% WR +$2,299

Live BEAT backtest on longs and collapsed on shorts. Uniform degradation
cannot produce that. The GNN hypothesis was tested and rejected (the signal
is gated out in both arms — see gnn_directional_effect.py), leaving one
leading candidate: for most of that window, live entered off candles that
had not closed yet.

`SmartMoneyAnalyzer` is explicit that this is wrong — `update()` warns
"feeding in-progress candles here corrupts OB/FVG/market-structure
detection", and `peek_signals()` exists as the read-only alternative that
checks proximity to ALREADY-detected zones at the current forming price.
Live used peek_signals for entries; the backtest only ever evaluates at
candle close. That is a real behavioural difference, and it was fixed in
production on 2026-09-03 — but the post-fix live sample is far too small to
confirm anything (crypto n=10, 2 shorts).

So simulate it instead, using the SAME mechanism live used rather than an
invented approximation:

  CLOSED  — evaluate entries only at 1h close (backtest behaviour, and
            live's behaviour since the fix)
  FORMING — additionally evaluate at each 5m step inside the hour, using
            peek_signals() against a partial 1h candle assembled from the
            5m bars so far (live's pre-fix behaviour)

Both arms feed IDENTICAL closed 1h candles into SMC history, so pattern
detection is the same. They differ only in when an entry may fire and at
what price.

The hypothesis predicts FORMING degrades SHORT substantially more than
LONG. If instead it hurts both roughly equally, or neither, then
intra-candle entry is NOT the explanation and the SHORT gap needs a
different cause.

Data is 5m from Binance (yfinance caps intraday history and we need sub-hour
resolution over a multi-month window).

Usage: python -m experiments.forming_candle_entry_effect [--days 90] [--segments 3]
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

SUB_TF, SUB_MS, PER_HOUR = "5m", 300_000, 12


def _fetch_5m(symbol: str, start_ms: int, end_ms: int) -> list[Candle]:
    out, cursor = [], start_ms
    while cursor < end_ms:
        try:
            r = requests.get(f"{settings.BINANCE_REST_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": SUB_TF,
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
            cursor = batch[-1][0] + SUB_MS
            time.sleep(0.1)
        except Exception:
            break
    return out


def _to_hours(bars: list[Candle]) -> list[tuple[Candle, list[Candle]]]:
    """Group 5m bars into (closed 1h candle, its 5m constituents)."""
    buckets: dict[datetime, list[Candle]] = {}
    for b in bars:
        h = b.timestamp.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(h, []).append(b)
    out = []
    for h in sorted(buckets):
        subs = sorted(buckets[h], key=lambda c: c.timestamp)
        if len(subs) < PER_HOUR:          # partial hour at either edge
            continue
        out.append((Candle(timestamp=h, open=subs[0].open,
                           high=max(s.high for s in subs), low=min(s.low for s in subs),
                           close=subs[-1].close, volume=sum(s.volume for s in subs)), subs))
    return out


def _summarise(trades, label):
    if not trades:
        return f"{label:<20} (no trades)"
    w = sum(1 for t in trades if t["pnl_dollars"] > 0)
    pnl = sum(t["pnl_dollars"] for t in trades)
    wins = [t["pnl_dollars"] for t in trades if t["pnl_dollars"] > 0]
    loss = [t["pnl_dollars"] for t in trades if t["pnl_dollars"] <= 0]
    pf = (sum(wins) / abs(sum(loss))) if loss and sum(loss) != 0 else float("inf")
    return (f"{label:<20}{len(trades):<7}{w/len(trades)*100:>6.1f}%  "
            f"${pnl:>+11,.0f}  ${pnl/len(trades):>+9,.2f}  {pf:>6.2f}")


def run_segment(series: dict, label: str) -> dict:
    ledgers = {}
    for name in ("CLOSED", "FORMING"):
        led = PaperTradingLedger(path=f"backtrack_logs/_fce_{label}_{name}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000
        ledgers[name] = led

    smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in series}
    fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in series}

    n = min(len(v) for v in series.values())
    for i in range(n):
        for sym, hours in series.items():
            hour_candle, subs = hours[i]

            # ---- FORMING arm: entries fire mid-hour off peek_signals ----
            run_high, run_low = subs[0].high, subs[0].low
            for j, sub in enumerate(subs[:-1]):       # last sub == the close, handled below
                run_high = max(run_high, sub.high)
                run_low = min(run_low, sub.low)
                partial = Candle(timestamp=sub.timestamp, open=subs[0].open,
                                 high=run_high, low=run_low, close=sub.close,
                                 volume=sum(s.volume for s in subs[:j + 1]))
                peek = smc[sym].peek_signals(partial)
                pctx = {**peek, **fvg[sym].get_trading_signal(
                    current_price=partial.close, trend=peek.get("market_structure", "unknown")),
                    "timestamp": partial.timestamp.isoformat()}
                at = partial.timestamp.isoformat()
                led = ledgers["FORMING"]
                led.check_and_close_ohlc(sym, sub.low, sub.high, closed_at=at, timeframe="1h")
                ins = build_entry_insight(sym, partial.close, pctx, gnn_signal=0.0)
                led.maybe_open(sym, "spot", ins["spot"], timeframe="1h", opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe="1h", opened_at=at)

            # ---- both arms: identical closed-candle history + close entry ----
            sig = smc[sym].update(hour_candle)
            hist = list(smc[sym].candles)
            if len(hist) >= 3:
                fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2,
                                    current_price=hour_candle.close, volume=hour_candle.volume)
            fvg[sym].update_all(hour_candle.close)
            ctx = {**sig, **fvg[sym].get_trading_signal(
                current_price=hour_candle.close, trend=sig.get("market_structure", "unknown")),
                "timestamp": hour_candle.timestamp.isoformat()}
            at = hour_candle.timestamp.isoformat()
            ins = build_entry_insight(sym, hour_candle.close, ctx, gnn_signal=0.0)
            for name, led in ledgers.items():
                led.check_and_close_ohlc(sym, hour_candle.low, hour_candle.high,
                                         closed_at=at, timeframe="1h")
                led.maybe_open(sym, "spot", ins["spot"], timeframe="1h", opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe="1h", opened_at=at)

    return {k: list(v.closed_trades) for k, v in ledgers.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--segments", type=int, default=3)
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    print(f"Fetching {args.days}d of 5m from {start:%Y-%m-%d} to {end:%Y-%m-%d}...\n", flush=True)

    series = {}
    for sym in settings.WATCHLIST:
        bars = _fetch_5m(sym, s_ms, e_ms)
        hrs = _to_hours(bars)
        if hrs:
            series[sym] = hrs
            print(f"  {sym:<9}{len(bars):>7} 5m bars -> {len(hrs):>5} hours", flush=True)
    if not series:
        print("no data"); return

    n = min(len(v) for v in series.values())
    b = [round(n * i / args.segments) for i in range(args.segments + 1)]
    out = {"CLOSED": [], "FORMING": []}
    for i in range(args.segments):
        part = {s: v[b[i]:b[i + 1]] for s, v in series.items()}
        for name, tr in run_segment(part, f"s{i}").items():
            out[name].extend(tr)
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 80}")
    print(f"FORMING-CANDLE ENTRY EFFECT — {args.days}d, {args.segments} segments, 1h")
    print(f"{'=' * 80}")
    print(f"{'Arm':<20}{'n':<7}{'WR':>7}  {'PnL':>12}  {'exp/trade':>10}  {'PF':>6}")
    print("-" * 80)
    print(_summarise(out["CLOSED"], "CLOSED  all"))
    print(_summarise(out["FORMING"], "FORMING all"))
    print("-" * 80)
    deltas = {}
    for dirn in ("long", "short"):
        c = [t for t in out["CLOSED"] if t.get("direction") == dirn]
        f = [t for t in out["FORMING"] if t.get("direction") == dirn]
        print(_summarise(c, f"CLOSED  {dirn}"))
        print(_summarise(f, f"FORMING {dirn}"))
        if c and f:
            cw = sum(1 for t in c if t["pnl_dollars"] > 0) / len(c) * 100
            fw = sum(1 for t in f if t["pnl_dollars"] > 0) / len(f) * 100
            deltas[dirn] = cw - fw
            print(f"{'  WR damage':<20}{'':<7}{deltas[dirn]:>6.1f}pp")
        print("-" * 80)

    print("\nVERDICT:")
    if len(deltas) == 2:
        dl, ds = deltas["long"], deltas["short"]
        print(f"  Forming-candle entry costs LONG {dl:.1f}pp and SHORT {ds:.1f}pp of win rate.")
        if ds > dl * 2 and ds > 5:
            print("  SHORT is hit disproportionately -> this IS a strong candidate for the")
            print("  live/backtest gap, and the 2026-09-03 fix should close it. Confirm on")
            print("  live post-fix data before treating it as settled.")
        elif abs(ds - dl) < 5:
            print("  Both directions are hit about equally -> intra-candle entry is NOT the")
            print("  asymmetry's cause. The SHORT gap needs a different explanation.")
        else:
            print("  Read the table — the effect does not match the hypothesis cleanly.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
