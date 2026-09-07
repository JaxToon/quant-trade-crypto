"""
Does re-evaluating entries many times per candle change WHICH setups get
taken — the last standing explanation for the live/backtest SHORT gap?

Established: the gap is a SELECTION problem. Of 37 live shorts only 9 (24%)
correspond to a backtest short; on those 9 the outcomes agree 8/9 and entry
prices differ by a median 0.047%, so execution is fine. Live took 28 shorts
the backtest never proposed (-$4,227, 39% WR) and missed 43 it did (+$4,093,
60% WR).

Five explanations have been tested and rejected: the GNN signal (gated out
98.9% of the time in both arms), forming-candle entry (helps shorts if
anything), a cold analyzer after restart (backfill rebuilds SMC *and* FVG),
testnet data contamination (backfill returns production data, verified to
the cent), and FVG memory truncation across 39 restarts (LONG -1.6pp, SHORT
+0.4pp — not asymmetric).

What remains is structural and was never addressed by the 2026-09-03 fix.
Live calls:

    build_entry_insight(symbol, CURRENT_PRICE, last_CLOSED_candle_context)

and re-runs it every cycle — formerly every 60s, now every second. The SMC
context is correctly taken from a closed candle, but the PRICE keeps moving,
and entry_insight's actionability depends on where price sits relative to
the entry zone. So within one 1h candle live gets dozens of chances to catch
price inside a zone; the backtest gets exactly one, at the close.

That would produce a larger, differently-composed trade population — which
is precisely what the attribution found.

  ONCE_PER_CANDLE — evaluate only at candle close (every backtest in this repo)
  INTRA_CANDLE    — additionally evaluate at each 5m step, using the SAME
                    closed-candle SMC context but that step's live price
                    (what production actually does)

Note the difference from forming_candle_entry_effect.py: that varied the SMC
CONTEXT via peek_signals (the pre-fix bug, now fixed). This varies only the
PRICE, holding context fixed — the behaviour that is still live today.

PRE-REGISTERED: supported only if INTRA_CANDLE degrades SHORT materially
more than LONG (>2x the win-rate drop, and at least 5pp). Equal damage means
intra-candle evaluation changes selection but does not explain the
ASYMMETRY, and we are out of mechanistic hypotheses — in which case say so
rather than inventing a seventh.

Usage: python -m experiments.intracandle_evaluation_effect [--days 90] [--segments 3]
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

SUB_MS, PER_HOUR = 300_000, 12
WARMUP = 150


def _fetch_5m(symbol, start_ms, end_ms):
    out, cursor = [], start_ms
    while cursor < end_ms:
        try:
            r = requests.get(f"{settings.BINANCE_REST_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": "5m",
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


def _to_hours(bars):
    buckets = {}
    for b in bars:
        buckets.setdefault(b.timestamp.replace(minute=0, second=0, microsecond=0), []).append(b)
    out = []
    for h in sorted(buckets):
        subs = sorted(buckets[h], key=lambda c: c.timestamp)
        if len(subs) < PER_HOUR:
            continue
        out.append((Candle(timestamp=h, open=subs[0].open, high=max(s.high for s in subs),
                           low=min(s.low for s in subs), close=subs[-1].close,
                           volume=sum(s.volume for s in subs)), subs))
    return out


def _sm(trades, label):
    if not trades:
        return f"{label:<24} (no trades)"
    w = sum(1 for t in trades if t["pnl_dollars"] > 0)
    pnl = sum(t["pnl_dollars"] for t in trades)
    return f"{label:<24}{len(trades):<7}{w/len(trades)*100:>6.1f}%  ${pnl:>+11,.0f}  ${pnl/len(trades):>+9,.2f}"


def run_segment(series, label):
    out = {}
    for arm in ("ONCE_PER_CANDLE", "INTRA_CANDLE"):
        led = PaperTradingLedger(path=f"backtrack_logs/_ice_{label}_{arm}.json",
                                 starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        led.balance = led.starting_balance
        led.open_positions.clear(); led.closed_trades.clear()
        led.max_closed_trades = 1_000_000

        smc = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in series}
        fvg = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in series}

        n = min(len(v) for v in series.values())
        for i in range(n):
            for sym, hours in series.items():
                hour, subs = hours[i]
                # Closed-candle context, computed once — identical in both arms.
                sig = smc[sym].update(hour)
                hist = list(smc[sym].candles)
                if len(hist) >= 3:
                    fvg[sym].detect_fvg(candles=hist, index=len(hist) - 2,
                                        current_price=hour.close, volume=hour.volume)
                fvg[sym].update_all(hour.close)
                base_trend = sig.get("market_structure", "unknown")

                if arm == "INTRA_CANDLE" and i >= WARMUP:
                    # Same closed-candle context, price walks forward.
                    for sub in subs:
                        ctx = {**sig, **fvg[sym].get_trading_signal(
                            current_price=sub.close, trend=base_trend),
                            "timestamp": sub.timestamp.isoformat()}
                        at = sub.timestamp.isoformat()
                        led.check_and_close_ohlc(sym, sub.low, sub.high, closed_at=at, timeframe="1h")
                        ins = build_entry_insight(sym, sub.close, ctx, gnn_signal=0.0)
                        led.maybe_open(sym, "spot", ins["spot"], timeframe="1h", opened_at=at)
                        led.maybe_open(sym, "derivative", ins["derivative"], timeframe="1h", opened_at=at)
                    continue

                ctx = {**sig, **fvg[sym].get_trading_signal(
                    current_price=hour.close, trend=base_trend),
                    "timestamp": hour.timestamp.isoformat()}
                at = hour.timestamp.isoformat()
                led.check_and_close_ohlc(sym, hour.low, hour.high, closed_at=at, timeframe="1h")
                if i < WARMUP:
                    continue
                ins = build_entry_insight(sym, hour.close, ctx, gnn_signal=0.0)
                led.maybe_open(sym, "spot", ins["spot"], timeframe="1h", opened_at=at)
                led.maybe_open(sym, "derivative", ins["derivative"], timeframe="1h", opened_at=at)
        out[arm] = list(led.closed_trades)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--segments", type=int, default=3)
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"Fetching {args.days}d of 5m...\n", flush=True)
    series = {}
    for s in settings.WATCHLIST:
        hrs = _to_hours(_fetch_5m(s, int(start.timestamp() * 1000), int(end.timestamp() * 1000)))
        if hrs:
            series[s] = hrs
            print(f"  {s:<9}{len(hrs):>5} hours", flush=True)
    if not series:
        print("no data"); return

    n = min(len(v) for v in series.values())
    b = [round(n * i / args.segments) for i in range(args.segments + 1)]
    agg = {"ONCE_PER_CANDLE": [], "INTRA_CANDLE": []}
    for i in range(args.segments):
        part = {s: v[b[i]:b[i + 1]] for s, v in series.items()}
        for k, tr in run_segment(part, f"s{i}").items():
            agg[k].extend(tr)
        print(f"  seg{i} done", flush=True)

    print(f"\n{'=' * 82}")
    print(f"INTRA-CANDLE EVALUATION EFFECT — {args.days}d, {args.segments} segments, 1h")
    print(f"{'=' * 82}")
    print(f"{'Arm':<24}{'n':<7}{'WR':>7}  {'PnL':>12}  {'exp/trade':>10}")
    print("-" * 82)
    for a in agg:
        print(_sm(agg[a], f"{a}  all"))
    print("-" * 82)
    wr = {}
    for dirn in ("long", "short"):
        for a in agg:
            g = [t for t in agg[a] if t.get("direction") == dirn]
            print(_sm(g, f"{a}  {dirn}"))
            if g:
                wr[(a, dirn)] = sum(1 for t in g if t["pnl_dollars"] > 0) / len(g) * 100
        print("-" * 82)

    dl = wr.get(("ONCE_PER_CANDLE", "long"), 0) - wr.get(("INTRA_CANDLE", "long"), 0)
    ds = wr.get(("ONCE_PER_CANDLE", "short"), 0) - wr.get(("INTRA_CANDLE", "short"), 0)
    print(f"\nVERDICT: LONG loses {dl:+.1f}pp, SHORT loses {ds:+.1f}pp")
    if ds > max(2 * dl, 5):
        print("  SUPPORTS the hypothesis — intra-candle re-evaluation damages shorts")
        print("  disproportionately. Fix is cheap: evaluate entries once per closed candle.")
    elif abs(ds - dl) < 5:
        print("  Equal damage. Intra-candle evaluation changes selection but does NOT")
        print("  explain the asymmetry. We are out of mechanistic hypotheses — say so and")
        print("  wait for live data rather than inventing a seventh.")
    else:
        print("  Does not match the hypothesis cleanly — read the table.")
    print(f"{'=' * 82}")


if __name__ == "__main__":
    main()
