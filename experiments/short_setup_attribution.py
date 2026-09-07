"""
Is the live SHORT gap a SELECTION problem or an EXECUTION problem?

Established so far. On the identical window, live shorts won 13/37 (35.1%)
where the backtest predicts 55.8% — z = -2.53, p = 0.011, so not small-sample
luck. Two candidate causes have been eliminated: the GNN (gated out in both
arms, 98.9% of calls return early) and forming-candle entry (simulated, and
if anything it HELPS shorts).

That leaves a fork this script exists to resolve:

  SELECTION — live and backtest took DIFFERENT shorts. Live saw setups the
              backtest never generated, or missed ones it did. The problem
              is upstream, in what gets proposed and accepted.

  EXECUTION — live took the SAME shorts and did worse on them. The problem
              is downstream, in entry price, timing, or how they were closed.

The distinction matters because it halves the search space, and the two have
nothing in common as fixes.

Method: replay the matched window exactly as live_vs_backtest_matched_window
does (same Binance data, same symbols, same three concurrent timeframe
layers), then pair each live short against backtest shorts on
(symbol, timeframe) with open times within MATCH_TOLERANCE_MIN. Report:

  - matched pairs, and whether their OUTCOMES agree
  - live-only shorts (backtest never proposed them)
  - backtest-only shorts (live never took them)
  - for matched pairs, the entry-price difference

A high matched rate with diverging outcomes points to EXECUTION. A low
matched rate points to SELECTION.

Usage: python -m experiments.short_setup_attribution
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import requests

from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

TIMEFRAMES = ["5m", "15m", "1h"]
TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
WARMUP_BARS = 150
MATCH_TOLERANCE_MIN = 60      # a live decision and a backtest bar can be up to an hour apart
LIVE_LEDGER = "backtrack_logs/paper_ledger_crypto.json"


def _fetch(symbol, interval, start_ms, end_ms):
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
                out.append(Candle(timestamp=datetime.utcfromtimestamp(b[0] / 1000),
                                  open=float(b[1]), high=float(b[2]), low=float(b[3]),
                                  close=float(b[4]), volume=float(b[5])))
            cursor = batch[-1][0] + TF_MS[interval]
            time.sleep(0.1)
        except Exception:
            break
    return out


def _parse(s):
    d = datetime.fromisoformat(s)
    return d.replace(tzinfo=None) if d.tzinfo else d


def main():
    live = json.load(open(LIVE_LEDGER))["closed_trades"]
    live = [t for t in live if t.get("opened_at") and t.get("closed_at")]
    lo, hi = min(t["opened_at"] for t in live), max(t["closed_at"] for t in live)
    to_ms = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)
    start_ms, end_ms = to_ms(lo), to_ms(hi)
    window_start = datetime.utcfromtimestamp(start_ms / 1000)
    print(f"Window {window_start} -> {datetime.utcfromtimestamp(end_ms/1000)}\n", flush=True)

    led = PaperTradingLedger(path="backtrack_logs/_ssa.json",
                             starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    led.balance = led.starting_balance
    led.open_positions.clear(); led.closed_trades.clear()
    led.max_closed_trades = 1_000_000

    series = {}
    for tf in TIMEFRAMES:
        for sym in settings.WATCHLIST:
            c = _fetch(sym, tf, start_ms - WARMUP_BARS * TF_MS[tf], end_ms)
            if c:
                series[(sym, tf)] = c
    print(f"fetched {len(series)} (symbol, timeframe) streams", flush=True)

    events = []
    for (sym, tf), cs in series.items():
        for c in cs:
            events.append((c.timestamp, sym, tf, c))
    events.sort(key=lambda e: e[0])

    smc = {k: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for k in series}
    fvg = {k: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for k in series}
    print(f"replaying {len(events):,} events...", flush=True)
    for ts, sym, tf, c in events:
        k = (sym, tf)
        sig = smc[k].update(c)
        hist = list(smc[k].candles)
        if len(hist) >= 3:
            fvg[k].detect_fvg(candles=hist, index=len(hist) - 2, current_price=c.close, volume=c.volume)
        fvg[k].update_all(c.close)
        ctx = {**sig, **fvg[k].get_trading_signal(
            current_price=c.close, trend=sig.get("market_structure", "unknown")),
            "timestamp": ts.isoformat()}
        led.check_and_close_ohlc(sym, c.low, c.high, closed_at=ts.isoformat(), timeframe=tf)
        if ts < window_start:
            continue
        ins = build_entry_insight(sym, c.close, ctx, gnn_signal=0.0)
        led.maybe_open(sym, "spot", ins["spot"], timeframe=tf, opened_at=ts.isoformat())
        led.maybe_open(sym, "derivative", ins["derivative"], timeframe=tf, opened_at=ts.isoformat())

    bt = [t for t in led.closed_trades if t.get("direction") == "short"]
    lv = [t for t in live if t.get("direction") == "short"]
    print(f"\nlive shorts: {len(lv)}   backtest shorts: {len(bt)}")

    tol = timedelta(minutes=MATCH_TOLERANCE_MIN)
    used, pairs, live_only = set(), [], []
    for l in lv:
        lt = _parse(l["opened_at"])
        best, best_gap = None, None
        for i, b in enumerate(bt):
            if i in used or b["symbol"] != l["symbol"] or b.get("timeframe") != l.get("timeframe"):
                continue
            gap = abs(_parse(b["opened_at"]) - lt)
            if gap <= tol and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best is None:
            live_only.append(l)
        else:
            used.add(best)
            pairs.append((l, bt[best], best_gap))
    bt_only = [b for i, b in enumerate(bt) if i not in used]

    print(f"\n{'=' * 84}")
    print("SHORT SETUP ATTRIBUTION — did live and backtest take the SAME shorts?")
    print(f"{'=' * 84}")
    tot = len(lv)
    print(f"  matched (same symbol+timeframe, within {MATCH_TOLERANCE_MIN}min): "
          f"{len(pairs)}/{tot} ({len(pairs)/tot*100:.0f}% of live shorts)")
    print(f"  live-only     (backtest never proposed): {len(live_only)}")
    print(f"  backtest-only (live never took):         {len(bt_only)}")

    if pairs:
        agree = sum(1 for l, b, _ in pairs if (l["pnl_dollars"] > 0) == (b["pnl_dollars"] > 0))
        lw = sum(1 for l, _, _ in pairs if l["pnl_dollars"] > 0)
        bw = sum(1 for _, b, _ in pairs if b["pnl_dollars"] > 0)
        entry_diffs = [abs(l["entry_price"] - b["entry_price"]) / b["entry_price"] * 100
                       for l, b, _ in pairs if b.get("entry_price")]
        print(f"\n  On the {len(pairs)} MATCHED setups:")
        print(f"    live won {lw}/{len(pairs)} ({lw/len(pairs)*100:.0f}%)   "
              f"backtest won {bw}/{len(pairs)} ({bw/len(pairs)*100:.0f}%)")
        print(f"    outcomes agree on {agree}/{len(pairs)} ({agree/len(pairs)*100:.0f}%)")
        if entry_diffs:
            entry_diffs.sort()
            print(f"    entry-price difference: median {entry_diffs[len(entry_diffs)//2]:.3f}%  "
                  f"max {max(entry_diffs):.3f}%")
        print(f"    live PnL ${sum(l['pnl_dollars'] for l,_,_ in pairs):+,.0f}   "
              f"backtest PnL ${sum(b['pnl_dollars'] for _,b,_ in pairs):+,.0f}")

    if live_only:
        print(f"\n  Live-only shorts PnL: ${sum(t['pnl_dollars'] for t in live_only):+,.0f} "
              f"(won {sum(1 for t in live_only if t['pnl_dollars']>0)}/{len(live_only)})")
    if bt_only:
        print(f"  Backtest-only shorts PnL: ${sum(t['pnl_dollars'] for t in bt_only):+,.0f} "
              f"(won {sum(1 for t in bt_only if t['pnl_dollars']>0)}/{len(bt_only)})")

    print("\nVERDICT:")
    rate = len(pairs) / tot if tot else 0
    if rate < 0.4:
        print(f"  Only {rate*100:.0f}% of live shorts correspond to a backtest short.")
        print("  This is a SELECTION problem: the two systems are not trading the same")
        print("  setups, so comparing their win rates was never apples-to-apples. Look")
        print("  upstream — at what each one proposes and accepts, not at execution.")
    elif pairs and abs(lw - bw) / len(pairs) > 0.15:
        print("  Most live shorts DO correspond to a backtest short, but the outcomes")
        print("  diverge. This is an EXECUTION problem: same setups, worse results.")
        print("  Look at entry price, timing, and how positions are closed.")
    else:
        print("  Setups match and outcomes largely agree — the aggregate gap is then")
        print("  driven by the unmatched populations, not by the shared ones. Read the")
        print("  live-only / backtest-only PnL lines above.")
    print(f"{'=' * 84}")


if __name__ == "__main__":
    main()
