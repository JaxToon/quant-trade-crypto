"""
Is one of the three concurrent SMC/FVG timeframe layers (5m / 15m / 1h) a
net drag on the system?

Live data hinted at it -- 1h: 46 trades, 63.0% WR, +$4,400, PF 1.47; 5m: 15
trades, 60.0% WR, +$476, PF 1.15; 15m: 25 trades, 56.0% WR, **-$454, PF
0.93** -- but n=15-46 per layer is far too small to act on (the standing
rule: don't react to small samples). This runs the same question against a
proper multi-segment backtest instead.

`main.py` runs all three layers concurrently (settings.BINANCE_SMC_TIMEFRAMES),
each with independent SMC/FVG state and its own positions, so a
consistently-losing layer is pure drag that could simply be switched off --
a cleaner, simpler improvement than any entry filter this project has
tested and rejected.

IMPORTANT constraint on the comparison: yfinance serves 5m and 15m for only
60 days (YF_MAX_DAYS), vs 730 for 1h. To keep this apples-to-apples all
three layers are run over the SAME 60-day calendar window -- so this is a
shorter, fewer-regime test than the usual 730d/6-segment bar. Segments are
still used to check consistency within that window, but a result here is
weaker evidence than a 2-year pass: treat it as a screen, not a verdict.
Anything that looks decisive should be re-checked on 1h/730d for the layers
that support it before acting.

Usage: python -m experiments.per_timeframe_layer_analysis [--days 60] [--segments 4]
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

TIMEFRAMES = ["5m", "15m", "1h"]


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict:
    ledger = PaperTradingLedger(
        path=f"backtrack_logs/_tflayer_{label}_{timeframe}.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    ledger.balance = ledger.starting_balance
    ledger.open_positions.clear()
    ledger.closed_trades.clear()
    ledger.max_closed_trades = 1_000_000

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(min_len):
        for sym, candles in symbols_candles.items():
            candle = candles[t]
            smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
            smc_signals = smc.update(candle)
            hist = list(smc.candles)
            if len(hist) >= 3:
                fvg.detect_fvg(candles=hist, index=len(hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    ledger.open_positions.clear()
    return ledger.stats()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="capped at 60 -- the 5m/15m yfinance limit")
    parser.add_argument("--segments", type=int, default=4)
    args = parser.parse_args()
    days = min(args.days, min(YF_MAX_DAYS[tf] for tf in TIMEFRAMES))

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols")
    print(f"Window: {days}d (capped by the 5m/15m limit so all three layers share one calendar window)\n", flush=True)

    per_tf_candles: dict[str, dict[str, list[Candle]]] = {}
    for tf in TIMEFRAMES:
        got = {}
        for symbol in settings.WATCHLIST:
            candles = fetch_candles(symbol, days, tf)
            if candles:
                got[symbol] = candles
        per_tf_candles[tf] = got
        counts = {s: len(c) for s, c in got.items()}
        print(f"  {tf:4s}: {len(got)} symbols, ~{min(counts.values()) if counts else 0} candles each", flush=True)

    n_seg = args.segments
    print(f"\n{'=' * 96}")
    print(f"[{settings.ASSET_TYPE.upper()}] PER-TIMEFRAME LAYER ANALYSIS — {n_seg} segments across {days}d")
    print(f"{'=' * 96}", flush=True)

    agg: dict[str, dict] = {tf: {"pnl": 0.0, "trades": 0, "wins": 0, "seg_pnls": []} for tf in TIMEFRAMES}

    for tf in TIMEFRAMES:
        candles_map = per_tf_candles[tf]
        if not candles_map:
            print(f"\n{tf}: no data, skipping")
            continue
        n = min(len(c) for c in candles_map.values())
        bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]
        print(f"\n--- {tf} ---")
        for i in range(n_seg):
            seg = {s: c[bounds[i]:bounds[i + 1]] for s, c in candles_map.items()}
            o = run_segment(seg, tf, f"seg{i}")["overall"]
            wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
            print(f"  seg{i}: {o['trades']:4d}t  WR={wr:<8} PnL=${o['total_pnl_dollars']:>+12,.2f}", flush=True)
            agg[tf]["pnl"] += o["total_pnl_dollars"]
            agg[tf]["trades"] += o["trades"]
            agg[tf]["wins"] += o["wins"]
            agg[tf]["seg_pnls"].append(o["total_pnl_dollars"])

    print(f"\n{'=' * 96}")
    print(f"{'Layer':<8}{'Trades':<10}{'Win Rate':<12}{'Total PnL':<20}{'Profitable segments':<22}")
    print("-" * 96)
    for tf in TIMEFRAMES:
        a = agg[tf]
        if not a["trades"]:
            continue
        wr = a["wins"] / a["trades"] * 100
        pos_segs = sum(1 for p in a["seg_pnls"] if p > 0)
        print(f"{tf:<8}{a['trades']:<10}{wr:<12.1f}${a['pnl']:>+15,.2f}     {pos_segs}/{len(a['seg_pnls'])}")
    print("-" * 96)
    print("A layer worth switching off would be consistently negative -- losing in most/all segments,")
    print("not just aggregate. One bad segment in an otherwise positive layer is noise, not a verdict.")
    print("Remember this is a 60-day window (5m/15m data limit), so it screens rather than concludes.")
    print(f"{'=' * 96}")


if __name__ == "__main__":
    main()
