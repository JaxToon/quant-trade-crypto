"""
New hypothesis from live-trade forensics (2026-08-28): the only 6 crypto
SHORT trades matchable against the candidate log (which logs SMC's own
market_structure reading at decision time) were ALL taken while
market_structure=="bullish" -- i.e. the FVG-derived direction and the
broader swing-structure reading disagreed, and nothing currently checks
that. build_entry_insight's direction comes from smc["direction"] (the
FVG signal) alone; market_structure is a separate field never
cross-checked against it for Path 1.

Tests a market_structure-alignment filter: veto SHORT when
market_structure=="bullish", veto LONG when market_structure=="bearish"
(symmetric). Same 6-segment/~2yr multi-regime bar as every other
robustness pass in this research line, crypto only (where the live
finding originated).

Usage: python -m experiments.market_structure_alignment_filter --segments 6 --timeframe 1h
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer


def apply_alignment_filter(insight: dict, market_structure: str) -> dict:
    direction = insight.get("direction")
    if direction not in ("long", "short"):
        return insight
    disagree = (direction == "short" and market_structure == "bullish") or \
               (direction == "long" and market_structure == "bearish")
    if not disagree:
        return insight
    vetoed = dict(insight)
    for leg in ("spot", "derivative"):
        if vetoed[leg]["actionable"]:
            vetoed[leg] = dict(vetoed[leg])
            vetoed[leg]["actionable"] = False
    return vetoed


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    arm_names = ["BASELINE", "MS_ALIGN"]
    ledgers = {}
    for name in arm_names:
        path = f"backtrack_logs/_ms_align_{label}_{name}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(min_len):
        for sym, candles in symbols_candles.items():
            candle = candles[t]
            smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
            smc_signals = smc.update(candle)
            candle_hist = list(smc.candles)
            if len(candle_hist) >= 3:
                fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()

            ledgers["BASELINE"].check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledgers["BASELINE"].maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledgers["BASELINE"].maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            filtered_insight = apply_alignment_filter(insight, ms_trend)
            ledgers["MS_ALIGN"].check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledgers["MS_ALIGN"].maybe_open(sym, "spot", filtered_insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledgers["MS_ALIGN"].maybe_open(sym, "derivative", filtered_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    results = {}
    for name, ledger in ledgers.items():
        ledger.open_positions.clear()
        results[name] = ledger.stats()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, 730)
    days = min(args.days if args.days is not None else max_days, max_days)

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols: {settings.WATCHLIST}")
    print(f"Fetching {days}d of {timeframe} candles (via yfinance, max available)...\n")

    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        candles = fetch_candles(symbol, days, timeframe)
        if candles:
            all_candles[symbol] = candles
            print(f"  {symbol}: {len(candles)} candles")

    n = min(len(c) for c in all_candles.values())
    n_seg = args.segments
    bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]
    segments = [{sym: candles[bounds[i]:bounds[i + 1]] for sym, candles in all_candles.items()} for i in range(n_seg)]

    wins = 0
    total_base_pnl = 0.0
    total_filt_pnl = 0.0
    print(f"\n{'=' * 90}")
    print(f"[{settings.ASSET_TYPE.upper()}] MARKET-STRUCTURE ALIGNMENT FILTER — {n_seg} segments across {days}d")
    print(f"{'=' * 90}")
    for i, seg in enumerate(segments):
        results = run_segment(seg, timeframe, f"seg{i}")
        bo, fo = results["BASELINE"]["overall"], results["MS_ALIGN"]["overall"]
        total_base_pnl += bo["total_pnl_dollars"]
        total_filt_pnl += fo["total_pnl_dollars"]
        beats = fo["total_pnl_dollars"] > bo["total_pnl_dollars"]
        if beats:
            wins += 1
        bwr = f"{bo['win_rate_pct']:.1f}%" if bo["win_rate_pct"] is not None else "-"
        fwr = f"{fo['win_rate_pct']:.1f}%" if fo["win_rate_pct"] is not None else "-"
        print(f"seg{i}: BASELINE {bo['trades']}t/{bwr}/${bo['total_pnl_dollars']:+.0f}   "
              f"MS_ALIGN {fo['trades']}t/{fwr}/${fo['total_pnl_dollars']:+.0f}   {'WINS' if beats else ''}")

    print(f"\n{'=' * 90}")
    print(f"SUMMARY: MS_ALIGN wins {wins}/{n_seg} segments. Aggregate: baseline ${total_base_pnl:+.2f}, "
          f"MS_ALIGN ${total_filt_pnl:+.2f} (delta ${total_filt_pnl - total_base_pnl:+.2f})")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
