"""
Cheap gate before Stage 2 (v3 track) -- tests whether a SIMPLE RULE
combining already-validated signals (full timeframe concurrence, direction)
beats the current baseline (v2.5, unfiltered), before committing to a full
ML meta-label model. If a simple if/else rule doesn't clear this bar, a
trained model on the same 58-day/261-trade dataset almost certainly
wouldn't either -- see the GNN ablation precedent (a genuinely more
sophisticated model added ZERO measured value over SMC-only). Cheaper to
find that out now than after building the model.

Reuses backtest_concurrence_experiment.py's already-validated scaffolding
directly (_fetch_all, _split_into_segments, _classify_concurrence, _score,
TIMEFRAMES) rather than reimplementing it -- this is a variant-comparison
on top of that same causally-correct replay, not a new experiment design.

Filters tested, pre-registered before running (not chosen after seeing
results):
  baseline           - every trade taken exactly as entry_insight/
                        maybe_open already decides today (= current v2.5
                        live behavior, unfiltered)
  concurrence_only    - baseline trades restricted to full_agreement
                        (the strongest bucket from the earlier concurrence
                        experiment: 171 trades, 61.4% win, +$110.55 expectancy)
  direction_only      - baseline trades restricted to long/bullish (the
                        directional main effect from the regime x direction
                        study: bullish consistently outperforms bearish)
  concurrence_and_direction - both conditions at once

This is purely a SEGMENTATION of the same baseline trade set into
overlapping subsets, like the original concurrence experiment -- no new
trades are added, nothing is removed from what the live system actually
does. Read/replay only, does not modify any live code or touch the live
process.

Usage: python backtest_meta_filter_experiment.py [--days N] [--segments N]
"""
from __future__ import annotations

import argparse

from backtest import CRYPTO_TICKERS
from .backtest_concurrence_experiment import (
    TIMEFRAMES,
    _classify_concurrence,
    _fetch_all,
    _score,
    _split_into_segments,
)
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

MAX_DAYS_FOR_5M_15M = 58  # safely under yfinance's 60d cap for 5m/15m


def replay_with_tags(candles_by_symbol_tf: dict) -> list[dict]:
    """Same causal replay as backtest_concurrence_experiment.py's replay(),
    extended to tag each closed trade with BOTH concurrence class and
    direction (that script only tracked concurrence). Returns closed
    trades directly, each with "concurrence_class" and "direction" keys
    added -- no separate id-matching dict needed since we control the
    trade dicts here."""
    events: list[tuple] = []
    for (symbol, tf), candles in candles_by_symbol_tf.items():
        for candle in candles:
            events.append((candle.timestamp, symbol, tf, candle))
    events.sort(key=lambda e: e[0])

    ledger = PaperTradingLedger(path="backtrack_logs/_meta_filter_scratch.json", starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    ledger.balance = ledger.starting_balance
    ledger.max_closed_trades = 1_000_000

    smc_by_key = {k: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for k in candles_by_symbol_tf}
    fvg_by_key = {k: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for k in candles_by_symbol_tf}
    latest_structure: dict[tuple[str, str], str] = {}
    tags_by_trade: dict[str, dict] = {}  # "symbol|market|timeframe|opened_at" -> {concurrence_class, direction}

    for timestamp, symbol, tf, candle in events:
        key = (symbol, tf)
        ledger.check_and_close_ohlc(symbol, candle.low, candle.high, closed_at=timestamp.isoformat(), timeframe=tf)

        smc, fvg = smc_by_key[key], fvg_by_key[key]
        smc_signals = smc.update(candle)
        candle_hist = list(smc.candles)
        if len(candle_hist) >= 3:
            fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
        fvg.update_all(candle.close)
        ms_trend = smc_signals.get("market_structure", "unknown")
        latest_structure[key] = ms_trend
        fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
        smc_context = {**smc_signals, **fvg_signal, "timestamp": timestamp.isoformat()}

        insight = build_entry_insight(symbol, candle.close, smc_context, gnn_signal=0.0)
        opened_at = timestamp.isoformat()
        direction = insight.get("direction", "neutral")

        before_open = set(ledger.open_positions.keys())
        ledger.maybe_open(symbol, "spot", insight["spot"], timeframe=tf, opened_at=opened_at)
        ledger.maybe_open(symbol, "derivative", insight["derivative"], timeframe=tf, opened_at=opened_at)
        newly_opened = set(ledger.open_positions.keys()) - before_open

        if newly_opened and direction in ("long", "short"):
            other_structures = [latest_structure.get((symbol, other_tf), "unknown") for other_tf in TIMEFRAMES if other_tf != tf]
            cls = _classify_concurrence(direction, other_structures)
            for pos_key in newly_opened:
                marker = f"{pos_key[0]}|{pos_key[1]}|{pos_key[2]}|{opened_at}"
                tags_by_trade[marker] = {"concurrence_class": cls, "direction": direction}

    ledger.open_positions.clear()

    tagged_trades = []
    for t in ledger.closed_trades:
        marker = f"{t['symbol']}|{t['market']}|{t['timeframe']}|{t['opened_at']}"
        tags = tags_by_trade.get(marker, {"concurrence_class": None, "direction": None})
        tagged_trades.append({**t, **tags})
    return tagged_trades


def apply_filter(trades: list[dict], name: str) -> list[dict]:
    if name == "baseline":
        return trades
    if name == "concurrence_only":
        return [t for t in trades if t["concurrence_class"] == "full_agreement"]
    if name == "direction_only":
        return [t for t in trades if t["direction"] == "long"]
    if name == "concurrence_and_direction":
        return [t for t in trades if t["concurrence_class"] == "full_agreement" and t["direction"] == "long"]
    raise ValueError(name)


FILTERS = ["baseline", "concurrence_only", "direction_only", "concurrence_and_direction"]


def print_scores(trades: list[dict], label: str) -> None:
    header = f"  {'Filter':<26}{'Trades':<8}{'WinRate%':<10}{'Expectancy$':<13}{'MaxDD%':<9}{'ProfitFactor':<13}"
    print(f"\n{label}")
    print(header)
    for f in FILTERS:
        subset = apply_filter(trades, f)
        s = _score(subset)
        if s["trades"] == 0:
            print(f"  {f:<26}{'0':<8}(no trades)")
            continue
        print(f"  {f:<26}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=MAX_DAYS_FOR_5M_15M)
    parser.add_argument("--segments", type=int, default=3)
    args = parser.parse_args()
    days = min(args.days, MAX_DAYS_FOR_5M_15M)
    n_segments = max(2, args.segments)

    symbols = list(CRYPTO_TICKERS.keys())
    print(f"Meta-filter cheap gate: {days}d, {n_segments} non-overlapping walk-forward segments, symbols: {symbols}")
    print(f"Filters (pre-registered): {FILTERS}\n")

    full = _fetch_all(symbols, days)
    segments = _split_into_segments(full, n_segments)

    all_trades: list[dict] = []
    for seg_idx, seg in enumerate(segments):
        n_candles = sum(len(c) for c in seg.values())
        trades = replay_with_tags(seg)
        all_trades.extend(trades)
        print_scores(trades, f"segment {seg_idx + 1}/{n_segments}: {n_candles} candles, {len(trades)} baseline trades")

    print(f"\n{'=' * 78}\nPOOLED ACROSS ALL {n_segments} SEGMENTS\n{'=' * 78}")
    print_scores(all_trades, f"{len(all_trades)} baseline trades total")

    print(f"\n{'=' * 78}")
    print("Decision gate: if concurrence_and_direction does not clearly and")
    print("consistently beat baseline (win rate AND expectancy, across most")
    print("segments, not just pooled), a full ML meta-label model built on")
    print("the same 261-trade historical dataset is unlikely to do better --")
    print("stop here rather than building it. Purely observational either")
    print("way -- nothing here is deployed, no live code touched.")


if __name__ == "__main__":
    main()
