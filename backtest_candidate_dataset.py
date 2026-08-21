"""
Historical bootstrap for the candidate-trade dataset (v3 track, Stage 1
cont'd) -- builds thousands of candidate rows from real historical replay
in minutes, instead of waiting weeks for live paper trading to accumulate
the same volume one trade at a time (today's live run produced a handful).

Reuses the exact joint multi-timeframe replay pattern already validated in
market_structure_fix_ab_multitimeframe.py (merged event queue across
every (symbol, timeframe), ONE shared ledger so the correlation cap sees
true cross-timeframe concurrency, matching live) -- not a new replay
engine, just the existing one with candidate logging added inside it,
using the IDENTICAL schema main.py's _record_candidate() writes live, so
this output and dashboard/candidate_trades_{asset_type}.json can be
concatenated into one training set without reconciling two formats.

Live candidate logging (deployed 2026-08-22) is NOT replaced by this --
it remains the genuinely out-of-sample stream for validating anything
trained on this historical bootstrap, since a backtest can by definition
only replay what's already happened. yfinance is also a different exact
data source than Binance's live feed (confirmed during the 2026-08-21
losing-streak investigation, where a yfinance replay didn't reproduce the
actual live trades) -- this dataset is for feature/label volume, not a
perfect stand-in for live execution fidelity. Output is saved to
*_historical.json files, kept separate from the live-collected ones, so a
future training step can weight or filter by source explicitly rather than
silently blending them.

Usage: python backtest_candidate_dataset.py [--days N]
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime

from backtest import CRYPTO_TICKERS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from build_candidate_dataset import build_dataset, summarize
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

TIMEFRAMES = settings.BINANCE_SMC_TIMEFRAMES  # ['5m', '15m', '1h'], matches live exactly


def _candidate_row(symbol: str, tf: str, market: str, rec: dict, smc_raw: dict,
                    decision: str, reason: str | None, opened_at: str | None, timestamp: datetime) -> dict:
    """Identical schema to main.py's _record_candidate() -- see that
    function's docstring for field meanings. Kept in sync deliberately;
    if that schema changes, this must change with it."""
    return {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "timeframe": tf,
        "market": market,
        "direction": rec.get("direction"),
        "entry_zone": rec.get("entry_zone"),
        "stop_loss": rec.get("stop_loss"),
        "target": rec.get("target"),
        "confidence": rec.get("confidence", 0.0),
        "atr": rec.get("atr", 0.0),
        "decision": decision,
        "reason": reason,
        "opened_at": opened_at,
        "features": {
            "market_structure": smc_raw.get("market_structure"),
            "smc_signal": smc_raw.get("smc_signal"),
            "active_order_blocks": smc_raw.get("active_order_blocks"),
            "active_fvgs": smc_raw.get("active_fvgs"),
            "liquidity_zones": smc_raw.get("liquidity_zones"),
            "in_bullish_ob": smc_raw.get("in_bullish_ob"),
            "in_bearish_ob": smc_raw.get("in_bearish_ob"),
            "in_bullish_fvg": smc_raw.get("in_bullish_fvg"),
            "in_bearish_fvg": smc_raw.get("in_bearish_fvg"),
        },
    }


def replay_and_log(candles_by_symbol_tf: dict) -> tuple[PaperTradingLedger, list[dict], list[dict]]:
    """Joint multi-timeframe replay (verbatim pattern from
    market_structure_fix_ab_multitimeframe.py), with candidate logging
    added at the same point live logs it -- right after maybe_open()
    decides, never influencing that decision."""
    events: list[tuple] = []
    for (symbol, tf), candles in candles_by_symbol_tf.items():
        for candle in candles:
            events.append((candle.timestamp, symbol, tf, candle))
    events.sort(key=lambda e: e[0])

    ledger = PaperTradingLedger(
        path="backtrack_logs/_candidate_dataset_bootstrap_scratch.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    ledger.balance = ledger.starting_balance
    ledger.open_positions.clear()
    ledger.closed_trades.clear()
    ledger.max_closed_trades = 1_000_000

    smc_by_key = {k: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for k in candles_by_symbol_tf}
    fvg_by_key = {k: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for k in candles_by_symbol_tf}
    candidate_log: list[dict] = []

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
        fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
        smc_context = {**smc_signals, **fvg_signal, "timestamp": timestamp.isoformat()}

        insight = build_entry_insight(symbol, candle.close, smc_context, gnn_signal=0.0)
        opened_at = timestamp.isoformat()
        spot = {**insight.get("spot", {}), "confidence": insight.get("confidence", 0.0)}
        derivative = {**insight.get("derivative", {}), "confidence": insight.get("confidence", 0.0)}

        # See main.py's _record_candidate call site for why this check is
        # needed: maybe_open() returns None both when it opens a NEW
        # position and when one is already open at this key (a no-op) --
        # only the former is a genuine new candidate.
        spot_already_open = (symbol, "spot", tf) in ledger.open_positions
        deriv_already_open = (symbol, "derivative", tf) in ledger.open_positions
        spot_reason = ledger.maybe_open(symbol, "spot", spot, timeframe=tf, opened_at=opened_at)
        deriv_reason = ledger.maybe_open(symbol, "derivative", derivative, timeframe=tf, opened_at=opened_at)

        if spot.get("actionable") and not spot_already_open:
            candidate_log.append(_candidate_row(symbol, tf, "spot", spot, smc_context,
                                                 "rejected" if spot_reason else "accepted", spot_reason,
                                                 None if spot_reason else opened_at, timestamp))
        if derivative.get("actionable") and not deriv_already_open:
            candidate_log.append(_candidate_row(symbol, tf, "derivative", derivative, smc_context,
                                                 "rejected" if deriv_reason else "accepted", deriv_reason,
                                                 None if deriv_reason else opened_at, timestamp))

    # open_positions is dict[(symbol, market, timeframe), PaperPosition] --
    # need the dataclass VALUES as dicts (build_candidate_dataset.py's
    # join logic expects .get()-able dicts, same shape as closed_trades).
    open_positions = [asdict(p) for p in ledger.open_positions.values()]
    ledger.open_positions.clear()
    return ledger, candidate_log, open_positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    symbols = list(CRYPTO_TICKERS.keys())
    print(f"Fetching {args.days}d x {TIMEFRAMES} for {len(symbols)} symbols...\n")

    candles_by_symbol_tf: dict[tuple[str, str], list] = {}
    for symbol in symbols:
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, args.days, tf)
            if candles:
                candles_by_symbol_tf[(symbol, tf)] = candles
                print(f"  {symbol} [{tf}]: {len(candles)} candles")

    if not candles_by_symbol_tf:
        print("No candles fetched, aborting.")
        return

    print("\nReplaying (joint multi-timeframe, causal, matches live pipeline)...")
    ledger, candidate_log, open_positions = replay_and_log(candles_by_symbol_tf)
    print(f"  {len(candidate_log)} actionable candidates logged, {len(ledger.closed_trades)} resolved trades")

    candidate_path = "dashboard/candidate_trades_crypto_historical.json"
    os.makedirs("dashboard", exist_ok=True)
    with open(candidate_path, "w") as f:
        json.dump(candidate_log, f)
    print(f"  raw candidate log -> {candidate_path}")

    rows = build_dataset(candidate_log, ledger.closed_trades, open_positions)
    print()
    summarize(rows)

    dataset_path = "dashboard/candidate_dataset_crypto_historical.json"
    with open(dataset_path, "w") as f:
        json.dump(rows, f)
    print(f"Saved {len(rows)} labeled rows -> {dataset_path}")


if __name__ == "__main__":
    main()
