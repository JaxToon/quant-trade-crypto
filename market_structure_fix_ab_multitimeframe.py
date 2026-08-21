"""
A/B backtest for the market_structure fix, take 2 (2026-08-22) -- fixes a
real methodology flaw in market_structure_fix_ab_backtest.py: that script
replayed each timeframe (5m, 15m, 1h) through its OWN separate ledger, but
live (main.py) and backtest.py's own run_multitimeframe() process all three
timeframes for a symbol through ONE SHARED ledger with a joint,
chronologically-merged event queue -- the portfolio correlation cap sees
concurrent positions ACROSS timeframes, not per-timeframe in isolation.
Running timeframes separately breaks that interaction and barely
reproduced any of the actual overnight live trades when checked directly.

This reuses run_multitimeframe()'s exact merged-event-queue pattern (global
queue of (timestamp, symbol, timeframe, candle) events, sorted real-time,
ONE PaperTradingLedger instance for the whole run), parameterized only by
which SmartMoneyAnalyzer subclass computes market_structure -- same
isolation principle as the first A/B script: identical everything else
(FVG, order blocks, entry/stop/target, fees, correlation cap), OLD-buggy vs
NEW-fixed is the only variable.

Fully isolated from the live process: separate scratch ledger paths
(backtrack_logs/_ms_fix_ab_mtf_scratch_*.json), no live ledger file
touched, no live process signaled or read from except its own log (not
even that, here). Safe to run alongside the live system.

Also reports a TAIL breakdown (last N days of the window) specifically,
since the live concern that prompted this was a recent (~7h) losing
streak -- the full-window aggregate could look fine while masking a real
recent divergence, or vice versa.

Usage: python market_structure_fix_ab_multitimeframe.py [--days N]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta

from backtest import CRYPTO_TICKERS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from hmm_regime_filter_backtest import compute_full_stats, print_stats_row
from market_structure_fix_ab_backtest import SmartMoneyAnalyzerOldBuggy
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

TIMEFRAMES = settings.BINANCE_SMC_TIMEFRAMES  # ['5m', '15m', '1h'], matches live exactly


def replay_joint_multitimeframe(candles_by_symbol_tf: dict, analyzer_cls: type) -> PaperTradingLedger:
    """Verbatim adaptation of backtest.py's run_multitimeframe() replay
    core -- ONE merged, chronologically-sorted event queue across every
    (symbol, timeframe), ONE shared ledger (so the correlation cap sees
    true cross-timeframe concurrency, matching live) -- parameterized by
    analyzer class instead of hardcoding SmartMoneyAnalyzer."""
    events: list[tuple] = []
    for (symbol, tf), candles in candles_by_symbol_tf.items():
        for candle in candles:
            events.append((candle.timestamp, symbol, tf, candle))
    events.sort(key=lambda e: e[0])

    ledger = PaperTradingLedger(
        path=f"backtrack_logs/_ms_fix_ab_mtf_scratch_{analyzer_cls.__name__}.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    ledger.balance = ledger.starting_balance
    ledger.open_positions.clear()
    ledger.closed_trades.clear()
    ledger.max_closed_trades = 1_000_000

    smc_by_key = {k: analyzer_cls(lookback=settings.SMC_LOOKBACK_CANDLES) for k in candles_by_symbol_tf}
    fvg_by_key = {k: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for k in candles_by_symbol_tf}

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
        ledger.maybe_open(symbol, "spot", insight["spot"], timeframe=tf, opened_at=opened_at)
        ledger.maybe_open(symbol, "derivative", insight["derivative"], timeframe=tf, opened_at=opened_at)

    ledger.open_positions.clear()
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--tail-days", type=int, default=3, help="recent-window breakdown size")
    args = parser.parse_args()

    symbols = list(CRYPTO_TICKERS.keys())
    print(f"Fetching {args.days}d x {TIMEFRAMES} for {len(symbols)} symbols (joint multi-timeframe replay)...\n")

    candles_by_symbol_tf: dict[tuple[str, str], list] = {}
    for symbol in symbols:
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, args.days, tf)
            if candles:
                candles_by_symbol_tf[(symbol, tf)] = candles
                print(f"  {symbol} [{tf}]: {len(candles)} candles, {candles[0].timestamp} -> {candles[-1].timestamp}")

    if not candles_by_symbol_tf:
        print("No candles fetched, aborting.")
        return

    window_end = max(c[-1].timestamp for c in candles_by_symbol_tf.values())
    tail_start = window_end - timedelta(days=args.tail_days)
    print(f"\nFull window ends {window_end}. Tail breakdown: last {args.tail_days}d (>= {tail_start}).\n")

    header = f"{'Variant':<20}{'Trades':<8}{'Win%':<10}{'PnL$':<11}{'Return%':<10}{'Exp$':<12}{'MaxDD%':<9}{'PF':<9}{'AvgWin$':<10}{'AvgLoss$':<10}"

    print(f"\n{'=' * 100}\nFULL WINDOW ({args.days}d, joint multi-timeframe, correlation cap shared across tf)\n{'=' * 100}")
    print(header)
    print("-" * len(header))
    ledgers = {}
    for label, cls in [("OLD (buggy)", SmartMoneyAnalyzerOldBuggy), ("NEW (fixed, live)", SmartMoneyAnalyzer)]:
        ledger = replay_joint_multitimeframe(candles_by_symbol_tf, cls)
        ledgers[label] = ledger
        stats = compute_full_stats(ledger.closed_trades, ledger.starting_balance)
        print_stats_row(label, stats)

    print(f"\n{'=' * 100}\nTAIL WINDOW (last {args.tail_days}d only -- where tonight's live losing streak happened)\n{'=' * 100}")
    print(header)
    print("-" * len(header))
    for label, ledger in ledgers.items():
        tail_trades = [t for t in ledger.closed_trades if datetime.fromisoformat(t["opened_at"]) >= tail_start]
        stats = compute_full_stats(tail_trades, ledger.starting_balance)
        print_stats_row(label, stats)
        for t in tail_trades:
            print(f"      {t['opened_at']} {t['symbol']} [{t['timeframe']}] {t['direction']} {t['outcome']} pnl={t['pnl_dollars']:+.2f}")

    print(f"\n{'=' * 100}")
    print("Isolated from the live process -- separate scratch ledgers, live paper_ledger_crypto.json untouched.")


if __name__ == "__main__":
    main()
