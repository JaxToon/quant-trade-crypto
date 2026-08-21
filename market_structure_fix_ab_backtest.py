"""
A/B backtest for the market_structure.trend fix (2026-08-21) -- answers a
narrower, more honest question than "is the fix good" (it is; it's a
correctness fix, see strategy/smart_money.py and tests/test_market_structure.py)
namely: since the bug is CONFIRMED to feed live decisions (entry_insight.py,
LLM debate context, FVG confidence bonus, GNN features, Neo4j graph state --
unlike the earlier FVG timestamp bug, which traced to zero behavioral
impact), what does fixing it actually change about Path 1's own trade
selection and P&L, before the live process gets restarted onto it?

Controlled A/B: replays the SAME historical candles through the SAME
entry_insight -> paper_trading pipeline (replay_lockstep's logic, without
GNN -- gnn_signal=0.0 throughout, matching every non-GNN backtest in this
repo), varying ONLY which SmartMoneyAnalyzer subclass computes
market_structure.trend:
  OLD  = SmartMoneyAnalyzerOldBuggy -- the exact pre-fix
         _update_market_structure() body (verbatim from git history,
         commit 58fce8b~1), reproducing the freeze bug faithfully.
  NEW  = SmartMoneyAnalyzer (current, fixed) -- confirmed swing pivots,
         BOS/ChoCH semantics.
FVG detection, order blocks, liquidity zones, entry/stop/target logic,
fees, correlation cap -- all identical between arms; only the market
structure trend fed into build_entry_insight's smc_context differs.

(days, timeframe) runs mirror hmm_regime_filter_backtest.py's choices for
direct comparability with that earlier Path 1 A/B result.

Does not modify strategy/smart_money.py, paper_trading.py, entry_insight.py,
or any live/deployed code. Does not touch the live process.

Usage: python market_structure_fix_ab_backtest.py
"""
from __future__ import annotations

from datetime import datetime

from backtest import CRYPTO_TICKERS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from hmm_regime_filter_backtest import compute_full_stats, print_stats_row
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, MarketStructure, SmartMoneyAnalyzer

RUNS = [(60, "5m"), (60, "15m"), (180, "1h")]  # mirrors hmm_regime_filter_backtest.py's RUNS


class SmartMoneyAnalyzerOldBuggy(SmartMoneyAnalyzer):
    """Reproduces the EXACT pre-fix _update_market_structure() (verbatim
    from `git show 58fce8b~1:strategy/smart_money.py`) -- last_high only
    updates in the bullish-flip branch, last_low only in the bearish-flip
    branch, so whichever one doesn't fire first freezes permanently. Every
    other method (FVG, order blocks, liquidity zones) is inherited
    unchanged from the current, fixed SmartMoneyAnalyzer -- this isolates
    market_structure.trend as the ONLY variable between the two A/B arms."""

    def _update_market_structure(self) -> None:
        if len(self.candles) < 10:
            return

        candles = list(self.candles)
        highs = [c.high for c in candles[-10:]]
        lows = [c.low for c in candles[-10:]]

        recent_high = max(highs)
        recent_low = min(lows)

        if self.market_structure is None:
            self.market_structure = MarketStructure(
                trend="ranging", last_high=recent_high, last_low=recent_low, timestamp=candles[-1].timestamp,
            )
        else:
            ms = self.market_structure
            if recent_high > ms.last_high:
                ms.higher_high = recent_high
                ms.trend = "bullish"
                ms.last_high = recent_high
            elif recent_low < ms.last_low:
                ms.lower_low = recent_low
                ms.trend = "bearish"
                ms.last_low = recent_low
            ms.timestamp = candles[-1].timestamp


def replay_variant(symbols_candles: dict, timeframe: str, analyzer_cls: type) -> PaperTradingLedger:
    ledger = PaperTradingLedger(
        path=f"backtrack_logs/_ms_fix_ab_scratch_{analyzer_cls.__name__}.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    ledger.balance = ledger.starting_balance
    ledger.max_closed_trades = 1_000_000

    smc_by_symbol = {sym: analyzer_cls(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
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

            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=candle.timestamp.isoformat(), timeframe=timeframe)
            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    ledger.open_positions.clear()
    return ledger


def trend_distribution(symbols_candles: dict, analyzer_cls: type) -> dict:
    """Diagnostic: how often each trend label actually gets produced by
    this analyzer variant, across all symbols -- useful to sanity check
    that the OLD arm really does freeze (should show far fewer BULLISH<->
    BEARISH transitions than NEW) before trusting the P&L comparison."""
    counts = {"bullish": 0, "bearish": 0, "ranging": 0, "unknown": 0}
    transitions = 0
    for sym, candles in symbols_candles.items():
        smc = analyzer_cls(lookback=settings.SMC_LOOKBACK_CANDLES)
        last_trend = None
        for candle in candles:
            signals = smc.update(candle)
            trend = signals.get("market_structure", "unknown")
            counts[trend] = counts.get(trend, 0) + 1
            if last_trend is not None and trend != last_trend and trend in ("bullish", "bearish"):
                transitions += 1
            last_trend = trend
    return {"counts": counts, "transitions": transitions}


def main() -> None:
    symbols = list(CRYPTO_TICKERS.keys())
    header = f"{'Variant':<14}{'Trades':<8}{'Win%':<10}{'PnL$':<11}{'Return%':<10}{'Exp$':<12}{'MaxDD%':<9}{'PF':<9}{'AvgWin$':<10}{'AvgLoss$':<10}"

    for days, timeframe in RUNS:
        print(f"\n{'=' * 100}\nA/B: market_structure fix -- {days}d {timeframe}\n{'=' * 100}")
        symbols_candles: dict[str, list[Candle]] = {}
        for symbol in symbols:
            candles = fetch_candles(symbol, days, timeframe)
            if candles:
                symbols_candles[symbol] = candles
                print(f"  {symbol}: {len(candles)} candles")

        if not symbols_candles:
            print("  no candles fetched, skipping")
            continue

        print("\n-- Trend distribution sanity check (does OLD actually freeze more than NEW?) --")
        for label, cls in [("OLD (buggy)", SmartMoneyAnalyzerOldBuggy), ("NEW (fixed)", SmartMoneyAnalyzer)]:
            dist = trend_distribution(symbols_candles, cls)
            print(f"  {label:<14} counts={dist['counts']}  bullish<->bearish transitions={dist['transitions']}")

        print(f"\n{header}")
        print("-" * len(header))
        variant_stats = {}
        for label, cls in [("OLD (buggy)", SmartMoneyAnalyzerOldBuggy), ("NEW (fixed)", SmartMoneyAnalyzer)]:
            ledger = replay_variant(symbols_candles, timeframe, cls)
            stats = compute_full_stats(ledger.closed_trades, ledger.starting_balance)
            print_stats_row(label, stats)
            variant_stats[label] = stats

        old_stats, new_stats = variant_stats["OLD (buggy)"], variant_stats["NEW (fixed)"]
        if old_stats["trades"] and new_stats["trades"]:
            print(f"\n  Delta (NEW - OLD): trades {new_stats['trades'] - old_stats['trades']:+d}, "
                  f"win_rate {new_stats['win_rate_pct'] - old_stats['win_rate_pct']:+.1f}pp, "
                  f"expectancy ${new_stats['expectancy'] - old_stats['expectancy']:+.2f}, "
                  f"max_drawdown {new_stats['max_drawdown_pct'] - old_stats['max_drawdown_pct']:+.2f}pp")

    print(f"\n{'=' * 100}")
    print("Purely observational -- does not modify strategy/smart_money.py or touch the live process.")


if __name__ == "__main__":
    main()
