"""
Higher-timeframe (HTF) bias filter experiment — Path 1 (SMC/FVG paper
trading) ONLY.

Tests a well-established ICT/SMC principle this system's own code doesn't
currently implement: "the higher timeframe has authority over the lower
timeframe" -- a 5m or 15m entry should only fire when it agrees with the
1h layer's current market structure, not trade independently of it.
Checked the code before building this: BINANCE_SMC_TIMEFRAMES (5m/15m/1h)
currently run as fully independent, parallel signal generators with zero
cross-timeframe filtering -- confirmed via ingestion/integrated_stream.py
and strategy/entry_insight.py, no bias/htf logic anywhere. This is a
genuine, literature-backed gap, not a re-hash of the momentum experiment:
it's a FILTER on existing SMC/FVG signals, not a new entry mechanism, so
it doesn't inherit that rejection.

Built on top of backtest.py's run_multitimeframe() event-queue structure
(not reinvented): candles from every (symbol, timeframe) pair are merged
into one globally time-sorted stream, each (symbol, timeframe) keeps its
own independent, incrementally-updated SmartMoneyAnalyzer/FVG state. This
already guarantees the causal invariant this project cares about for free
— at the moment a 5m or 15m event is processed, the 1h analyzer's current
state reflects only 1h candles that have actually closed by that
timestamp, never a future one, because it's the same object only ever
updated when its OWN chronological event arrives.

HTF bias source: the 1h layer's SmartMoneyAnalyzer.market_structure.trend
("bullish" / "bearish" / "unknown") for the same symbol, read at whatever
its current state is when a 5m/15m entry is being considered. The 1h
layer's own trades are never filtered (no higher timeframe available in
this 3-layer scope).

Three variants:
  baseline        - unchanged, exactly backtest.py's run_multitimeframe()
                     behavior: all 3 timeframes trade fully independently
  htf_bias_hard   - a 5m/15m entry only fires if its direction matches 1h
                     market_structure exactly (long+bullish, short+bearish)
                     -- blocked if 1h structure is "unknown" too
  htf_bias_soft   - a 5m/15m entry is blocked only if it actively opposes
                     1h market_structure (long while 1h is bearish, or vice
                     versa) -- allowed through if 1h structure is "unknown"
                     (missing information isn't treated as disagreement)

Scored by: win rate, total return, expectancy ($/trade), max drawdown,
profit factor -- and specifically whether either filter variant improves
these over baseline, not just "does it look plausible." Expectation going
in: fewer trades, and the open question is whether the ones that survive
the filter are meaningfully higher quality.

Does not modify backtrack/paper_trading.py, strategy/fvg_analyzer.py,
strategy/smart_money.py, strategy/entry_insight.py, or backtest.py. Uses
its own scratch ledger files, one per variant. Fully isolated from any
live main.py process — pure historical replay, no live data stream, no
shared state, no interference with whatever main.py is currently doing.

Usage: python backtest_htf_bias_experiment.py [--days N]
(always all 3 timeframes at once -- that's the whole point of this one)
"""
from __future__ import annotations

import argparse

from backtest import CRYPTO_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

VARIANTS = ["baseline", "htf_bias_hard", "htf_bias_soft"]
HTF = "1h"  # the layer whose structure governs the others in this experiment


def _htf_allows(direction: str, htf_structure: str, variant: str) -> bool:
    if direction not in ("long", "short"):
        return True  # neutral/no-direction insight -- nothing for this filter to block
    if variant == "htf_bias_hard":
        return (direction == "long" and htf_structure == "bullish") or (direction == "short" and htf_structure == "bearish")
    if variant == "htf_bias_soft":
        opposes = (direction == "long" and htf_structure == "bearish") or (direction == "short" and htf_structure == "bullish")
        return not opposes
    return True  # baseline: no filter


def replay_variant(symbols: list[str], days: int, variant: str) -> tuple[PaperTradingLedger, int]:
    timeframes = settings.BINANCE_SMC_TIMEFRAMES

    candles_by_symbol_tf: dict[tuple[str, str], list] = {}
    for symbol in symbols:
        for tf in timeframes:
            candles = fetch_candles(symbol, days, tf)
            if candles:
                candles_by_symbol_tf[(symbol, tf)] = candles

    events: list[tuple] = []
    for (symbol, tf), candles in candles_by_symbol_tf.items():
        for candle in candles:
            events.append((candle.timestamp, symbol, tf, candle))
    events.sort(key=lambda e: e[0])

    ledger = PaperTradingLedger(
        path=f"backtrack_logs/_htf_bias_scratch_{variant}.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    ledger.balance = ledger.starting_balance
    ledger.max_closed_trades = 1_000_000

    smc_by_key = {k: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for k in candles_by_symbol_tf}
    fvg_by_key = {k: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for k in candles_by_symbol_tf}
    # Latest known market_structure per (symbol, tf) -- causally correct by
    # construction: only updated when that (symbol, tf)'s own event fires.
    latest_structure: dict[tuple[str, str], str] = {}

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

        if tf == HTF or variant == "baseline":
            allowed = True
        else:
            htf_structure = latest_structure.get((symbol, HTF), "unknown")
            allowed = _htf_allows(insight.get("direction", "neutral"), htf_structure, variant)

        if allowed:
            ledger.maybe_open(symbol, "spot", insight["spot"], timeframe=tf, opened_at=opened_at)
            ledger.maybe_open(symbol, "derivative", insight["derivative"], timeframe=tf, opened_at=opened_at)

    ledger.open_positions.clear()  # unresolved at window end -- not counted
    return ledger, len(events)


def compute_stats(ledger: PaperTradingLedger) -> dict:
    trades = ledger.closed_trades
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    equity = ledger.starting_balance
    peak = equity
    max_dd_pct = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100)

    return {
        "trades": n,
        "win_rate_pct": len(wins) / n * 100,
        "total_return_pct": total_pnl / ledger.starting_balance * 100,
        "expectancy_dollars": total_pnl / n,
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HTF bias filter experiment: does gating 5m/15m entries on 1h market structure improve Path 1? (Path 1 only)")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()
    days = min(args.days if args.days is not None else settings.BACKTEST_DAYS, YF_MAX_DAYS["5m"])

    tickers = CRYPTO_TICKERS if settings.ASSET_TYPE == "crypto" else {s: [s] for s in settings.STOCK_WATCHLIST}
    symbols = list(tickers.keys())
    print(f"ASSET_TYPE={settings.ASSET_TYPE} — HTF bias experiment over {days}d, symbols: {symbols}")
    print(f"Timeframes: {settings.BINANCE_SMC_TIMEFRAMES} (HTF={HTF} governs 5m/15m entries in the filtered variants)\n")

    results = {}
    for variant in VARIANTS:
        ledger, n_events = replay_variant(symbols, days, variant)
        results[variant] = compute_stats(ledger)
        results[variant]["events"] = n_events

    header = f"{'Variant':<18}{'Trades':<8}{'WinRate%':<10}{'Return%':<10}{'MaxDD%':<9}{'ProfitFactor':<13}"
    print(header)
    print("-" * len(header))
    for name, s in results.items():
        if s["trades"] == 0:
            print(f"{name:<18}{'0':<8}(no trades)")
            continue
        print(
            f"{name:<18}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['total_return_pct']:<+10.2f}"
            f"{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}"
        )

    ranked = sorted(
        ((name, s) for name, s in results.items() if s["trades"] > 0),
        key=lambda kv: kv[1]["expectancy_dollars"], reverse=True,
    )
    print(f"\nRanked by expectancy ($/trade): {' > '.join(name for name, _ in ranked)}")
    baseline = results.get("baseline", {})
    if baseline.get("trades"):
        for name in ("htf_bias_hard", "htf_bias_soft"):
            s = results.get(name, {})
            if s.get("trades"):
                print(
                    f"{name}: {baseline['trades']} -> {s['trades']} trades ({s['trades']/baseline['trades']*100:.0f}% of baseline), "
                    f"win rate {baseline['win_rate_pct']:.1f}% -> {s['win_rate_pct']:.1f}%, "
                    f"expectancy ${baseline['expectancy_dollars']:+.2f} -> ${s['expectancy_dollars']:+.2f}, "
                    f"drawdown {baseline['max_drawdown_pct']:.2f}% -> {s['max_drawdown_pct']:.2f}%"
                )
            else:
                print(f"{name}: no trades survived the filter at this window")

    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("paper_trading.py, entry_insight.py, or any live strategy code regardless of which variant wins.")


if __name__ == "__main__":
    main()
