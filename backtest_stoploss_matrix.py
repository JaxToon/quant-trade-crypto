"""
Stop-loss placement backtest matrix — Path 1 (SMC/FVG paper trading) ONLY.

Quantitative experiment, not a strategy change: entry signal generation
(FVG detection, direction, confidence, target) is completely untouched.
This only overrides the `stop_loss` price before it reaches
build_entry_insight, comparing variants against the exact same historical
candles/entries/targets. Mirrors backtest_target_matrix.py's structure and
safety guarantees exactly, just on the other side of the trade (stop
placement instead of target placement).

Motivation: the live stop is always gap_boundary -/+ 0.2*gap_size
(strategy/fvg_analyzer.py's FVGZone.calculate_stop_loss) — a fixed,
FVG-anchored buffer that ignores the instrument's actual volatility. A
tight stop on a volatile symbol/timeframe gets stopped out by ordinary
noise; a loose stop on a calm one gives back more than necessary. This
matrix asks: does a wider FVG-anchored buffer, or switching the stop's
basis entirely to ATR (volatility-relative), improve EXPECTANCY and
drawdown, or just reduce stop-outs while giving back more per loss than
it saves?

Note: MIN_STOP_DISTANCE_PCT (paper_trading.py) already rejects stops too
tight to survive fees regardless of variant, so a wider-stop variant isn't
just "more trades survive the fee filter" by construction alone — it also
changes position size (quantity = risk_amount / stop_distance), so a wider
stop means a smaller position for the same $ risk, same as it would live.

Variants:
  baseline    - unchanged, the live gap_boundary -/+ 0.2*gap_size stop
  gap_0.5x / gap_1.0x - same FVG-anchored method, wider buffer multiple
  ATR_1.0x / ATR_1.5x / ATR_2.0x - stop_distance = multiple * ATR from
                                     entry, ignoring the FVG's own buffer

Scored by: win rate, total return, expectancy ($/trade), max drawdown (%
from peak equity), and profit factor — a wider stop is expected to raise
hit-rate at the cost of a bigger average loss; the question is net effect.

Does not modify backtrack/paper_trading.py, strategy/fvg_analyzer.py, or
strategy/entry_insight.py — everything here is applied AFTER those produce
their normal output, on a private copy of the signal dict passed into
build_entry_insight. Uses its own scratch ledger file (never touches the
live paper-trading ledger main.py maintains, and runs fully isolated from
any currently-running main.py process -- pure historical replay, no live
data stream, no shared state).

Usage: python backtest_stoploss_matrix.py [--days N] [--timeframe 5m|15m|1h]
"""
from __future__ import annotations

import argparse

from backtest import CRYPTO_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer


def _gap_multiple_stop(direction: str, entry_zone: tuple[float, float], multiple: float) -> float:
    low, high = entry_zone
    gap_size = high - low
    return (low - multiple * gap_size) if direction == "long" else (high + multiple * gap_size)


def _atr_multiple_stop(direction: str, entry_zone: tuple[float, float], atr: float, multiple: float) -> float:
    entry_price = sum(entry_zone) / 2
    stop_distance = multiple * atr
    return (entry_price - stop_distance) if direction == "long" else (entry_price + stop_distance)


def _make_variant(kind: str, multiple: float | None = None):
    """Returns a fn(smc_context: dict) -> float | None overriding stop_loss, or None for baseline (no override)."""
    if kind == "baseline":
        return None

    def override(smc_context: dict) -> float | None:
        direction = smc_context.get("direction")
        entry_zone = smc_context.get("entry_zone")
        if direction not in ("long", "short") or not entry_zone:
            return None
        if kind == "gap":
            return _gap_multiple_stop(direction, entry_zone, multiple)
        if kind == "atr":
            atr = smc_context.get("atr", 0.0)
            if not atr or atr <= 0:
                return None  # no volatility reading yet -- fall back to live stop rather than a zero/negative distance
            return _atr_multiple_stop(direction, entry_zone, atr, multiple)
        raise ValueError(kind)

    return override


VARIANTS: dict[str, tuple[str, float | None]] = {
    "baseline": ("baseline", None),
    "gap_0.5x": ("gap", 0.5),
    "gap_1.0x": ("gap", 1.0),
    "ATR_1.0x": ("atr", 1.0),
    "ATR_1.5x": ("atr", 1.5),
    "ATR_2.0x": ("atr", 2.0),
}


def replay_variant(symbols_candles: dict, timeframe: str, stop_override) -> PaperTradingLedger:
    """
    Same lockstep structure as backtest_target_matrix.py's replay_variant:
    no GNN (gnn_signal=0.0 throughout) and no persistence -- comparison
    tool only, ledgers stay in-memory.
    """
    ledger = PaperTradingLedger(
        path="backtrack_logs/_stoploss_matrix_scratch.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    ledger.balance = ledger.starting_balance
    ledger.max_closed_trades = 1_000_000

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

            if stop_override is not None:
                new_stop = stop_override(smc_context)
                if new_stop is not None:
                    smc_context["stop_loss"] = new_stop

            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=candle.timestamp.isoformat(), timeframe=timeframe)
            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    ledger.open_positions.clear()  # unresolved at window end -- not counted
    return ledger


def compute_stats(ledger: PaperTradingLedger) -> dict:
    trades = ledger.closed_trades
    n = len(trades)
    if n == 0:
        return {"trades": 0}

    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    win_rate = len(wins) / n * 100
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    expectancy = total_pnl / n
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

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
        "win_rate_pct": win_rate,
        "total_return_pct": total_pnl / ledger.starting_balance * 100,
        "expectancy_dollars": expectancy,
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare stop-loss placement variants on identical historical entries (Path 1 only).")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", choices=list(YF_MAX_DAYS.keys()), default="15m")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS[timeframe]
    days = min(args.days if args.days is not None else settings.BACKTEST_DAYS, max_days)

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — stop-loss matrix over {days}d of {timeframe} candles\n")
    all_candles: dict[str, list] = {}
    tickers = CRYPTO_TICKERS if settings.ASSET_TYPE == "crypto" else {s: [s] for s in settings.STOCK_WATCHLIST}
    for symbol, yf_tickers in tickers.items():
        candles = fetch_candles(symbol, days, timeframe)
        if candles:
            all_candles[symbol] = candles
            print(f"  {symbol}: {len(candles)} candles")
    print()

    results = {}
    for name, (kind, multiple) in VARIANTS.items():
        override = _make_variant(kind, multiple)
        ledger = replay_variant(all_candles, timeframe, override)
        results[name] = compute_stats(ledger)

    header = f"{'Variant':<10}{'Trades':<8}{'WinRate%':<10}{'Return%':<10}{'Expectancy$':<13}{'MaxDD%':<9}{'ProfitFactor':<13}"
    print(header)
    print("-" * len(header))
    for name, s in results.items():
        if s["trades"] == 0:
            print(f"{name:<10}{'0':<8}(no trades)")
            continue
        print(
            f"{name:<10}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['total_return_pct']:<+10.2f}"
            f"{s['expectancy_dollars']:<+13.2f}{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}"
        )

    ranked = sorted(
        ((name, s) for name, s in results.items() if s["trades"] > 0),
        key=lambda kv: kv[1]["expectancy_dollars"], reverse=True,
    )
    print(f"\nRanked by expectancy ($/trade): {' > '.join(name for name, _ in ranked)}")
    if ranked:
        best_name, best = ranked[0]
        baseline = results.get("baseline", {})
        if baseline.get("trades"):
            print(
                f"\nBest ({best_name}) vs baseline: expectancy {baseline['expectancy_dollars']:+.2f} -> "
                f"{best['expectancy_dollars']:+.2f}, drawdown {baseline['max_drawdown_pct']:.2f}% -> "
                f"{best['max_drawdown_pct']:.2f}%, win rate {baseline['win_rate_pct']:.1f}% -> {best['win_rate_pct']:.1f}%"
            )
    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("paper_trading.py, fvg_analyzer.py, or entry_insight.py regardless of which variant wins.")


if __name__ == "__main__":
    main()
