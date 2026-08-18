"""
Target/exit backtest matrix — Path 1 (SMC/FVG paper trading) ONLY.

Quantitative experiment, not a strategy change: entry signal generation
(FVG detection, direction, confidence, stop-loss) is completely untouched.
This only overrides the `target` price before it reaches build_entry_insight,
comparing variants against the exact same historical candles/entries/stops.

Motivation: the live target is always gap_boundary +/- 0.5*gap_size
(strategy/fvg_analyzer.py's FVGZone.calculate_target), and FVG gaps across
all 8 crypto symbols average ~0.18-0.22% of price (measured directly this
session) — so realized trade PnL is structurally capped near ~1% regardless
of setup quality. This matrix asks: does widening the target (either as a
larger multiple of the same gap, or as a fixed R-multiple of the stop
distance) improve EXPECTANCY and drawdown, or just look better per-trade
while actually hurting hit-rate enough to net worse?

Variants:
  baseline   - unchanged, the live gap_boundary + 0.5*gap_size target
  gap_1.0x / gap_1.5x / gap_2.0x - same FVG-anchored method, wider multiple
  R2 / R3 / R5 - fixed reward:risk multiple of the SAME stop distance
                 (ignores the FVG's own gap size entirely for the target)

Scored by: win rate, total return, expectancy ($/trade), max drawdown (%
from peak equity), and profit factor (gross wins / gross losses) — not
win rate alone, since wider targets are expected to lower hit-rate and the
question is whether the bigger wins more than compensate.

Does not modify backtrack/paper_trading.py, strategy/fvg_analyzer.py, or
strategy/entry_insight.py — everything here is applied AFTER those produce
their normal output, on a private copy of the signal dict passed into
build_entry_insight. Uses its own ledger files (never touches the live
paper-trading ledger main.py maintains).

Usage: python backtest_target_matrix.py [--days N] [--timeframe 5m|15m|1h]
"""
from __future__ import annotations

import argparse

from backtest import CRYPTO_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer


def _gap_multiple_target(direction: str, entry_zone: tuple[float, float], multiple: float) -> float:
    low, high = entry_zone
    gap_size = high - low
    return (high + multiple * gap_size) if direction == "long" else (low - multiple * gap_size)


def _r_multiple_target(direction: str, entry_zone: tuple[float, float], stop_loss: float, multiple: float) -> float:
    entry_price = sum(entry_zone) / 2
    stop_distance = abs(entry_price - stop_loss)
    return (entry_price + multiple * stop_distance) if direction == "long" else (entry_price - multiple * stop_distance)


def _make_variant(kind: str, multiple: float | None = None):
    """Returns a fn(smc_context: dict) -> float | None overriding target, or None for baseline (no override)."""
    if kind == "baseline":
        return None

    def override(smc_context: dict) -> float | None:
        direction = smc_context.get("direction")
        entry_zone = smc_context.get("entry_zone")
        stop_loss = smc_context.get("stop_loss")
        if direction not in ("long", "short") or not entry_zone or stop_loss is None:
            return None
        if kind == "gap":
            return _gap_multiple_target(direction, entry_zone, multiple)
        if kind == "r":
            return _r_multiple_target(direction, entry_zone, stop_loss, multiple)
        raise ValueError(kind)

    return override


VARIANTS: dict[str, tuple[str, float | None]] = {
    "baseline": ("baseline", None),
    "gap_1.0x": ("gap", 1.0),
    "gap_1.5x": ("gap", 1.5),
    "gap_2.0x": ("gap", 2.0),
    "R2": ("r", 2.0),
    "R3": ("r", 3.0),
    "R5": ("r", 5.0),
}


def replay_variant(symbols_candles: dict, timeframe: str, target_override) -> PaperTradingLedger:
    """
    Same lockstep structure as backtest.py's replay_lockstep, simplified:
    no GNN (gnn_signal=0.0 throughout, matching every non---with-gnn run
    elsewhere in this project) and no persistence — this is a comparison
    tool, not a report generator, so ledgers stay in-memory only.
    """
    ledger = PaperTradingLedger(
        path="backtrack_logs/_target_matrix_scratch.json",
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

            if target_override is not None:
                new_target = target_override(smc_context)
                if new_target is not None:
                    smc_context["target"] = new_target

            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=candle.timestamp.isoformat(), timeframe=timeframe)
            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    ledger.open_positions.clear()  # unresolved at window end — not counted
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

    # Equity curve from starting balance, in the trades' own chronological
    # close order (they're appended in that order during replay) -> max
    # drawdown as a % pullback from the running peak.
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
    parser = argparse.ArgumentParser(description="Compare target/exit variants on identical historical entries (Path 1 only).")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", choices=list(YF_MAX_DAYS.keys()), default="15m")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS[timeframe]
    days = min(args.days if args.days is not None else settings.BACKTEST_DAYS, max_days)

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — target/exit matrix over {days}d of {timeframe} candles\n")
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
    print("paper_trading.py or entry_insight.py regardless of which variant wins.")


if __name__ == "__main__":
    main()
