"""
SMC confidence-weighting backtest matrix — Path 1 only, quantitative
experiment. Does NOT modify strategy/fvg_analyzer.py, entry_insight.py,
paper_trading.py, or touch any live process. Reuses the same real
SmartMoneyAnalyzer/FairValueGapAnalyzer/PaperTradingLedger classes those
files use, exactly like backtest_target_matrix.py and
hmm_regime_filter_backtest.py did.

Tests two ideas surfaced from reviewing outside trading content, translated
into concrete testable rules instead of taken on faith:

  respect_bonus  - boost confidence when the FVG being traded has already
                    been touched and held before (FVGZone.respect_count),
                    not just freshly formed. The "wait for 2+ touches"
                    instinct, but as a graded confidence adjustment rather
                    than a hard filter.
  atr_bonus      - boost confidence when the FVG's gap is unusually large
                    relative to the symbol's own recent ATR% (an
                    "unhealthy/overextended move" proxy) -- already-computed
                    ATR (SmartMoneyAnalyzer._compute_atr(), exposed via
                    smc_signals["atr"]) compared against the FVG's own
                    gap_size_percent.
  combined       - both bonuses applied together.

All variants use the SAME entry_zone/stop_loss/target as baseline -- only
the confidence value (which drives conviction-scaled position sizing and
the ENTRY_INSIGHT_MIN_CONFIDENCE veto) differs.

Confidence is intercepted and overridden AFTER FairValueGapAnalyzer.
get_trading_signal() computes it (same call, real object), not by
reimplementing FVG detection -- get_nearest_fvg() is called a second time
with the same current_price to fetch the actual FVGZone object used, so
respect_count/gap_size_percent come from the real, live analyzer state.

Usage: python smc_confidence_matrix_backtest.py
"""
from __future__ import annotations

from backtest import CRYPTO_TICKERS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

RESPECT_BONUS_PER_TOUCH = 8.0   # confidence points per prior respected touch
RESPECT_BONUS_CAP = 24.0        # max total bonus (3 touches)
ATR_GAP_MULTIPLE = 1.5          # gap must be this many multiples of ATR% to earn the bonus
ATR_BONUS = 15.0                # confidence points awarded when it clears that bar

RUNS = [(60, "5m"), (60, "15m"), (180, "1h")]  # matches backtest_target_matrix.py


def _confidence_bonus(fvg, smc_signals: dict, mode: str, price: float) -> float:
    if fvg is None:
        return 0.0
    bonus = 0.0
    if mode in ("respect", "combined"):
        bonus += min(fvg.respect_count * RESPECT_BONUS_PER_TOUCH, RESPECT_BONUS_CAP)
    if mode in ("atr", "combined"):
        atr = smc_signals.get("atr", 0.0)
        atr_pct = (atr / price * 100) if price else 0.0
        if atr_pct > 0 and fvg.gap_size_percent >= ATR_GAP_MULTIPLE * atr_pct:
            bonus += ATR_BONUS
    return bonus


def replay(symbols_candles: dict, timeframe: str, mode: str) -> PaperTradingLedger:
    """mode: 'baseline' | 'respect' | 'atr' | 'combined'"""
    ledger = PaperTradingLedger(path="backtrack_logs/_smc_confidence_scratch.json", starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
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

            if mode != "baseline" and fvg_signal.get("direction") in ("long", "short"):
                fvg_type = "bullish" if fvg_signal["direction"] == "long" else "bearish"
                nearest = fvg.get_nearest_fvg(candle.close, fvg_type)
                bonus = _confidence_bonus(nearest, smc_signals, mode, candle.close)
                if bonus:
                    fvg_signal["confidence"] = min(100.0, fvg_signal["confidence"] + bonus)

            smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=candle.timestamp.isoformat(), timeframe=timeframe)
            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    ledger.open_positions.clear()
    return ledger


def compute_full_stats(trades: list[dict], starting_balance: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    equity, peak, max_dd = starting_balance, starting_balance, 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
    return {
        "trades": n,
        "win_rate_pct": len(wins) / n * 100,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / starting_balance * 100,
        "expectancy": total_pnl / n,
        "max_drawdown_pct": max_dd,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
    }


def print_stats_row(name: str, s: dict) -> None:
    if s["trades"] == 0:
        print(f"{name:<12}{'0':<8}(no trades)")
        return
    print(
        f"{name:<12}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['total_pnl']:<+11.2f}"
        f"{s['total_return_pct']:<+10.2f}{s['expectancy']:<+12.2f}{s['max_drawdown_pct']:<9.2f}"
        f"{s['profit_factor']:<9.2f}{s['avg_win']:<+10.2f}{s['avg_loss']:<+10.2f}"
    )


def main() -> None:
    header = (
        f"{'Variant':<12}{'Trades':<8}{'WinRate%':<10}{'TotalPnL$':<11}{'Return%':<10}"
        f"{'Expect$':<12}{'MaxDD%':<9}{'PF':<9}{'AvgWin$':<10}{'AvgLoss$':<10}"
    )

    for days, timeframe in RUNS:
        print(f"\n{'=' * len(header)}")
        print(f"TIMEFRAME {timeframe} ({days}d)")
        print(f"{'=' * len(header)}")

        all_candles = {}
        for symbol in CRYPTO_TICKERS:
            c = fetch_candles(symbol, days, timeframe)
            if c:
                all_candles[symbol] = c

        starting_balance = settings.PAPER_TRADING_STARTING_BALANCE
        print(header)
        print("-" * len(header))
        for mode in ["baseline", "respect", "atr", "combined"]:
            ledger = replay(all_candles, timeframe, mode)
            print_stats_row(mode, compute_full_stats(ledger.closed_trades, starting_balance))

    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("strategy/fvg_analyzer.py, entry_insight.py, or paper_trading.py regardless of outcome.")


if __name__ == "__main__":
    main()
