"""
Every prior filter test in this research line (VWAP, the 3-filter matrix,
the 10-indicator survey, all robustness passes) ONLY vetoes the existing
SMC/FVG baseline's own signals — it can never generate a trade the
baseline wouldn't already have flagged, so the best any of them could
structurally do is match baseline (never veto) or underperform it (veto a
winner). That ceiling is exactly why nothing ever beat baseline.

This asks the different question the user raised: independent of SMC/FVG
entirely, do the classic standalone technical-analysis STRATEGIES
(entries generated directly from an indicator crossover/breakout, the way
retail/systematic traders actually run them standalone) do any better?

Five textbook strategies, entries fire only on the crossing/breakout bar
(not persistently while the condition holds — matches how these are
actually defined and traded):
  EMA_CROSS      — EMA(12) crosses EMA(26): golden cross long, death cross short
  MACD_CROSS     — MACD line crosses its signal line
  RSI_REVERSION  — RSI(14) crosses back above 30 (long) / below 70 (short)
  BB_BREAKOUT    — close breaks outside a 20-period, 2sigma Bollinger Band
  DONCHIAN_BREAK — close makes a new 20-period high/low (Turtle-style)

Since there's no FVG/order-block structure to anchor a stop to, sizing
uses ATR(14): stop = 1.5x ATR, target = 3x ATR (2:1 reward:risk, standard
practice for this style of system). Confidence is a flat 50.0 (no
respect-count-style conviction signal exists for a pure crossover system)
— a deliberate simplification, noted as a limitation, not hidden.

Every strategy is run through the SAME PaperTradingLedger machinery as
baseline — same fee-viability filter, same portfolio correlation cap,
same conviction-scaled sizing — so results are directly comparable, not
an apples-to-oranges toy comparison. Same 6-segment, ~2-year multi-regime
bar as every other robustness pass in this line.

Usage: python standalone_ta_strategies_robustness.py [--segments N] [--days N] [--timeframe 1h]
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer
from .ta_indicators import (
    atr_series, bollinger_bands_series, donchian_mid_series, macd_hist_series, rsi_series,
)
from train_gnn import _ema

STRATEGY_NAMES = ["EMA_CROSS", "MACD_CROSS", "RSI_REVERSION", "BB_BREAKOUT", "DONCHIAN_BREAK"]
ATR_STOP_MULT = 1.5
ATR_TARGET_MULT = 3.0
FLAT_CONFIDENCE = 50.0


def compute_strategy_state(candles: list[Candle]) -> dict:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    ema_fast, ema_slow = _ema(12, closes), _ema(26, closes)
    macd_hist = macd_hist_series(closes)
    rsi = rsi_series(closes, 14)
    bb_upper, bb_lower = bollinger_bands_series(closes, 20, 2.0)
    atr = atr_series(highs, lows, closes, 14)

    n = len(closes)
    donchian_high = [0.0] * n
    donchian_low = [0.0] * n
    for i in range(n):
        window_highs = highs[max(0, i - 20):i]  # EXCLUDES current bar — a breakout must clear the prior window
        window_lows = lows[max(0, i - 20):i]
        donchian_high[i] = max(window_highs) if window_highs else highs[i]
        donchian_low[i] = min(window_lows) if window_lows else lows[i]

    return {
        "closes": closes, "ema_fast": ema_fast, "ema_slow": ema_slow,
        "macd_hist": macd_hist, "rsi": rsi, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "atr": atr, "donchian_high": donchian_high, "donchian_low": donchian_low,
    }


def strategy_direction(name: str, state: dict, t: int) -> str:
    """Returns 'long'/'short'/'neutral' — 'long'/'short' only on the exact
    crossing/breakout bar, never persistently."""
    if t < 1:
        return "neutral"
    if name == "EMA_CROSS":
        prev_diff = state["ema_fast"][t - 1] - state["ema_slow"][t - 1]
        cur_diff = state["ema_fast"][t] - state["ema_slow"][t]
        if prev_diff <= 0 < cur_diff:
            return "long"
        if prev_diff >= 0 > cur_diff:
            return "short"
        return "neutral"
    if name == "MACD_CROSS":
        prev_h, cur_h = state["macd_hist"][t - 1], state["macd_hist"][t]
        if prev_h <= 0 < cur_h:
            return "long"
        if prev_h >= 0 > cur_h:
            return "short"
        return "neutral"
    if name == "RSI_REVERSION":
        prev_r, cur_r = state["rsi"][t - 1], state["rsi"][t]
        if prev_r < 30 <= cur_r:
            return "long"
        if prev_r > 70 >= cur_r:
            return "short"
        return "neutral"
    if name == "BB_BREAKOUT":
        prev_c, cur_c = state["closes"][t - 1], state["closes"][t]
        if prev_c <= state["bb_upper"][t - 1] and cur_c > state["bb_upper"][t]:
            return "long"
        if prev_c >= state["bb_lower"][t - 1] and cur_c < state["bb_lower"][t]:
            return "short"
        return "neutral"
    if name == "DONCHIAN_BREAK":
        cur_c = state["closes"][t]
        if cur_c > state["donchian_high"][t]:
            return "long"
        if cur_c < state["donchian_low"][t]:
            return "short"
        return "neutral"
    raise ValueError(name)


def build_standalone_insight(price: float, direction: str, atr_val: float) -> dict:
    no_setup = {
        "actionable": False, "direction": "neutral", "entry_zone": None,
        "stop_loss": None, "target": None, "confidence": 0.0, "atr": atr_val, "note": "",
    }
    if direction == "neutral" or atr_val <= 0:
        return {"spot": dict(no_setup), "derivative": dict(no_setup)}

    stop_dist, target_dist = ATR_STOP_MULT * atr_val, ATR_TARGET_MULT * atr_val
    if direction == "long":
        stop_loss, target = price - stop_dist, price + target_dist
    else:
        stop_loss, target = price + stop_dist, price - target_dist

    base = {
        "direction": direction, "entry_zone": (price, price),
        "stop_loss": stop_loss, "target": target, "confidence": FLAT_CONFIDENCE,
        "atr": atr_val, "note": "Standalone TA crossover signal (no SMC/FVG).",
    }
    spot = dict(base, actionable=(direction == "long"))
    derivative = dict(base, actionable=True)
    return {"spot": spot, "derivative": derivative}


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    arm_names = ["BASELINE"] + STRATEGY_NAMES
    ledgers = {}
    for name in arm_names:
        path = f"backtrack_logs/_standalone_ta_{label}_{name}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    state_by_symbol = {sym: compute_strategy_state(candles) for sym, candles in symbols_candles.items()}

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(min_len):
        for sym, candles in symbols_candles.items():
            candle = candles[t]
            state = state_by_symbol[sym]

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

            for name in STRATEGY_NAMES:
                ledger = ledgers[name]
                direction = strategy_direction(name, state, t)
                strat_insight = build_standalone_insight(candle.close, direction, state["atr"][t])
                ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                ledger.maybe_open(sym, "spot", strat_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledger.maybe_open(sym, "derivative", strat_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

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
        if not candles:
            print(f"  {symbol}: no historical data, skipping")
            continue
        all_candles[symbol] = candles
        print(f"  {symbol}: {len(candles)} candles ({candles[0].timestamp} -> {candles[-1].timestamp})")

    if not all_candles:
        print("No data — aborting.")
        return

    n = min(len(c) for c in all_candles.values())
    n_seg = args.segments
    bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]
    segments = [{sym: candles[bounds[i]:bounds[i + 1]] for sym, candles in all_candles.items()} for i in range(n_seg)]

    print(f"\n{'=' * 100}")
    print(f"[{settings.ASSET_TYPE.upper()}] STANDALONE TA STRATEGIES vs SMC/FVG BASELINE — {n_seg} segments across {days}d")
    print(f"{'=' * 100}")

    wins_per_strategy = {name: 0 for name in STRATEGY_NAMES}
    total_pnl_per_strategy = {name: 0.0 for name in STRATEGY_NAMES}
    total_baseline_pnl = 0.0

    for i, seg in enumerate(segments):
        results = run_segment(seg, timeframe, f"seg{i}")
        base_pnl = results["BASELINE"]["overall"]["total_pnl_dollars"]
        base_wr = results["BASELINE"]["overall"]["win_rate_pct"]
        total_baseline_pnl += base_pnl
        start_ts = min((c[0].timestamp for c in seg.values() if c), default=None)
        end_ts = max((c[-1].timestamp for c in seg.values() if c), default=None)
        period_str = f"{start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d}" if start_ts and end_ts else "-"
        bwr_str = f"{base_wr:.1f}%" if base_wr is not None else "n/a"

        print(f"\n--- seg{i} ({period_str}) --- BASELINE (SMC/FVG): {results['BASELINE']['overall']['trades']}t / {bwr_str} WR / ${base_pnl:+.0f}")
        print(f"{'Strategy':<16}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}")
        for name in STRATEGY_NAMES:
            o = results[name]["overall"]
            wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
            beats = o["total_pnl_dollars"] > base_pnl
            if beats:
                wins_per_strategy[name] += 1
            total_pnl_per_strategy[name] += o["total_pnl_dollars"]
            tag = "WINS" if beats else ""
            print(f"{name:<16}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+10.2f}   {tag}")

    print(f"\n{'=' * 100}")
    print(f"[{settings.ASSET_TYPE.upper()}] SUMMARY — segments won (of {n_seg}), aggregate vs baseline ${total_baseline_pnl:+.0f}")
    print(f"{'=' * 100}")
    ranked = sorted(STRATEGY_NAMES, key=lambda n: (wins_per_strategy[n], total_pnl_per_strategy[n]), reverse=True)
    print(f"{'Strategy':<16}{'Segments won':<16}{'Aggregate PnL':<18}{'vs baseline':<16}")
    print("-" * 100)
    for name in ranked:
        w = wins_per_strategy[name]
        agg = total_pnl_per_strategy[name]
        delta = agg - total_baseline_pnl
        robust = w == n_seg
        tag = " <- ROBUST" if robust else ""
        print(f"{name:<16}{w}/{n_seg:<14}${agg:>+13.2f}   ${delta:>+13.2f}{tag}")
    print("-" * 100)
    robust = [n for n in STRATEGY_NAMES if wins_per_strategy[n] == n_seg]
    print(f"\nStandalone strategies beating SMC/FVG baseline in ALL {n_seg} segments: {robust if robust else 'NONE'}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
