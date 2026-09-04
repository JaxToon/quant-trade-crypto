"""
Tests dynamic stop movement (breakeven + ATR trailing) against baseline —
the one exit-side lever this project has never evaluated.

Motivated directly by `experiments/mfe_mae_exit_analysis.py`'s finding: the
median LOSING trade reaches +0.72-0.77% profit before reversing into a full
stop-loss, and the top quartile of losers reaches +1.05-1.20%, versus a
median WINNER realizing only 1.10-1.275%. Many losers therefore get most of
the way to target, give it all back, and take a full stop. Moving the stop
to breakeven (or trailing it) once a trade is meaningfully in profit should
convert some slice of those full losses into scratches or small wins.

Every prior exit-side experiment here tested a STATIC distance:
`backtest_target_matrix.py` (target distance — wider is monotonically
worse, closed) and `backtest_stoploss_matrix.py` (stop distance — ATR-based
stops rejected). Neither tested a stop that MOVES during the trade.
`PAPER_TRADING_TRAILING_STOP_ENABLED` ships disabled and a grep of
RESEARCH_NOTES.md finds zero prior research on it.

Variants (all vs the same BASELINE, same 6-segment/~2yr multi-regime bar
every other robustness pass in this project uses):

  BE_0.5R / BE_0.7R / BE_1.0R
      Breakeven: once profit reaches N x initial risk, move the stop to
      entry. Implemented here (production has no breakeven rule) by
      tightening pos.stop_loss in place -- the ledger's own
      `_effective_stop` then honours it like any other stop.
  TRAIL_1.0R_2ATR / TRAIL_0.5R_1.5ATR / TRAIL_1.0R_1ATR
      The EXISTING production trailing implementation
      (`_maybe_update_trailing_stop`), enabled and swept over its two
      parameters (activation in R, ATR multiple).

A variant only counts as robust if it beats baseline in ALL segments --
same bar EMA50 failed and SHORT-sizing passed.

Usage: python -m experiments.trailing_breakeven_robustness [--segments 6] [--days 730]
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

# name -> (breakeven_activation_R or None, trailing_activation_R or None, atr_multiple or None)
VARIANTS: dict[str, tuple] = {
    "BE_0.5R": (0.5, None, None),
    "BE_0.7R": (0.7, None, None),
    "BE_1.0R": (1.0, None, None),
    "TRAIL_1.0R_2ATR": (None, 1.0, 2.0),
    "TRAIL_0.5R_1.5ATR": (None, 0.5, 1.5),
    "TRAIL_1.0R_1ATR": (None, 1.0, 1.0),
}


def _apply_breakeven(ledger: PaperTradingLedger, symbol: str, high: float, low: float, activation_r: float) -> None:
    """Move the stop to entry once profit >= activation_r x initial risk.

    Uses the candle's favorable extreme (high for longs, low for shorts) as
    'best price reached this bar' -- the same approximation
    `check_and_close_ohlc` already makes for trailing, and the same
    inherent OHLC-only limitation documented there.
    """
    for pos in ledger.open_positions.values():
        if pos.symbol != symbol:
            continue
        initial_risk = abs(pos.entry_price - pos.stop_loss)
        if initial_risk <= 0:
            continue
        if pos.direction == "long":
            if (high - pos.entry_price) / initial_risk >= activation_r:
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
        else:
            if (pos.entry_price - low) / initial_risk >= activation_r:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price)


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    arm_names = ["BASELINE"] + list(VARIANTS)
    ledgers = {}
    for name in arm_names:
        ledger = PaperTradingLedger(
            path=f"backtrack_logs/_trailing_{label}_{name}.json",
            starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
        )
        ledger.balance = ledger.starting_balance
        ledger.open_positions.clear()
        ledger.closed_trades.clear()
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

    smc_by_symbol = {s: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for s in symbols_candles}
    fvg_by_symbol = {s: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for s in symbols_candles}

    # Saved so each arm can flip the global trailing settings around its own
    # ledger calls without leaking into the others (settings is module-level
    # global state the ledger reads directly).
    orig_enabled = settings.PAPER_TRADING_TRAILING_STOP_ENABLED
    orig_activation = settings.PAPER_TRADING_TRAILING_ACTIVATION_R
    orig_atr_mult = settings.PAPER_TRADING_TRAILING_ATR_MULTIPLE

    try:
        min_len = min((len(c) for c in symbols_candles.values()), default=0)
        for t in range(min_len):
            for sym, candles in symbols_candles.items():
                candle = candles[t]
                smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
                smc_signals = smc.update(candle)
                hist = list(smc.candles)
                if len(hist) >= 3:
                    fvg.detect_fvg(candles=hist, index=len(hist) - 2, current_price=candle.close, volume=candle.volume)
                fvg.update_all(candle.close)
                ms_trend = smc_signals.get("market_structure", "unknown")
                fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
                smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}

                insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
                opened_at = candle.timestamp.isoformat()

                for name in arm_names:
                    ledger = ledgers[name]
                    be_r, trail_r, atr_mult = VARIANTS.get(name, (None, None, None))

                    if be_r is not None:
                        _apply_breakeven(ledger, sym, candle.high, candle.low, be_r)

                    settings.PAPER_TRADING_TRAILING_STOP_ENABLED = trail_r is not None
                    if trail_r is not None:
                        settings.PAPER_TRADING_TRAILING_ACTIVATION_R = trail_r
                        settings.PAPER_TRADING_TRAILING_ATR_MULTIPLE = atr_mult

                    ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                    ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
                    ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)
    finally:
        settings.PAPER_TRADING_TRAILING_STOP_ENABLED = orig_enabled
        settings.PAPER_TRADING_TRAILING_ACTIVATION_R = orig_activation
        settings.PAPER_TRADING_TRAILING_ATR_MULTIPLE = orig_atr_mult

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

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols")
    print(f"Fetching {days}d of {timeframe} candles...\n", flush=True)

    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        candles = fetch_candles(symbol, days, timeframe)
        if candles:
            all_candles[symbol] = candles
            print(f"  {symbol}: {len(candles)} candles", flush=True)

    if not all_candles:
        print("No data — aborting.")
        return

    n = min(len(c) for c in all_candles.values())
    n_seg = args.segments
    bounds = [round(n * i / n_seg) for i in range(n_seg + 1)]
    segments = [{s: c[bounds[i]:bounds[i + 1]] for s, c in all_candles.items()} for i in range(n_seg)]

    print(f"\n{'=' * 100}")
    print(f"[{settings.ASSET_TYPE.upper()}] TRAILING / BREAKEVEN STOP ROBUSTNESS — {n_seg} segments across {days}d")
    print(f"{'=' * 100}", flush=True)

    names = list(VARIANTS)
    wins = {k: 0 for k in names}
    totals = {k: 0.0 for k in names}
    base_total = 0.0

    for i, seg in enumerate(segments):
        res = run_segment(seg, timeframe, f"seg{i}")
        b = res["BASELINE"]["overall"]
        base_total += b["total_pnl_dollars"]
        bwr = f"{b['win_rate_pct']:.1f}%" if b["win_rate_pct"] is not None else "n/a"
        print(f"\n--- seg{i} --- BASELINE: {b['trades']}t / {bwr} WR / ${b['total_pnl_dollars']:+,.0f}")
        print(f"{'Variant':<20}{'Trades':<9}{'Win Rate':<11}{'PnL $':<16}")
        for name in names:
            o = res[name]["overall"]
            wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
            beat = o["total_pnl_dollars"] > b["total_pnl_dollars"]
            wins[name] += 1 if beat else 0
            totals[name] += o["total_pnl_dollars"]
            print(f"{name:<20}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+12,.2f}   {'WINS' if beat else ''}", flush=True)

    print(f"\n{'=' * 100}")
    print(f"SUMMARY — segments won (of {n_seg}), vs baseline aggregate ${base_total:+,.0f}")
    print(f"{'=' * 100}")
    print(f"{'Variant':<20}{'Segments won':<16}{'Aggregate PnL':<20}{'vs baseline':<18}")
    print("-" * 100)
    for name in sorted(names, key=lambda k: (wins[k], totals[k]), reverse=True):
        tag = " <- ROBUST" if wins[name] == n_seg else ""
        print(f"{name:<20}{wins[name]}/{n_seg:<14}${totals[name]:>+15,.2f}   ${totals[name]-base_total:>+14,.2f}{tag}")
    print("-" * 100)
    robust = [k for k in names if wins[k] == n_seg]
    print(f"\nBeating baseline in ALL {n_seg} segments: {robust if robust else 'NONE'}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
