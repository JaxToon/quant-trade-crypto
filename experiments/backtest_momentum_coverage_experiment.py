"""
Momentum-coverage-gap experiment — Path 1 (SMC/FVG paper trading) ONLY.

Tests whether a momentum/breakout overlay adds genuine incremental value
specifically in market conditions SMC/FVG structurally can't see (clean,
low-pullback directional moves — no order-block retest, no fresh FVG),
WITHOUT hurting the existing SMC/FVG edge where it already works. This is
a hypothesis TEST, not a strategy change — same discipline as the target-
width, regime-filter, confidence-weighting, and stop-loss experiments:
backtest first, deploy only if the result clearly earns it.

Motivation: 2026-08-19's BTC rally (~+8.7% over ~18h, low pullback)
produced almost no SMC/FVG signal for hours. Not a bug — SMC/FVG trades
structural retracements by design — but a real open question: is that a
genuine, uncaptured opportunity, or would trying to capture it just add
noise/drawdown without real edge?

HARD INVARIANT — causal-only features, no exceptions:
The momentum signal at bar t is computed ONLY from candles up to and
including t (via smc.candles, the same incrementally-built rolling
window SmartMoneyAnalyzer.update() already uses one candle at a time in
strict chronological order). No future candle, no realized subsequent
return, no hindsight "this was a breakout" labeling. This exists
specifically to avoid the same class of bug already found and fixed
twice elsewhere in this project: the regime-detection bug (single-
observation decode dominated by startprob_) and the opened_at backtest
bug (recorded wall-clock time instead of candle time, silently breaking
time-based analysis). A momentum feature that peeks forward would be a
new instance of the same mistake, just relabeled.

Momentum signal (all causal):
  - directional consistency: >= MOMENTUM_MIN_CONSISTENCY of the closes
    in the last MOMENTUM_LOOKBACK bars move in the same direction as the
    net move over that window
  - magnitude: |net move| over that window >= MOMENTUM_MIN_ATR_MULTIPLE * ATR
    (volatility-normalized, so a "big move" means big relative to the
    instrument's own noise, not a fixed dollar/percent threshold)
  Both must hold, else momentum_direction = "neutral" for that bar.
  These thresholds are principled first-pass defaults, NOT tuned against
  this evaluation window — sweeping/cherry-picking them here would be the
  exact p-hacking this project's backtest discipline exists to avoid. If
  this direction shows promise, threshold sensitivity is a SEPARATE,
  later question.

Momentum SL/TP are ATR-based (no FVG/order-block to structurally anchor
to, unlike SMC/FVG trades — see backtest_stoploss_matrix.py for why
ATR-based stops were rejected FOR THE SMC/FVG SIGNAL specifically; that
finding doesn't automatically transfer to a differently-sourced signal,
which is part of what this experiment checks):
  SL = entry -/+ MOMENTUM_SL_ATR_MULTIPLE * ATR
  TP = entry +/- MOMENTUM_TP_ATR_MULTIPLE * ATR

Three variants, identical historical candles/entries otherwise:
  baseline               - SMC/FVG only, unchanged (matches live exactly)
  smc_momentum_fallback  - momentum only opens a trade when SMC/FVG has
                            NO actionable setup for that symbol/tick this
                            bar — tests the core hypothesis directly
  smc_momentum_always    - momentum can open regardless of whether SMC/FVG
                            also has a setup — control, tests whether
                            momentum has standalone value or just degrades
                            the baseline when blanket-applied (this is the
                            variant the "coverage gap" hypothesis predicts
                            should NOT clearly beat smc_momentum_fallback)

Momentum-sourced trades are tagged via a distinct timeframe suffix
(f"{timeframe}_momentum") so they land in PaperTradingLedger's own
by-timeframe bucketing — separable from SMC-sourced trades in the SAME
ledger and equity curve, with zero changes to backtrack/paper_trading.py.

Evaluated on whether the fallback variant's OVERALL expectancy/drawdown
holds or improves vs baseline — NOT just "does trade count go up" (more
trades alone is not the goal) — plus the momentum-only trades' own
expectancy/win-rate in isolation, i.e. does the coverage-gap-filling
carry real edge on its own, or is it just adding noise that happens not
to sink the total.

Does not modify backtrack/paper_trading.py, strategy/fvg_analyzer.py,
strategy/smart_money.py, or strategy/entry_insight.py. Uses its own
scratch ledger files, one per variant. Fully isolated from any live
main.py process — pure historical replay, no live data stream, no shared
state.

Usage: python backtest_momentum_coverage_experiment.py [--days N] [--timeframe 5m|15m|1h]
"""
from __future__ import annotations

import argparse

from backtest import CRYPTO_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

MOMENTUM_LOOKBACK = 12
MOMENTUM_MIN_CONSISTENCY = 0.65
MOMENTUM_MIN_ATR_MULTIPLE = 2.0
MOMENTUM_SL_ATR_MULTIPLE = 1.5
MOMENTUM_TP_ATR_MULTIPLE = 3.0
MOMENTUM_CONFIDENCE = 50.0  # fixed, mid-range -- not scaled by signal strength, to avoid an extra tunable that could be backfit

VARIANTS = ["baseline", "smc_momentum_fallback", "smc_momentum_always"]


def _compute_momentum(candle_hist: list, atr: float) -> str:
    """Causal: candle_hist is the history up to and including 'now' — see
    module docstring's HARD INVARIANT. Returns 'long', 'short', or 'neutral'."""
    if len(candle_hist) < MOMENTUM_LOOKBACK + 1 or not atr or atr <= 0:
        return "neutral"
    window = candle_hist[-MOMENTUM_LOOKBACK:]
    closes = [c.close for c in window]
    net_move = closes[-1] - closes[0]
    if abs(net_move) < MOMENTUM_MIN_ATR_MULTIPLE * atr:
        return "neutral"
    direction_sign = 1 if net_move > 0 else -1
    step_count = len(closes) - 1
    consistent_steps = sum(1 for i in range(1, len(closes)) if (closes[i] - closes[i - 1]) * direction_sign > 0)
    if consistent_steps / step_count < MOMENTUM_MIN_CONSISTENCY:
        return "neutral"
    return "long" if direction_sign > 0 else "short"


def _momentum_recs(direction: str, entry_price: float, atr: float) -> tuple[dict, dict]:
    sl_distance = MOMENTUM_SL_ATR_MULTIPLE * atr
    tp_distance = MOMENTUM_TP_ATR_MULTIPLE * atr
    if direction == "long":
        stop_loss, target = entry_price - sl_distance, entry_price + tp_distance
    else:
        stop_loss, target = entry_price + sl_distance, entry_price - tp_distance

    spot = {
        "actionable": direction == "long",  # spot can't short -- same rule as entry_insight._spot_recommendation
        "direction": direction, "entry_zone": [entry_price, entry_price],
        "stop_loss": stop_loss, "target": target, "confidence": MOMENTUM_CONFIDENCE,
    }
    derivative = {
        "actionable": True,
        "direction": direction, "entry_zone": [entry_price, entry_price],
        "stop_loss": stop_loss, "target": target, "confidence": MOMENTUM_CONFIDENCE,
        "suggested_leverage": settings.INSIGHT_DERIVATIVE_LEVERAGE,
    }
    return spot, derivative


def replay_variant(symbols_candles: dict, timeframe: str, variant: str) -> PaperTradingLedger:
    """Same lockstep structure as the other *_matrix backtest scripts in
    this repo: no GNN (gnn_signal=0.0 throughout), no persistence."""
    ledger = PaperTradingLedger(
        path=f"backtrack_logs/_momentum_scratch_{variant}.json",
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
            candle_hist = list(smc.candles)  # causal: only bars up to and including `candle`
            if len(candle_hist) >= 3:
                fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}
            atr = smc_context.get("atr", 0.0)

            # timeframe=None: checks both the plain-timeframe (SMC) and
            # "{timeframe}_momentum" (momentum) open positions for this
            # symbol against this candle's [low, high] range.
            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=candle.timestamp.isoformat(), timeframe=None)

            opened_at = candle.timestamp.isoformat()
            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            ledger.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)
            smc_has_setup = bool(insight["spot"].get("actionable") or insight["derivative"].get("actionable"))

            if variant != "baseline":
                fire_momentum = variant == "smc_momentum_always" or (variant == "smc_momentum_fallback" and not smc_has_setup)
                if fire_momentum:
                    mom_dir = _compute_momentum(candle_hist, atr)
                    if mom_dir != "neutral":
                        mom_spot, mom_deriv = _momentum_recs(mom_dir, candle.close, atr)
                        mom_tf = f"{timeframe}_momentum"
                        ledger.maybe_open(sym, "spot", mom_spot, timeframe=mom_tf, opened_at=opened_at)
                        ledger.maybe_open(sym, "derivative", mom_deriv, timeframe=mom_tf, opened_at=opened_at)

    ledger.open_positions.clear()  # unresolved at window end -- not counted
    return ledger


def _score(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate_pct": None, "total_pnl_dollars": 0.0, "expectancy_dollars": 0.0, "profit_factor": 0.0}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    return {
        "trades": n,
        "win_rate_pct": len(wins) / n * 100,
        "total_pnl_dollars": total_pnl,
        "expectancy_dollars": total_pnl / n,
        "profit_factor": profit_factor,
    }


def compute_stats(ledger: PaperTradingLedger) -> dict:
    trades = ledger.closed_trades
    smc_trades = [t for t in trades if not t["timeframe"].endswith("_momentum")]
    mom_trades = [t for t in trades if t["timeframe"].endswith("_momentum")]

    overall = _score(trades)
    equity = ledger.starting_balance
    peak = equity
    max_dd_pct = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100)
    overall["max_drawdown_pct"] = max_dd_pct
    overall["total_return_pct"] = overall["total_pnl_dollars"] / ledger.starting_balance * 100
    overall["smc_only"] = _score(smc_trades)
    overall["momentum_only"] = _score(mom_trades)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Momentum-coverage-gap experiment: does a momentum overlay fill SMC/FVG's structural blind spot without hurting its existing edge? (Path 1 only)")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", choices=list(YF_MAX_DAYS.keys()), default="15m")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS[timeframe]
    days = min(args.days if args.days is not None else settings.BACKTEST_DAYS, max_days)

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — momentum-coverage-gap experiment over {days}d of {timeframe} candles")
    print(f"Momentum signal: lookback={MOMENTUM_LOOKBACK} bars, min_consistency={MOMENTUM_MIN_CONSISTENCY}, min_move={MOMENTUM_MIN_ATR_MULTIPLE}xATR")
    print(f"Momentum risk: SL={MOMENTUM_SL_ATR_MULTIPLE}xATR, TP={MOMENTUM_TP_ATR_MULTIPLE}xATR (untuned first-pass defaults)\n")

    all_candles: dict[str, list] = {}
    tickers = CRYPTO_TICKERS if settings.ASSET_TYPE == "crypto" else {s: [s] for s in settings.STOCK_WATCHLIST}
    for symbol, yf_tickers in tickers.items():
        candles = fetch_candles(symbol, days, timeframe)
        if candles:
            all_candles[symbol] = candles
            print(f"  {symbol}: {len(candles)} candles")
    print()

    results = {}
    for variant in VARIANTS:
        ledger = replay_variant(all_candles, timeframe, variant)
        results[variant] = compute_stats(ledger)

    print("=== Overall (all trades combined per variant) ===")
    header = f"{'Variant':<24}{'Trades':<8}{'WinRate%':<10}{'Return%':<10}{'MaxDD%':<9}{'ProfitFactor':<13}"
    print(header)
    print("-" * len(header))
    for name, s in results.items():
        if s["trades"] == 0:
            print(f"{name:<24}{'0':<8}(no trades)")
            continue
        print(
            f"{name:<24}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['total_return_pct']:<+10.2f}"
            f"{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}"
        )

    print("\n=== Momentum-sourced trades only (within each variant that has them) ===")
    header2 = f"{'Variant':<24}{'Trades':<8}{'WinRate%':<10}{'Expectancy$':<13}{'ProfitFactor':<13}"
    print(header2)
    print("-" * len(header2))
    for name in VARIANTS:
        s = results[name]["momentum_only"]
        if s["trades"] == 0:
            print(f"{name:<24}{'0':<8}(no momentum trades)")
            continue
        print(f"{name:<24}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['profit_factor']:<13.2f}")

    print("\n=== SMC-sourced trades only (within each variant, for reference) ===")
    print(header2)
    print("-" * len(header2))
    for name in VARIANTS:
        s = results[name]["smc_only"]
        if s["trades"] == 0:
            print(f"{name:<24}{'0':<8}(no SMC trades)")
            continue
        print(f"{name:<24}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['profit_factor']:<13.2f}")

    baseline = results.get("baseline", {})
    fallback = results.get("smc_momentum_fallback", {})
    always = results.get("smc_momentum_always", {})
    print("\n=== Verdict ===")
    if baseline.get("trades") and fallback.get("trades"):
        print(
            f"baseline -> fallback: expectancy n/a (see per-trade above), return {baseline['total_return_pct']:+.2f}% -> "
            f"{fallback['total_return_pct']:+.2f}%, drawdown {baseline['max_drawdown_pct']:.2f}% -> {fallback['max_drawdown_pct']:.2f}%"
        )
        mom_in_fallback = fallback["momentum_only"]
        if mom_in_fallback["trades"] > 0:
            print(
                f"Momentum-sourced trades in fallback mode: {mom_in_fallback['trades']} trades, "
                f"{mom_in_fallback['win_rate_pct']:.1f}% win rate, expectancy ${mom_in_fallback['expectancy_dollars']:+.2f}/trade, "
                f"profit factor {mom_in_fallback['profit_factor']:.2f}"
            )
            coverage_hypothesis_supported = (
                mom_in_fallback["expectancy_dollars"] > 0
                and fallback["max_drawdown_pct"] <= baseline["max_drawdown_pct"] * 1.1  # allow small noise margin
                and fallback["total_return_pct"] >= baseline["total_return_pct"]
            )
            print(f"Coverage-gap hypothesis supported at this timeframe: {coverage_hypothesis_supported}")
        else:
            print("Momentum never fired in fallback mode at this timeframe/window -- no coverage gap found here, or thresholds too strict for this sample.")
    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("paper_trading.py, entry_insight.py, or any live strategy code regardless of which variant wins.")


if __name__ == "__main__":
    main()
