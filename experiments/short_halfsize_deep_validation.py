"""
Deep validation for SHORT_20PCT (the directional-sizing finding from
directional_sizing_robustness.py -- robust 6/6 segments, broad plateau
5-25%, crypto-only) before it's trusted enough to deploy. Applies the
same scrutiny that caught EMA50's overfit: per-symbol concentration,
drawdown, statistical significance -- plus one question specific to this
finding: is the extra profit coming from better-sized SHORT trades
themselves, or from smaller SHORT positions freeing up the portfolio
correlation cap's $ budget so MORE trades (of either direction) get
through that baseline would have rejected? Both are legitimate, but they
have different implications -- the first is a direct sizing edge, the
second is an indirect capacity effect worth knowing about explicitly
rather than assuming.

Reuses the exact drawdown/significance helpers from
ema50_robustness_validation.py so the methodology is identical to what
already caught one false positive this session.

Usage: python -m experiments.short_halfsize_deep_validation [--days N] [--timeframe 1h]
"""
from __future__ import annotations

import argparse
import math

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import MIN_STOP_DISTANCE_PCT, PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SHORT_MULTIPLIER = 0.20  # middle of the validated 5-25% robust plateau


def apply_short_halfsize(insight: dict) -> dict:
    if insight.get("direction") != "short":
        return insight
    result = dict(insight)
    deriv = dict(result["derivative"])
    deriv["confidence"] = deriv.get("confidence", 0.0) * SHORT_MULTIPLIER
    result["derivative"] = deriv
    return result


def max_drawdown(ledger: PaperTradingLedger) -> float:
    trades = sorted(ledger.closed_trades, key=lambda t: t["closed_at"])
    equity = ledger.starting_balance
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
    return max_dd


def two_proportion_z(wins_a: int, n_a: int, wins_b: int, n_b: int) -> tuple[float, float]:
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a, p_b = wins_a / n_a, wins_b / n_b
    p_pool = (wins_a + wins_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def per_symbol_breakdown(ledger: PaperTradingLedger) -> dict[str, tuple[int, float]]:
    out: dict[str, list] = {}
    for t in ledger.closed_trades:
        sym = t["symbol"]
        out.setdefault(sym, [0, 0.0])
        out[sym][0] += 1
        out[sym][1] += t["pnl_dollars"]
    return {k: (v[0], round(v[1], 2)) for k, v in out.items()}


def run_full(symbols_candles: dict[str, list[Candle]], timeframe: str) -> tuple[PaperTradingLedger, PaperTradingLedger, dict]:
    """Single pass, both arms, PLUS instrumentation: counts how many
    derivative-short attempts baseline's correlation cap rejected vs how
    many the smaller-sized variant let through -- isolates the "more
    trades pass the cap" mechanism from "better per-trade sizing"."""
    baseline = PaperTradingLedger(path="backtrack_logs/_short_deep_baseline.json", starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    baseline.balance = baseline.starting_balance
    baseline.max_closed_trades = 1_000_000
    variant = PaperTradingLedger(path="backtrack_logs/_short_deep_variant.json", starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    variant.balance = variant.starting_balance
    variant.max_closed_trades = 1_000_000

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}

    stats = {
        "short_signals_seen": 0,
        "baseline_short_opened": 0, "baseline_short_capcap_rejected": 0,
        "variant_short_opened": 0, "variant_short_capcap_rejected": 0,
    }

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

            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()
            is_short_signal = insight.get("direction") == "short" and insight["derivative"]["actionable"]
            if is_short_signal:
                stats["short_signals_seen"] += 1

            baseline.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            base_before = len(baseline.closed_trades) + len(baseline.open_positions)
            reject_reason = baseline.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            deriv_reject = baseline.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)
            if is_short_signal:
                opened_now = (len(baseline.closed_trades) + len(baseline.open_positions)) > base_before
                if opened_now:
                    stats["baseline_short_opened"] += 1
                elif deriv_reject and "correlated-risk cap" in deriv_reject:
                    stats["baseline_short_capcap_rejected"] += 1

            variant_insight = apply_short_halfsize(insight)
            variant.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            var_before = len(variant.closed_trades) + len(variant.open_positions)
            variant.maybe_open(sym, "spot", variant_insight["spot"], timeframe=timeframe, opened_at=opened_at)
            v_deriv_reject = variant.maybe_open(sym, "derivative", variant_insight["derivative"], timeframe=timeframe, opened_at=opened_at)
            if is_short_signal:
                opened_now = (len(variant.closed_trades) + len(variant.open_positions)) > var_before
                if opened_now:
                    stats["variant_short_opened"] += 1
                elif v_deriv_reject and "correlated-risk cap" in v_deriv_reject:
                    stats["variant_short_capcap_rejected"] += 1

    baseline.open_positions.clear()
    variant.open_positions.clear()
    return baseline, variant, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, 730)
    days = min(args.days if args.days is not None else max_days, max_days)

    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols: {settings.WATCHLIST}")
    print(f"Fetching {days}d of {timeframe} candles (via yfinance, max available)...\n")
    print(f"MIN_STOP_DISTANCE_PCT (fee-viability floor, unaffected by this sizing change): {MIN_STOP_DISTANCE_PCT*100:.3f}%\n")

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

    baseline, variant, stats = run_full(all_candles, timeframe)
    bs, vs = baseline.stats()["overall"], variant.stats()["overall"]

    print(f"\n{'=' * 90}")
    print(f"[{settings.ASSET_TYPE.upper()}] SHORT_{int(SHORT_MULTIPLIER*100)}PCT DEEP VALIDATION — full {days}d window")
    print(f"{'=' * 90}")
    print(f"Baseline: {bs['trades']} trades, {bs['win_rate_pct']}% WR, ${bs['total_pnl_dollars']:+.2f}")
    print(f"Variant:  {vs['trades']} trades, {vs['win_rate_pct']}% WR, ${vs['total_pnl_dollars']:+.2f}")

    print(f"\n--- Mechanism check: correlation-cap capacity effect ---")
    print(f"SHORT signals seen (actionable derivative short, either arm): {stats['short_signals_seen']}")
    print(f"Baseline: {stats['baseline_short_opened']} opened, {stats['baseline_short_capcap_rejected']} rejected specifically by correlation cap")
    print(f"Variant:  {stats['variant_short_opened']} opened, {stats['variant_short_capcap_rejected']} rejected specifically by correlation cap")
    extra_shorts_opened = stats['variant_short_opened'] - stats['baseline_short_opened']
    print(f"Extra SHORT trades opened by the smaller-sized variant (freed correlation-cap capacity): {extra_shorts_opened}")

    print(f"\n--- Per-symbol concentration ---")
    base_by_sym = per_symbol_breakdown(baseline)
    var_by_sym = per_symbol_breakdown(variant)
    print(f"{'Symbol':<12}{'Baseline t/PnL':<26}{'Variant t/PnL':<26}")
    for sym in settings.WATCHLIST:
        bt, bp = base_by_sym.get(sym, (0, 0.0))
        vt, vp = var_by_sym.get(sym, (0, 0.0))
        print(f"{sym:<12}{bt:>3}t / ${bp:>+9.2f}          {vt:>3}t / ${vp:>+9.2f}")
    total_var_pnl = sum(p for _, p in var_by_sym.values())
    top_sym_pnl = max((p for _, p in var_by_sym.values()), default=0)
    concentration = (top_sym_pnl / total_var_pnl * 100) if total_var_pnl else 0
    print(f"\nLargest single symbol's share of variant's total PnL: {concentration:.1f}%")

    print(f"\n--- Drawdown & significance ---")
    base_dd, var_dd = max_drawdown(baseline), max_drawdown(variant)
    z, p = two_proportion_z(bs["wins"], bs["trades"], vs["wins"], vs["trades"])
    print(f"Baseline max drawdown: {base_dd:.1f}%   |   Variant max drawdown: {var_dd:.1f}%")
    print(f"Win-rate delta z-stat: {z:+.2f}   p-value: {p:.4f}  {'(significant at p<0.05)' if p < 0.05 else '(NOT significant at p<0.05 on win-rate alone -- PnL is the primary signal here, not WR)'}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
