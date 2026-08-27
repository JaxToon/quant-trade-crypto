"""
Competing-hypothesis test flagged by short_halfsize_deep_validation.py's
mechanism analysis: SHORT_20PCT's edge turned out to come almost entirely
from freeing portfolio correlation-cap capacity (779 extra SHORT trades
got through once they consumed less of the shared $800 budget), NOT from
shorts individually trading better (win rate was statistically flat,
p=0.69). That raises the obvious question this script answers: if the
real lever is "the correlation cap has slack," does simply RAISING
PORTFOLIO_MAX_CORRELATED_RISK_PCT achieve the same unlock WITHOUT being
short-specific -- a simpler, more general fix than shrinking shorts?

Variants, same lockstep-all-arms engine, same 6-segment/~2yr bar as every
other robustness pass in this line:
  SHORT_20PCT        -- the already-validated finding (short confidence x0.2, cap unchanged)
  CAP_1.5X / 2X / 3X  -- raise PORTFOLIO_MAX_CORRELATED_RISK_PCT alone, shorts untouched
  COMBINED_2X_SHORT20 -- both together (tests whether they're additive or redundant)

settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT is read fresh by
PaperTradingLedger.maybe_open() on every call (not cached at import time),
so it's safe to monkeypatch it right before each arm's maybe_open call and
restore it immediately after -- single-threaded, no race.

Usage: python -m experiments.correlation_cap_vs_short_sizing [--segments N] [--days N] [--timeframe 1h]
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

BASE_CAP = settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT
SHORT_MULT = 0.20

VARIANTS = {
    "SHORT_20PCT": {"cap_mult": 1.0, "short_mult": SHORT_MULT},
    "CAP_1.5X": {"cap_mult": 1.5, "short_mult": 1.0},
    "CAP_2X": {"cap_mult": 2.0, "short_mult": 1.0},
    "CAP_3X": {"cap_mult": 3.0, "short_mult": 1.0},
    "COMBINED_2X_SHORT20": {"cap_mult": 2.0, "short_mult": SHORT_MULT},
}


def apply_short_mult(insight: dict, mult: float) -> dict:
    if mult == 1.0 or insight.get("direction") != "short":
        return insight
    result = dict(insight)
    deriv = dict(result["derivative"])
    deriv["confidence"] = deriv.get("confidence", 0.0) * mult
    result["derivative"] = deriv
    return result


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    arm_names = ["BASELINE"] + list(VARIANTS.keys())
    ledgers = {}
    for name in arm_names:
        path = f"backtrack_logs/_cap_vs_short_{label}_{name}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

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

            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()

            settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT = BASE_CAP
            ledgers["BASELINE"].check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledgers["BASELINE"].maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledgers["BASELINE"].maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            for name, cfg in VARIANTS.items():
                ledger = ledgers[name]
                variant_insight = apply_short_mult(insight, cfg["short_mult"])
                settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT = BASE_CAP * cfg["cap_mult"]
                ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                ledger.maybe_open(sym, "spot", variant_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledger.maybe_open(sym, "derivative", variant_insight["derivative"], timeframe=timeframe, opened_at=opened_at)
                settings.PORTFOLIO_MAX_CORRELATED_RISK_PCT = BASE_CAP

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
    print(f"Base PORTFOLIO_MAX_CORRELATED_RISK_PCT = {BASE_CAP} (${BASE_CAP * settings.PAPER_TRADING_STARTING_BALANCE:.0f})")
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

    variant_names = list(VARIANTS.keys())
    wins_per_variant = {name: 0 for name in variant_names}
    total_pnl_per_variant = {name: 0.0 for name in variant_names}
    total_baseline_pnl = 0.0

    print(f"\n{'=' * 100}")
    print(f"[{settings.ASSET_TYPE.upper()}] CORRELATION CAP vs SHORT SIZING — {n_seg} segments across {days}d")
    print(f"{'=' * 100}")

    for i, seg in enumerate(segments):
        results = run_segment(seg, timeframe, f"seg{i}")
        base_pnl = results["BASELINE"]["overall"]["total_pnl_dollars"]
        base_wr = results["BASELINE"]["overall"]["win_rate_pct"]
        total_baseline_pnl += base_pnl
        start_ts = min((c[0].timestamp for c in seg.values() if c), default=None)
        end_ts = max((c[-1].timestamp for c in seg.values() if c), default=None)
        period_str = f"{start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d}" if start_ts and end_ts else "-"
        bwr_str = f"{base_wr:.1f}%" if base_wr is not None else "n/a"

        print(f"\n--- seg{i} ({period_str}) --- BASELINE: {results['BASELINE']['overall']['trades']}t / {bwr_str} WR / ${base_pnl:+.0f}")
        print(f"{'Variant':<22}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}")
        for name in variant_names:
            o = results[name]["overall"]
            wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
            beats = o["total_pnl_dollars"] > base_pnl
            if beats:
                wins_per_variant[name] += 1
            total_pnl_per_variant[name] += o["total_pnl_dollars"]
            tag = "WINS" if beats else ""
            print(f"{name:<22}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+10.2f}   {tag}")

    print(f"\n{'=' * 100}")
    print(f"[{settings.ASSET_TYPE.upper()}] SUMMARY — segments won (of {n_seg}), aggregate vs baseline ${total_baseline_pnl:+.0f}")
    print(f"{'=' * 100}")
    ranked = sorted(variant_names, key=lambda n: (wins_per_variant[n], total_pnl_per_variant[n]), reverse=True)
    print(f"{'Variant':<22}{'Segments won':<16}{'Aggregate PnL':<18}{'vs baseline':<16}")
    print("-" * 100)
    for name in ranked:
        w = wins_per_variant[name]
        agg = total_pnl_per_variant[name]
        delta = agg - total_baseline_pnl
        robust = w == n_seg
        tag = " <- ROBUST" if robust else ""
        print(f"{name:<22}{w}/{n_seg:<14}${agg:>+13.2f}   ${delta:>+13.2f}{tag}")
    print("-" * 100)
    robust = [n for n in variant_names if wins_per_variant[n] == n_seg]
    print(f"\nVariants beating baseline in ALL {n_seg} segments: {robust if robust else 'NONE'}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
