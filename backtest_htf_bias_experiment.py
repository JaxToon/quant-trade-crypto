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
("bullish" / "bearish" / "unknown") for the same symbol. The soft rule is
used throughout (blocked only on active disagreement, "unknown" passes
through) -- an earlier version of this script also tested a hard variant
(blocked on "unknown" too) and found it produced IDENTICAL results to
soft in both windows tested, because "unknown" 1h structure only ever
occurs during the initial SMC_LOOKBACK_CANDLES warmup, before any
lower-timeframe entry signal happened to fire in that data. Per review:
"filter logic = validated candidate, hard vs soft = unresolved
implementation detail" -- not tuning that choice further until the
robustness question below is settled.

REVISION (walk-forward + ablation): the first version of this experiment
compared a 30-day window against a 60-day window and called them two
independent confirmations -- they weren't. 30 days ending "now" is fully
CONTAINED inside 60 days ending "now"; that's the same data measured
twice; it says nothing about whether the effect holds on unseen dates.
Fixed here two ways:
  1. WALK-FORWARD: fetch the full historical window once, split it into
     N genuinely non-overlapping, chronologically-ordered segments by
     calendar time (not by trailing "now"), and replay each segment with
     COLD-STARTED analyzer state (no carryover between segments) -- N
     independent samples, not one window nested in another.
  2. ABLATION: is the edge from hierarchical alignment specifically, or
     is 1h's own market_structure just a good filter regardless of what
     it's filtering? Runs 4 variants: baseline (no filter anywhere),
     htf_5m_only (filter applied to 5m entries only), htf_15m_only
     (15m entries only), htf_both (both -- the original design). 1h's
     own trades are never filtered in any variant (no HTF above it in
     this 3-layer scope) -- reported per-timeframe per-variant so it's
     visible whether 1h's own numbers stay flat (they should, as a
     sanity check) and whether the improvement is concentrated in 5m,
     15m, or both.

Acceptance framing per review, not just "does total return go up":
delta-expectancy > 0, delta-max_drawdown < 0, no systematic win-rate collapse, no
excessively aggressive trade reduction, consistent DIRECTION across
segments (e.g. +15%/+8%/+22%/+5%/-3% is more convincing than one +180%)
-- not a single magnitude number.

Does not modify backtrack/paper_trading.py, strategy/fvg_analyzer.py,
strategy/smart_money.py, strategy/entry_insight.py, or backtest.py. Uses
its own scratch ledger files, one per (variant, segment) combination,
never written to disk (ledger.save() is never called). Fully isolated
from any live main.py process — pure historical replay, no live data
stream, no shared state, no interference with whatever main.py is
currently doing.

Usage: python backtest_htf_bias_experiment.py [--days N] [--segments N]
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

VARIANTS = ["baseline", "htf_5m_only", "htf_15m_only", "htf_both"]
HTF = "1h"  # the layer whose structure governs the others in this experiment
FILTERED_TF_BY_VARIANT = {
    "baseline": set(),
    "htf_5m_only": {"5m"},
    "htf_15m_only": {"15m"},
    "htf_both": {"5m", "15m"},
}


def _htf_allows(direction: str, htf_structure: str) -> bool:
    """Soft rule only -- see module docstring for why hard isn't tested here."""
    if direction not in ("long", "short"):
        return True  # neutral/no-direction insight -- nothing for this filter to block
    opposes = (direction == "long" and htf_structure == "bearish") or (direction == "short" and htf_structure == "bullish")
    return not opposes


def _fetch_all(symbols: list[str], days: int) -> dict[tuple[str, str], list]:
    timeframes = settings.BINANCE_SMC_TIMEFRAMES
    candles_by_symbol_tf: dict[tuple[str, str], list] = {}
    for symbol in symbols:
        for tf in timeframes:
            candles = fetch_candles(symbol, days, tf)
            if candles:
                candles_by_symbol_tf[(symbol, tf)] = candles
    return candles_by_symbol_tf


def _split_into_segments(candles_by_symbol_tf: dict, n_segments: int) -> list[dict]:
    """Splits by CALENDAR TIME (not candle index -- 5m/15m/1h have very
    different candle counts for the same days), so segment i covers the
    same date range across every (symbol, timeframe). Returns a list of
    n_segments dicts, each shaped like candles_by_symbol_tf but sliced."""
    all_ts = [c.timestamp for candles in candles_by_symbol_tf.values() for c in candles]
    start, end = min(all_ts), max(all_ts)
    span = (end - start) / n_segments
    bounds = [start + span * i for i in range(n_segments + 1)]

    segments: list[dict] = [dict() for _ in range(n_segments)]
    for key, candles in candles_by_symbol_tf.items():
        for i in range(n_segments):
            lo, hi = bounds[i], bounds[i + 1]
            seg_candles = [c for c in candles if lo <= c.timestamp < hi] if i < n_segments - 1 else [c for c in candles if lo <= c.timestamp <= hi]
            if seg_candles:
                segments[i][key] = seg_candles
    return segments


def replay_variant(candles_by_symbol_tf: dict, variant: str) -> PaperTradingLedger:
    """Cold-started: fresh analyzer state, no carryover from any other
    segment or variant -- each call is a fully independent replay."""
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
    latest_structure: dict[tuple[str, str], str] = {}
    filtered_tfs = FILTERED_TF_BY_VARIANT[variant]

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

        if tf not in filtered_tfs:
            allowed = True
        else:
            htf_structure = latest_structure.get((symbol, HTF), "unknown")
            allowed = _htf_allows(insight.get("direction", "neutral"), htf_structure)

        if allowed:
            ledger.maybe_open(symbol, "spot", insight["spot"], timeframe=tf, opened_at=opened_at)
            ledger.maybe_open(symbol, "derivative", insight["derivative"], timeframe=tf, opened_at=opened_at)

    ledger.open_positions.clear()  # unresolved at segment end -- not counted
    return ledger


def _score(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    equity = 10_000.0
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
        "expectancy_dollars": total_pnl / n,
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
    }


def compute_stats(ledger: PaperTradingLedger) -> dict:
    trades = ledger.closed_trades
    overall = _score(trades)
    for tf in settings.BINANCE_SMC_TIMEFRAMES:
        overall[tf] = _score([t for t in trades if t["timeframe"] == tf])
    return overall


def _print_variant_row(name: str, s: dict) -> None:
    if s["trades"] == 0:
        print(f"  {name:<14}{'0':<8}(no trades)")
        return
    print(f"  {name:<14}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['expectancy_dollars']:<+13.2f}{s['max_drawdown_pct']:<9.2f}{s['profit_factor']:<13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HTF bias filter: walk-forward non-overlapping segments + per-timeframe ablation (Path 1 only)")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--segments", type=int, default=3)
    args = parser.parse_args()
    days = min(args.days if args.days is not None else YF_MAX_DAYS["5m"], YF_MAX_DAYS["5m"])
    n_segments = max(2, args.segments)

    tickers = CRYPTO_TICKERS if settings.ASSET_TYPE == "crypto" else {s: [s] for s in settings.STOCK_WATCHLIST}
    symbols = list(tickers.keys())
    print(f"ASSET_TYPE={settings.ASSET_TYPE} — HTF bias walk-forward over {days}d in {n_segments} non-overlapping segments, symbols: {symbols}")
    print(f"Timeframes: {settings.BINANCE_SMC_TIMEFRAMES} (HTF={HTF}, soft rule, cold-started per segment)\n")

    full = _fetch_all(symbols, days)
    segments = _split_into_segments(full, n_segments)
    for i, seg in enumerate(segments):
        n_candles = sum(len(c) for c in seg.values())
        print(f"  segment {i+1}/{n_segments}: {n_candles} total candles across {len(seg)} (symbol, timeframe) streams")
    print()

    header = f"  {'Variant':<14}{'Trades':<8}{'WinRate%':<10}{'Expectancy$':<13}{'MaxDD%':<9}{'ProfitFactor':<13}"

    all_results: list[dict] = []  # one entry per (segment, variant)
    for seg_idx, seg in enumerate(segments):
        print(f"{'=' * 70}\nSEGMENT {seg_idx + 1}/{n_segments}\n{'=' * 70}")
        print(header)
        seg_results = {}
        for variant in VARIANTS:
            ledger = replay_variant(seg, variant)
            stats = compute_stats(ledger)
            seg_results[variant] = stats
            _print_variant_row(variant, stats)
        all_results.append(seg_results)

        baseline = seg_results["baseline"]
        if baseline["trades"] > 0:
            print("\n  vs baseline (overall):")
            for variant in ("htf_5m_only", "htf_15m_only", "htf_both"):
                s = seg_results[variant]
                if s["trades"] == 0:
                    print(f"    {variant}: no trades survived")
                    continue
                d_exp = s["expectancy_dollars"] - baseline["expectancy_dollars"]
                d_dd = s["max_drawdown_pct"] - baseline["max_drawdown_pct"]
                print(
                    f"    {variant}: trades {baseline['trades']}->{s['trades']}, "
                    f"expectancy {baseline['expectancy_dollars']:+.2f}->{s['expectancy_dollars']:+.2f} ({d_exp:+.2f}), "
                    f"drawdown {baseline['max_drawdown_pct']:.2f}->{s['max_drawdown_pct']:.2f} ({d_dd:+.2f})"
                )
            print("\n  per-timeframe breakdown (expectancy $/trade, baseline -> htf_both):")
            for tf in settings.BINANCE_SMC_TIMEFRAMES:
                b_tf = baseline[tf]
                f_tf = seg_results["htf_both"][tf]
                b_str = f"{b_tf['expectancy_dollars']:+.2f} ({b_tf['trades']}t)" if b_tf["trades"] else "no trades"
                f_str = f"{f_tf['expectancy_dollars']:+.2f} ({f_tf['trades']}t)" if f_tf["trades"] else "no trades"
                flag = " <- filtered in htf_both" if tf in FILTERED_TF_BY_VARIANT["htf_both"] else " <- never filtered (sanity check: should ~match baseline)"
                print(f"    {tf}: {b_str} -> {f_str}{flag}")
        print()

    print(f"{'=' * 70}\nACCEPTANCE SUMMARY ACROSS ALL {n_segments} SEGMENTS\n{'=' * 70}")
    print("Per review: consistent DIRECTION across independent segments matters more than any single magnitude.")
    for variant in ("htf_5m_only", "htf_15m_only", "htf_both"):
        deltas_exp = []
        deltas_dd = []
        for seg_results in all_results:
            baseline = seg_results["baseline"]
            s = seg_results[variant]
            if baseline["trades"] > 0 and s["trades"] > 0:
                deltas_exp.append(s["expectancy_dollars"] - baseline["expectancy_dollars"])
                deltas_dd.append(s["max_drawdown_pct"] - baseline["max_drawdown_pct"])
        if not deltas_exp:
            print(f"  {variant}: insufficient data across segments to judge")
            continue
        positive_exp = sum(1 for d in deltas_exp if d > 0)
        improved_dd = sum(1 for d in deltas_dd if d < 0)
        print(
            f"  {variant}: delta-expectancy positive in {positive_exp}/{len(deltas_exp)} segments "
            f"({[f'{d:+.2f}' for d in deltas_exp]}), delta-drawdown improved in {improved_dd}/{len(deltas_dd)} segments"
        )

    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("paper_trading.py, entry_insight.py, or any live strategy code regardless of which variant wins.")


if __name__ == "__main__":
    main()
