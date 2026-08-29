"""
Tests the research question RESEARCH_NOTES.md's regime x direction arc
explicitly "parked, not pursued": SMC direction carries a real directional
main effect (bullish consistently outperforms bearish, confirmed across a
700d and then a full 2019-2026 Binance study with block-bootstrapped
interaction tests) -- does that translate into a real edge if applied to
Path 1 entry SELECTION or SIZING, as opposed to the circuit-breaker
question that arc actually closed?

Motivated by a live-data prompt (49 real crypto paper trades): SHORT/
derivative trades showed 38% win rate vs LONG/spot's 76%, but chronological
inspection showed that gap is mostly ONE event (the 2026-08-20/25 rally --
9 of 10 post-08-18 shorts lost, vs 5/5 wins before it) -- not standalone
evidence. This is the proper multi-regime test that live sample couldn't be.

Two variants, both real code (build_entry_insight -> PaperTradingLedger,
same fee-viability filter, correlation cap, conviction sizing as baseline),
same 6-segment/~2yr lockstep engine as every other robustness pass today:

  LONG_ONLY      -- veto every SHORT recommendation outright (derivative
                    only; spot already can't short). Tests the strong form.
  SHORT_HALFSIZE -- keep SHORT trades, but halve their confidence score
                    before sizing -- reduces conviction_multiplier (see
                    paper_trading.py's maybe_open), i.e. smaller position
                    on shorts rather than banning them. Tests a softer,
                    de-risk-not-ban form.

Usage: python -m experiments.directional_sizing_robustness [--segments N] [--days N] [--timeframe 1h] [--with-gnn]
Run once per asset type (default settings.ASSET_TYPE, and again with
ASSET_TYPE=stock in the environment). --with-gnn layers in the real,
current trained checkpoint's point-in-time signal (backtest.py's pattern)
instead of gnn_signal=0.0 for every step -- use this to re-check the
SHORT-viability question against the actual current model, since every
run of this script before 2026-08-29 (including the original 38%-live-WR
motivation above) used gnn_signal=0.0 for methodological consistency with
the other robustness passes, which meant it never actually reflected
whatever GNN state (fresh, stale, or the corrupted checkpoint window --
see RESEARCH_NOTES.md) was live at any given time.
"""
from __future__ import annotations

import argparse

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SWEEP_MULTIPLIERS = [0.05, 0.1, 0.15, 0.2, 0.25]
VARIANT_NAMES = ["LONG_ONLY"] + [f"SHORT_{int(m*100)}PCT" for m in SWEEP_MULTIPLIERS]
_MULT_BY_NAME = {f"SHORT_{int(m*100)}PCT": m for m in SWEEP_MULTIPLIERS}


def apply_variant(insight: dict, name: str) -> dict:
    direction = insight.get("direction")
    if direction != "short":
        return insight
    result = dict(insight)
    if name == "LONG_ONLY":
        deriv = dict(result["derivative"])
        deriv["actionable"] = False
        result["derivative"] = deriv
        # spot already can't short (build_entry_insight already marks it
        # not-actionable for bearish direction) -- nothing to change there.
    elif name in _MULT_BY_NAME:
        deriv = dict(result["derivative"])
        deriv["confidence"] = deriv.get("confidence", 0.0) * _MULT_BY_NAME[name]
        result["derivative"] = deriv
    return result


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str, trainer=None) -> dict[str, dict]:
    """
    trainer: optional GNNTrainer (already .load()'ed). If given, every step
    recomputes the same 18-dim TOKEN_FEATURES-order vector backtest.py's
    replay_lockstep uses and does a joint predict_from_features() call
    across all symbols, so entry_insight sees a real point-in-time GNN
    signal instead of the gnn_signal=0.0 used everywhere else this session.
    """
    arm_names = ["BASELINE"] + VARIANT_NAMES
    ledgers = {}
    for name in arm_names:
        path = f"backtrack_logs/_directional_sizing_{label}_{name}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    closes_by_symbol: dict[str, list[float]] = {sym: [] for sym in symbols_candles}

    if trainer is not None:
        from train_gnn import _rsi, _macd, _bollinger, _encode_market_structure  # local: keeps the no-GNN path torch-free

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    for t in range(min_len):
        step_candles: dict[str, Candle] = {}
        smc_context_by_symbol: dict[str, dict] = {}

        for sym, candles in symbols_candles.items():
            candle = candles[t]
            step_candles[sym] = candle
            smc, fvg = smc_by_symbol[sym], fvg_by_symbol[sym]
            smc_signals = smc.update(candle)
            candle_hist = list(smc.candles)
            if len(candle_hist) >= 3:
                fvg.detect_fvg(candles=candle_hist, index=len(candle_hist) - 2, current_price=candle.close, volume=candle.volume)
            fvg.update_all(candle.close)
            ms_trend = smc_signals.get("market_structure", "unknown")
            fvg_signal = fvg.get_trading_signal(current_price=candle.close, trend=ms_trend)
            smc_context_by_symbol[sym] = {**smc_signals, **fvg_signal, "timestamp": candle.timestamp.isoformat()}
            if trainer is not None:
                closes_by_symbol[sym].append(candle.close)

        if trainer is not None:
            # Same 18-dim feature vector (TOKEN_FEATURES order) as
            # backtest.py's replay_lockstep — a joint forward pass across
            # all Token nodes since the GNN message-passes over all of them
            # at once.
            features: dict[str, list[float]] = {}
            for sym, candle in step_candles.items():
                closes = closes_by_symbol[sym]
                rsi = _rsi(closes)
                macd, macd_signal, macd_hist = _macd(closes)
                bb_upper, bb_mid, bb_lower = _bollinger(closes)
                smc_signals = smc_context_by_symbol[sym]
                tech = [candle.close, candle.volume, rsi, macd, macd_signal, macd_hist, bb_upper, bb_mid, bb_lower]
                smc_feats = [
                    smc_signals.get("smc_signal", 0.0),
                    _encode_market_structure(smc_signals.get("market_structure", "unknown")),
                    float(smc_signals.get("active_order_blocks", 0)),
                    float(smc_signals.get("active_fvgs", 0)),
                    float(smc_signals.get("liquidity_zones", 0)),
                    1.0 if smc_signals.get("in_bullish_ob") else 0.0,
                    1.0 if smc_signals.get("in_bearish_ob") else 0.0,
                    1.0 if smc_signals.get("in_bullish_fvg") else 0.0,
                    1.0 if smc_signals.get("in_bearish_fvg") else 0.0,
                ]
                features[sym] = tech + smc_feats
            gnn_signals = trainer.predict_from_features(features)
        else:
            gnn_signals = {sym: 0.0 for sym in step_candles}

        for sym, candle in step_candles.items():
            smc_context = smc_context_by_symbol[sym]
            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=gnn_signals.get(sym, 0.0))
            opened_at = candle.timestamp.isoformat()

            ledgers["BASELINE"].check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledgers["BASELINE"].maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledgers["BASELINE"].maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            for name in VARIANT_NAMES:
                ledger = ledgers[name]
                variant_insight = apply_variant(insight, name)
                ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                ledger.maybe_open(sym, "spot", variant_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledger.maybe_open(sym, "derivative", variant_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

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
    parser.add_argument("--with-gnn", action="store_true",
                         help="Layer in the real, current trained GNN checkpoint's point-in-time signal "
                              "(same pattern as backtest.py) instead of gnn_signal=0.0 for every step.")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, 730)
    days = min(args.days if args.days is not None else max_days, max_days)

    trainer = None
    if args.with_gnn:
        from gnn.trainer import GNNTrainer
        trainer = GNNTrainer()
        if not trainer.load():
            print(f"No trained GNN checkpoint found for ASSET_TYPE={settings.ASSET_TYPE} — "
                  "run train_gnn.py first. Aborting --with-gnn run.")
            return
        print(f"Loaded GNN checkpoint for ASSET_TYPE={settings.ASSET_TYPE} — real point-in-time signal active.\n")

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
    print(f"[{settings.ASSET_TYPE.upper()}] DIRECTIONAL SIZING ROBUSTNESS — {n_seg} segments across {days}d")
    print(f"{'=' * 100}")

    wins_per_variant = {name: 0 for name in VARIANT_NAMES}
    total_pnl_per_variant = {name: 0.0 for name in VARIANT_NAMES}
    total_baseline_pnl = 0.0

    for i, seg in enumerate(segments):
        results = run_segment(seg, timeframe, f"seg{i}", trainer=trainer)
        base_pnl = results["BASELINE"]["overall"]["total_pnl_dollars"]
        base_wr = results["BASELINE"]["overall"]["win_rate_pct"]
        total_baseline_pnl += base_pnl
        start_ts = min((c[0].timestamp for c in seg.values() if c), default=None)
        end_ts = max((c[-1].timestamp for c in seg.values() if c), default=None)
        period_str = f"{start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d}" if start_ts and end_ts else "-"
        bwr_str = f"{base_wr:.1f}%" if base_wr is not None else "n/a"

        print(f"\n--- seg{i} ({period_str}) --- BASELINE: {results['BASELINE']['overall']['trades']}t / {bwr_str} WR / ${base_pnl:+.0f}")
        print(f"{'Variant':<16}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}")
        for name in VARIANT_NAMES:
            o = results[name]["overall"]
            wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
            beats = o["total_pnl_dollars"] > base_pnl
            if beats:
                wins_per_variant[name] += 1
            total_pnl_per_variant[name] += o["total_pnl_dollars"]
            tag = "WINS" if beats else ""
            print(f"{name:<16}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+10.2f}   {tag}")

    print(f"\n{'=' * 100}")
    print(f"[{settings.ASSET_TYPE.upper()}] SUMMARY — segments won (of {n_seg}), aggregate vs baseline ${total_baseline_pnl:+.0f}")
    print(f"{'=' * 100}")
    ranked = sorted(VARIANT_NAMES, key=lambda n: (wins_per_variant[n], total_pnl_per_variant[n]), reverse=True)
    print(f"{'Variant':<16}{'Segments won':<16}{'Aggregate PnL':<18}{'vs baseline':<16}")
    print("-" * 100)
    for name in ranked:
        w = wins_per_variant[name]
        agg = total_pnl_per_variant[name]
        delta = agg - total_baseline_pnl
        robust = w == n_seg
        tag = " <- ROBUST" if robust else ""
        print(f"{name:<16}{w}/{n_seg:<14}${agg:>+13.2f}   ${delta:>+13.2f}{tag}")
    print("-" * 100)
    robust = [n for n in VARIANT_NAMES if wins_per_variant[n] == n_seg]
    print(f"\nVariants beating baseline in ALL {n_seg} segments: {robust if robust else 'NONE'}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
