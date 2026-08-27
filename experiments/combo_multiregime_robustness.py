"""
Extends the full 11-filter multi-regime robustness pass to COMBINATIONS,
using the exact same "must win the segment on PnL, across all 6 segments
of ~2 years of real data" bar established in full_ta_robustness_validation.py
as the only credible definition of "regime-robust" (RESEARCH_NOTES.md).

No individual filter cleared that bar. This asks the natural follow-up:
does combining the LEAST-BAD individual filters — the ones that won at
least one segment somewhere — do any better? Full 2^11 combos across 11
filters would be a severe multiple-comparison fishing expedition (2,048
combinations), so this restricts to the filters that already showed
SOME signal in the full-11-filter pass, per asset type:

  Crypto finalists (won >=1 segment there): EMA50, STOCH14, ADX14  (7 combos)
  Stock finalists  (won >=1 segment there): BB20, EMA50, MACD, DONCHIAN (15 combos)

Same lockstep-all-arms-per-segment engine, same real code path, same
scratch-ledger isolation as every other script in this line.

Usage:
  python combo_multiregime_robustness.py --asset crypto --segments 6
  ASSET_TYPE=stock python combo_multiregime_robustness.py --asset stock --segments 6
"""
from __future__ import annotations

import argparse
import itertools

from backtest import YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from .full_ta_robustness_validation import apply_veto, compute_state
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

FINALISTS = {
    "crypto": ["EMA50", "STOCH14", "ADX14"],
    "stock": ["BB20", "EMA50", "MACD", "DONCHIAN"],
}


def all_combos(names: list[str]) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = []
    for r in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, r))
    return combos


def combo_label(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def apply_combo(insight: dict, candle: Candle, state: dict, t: int, combo: tuple[str, ...]) -> dict:
    result = insight
    for name in combo:
        result = apply_veto(result, candle, state, t, name)
    return result


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str, combos: list[tuple[str, ...]]) -> dict[str, dict]:
    arm_names = ["BASELINE"] + [combo_label(c) for c in combos]
    ledgers = {}
    for name in arm_names:
        path = f"backtrack_logs/_combo_robustness_{label}_{name}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    state_by_symbol = {sym: compute_state(candles) for sym, candles in symbols_candles.items()}

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

            for combo in combos:
                name = combo_label(combo)
                ledger = ledgers[name]
                combo_insight = apply_combo(insight, candle, state, t, combo)
                ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                ledger.maybe_open(sym, "spot", combo_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledger.maybe_open(sym, "derivative", combo_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    results = {}
    for name, ledger in ledgers.items():
        ledger.open_positions.clear()
        results[name] = ledger.stats()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["crypto", "stock"], required=True)
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, 730)
    days = min(args.days if args.days is not None else max_days, max_days)

    finalists = FINALISTS[args.asset]
    combos = all_combos(finalists)
    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols: {settings.WATCHLIST}")
    print(f"Finalist filters for {args.asset}: {finalists} -> {len(combos)} combinations")
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

    combo_names = [combo_label(c) for c in combos]
    wins_per_combo = {name: 0 for name in combo_names}
    total_pnl_per_combo = {name: 0.0 for name in combo_names}
    total_baseline_pnl = 0.0

    print(f"\n{'=' * 100}")
    print(f"[{args.asset.upper()}] COMBINATION MULTI-REGIME ROBUSTNESS — {len(combos)} combos, {n_seg} segments across {days}d")
    print(f"{'=' * 100}")

    for i, seg in enumerate(segments):
        results = run_segment(seg, timeframe, f"{args.asset}_seg{i}", combos)
        base_pnl = results["BASELINE"]["overall"]["total_pnl_dollars"]
        base_wr = results["BASELINE"]["overall"]["win_rate_pct"]
        total_baseline_pnl += base_pnl
        start_ts = min((c[0].timestamp for c in seg.values() if c), default=None)
        end_ts = max((c[-1].timestamp for c in seg.values() if c), default=None)
        period_str = f"{start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d}" if start_ts and end_ts else "-"

        bwr_str = f"{base_wr:.1f}%" if base_wr is not None else "n/a"
        print(f"\n--- seg{i} ({period_str}) --- BASELINE: {results['BASELINE']['overall']['trades']}t / {bwr_str} WR / ${base_pnl:+.0f}")
        print(f"{'Combo':<22}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}")
        for name in combo_names:
            o = results[name]["overall"]
            wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
            beats = o["total_pnl_dollars"] > base_pnl
            if beats:
                wins_per_combo[name] += 1
            total_pnl_per_combo[name] += o["total_pnl_dollars"]
            tag = "WINS" if beats else ""
            print(f"{name:<22}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+10.2f}   {tag}")

    print(f"\n{'=' * 100}")
    print(f"[{args.asset.upper()}] SUMMARY — segments won (of {n_seg}), aggregate vs baseline ${total_baseline_pnl:+.0f}")
    print(f"{'=' * 100}")
    ranked = sorted(combo_names, key=lambda n: (wins_per_combo[n], total_pnl_per_combo[n]), reverse=True)
    print(f"{'Combo':<22}{'Segments won':<16}{'Aggregate PnL':<18}{'vs baseline':<16}")
    print("-" * 100)
    for name in ranked:
        w = wins_per_combo[name]
        agg = total_pnl_per_combo[name]
        delta = agg - total_baseline_pnl
        robust = w == n_seg
        tag = " <- ROBUST" if robust else ""
        print(f"{name:<22}{w}/{n_seg:<14}${agg:>+13.2f}   ${delta:>+13.2f}{tag}")
    print("-" * 100)
    robust_combos = [n for n in combo_names if wins_per_combo[n] == n_seg]
    print(f"\nCombos winning ALL {n_seg} segments: {robust_combos if robust_combos else 'NONE'}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
