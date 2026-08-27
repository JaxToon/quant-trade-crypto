"""
Full technical-analysis confluence survey, two phases, per user request to
test "semua model technical analysis yang ada" (all TA models) rather than
just VWAP/EMA/RSI/VOL.

PHASE 1 (--phase 1): screen each candidate filter INDIVIDUALLY (main effect
only) against the SMC/FVG baseline, in-sample + out-of-sample, separately
for crypto and stock. This is deliberately NOT a full combinatorial grid —
with 10 candidate filters a full 2^10 combo matrix is a severe multiple-
comparison problem (near-guaranteed to find a "winning" combo by chance
alone) and would contradict this project's own established discipline
(pre-register, don't fish — see feedback_quant_research_discipline).
Screening main effects first, then only combining the survivors, mirrors
exactly how the regime x direction research arc in RESEARCH_NOTES.md was
run (main effect first, interaction only on what survived).

PHASE 2 (--phase 2 --filters A,B,C): full combination matrix restricted to
whichever filters phase 1 showed a CONSISTENT (both in-sample AND
out-of-sample) individual edge on, same lockstep-all-arms-at-once engine as
confluence_matrix_ab_experiment.py.

Candidates (10, one per major TA category, chosen from web-survey +
standard practice; VWAP already tested separately and rejected, not
repeated):
  Trend:       EMA50, MACD (histogram sign), DONCHIAN (channel midpoint)
  Momentum:    RSI14 (extreme-avoid), STOCH14 (extreme-avoid), CCI20 (extreme-avoid)
  Volatility:  BB20 (Bollinger extreme-avoid), ADX14 (trend-strength gate, direction-agnostic)
  Volume:      VOL20 (above-average confirm), OBV20 (on-balance-volume trend confirm)

Every filter is a pure post-hoc veto on build_entry_insight()'s output —
zero live-pipeline files touched, every arm on its own scratch ledger
(backtrack_logs/_ta_survey_*.json), fully isolated.

Usage:
  python ta_full_survey_experiment.py --phase 1 --timeframe 1h --days 90
  python ta_full_survey_experiment.py --phase 2 --filters EMA50,MACD --timeframe 1h --days 90
Run each phase once with default ASSET_TYPE and once with
ASETTYPE=stock in the environment (matches start_system.sh's convention).
"""
from __future__ import annotations

import argparse
import itertools

from backtest import CRYPTO_TICKERS, STOCK_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer
from .ta_indicators import (
    adx_series, bollinger_bands_series, cci_series, donchian_mid_series,
    macd_hist_series, obv_series, rsi_series, stochastic_k_series, volume_avg_series,
)
from train_gnn import _ema

SYMBOL_TICKERS = STOCK_TICKERS if settings.ASSET_TYPE == "stock" else CRYPTO_TICKERS

ALL_FILTER_NAMES = ["EMA50", "MACD", "DONCHIAN", "RSI14", "STOCH14", "CCI20", "BB20", "ADX14", "VOL20", "OBV20"]


def compute_state(candles: list[Candle]) -> dict:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]
    bb_upper, bb_lower = bollinger_bands_series(closes, 20, 2.0)
    obv = obv_series(closes, volumes)
    return {
        "ema50": _ema(50, closes),
        "macd_hist": macd_hist_series(closes),
        "donchian_mid": donchian_mid_series(highs, lows, 20),
        "rsi14": rsi_series(closes, 14),
        "stoch14": stochastic_k_series(highs, lows, closes, 14),
        "cci20": cci_series(highs, lows, closes, 20),
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "adx14": adx_series(highs, lows, closes, 14),
        "vol20": volume_avg_series(volumes, 20),
        "obv": obv,
        "obv_ema20": _ema(20, obv) if len(obv) >= 2 else obv,
    }


def filter_passes(name: str, direction: str, candle: Candle, state: dict, t: int) -> bool:
    if name == "EMA50":
        ema = state["ema50"][t]
        return candle.close >= ema if direction == "long" else candle.close <= ema
    if name == "MACD":
        h = state["macd_hist"][t]
        return h >= 0 if direction == "long" else h <= 0
    if name == "DONCHIAN":
        mid = state["donchian_mid"][t]
        return candle.close >= mid if direction == "long" else candle.close <= mid
    if name == "RSI14":
        r = state["rsi14"][t]
        return r <= 70 if direction == "long" else r >= 30
    if name == "STOCH14":
        k = state["stoch14"][t]
        return k <= 80 if direction == "long" else k >= 20
    if name == "CCI20":
        c = state["cci20"][t]
        return c <= 100 if direction == "long" else c >= -100
    if name == "BB20":
        price = candle.close
        return price <= state["bb_upper"][t] if direction == "long" else price >= state["bb_lower"][t]
    if name == "ADX14":
        return state["adx14"][t] >= 25  # trend-strength gate, same for both directions
    if name == "VOL20":
        return candle.volume >= state["vol20"][t]
    if name == "OBV20":
        obv_up = state["obv"][t] >= state["obv_ema20"][t]
        return obv_up if direction == "long" else not obv_up
    raise ValueError(name)


def apply_combo_veto(insight: dict, candle: Candle, state: dict, t: int, combo: tuple[str, ...]) -> dict:
    direction = insight.get("direction")
    if direction not in ("long", "short") or not combo:
        return insight
    if all(filter_passes(name, direction, candle, state, t) for name in combo):
        return insight
    vetoed = dict(insight)
    for leg in ("spot", "derivative"):
        if vetoed[leg]["actionable"]:
            vetoed[leg] = dict(vetoed[leg])
            vetoed[leg]["actionable"] = False
            vetoed[leg]["note"] = vetoed[leg].get("note", "") + f" [veto: {'+'.join(combo)}]"
    return vetoed


def run_arms(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str, combos: list[tuple[str, ...]]) -> dict[str, dict]:
    ledgers = {}
    for combo in combos:
        name = "BASELINE" if not combo else "+".join(combo)
        path = f"backtrack_logs/_ta_survey_{label}_{name}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[combo] = ledger

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

            for combo, ledger in ledgers.items():
                ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                combo_insight = apply_combo_veto(insight, candle, state, t, combo)
                ledger.maybe_open(sym, "spot", combo_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledger.maybe_open(sym, "derivative", combo_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    results = {}
    for combo, ledger in ledgers.items():
        ledger.open_positions.clear()
        name = "BASELINE" if not combo else "+".join(combo)
        results[name] = ledger.stats()
    return results


def fetch_all(days: int, timeframe: str) -> dict[str, list[Candle]]:
    print(f"ASSET_TYPE={settings.ASSET_TYPE} — {len(settings.WATCHLIST)} symbols: {settings.WATCHLIST}")
    print(f"Fetching {days}d of {timeframe} candles (via yfinance)...\n")
    all_candles: dict[str, list[Candle]] = {}
    for symbol in settings.WATCHLIST:
        candles = fetch_candles(symbol, days, timeframe)
        if not candles:
            print(f"  {symbol}: no historical data, skipping")
            continue
        all_candles[symbol] = candles
        print(f"  {symbol}: {len(candles)} candles ({candles[0].timestamp} -> {candles[-1].timestamp})")
    return all_candles


def split_halves(all_candles: dict[str, list[Candle]]) -> tuple[dict, dict]:
    n = min(len(c) for c in all_candles.values())
    mid = n // 2
    return (
        {sym: candles[:mid] for sym, candles in all_candles.items()},
        {sym: candles[mid:] for sym, candles in all_candles.items()},
    )


def print_screen(title: str, results: dict[str, dict]) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")
    print(f"{'Filter':<12}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}{'Return %':<10}")
    print("-" * 78)
    for name, s in results.items():
        o = s["overall"]
        wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
        marker = " <- baseline" if name == "BASELINE" else ""
        print(f"{name:<12}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+10.2f}   {s['total_return_pct']:>+7.2f}%{marker}")


def phase1(days: int, timeframe: str) -> None:
    all_candles = fetch_all(days, timeframe)
    if not all_candles:
        print("No data fetched — aborting.")
        return
    in_sample, out_sample = split_halves(all_candles)

    combos = [()] + [(f,) for f in ALL_FILTER_NAMES]
    results_is = run_arms(in_sample, timeframe, f"{settings.ASSET_TYPE}_p1is", combos)
    results_oos = run_arms(out_sample, timeframe, f"{settings.ASSET_TYPE}_p1oos", combos)

    print_screen(f"[{settings.ASSET_TYPE.upper()}] PHASE 1 IN-SAMPLE [{timeframe}] (~{days // 2}d)", results_is)
    print_screen(f"[{settings.ASSET_TYPE.upper()}] PHASE 1 OUT-OF-SAMPLE [{timeframe}] (~{days - days // 2}d)", results_oos)

    base_is_pnl = results_is["BASELINE"]["overall"]["total_pnl_dollars"]
    base_oos_pnl = results_oos["BASELINE"]["overall"]["total_pnl_dollars"]
    base_is_wr = results_is["BASELINE"]["overall"]["win_rate_pct"] or 0
    base_oos_wr = results_oos["BASELINE"]["overall"]["win_rate_pct"] or 0

    print(f"\n{'=' * 78}")
    print(f"[{settings.ASSET_TYPE.upper()}] PHASE 1 SCREEN RESULT — survivors beat baseline PnL in BOTH halves")
    print(f"{'=' * 78}")
    survivors = []
    for name in ALL_FILTER_NAMES:
        is_pnl = results_is[name]["overall"]["total_pnl_dollars"]
        oos_pnl = results_oos[name]["overall"]["total_pnl_dollars"]
        is_wr = results_is[name]["overall"]["win_rate_pct"] or 0
        oos_wr = results_oos[name]["overall"]["win_rate_pct"] or 0
        beats_pnl_both = is_pnl > base_is_pnl and oos_pnl > base_oos_pnl
        beats_wr_both = is_wr > base_is_wr and oos_wr > base_oos_wr
        status = "SURVIVOR (pnl+WR both halves)" if (beats_pnl_both and beats_wr_both) else ("pnl only" if beats_pnl_both else "no")
        if beats_pnl_both and beats_wr_both:
            survivors.append(name)
        print(f"  {name:<10} IS ${is_pnl:+9.2f} (base ${base_is_pnl:+.2f})  OOS ${oos_pnl:+9.2f} (base ${base_oos_pnl:+.2f})  -> {status}")
    print(f"\nSurvivors to carry into Phase 2: {survivors if survivors else 'NONE'}")
    print(f"{'=' * 78}")


def phase2(days: int, timeframe: str, filters: list[str]) -> None:
    all_candles = fetch_all(days, timeframe)
    if not all_candles:
        print("No data fetched — aborting.")
        return
    in_sample, out_sample = split_halves(all_candles)

    combos: list[tuple[str, ...]] = [()]
    for r in range(1, len(filters) + 1):
        combos.extend(itertools.combinations(filters, r))

    results_is = run_arms(in_sample, timeframe, f"{settings.ASSET_TYPE}_p2is", combos)
    results_oos = run_arms(out_sample, timeframe, f"{settings.ASSET_TYPE}_p2oos", combos)

    print_screen(f"[{settings.ASSET_TYPE.upper()}] PHASE 2 IN-SAMPLE [{timeframe}] (~{days // 2}d)", results_is)
    print_screen(f"[{settings.ASSET_TYPE.upper()}] PHASE 2 OUT-OF-SAMPLE [{timeframe}] (~{days - days // 2}d)", results_oos)

    base_is_pnl = results_is["BASELINE"]["overall"]["total_pnl_dollars"]
    base_oos_pnl = results_oos["BASELINE"]["overall"]["total_pnl_dollars"]
    print(f"\n{'=' * 78}")
    print(f"[{settings.ASSET_TYPE.upper()}] PHASE 2 — combos beating baseline PnL in BOTH halves")
    print(f"{'=' * 78}")
    any_consistent = False
    for name in results_is:
        if name == "BASELINE":
            continue
        is_pnl = results_is[name]["overall"]["total_pnl_dollars"]
        oos_pnl = results_oos[name]["overall"]["total_pnl_dollars"]
        if is_pnl > base_is_pnl and oos_pnl > base_oos_pnl:
            any_consistent = True
            print(f"  {name}: IS ${is_pnl:+.2f} (base ${base_is_pnl:+.2f})   OOS ${oos_pnl:+.2f} (base ${base_oos_pnl:+.2f})")
    if not any_consistent:
        print("  None.")
    print(f"{'=' * 78}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--filters", type=str, default="", help="Comma-separated filter names, phase 2 only")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--timeframe", choices=settings.BINANCE_SMC_TIMEFRAMES, default="1h")
    args = parser.parse_args()

    max_days = YF_MAX_DAYS.get(args.timeframe, 60)
    days = min(args.days, max_days)

    if args.phase == 1:
        phase1(days, args.timeframe)
    else:
        filters = [f.strip() for f in args.filters.split(",") if f.strip()]
        if not filters:
            print("Phase 2 requires --filters (comma-separated survivor names from phase 1).")
            return
        phase2(days, args.timeframe, filters)


if __name__ == "__main__":
    main()
