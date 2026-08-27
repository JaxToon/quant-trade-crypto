"""
Extends ema50_robustness_validation.py's multi-regime test to EVERY
technical-analysis filter tested so far in this project — not just EMA50.

EMA50 looked like a winner on a 90-day window and then failed an
identical multi-regime check across 730 days (2 of 6 segments won, every
EMA period 20-200 lost to no-filter over the long run — see
RESEARCH_NOTES.md's "EMA50 robustness validation" entry). The obvious
next question: does ANY of the other 10 filters already screened in this
project's research hold up any better once tested the same way, or does
the same single-regime-artifact failure mode apply across the board?

Filters (11 total — the full set from ta_full_survey_experiment.py plus
VWAP from vwap_confluence_ab_experiment.py, i.e. every TA method surveyed
from the web + standard practice so far in this research line):
  EMA50, MACD, DONCHIAN, RSI14, STOCH14, CCI20, BB20, ADX14, VOL20,
  OBV20, VWAP (daily-anchored)

Same real-code, isolated-scratch-ledger, no-lookahead discipline as every
other AB script here. Runs baseline + all 11 filters in ONE lockstep pass
per segment (12 ledgers, identical candle sequence) so results are
directly comparable.

Usage: python full_ta_robustness_validation.py [--segments N] [--days N] [--timeframe 1h]
Run once per asset type (default settings.ASSET_TYPE, and again with
ASSET_TYPE=stock in the environment).
"""
from __future__ import annotations

import argparse
from datetime import datetime

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
FILTER_NAMES = ["EMA50", "MACD", "DONCHIAN", "RSI14", "STOCH14", "CCI20", "BB20", "ADX14", "VOL20", "OBV20", "VWAP"]


def compute_daily_vwap(candles: list[Candle]) -> list[float]:
    vwaps: list[float] = []
    cum_pv = cum_vol = 0.0
    current_day = None
    for c in candles:
        day = c.timestamp.date()
        if day != current_day:
            current_day, cum_pv, cum_vol = day, 0.0, 0.0
        typical = (c.high + c.low + c.close) / 3.0
        vol = c.volume if c.volume and c.volume > 0 else 1.0
        cum_pv += typical * vol
        cum_vol += vol
        vwaps.append(cum_pv / cum_vol if cum_vol > 0 else c.close)
    return vwaps


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
        "vwap": compute_daily_vwap(candles),
    }


def filter_passes(name: str, direction: str, candle: Candle, state: dict, t: int) -> bool:
    if name == "EMA50":
        v = state["ema50"][t]
        return candle.close >= v if direction == "long" else candle.close <= v
    if name == "MACD":
        h = state["macd_hist"][t]
        return h >= 0 if direction == "long" else h <= 0
    if name == "DONCHIAN":
        v = state["donchian_mid"][t]
        return candle.close >= v if direction == "long" else candle.close <= v
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
        return state["adx14"][t] >= 25
    if name == "VOL20":
        return candle.volume >= state["vol20"][t]
    if name == "OBV20":
        obv_up = state["obv"][t] >= state["obv_ema20"][t]
        return obv_up if direction == "long" else not obv_up
    if name == "VWAP":
        v = state["vwap"][t]
        return candle.close >= v if direction == "long" else candle.close <= v
    raise ValueError(name)


def apply_veto(insight: dict, candle: Candle, state: dict, t: int, name: str) -> dict:
    direction = insight.get("direction")
    if direction not in ("long", "short") or filter_passes(name, direction, candle, state, t):
        return insight
    vetoed = dict(insight)
    for leg in ("spot", "derivative"):
        if vetoed[leg]["actionable"]:
            vetoed[leg] = dict(vetoed[leg])
            vetoed[leg]["actionable"] = False
    return vetoed


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    arm_names = ["BASELINE"] + FILTER_NAMES
    ledgers = {}
    for name in arm_names:
        path = f"backtrack_logs/_full_ta_robustness_{label}_{name}.json"
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

            for name in FILTER_NAMES:
                ledger = ledgers[name]
                f_insight = apply_veto(insight, candle, state, t, name)
                ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                ledger.maybe_open(sym, "spot", f_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledger.maybe_open(sym, "derivative", f_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

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
    print(f"[{settings.ASSET_TYPE.upper()}] MULTI-REGIME ROBUSTNESS — all {len(FILTER_NAMES)} filters, {n_seg} segments across {days}d")
    print(f"{'=' * 100}")

    wins_per_filter = {name: 0 for name in FILTER_NAMES}
    total_pnl_per_filter = {name: 0.0 for name in FILTER_NAMES}
    total_baseline_pnl = 0.0

    for i, seg in enumerate(segments):
        results = run_segment(seg, timeframe, f"seg{i}")
        base_pnl = results["BASELINE"]["overall"]["total_pnl_dollars"]
        base_wr = results["BASELINE"]["overall"]["win_rate_pct"]
        total_baseline_pnl += base_pnl
        start_ts = min((c[0].timestamp for c in seg.values() if c), default=None)
        end_ts = max((c[-1].timestamp for c in seg.values() if c), default=None)
        period_str = f"{start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d}" if start_ts and end_ts else "-"

        print(f"\n--- seg{i} ({period_str}) --- BASELINE: {results['BASELINE']['overall']['trades']}t / "
              f"{base_wr:.1f}% WR / ${base_pnl:+.0f}" if base_wr is not None else f"\n--- seg{i} ({period_str}) --- BASELINE: n/a")
        print(f"{'Filter':<10}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}{'vs baseline':<14}")
        for name in FILTER_NAMES:
            o = results[name]["overall"]
            wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
            beats = o["total_pnl_dollars"] > base_pnl
            if beats:
                wins_per_filter[name] += 1
            total_pnl_per_filter[name] += o["total_pnl_dollars"]
            tag = "WINS" if beats else ""
            print(f"{name:<10}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+10.2f}   {tag}")

    print(f"\n{'=' * 100}")
    print(f"[{settings.ASSET_TYPE.upper()}] SUMMARY — segments won (out of {n_seg}) and aggregate PnL vs baseline's ${total_baseline_pnl:+.0f}")
    print(f"{'=' * 100}")
    print(f"{'Filter':<10}{'Segments won':<16}{'Aggregate PnL':<18}{'vs baseline aggregate':<20}")
    print("-" * 100)
    ranked = sorted(FILTER_NAMES, key=lambda n: wins_per_filter[n], reverse=True)
    for name in ranked:
        w = wins_per_filter[name]
        agg = total_pnl_per_filter[name]
        delta = agg - total_baseline_pnl
        robust = w == n_seg
        tag = " <- ROBUST (wins every segment)" if robust else (f" ({w}/{n_seg})" if w > n_seg // 2 else "")
        print(f"{name:<10}{w}/{n_seg:<14}${agg:>+13.2f}   ${delta:>+13.2f}{tag}")
    print("-" * 100)
    any_robust = [n for n in FILTER_NAMES if wins_per_filter[n] == n_seg]
    print(f"\nFilters winning ALL {n_seg} segments (the only credible bar for 'regime-robust'): {any_robust if any_robust else 'NONE'}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
