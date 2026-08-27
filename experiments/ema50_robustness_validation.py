"""
Robustness validation for the EMA(50) crypto entry filter before any live
deployment — closes four specific gaps the original 90-day, single-parameter
test could not rule out:

1. REGIME DEPENDENCE: the original 90d window overlaps almost entirely with
   a real +25% BTC rally (2026-08-20/25, see RESEARCH_NOTES.md's "Persistent
   turbulent regime" entry). An EMA trend filter trivially "wins" during any
   window dominated by one strong trend — that's not evidence it works in
   general. Fetches the longest available yfinance 1h history and splits it
   into N>=6 chronological segments so the edge has to hold across MULTIPLE
   distinct regimes, not one lucky stretch.

2. PARAMETER OVERFITTING: only period=50 was ever tested. Sweeps EMA period
   over [20,30,40,50,60,80,100,150,200] on the same long window — a real
   effect should hold across a broad neighborhood of periods, not spike
   narrowly at exactly 50.

3. CONCENTRATION RISK: an aggregate win rate can hide a filter whose entire
   edge comes from one or two symbols (e.g. just BTC riding the rally).
   Reports per-symbol trade count and PnL for baseline vs EMA50.

4. DRAWDOWN + STATISTICAL SIGNIFICANCE: aggregate PnL alone doesn't show
   risk profile. Computes peak-to-trough max drawdown on each arm's equity
   curve, and a two-proportion z-test on the win-rate delta so "68% vs 54%"
   can be judged against its actual sample size instead of eyeballed.

Same isolation discipline as every other AB script here: real yfinance
data, real SmartMoneyAnalyzer/FairValueGapAnalyzer/build_entry_insight/
PaperTradingLedger code, one lockstep pass per segment so every arm sees
identical candles, scratch ledgers only, zero live files touched.

Usage: python ema50_robustness_validation.py [--segments N] [--days N]
"""
from __future__ import annotations

import argparse
import math

from backtest import CRYPTO_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer
from train_gnn import _ema

EMA_PERIODS_TO_SWEEP = [20, 30, 40, 50, 60, 80, 100, 150, 200]


def compute_ema_states(candles: list[Candle], periods: list[int]) -> dict[int, list[float]]:
    closes = [c.close for c in candles]
    return {p: _ema(p, closes) for p in periods}


def ema_passes(direction: str, price: float, ema_val: float) -> bool:
    return price >= ema_val if direction == "long" else price <= ema_val


def apply_ema_veto(insight: dict, price: float, ema_val: float) -> dict:
    direction = insight.get("direction")
    if direction not in ("long", "short") or ema_passes(direction, price, ema_val):
        return insight
    vetoed = dict(insight)
    for leg in ("spot", "derivative"):
        if vetoed[leg]["actionable"]:
            vetoed[leg] = dict(vetoed[leg])
            vetoed[leg]["actionable"] = False
    return vetoed


def run_segment(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str, ema_periods: list[int]) -> dict[str, PaperTradingLedger]:
    """Runs BASELINE plus one arm per ema_period, all in one lockstep pass."""
    arm_names = ["BASELINE"] + [f"EMA{p}" for p in ema_periods]
    ledgers = {}
    for name in arm_names:
        path = f"backtrack_logs/_ema_robustness_{label}_{name}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[name] = ledger

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    ema_by_symbol = {sym: compute_ema_states(candles, ema_periods) for sym, candles in symbols_candles.items()}

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

            ledgers["BASELINE"].check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledgers["BASELINE"].maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledgers["BASELINE"].maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            for p in ema_periods:
                name = f"EMA{p}"
                ema_val = ema_by_symbol[sym][p][t]
                ema_insight = apply_ema_veto(insight, candle.close, ema_val)
                ledgers[name].check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
                ledgers[name].maybe_open(sym, "spot", ema_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledgers[name].maybe_open(sym, "derivative", ema_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    for ledger in ledgers.values():
        ledger.open_positions.clear()
    return ledgers


def max_drawdown(ledger: PaperTradingLedger) -> float:
    """Peak-to-trough max drawdown (%) on the equity curve implied by
    closed_trades in chronological order."""
    trades = sorted(ledger.closed_trades, key=lambda t: t["closed_at"])
    equity = ledger.starting_balance
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
    return max_dd


def two_proportion_z(wins_a: int, n_a: int, wins_b: int, n_b: int) -> tuple[float, float]:
    """Two-proportion z-test. Returns (z, two_tailed_p_approx). Uses a
    normal approximation to the standard error of the pooled proportion —
    fine for the sample sizes here (all arms n>=20)."""
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a, p_b = wins_a / n_a, wins_b / n_b
    p_pool = (wins_a + wins_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    # two-tailed p from standard normal via erf
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def per_symbol_breakdown(ledger: PaperTradingLedger) -> dict[str, tuple[int, float]]:
    out: dict[str, list] = {}
    for t in ledger.closed_trades:
        sym = t["symbol"]
        if sym not in out:
            out[sym] = [0, 0.0]
        out[sym][0] += 1
        out[sym][1] += t["pnl_dollars"]
    return {k: (v[0], round(v[1], 2)) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe
    max_days = YF_MAX_DAYS.get(timeframe, 730)
    days = min(args.days if args.days is not None else max_days, max_days)

    print(f"ASSET_TYPE=crypto — {len(settings.WATCHLIST)} symbols: {settings.WATCHLIST}")
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

    # ---------- TEST 1: multi-regime segmentation, EMA50 only ----------
    print(f"\n{'=' * 90}\nTEST 1 — MULTI-REGIME: baseline vs EMA50 across {n_seg} chronological segments\n{'=' * 90}")
    print(f"{'Seg':<6}{'Period':<24}{'Base trades/WR/PnL':<32}{'EMA50 trades/WR/PnL':<32}")
    print("-" * 90)
    seg_results = []
    for i, seg in enumerate(segments):
        ledgers = run_segment(seg, timeframe, f"seg{i}", [50])
        base, ema = ledgers["BASELINE"], ledgers["EMA50"]
        bs, es = base.stats()["overall"], ema.stats()["overall"]
        start_ts = min((c[0].timestamp for c in seg.values() if c), default=None)
        end_ts = max((c[-1].timestamp for c in seg.values() if c), default=None)
        period_str = f"{start_ts:%Y-%m-%d} -> {end_ts:%Y-%m-%d}" if start_ts and end_ts else "-"
        bwr = f"{bs['win_rate_pct']:.1f}%" if bs["win_rate_pct"] is not None else "-"
        ewr = f"{es['win_rate_pct']:.1f}%" if es["win_rate_pct"] is not None else "-"
        print(f"seg{i:<3}{period_str:<24}{bs['trades']:>3}t/{bwr:>6}/${bs['total_pnl_dollars']:>+9.0f}   {es['trades']:>3}t/{ewr:>6}/${es['total_pnl_dollars']:>+9.0f}")
        seg_results.append((i, period_str, base, ema, bs, es))

    beats = sum(1 for *_r, bs, es in seg_results if es["total_pnl_dollars"] > bs["total_pnl_dollars"])
    print("-" * 90)
    print(f"EMA50 beats baseline PnL in {beats}/{n_seg} segments.")
    if beats < n_seg:
        print("NOTE: not unanimous — see per-segment breakdown above for which regime(s) it fails in.")

    # Drawdown + significance on segment 0 vs last segment (or pooled overall not directly summable across independent ledgers,
    # so report per-segment drawdown/significance individually).
    print(f"\n{'Seg':<6}{'Base maxDD':<14}{'EMA50 maxDD':<14}{'WR z-stat':<12}{'p-value':<10}")
    print("-" * 60)
    for i, period_str, base, ema, bs, es in seg_results:
        base_dd = max_drawdown(base)
        ema_dd = max_drawdown(ema)
        z, p = two_proportion_z(bs["wins"], bs["trades"], es["wins"], es["trades"])
        sig = "*" if p < 0.05 else ""
        print(f"seg{i:<3}{base_dd:>10.1f}%   {ema_dd:>10.1f}%   {z:>+8.2f}    {p:>6.3f}{sig}")
    print("(* = win-rate delta significant at p<0.05, two-proportion z-test)")

    # ---------- TEST 2: EMA period sensitivity sweep, full window ----------
    print(f"\n{'=' * 90}\nTEST 2 — PARAMETER SENSITIVITY: EMA period sweep on full {days}d window\n{'=' * 90}")
    full_ledgers = run_segment(all_candles, timeframe, "full_sweep", EMA_PERIODS_TO_SWEEP)
    base = full_ledgers["BASELINE"]
    bs = base.stats()["overall"]
    print(f"{'Period':<10}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}{'vs baseline':<14}")
    print("-" * 90)
    bwr = f"{bs['win_rate_pct']:.1f}%" if bs["win_rate_pct"] is not None else "-"
    print(f"{'BASELINE':<10}{bs['trades']:<9}{bwr:<11}{bs['total_pnl_dollars']:>+10.2f}   -")
    for p in EMA_PERIODS_TO_SWEEP:
        s = full_ledgers[f"EMA{p}"].stats()["overall"]
        wr = f"{s['win_rate_pct']:.1f}%" if s["win_rate_pct"] is not None else "-"
        beats = "beats base" if s["total_pnl_dollars"] > bs["total_pnl_dollars"] else ""
        print(f"{p:<10}{s['trades']:<9}{wr:<11}{s['total_pnl_dollars']:>+10.2f}   {beats}")

    # ---------- TEST 3: per-symbol concentration, EMA50 vs baseline, full window ----------
    print(f"\n{'=' * 90}\nTEST 3 — PER-SYMBOL CONCENTRATION: EMA50 vs baseline, full {days}d window\n{'=' * 90}")
    ema50 = full_ledgers["EMA50"]
    base_by_sym = per_symbol_breakdown(base)
    ema_by_sym = per_symbol_breakdown(ema50)
    print(f"{'Symbol':<12}{'Base trades/PnL':<24}{'EMA50 trades/PnL':<24}")
    print("-" * 90)
    for sym in settings.WATCHLIST:
        bt, bp = base_by_sym.get(sym, (0, 0.0))
        et, ep = ema_by_sym.get(sym, (0, 0.0))
        print(f"{sym:<12}{bt:>3}t / ${bp:>+9.2f}         {et:>3}t / ${ep:>+9.2f}")
    total_base_pnl = sum(p for _, p in base_by_sym.values())
    total_ema_pnl = sum(p for _, p in ema_by_sym.values())
    top_sym_ema_pnl = max((p for _, p in ema_by_sym.values()), default=0)
    concentration = (top_sym_ema_pnl / total_ema_pnl * 100) if total_ema_pnl else 0
    print(f"\nLargest single symbol's share of EMA50's total PnL: {concentration:.1f}%")
    if concentration > 50:
        print("WARNING: more than half of EMA50's edge comes from one symbol — concentration risk.")
    else:
        print("Edge is reasonably distributed across symbols, not concentrated in one.")

    # ---------- TEST 4: overall drawdown + significance, full window ----------
    print(f"\n{'=' * 90}\nTEST 4 — DRAWDOWN + SIGNIFICANCE: full {days}d window, baseline vs EMA50\n{'=' * 90}")
    base_dd = max_drawdown(base)
    ema_dd = max_drawdown(ema50)
    z, p = two_proportion_z(bs["wins"], bs["trades"], ema50.stats()["overall"]["wins"], ema50.stats()["overall"]["trades"])
    print(f"Baseline max drawdown: {base_dd:.1f}%   |   EMA50 max drawdown: {ema_dd:.1f}%")
    print(f"Win-rate delta z-stat: {z:+.2f}   p-value: {p:.4f}  {'(significant at p<0.05)' if p < 0.05 else '(NOT significant at p<0.05)'}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
