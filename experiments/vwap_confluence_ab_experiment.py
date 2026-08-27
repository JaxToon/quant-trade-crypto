"""
Pre-registered A/B experiment: does adding a VWAP directional-confluence
filter to the existing SMC/FVG entry pipeline improve trade quality?

HYPOTHESIS (fixed before running): a trade whose direction disagrees with
price's position relative to daily-anchored VWAP (long while price < VWAP,
short while price > VWAP) is fighting the day's volume-weighted average
price — vetoing those trades should raise win rate and/or PF versus the
current SMC/FVG-only baseline, without cutting trade count so much the
strategy is effectively starved.

VWAP definition: daily-anchored (reset at each UTC midnight), the
standard convention for continuous 24/7 crypto markets — cumulative
(typical_price * volume) / cumulative volume since the day's first candle.
Typical price = (high + low + close) / 3.

Design, matching the discipline already used for market_structure_fix and
the GNN ablation in backtest.py:
  - SAME lockstep replay engine (replay_lockstep's index-aligned execution,
    not sequential-per-symbol — see backtest.py's docstring for why that
    confound matters for the correlation cap) for BOTH arms, differing
    ONLY in whether the VWAP veto is applied.
  - Real yfinance OHLCV data, not synthetic.
  - Reported on BOTH an in-sample and out-of-sample half, not one window.
  - gnn_signal held at 0.0 for both arms (isolates VWAP's effect only,
    same reasoning as run_gnn_ablation).

Usage: python vwap_confluence_ab_experiment.py [--days N] [--timeframe 5m|15m|1h]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime

from backtest import CRYPTO_TICKERS, STOCK_TICKERS, YF_MAX_DAYS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import Candle, SmartMoneyAnalyzer

SYMBOL_TICKERS = STOCK_TICKERS if settings.ASSET_TYPE == "stock" else CRYPTO_TICKERS


def compute_daily_vwap_series(candles: list[Candle]) -> list[float]:
    """Returns one VWAP value per candle, anchored to reset at each UTC
    calendar-day boundary present in the data."""
    vwaps: list[float] = []
    cum_pv = 0.0
    cum_vol = 0.0
    current_day = None
    for c in candles:
        day = c.timestamp.date()
        if day != current_day:
            current_day = day
            cum_pv = 0.0
            cum_vol = 0.0
        typical = (c.high + c.low + c.close) / 3.0
        vol = c.volume if c.volume and c.volume > 0 else 1.0  # yfinance sometimes reports 0 volume on thin crypto ticks
        cum_pv += typical * vol
        cum_vol += vol
        vwaps.append(cum_pv / cum_vol if cum_vol > 0 else c.close)
    return vwaps


def apply_vwap_veto(insight: dict, price: float, vwap: float) -> dict:
    """Vetoes (actionable=False) any spot/derivative recommendation whose
    direction disagrees with price's side of VWAP. Neutral/no-setup
    insights pass through unchanged."""
    direction = insight.get("direction")
    if direction not in ("long", "short"):
        return insight

    agrees = (direction == "long" and price >= vwap) or (direction == "short" and price <= vwap)
    if agrees:
        return insight

    vetoed = dict(insight)
    for leg in ("spot", "derivative"):
        if vetoed[leg]["actionable"]:
            vetoed[leg] = dict(vetoed[leg])
            vetoed[leg]["actionable"] = False
            vetoed[leg]["note"] = vetoed[leg].get("note", "") + " [VWAP-veto: arah berlawanan dengan VWAP harian]"
    return vetoed


def replay_lockstep_ab(
    symbols_candles: dict[str, list[Candle]],
    ledger_baseline: PaperTradingLedger,
    ledger_vwap: PaperTradingLedger,
    timeframe: str,
) -> int:
    """Runs BOTH arms through the identical event sequence in one pass —
    guarantees identical execution order/correlation-cap state evolution
    up to the point each arm's own open-position set diverges, same as
    market_structure_fix_ab_multitimeframe.py's approach."""
    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    vwap_by_symbol = {sym: compute_daily_vwap_series(candles) for sym, candles in symbols_candles.items()}

    min_len = min((len(c) for c in symbols_candles.values()), default=0)
    n_candles = 0
    for t in range(min_len):
        for sym, candles in symbols_candles.items():
            candle = candles[t]
            vwap = vwap_by_symbol[sym][t]

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

            ledger_baseline.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledger_baseline.maybe_open(sym, "spot", insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger_baseline.maybe_open(sym, "derivative", insight["derivative"], timeframe=timeframe, opened_at=opened_at)

            vwap_insight = apply_vwap_veto(insight, candle.close, vwap)
            ledger_vwap.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=opened_at, timeframe=timeframe)
            ledger_vwap.maybe_open(sym, "spot", vwap_insight["spot"], timeframe=timeframe, opened_at=opened_at)
            ledger_vwap.maybe_open(sym, "derivative", vwap_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

        n_candles += len(symbols_candles)
    return n_candles


def run_arm_pair(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> tuple[dict, dict]:
    baseline = PaperTradingLedger(
        path=f"backtrack_logs/_vwap_ab_baseline_{label}.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    baseline.balance = baseline.starting_balance
    baseline.max_closed_trades = 1_000_000

    vwap_arm = PaperTradingLedger(
        path=f"backtrack_logs/_vwap_ab_vwap_{label}.json",
        starting_balance=settings.PAPER_TRADING_STARTING_BALANCE,
    )
    vwap_arm.balance = vwap_arm.starting_balance
    vwap_arm.max_closed_trades = 1_000_000

    replay_lockstep_ab(symbols_candles, baseline, vwap_arm, timeframe)

    baseline.open_positions.clear()
    vwap_arm.open_positions.clear()
    return baseline.stats(), vwap_arm.stats()


def print_pair(title: str, base_stats: dict, vwap_stats: dict) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    for arm_name, s in [("BASELINE (SMC/FVG only)", base_stats), ("VWAP-VETO (SMC/FVG + VWAP confluence)", vwap_stats)]:
        o = s["overall"]
        wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
        print(f"{arm_name:<42} trades={o['trades']:<5} win_rate={wr:<8} pnl=${o['total_pnl_dollars']:+.2f}  return={s['total_return_pct']:+.2f}%")
    b, v = base_stats["overall"], vwap_stats["overall"]
    if b["win_rate_pct"] is not None and v["win_rate_pct"] is not None:
        print(f"\nWin rate delta: {v['win_rate_pct'] - b['win_rate_pct']:+.1f}pp   Trade count delta: {v['trades'] - b['trades']:+d} ({100*(v['trades']-b['trades'])/max(b['trades'],1):+.1f}%)")
    if b["trades"] < 20 or v["trades"] < 20:
        print("NOTE: one or both arms have <20 trades — treat this segment as directional, not conclusive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test: SMC/FVG baseline vs SMC/FVG + VWAP directional-confluence veto.")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", choices=settings.BINANCE_SMC_TIMEFRAMES, default=settings.BINANCE_KLINE_INTERVAL)
    args = parser.parse_args()
    timeframe = args.timeframe

    max_days = YF_MAX_DAYS.get(timeframe, 60)
    days = min(args.days if args.days is not None else settings.BACKTEST_DAYS, max_days)

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

    n = min(len(c) for c in all_candles.values())
    mid = n // 2
    in_sample = {sym: candles[:mid] for sym, candles in all_candles.items()}
    out_sample = {sym: candles[mid:] for sym, candles in all_candles.items()}

    base_is, vwap_is = run_arm_pair(in_sample, timeframe, "is")
    base_oos, vwap_oos = run_arm_pair(out_sample, timeframe, "oos")

    print_pair(f"IN-SAMPLE [{timeframe}] (first half, ~{days // 2}d)", base_is, vwap_is)
    print_pair(f"OUT-OF-SAMPLE [{timeframe}] (second half, ~{days - days // 2}d)", base_oos, vwap_oos)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    for label, (b, v) in [("IN-SAMPLE", (base_is, vwap_is)), ("OUT-OF-SAMPLE", (base_oos, vwap_oos))]:
        bo, vo = b["overall"], v["overall"]
        bwr = bo["win_rate_pct"]
        vwr = vo["win_rate_pct"]
        delta = (vwr - bwr) if (bwr is not None and vwr is not None) else None
        delta_str = f"{delta:+.1f}pp" if delta is not None else "n/a"
        print(f"{label:<15} baseline={bo['trades']:>4} trades / {bwr if bwr is not None else 0:.1f}% WR   "
              f"vwap-veto={vo['trades']:>4} trades / {vwr if vwr is not None else 0:.1f}% WR   delta={delta_str}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
