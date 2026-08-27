"""
Pre-registered A/B/combination-matrix experiment: does layering established,
non-ML technical confluence filters on top of the existing SMC/FVG entry
pipeline improve trade quality — individually or combined? Run separately
for crypto and stock (each asset class already has its own GNN
checkpoint/ledger per DEPLOYMENT.md, so results are reported separately
too, not pooled).

Candidate filters (chosen from a web survey of FVG/SMC confluence
practice, 2026-08-27 — see RESEARCH_NOTES.md for sources; VWAP was
already tested separately and rejected, not repeated here):

  EMA  - price vs EMA(50) trend alignment: veto longs below EMA50, shorts
         above it. Cited repeatedly as a standard FVG confluence layer.
  RSI  - extreme-avoidance: veto longs when RSI(14) > 70 (overbought),
         shorts when RSI(14) < 30 (oversold). Simpler and more robustly
         computable than RSI-divergence, which the same sources cite as
         the higher-conviction but harder-to-define version.
  VOL  - volume confirmation: veto entries on below-(20-period)-average
         volume, regardless of direction — "high volume rejection
         confirms the level is defended" per the same survey.

Each is a pure post-hoc veto on build_entry_insight()'s output (same
technique as vwap_confluence_ab_experiment.py) — no live pipeline file is
touched, and every arm writes to its own scratch ledger path under
backtrack_logs/_matrix_ab_*.json, fully isolated from the live paper
ledgers and from each other.

Design discipline (same as every prior AB script in this repo):
  - ALL arms (baseline + 7 filter combinations = 8 total) replayed in ONE
    lockstep pass per (asset_type, half) so every arm sees the identical
    candle sequence and SMC/FVG state evolution — only the veto layer
    differs between arms.
  - Real yfinance OHLCV data.
  - In-sample / out-of-sample split, both reported — a combo that only
    wins on one half is flagged, not declared a winner.
  - ATR-based filters and HTF-bias filtering are deliberately NOT
    reproduced here — both already have dedicated prior experiments
    (backtest_stoploss_matrix.py: ATR stops rejected; smc_confidence_matrix_backtest.py:
    ATR-relative confidence mixed; backtest_htf_bias_experiment.py: open
    question) and reopening them here would duplicate, not extend, that
    work.

Usage: python confluence_matrix_ab_experiment.py [--days N] [--timeframe 1h]
Run once per asset type: default settings.ASSET_TYPE, and again with
ASSET_TYPE=stock in the environment (matching start_system.sh's own
convention for switching asset class).
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
from train_gnn import _ema

SYMBOL_TICKERS = STOCK_TICKERS if settings.ASSET_TYPE == "stock" else CRYPTO_TICKERS

FILTER_NAMES = ["EMA", "RSI", "VOL"]


def compute_rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """Rolling RSI at every index, same simple-average methodology as
    train_gnn._rsi (not Wilder-smoothed) for consistency with the rest of
    this codebase's existing RSI usage."""
    n = len(closes)
    rsi = [50.0] * n
    if n < period + 1:
        return rsi
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    for i in range(period, n):
        avg_gain = sum(gains[i - period:i]) / period
        avg_loss = sum(losses[i - period:i]) / period
        rsi[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def compute_vol_avg_series(volumes: list[float], period: int = 20) -> list[float]:
    n = len(volumes)
    avg = [0.0] * n
    for i in range(n):
        window = volumes[max(0, i - period + 1):i + 1]
        avg[i] = sum(window) / len(window) if window else 0.0
    return avg


def compute_indicator_state(candles: list[Candle]) -> dict:
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    return {
        "ema50": _ema(50, closes),
        "rsi14": compute_rsi_series(closes, 14),
        "vol_avg20": compute_vol_avg_series(volumes, 20),
    }


def filter_passes(name: str, direction: str, price: float, volume: float, state: dict, t: int) -> bool:
    if name == "EMA":
        ema = state["ema50"][t]
        return price >= ema if direction == "long" else price <= ema
    if name == "RSI":
        rsi = state["rsi14"][t]
        return rsi <= 70 if direction == "long" else rsi >= 30
    if name == "VOL":
        return volume >= state["vol_avg20"][t]
    raise ValueError(name)


def apply_combo_veto(insight: dict, price: float, volume: float, state: dict, t: int, combo: tuple[str, ...]) -> dict:
    direction = insight.get("direction")
    if direction not in ("long", "short") or not combo:
        return insight
    if all(filter_passes(name, direction, price, volume, state, t) for name in combo):
        return insight
    vetoed = dict(insight)
    for leg in ("spot", "derivative"):
        if vetoed[leg]["actionable"]:
            vetoed[leg] = dict(vetoed[leg])
            vetoed[leg]["actionable"] = False
            vetoed[leg]["note"] = vetoed[leg].get("note", "") + f" [veto: {'+'.join(combo)}]"
    return vetoed


def all_combos() -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = [()]
    for r in range(1, len(FILTER_NAMES) + 1):
        combos.extend(itertools.combinations(FILTER_NAMES, r))
    return combos


def combo_label(combo: tuple[str, ...]) -> str:
    return "BASELINE" if not combo else "+".join(combo)


def run_matrix(symbols_candles: dict[str, list[Candle]], timeframe: str, label: str) -> dict[str, dict]:
    combos = all_combos()
    ledgers = {}
    for combo in combos:
        path = f"backtrack_logs/_matrix_ab_{label}_{combo_label(combo)}.json"
        ledger = PaperTradingLedger(path=path, starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
        ledger.balance = ledger.starting_balance
        ledger.max_closed_trades = 1_000_000
        ledgers[combo] = ledger

    smc_by_symbol = {sym: SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES) for sym in symbols_candles}
    fvg_by_symbol = {sym: FairValueGapAnalyzer(max_fvgs=50, cluster_proximity=0.02) for sym in symbols_candles}
    state_by_symbol = {sym: compute_indicator_state(candles) for sym, candles in symbols_candles.items()}

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
                combo_insight = apply_combo_veto(insight, candle.close, candle.volume, state, t, combo)
                ledger.maybe_open(sym, "spot", combo_insight["spot"], timeframe=timeframe, opened_at=opened_at)
                ledger.maybe_open(sym, "derivative", combo_insight["derivative"], timeframe=timeframe, opened_at=opened_at)

    results = {}
    for combo, ledger in ledgers.items():
        ledger.open_positions.clear()
        results[combo_label(combo)] = ledger.stats()
    return results


def print_matrix(title: str, results: dict[str, dict]) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")
    print(f"{'Combo':<18}{'Trades':<9}{'Win Rate':<11}{'PnL $':<14}{'Return %':<10}")
    print("-" * 78)
    base = results["BASELINE"]["overall"]
    for combo_name, s in results.items():
        o = s["overall"]
        wr = f"{o['win_rate_pct']:.1f}%" if o["win_rate_pct"] is not None else "-"
        marker = " <- baseline" if combo_name == "BASELINE" else ""
        print(f"{combo_name:<18}{o['trades']:<9}{wr:<11}{o['total_pnl_dollars']:>+10.2f}   {s['total_return_pct']:>+7.2f}%{marker}")
    print("-" * 78)
    # Rank combos (excluding baseline) by total PnL, flag anything beating baseline on BOTH pnl and win rate.
    non_base = {k: v for k, v in results.items() if k != "BASELINE"}
    ranked = sorted(non_base.items(), key=lambda kv: kv[1]["overall"]["total_pnl_dollars"], reverse=True)
    print("Ranked by total PnL (excl. baseline):")
    for name, s in ranked:
        o = s["overall"]
        beats_pnl = o["total_pnl_dollars"] > base["total_pnl_dollars"]
        beats_wr = (o["win_rate_pct"] or 0) > (base["win_rate_pct"] or 0)
        tag = " [beats baseline on BOTH pnl & WR]" if beats_pnl and beats_wr else (" [beats pnl only]" if beats_pnl else "")
        print(f"  {name:<18} pnl=${o['total_pnl_dollars']:+.2f}{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confluence-filter combination matrix A/B test (isolated, scratch ledgers only).")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--timeframe", choices=settings.BINANCE_SMC_TIMEFRAMES, default="1h")
    args = parser.parse_args()
    timeframe = args.timeframe

    max_days = YF_MAX_DAYS.get(timeframe, 60)
    days = min(args.days if args.days is not None else 90, max_days)

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

    if not all_candles:
        print("No data fetched for any symbol — aborting.")
        return

    n = min(len(c) for c in all_candles.values())
    mid = n // 2
    in_sample = {sym: candles[:mid] for sym, candles in all_candles.items()}
    out_sample = {sym: candles[mid:] for sym, candles in all_candles.items()}

    results_is = run_matrix(in_sample, timeframe, f"{settings.ASSET_TYPE}_is")
    results_oos = run_matrix(out_sample, timeframe, f"{settings.ASSET_TYPE}_oos")

    print_matrix(f"[{settings.ASSET_TYPE.upper()}] IN-SAMPLE [{timeframe}] (first half, ~{days // 2}d)", results_is)
    print_matrix(f"[{settings.ASSET_TYPE.upper()}] OUT-OF-SAMPLE [{timeframe}] (second half, ~{days - days // 2}d)", results_oos)

    print(f"\n{'=' * 78}")
    print(f"[{settings.ASSET_TYPE.upper()}] CONSISTENCY CHECK (combos beating baseline on PnL in BOTH halves)")
    print(f"{'=' * 78}")
    base_is_pnl = results_is["BASELINE"]["overall"]["total_pnl_dollars"]
    base_oos_pnl = results_oos["BASELINE"]["overall"]["total_pnl_dollars"]
    any_consistent = False
    for combo_name in results_is:
        if combo_name == "BASELINE":
            continue
        is_pnl = results_is[combo_name]["overall"]["total_pnl_dollars"]
        oos_pnl = results_oos[combo_name]["overall"]["total_pnl_dollars"]
        if is_pnl > base_is_pnl and oos_pnl > base_oos_pnl:
            any_consistent = True
            print(f"  {combo_name}: IS ${is_pnl:+.2f} (base ${base_is_pnl:+.2f})   OOS ${oos_pnl:+.2f} (base ${base_oos_pnl:+.2f})")
    if not any_consistent:
        print("  None. No combo beat the baseline's total PnL in both halves.")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
