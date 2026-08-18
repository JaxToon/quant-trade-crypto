"""
Regime-filtered Path 1 backtest — controlled A/B/C, nothing deployed.

Tests whether skipping (or down-weighting) Path 1 entries during HMM-
flagged turbulent periods improves expectancy/drawdown. Follows directly
from hmm_regime_ab_experiment.py's finding that turbulent-tagged trades
have meaningfully worse expectancy (+$41.74 vs +$100.32) and win rate
(54.8% vs 60.8%).

IMPORTANT: this originally used detect_regime()'s OLD single-observation
method (one point at a time -- causal, but not live-equivalent). That
turned out to be a real, separate bug: startprob_ on the live checkpoint
concentrates ~100% of its mass on one state, so single-frame predict()
returns that same state for virtually any input (verified: 0.00%
turbulent flagged across a 60+ day walk-forward test). Fixed in
risk/hmm_regime.py via a rolling-window decode (see
hmm_windowed_decode_experiment.py for that validation) -- this script now
uses the SAME windowed-decode logic and the ACTUAL deployed checkpoint,
so the regime labels here are genuinely live-equivalent, not just causal.

Variants (all identical entry/stop/target logic to live Path 1 — only
whether maybe_open() gets called differs):
  baseline     - unchanged
  hard_filter  - skip ALL entries (spot+derivative) while regime==turbulent
  soft_filter  - during turbulent, only skip entries below a confidence
                 bar (SOFT_CONFIDENCE_THRESHOLD); calm periods untouched

Run separately per timeframe (5m/15m/1h) since Path 1 behaves differently
at each — matches backtest_target_matrix.py's days-per-timeframe choices
for direct comparability with that earlier result.

Reports the requested metric set (trades, win_rate%, total_pnl$,
total_return%, expectancy$/trade, max_drawdown%, profit_factor, avg_win$,
avg_loss$) PLUS the quality-vs-exposure breakdown: of baseline's own
trades that fall in a turbulent window, how many were wins vs losses, and
what's the $ split (foregone winning PnL vs avoided losing PnL) — that
tells you whether the filter is discriminating (avoided >> foregone) or
just cutting exposure (comparable, or foregone is a big chunk of it).

Does not modify paper_trading.py, entry_insight.py, fvg_analyzer.py,
risk/hmm_regime.py, model_checkpoints/, or config/settings.py.

Usage: python hmm_regime_filter_backtest.py
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from backtest import CRYPTO_TICKERS, fetch_candles
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from hmm_windowed_decode_experiment import fetch_features, load_live_model, order_remap, windowed_label
from strategy.entry_insight import build_entry_insight
from strategy.fvg_analyzer import FairValueGapAnalyzer
from strategy.smart_money import SmartMoneyAnalyzer

REGIME_FETCH_DAYS = 200  # covers every Path 1 window tested below (max 180d for 1h)
SOFT_CONFIDENCE_THRESHOLD = 60.0  # skip only sub-60%-confidence entries while turbulent
DECODE_WINDOW_HOURS = settings.HMM_DECODE_WINDOW_HOURS  # 336h, validated in hmm_windowed_decode_experiment.py

# (days, timeframe) per run — mirrors backtest_target_matrix.py's choices
# so results are directly comparable to that earlier matrix.
RUNS = [(60, "5m"), (60, "15m"), (180, "1h")]


# ── Regime model: the ACTUAL deployed checkpoint, decoded via the fixed
# windowed method (risk/hmm_regime.py), walked forward hour by hour ────────

def build_live_equivalent_regime_series():
    X, feature_timestamps = fetch_features(days=REGIME_FETCH_DAYS)
    model = load_live_model()
    remap = order_remap(model)

    # Windowed decode at every hour, using only that hour and everything
    # before it — genuinely causal, and matches the FIXED detect_regime()
    # exactly (see hmm_windowed_decode_experiment.py's windowed_label).
    ordered_states = np.array([windowed_label(model, remap, X, i, DECODE_WINDOW_HOURS) for i in range(len(X))])
    ts_epoch = np.array([t.timestamp() for t in feature_timestamps])
    return ts_epoch, ordered_states


def regime_at(ts_epoch: np.ndarray, ordered_states: np.ndarray, query_dt: datetime) -> int | None:
    idx = np.searchsorted(ts_epoch, query_dt.timestamp(), side="right") - 1
    if idx < 0 or idx >= len(ordered_states):
        return None
    return int(ordered_states[idx])


# ── Replay ──────────────────────────────────────────────────────────────

def replay(symbols_candles: dict, timeframe: str, ts_epoch, ordered_states, mode: str) -> PaperTradingLedger:
    """mode: 'baseline' | 'hard' | 'soft'"""
    ledger = PaperTradingLedger(path="backtrack_logs/_hmm_filter_scratch.json", starting_balance=settings.PAPER_TRADING_STARTING_BALANCE)
    ledger.balance = ledger.starting_balance
    ledger.max_closed_trades = 1_000_000

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

            regime = regime_at(ts_epoch, ordered_states, candle.timestamp)
            is_turbulent = regime == settings.HMM_TURBULENT_STATE

            ledger.check_and_close_ohlc(sym, candle.low, candle.high, closed_at=candle.timestamp.isoformat(), timeframe=timeframe)
            insight = build_entry_insight(sym, candle.close, smc_context, gnn_signal=0.0)
            opened_at = candle.timestamp.isoformat()

            for market in ("spot", "derivative"):
                rec = insight[market]
                if mode == "hard" and is_turbulent:
                    continue
                if mode == "soft" and is_turbulent and rec.get("confidence", 0.0) < SOFT_CONFIDENCE_THRESHOLD:
                    continue
                ledger.maybe_open(sym, market, rec, timeframe=timeframe, opened_at=opened_at)

    ledger.open_positions.clear()
    return ledger


# ── Stats ───────────────────────────────────────────────────────────────

def compute_full_stats(trades: list[dict], starting_balance: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_pnl = sum(t["pnl_dollars"] for t in trades)
    gross_win = sum(t["pnl_dollars"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollars"] for t in losses))
    equity, peak, max_dd = starting_balance, starting_balance, 0.0
    for t in trades:
        equity += t["pnl_dollars"]
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
    return {
        "trades": n,
        "win_rate_pct": len(wins) / n * 100,
        "total_pnl": total_pnl,
        "total_return_pct": total_pnl / starting_balance * 100,
        "expectancy": total_pnl / n,
        "max_drawdown_pct": max_dd,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
    }


def print_stats_row(name: str, s: dict) -> None:
    if s["trades"] == 0:
        print(f"{name:<14}{'0':<8}(no trades)")
        return
    print(
        f"{name:<14}{s['trades']:<8}{s['win_rate_pct']:<10.1f}{s['total_pnl']:<+11.2f}"
        f"{s['total_return_pct']:<+10.2f}{s['expectancy']:<+12.2f}{s['max_drawdown_pct']:<9.2f}"
        f"{s['profit_factor']:<9.2f}{s['avg_win']:<+10.2f}{s['avg_loss']:<+10.2f}"
    )


def quality_vs_exposure(baseline_trades: list[dict], ts_epoch, ordered_states, starting_balance: float) -> None:
    turbulent, calm = [], []
    for t in baseline_trades:
        opened = datetime.fromisoformat(t["opened_at"])
        regime = regime_at(ts_epoch, ordered_states, opened)
        if regime is None:
            continue
        (turbulent if regime == settings.HMM_TURBULENT_STATE else calm).append(t)

    wins = [t for t in turbulent if t["outcome"] == "win"]
    losses = [t for t in turbulent if t["outcome"] == "loss"]
    foregone_win_pnl = sum(t["pnl_dollars"] for t in wins)
    avoided_loss_pnl = -sum(t["pnl_dollars"] for t in losses)
    net = avoided_loss_pnl - foregone_win_pnl

    print(f"\n  Baseline trades that fall in a turbulent window (would be hard-filtered): {len(turbulent)}")
    if turbulent:
        print(f"    of those: {len(wins)} wins (foregone PnL ${foregone_win_pnl:+.2f}), "
              f"{len(losses)} losses (avoided PnL ${avoided_loss_pnl:+.2f})")
        print(f"    net $ effect of removing them: ${net:+.2f}  "
              f"({'filter net-positive' if net > 0 else 'filter net-negative — cut more winners than losers'})")

    calm_stats = compute_full_stats(calm, starting_balance)
    overall_stats = compute_full_stats(baseline_trades, starting_balance)
    if calm_stats["trades"] and overall_stats["trades"]:
        print(f"\n  Quality check: expectancy of calm-only subset (${calm_stats['expectancy']:+.2f}) "
              f"vs baseline overall (${overall_stats['expectancy']:+.2f})")
        if calm_stats["expectancy"] > overall_stats["expectancy"] * 1.1:
            print("    -> Calm subset is meaningfully BETTER than the average trade: filter discriminates quality.")
        elif calm_stats["expectancy"] < overall_stats["expectancy"] * 0.9:
            print("    -> Calm subset is WORSE than the average trade: filtering would hurt, not help.")
        else:
            print("    -> Calm subset is close to the average trade: filter mostly just cuts exposure, doesn't discriminate.")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Building live-equivalent HMM regime series ({REGIME_FETCH_DAYS}d BTC-USD, "
          f"windowed decode, {DECODE_WINDOW_HOURS}h window, deployed checkpoint)...")
    ts_epoch, ordered_states = build_live_equivalent_regime_series()
    turbulent_frac = (ordered_states == settings.HMM_TURBULENT_STATE).mean() * 100
    print(f"  {len(ordered_states)} hourly steps, {turbulent_frac:.1f}% flagged turbulent (windowed, live-equivalent)\n")

    header = (
        f"{'Variant':<14}{'Trades':<8}{'WinRate%':<10}{'TotalPnL$':<11}{'Return%':<10}"
        f"{'Expect$':<12}{'MaxDD%':<9}{'PF':<9}{'AvgWin$':<10}{'AvgLoss$':<10}"
    )

    for days, timeframe in RUNS:
        print(f"\n{'=' * len(header)}")
        print(f"TIMEFRAME {timeframe} ({days}d)")
        print(f"{'=' * len(header)}")

        all_candles = {}
        for symbol in CRYPTO_TICKERS:
            c = fetch_candles(symbol, days, timeframe)
            if c:
                all_candles[symbol] = c

        starting_balance = settings.PAPER_TRADING_STARTING_BALANCE
        baseline_ledger = replay(all_candles, timeframe, ts_epoch, ordered_states, "baseline")
        hard_ledger = replay(all_candles, timeframe, ts_epoch, ordered_states, "hard")
        soft_ledger = replay(all_candles, timeframe, ts_epoch, ordered_states, "soft")

        print(header)
        print("-" * len(header))
        print_stats_row("baseline", compute_full_stats(baseline_ledger.closed_trades, starting_balance))
        print_stats_row("hard_filter", compute_full_stats(hard_ledger.closed_trades, starting_balance))
        print_stats_row(f"soft_filter", compute_full_stats(soft_ledger.closed_trades, starting_balance))

        quality_vs_exposure(baseline_ledger.closed_trades, ts_epoch, ordered_states, starting_balance)

    print("\nThis is a quantitative comparison only — nothing here is deployed to")
    print("paper_trading.py, entry_insight.py, or risk/hmm_regime.py regardless of outcome.")


if __name__ == "__main__":
    main()
