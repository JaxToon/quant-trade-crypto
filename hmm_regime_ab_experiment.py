"""
Controlled A/B experiment: does a 2-state HMM regime classifier separate
market conditions any better than the live 3-state one, and — the
question that actually matters — does either model's "turbulent" flag
correlate with worse Path 1 (SMC/FVG) trade outcomes?

This does NOT touch model_checkpoints/hmm_regime_*.pkl, config/settings.py,
or risk/hmm_regime.py. Both HMM variants are fit fresh in-memory here, on
identical feature data, changing exactly one variable (n_components) —
everything else (covariance_type, n_iter, random_state, feature
engineering) matches risk/hmm_regime.py's fit() exactly so the comparison
isolates state count, not some other confound.

Known baseline (already established this session, re-verified below):
states 0/1 of the live 3-state model are near-identical (0.17-0.3% mean_vol
separation) — this experiment checks whether collapsing to 2 states fixes
that AND whether it changes anything that would actually matter to
trading — i.e., whether "turbulent" periods (by either model's definition)
are where Path 1 trades actually perform worse. If they're not, regime
filtering isn't the lever it looks like on paper, independent of state count.

Two parts:
  1. Cluster quality: mean volatility per ordered state, separation.
  2. Outcome relevance: replay real Path 1 trades (backtest_target_matrix's
     baseline target, unmodified), tag each trade's regime at its own
     opened_at timestamp (nearest-hour lookup) under BOTH HMM variants,
     and compare win rate / expectancy for trades-in-turbulent vs
     trades-in-calm.

Caveat, stated plainly: regime labels here come from hmm.predict() over
the FULL historical sequence (Viterbi decoding), which is a smoothed,
retrospective label — not the causal, point-in-time label live inference
actually produces. This is standard practice for offline regime-quality
analysis, but it means part 2's numbers are an upper bound on how well a
live filter would have done, not a live-equivalent backtest.

Usage: python hmm_regime_ab_experiment.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from hmmlearn import hmm

from backtest import CRYPTO_TICKERS, fetch_candles
from backtest_target_matrix import _make_variant, replay_variant

DAYS = 180  # 1h benchmark history window


def fetch_benchmark_closes(days: int) -> tuple[list[float], list[datetime]]:
    df = yf.download("BTC-USD", period=f"{days}d", interval="1h", progress=False, auto_adjust=True)
    closes = df["Close"].dropna().values.flatten().tolist()
    timestamps = [ts.to_pydatetime().replace(tzinfo=None) for ts in df.index[: len(closes)]]
    return closes, timestamps


def build_features(closes: list[float]) -> np.ndarray:
    """Exactly matches risk/hmm_regime.py's fit() feature engineering
    (vol, abs_change), minus Fear & Greed — isolating state count as the
    only variable under test, not feature set too."""
    log_returns = np.array([np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))])
    vol = np.array([np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))])
    abs_change = np.array([
        abs((closes[i + 1] - closes[max(0, i + 1 - 24)]) / closes[max(0, i + 1 - 24)])
        for i in range(len(log_returns))
    ])
    return np.column_stack([vol, abs_change])


def fit_hmm(X: np.ndarray, n_states: int) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42)
    model.fit(X)
    return model


def ordered_states(model: hmm.GaussianHMM, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ordered_state_per_observation, mean_vol_per_ordered_state) —
    0=lowest vol ... N-1=highest vol, same re-mapping risk/hmm_regime.py's
    detect_regime() applies."""
    raw = model.predict(X)
    means = model.means_[:, 0]
    order = np.argsort(means)
    remap = {raw_state: ordered for ordered, raw_state in enumerate(order)}
    ordered = np.array([remap[s] for s in raw])
    return ordered, means[order]


def report_cluster_quality(name: str, n_states: int, mean_vols: np.ndarray, ordered: np.ndarray) -> None:
    print(f"\n--- {name} ({n_states}-state) ---")
    for i, mv in enumerate(mean_vols):
        frac = (ordered == i).mean() * 100
        print(f"  state {i}: mean_vol={mv:.5f}  time_spent={frac:.1f}%")
    if len(mean_vols) >= 2:
        gaps = [(mean_vols[i + 1] - mean_vols[i]) / mean_vols[i] * 100 for i in range(len(mean_vols) - 1)]
        print(f"  relative separation between consecutive states: {['%.1f%%' % g for g in gaps]}")


def tag_trades_with_regime(trades: list[dict], timestamps: list[datetime], ordered: np.ndarray, turbulent_state: int) -> list[bool]:
    """For each trade, find the nearest benchmark hour <= opened_at and
    return whether that hour's regime was turbulent. Trades before the
    benchmark window starts are skipped (None-filtered by caller)."""
    ts_array = np.array([t.timestamp() for t in timestamps])
    flags = []
    for t in trades:
        opened = datetime.fromisoformat(t["opened_at"]).replace(tzinfo=None)
        idx = np.searchsorted(ts_array, opened.timestamp(), side="right") - 1
        if idx < 0 or idx >= len(ordered):
            flags.append(None)
            continue
        flags.append(bool(ordered[idx] == turbulent_state))
    return flags


def report_outcome_relevance(name: str, trades: list[dict], flags: list[bool | None]) -> None:
    turbulent = [t for t, f in zip(trades, flags) if f is True]
    calm = [t for t, f in zip(trades, flags) if f is False]

    def stats(group: list[dict]) -> str:
        if not group:
            return "n=0"
        wins = sum(1 for t in group if t["outcome"] == "win")
        wr = wins / len(group) * 100
        exp = sum(t["pnl_dollars"] for t in group) / len(group)
        return f"n={len(group):<4} win_rate={wr:5.1f}%  expectancy=${exp:+.2f}"

    print(f"\n--- {name}: Path 1 trade outcomes by regime-at-entry ---")
    print(f"  turbulent-flagged trades: {stats(turbulent)}")
    print(f"  calm-flagged trades:      {stats(calm)}")
    skipped = sum(1 for f in flags if f is None)
    if skipped:
        print(f"  ({skipped} trades before benchmark window start, excluded)")


def main() -> None:
    print(f"Fetching {DAYS}d of hourly BTC-USD data for HMM fitting...")
    closes, timestamps = fetch_benchmark_closes(DAYS)
    print(f"  {len(closes)} hourly closes, {timestamps[0]} -> {timestamps[-1]}")

    X = build_features(closes)
    feature_timestamps = timestamps[1:]  # log_returns drops the first close

    print("\nFitting 3-state (live config) and 2-state (candidate) HMMs on identical features...")
    model3 = fit_hmm(X, 3)
    model2 = fit_hmm(X, 2)

    ordered3, means3 = ordered_states(model3, X)
    ordered2, means2 = ordered_states(model2, X)

    print("\n=== PART 1: Cluster quality ===")
    report_cluster_quality("3-state (live)", 3, means3, ordered3)
    report_cluster_quality("2-state (candidate)", 2, means2, ordered2)

    print("\nFetching Path 1 (SMC/FVG) crypto candles for the outcome-relevance check...")
    all_candles = {}
    for symbol in CRYPTO_TICKERS:
        c = fetch_candles(symbol, 60, "1h")  # 1h to align cleanly with the hourly regime series
        if c:
            all_candles[symbol] = c
    ledger = replay_variant(all_candles, "1h", _make_variant("baseline", None))
    trades = ledger.closed_trades
    print(f"  {len(trades)} Path 1 baseline trades replayed (60d/1h, unmodified target)")

    flags3 = tag_trades_with_regime(trades, feature_timestamps, ordered3, turbulent_state=2)
    flags2 = tag_trades_with_regime(trades, feature_timestamps, ordered2, turbulent_state=1)

    print("\n=== PART 2: Does 'turbulent' actually predict worse Path 1 trades? ===")
    report_outcome_relevance("3-state (live)", trades, flags3)
    report_outcome_relevance("2-state (candidate)", trades, flags2)

    print("\nNote: regime labels are retrospective (full-sequence Viterbi decode),")
    print("an upper bound on live filtering quality, not a live-equivalent backtest.")
    print("Nothing here touches model_checkpoints/hmm_regime_*.pkl or config/settings.py.")


if __name__ == "__main__":
    main()
