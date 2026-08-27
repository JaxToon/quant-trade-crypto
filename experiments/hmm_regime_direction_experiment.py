"""
Regime x direction decomposition study (2026-08-21).

Question: does HMM-flagged turbulence have symmetric consequences regardless
of direction, or does {turbulent + bullish} carry a materially different
forward risk distribution than {turbulent + bearish}? This is the empirical
evidence circuit_breaker.py's bear_veto_active would need before becoming
direction-aware instead of firing on regime alone (see risk/circuit_breaker.py
-- currently bear_veto_active == (regime == TURBULENT), full stop).

Purely observational. Does not modify risk/hmm_regime.py, risk/circuit_breaker.py,
model_checkpoints/, config/settings.py, or any live/Path 1 code. No trades are
opened -- this measures market-conditional forward behavior, not strategy P&L.

Design decisions locked BEFORE running (per this session's "preserve the
experiment definition before seeing results" discipline):

  Benchmark: BTC hourly only (matches what the live HMM itself trains on --
  see risk/hmm_regime.py's BENCHMARK dict). Not per-symbol.

  Regime: reuses the ACTUAL production HMM machinery -- same features
  (rolling 24h vol, rolling 24h abs% change, Fear&Greed), same
  n_components/covariance_type/n_iter/random_state as risk/hmm_regime.py's
  fit(), same HMM_DECODE_WINDOW_HOURS windowed-decode logic validated in
  hmm_windowed_decode_experiment.py. Two parallel views:
    - retrospective: one HMM fit on the full history, full-Viterbi decoded.
      Descriptive/diagnostic only -- NOT causal (the model has "seen" the
      whole span), reported for sanity-checking, not as deployment evidence.
    - causal walk-forward: refit weekly (matching main.py's
      `schedule.every().monday...regime_detector.fit(..., days=90)`) on
      ONLY the trailing 90d of features available as of that refit, then
      windowed-decode the following week with that fixed model before
      refitting again. This is decision-grade: state at hour t never uses
      a model that has seen data past t. (hmm_windowed_decode_experiment.py
      proved the DECODE window is causal; it never addressed whether the
      FIT itself was causal, since it decodes from one static loaded
      checkpoint across the whole test span. This closes that gap.)

  Direction: SmartMoneyAnalyzer.market_structure.trend, fed BTC candles one
  closed bar at a time (same object/update() call hmm_regime_filter_backtest.py
  already uses for Path 1's SMC context) -- benchmark-level, causal by
  construction, already-production logic, zero new hyperparameters.
  bullish -> bullish, bearish -> bearish, ranging/unknown -> neutral.

  Forward horizons: 4h / 12h / 24h, fixed upfront, all three always reported
  together (no post-hoc horizon selection).

  Boundary: classification happens using data available through candle t's
  close (t = the candle a regime/direction pair is keyed to). Forward return
  is close[t] -> close[t+h]. MAE/MFE search the high/low range starting at
  candle t+1 (next bar), never candle t itself, so the excursion doesn't
  double-count the bar that produced the classification.

  Direction convention for return/MAE/MFE: reported WITH the direction
  label's implied trade (long when bullish, short when bearish -- sign-
  flipped, MAE/MFE swapped accordingly), because the real question is "would
  a trade taken with this structural bias behave differently by regime",
  not an unconditional market return. Neutral has no implied trade -- raw
  (long-convention) numbers are reported but flagged informational-only.

  Metrics per {calm,turbulent} x {bullish,neutral,bearish} cell, per
  horizon: n, win rate, mean/expectancy, median, mean MAE, mean MFE, 10th-
  percentile MAE (downside tail). Fee-adjusted return (2x
  PAPER_TRADING_FEE_PCT round trip, the actual configured spot taker fee) is
  reported as a secondary sensitivity column, not folded into the core
  metrics -- this is a market-conditional study, not a strategy P&L
  simulation, and no entry/exit/turnover model is being assumed.

Usage: python hmm_regime_direction_experiment.py
"""
from __future__ import annotations

import numpy as np
import requests
from datetime import date
from hmmlearn import hmm

from backtest import fetch_candles
from config import settings
from strategy.smart_money import SmartMoneyAnalyzer

FETCH_DAYS = 700  # near yfinance's ~730d cap for 1h data
TRAIN_DAYS = 90  # matches main.py's weekly refit: regime_detector.fit(..., days=90)
REFIT_STEP_HOURS = 24 * 7  # weekly, matches schedule.every().monday
DECODE_WINDOW_HOURS = settings.HMM_DECODE_WINDOW_HOURS  # 336h, reused as-is
HORIZONS = [4, 12, 24]
ROUND_TRIP_FEE_PCT = 2 * settings.PAPER_TRADING_FEE_PCT  # 0.1%/side, configured spot taker fee


# ── Fear & Greed history (crypto-only 3rd HMM feature, duplicated from
# risk/hmm_regime.py's _fetch_fear_greed_history -- must stay in sync with
# that method's exact behavior; not imported since it's an instance method
# tied to RegimeDetector's own checkpoint-loading __init__) ────────────────

def fetch_fear_greed_history(days: int) -> dict:
    try:
        resp = requests.get("https://api.alternative.me/fng/", params={"limit": days}, timeout=10)
        resp.raise_for_status()
        history = {}
        for entry in resp.json().get("data", []):
            day = date.fromtimestamp(int(entry["timestamp"]))
            history[day] = float(entry["value"]) / 100.0
        return history
    except Exception as exc:
        print(f"Fear & Greed history fetch failed: {exc} -- proceeding with neutral (0.5) fill.")
        return {}


# ── Feature engineering (exactly risk/hmm_regime.py's fit()) ──────────────

def build_features(candles) -> tuple[np.ndarray, list]:
    closes = [c.close for c in candles]
    log_returns = np.array([np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))])
    vol = np.array([np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))])
    abs_change = np.array([
        abs((closes[i + 1] - closes[max(0, i + 1 - 24)]) / closes[max(0, i + 1 - 24)])
        for i in range(len(log_returns))
    ])
    timestamps = [c.timestamp for c in candles[1:]]  # X[i] <-> candles[i+1]

    fg_history = fetch_fear_greed_history(FETCH_DAYS)
    fg_feature = np.array([fg_history.get(ts.date(), 0.5) for ts in timestamps])

    X = np.column_stack([vol, abs_change, fg_feature])
    return X, timestamps


def order_remap(model) -> dict:
    means = model.means_[:, 0]  # mean volatility per raw state
    order = np.argsort(means)   # 0 = lowest vol ... N-1 = highest vol
    return {raw: ordered for ordered, raw in enumerate(order)}


def fit_hmm(X_train: np.ndarray):
    model = hmm.GaussianHMM(
        n_components=settings.HMM_N_STATES,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
    )
    model.fit(X_train)
    return model


def windowed_label(model, remap: dict, X: np.ndarray, end_idx: int, window: int) -> int:
    start = max(0, end_idx - window + 1)
    raw_path = model.predict(X[start:end_idx + 1])
    return remap[raw_path[-1]]


# ── Regime: retrospective (descriptive) ────────────────────────────────────

def retrospective_regime(X: np.ndarray) -> np.ndarray:
    model = fit_hmm(X)
    remap = order_remap(model)
    raw_path = model.predict(X)
    return np.array([remap[s] for s in raw_path])


# ── Regime: causal walk-forward (decision-grade) ───────────────────────────

def causal_walk_forward_regime(X: np.ndarray) -> np.ndarray:
    train_hours = TRAIN_DAYS * 24
    n = len(X)
    labels = np.full(n, -1, dtype=int)

    refit_idx = train_hours
    if refit_idx >= n:
        raise ValueError(f"Not enough history for a single walk-forward step: need >{train_hours}h, have {n}h")

    model, remap = None, None
    while refit_idx < n:
        fit_start = max(0, refit_idx - train_hours)
        X_train = X[fit_start:refit_idx]
        try:
            model = fit_hmm(X_train)
            remap = order_remap(model)
        except Exception as exc:
            print(f"  walk-forward refit at hour {refit_idx} failed ({exc}) -- carrying previous model forward.")
            if model is None:
                refit_idx += REFIT_STEP_HOURS
                continue

        decode_end = min(refit_idx + REFIT_STEP_HOURS, n)
        for j in range(refit_idx, decode_end):
            labels[j] = windowed_label(model, remap, X, j, DECODE_WINDOW_HOURS)
        refit_idx = decode_end

    n_cycles = (n - train_hours + REFIT_STEP_HOURS - 1) // REFIT_STEP_HOURS
    print(f"  causal walk-forward: {n_cycles} weekly refit cycles, "
          f"{(labels >= 0).sum()}/{n} hours labeled (first {train_hours}h are training-only, unlabeled)")
    return labels


# ── Direction: SMC market_structure.trend, replayed candle by candle ──────

def direction_series(candles) -> list[str]:
    smc = SmartMoneyAnalyzer(lookback=settings.SMC_LOOKBACK_CANDLES)
    trends = []
    for candle in candles:
        signals = smc.update(candle)
        raw = signals.get("market_structure", "unknown")
        trends.append({"bullish": "bullish", "bearish": "bearish"}.get(raw, "neutral"))
    return trends


# ── Forward outcome, direction-signed ──────────────────────────────────────

def forward_outcomes(candles, t_idx: int) -> dict[int, tuple[float, float, float]] | None:
    """Returns {horizon: (raw_return, raw_mfe, raw_mae)} or None if not enough
    forward data. MAE/MFE search candles[t_idx+1 : t_idx+h+1] (next bar
    onward), never candle t_idx itself."""
    n = len(candles)
    ref_price = candles[t_idx].close
    out = {}
    for h in HORIZONS:
        fwd_idx = t_idx + h
        if fwd_idx >= n:
            return None
        window = candles[t_idx + 1: fwd_idx + 1]
        if not window:
            return None
        fwd_close = candles[fwd_idx].close
        raw_return = (fwd_close - ref_price) / ref_price
        raw_mfe = (max(c.high for c in window) - ref_price) / ref_price
        raw_mae = (min(c.low for c in window) - ref_price) / ref_price
        out[h] = (raw_return, raw_mfe, raw_mae)
    return out


def sign_for_direction(direction: str, raw_return: float, raw_mfe: float, raw_mae: float) -> tuple[float, float, float]:
    if direction == "bullish":
        return raw_return, raw_mfe, raw_mae
    if direction == "bearish":
        return -raw_return, -raw_mae, -raw_mfe  # short: favorable/adverse swap
    return raw_return, raw_mfe, raw_mae  # neutral: informational, long-convention


# ── Cell aggregation & reporting ───────────────────────────────────────────

REGIME_LABELS = {0: "calm", 1: "turbulent"}
DIRECTIONS = ["bullish", "neutral", "bearish"]


def build_cells(regime_labels: np.ndarray, directions: list[str], candles) -> dict:
    """cells[horizon][(regime_name, direction)] = list of signed returns/mfe/mae"""
    cells = {h: {(r, d): [] for r in REGIME_LABELS.values() for d in DIRECTIONS} for h in HORIZONS}

    n = len(candles)
    for i in range(len(regime_labels)):
        t_idx = i + 1  # X[i] <-> candles[i+1]
        if t_idx >= n:
            continue
        regime = regime_labels[i]
        if regime < 0:
            continue  # unlabeled (causal walk-forward warm-up)
        regime_name = REGIME_LABELS.get(int(regime))
        if regime_name is None:
            continue
        direction = directions[t_idx]

        outcomes = forward_outcomes(candles, t_idx)
        if outcomes is None:
            continue
        for h in HORIZONS:
            raw_return, raw_mfe, raw_mae = outcomes[h]
            signed_return, signed_mfe, signed_mae = sign_for_direction(direction, raw_return, raw_mfe, raw_mae)
            cells[h][(regime_name, direction)].append((signed_return, signed_mfe, signed_mae))
    return cells


def print_table(cells: dict, mode_label: str) -> None:
    print(f"\n{'=' * 100}\n{mode_label}\n{'=' * 100}")
    for h in HORIZONS:
        print(f"\n-- Forward horizon: {h}h --")
        header = f"{'Regime':<11}{'Direction':<10}{'n':>6}{'WinRate%':>10}{'Mean%':>9}{'Median%':>9}{'MeanMFE%':>10}{'MeanMAE%':>10}{'P10 MAE%':>10}{'FeeAdjMean%':>13}"
        print(header)
        print("-" * len(header))
        for regime_name in REGIME_LABELS.values():
            for direction in DIRECTIONS:
                rows = cells[h][(regime_name, direction)]
                n = len(rows)
                if n == 0:
                    print(f"{regime_name:<11}{direction:<10}{0:>6}  (no observations)")
                    continue
                returns = np.array([r[0] for r in rows])
                mfes = np.array([r[1] for r in rows])
                maes = np.array([r[2] for r in rows])
                win_rate = (returns > 0).mean() * 100
                mean_ret = returns.mean() * 100
                median_ret = np.median(returns) * 100
                mean_mfe = mfes.mean() * 100
                mean_mae = maes.mean() * 100
                p10_mae = np.percentile(maes, 10) * 100
                fee_adj_mean = (returns.mean() - ROUND_TRIP_FEE_PCT) * 100
                flag = "" if direction != "neutral" else "  (no implied trade -- informational)"
                print(f"{regime_name:<11}{direction:<10}{n:>6}{win_rate:>10.1f}{mean_ret:>9.2f}{median_ret:>9.2f}"
                      f"{mean_mfe:>10.2f}{mean_mae:>10.2f}{p10_mae:>10.2f}{fee_adj_mean:>13.2f}{flag}")


def main() -> None:
    print(f"Fetching {FETCH_DAYS}d of BTCUSDT hourly candles...")
    candles = fetch_candles("BTCUSDT", days=FETCH_DAYS, timeframe="1h")
    print(f"  {len(candles)} candles: {candles[0].timestamp} -> {candles[-1].timestamp}\n")

    print("Building HMM features (vol, abs_change, fear&greed)...")
    X, timestamps = build_features(candles)
    print(f"  {len(X)} feature rows\n")

    print("Replaying SMC market_structure.trend (direction, causal by construction)...")
    directions = direction_series(candles)

    print("Fitting retrospective (descriptive, full-history) regime decode...")
    retro_labels = retrospective_regime(X)
    turbulent_frac_retro = (retro_labels == settings.HMM_TURBULENT_STATE).mean() * 100
    print(f"  retrospective turbulent fraction: {turbulent_frac_retro:.1f}%\n")

    print("Running causal walk-forward (weekly refit, decision-grade) regime decode...")
    causal_labels = causal_walk_forward_regime(X)
    labeled = causal_labels[causal_labels >= 0]
    turbulent_frac_causal = (labeled == settings.HMM_TURBULENT_STATE).mean() * 100
    agree = (causal_labels[causal_labels >= 0] == retro_labels[causal_labels >= 0])
    print(f"  causal turbulent fraction: {turbulent_frac_causal:.1f}%")
    print(f"  causal vs retrospective agreement (where both defined): {agree.mean() * 100:.1f}%\n")

    retro_cells = build_cells(retro_labels, directions, candles)
    causal_cells = build_cells(causal_labels, directions, candles)

    print_table(retro_cells, "DESCRIPTIVE VIEW (retrospective full-history decode -- NOT causal, diagnostic only)")
    print_table(causal_cells, "DECISION-GRADE VIEW (causal walk-forward, weekly refit -- what the live system would actually have known)")

    print(f"\n{'=' * 100}")
    print("This is a market-conditional observational study only. Nothing here is")
    print("deployed to risk/circuit_breaker.py, risk/hmm_regime.py, or any live")
    print("decision path regardless of which cell looks best. No trades were")
    print("opened/closed -- these are hypothetical WITH-direction excursions from")
    print("close[t], not simulated strategy P&L.")


if __name__ == "__main__":
    main()
