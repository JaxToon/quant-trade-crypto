"""
Validates the windowed-decode fix for detect_regime()'s single-observation
Viterbi bug BEFORE touching risk/hmm_regime.py or any live checkpoint.

Root cause (confirmed against the actual deployed
model_checkpoints/hmm_regime_crypto.pkl): detect_regime() calls
model.predict(obs) with a SINGLE observation every time. For a length-1
sequence, Viterbi reduces to argmax(log(startprob) + log(emission)).
startprob_ on the live checkpoint is [6.97e-20, 1.0, 2.76e-134] — one raw
state has ~100% of the prior mass, so single-frame predict() returns that
SAME state for virtually any input (verified directly: even an observation
sitting almost exactly on the turbulent state's own fitted mean still
classified as the dominant state). This is fully consistent with live logs
showing "REGIME: LOW_VOLATILITY" continuously regardless of actual
conditions.

Fix under test: give predict() a WINDOW of the N most recent hourly
observations (not just one), so startprob_ only influences the window's
first step and transmat_ + emission drive the rest — genuine HMM state
inference instead of a startprob-dominated point estimate. This is causal
(the window always ends at "now", never reaches into the future) — proven
below via true walk-forward simulation, not just a single retrospective
slice.

Validates all six of the stated requirements:
  1. Window uses only real, already-elapsed data (no lookahead)          -> walk-forward construction guarantees this by design
  2. Feature preprocessing identical to training                        -> reuses the exact vol/abs_change formula from fit()
  3. State mapping stays consistent (turbulent = correct state)         -> same order/remap logic as detect_regime()
  4. No lookahead                                                       -> explicit walk-forward, checked
  5. Latency acceptable                                                 -> reports wall-clock cost of one decode call
  6. Detector doesn't lag turbulence onset by too much                  -> measures transition lag vs retrospective ground truth

Does NOT modify risk/hmm_regime.py, model_checkpoints/*, or config/settings.py.

Usage: python hmm_windowed_decode_experiment.py
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import yfinance as yf

FETCH_DAYS = 120  # walk-forward test period + enough lookback for the largest window
CANDIDATE_WINDOWS_HOURS = [72, 168, 336]  # 3d, 7d, 14d
WALK_FORWARD_START_HOUR = 400  # skip enough that even the largest window is fully populated


def fetch_features(days: int = FETCH_DAYS):
    df = yf.download("BTC-USD", period=f"{days}d", interval="1h", progress=False, auto_adjust=True)
    closes = df["Close"].dropna().values.flatten().tolist()
    timestamps = [ts.to_pydatetime().replace(tzinfo=None) for ts in df.index[: len(closes)]]

    # Exactly risk/hmm_regime.py's fit() feature engineering.
    log_returns = np.array([np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))])
    vol = np.array([np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))])
    abs_change = np.array([
        abs((closes[i + 1] - closes[max(0, i + 1 - 24)]) / closes[max(0, i + 1 - 24)])
        for i in range(len(log_returns))
    ])
    X = np.column_stack([vol, abs_change])
    return X, timestamps[1:]


def load_live_model():
    with open("model_checkpoints/hmm_regime_crypto.pkl", "rb") as f:
        return pickle.load(f)


def order_remap(model):
    """Exactly detect_regime()'s re-ordering: 0=lowest vol ... N-1=highest vol."""
    means = model.means_[:, 0]
    order = np.argsort(means)
    return {raw: ordered for ordered, raw in enumerate(order)}


def single_frame_label(model, remap, obs_row: np.ndarray) -> int:
    """The CURRENT live method — single observation, startprob-dominated."""
    raw = model.predict(obs_row.reshape(1, -1))[0]
    return remap[raw]


def windowed_label(model, remap, X: np.ndarray, end_idx: int, window: int) -> int:
    """Proposed fix — Viterbi over the last `window` observations ending at
    (and including) end_idx, take the LAST state. Never reads past end_idx."""
    start = max(0, end_idx - window + 1)
    raw_path = model.predict(X[start:end_idx + 1])
    return remap[raw_path[-1]]


def retrospective_label(model, remap, X: np.ndarray) -> np.ndarray:
    """Full-history smoothed Viterbi — NOT causal, used only as a ground-
    truth proxy for 'what regime was this hour really in', matching the
    earlier hmm_regime_ab_experiment.py methodology."""
    raw_path = model.predict(X)
    return np.array([remap[s] for s in raw_path])


def main() -> None:
    print(f"Fetching {FETCH_DAYS}d of hourly BTC-USD data...")
    X, timestamps = fetch_features()
    print(f"  {len(X)} hourly feature rows, {timestamps[0]} -> {timestamps[-1]}\n")

    model = load_live_model()
    remap = order_remap(model)
    turbulent_ordered = remap[int(np.argmax(model.means_[:, 0]))]
    print(f"Loaded live checkpoint: n_states={model.n_components}, startprob_={model.startprob_}")
    print(f"Turbulent = ordered state {turbulent_ordered}\n")

    # Ground truth: full-history retrospective smoothed labels.
    retro = retrospective_label(model, remap, X)
    print(f"Retrospective (ground-truth proxy) turbulent fraction: {(retro == turbulent_ordered).mean() * 100:.1f}%\n")

    # --- Requirement 5: latency ---
    t0 = time.perf_counter()
    for _ in range(20):
        windowed_label(model, remap, X, len(X) - 1, 168)
    latency_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"Latency (168h window, single decode call): {latency_ms:.2f}ms — negligible vs network fetch cost\n")

    # --- Old method: single-frame, walked forward (should collapse) ---
    old_labels = np.array([single_frame_label(model, remap, X[i]) for i in range(WALK_FORWARD_START_HOUR, len(X))])
    old_turbulent_frac = (old_labels == turbulent_ordered).mean() * 100
    print(f"OLD (single-observation) method, walked forward: turbulent flagged {old_turbulent_frac:.2f}% of the time")
    print(f"  (vs retrospective ground truth for the same span: "
          f"{(retro[WALK_FORWARD_START_HOUR:] == turbulent_ordered).mean() * 100:.1f}%)\n")

    # --- New method: windowed, walked forward, for each candidate window ---
    print(f"{'Window':<10}{'AgreeWithRetro%':<18}{'TurbulentFrac%':<16}{'AvgTransitionLag(h)':<20}")
    print("-" * 64)
    for window in CANDIDATE_WINDOWS_HOURS:
        labels = np.array([
            windowed_label(model, remap, X, i, window)
            for i in range(WALK_FORWARD_START_HOUR, len(X))
        ])
        retro_slice = retro[WALK_FORWARD_START_HOUR:]
        agree = (labels == retro_slice).mean() * 100
        turb_frac = (labels == turbulent_ordered).mean() * 100

        # Transition lag: for each retrospective turbulent-onset (calm -> turbulent),
        # find how many hours later (if any, within 48h) the windowed method also flags it.
        lags = []
        for i in range(1, len(retro_slice)):
            if retro_slice[i] == turbulent_ordered and retro_slice[i - 1] != turbulent_ordered:
                for lag in range(0, 48):
                    if i + lag < len(labels) and labels[i + lag] == turbulent_ordered:
                        lags.append(lag)
                        break
        avg_lag = np.mean(lags) if lags else float("nan")
        print(f"{window:<10}{agree:<18.1f}{turb_frac:<16.2f}{avg_lag:<20.1f}")

    print("\nAgreement with retrospective ground truth should be high, turbulent")
    print("fraction should be well above ~0%, and lag should be small (few hours,")
    print("not days) for a window size to be a reasonable live replacement.")


if __name__ == "__main__":
    main()
