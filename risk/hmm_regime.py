"""
Hidden Markov Model regime classifier.
States (settings.HMM_N_STATES): 0=LOW_VOLATILITY, 1=HIGH_STRESS_TURBULENT

Features:
  - Benchmark 24h volatility (std of log returns)
  - Benchmark 24h absolute price change %
  - Fear & Greed Index (crypto only — see below)

Benchmark is BTC for crypto, SPY for stock (see BENCHMARK below) — found
this was hardcoded to BTC regardless of ASSET_TYPE while checking whether
crypto and stock could run simultaneously: a stock process's bear-veto
circuit breaker would have been reacting to crypto market stress, not
equity market stress. Checkpoint is asset-type-suffixed for the same
reason — a crypto and a stock process running at once would otherwise
silently overwrite each other's fitted regime model.

Fear & Greed is fetched from alternative.me, which is specifically a
CRYPTO sentiment index — not a general market sentiment gauge. It used to
be fetched every detect_regime() call and then thrown away entirely
("can be added as 3rd dim if desired") despite being free/already paid
for in API-call terms. Now actually used as a 3rd feature dimension, but
ONLY for crypto — reusing it for stock would repeat the exact mismatch
this file's benchmark fix (BTC vol used for stock regime) just corrected.
Stock keeps the original 2-feature (vol, change) model.

Also found live (user's own observation, not assumed): requesting 3 HMM
states from vol+change alone consistently collapsed 2 of them into a
near-identical "calm" cluster (~0.3% mean-vol separation, vs ~74-80%
against the genuinely distinct turbulent state) — a real 2-regime
process at this resolution, not 3. This had ZERO behavioral effect at the
time (circuit_breaker.py only ever checked state == HMM_TURBULENT_STATE,
and states 0/1 were treated identically either way), confirmed directly
via an A/B: 2-state vs 3-state produce IDENTICAL turbulent/calm
classification for every trade tested. HMM_N_STATES is now 2 — pure
correctness/parsimony, not a behavior change (see
hmm_regime_ab_experiment.py).

detect_regime() decodes a rolling window of recent observations (see
HMM_DECODE_WINDOW_HOURS), not a single point — feeding predict() one
observation at a time let a peaked startprob_ dominate the decision
regardless of the actual input; see hmm_windowed_decode_experiment.py.

Fitted on recent hourly benchmark history; refitted weekly.
"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests
from hmmlearn import hmm

from config import settings

logger = logging.getLogger(__name__)

# Binance only lists crypto pairs — SPY (S&P 500 ETF) is the stock-side
# proxy for "is the broad market stressed right now", fetched via yfinance
# only (no Binance leg to try for an equity ticker).
BENCHMARK = {"crypto": "BTCUSDT", "stock": "SPY"}
BENCHMARK_YF_TICKER = {"crypto": "BTC-USD", "stock": "SPY"}

HMM_CHECKPOINT = f"model_checkpoints/hmm_regime_{settings.ASSET_TYPE}.pkl"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


class RegimeDetector:

    def __init__(self) -> None:
        self._model: Optional[hmm.GaussianHMM] = None
        self._last_state: int = 1  # default: NORMAL
        self._benchmark = BENCHMARK.get(settings.ASSET_TYPE, "BTCUSDT")
        self._benchmark_yf = BENCHMARK_YF_TICKER.get(settings.ASSET_TYPE, "BTC-USD")
        self._load()

    # ── Model persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(HMM_CHECKPOINT):
            with open(HMM_CHECKPOINT, "rb") as f:
                self._model = pickle.load(f)
            logger.info("HMM model loaded from %s", HMM_CHECKPOINT)

    def _save(self) -> None:
        os.makedirs("model_checkpoints", exist_ok=True)
        with open(HMM_CHECKPOINT, "wb") as f:
            pickle.dump(self._model, f)

    # ── Data collection ────────────────────────────────────────────────────

    def _fetch_fear_greed(self) -> float:
        """Returns Fear & Greed index 0-100 (lower = more fear)."""
        try:
            resp = requests.get(FEAR_GREED_URL, timeout=5)
            data = resp.json()
            return float(data["data"][0]["value"]) / 100.0
        except Exception:
            return 0.5  # neutral on failure

    def _fetch_fear_greed_history(self, days: int) -> dict:
        """Returns {date: normalized F&G value} for fit()'s training window
        (crypto only — see module docstring for why). alternative.me serves
        one value PER DAY, while the price data fit() trains on is hourly —
        this dict gets looked up per-hour by calendar date, forward-filled
        naturally since the caller just skips missing dates."""
        try:
            resp = requests.get("https://api.alternative.me/fng/", params={"limit": days}, timeout=10)
            resp.raise_for_status()
            history = {}
            for entry in resp.json().get("data", []):
                day = datetime.utcfromtimestamp(int(entry["timestamp"])).date()
                history[day] = float(entry["value"]) / 100.0
            return history
        except Exception as exc:
            logger.warning("Fear & Greed history fetch failed: %s", exc)
            return {}

    def _fetch_benchmark_volatility(self, client=None) -> tuple[float, float]:
        """Returns (24h_vol, 24h_change_pct) for this asset type's benchmark
        (BTCUSDT for crypto, SPY for stock). Binance REST first (crypto
        only — it doesn't list SPY), yfinance fallback otherwise."""
        if settings.ASSET_TYPE == "crypto":
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": self._benchmark, "interval": "1h", "limit": 25},
                    timeout=5,
                )
                resp.raise_for_status()
                closes = [float(k[4]) for k in resp.json()]
                log_returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
                return float(np.std(log_returns)), abs((closes[-1] - closes[0]) / closes[0])
            except Exception:
                pass

        try:
            import yfinance as yf
            df = yf.download(self._benchmark_yf, period="2d", interval="1h", progress=False, auto_adjust=True)
            closes = df["Close"].dropna().values.flatten().tolist()
            if len(closes) >= 2:
                log_returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
                return float(np.std(log_returns)), abs((closes[-1] - closes[0]) / closes[0])
        except Exception as exc:
            logger.warning("%s volatility yfinance fallback failed: %s", self._benchmark, exc)

        return 0.01, 0.01

    def _fetch_recent_window_features(self, window_hours: int) -> tuple[np.ndarray, list] | None:
        """
        Builds the (window_hours, n_features) feature matrix ending "now",
        for detect_regime()'s windowed decode — same vol/abs_change
        preprocessing fit() uses, just over a recent live window instead
        of the full training history. Needs window_hours+24 raw closes
        (each feature row needs its own rolling 24h lookback). Returns
        (X, hour_timestamps) or None on fetch failure — caller falls back
        to the last known regime rather than guessing.
        """
        need = window_hours + 24
        closes: list[float] | None = None

        if settings.ASSET_TYPE == "crypto":
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": self._benchmark, "interval": "1h", "limit": min(need, 1000)},
                    timeout=15,
                )
                resp.raise_for_status()
                closes = [float(k[4]) for k in resp.json()]
            except Exception as exc:
                logger.warning("HMM window fetch: Binance klines failed (%s), trying yfinance", exc)

        if not closes:
            try:
                import yfinance as yf
                df = yf.download(
                    self._benchmark_yf, period=f"{min(need // 24 + 2, 730)}d", interval="1h",
                    progress=False, auto_adjust=True,
                )
                closes = df["Close"].dropna().values.flatten().tolist()
            except Exception as exc:
                logger.warning("HMM window fetch: yfinance fallback failed: %s", exc)
                return None

        if not closes or len(closes) < 25:
            return None

        log_returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        vol = [np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))]
        abs_change = [
            abs((closes[i + 1] - closes[max(0, i + 1 - 24)]) / closes[max(0, i + 1 - 24)])
            for i in range(len(log_returns))
        ]
        X = np.column_stack([vol, abs_change])[-window_hours:]

        # Approximate real hour timestamps for the returned rows — used only
        # to look up per-day Fear & Greed values below, same approximation
        # fit() already makes (contiguous hourly candles ending "now").
        now = datetime.utcnow()
        hour_timestamps = [now - timedelta(hours=len(X) - 1 - i) for i in range(len(X))]
        return X, hour_timestamps

    # ── Training ───────────────────────────────────────────────────────────

    def _fetch_benchmark_closes(self, days: int) -> list[float]:
        """Hourly benchmark closes for `fit()`. Binance REST first (crypto
        only), yfinance fallback otherwise (fit() previously had NO fallback
        at all for crypto, so any transient Binance hiccup silently left
        fit() as a no-op, quietly keeping whatever model — possibly a
        broken one — was already on disk)."""
        if settings.ASSET_TYPE == "crypto":
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": self._benchmark, "interval": "1h", "limit": min(days * 24, 1000)},
                    timeout=15,
                )
                resp.raise_for_status()
                return [float(k[4]) for k in resp.json()]
            except Exception as exc:
                logger.warning("HMM fit: Binance klines fetch failed (%s), trying yfinance fallback", exc)

        import yfinance as yf
        df = yf.download(self._benchmark_yf, period=f"{min(days, 730)}d", interval="1h", progress=False, auto_adjust=True)
        return df["Close"].dropna().values.flatten().tolist()

    def fit(self, client, days: int = 60) -> None:
        """
        Fit HMM on `days` of hourly benchmark data.
        Called once at startup and weekly via scheduler.
        """
        logger.info("Fitting HMM on %d days of %s data...", days, self._benchmark)
        try:
            closes = self._fetch_benchmark_closes(days)
            log_returns = np.array([np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))])

            # Sanity check BEFORE fitting: a bad/garbled API response (e.g. an
            # error body that still happened to parse) previously produced a
            # model with mean_abs_change=1.01 for the "turbulent" state —
            # implying >100% single-hour moves on average, which BTC has never
            # done. A single hourly log-return magnitude above 1.0 (+-171%/-63%)
            # is not a real market event; treat it as corrupted input and bail
            # rather than silently persisting a broken model.
            if len(log_returns) < 48 or not np.all(np.isfinite(log_returns)) or np.max(np.abs(log_returns)) > 1.0:
                logger.error(
                    "HMM fitting aborted: BTC kline data looks corrupted "
                    "(n=%d, max|log_return|=%.4f) — keeping previous model.",
                    len(log_returns), float(np.max(np.abs(log_returns))) if len(log_returns) else float("nan"),
                )
                return

            vol = np.array([np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))])
            # 24h cumulative % change over the SAME rolling window `vol` uses —
            # matches _fetch_benchmark_volatility()'s live feature exactly (that
            # computes abs total move over a 24h window, not a single candle).
            # This used to be np.abs(log_returns) — a single hour's move — so
            # the model was trained on one statistic and fed a completely
            # different one (24h cumulative move, routinely 10-50x larger) at
            # live inference time, guaranteeing live observations looked like
            # extreme outliers even on calm days.
            abs_change = np.array([
                abs((closes[i + 1] - closes[max(0, i + 1 - 24)]) / closes[max(0, i + 1 - 24)])
                for i in range(len(log_returns))
            ])

            # Fear & Greed as a 3rd feature — crypto only (see module
            # docstring). closes has no real timestamps attached (it's a
            # plain list of floats), so approximate: assume the fetch is
            # contiguous hourly candles ending "now" — the same implicit
            # assumption vol/abs_change's positional-index windowing above
            # already relies on, not a new one. Missing days (weekends in
            # the daily F&G feed, gaps, etc.) fall back to neutral (0.5)
            # rather than skewing the fit with a fabricated value.
            fg_feature = None
            if settings.ASSET_TYPE == "crypto":
                fg_history = self._fetch_fear_greed_history(days)
                if fg_history:
                    now = datetime.utcnow()
                    n_closes = len(closes)
                    fg_feature = np.array([
                        fg_history.get((now - timedelta(hours=n_closes - 1 - (i + 1))).date(), 0.5)
                        for i in range(len(log_returns))
                    ])

            # Feature matrix: [rolling 24h volatility, rolling 24h abs %
            # change] (+ Fear & Greed for crypto)
            X = np.column_stack([vol, abs_change, fg_feature]) if fg_feature is not None else np.column_stack([vol, abs_change])

            self._model = hmm.GaussianHMM(
                n_components=settings.HMM_N_STATES,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
            )
            self._model.fit(X)
            self._save()
            logger.info("HMM fitted and saved.")
        except Exception as exc:
            logger.error("HMM fitting failed: %s", exc)

    # ── Inference ──────────────────────────────────────────────────────────

    def detect_regime(self, client) -> int:
        """
        Returns current market regime state (0, 1, or 2).
        State mapping is determined by variance of fitted Gaussians:
          lowest variance → state 0 (low-vol)
          highest variance → state 2 (turbulent)

        Decodes a rolling window of the last HMM_DECODE_WINDOW_HOURS
        observations (Viterbi), not a single point — feeding predict() one
        observation at a time reduces a length-1 sequence's Viterbi to
        argmax(log(startprob) + log(emission)), and startprob_ on a
        Baum-Welch-fitted model tends to concentrate almost all its mass on
        one state (confirmed on the live checkpoint: ~1e-134 to ~1.0 spread
        across states), so single-frame predict() returned that SAME state
        for virtually any input — verified empirically: 0.00% turbulent
        flagged across a 60+ day walk-forward test, vs 23.6% by a
        retrospective full-history decode. A window lets transmat_ and each
        observation's own emission likelihood actually drive the decision
        past its first step. 336h was the smallest window that reached the
        agreement plateau against retrospective ground truth in validation
        (84.7%, same as 500h+), with ~1.6h average detection lag and
        sub-millisecond decode latency — see
        hmm_windowed_decode_experiment.py.
        """
        if self._model is None:
            logger.warning("HMM not fitted; returning NORMAL regime.")
            return 1

        fetched = self._fetch_recent_window_features(settings.HMM_DECODE_WINDOW_HOURS)
        if fetched is None:
            logger.warning("HMM window fetch failed; reusing last known regime.")
            return self._last_state
        X, hour_timestamps = fetched

        # Match whatever the LOADED model was actually trained with, rather
        # than re-deriving from settings.ASSET_TYPE — makes an old 2-feature
        # checkpoint (from before this change) still usable without a shape
        # mismatch until it's refit under the new 3-feature schema.
        if self._model.means_.shape[1] >= 3:
            fg_history = self._fetch_fear_greed_history(settings.HMM_DECODE_WINDOW_HOURS // 24 + 2)
            fg_col = np.array([fg_history.get(ts.date(), 0.5) for ts in hour_timestamps])
            X = np.column_stack([X, fg_col])

        try:
            raw_path = self._model.predict(X)
            raw_state = raw_path[-1]
            # Re-map raw states to ordered volatility regimes
            means = self._model.means_[:, 0]  # mean volatility per state
            order = np.argsort(means)          # 0=lowest vol, 2=highest vol
            ordered_state = int(np.where(order == raw_state)[0][0])
            self._last_state = ordered_state
            logger.debug(
                "HMM regime (windowed, %dh): raw=%d → ordered=%d (vol=%.5f)",
                settings.HMM_DECODE_WINDOW_HOURS, raw_state, ordered_state, float(X[-1, 0]),
            )
            return ordered_state
        except Exception as exc:
            logger.warning("HMM predict error: %s", exc)
            return self._last_state

    @property
    def is_turbulent(self) -> bool:
        return self._last_state == settings.HMM_TURBULENT_STATE
