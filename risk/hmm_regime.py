"""
Hidden Markov Model regime classifier.
States: 0=LOW_VOLATILITY, 1=NORMAL, 2=HIGH_STRESS_TURBULENT

Features:
  - BTC 24h volatility (std of log returns)
  - BTC 24h absolute price change %
  - Fear & Greed Index (free alternative.me API)

Fitted on recent Binance kline history; refitted weekly.
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

HMM_CHECKPOINT = "model_checkpoints/hmm_regime.pkl"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


class RegimeDetector:

    def __init__(self) -> None:
        self._model: Optional[hmm.GaussianHMM] = None
        self._last_state: int = 1  # default: NORMAL
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

    def _fetch_btc_volatility(self, client=None) -> tuple[float, float]:
        """Returns (24h_vol, 24h_change_pct). Tries Binance REST, falls back to yfinance."""
        # Try Binance public REST first
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1h", "limit": 25},
                timeout=5,
            )
            resp.raise_for_status()
            closes = [float(k[4]) for k in resp.json()]
            log_returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
            return float(np.std(log_returns)), abs((closes[-1] - closes[0]) / closes[0])
        except Exception:
            pass

        # Fallback: yfinance (works globally)
        try:
            import yfinance as yf
            df = yf.download("BTC-USD", period="2d", interval="1h", progress=False, auto_adjust=True)
            closes = df["Close"].dropna().values.flatten().tolist()
            if len(closes) >= 2:
                log_returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
                return float(np.std(log_returns)), abs((closes[-1] - closes[0]) / closes[0])
        except Exception as exc:
            logger.warning("BTC volatility yfinance fallback failed: %s", exc)

        return 0.01, 0.01

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, client, days: int = 60) -> None:
        """
        Fit HMM on `days` of hourly BTC data.
        Called once at startup and weekly via scheduler.
        """
        logger.info("Fitting HMM on %d days of BTC data...", days)
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1h", "limit": min(days * 24, 1000)},
                timeout=15,
            )
            resp.raise_for_status()
            klines = resp.json()
            closes = [float(k[4]) for k in klines]
            log_returns = np.array([np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))])
            vol = np.array([np.std(log_returns[max(0, i - 23):i + 1]) for i in range(len(log_returns))])
            abs_change = np.abs(log_returns)
            # Feature matrix: [volatility, |return|]
            X = np.column_stack([vol, abs_change])

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
        """
        if self._model is None:
            logger.warning("HMM not fitted; returning NORMAL regime.")
            return 1

        vol, change = self._fetch_btc_volatility(client)
        fg = self._fetch_fear_greed()
        # Use vol + change as features (fear_greed can be added as 3rd dim if desired)
        obs = np.array([[vol, change]])
        try:
            raw_state = self._model.predict(obs)[0]
            # Re-map raw states to ordered volatility regimes
            means = self._model.means_[:, 0]  # mean volatility per state
            order = np.argsort(means)          # 0=lowest vol, 2=highest vol
            ordered_state = int(np.where(order == raw_state)[0][0])
            self._last_state = ordered_state
            logger.debug("HMM regime: raw=%d → ordered=%d (vol=%.5f)", raw_state, ordered_state, vol)
            return ordered_state
        except Exception as exc:
            logger.warning("HMM predict error: %s", exc)
            return self._last_state

    @property
    def is_turbulent(self) -> bool:
        return self._last_state == settings.HMM_TURBULENT_STATE
