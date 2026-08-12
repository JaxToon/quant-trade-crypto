"""
HMM-based circuit breaker.
In HIGH_STRESS_TURBULENT regime:
  - Slash risk limits by 70%
  - Grant Bear Agent absolute veto power
"""
from __future__ import annotations

import logging

from config import settings
from risk.hmm_regime import RegimeDetector

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "LOW_VOLATILITY", 1: "NORMAL", 2: "HIGH_STRESS_TURBULENT"}


class CircuitBreaker:

    def __init__(self, regime_detector: RegimeDetector) -> None:
        self._detector = regime_detector
        self._current_regime: int = 1

    def update(self, binance_client) -> int:
        """Re-evaluate market regime. Call this on each major tick cycle."""
        self._current_regime = self._detector.detect_regime(binance_client)
        label = REGIME_LABELS.get(self._current_regime, "UNKNOWN")
        if self._current_regime == settings.HMM_TURBULENT_STATE:
            logger.warning(
                "REGIME: %s — risk reduced by %.0f%%, Bear veto active",
                label, settings.HMM_RISK_REDUCTION * 100,
            )
        else:
            logger.info("REGIME: %s", label)
        return self._current_regime

    @property
    def bear_veto_active(self) -> bool:
        """True when Bear Agent has absolute veto (turbulent regime)."""
        return self._current_regime == settings.HMM_TURBULENT_STATE

    @property
    def effective_max_position_pct(self) -> float:
        """Position size limit adjusted for current regime."""
        if self._current_regime == settings.HMM_TURBULENT_STATE:
            return settings.MAX_POSITION_PCT * (1 - settings.HMM_RISK_REDUCTION)
        return settings.MAX_POSITION_PCT

    @property
    def regime_label(self) -> str:
        return REGIME_LABELS.get(self._current_regime, "UNKNOWN")
