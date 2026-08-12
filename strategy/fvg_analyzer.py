"""
Fair Value Gap (FVG) Advanced Analysis
Enhanced FVG detection with mitigation tracking, confluence zones, and trading signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class FVGZone:
    """
    Fair Value Gap Zone with advanced tracking.
    """
    id: str
    type: Literal["bullish", "bearish"]
    high: float
    low: float
    created_at: datetime

    # Candle references
    candle_before_high: float
    candle_before_low: float
    candle_after_high: float
    candle_after_low: float

    # Gap metrics
    gap_size: float
    gap_size_percent: float

    # Mitigation tracking
    mitigated: bool = False
    mitigation_percentage: float = 0.0
    mitigation_time: datetime | None = None
    touches: int = 0

    # Trading metrics
    respect_count: int = 0  # How many times price respected this FVG
    volume_at_creation: float = 0.0
    strength_score: float = 0.0  # 0-100

    # Confluence factors
    near_order_block: bool = False
    near_liquidity_zone: bool = False
    at_premium_discount: Literal["premium", "discount", "equilibrium"] | None = None

    # Status
    active: bool = True
    invalidated: bool = False

    def update_mitigation(self, current_price: float) -> None:
        """Update mitigation status based on current price."""
        if self.mitigated or self.invalidated:
            return

        in_gap = self.low <= current_price <= self.high

        if in_gap:
            self.touches += 1

        # Calculate mitigation percentage
        if self.type == "bullish":
            # Bullish FVG fills from top down
            if current_price >= self.high:
                self.mitigation_percentage = 0.0
            elif current_price <= self.low:
                self.mitigation_percentage = 100.0
                self.mitigated = True
                self.mitigation_time = datetime.utcnow()
            else:
                filled = self.high - current_price
                self.mitigation_percentage = (filled / self.gap_size) * 100

        else:  # bearish
            # Bearish FVG fills from bottom up
            if current_price <= self.low:
                self.mitigation_percentage = 0.0
            elif current_price >= self.high:
                self.mitigation_percentage = 100.0
                self.mitigated = True
                self.mitigation_time = datetime.utcnow()
            else:
                filled = current_price - self.low
                self.mitigation_percentage = (filled / self.gap_size) * 100

        # Partial mitigation threshold (50%)
        if self.mitigation_percentage >= 50:
            self.respect_count += 1

    def is_valid_entry(self, current_price: float, min_gap_remaining: float = 0.5) -> bool:
        """
        Check if FVG is valid for entry.
        min_gap_remaining: minimum percentage of gap that should be unfilled (0-1)
        """
        if self.invalidated or self.mitigated:
            return False

        remaining = 1 - (self.mitigation_percentage / 100)
        if remaining < min_gap_remaining:
            return False

        return self.low <= current_price <= self.high

    def calculate_target(self, entry_price: float) -> float:
        """Calculate target price based on FVG type and entry."""
        if self.type == "bullish":
            # Target is opposite end of gap
            return self.high + (self.gap_size * 0.5)  # Add 50% of gap as target
        else:
            return self.low - (self.gap_size * 0.5)

    def calculate_stop_loss(self, entry_price: float) -> float:
        """Calculate stop loss (invalidation point)."""
        if self.type == "bullish":
            # Stop below the gap
            return self.low - (self.gap_size * 0.2)
        else:
            # Stop above the gap
            return self.high + (self.gap_size * 0.2)


@dataclass
class FVGCluster:
    """
    Cluster of multiple FVGs in proximity (confluence zone).
    Strong institutional interest area.
    """
    fvgs: list[FVGZone] = field(default_factory=list)
    type: Literal["bullish", "bearish"] | None = None
    high: float = 0.0
    low: float = 0.0
    strength: float = 0.0

    def add_fvg(self, fvg: FVGZone) -> None:
        """Add FVG to cluster."""
        self.fvgs.append(fvg)
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate cluster boundaries and strength."""
        if not self.fvgs:
            return

        self.high = max(fvg.high for fvg in self.fvgs)
        self.low = min(fvg.low for fvg in self.fvgs)

        # Determine dominant type
        bullish_count = sum(1 for fvg in self.fvgs if fvg.type == "bullish")
        bearish_count = len(self.fvgs) - bullish_count
        self.type = "bullish" if bullish_count > bearish_count else "bearish"

        # Strength based on number of FVGs and their individual strength
        self.strength = sum(fvg.strength_score for fvg in self.fvgs) / len(self.fvgs)


class FairValueGapAnalyzer:
    """
    Advanced Fair Value Gap analyzer with mitigation tracking,
    confluence detection, and trade signal generation.
    """

    def __init__(self, max_fvgs: int = 50, cluster_proximity: float = 0.02):
        """
        Args:
            max_fvgs: Maximum number of active FVGs to track
            cluster_proximity: Price distance threshold for clustering (2% default)
        """
        self.max_fvgs = max_fvgs
        self.cluster_proximity = cluster_proximity

        self.fvgs: deque[FVGZone] = deque(maxlen=max_fvgs)
        self.clusters: list[FVGCluster] = []

        self._fvg_counter = 0

    def detect_fvg(
        self,
        candles: list,
        index: int,
        current_price: float,
        volume: float
    ) -> FVGZone | None:
        """
        Detect FVG at given candle index (requires 3 candles: i-1, i, i+1).

        Args:
            candles: List of candle objects with open, high, low, close
            index: Current candle index (must be >= 1 and < len-1)
            current_price: Current market price
            volume: Volume of middle candle
        """
        if index < 1 or index >= len(candles) - 1:
            return None

        prev = candles[index - 1]
        curr = candles[index]
        next_candle = candles[index + 1]

        fvg = None

        # Bullish FVG: prev.high < next.low (gap up with no trading in between)
        if prev.high < next_candle.low:
            gap_size = next_candle.low - prev.high
            gap_percent = (gap_size / prev.high) * 100

            # Filter out tiny gaps (< 0.1%)
            if gap_percent < 0.1:
                return None

            self._fvg_counter += 1
            fvg = FVGZone(
                id=f"FVG_{self._fvg_counter}",
                type="bullish",
                high=next_candle.low,
                low=prev.high,
                created_at=datetime.utcnow(),
                candle_before_high=prev.high,
                candle_before_low=prev.low,
                candle_after_high=next_candle.high,
                candle_after_low=next_candle.low,
                gap_size=gap_size,
                gap_size_percent=gap_percent,
                volume_at_creation=volume,
                strength_score=self._calculate_fvg_strength(
                    gap_percent, volume, "bullish", candles[index-2:index+2]
                )
            )

            logger.info(
                "Bullish FVG detected: %.2f-%.2f (%.2f%%, strength=%.1f)",
                fvg.low, fvg.high, gap_percent, fvg.strength_score
            )

        # Bearish FVG: prev.low > next.high (gap down)
        elif prev.low > next_candle.high:
            gap_size = prev.low - next_candle.high
            gap_percent = (gap_size / prev.low) * 100

            if gap_percent < 0.1:
                return None

            self._fvg_counter += 1
            fvg = FVGZone(
                id=f"FVG_{self._fvg_counter}",
                type="bearish",
                high=prev.low,
                low=next_candle.high,
                created_at=datetime.utcnow(),
                candle_before_high=prev.high,
                candle_before_low=prev.low,
                candle_after_high=next_candle.high,
                candle_after_low=next_candle.low,
                gap_size=gap_size,
                gap_size_percent=gap_percent,
                volume_at_creation=volume,
                strength_score=self._calculate_fvg_strength(
                    gap_percent, volume, "bearish", candles[index-2:index+2]
                )
            )

            logger.info(
                "Bearish FVG detected: %.2f-%.2f (%.2f%%, strength=%.1f)",
                fvg.low, fvg.high, gap_percent, fvg.strength_score
            )

        if fvg:
            self.fvgs.append(fvg)
            self._update_clusters()

        return fvg

    def update_all(self, current_price: float) -> None:
        """Update mitigation status for all active FVGs."""
        for fvg in self.fvgs:
            if fvg.active and not fvg.mitigated:
                fvg.update_mitigation(current_price)

    def get_nearest_fvg(
        self,
        current_price: float,
        fvg_type: Literal["bullish", "bearish"] | None = None,
        max_distance_percent: float = 2.0
    ) -> FVGZone | None:
        """
        Get nearest active FVG to current price.

        Args:
            current_price: Current market price
            fvg_type: Filter by FVG type (None = any)
            max_distance_percent: Maximum distance as percentage of price
        """
        nearest = None
        min_distance = float('inf')

        for fvg in self.fvgs:
            if not fvg.active or fvg.mitigated:
                continue

            if fvg_type and fvg.type != fvg_type:
                continue

            # Calculate distance to FVG zone
            if fvg.low <= current_price <= fvg.high:
                distance = 0  # Inside the gap
            elif current_price < fvg.low:
                distance = fvg.low - current_price
            else:
                distance = current_price - fvg.high

            distance_percent = (distance / current_price) * 100

            if distance_percent <= max_distance_percent and distance < min_distance:
                min_distance = distance
                nearest = fvg

        return nearest

    def get_trading_signal(self, current_price: float, trend: str = "unknown") -> dict:
        """
        Generate trading signal based on FVG analysis.

        Returns:
            dict with signal strength, direction, entry, stop, target
        """
        # Find nearest FVGs
        nearest_bull_fvg = self.get_nearest_fvg(current_price, "bullish")
        nearest_bear_fvg = self.get_nearest_fvg(current_price, "bearish")

        signal = {
            "fvg_signal": 0.0,
            "direction": "neutral",
            "confidence": 0.0,
            "entry_zone": None,
            "stop_loss": None,
            "target": None,
            "fvg_context": ""
        }

        # Bullish FVG signal
        if nearest_bull_fvg and nearest_bull_fvg.is_valid_entry(current_price):
            # Price in bullish FVG = potential long
            strength = nearest_bull_fvg.strength_score / 100
            confluence_bonus = 0.2 if (nearest_bull_fvg.near_order_block or
                                       nearest_bull_fvg.near_liquidity_zone) else 0
            trend_bonus = 0.2 if trend == "bullish" else 0

            signal_strength = min(1.0, strength + confluence_bonus + trend_bonus)

            signal.update({
                "fvg_signal": signal_strength,
                "direction": "long",
                "confidence": signal_strength * 100,
                "entry_zone": (nearest_bull_fvg.low, nearest_bull_fvg.high),
                "stop_loss": nearest_bull_fvg.calculate_stop_loss(current_price),
                "target": nearest_bull_fvg.calculate_target(current_price),
                "fvg_context": f"Bullish FVG {nearest_bull_fvg.id} (unfilled: {100-nearest_bull_fvg.mitigation_percentage:.1f}%)"
            })

        # Bearish FVG signal
        elif nearest_bear_fvg and nearest_bear_fvg.is_valid_entry(current_price):
            # Price in bearish FVG = potential short
            strength = nearest_bear_fvg.strength_score / 100
            confluence_bonus = 0.2 if (nearest_bear_fvg.near_order_block or
                                       nearest_bear_fvg.near_liquidity_zone) else 0
            trend_bonus = 0.2 if trend == "bearish" else 0

            signal_strength = min(1.0, strength + confluence_bonus + trend_bonus)

            signal.update({
                "fvg_signal": -signal_strength,
                "direction": "short",
                "confidence": signal_strength * 100,
                "entry_zone": (nearest_bear_fvg.low, nearest_bear_fvg.high),
                "stop_loss": nearest_bear_fvg.calculate_stop_loss(current_price),
                "target": nearest_bear_fvg.calculate_target(current_price),
                "fvg_context": f"Bearish FVG {nearest_bear_fvg.id} (unfilled: {100-nearest_bear_fvg.mitigation_percentage:.1f}%)"
            })

        return signal

    def _calculate_fvg_strength(
        self,
        gap_percent: float,
        volume: float,
        fvg_type: str,
        surrounding_candles: list
    ) -> float:
        """
        Calculate FVG strength score (0-100).

        Factors:
        - Gap size (larger = stronger)
        - Volume (higher = stronger)
        - Momentum of move creating gap
        """
        # Gap size component (0-40 points)
        gap_score = min(40, gap_percent * 10)

        # Volume component (0-30 points)
        if surrounding_candles:
            avg_volume = sum(getattr(c, 'volume', 0) for c in surrounding_candles) / len(surrounding_candles)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            volume_score = min(30, volume_ratio * 15)
        else:
            volume_score = 15

        # Momentum component (0-30 points)
        if len(surrounding_candles) >= 3:
            price_move = abs(surrounding_candles[-1].close - surrounding_candles[0].close)
            avg_price = (surrounding_candles[-1].close + surrounding_candles[0].close) / 2
            momentum_percent = (price_move / avg_price) * 100 if avg_price > 0 else 0
            momentum_score = min(30, momentum_percent * 5)
        else:
            momentum_score = 15

        total_score = gap_score + volume_score + momentum_score
        return round(total_score, 2)

    def _update_clusters(self) -> None:
        """Identify and update FVG clusters (confluence zones)."""
        self.clusters = []

        active_fvgs = [fvg for fvg in self.fvgs if fvg.active and not fvg.mitigated]
        if len(active_fvgs) < 2:
            return

        # Group nearby FVGs
        visited = set()

        for i, fvg1 in enumerate(active_fvgs):
            if i in visited:
                continue

            cluster = FVGCluster()
            cluster.add_fvg(fvg1)
            visited.add(i)

            # Find nearby FVGs
            for j, fvg2 in enumerate(active_fvgs[i+1:], start=i+1):
                if j in visited:
                    continue

                # Check if FVGs are close
                avg_price = (fvg1.high + fvg1.low + fvg2.high + fvg2.low) / 4
                distance = abs(fvg1.low - fvg2.high) if fvg1.low > fvg2.high else abs(fvg2.low - fvg1.high)
                distance_percent = (distance / avg_price) * 100

                if distance_percent <= self.cluster_proximity:
                    cluster.add_fvg(fvg2)
                    visited.add(j)

            # Only keep clusters with 2+ FVGs
            if len(cluster.fvgs) >= 2:
                self.clusters.append(cluster)
                logger.info(
                    "FVG Cluster detected: %d FVGs at %.2f-%.2f (strength=%.1f)",
                    len(cluster.fvgs), cluster.low, cluster.high, cluster.strength
                )

    def cleanup_old_fvgs(self, max_age_hours: int = 168) -> None:
        """Remove FVGs older than max_age_hours (default 1 week)."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        before_count = len(self.fvgs)
        self.fvgs = deque(
            (fvg for fvg in self.fvgs if fvg.created_at > cutoff),
            maxlen=self.max_fvgs
        )

        removed = before_count - len(self.fvgs)
        if removed > 0:
            logger.info("Cleaned up %d old FVGs (older than %dh)", removed, max_age_hours)
