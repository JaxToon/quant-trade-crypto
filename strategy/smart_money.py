"""
Smart Money Concept (SMC) Indicators
Detects institutional trading patterns: Order Blocks, BOS, ChoCH, Liquidity Zones, MSS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def total_range(self) -> float:
        return self.high - self.low


@dataclass
class OrderBlock:
    """Order Block: last candle before strong move."""
    type: Literal["bullish", "bearish"]
    high: float
    low: float
    timestamp: datetime
    volume: float
    strength: float  # 0-100: based on follow-through momentum
    tested: bool = False
    mitigated: bool = False

    def is_price_in_zone(self, price: float) -> bool:
        """Check if price is within order block zone."""
        return self.low <= price <= self.high


@dataclass
class FairValueGap:
    """Fair Value Gap: imbalance/inefficiency zone."""
    type: Literal["bullish", "bearish"]
    high: float
    low: float
    timestamp: datetime
    mitigated: bool = False
    mitigation_percentage: float = 0.0  # 0-100%

    def is_price_in_gap(self, price: float) -> bool:
        """Check if price is within FVG zone."""
        return self.low <= price <= self.high

    def calculate_mitigation(self, price: float) -> float:
        """Calculate what % of gap has been filled."""
        if self.mitigated:
            return 100.0

        gap_size = self.high - self.low
        if gap_size == 0:
            return 0.0

        if self.type == "bullish":
            # Bullish FVG fills from top down
            if price >= self.high:
                return 0.0
            elif price <= self.low:
                return 100.0
            else:
                filled = self.high - price
                return (filled / gap_size) * 100
        else:
            # Bearish FVG fills from bottom up
            if price <= self.low:
                return 0.0
            elif price >= self.high:
                return 100.0
            else:
                filled = price - self.low
                return (filled / gap_size) * 100


@dataclass
class MarketStructure:
    """Market Structure: Higher Highs, Lower Lows, etc."""
    trend: Literal["bullish", "bearish", "ranging"]
    last_high: float
    last_low: float
    higher_high: float | None = None
    lower_low: float | None = None
    timestamp: datetime = None


@dataclass
class LiquidityZone:
    """Liquidity Zone: areas where stops are likely clustered."""
    type: Literal["buy_side", "sell_side"]  # buy_side = above, sell_side = below
    price: float
    timestamp: datetime
    swept: bool = False
    volume_profile: float = 0.0  # relative volume at this level


class SmartMoneyAnalyzer:
    """
    Analyzes price action for Smart Money Concepts.
    Tracks Order Blocks, FVG, Market Structure, and Liquidity.
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.candles: deque[Candle] = deque(maxlen=lookback)
        self.order_blocks: list[OrderBlock] = []
        self.fvgs: list[FairValueGap] = []
        self.liquidity_zones: list[LiquidityZone] = []
        self.market_structure: MarketStructure | None = None

    def update(self, candle: Candle) -> dict:
        """
        Update with a *closed* candle: appends it to history and re-runs
        pattern detection. Only call this once a candle has actually closed —
        feeding in-progress candles here corrupts OB/FVG/market-structure
        detection with a stream of partial bars.
        """
        self.candles.append(candle)

        if len(self.candles) < 5:
            return self._empty_signals()

        # Detect patterns
        self._detect_order_blocks()
        self._detect_fair_value_gaps()
        self._update_market_structure()
        self._detect_liquidity_zones()

        return self._generate_signals(candle)

    def peek_signals(self, candle: Candle) -> dict:
        """
        Read-only signal check against the current (still-forming) price —
        does NOT append to history or re-run detection. Use this for
        in-progress candles so live price still updates proximity to
        already-detected OBs/FVGs without polluting pattern detection.
        """
        if not self.market_structure:
            return self._empty_signals()
        return self._generate_signals(candle)

    def _detect_order_blocks(self) -> None:
        """
        Detect Order Blocks: last opposing candle before strong move.
        """
        if len(self.candles) < 5:
            return

        candles = list(self.candles)

        for i in range(2, len(candles) - 2):
            prev = candles[i - 1]
            curr = candles[i]
            next1 = candles[i + 1]
            next2 = candles[i + 2]

            # Bullish OB: bearish candle followed by strong bullish move
            if (curr.is_bearish and
                next1.is_bullish and next2.is_bullish and
                next1.close > curr.high):

                strength = self._calculate_ob_strength(curr, [next1, next2])

                ob = OrderBlock(
                    type="bullish",
                    high=curr.high,
                    low=curr.low,
                    timestamp=curr.timestamp,
                    volume=curr.volume,
                    strength=strength
                )

                # Only add if not duplicate
                if not self._is_duplicate_ob(ob):
                    self.order_blocks.append(ob)
                    logger.debug("Bullish OB detected at %.2f-%.2f", ob.low, ob.high)

            # Bearish OB: bullish candle followed by strong bearish move
            elif (curr.is_bullish and
                  next1.is_bearish and next2.is_bearish and
                  next1.close < curr.low):

                strength = self._calculate_ob_strength(curr, [next1, next2])

                ob = OrderBlock(
                    type="bearish",
                    high=curr.high,
                    low=curr.low,
                    timestamp=curr.timestamp,
                    volume=curr.volume,
                    strength=strength
                )

                if not self._is_duplicate_ob(ob):
                    self.order_blocks.append(ob)
                    logger.debug("Bearish OB detected at %.2f-%.2f", ob.low, ob.high)

        # Cleanup: remove old or mitigated OBs
        self.order_blocks = [ob for ob in self.order_blocks if not ob.mitigated][-20:]

    def _detect_fair_value_gaps(self) -> None:
        """
        Detect Fair Value Gaps (FVG): 3-candle imbalance.
        Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
        Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
        """
        if len(self.candles) < 3:
            return

        candles = list(self.candles)

        for i in range(1, len(candles) - 1):
            prev = candles[i - 1]
            curr = candles[i]
            next_candle = candles[i + 1]

            # Bullish FVG (gap up)
            if prev.high < next_candle.low:
                fvg = FairValueGap(
                    type="bullish",
                    high=next_candle.low,
                    low=prev.high,
                    timestamp=curr.timestamp
                )

                if not self._is_duplicate_fvg(fvg):
                    self.fvgs.append(fvg)
                    logger.debug("Bullish FVG detected: %.2f-%.2f", fvg.low, fvg.high)

            # Bearish FVG (gap down)
            elif prev.low > next_candle.high:
                fvg = FairValueGap(
                    type="bearish",
                    high=prev.low,
                    low=next_candle.high,
                    timestamp=curr.timestamp
                )

                if not self._is_duplicate_fvg(fvg):
                    self.fvgs.append(fvg)
                    logger.debug("Bearish FVG detected: %.2f-%.2f", fvg.low, fvg.high)

        # Update mitigation status
        current_price = candles[-1].close
        for fvg in self.fvgs:
            fvg.mitigation_percentage = fvg.calculate_mitigation(current_price)
            if fvg.mitigation_percentage >= 50:  # 50% mitigation = filled
                fvg.mitigated = True

        # Cleanup: remove fully mitigated FVGs
        self.fvgs = [fvg for fvg in self.fvgs if not fvg.mitigated][-20:]

    def _update_market_structure(self) -> None:
        """
        Update market structure: identify swing highs/lows and trend.
        BOS (Break of Structure) and ChoCH (Change of Character) detection.
        """
        if len(self.candles) < 10:
            return

        candles = list(self.candles)
        highs = [c.high for c in candles[-10:]]
        lows = [c.low for c in candles[-10:]]

        recent_high = max(highs)
        recent_low = min(lows)

        # Simple trend detection
        if self.market_structure is None:
            self.market_structure = MarketStructure(
                trend="ranging",
                last_high=recent_high,
                last_low=recent_low,
                timestamp=candles[-1].timestamp
            )
        else:
            ms = self.market_structure

            # Check for Higher High (HH)
            if recent_high > ms.last_high:
                ms.higher_high = recent_high
                if ms.trend != "bullish":
                    logger.info("ChoCH: Market structure changed to BULLISH")
                ms.trend = "bullish"
                ms.last_high = recent_high

            # Check for Lower Low (LL)
            elif recent_low < ms.last_low:
                ms.lower_low = recent_low
                if ms.trend != "bearish":
                    logger.info("ChoCH: Market structure changed to BEARISH")
                ms.trend = "bearish"
                ms.last_low = recent_low

            ms.timestamp = candles[-1].timestamp

    def _detect_liquidity_zones(self) -> None:
        """
        Detect liquidity zones: swing highs (buy-side) and swing lows (sell-side).
        These are areas where stop losses are likely clustered.
        """
        if len(self.candles) < 7:
            return

        candles = list(self.candles)

        for i in range(3, len(candles) - 3):
            curr = candles[i]

            # Swing High: local high with lower highs on both sides
            is_swing_high = (
                curr.high > candles[i-1].high and
                curr.high > candles[i-2].high and
                curr.high > candles[i+1].high and
                curr.high > candles[i+2].high
            )

            if is_swing_high:
                liq = LiquidityZone(
                    type="buy_side",
                    price=curr.high,
                    timestamp=curr.timestamp,
                    volume_profile=curr.volume
                )
                if not self._is_duplicate_liquidity(liq):
                    self.liquidity_zones.append(liq)

            # Swing Low: local low with higher lows on both sides
            is_swing_low = (
                curr.low < candles[i-1].low and
                curr.low < candles[i-2].low and
                curr.low < candles[i+1].low and
                curr.low < candles[i+2].low
            )

            if is_swing_low:
                liq = LiquidityZone(
                    type="sell_side",
                    price=curr.low,
                    timestamp=curr.timestamp,
                    volume_profile=curr.volume
                )
                if not self._is_duplicate_liquidity(liq):
                    self.liquidity_zones.append(liq)

        # Check for swept liquidity
        current_price = candles[-1].close
        for liq in self.liquidity_zones:
            if not liq.swept:
                if liq.type == "buy_side" and current_price > liq.price:
                    liq.swept = True
                    logger.info("Buy-side liquidity SWEPT at %.2f", liq.price)
                elif liq.type == "sell_side" and current_price < liq.price:
                    liq.swept = True
                    logger.info("Sell-side liquidity SWEPT at %.2f", liq.price)

        # Cleanup: keep swept zones around (capped) instead of purging them
        # immediately — otherwise the same historical swing point gets
        # rediscovered as "new" on every subsequent call (since the dedup
        # check in _is_duplicate_liquidity only sees what's still in this
        # list) and gets re-logged as SWEPT again and again.
        self.liquidity_zones = self.liquidity_zones[-40:]

    def _calculate_ob_strength(self, ob_candle: Candle, follow_candles: list[Candle]) -> float:
        """
        Calculate order block strength (0-100) based on:
        - Volume of OB candle
        - Momentum of follow-through move
        - Size of OB zone
        """
        if not follow_candles:
            return 50.0

        # Volume strength (some data sources — e.g. yfinance crypto — report
        # zero volume on certain bars; treat that as "no signal" rather than
        # dividing by zero)
        max_follow_volume = max((c.volume for c in follow_candles), default=0)
        volume_strength = min(100, ob_candle.volume / max_follow_volume * 100) if max_follow_volume > 0 else 50

        # Momentum strength (how far price moved after OB)
        ob_range = ob_candle.total_range
        follow_range = sum(c.total_range for c in follow_candles)
        momentum_strength = min(100, (follow_range / ob_range) * 20) if ob_range > 0 else 50

        # Weighted average
        strength = (volume_strength * 0.4) + (momentum_strength * 0.6)
        return round(strength, 2)

    def _is_duplicate_ob(self, new_ob: OrderBlock) -> bool:
        """Check if order block already exists (similar price level)."""
        for ob in self.order_blocks:
            if ob.type == new_ob.type:
                overlap = min(ob.high, new_ob.high) - max(ob.low, new_ob.low)
                if overlap > 0:
                    return True
        return False

    def _is_duplicate_fvg(self, new_fvg: FairValueGap) -> bool:
        """Check if FVG already exists."""
        for fvg in self.fvgs:
            if fvg.type == new_fvg.type:
                overlap = min(fvg.high, new_fvg.high) - max(fvg.low, new_fvg.low)
                if overlap > 0:
                    return True
        return False

    def _is_duplicate_liquidity(self, new_liq: LiquidityZone) -> bool:
        """Check if liquidity zone already exists."""
        for liq in self.liquidity_zones:
            if liq.type == new_liq.type and abs(liq.price - new_liq.price) < (new_liq.price * 0.001):
                return True
        return False

    def _generate_signals(self, current_candle: Candle) -> dict:
        """
        Generate SMC trading signals based on current market state.
        """
        price = current_candle.close

        # Check proximity to Order Blocks
        nearest_bullish_ob = None
        nearest_bearish_ob = None

        for ob in self.order_blocks:
            if ob.type == "bullish" and ob.is_price_in_zone(price):
                nearest_bullish_ob = ob
            elif ob.type == "bearish" and ob.is_price_in_zone(price):
                nearest_bearish_ob = ob

        # Check proximity to FVGs
        nearest_bullish_fvg = None
        nearest_bearish_fvg = None

        for fvg in self.fvgs:
            if fvg.type == "bullish" and fvg.is_price_in_gap(price):
                nearest_bullish_fvg = fvg
            elif fvg.type == "bearish" and fvg.is_price_in_gap(price):
                nearest_bearish_fvg = fvg

        # Market structure signal
        ms_signal = 0.0
        if self.market_structure:
            if self.market_structure.trend == "bullish":
                ms_signal = 0.3
            elif self.market_structure.trend == "bearish":
                ms_signal = -0.3

        # Order Block signal
        ob_signal = 0.0
        if nearest_bullish_ob:
            ob_signal = 0.4 * (nearest_bullish_ob.strength / 100)
        elif nearest_bearish_ob:
            ob_signal = -0.4 * (nearest_bearish_ob.strength / 100)

        # FVG signal
        fvg_signal = 0.0
        if nearest_bullish_fvg and nearest_bullish_fvg.mitigation_percentage < 80:
            fvg_signal = 0.3 * (1 - nearest_bullish_fvg.mitigation_percentage / 100)
        elif nearest_bearish_fvg and nearest_bearish_fvg.mitigation_percentage < 80:
            fvg_signal = -0.3 * (1 - nearest_bearish_fvg.mitigation_percentage / 100)

        # Combined SMC signal
        smc_signal = ms_signal + ob_signal + fvg_signal
        smc_signal = max(-1.0, min(1.0, smc_signal))  # Clamp to [-1, 1]

        return {
            "smc_signal": round(smc_signal, 4),
            "market_structure": self.market_structure.trend if self.market_structure else "unknown",
            "active_order_blocks": len([ob for ob in self.order_blocks if not ob.mitigated]),
            "active_fvgs": len([fvg for fvg in self.fvgs if not fvg.mitigated]),
            "liquidity_zones": len([liq for liq in self.liquidity_zones if not liq.swept]),
            "in_bullish_ob": nearest_bullish_ob is not None,
            "in_bearish_ob": nearest_bearish_ob is not None,
            "in_bullish_fvg": nearest_bullish_fvg is not None,
            "in_bearish_fvg": nearest_bearish_fvg is not None,
            "atr": self._compute_atr(),
        }

    def _compute_atr(self, period: int = 14) -> float:
        """
        Average True Range over the last `period` candles, reusing the
        candle history already maintained for OB/FVG/market-structure
        detection — no separate data feed needed. True Range per candle is
        max(high-low, |high-prev_close|, |low-prev_close|), which (unlike
        Candle.total_range) also captures gap moves between candles, not
        just each candle's own high-low spread.

        Returns 0.0 with fewer than 2 candles (nothing to compute a range
        from yet) — callers should treat 0.0 as "ATR not yet available",
        same convention as the other signals in _empty_signals().
        """
        candles = list(self.candles)
        if len(candles) < 2:
            return 0.0
        window = candles[-(period + 1):]
        true_ranges = []
        for i in range(1, len(window)):
            high, low, prev_close = window[i].high, window[i].low, window[i - 1].close
            true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        return round(sum(true_ranges) / len(true_ranges), 8) if true_ranges else 0.0

    def _empty_signals(self) -> dict:
        """Return empty signals when not enough data."""
        return {
            "smc_signal": 0.0,
            "market_structure": "unknown",
            "active_order_blocks": 0,
            "active_fvgs": 0,
            "liquidity_zones": 0,
            "in_bullish_ob": False,
            "in_bearish_ob": False,
            "in_bullish_fvg": False,
            "in_bearish_fvg": False,
            "atr": self._compute_atr(),
        }
