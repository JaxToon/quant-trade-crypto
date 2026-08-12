"""
Blue Chip Stock/Crypto Filter
Identifies high-quality assets based on market cap, liquidity, and stability metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class BlueChipCriteria:
    """Criteria for blue chip classification."""
    min_market_cap: float  # USD
    min_avg_volume_24h: float  # USD
    max_volatility: float  # percentage (0-100)
    min_age_days: int  # minimum days since launch
    asset_type: Literal["crypto", "stock"]


# Predefined criteria sets
CRYPTO_BLUE_CHIP = BlueChipCriteria(
    min_market_cap=10_000_000_000,  # $10B market cap
    min_avg_volume_24h=1_000_000_000,  # $1B daily volume
    max_volatility=15.0,  # 15% max daily volatility
    min_age_days=365 * 2,  # 2 years minimum
    asset_type="crypto"
)

STOCK_BLUE_CHIP = BlueChipCriteria(
    min_market_cap=50_000_000_000,  # $50B market cap
    min_avg_volume_24h=500_000_000,  # $500M daily volume
    max_volatility=5.0,  # 5% max daily volatility
    min_age_days=365 * 5,  # 5 years minimum
    asset_type="stock"
)


@dataclass
class AssetMetrics:
    """Market metrics for an asset."""
    symbol: str
    market_cap: float
    avg_volume_24h: float
    volatility_30d: float
    age_days: int
    price: float
    change_24h: float

    # Additional metrics
    liquidity_score: float = 0.0  # 0-100
    stability_score: float = 0.0  # 0-100
    quality_score: float = 0.0  # 0-100


class BlueChipFilter:
    """
    Filter and rank assets based on blue chip criteria.
    Integrates with knowledge graph for fundamental analysis.
    """

    def __init__(self, criteria: BlueChipCriteria = CRYPTO_BLUE_CHIP):
        self.criteria = criteria
        logger.info(
            "BlueChipFilter initialized: %s (min_mcap=$%.2fB, min_vol=$%.2fB)",
            criteria.asset_type,
            criteria.min_market_cap / 1e9,
            criteria.min_avg_volume_24h / 1e9
        )

    def is_blue_chip(self, metrics: AssetMetrics) -> bool:
        """Check if asset meets blue chip criteria."""
        checks = {
            "market_cap": metrics.market_cap >= self.criteria.min_market_cap,
            "volume": metrics.avg_volume_24h >= self.criteria.min_avg_volume_24h,
            "volatility": metrics.volatility_30d <= self.criteria.max_volatility,
            "age": metrics.age_days >= self.criteria.min_age_days,
        }

        passed = all(checks.values())

        if not passed:
            failed = [k for k, v in checks.items() if not v]
            logger.debug(
                "%s failed blue chip criteria: %s",
                metrics.symbol,
                ", ".join(failed)
            )

        return passed

    def calculate_quality_score(self, metrics: AssetMetrics) -> float:
        """
        Calculate overall quality score (0-100).
        Higher score = better blue chip quality.
        """
        # Liquidity score (based on volume relative to market cap)
        volume_ratio = metrics.avg_volume_24h / metrics.market_cap if metrics.market_cap > 0 else 0
        liquidity_score = min(100, volume_ratio * 100 * 30)  # Scale to 0-100

        # Stability score (inverse of volatility)
        stability_score = max(0, 100 - (metrics.volatility_30d * 5))

        # Age score
        age_score = min(100, (metrics.age_days / 365) * 10)  # Scale years to 0-100

        # Market cap score (log scale)
        import math
        mcap_score = min(100, math.log10(metrics.market_cap) * 10)

        # Weighted average
        quality_score = (
            liquidity_score * 0.30 +
            stability_score * 0.30 +
            age_score * 0.20 +
            mcap_score * 0.20
        )

        return round(quality_score, 2)

    def rank_assets(self, assets: list[AssetMetrics]) -> list[AssetMetrics]:
        """
        Rank assets by quality score.
        Returns sorted list (highest quality first).
        """
        for asset in assets:
            asset.quality_score = self.calculate_quality_score(asset)

        return sorted(assets, key=lambda x: x.quality_score, reverse=True)

    def filter_and_rank(self, assets: list[AssetMetrics]) -> list[AssetMetrics]:
        """
        Filter to blue chips only, then rank by quality.
        """
        blue_chips = [a for a in assets if self.is_blue_chip(a)]
        return self.rank_assets(blue_chips)


# Default watchlists
CRYPTO_BLUE_CHIPS = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "BNBUSDT",   # Binance Coin
    "SOLUSDT",   # Solana
    "XRPUSDT",   # Ripple
    "ADAUSDT",   # Cardano
    "AVAXUSDT",  # Avalanche
    "DOTUSDT",   # Polkadot
]

NASDAQ_BLUE_CHIPS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta
    "TSLA",   # Tesla
    "AVGO",   # Broadcom
]


def get_default_watchlist(asset_type: Literal["crypto", "stock"] = "crypto") -> list[str]:
    """Get default blue chip watchlist for asset type."""
    if asset_type == "crypto":
        return CRYPTO_BLUE_CHIPS
    elif asset_type == "stock":
        return NASDAQ_BLUE_CHIPS
    else:
        return CRYPTO_BLUE_CHIPS
