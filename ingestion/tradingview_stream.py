"""
TradingView WebSocket stream for real-time market data.
Supports both crypto and stock markets with configurable timeframes.
"""
from __future__ import annotations

import asyncio
import logging
import json
from typing import Callable, Literal
from datetime import datetime
from collections import deque

try:
    from tradingview_ta import TA_Handler, Interval
except ImportError:
    TA_Handler = None
    Interval = None

from config import settings
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


MarketType = Literal["crypto", "stock"]


class TradingViewStream:
    """
    TradingView data stream for real-time market data.
    Falls back to polling mode if WebSocket not available.
    """

    def __init__(
        self,
        market_type: MarketType = "crypto",
        exchange: str = "BINANCE",
        screener: str = "crypto",
        interval: str = "1m",
    ) -> None:
        self.market_type = market_type
        self.exchange = exchange
        self.screener = screener
        self.interval = self._parse_interval(interval)
        self._callbacks: list[Callable[[str, dict], None]] = []
        self._running = False
        self._handlers: dict[str, TA_Handler] = {}

        if TA_Handler is None:
            logger.warning(
                "tradingview_ta not installed. Install with: pip install tradingview-ta"
            )

    def _parse_interval(self, interval: str):
        """Convert interval string to TradingView Interval enum."""
        if Interval is None:
            return interval

        mapping = {
            "1m": Interval.INTERVAL_1_MINUTE,
            "5m": Interval.INTERVAL_5_MINUTES,
            "15m": Interval.INTERVAL_15_MINUTES,
            "1h": Interval.INTERVAL_1_HOUR,
            "4h": Interval.INTERVAL_4_HOURS,
            "1d": Interval.INTERVAL_1_DAY,
        }
        return mapping.get(interval.lower(), Interval.INTERVAL_1_MINUTE)

    def add_callback(self, fn: Callable[[str, dict], None]) -> None:
        """Register callback to receive (symbol, data_dict) on each update."""
        self._callbacks.append(fn)

    async def start(self) -> None:
        """Start the TradingView data stream."""
        if TA_Handler is None:
            logger.error("Cannot start TradingView stream: tradingview_ta not installed")
            return

        self._running = True
        logger.info(
            "TradingView stream started: %s on %s (%s)",
            settings.WATCHLIST,
            self.exchange,
            self.screener,
        )

        # Initialize handlers for each symbol
        for symbol in settings.WATCHLIST:
            try:
                # For crypto: symbol format like "BTCUSDT"
                # For stocks: symbol format like "AAPL", "TSLA"
                handler = TA_Handler(
                    symbol=symbol,
                    screener=self.screener,
                    exchange=self.exchange,
                    interval=self.interval,
                )
                self._handlers[symbol] = handler
            except Exception as exc:
                logger.error("Failed to create handler for %s: %s", symbol, exc)

        # Polling loop
        while self._running:
            for symbol in settings.WATCHLIST:
                if symbol not in self._handlers:
                    continue

                try:
                    handler = self._handlers[symbol]
                    analysis = handler.get_analysis()

                    # Extract price data
                    indicators = analysis.indicators
                    oscillators = analysis.oscillators
                    moving_averages = analysis.moving_averages

                    data = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "price": indicators.get("close", 0),
                        "open": indicators.get("open", 0),
                        "high": indicators.get("high", 0),
                        "low": indicators.get("low", 0),
                        "volume": indicators.get("volume", 0),
                        "change": indicators.get("change", 0),
                        # Technical indicators
                        "rsi": oscillators.get("COMPUTE", {}).get("RSI", 50),
                        "macd": oscillators.get("COMPUTE", {}).get("MACD.macd", 0),
                        "macd_signal": oscillators.get("COMPUTE", {}).get(
                            "MACD.signal", 0
                        ),
                        # Moving averages
                        "ema_20": moving_averages.get("COMPUTE", {}).get("EMA20", 0),
                        "ema_50": moving_averages.get("COMPUTE", {}).get("EMA50", 0),
                        "sma_200": moving_averages.get("COMPUTE", {}).get("SMA200", 0),
                        # Recommendation
                        "recommendation": analysis.summary.get("RECOMMENDATION", "NEUTRAL"),
                    }

                    # Update Neo4j
                    try:
                        Neo4jClient.get().update_token_market_data(symbol, **data)
                    except Exception as exc:
                        logger.warning("Neo4j write failed for %s: %s", symbol, exc)

                    # Fire callbacks
                    for cb in self._callbacks:
                        try:
                            cb(symbol, data)
                        except Exception as exc:
                            logger.warning("Callback error for %s: %s", symbol, exc)

                except Exception as exc:
                    logger.error("Error fetching data for %s: %s", symbol, exc)

            # Poll interval based on timeframe
            await asyncio.sleep(self._get_poll_interval())

    def _get_poll_interval(self) -> int:
        """Get polling interval in seconds based on chart timeframe."""
        if self.interval == "1m":
            return 10  # Poll every 10 seconds for 1m charts
        elif self.interval == "5m":
            return 30
        elif self.interval == "15m":
            return 60
        elif self.interval == "1h":
            return 300  # 5 minutes
        elif self.interval == "4h":
            return 900  # 15 minutes
        else:
            return 60  # Default 1 minute

    async def stop(self) -> None:
        """Stop the TradingView stream."""
        self._running = False
        logger.info("TradingView stream stopped")
