"""
Binance WebSocket stream: tick-level price/volume + computed technical indicators.
Pushes data into the Neo4j token nodes in real-time.
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from datetime import datetime
from typing import Callable

from binance import AsyncClient, BinanceSocketManager
from config import settings
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# ── Technical indicator calculators ───────────────────────────────────────

class IndicatorBuffer:
    """Rolling window for a single symbol; computes RSI, MACD, Bollinger Bands."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.closes: deque[float] = deque(maxlen=max(settings.BB_PERIOD, settings.MACD_SLOW) + 10)
        self.last_price: float = 0.0
        self.last_volume: float = 0.0

    def update(self, price: float, volume: float) -> dict:
        self.last_price = price
        self.last_volume = volume
        self.closes.append(price)
        indicators = {"price": price, "volume": volume}
        if len(self.closes) >= settings.RSI_PERIOD + 1:
            indicators["rsi"] = self._rsi()
        if len(self.closes) >= settings.MACD_SLOW + settings.MACD_SIGNAL:
            indicators.update(self._macd())
        if len(self.closes) >= settings.BB_PERIOD:
            indicators.update(self._bollinger())
        return indicators

    def _rsi(self) -> float:
        closes = list(self.closes)[-(settings.RSI_PERIOD + 1):]
        gains, losses = [], []
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            (gains if delta > 0 else losses).append(abs(delta))
        avg_gain = sum(gains) / settings.RSI_PERIOD if gains else 0.0
        avg_loss = sum(losses) / settings.RSI_PERIOD if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _ema(self, period: int, data: list[float]) -> list[float]:
        k = 2 / (period + 1)
        emas = [data[0]]
        for p in data[1:]:
            emas.append(p * k + emas[-1] * (1 - k))
        return emas

    def _macd(self) -> dict:
        closes = list(self.closes)
        fast_ema = self._ema(settings.MACD_FAST, closes)
        slow_ema = self._ema(settings.MACD_SLOW, closes)
        macd_line = [f - s for f, s in zip(fast_ema[-len(slow_ema):], slow_ema)]
        signal_line = self._ema(settings.MACD_SIGNAL, macd_line)
        return {
            "macd": macd_line[-1],
            "macd_signal": signal_line[-1],
            "macd_hist": macd_line[-1] - signal_line[-1],
        }

    def _bollinger(self) -> dict:
        closes = list(self.closes)[-settings.BB_PERIOD:]
        mean = sum(closes) / settings.BB_PERIOD
        std = math.sqrt(sum((c - mean) ** 2 for c in closes) / settings.BB_PERIOD)
        return {
            "bb_upper": mean + settings.BB_STD * std,
            "bb_mid": mean,
            "bb_lower": mean - settings.BB_STD * std,
        }


# ── Binance WebSocket handler ─────────────────────────────────────────────

class StructuredStream:
    """
    Opens a Binance multi-symbol ticker stream.
    On each tick: computes indicators, writes to Neo4j, calls optional callbacks.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, IndicatorBuffer] = {
            sym: IndicatorBuffer(sym) for sym in settings.WATCHLIST
        }
        self._callbacks: list[Callable[[str, dict], None]] = []
        self._client: AsyncClient | None = None
        self._bm: BinanceSocketManager | None = None

    def add_callback(self, fn: Callable[[str, dict], None]) -> None:
        """Register a function that receives (symbol, indicators_dict) on each tick."""
        self._callbacks.append(fn)

    async def start(self) -> None:
        self._client = await AsyncClient.create(
            settings.BINANCE_API_KEY,
            settings.BINANCE_API_SECRET,
            testnet=settings.BINANCE_TESTNET,
        )
        self._bm = BinanceSocketManager(self._client)
        # Subscribe to real OHLC kline streams (not miniTicker) — SMC/FVG need
        # actual candle bodies+wicks, which a last-price ticker cannot provide.
        # One subscription per (symbol, timeframe) — SMC/FVG runs independently
        # on each layer in BINANCE_SMC_TIMEFRAMES; only the primary interval
        # (BINANCE_KLINE_INTERVAL) also feeds the technical-indicator buffer
        # and gets persisted to Neo4j, so RSI/MACD/BB never mix candles from
        # different timeframes.
        timeframes = settings.BINANCE_SMC_TIMEFRAMES
        streams = [f"{sym.lower()}@kline_{tf}" for sym in settings.WATCHLIST for tf in timeframes]
        async with self._bm.multiplex_socket(streams) as msock:
            logger.info("Binance kline streams open %s: %s", timeframes, settings.WATCHLIST)
            while True:
                msg = await msock.recv()
                if msg is None:
                    break
                await self._handle(msg)

    async def _handle(self, msg: dict) -> None:
        if msg.get("e") == "error":
            logger.error("Binance stream error: %s", msg)
            return
        data = msg.get("data", {})
        symbol = data.get("s", "")
        if symbol not in self._buffers:
            return
        k = data.get("k", {})
        if not k:
            return

        interval = k.get("i", settings.BINANCE_KLINE_INTERVAL)
        price = float(k.get("c", 0))
        volume = float(k.get("v", 0))
        is_closed = bool(k.get("x", False))
        is_primary = interval == settings.BINANCE_KLINE_INTERVAL

        # RSI/MACD/BB only make sense as a single rolling window on one
        # timeframe — secondary layers (15m/1h) skip technical indicators
        # entirely and pass through bare OHLCV for SMC/FVG to consume.
        if is_primary:
            indicators = self._buffers[symbol].update(price, volume)
        else:
            indicators = {"price": price, "volume": volume}

        indicators.update({
            "open": float(k.get("o", price)),
            "high": float(k.get("h", price)),
            "low": float(k.get("l", price)),
            "close": price,
            "interval": interval,
        })
        indicators["kline_closed"] = is_closed

        # Persist to Neo4j only for the primary timeframe's closed candles —
        # avoids write-storming Token nodes with mismatched-timeframe data
        # from three concurrent streams.
        if is_closed and is_primary:
            try:
                persisted = {k_: v for k_, v in indicators.items() if k_ not in ("kline_closed", "interval")}
                Neo4jClient.get().update_token_market_data(symbol, **persisted)
            except Exception as exc:
                logger.warning("Neo4j write failed for %s: %s", symbol, exc)

        # Fire callbacks (e.g. SMC enrichment, GNN inference trigger) on every tick
        for cb in self._callbacks:
            try:
                cb(symbol, indicators)
            except Exception as exc:
                logger.warning("Callback error: %s", exc)

    async def stop(self) -> None:
        if self._client:
            await self._client.close_connection()
