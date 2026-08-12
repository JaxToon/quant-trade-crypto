"""
Integrated Market Data Stream with Smart Money Concept Analysis.
Combines price data ingestion with real-time SMC/FVG detection.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable
from datetime import datetime
from collections import defaultdict

from config import settings
from graph.neo4j_client import Neo4jClient
from strategy.smart_money import SmartMoneyAnalyzer, Candle
from strategy.fvg_analyzer import FairValueGapAnalyzer

logger = logging.getLogger(__name__)


class IntegratedMarketStream:
    """
    Unified stream that:
    1. Ingests market data (Binance or TradingView)
    2. Runs SMC analysis (Order Blocks, Market Structure)
    3. Runs FVG analysis (Fair Value Gaps)
    4. Provides combined signals to trading system
    """

    def __init__(self, source: str = "auto"):
        """
        Args:
            source: "binance", "tradingview", or "auto" (auto-detect from config)
        """
        self.source = self._determine_source(source)
        self.smc_analyzers: dict[str, SmartMoneyAnalyzer] = {}
        self.fvg_analyzers: dict[str, FairValueGapAnalyzer] = {}
        self.candle_buffers: dict[str, list] = defaultdict(list)
        self._callbacks: list[Callable[[str, dict], None]] = []

        # Initialize analyzers for each symbol
        for symbol in settings.WATCHLIST:
            self.smc_analyzers[symbol] = SmartMoneyAnalyzer(
                lookback=settings.SMC_LOOKBACK_CANDLES
            )
            self.fvg_analyzers[symbol] = FairValueGapAnalyzer(
                max_fvgs=50,
                cluster_proximity=0.02
            )

        logger.info(
            "IntegratedMarketStream initialized: source=%s, SMC=%s, symbols=%s",
            self.source,
            "enabled" if settings.SMC_ENABLED else "disabled",
            len(settings.WATCHLIST)
        )

    def _determine_source(self, source: str) -> str:
        """Determine data source based on config or explicit parameter."""
        if source != "auto":
            return source

        market_source = settings.MARKET_SOURCE.lower()
        if market_source == "tradingview":
            return "tradingview"
        else:
            return "binance"

    def add_callback(self, fn: Callable[[str, dict], None]) -> None:
        """Register callback to receive (symbol, enriched_data) on each tick."""
        self._callbacks.append(fn)

    async def start(self) -> None:
        """Start the integrated stream based on configured source."""
        if self.source == "binance":
            self._backfill_history()
        if self.source == "tradingview":
            await self._start_tradingview_stream()
        else:
            await self._start_binance_stream()

    def _backfill_history(self) -> None:
        """
        Warm up SMC/FVG state with recently closed candles via REST so entry
        insight isn't blank for the first `SMC_LOOKBACK_CANDLES` bars after
        every restart (e.g. ~8 hours of dead air at the 5m default).
        """
        try:
            from binance.client import Client as SyncClient
            client = SyncClient(
                settings.BINANCE_API_KEY,
                settings.BINANCE_API_SECRET,
                testnet=settings.BINANCE_TESTNET,
            )
        except Exception as exc:
            logger.warning("SMC backfill skipped: could not create Binance client (%s)", exc)
            return

        interval = settings.BINANCE_KLINE_INTERVAL
        for symbol in settings.WATCHLIST:
            try:
                klines = client.get_klines(symbol=symbol, interval=interval, limit=settings.SMC_LOOKBACK_CANDLES)
            except Exception as exc:
                logger.warning("SMC backfill failed for %s: %s", symbol, exc)
                continue

            analyzer = self.smc_analyzers[symbol]
            fvg = self.fvg_analyzers[symbol]
            for row in klines:
                # kline row: [open_time, open, high, low, close, volume, close_time, ...]
                candle = Candle(
                    timestamp=datetime.utcfromtimestamp(row[0] / 1000),
                    open=float(row[1]), high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5]),
                )
                analyzer.update(candle)
                candles = list(analyzer.candles)
                if len(candles) >= 3:
                    fvg.detect_fvg(
                        candles=candles, index=len(candles) - 2,
                        current_price=candle.close, volume=candle.volume,
                    )
            if analyzer.candles:
                fvg.update_all(analyzer.candles[-1].close)
            logger.info("SMC backfilled %s: %d x %s candles", symbol, len(klines), interval)

    async def _start_binance_stream(self) -> None:
        """Start Binance WebSocket stream with SMC overlay."""
        from ingestion.structured_stream import StructuredStream

        binance_stream = StructuredStream()
        binance_stream.add_callback(self._on_binance_tick)

        logger.info("Starting Binance stream with SMC analysis...")
        await binance_stream.start()

    async def _start_tradingview_stream(self) -> None:
        """Start TradingView stream with SMC overlay."""
        from ingestion.tradingview_stream import TradingViewStream

        asset_type = settings.ASSET_TYPE
        exchange = settings.TRADINGVIEW_EXCHANGE
        screener = settings.TRADINGVIEW_SCREENER
        interval = settings.TRADINGVIEW_INTERVAL

        tv_stream = TradingViewStream(
            market_type=asset_type,
            exchange=exchange,
            screener=screener,
            interval=interval
        )
        tv_stream.add_callback(self._on_tradingview_tick)

        logger.info("Starting TradingView stream with SMC analysis...")
        await tv_stream.start()

    def _on_binance_tick(self, symbol: str, indicators: dict) -> None:
        """Process Binance tick data and enrich with SMC analysis."""
        enriched = self._enrich_with_smc(symbol, indicators)
        self._fire_callbacks(symbol, enriched)

    def _on_tradingview_tick(self, symbol: str, data: dict) -> None:
        """Process TradingView tick data and enrich with SMC analysis."""
        enriched = self._enrich_with_smc(symbol, data)
        self._fire_callbacks(symbol, enriched)

    def _enrich_with_smc(self, symbol: str, market_data: dict) -> dict:
        """
        Enrich market data with Smart Money Concept analysis.
        Returns combined dict with technical indicators + SMC signals.
        """
        if not settings.SMC_ENABLED:
            return market_data

        try:
            # Create candle from market data (real OHLC from the kline stream —
            # falls back to price-only if a non-Binance source is missing OHLC)
            candle = Candle(
                timestamp=datetime.utcnow(),
                open=market_data.get('open', market_data.get('price', 0)),
                high=market_data.get('high', market_data.get('price', 0)),
                low=market_data.get('low', market_data.get('price', 0)),
                close=market_data.get('price', 0),
                volume=market_data.get('volume', 0)
            )
            is_closed = market_data.get('kline_closed', True)

            if is_closed:
                # Candle finished — append to history and re-run pattern detection.
                smc_signals = self.smc_analyzers[symbol].update(candle)

                candles = list(self.smc_analyzers[symbol].candles)
                if len(candles) >= 3:
                    self.fvg_analyzers[symbol].detect_fvg(
                        candles=candles,
                        index=len(candles) - 2,  # Previous candle
                        current_price=candle.close,
                        volume=candle.volume
                    )
            else:
                # Candle still forming — just re-check proximity to already
                # detected zones against the live price, don't mutate history.
                smc_signals = self.smc_analyzers[symbol].peek_signals(candle)

            # Update all FVG mitigation % with the current live price
            self.fvg_analyzers[symbol].update_all(candle.close)

            # Get FVG trading signal
            ms_trend = smc_signals.get('market_structure', 'unknown')
            fvg_signal = self.fvg_analyzers[symbol].get_trading_signal(
                current_price=candle.close,
                trend=ms_trend
            )

            # Combine all SMC data
            smc_context = {
                **smc_signals,
                **fvg_signal,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Merge with original market data
            enriched = {
                **market_data,
                'smc': smc_context,
                'smc_enabled': True
            }

            # Log significant events
            if smc_signals.get('in_bullish_ob'):
                logger.info("%s: Price entered BULLISH Order Block", symbol)
            if smc_signals.get('in_bearish_ob'):
                logger.info("%s: Price entered BEARISH Order Block", symbol)
            if fvg_signal.get('direction') != 'neutral':
                logger.info(
                    "%s: FVG signal %s (confidence=%.1f%%)",
                    symbol,
                    fvg_signal.get('direction'),
                    fvg_signal.get('confidence', 0)
                )

            # Update Neo4j with SMC metadata
            try:
                Neo4jClient.get().update_token_smc_data(symbol, smc_context)
            except Exception as exc:
                logger.warning("Neo4j SMC write failed for %s: %s", symbol, exc)

            return enriched

        except Exception as exc:
            logger.error("SMC enrichment error for %s: %s", symbol, exc, exc_info=True)
            return {**market_data, 'smc_enabled': False, 'smc_error': str(exc)}

    def _fire_callbacks(self, symbol: str, enriched_data: dict) -> None:
        """Fire all registered callbacks with enriched data."""
        for cb in self._callbacks:
            try:
                cb(symbol, enriched_data)
            except Exception as exc:
                logger.warning("Callback error for %s: %s", symbol, exc)

    async def stop(self) -> None:
        """Stop the integrated stream."""
        logger.info("IntegratedMarketStream stopped")

    def get_smc_summary(self, symbol: str) -> dict:
        """Get current SMC state summary for a symbol."""
        if symbol not in self.smc_analyzers:
            return {}

        smc = self.smc_analyzers[symbol]
        fvg = self.fvg_analyzers[symbol]

        return {
            'symbol': symbol,
            'market_structure': smc.market_structure.trend if smc.market_structure else 'unknown',
            'active_order_blocks': len([ob for ob in smc.order_blocks if not ob.mitigated]),
            'active_fvgs': len([f for f in fvg.fvgs if not f.mitigated]),
            'liquidity_zones': len([liq for liq in smc.liquidity_zones if not liq.swept]),
            'fvg_clusters': len(fvg.clusters),
        }

    def get_all_summaries(self) -> dict[str, dict]:
        """Get SMC summaries for all tracked symbols."""
        return {symbol: self.get_smc_summary(symbol) for symbol in settings.WATCHLIST}
