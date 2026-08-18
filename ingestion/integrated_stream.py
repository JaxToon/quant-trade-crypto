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
            source: "binance", "tradingview", "yfinance", or "auto" (auto-detect from config)
        """
        self.source = self._determine_source(source)
        # Keyed by (symbol, timeframe) — SMC/FVG runs independently per layer
        # (BINANCE_SMC_TIMEFRAMES), since a 5m Order Block and a 1h Order Block
        # at "the same" price level are different patterns with different
        # stop distances. TradingView-sourced ticks (no `interval` tag) fall
        # back to the primary timeframe — see _enrich_with_smc.
        self.smc_analyzers: dict[tuple[str, str], SmartMoneyAnalyzer] = {}
        self.fvg_analyzers: dict[tuple[str, str], FairValueGapAnalyzer] = {}
        self.candle_buffers: dict[str, list] = defaultdict(list)
        self._callbacks: list[Callable[[str, dict], None]] = []

        # yfinance gets the SAME multi-timeframe layering as Binance/crypto
        # (not TradingView's single-interval limitation) — Stage 4 of the
        # stock data-path validation explicitly wants signal counts/rejection
        # reasons compared against the crypto baseline apples-to-apples.
        if self.source in ("binance", "yfinance"):
            timeframes = settings.BINANCE_SMC_TIMEFRAMES
        else:
            timeframes = [settings.TRADINGVIEW_INTERVAL]
        self._timeframes = timeframes
        for symbol in settings.WATCHLIST:
            for tf in timeframes:
                self.smc_analyzers[(symbol, tf)] = SmartMoneyAnalyzer(
                    lookback=settings.SMC_LOOKBACK_CANDLES
                )
                self.fvg_analyzers[(symbol, tf)] = FairValueGapAnalyzer(
                    max_fvgs=50,
                    cluster_proximity=0.02
                )

        logger.info(
            "IntegratedMarketStream initialized: source=%s, SMC=%s, symbols=%s, timeframes=%s",
            self.source,
            "enabled" if settings.SMC_ENABLED else "disabled",
            len(settings.WATCHLIST),
            timeframes,
        )

    def _determine_source(self, source: str) -> str:
        """Determine data source based on config or explicit parameter."""
        if source != "auto":
            return source

        market_source = settings.MARKET_SOURCE.lower()
        if market_source in ("tradingview", "yfinance"):
            return market_source
        return "binance"

    def add_callback(self, fn: Callable[[str, dict], None]) -> None:
        """Register callback to receive (symbol, enriched_data) on each tick."""
        self._callbacks.append(fn)

    async def start(self) -> None:
        """
        Start the integrated stream based on configured source.

        Was `if binance: backfill(); if tradingview: ...; else:
        start_binance()` — a 3rd source value (yfinance) matched neither
        `if`, fell through to the `else`, and would have silently started
        the BINANCE stream instead. Same class of bug as MARKET_SOURCE's
        earlier non-asset-aware default: a source dispatch that only
        explicitly handles 2 of N values and treats "no match" as "assume
        the first one" instead of erroring.
        """
        if self.source == "binance":
            self._backfill_history()
            await self._start_binance_stream()
        elif self.source == "tradingview":
            await self._start_tradingview_stream()
        elif self.source == "yfinance":
            self._backfill_yfinance_history()
            await self._start_yfinance_stream()
        else:
            raise ValueError(f"Unknown market data source: {self.source!r}")

    def _backfill_history(self) -> None:
        """
        Warm up SMC/FVG state with recently closed candles via REST so entry
        insight isn't blank for the first `SMC_LOOKBACK_CANDLES` bars after
        every restart — done once per (symbol, timeframe) layer.
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

        for symbol in settings.WATCHLIST:
            for tf in self._timeframes:
                try:
                    klines = client.get_klines(symbol=symbol, interval=tf, limit=settings.SMC_LOOKBACK_CANDLES)
                except Exception as exc:
                    logger.warning("SMC backfill failed for %s [%s]: %s", symbol, tf, exc)
                    continue

                analyzer = self.smc_analyzers[(symbol, tf)]
                fvg = self.fvg_analyzers[(symbol, tf)]
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
                logger.info("SMC backfilled %s [%s]: %d candles", symbol, tf, len(klines))

    def _backfill_yfinance_history(self) -> None:
        """
        Same purpose as _backfill_history() (Binance) — warm up SMC/FVG
        state so entry insight isn't blank for the first
        SMC_LOOKBACK_CANDLES bars after every restart. TradingViewStream's
        path never had this (stock previously started cold every time),
        so this is a genuine improvement for the yfinance path specifically,
        not just parity.

        Regular session only, matching ingestion/yfinance_stream.py's own
        reasoning — extended-hours data is thin/zero-volume noise.
        """
        import pandas as pd
        import yfinance as yf

        from ingestion.yfinance_stream import FETCH_PERIOD

        for symbol in settings.WATCHLIST:
            for tf in self._timeframes:
                period = FETCH_PERIOD.get(tf, "10d")
                try:
                    df = yf.download(symbol, period=period, interval=tf, progress=False, auto_adjust=True, prepost=False)
                except Exception as exc:
                    logger.warning("yfinance SMC backfill failed for %s [%s]: %s", symbol, tf, exc)
                    continue
                if df.empty:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Drop the last row — same still-forming-candle reasoning
                # as YFinanceStream._poll_symbol, so backfill never treats
                # an incomplete bar as historically closed.
                closed = df.iloc[:-1].tail(settings.SMC_LOOKBACK_CANDLES)

                analyzer = self.smc_analyzers[(symbol, tf)]
                fvg = self.fvg_analyzers[(symbol, tf)]
                for ts, row in closed.iterrows():
                    candle = Candle(
                        timestamp=ts.to_pydatetime().replace(tzinfo=None),
                        open=float(row["Open"]), high=float(row["High"]), low=float(row["Low"]),
                        close=float(row["Close"]), volume=float(row["Volume"]),
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
                logger.info("SMC backfilled %s [%s]: %d candles (yfinance)", symbol, tf, len(closed))

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

    async def _start_yfinance_stream(self) -> None:
        """
        Start yfinance polling with SMC overlay — Stage 3 of the staged
        stock data-path validation (see ingestion/yfinance_stream.py).
        Alternative to TradingView, whose free tier rate-limits after ~1
        poll cycle regardless of pacing.

        One YFinanceStream instance per timeframe, run concurrently — gives
        the SAME multi-timeframe layering as Binance/crypto (self._timeframes
        is BINANCE_SMC_TIMEFRAMES for this source, set in __init__) instead
        of TradingView's single-interval limitation, since Stage 4 wants
        signal counts compared against the crypto baseline apples-to-apples.
        """
        from ingestion.yfinance_stream import YFinanceStream

        logger.info("Starting yfinance stream with SMC analysis (timeframes=%s)...", self._timeframes)
        streams = [YFinanceStream(interval=tf) for tf in self._timeframes]
        for s in streams:
            s.add_callback(self._on_yfinance_tick)
        await asyncio.gather(*(s.start() for s in streams))

    def _on_binance_tick(self, symbol: str, indicators: dict) -> None:
        """Process Binance tick data and enrich with SMC analysis."""
        enriched = self._enrich_with_smc(symbol, indicators)
        self._fire_callbacks(symbol, enriched)

    def _on_tradingview_tick(self, symbol: str, data: dict) -> None:
        """Process TradingView tick data and enrich with SMC analysis."""
        enriched = self._enrich_with_smc(symbol, data)
        self._fire_callbacks(symbol, enriched)

    def _on_yfinance_tick(self, symbol: str, data: dict) -> None:
        """Process yfinance tick data and enrich with SMC analysis. data
        carries an 'interval' tag (unlike TradingView), so _enrich_with_smc
        routes it to the correct (symbol, timeframe) analyzer layer."""
        enriched = self._enrich_with_smc(symbol, data)
        self._fire_callbacks(symbol, enriched)

    def _enrich_with_smc(self, symbol: str, market_data: dict) -> dict:
        """
        Enrich market data with Smart Money Concept analysis, on whichever
        timeframe layer this tick belongs to (market_data['interval'], tagged
        by structured_stream.py; falls back to the primary timeframe for
        sources — like TradingView — that don't tag an interval).
        Returns combined dict with technical indicators + SMC signals.
        """
        if not settings.SMC_ENABLED:
            return market_data

        tf = market_data.get('interval', settings.BINANCE_KLINE_INTERVAL)
        key = (symbol, tf)
        if key not in self.smc_analyzers:
            # Unknown timeframe (shouldn't happen — every subscribed stream
            # has a matching analyzer) — pass through unenriched rather than
            # crash the whole tick pipeline over one bad tag.
            logger.debug("No SMC analyzer for %s [%s], skipping enrichment", symbol, tf)
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

            smc = self.smc_analyzers[key]
            fvg = self.fvg_analyzers[key]

            if is_closed:
                # Candle finished — append to history and re-run pattern detection.
                smc_signals = smc.update(candle)

                candles = list(smc.candles)
                if len(candles) >= 3:
                    fvg.detect_fvg(
                        candles=candles,
                        index=len(candles) - 2,  # Previous candle
                        current_price=candle.close,
                        volume=candle.volume
                    )
            else:
                # Candle still forming — just re-check proximity to already
                # detected zones against the live price, don't mutate history.
                smc_signals = smc.peek_signals(candle)

            # Update all FVG mitigation % with the current live price
            fvg.update_all(candle.close)

            # Get FVG trading signal
            ms_trend = smc_signals.get('market_structure', 'unknown')
            fvg_signal = fvg.get_trading_signal(
                current_price=candle.close,
                trend=ms_trend
            )

            # Combine all SMC data
            smc_context = {
                **smc_signals,
                **fvg_signal,
                'timeframe': tf,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Merge with original market data
            enriched = {
                **market_data,
                'smc': smc_context,
                'smc_enabled': True
            }

            # Log significant events (tag the timeframe — otherwise a 5m and
            # 1h Order Block on the same symbol are indistinguishable in logs)
            if smc_signals.get('in_bullish_ob'):
                logger.info("%s [%s]: Price entered BULLISH Order Block", symbol, tf)
            if smc_signals.get('in_bearish_ob'):
                logger.info("%s [%s]: Price entered BEARISH Order Block", symbol, tf)
            if fvg_signal.get('direction') != 'neutral':
                logger.info(
                    "%s [%s]: FVG signal %s (confidence=%.1f%%)",
                    symbol, tf,
                    fvg_signal.get('direction'),
                    fvg_signal.get('confidence', 0)
                )

            # Update Neo4j with SMC metadata — primary timeframe only, so the
            # Token node's smc_* properties (which feed GNN features) stay
            # consistent with one timeframe rather than flip-flopping between
            # three concurrent streams.
            if tf == settings.BINANCE_KLINE_INTERVAL:
                try:
                    Neo4jClient.get().update_token_smc_data(symbol, smc_context)
                except Exception as exc:
                    logger.warning("Neo4j SMC write failed for %s: %s", symbol, exc)

            return enriched

        except Exception as exc:
            logger.error("SMC enrichment error for %s [%s]: %s", symbol, tf, exc, exc_info=True)
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

    def get_smc_summary(self, symbol: str, timeframe: str | None = None) -> dict:
        """Get current SMC state summary for a symbol on one timeframe layer."""
        tf = timeframe or settings.BINANCE_KLINE_INTERVAL
        key = (symbol, tf)
        if key not in self.smc_analyzers:
            return {}

        smc = self.smc_analyzers[key]
        fvg = self.fvg_analyzers[key]

        return {
            'symbol': symbol,
            'timeframe': tf,
            'market_structure': smc.market_structure.trend if smc.market_structure else 'unknown',
            'active_order_blocks': len([ob for ob in smc.order_blocks if not ob.mitigated]),
            'active_fvgs': len([f for f in fvg.fvgs if not f.mitigated]),
            'liquidity_zones': len([liq for liq in smc.liquidity_zones if not liq.swept]),
            'fvg_clusters': len(fvg.clusters),
        }

    def get_all_summaries(self) -> dict[str, dict]:
        """Get SMC summaries for all tracked symbols/timeframes, keyed 'SYMBOL_tf'."""
        return {
            f"{symbol}_{tf}": self.get_smc_summary(symbol, tf)
            for symbol in settings.WATCHLIST
            for tf in self._timeframes
        }
