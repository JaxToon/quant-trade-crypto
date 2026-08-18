"""
Tests for ingestion/integrated_stream.py's source resolution and dispatch.

Regression guard for a real bug found while wiring in yfinance as a 3rd
source: start() was `if binance: ...; if tradingview: ...; else:
start_binance()` — a 3rd source value matched neither `if` and silently
fell through to starting the BINANCE stream instead of erroring or
dispatching correctly. Fixed to if/elif/elif/else-raise. These tests
never call start() itself (that's real network I/O) — they check
_determine_source()'s resolution and the timeframe layering __init__
derives from it, which is what actually exercises the fixed logic path.
"""
from __future__ import annotations

import asyncio

import pytest

from config import settings
from ingestion.integrated_stream import IntegratedMarketStream


class TestSourceResolution:
    def test_explicit_source_is_used_as_is(self):
        s = IntegratedMarketStream(source="yfinance")
        assert s.source == "yfinance"

    def test_auto_resolves_from_market_source_tradingview(self, monkeypatch):
        monkeypatch.setattr(settings, "MARKET_SOURCE", "tradingview")
        s = IntegratedMarketStream(source="auto")
        assert s.source == "tradingview"

    def test_auto_resolves_from_market_source_yfinance(self, monkeypatch):
        monkeypatch.setattr(settings, "MARKET_SOURCE", "yfinance")
        s = IntegratedMarketStream(source="auto")
        assert s.source == "yfinance"

    def test_auto_defaults_to_binance_for_unknown_value(self, monkeypatch):
        monkeypatch.setattr(settings, "MARKET_SOURCE", "something_else")
        s = IntegratedMarketStream(source="auto")
        assert s.source == "binance"


class TestTimeframeLayering:
    """yfinance gets the same multi-timeframe layering as Binance/crypto —
    NOT TradingView's single-interval limitation — so Stage 4 stock-vs-
    crypto comparisons are apples-to-apples on signal counts."""

    def test_yfinance_gets_multi_timeframe_like_binance(self):
        s = IntegratedMarketStream(source="yfinance")
        assert s._timeframes == settings.BINANCE_SMC_TIMEFRAMES
        assert len(s._timeframes) > 1

    def test_binance_gets_multi_timeframe(self):
        s = IntegratedMarketStream(source="binance")
        assert s._timeframes == settings.BINANCE_SMC_TIMEFRAMES

    def test_tradingview_gets_single_interval(self):
        s = IntegratedMarketStream(source="tradingview")
        assert s._timeframes == [settings.TRADINGVIEW_INTERVAL]

    def test_analyzer_count_matches_symbols_times_timeframes(self):
        s = IntegratedMarketStream(source="yfinance")
        expected = len(settings.WATCHLIST) * len(s._timeframes)
        assert len(s.smc_analyzers) == expected
        assert len(s.fvg_analyzers) == expected


class TestStartDispatchRaisesOnUnknownSource:
    """The actual regression: an unrecognized self.source must error, not
    silently fall through to starting a different stream than requested."""

    def test_unknown_source_raises_instead_of_defaulting_to_binance(self):
        s = IntegratedMarketStream(source="totally_unknown_source")
        with pytest.raises(ValueError, match="Unknown market data source"):
            asyncio.run(s.start())
