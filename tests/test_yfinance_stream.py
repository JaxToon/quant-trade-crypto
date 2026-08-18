"""
Tests for ingestion/yfinance_stream.py — Stage 1 of the staged stock
data-path validation (yfinance as an alternative live-ish poller to
TradingViewStream, whose free tier rate-limits after ~1 poll cycle).

No real network calls — build_tick_data and the closed-bar detection are
tested directly with synthetic DataFrames, matching the pure-function
extraction pattern tests/test_tradingview_stream.py already uses.

The live, real-network proof (repeated polling against genuinely static
market-closed data produces zero duplicate ticks) was run manually this
session, not automated here — see the session's scratchpad test script.
"""
from __future__ import annotations

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ingestion.yfinance_stream import (
    YFinanceStream,
    _bollinger,
    _macd,
    _rsi,
    build_tick_data,
    is_regular_session,
)

EASTERN = ZoneInfo("America/New_York")


def _synthetic_df(n: int = 30, start_price: float = 100.0) -> pd.DataFrame:
    """A plausible OHLCV frame with a real DatetimeIndex, 5m spacing."""
    idx = pd.date_range("2026-08-17 09:30", periods=n, freq="5min", tz=EASTERN)
    closes = [start_price + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "Open": [c - 0.05 for c in closes],
        "High": [c + 0.1 for c in closes],
        "Low": [c - 0.1 for c in closes],
        "Close": closes,
        "Volume": [1000 + i * 10 for i in range(n)],
    }, index=idx)


class TestMarketHoursGate:
    def test_weekday_during_session_is_open(self):
        # 2026-08-18 is a Tuesday
        now = datetime(2026, 8, 18, 10, 0, tzinfo=EASTERN)
        assert is_regular_session(now) is True

    def test_weekday_before_open_is_closed(self):
        now = datetime(2026, 8, 18, 8, 0, tzinfo=EASTERN)
        assert is_regular_session(now) is False

    def test_weekday_after_close_is_closed(self):
        now = datetime(2026, 8, 18, 17, 0, tzinfo=EASTERN)
        assert is_regular_session(now) is False

    def test_saturday_is_closed(self):
        now = datetime(2026, 8, 22, 10, 0, tzinfo=EASTERN)  # Saturday
        assert is_regular_session(now) is False

    def test_sunday_is_closed(self):
        now = datetime(2026, 8, 23, 10, 0, tzinfo=EASTERN)  # Sunday
        assert is_regular_session(now) is False

    def test_market_open_boundary_is_inclusive(self):
        now = datetime(2026, 8, 18, 9, 30, tzinfo=EASTERN)
        assert is_regular_session(now) is True

    def test_market_close_boundary_is_exclusive(self):
        now = datetime(2026, 8, 18, 16, 0, tzinfo=EASTERN)
        assert is_regular_session(now) is False


class TestIndicatorFormulas:
    def test_rsi_neutral_with_insufficient_data(self):
        assert _rsi([100.0, 101.0]) == 50.0

    def test_rsi_bounded_0_to_100(self):
        rising = [100.0 + i for i in range(20)]
        assert 0.0 <= _rsi(rising) <= 100.0

    def test_macd_zero_with_insufficient_data(self):
        assert _macd([100.0, 101.0]) == (0.0, 0.0, 0.0)

    def test_macd_hist_is_macd_minus_signal(self):
        closes = [100.0 + (i % 5) for i in range(60)]
        macd, signal, hist = _macd(closes)
        assert hist == pytest.approx(macd - signal)

    def test_bollinger_zero_with_insufficient_data(self):
        assert _bollinger([100.0]) == (0.0, 0.0, 0.0)

    def test_bollinger_upper_above_lower(self):
        closes = [100.0 + (i % 7) * 0.5 for i in range(30)]
        upper, mid, lower = _bollinger(closes)
        assert upper > mid > lower


class TestBuildTickData:
    def test_returns_numeric_indicator_fields(self):
        df = _synthetic_df(n=30)
        tick = build_tick_data(df, "5m", kline_closed=True)

        for key in ("price", "open", "high", "low", "volume", "rsi", "macd",
                    "macd_signal", "macd_hist", "ema_20", "ema_50", "sma_200",
                    "bb_upper", "bb_mid", "bb_lower"):
            assert isinstance(tick[key], (int, float)), f"{key} must be numeric, got {tick[key]!r}"

    def test_tags_interval_and_closed_flag(self):
        df = _synthetic_df(n=30)
        tick = build_tick_data(df, "15m", kline_closed=True)
        assert tick["interval"] == "15m"
        assert tick["kline_closed"] is True

    def test_price_matches_last_close(self):
        df = _synthetic_df(n=30)
        tick = build_tick_data(df, "5m", kline_closed=True)
        assert tick["price"] == pytest.approx(float(df.iloc[-1]["Close"]))


class TestClosedBarDetection:
    """The core correctness property: repeated polling against the SAME
    underlying data (e.g. market closed, or polled faster than a new bar
    forms) must not re-fire — mirrors tradingview_stream.py's _is_new_bar
    reasoning, but via comparing the last CLOSED bar's own timestamp
    rather than a fixed time-bucket."""

    def test_first_poll_fires_second_does_not(self, monkeypatch):
        stream = YFinanceStream(interval="5m")
        events = []
        stream.add_callback(lambda sym, data: events.append((sym, data)))

        df = _synthetic_df(n=30)
        monkeypatch.setattr(
            "ingestion.yfinance_stream.yf.download",
            lambda *a, **k: df,
        )
        monkeypatch.setattr(
            "ingestion.yfinance_stream.Neo4jClient.get",
            lambda: type("_C", (), {"update_token_market_data": lambda self, *a, **k: None})(),
        )

        stream._poll_symbol("AAPL", "5d")
        stream._poll_symbol("AAPL", "5d")  # identical data -- must not re-fire

        assert len(events) == 1

    def test_new_closed_bar_fires_again(self, monkeypatch):
        stream = YFinanceStream(interval="5m")
        events = []
        stream.add_callback(lambda sym, data: events.append((sym, data)))

        monkeypatch.setattr(
            "ingestion.yfinance_stream.Neo4jClient.get",
            lambda: type("_C", (), {"update_token_market_data": lambda self, *a, **k: None})(),
        )

        df1 = _synthetic_df(n=30)
        monkeypatch.setattr("ingestion.yfinance_stream.yf.download", lambda *a, **k: df1)
        stream._poll_symbol("AAPL", "5d")

        df2 = _synthetic_df(n=31)  # one more real bar has formed
        monkeypatch.setattr("ingestion.yfinance_stream.yf.download", lambda *a, **k: df2)
        stream._poll_symbol("AAPL", "5d")

        assert len(events) == 2

    def test_uses_second_to_last_row_not_the_forming_bar(self, monkeypatch):
        """The very last row of a yfinance pull is frequently the
        still-forming current candle -- must never be treated as closed."""
        stream = YFinanceStream(interval="5m")
        events = []
        stream.add_callback(lambda sym, data: events.append((sym, data)))

        df = _synthetic_df(n=30)
        monkeypatch.setattr("ingestion.yfinance_stream.yf.download", lambda *a, **k: df)
        monkeypatch.setattr(
            "ingestion.yfinance_stream.Neo4jClient.get",
            lambda: type("_C", (), {"update_token_market_data": lambda self, *a, **k: None})(),
        )

        stream._poll_symbol("AAPL", "5d")
        assert events[0][1]["price"] == pytest.approx(float(df.iloc[-2]["Close"]))

    def test_empty_dataframe_does_not_crash_or_fire(self, monkeypatch):
        stream = YFinanceStream(interval="5m")
        events = []
        stream.add_callback(lambda sym, data: events.append((sym, data)))
        monkeypatch.setattr("ingestion.yfinance_stream.yf.download", lambda *a, **k: pd.DataFrame())

        stream._poll_symbol("AAPL", "5d")
        assert events == []
