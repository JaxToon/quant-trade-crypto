"""
Regression test for _build_tick_data (ingestion/tradingview_stream.py).

Guards against the bug where technical indicators were read from
analysis.oscillators['COMPUTE']/analysis.moving_averages['COMPUTE'] — dicts
of BUY/SELL/NEUTRAL recommendation LABELS — instead of analysis.indicators
(the actual numeric values). The old code silently wrote string labels to
Neo4j, which gnn/dataset.py's _safe_float() coerced to 0.0 for every stock
token, every technical-indicator field, always.
"""
from __future__ import annotations

from types import SimpleNamespace

from ingestion.tradingview_stream import _build_tick_data


def _fake_analysis() -> SimpleNamespace:
    return SimpleNamespace(
        indicators={
            "close": 305.31, "open": 303.0, "high": 306.0, "low": 302.5,
            "volume": 1234567, "change": 0.75,
            "RSI": 51.153558105756105,
            "MACD.macd": -1.0011585800795615,
            "MACD.signal": -1.4978931966128866,
            "EMA20": 304.36, "EMA50": 307.49, "SMA200": 318.70,
            "BB.upper": 307.2563, "BB.lower": 300.1522,
        },
        # BUY/SELL/NEUTRAL labels per indicator — must NOT end up as the
        # value for rsi/macd/ema_*/etc in the built dict.
        oscillators={"RECOMMENDATION": "NEUTRAL", "COMPUTE": {"RSI": "NEUTRAL", "MACD.macd": None}},
        moving_averages={"RECOMMENDATION": "BUY", "COMPUTE": {"EMA20": "BUY", "SMA200": "SELL"}},
        summary={"RECOMMENDATION": "NEUTRAL"},
    )


def test_indicators_are_numeric_not_recommendation_labels():
    data = _build_tick_data(_fake_analysis(), "1h", kline_closed=True)

    assert data["rsi"] == 51.153558105756105
    assert data["macd"] == -1.0011585800795615
    assert data["macd_signal"] == -1.4978931966128866
    assert data["ema_20"] == 304.36
    assert data["ema_50"] == 307.49
    assert data["sma_200"] == 318.70
    for key in ("rsi", "macd", "macd_signal", "ema_20", "ema_50", "sma_200"):
        assert isinstance(data[key], (int, float)), f"{key} must be numeric, got {data[key]!r}"


def test_macd_hist_and_bollinger_bands_are_populated():
    data = _build_tick_data(_fake_analysis(), "1h", kline_closed=True)

    assert data["macd_hist"] == data["macd"] - data["macd_signal"]
    assert data["bb_upper"] == 307.2563
    assert data["bb_lower"] == 300.1522
    assert data["bb_mid"] == (307.2563 + 300.1522) / 2


def test_interval_and_kline_closed_are_tagged():
    data = _build_tick_data(_fake_analysis(), "1h", kline_closed=False)
    assert data["interval"] == "1h"
    assert data["kline_closed"] is False


def test_missing_indicator_keys_fall_back_to_defaults_not_crash():
    empty = SimpleNamespace(
        indicators={},
        oscillators={"RECOMMENDATION": "NEUTRAL", "COMPUTE": {}},
        moving_averages={"RECOMMENDATION": "NEUTRAL", "COMPUTE": {}},
        summary={"RECOMMENDATION": "NEUTRAL"},
    )
    data = _build_tick_data(empty, "5m", kline_closed=True)
    assert data["rsi"] == 50
    assert data["macd"] == 0
    assert data["bb_mid"] == 0
