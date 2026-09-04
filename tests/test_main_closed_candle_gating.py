"""
Tests for main.py's closed-candle gating on entry decisions.

Real forensics (2026-09): live crypto SHORT trades underperformed a
backtest replay of the identical historical window (30-50% WR live vs
58-61% backtest), and the gap was never fully explained by any of the
GNN-checkpoint-quality or filter-based hypotheses tested earlier this
session. Reading the live tick pipeline found a real, previously-unnoticed
methodology mismatch: `on_tick` stored EVERY tick's indicators (including
`SmartMoneyAnalyzer.peek_signals()` reads against a still-forming candle's
live, not-yet-settled price) into `_latest_indicators`, and
`_build_entry_insights` read new-position decisions straight from
whatever happened to be sitting there at the moment the 60s decision loop
polled it — which is disproportionately likely to be a forming-candle
peek, not a freshly-closed candle. `backtest.py`'s replay never evaluates
anything but fully-closed candles, so this was a real, silent difference
in what the two paths were actually deciding on.

Fix: a second dict, `_latest_closed_indicators`, only ever updated when
`kline_closed=True`; `_build_entry_insights` (and the candidate-log's
`smc_raw` lookup, and Path 2's decision_loop) now read from that one.
`_latest_indicators` itself is untouched and keeps updating every tick --
SL/TP monitoring (`_current_prices` -> `check_and_close`) SHOULD react to
real-time price; only NEW-POSITION decisions needed gating.

These tests exercise `on_tick` and `_build_entry_insights` directly
against fresh local dicts (module globals are saved/restored around each
test to avoid cross-test pollution) rather than running the full
decision_loop, which needs a live Neo4j/GNN/Binance client.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import main


@pytest.fixture(autouse=True)
def _isolated_globals():
    """main._latest_indicators / _latest_closed_indicators are module-level
    mutable dicts shared across the whole process — snapshot and restore
    them so this test file can't leak state into (or pick up state from)
    any other test."""
    saved_all = dict(main._latest_indicators)
    saved_closed = dict(main._latest_closed_indicators)
    main._latest_indicators.clear()
    main._latest_closed_indicators.clear()
    yield
    main._latest_indicators.clear()
    main._latest_indicators.update(saved_all)
    main._latest_closed_indicators.clear()
    main._latest_closed_indicators.update(saved_closed)


def _tick(price: float, kline_closed: bool, interval: str = "5m") -> dict:
    return {
        "price": price,
        "interval": interval,
        "kline_closed": kline_closed,
        "smc": {"direction": "short", "confidence": 40.0, "smc_signal": -0.5},
    }


def test_on_tick_always_updates_latest_but_only_closed_updates_closed_dict():
    main.on_tick("BTCUSDT", _tick(100.0, kline_closed=False))
    assert ("BTCUSDT", "5m") in main._latest_indicators
    assert ("BTCUSDT", "5m") not in main._latest_closed_indicators

    main.on_tick("BTCUSDT", _tick(101.0, kline_closed=True))
    assert main._latest_indicators[("BTCUSDT", "5m")]["price"] == 101.0
    assert main._latest_closed_indicators[("BTCUSDT", "5m")]["price"] == 101.0

    # A subsequent forming-candle tick keeps _latest_indicators current for
    # price/dashboard purposes but must NOT advance the closed-only dict.
    main.on_tick("BTCUSDT", _tick(999.0, kline_closed=False))
    assert main._latest_indicators[("BTCUSDT", "5m")]["price"] == 999.0
    assert main._latest_closed_indicators[("BTCUSDT", "5m")]["price"] == 101.0


def test_build_entry_insights_ignores_a_forming_candle_with_no_prior_close():
    """The exact real-world failure mode: a symbol whose only data so far
    is an in-progress (unclosed) candle must not produce an entry insight
    at all -- there's nothing settled yet to decide on."""
    main.on_tick("ETHUSDT", _tick(2000.0, kline_closed=False))

    with patch("main.build_entry_insight") as mock_build:
        insights = main._build_entry_insights({})

    mock_build.assert_not_called()
    assert insights == {}


def test_build_entry_insights_uses_the_closed_candle_not_a_later_forming_one():
    """Once a candle has closed, entry decisions must be built from THAT
    settled state -- even if a noisier forming-candle tick arrived after
    it and is sitting in _latest_indicators when the decision loop polls."""
    main.on_tick("ETHUSDT", _tick(2000.0, kline_closed=True))
    main.on_tick("ETHUSDT", _tick(2500.0, kline_closed=False))  # noisy mid-candle spike

    with patch("main.build_entry_insight", return_value={"spot": {}, "derivative": {}}) as mock_build:
        main._build_entry_insights({})

    assert mock_build.call_count == 1
    called_price = mock_build.call_args[0][1]
    assert called_price == 2000.0  # the closed candle's price, not the 2500.0 spike
