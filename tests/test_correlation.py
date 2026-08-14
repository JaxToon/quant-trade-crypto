"""Tests for strategy/correlation.py's get_correlation() lookup."""
from __future__ import annotations

from strategy.correlation import get_correlation


def test_self_correlation_is_one():
    assert get_correlation("BTCUSDT", "BTCUSDT") == 1.0
    assert get_correlation("AAPL", "AAPL") == 1.0


def test_lookup_is_symmetric():
    assert get_correlation("BTCUSDT", "ETHUSDT") == get_correlation("ETHUSDT", "BTCUSDT")
    assert get_correlation("AAPL", "AMZN") == get_correlation("AMZN", "AAPL")


def test_known_crypto_pair_returns_expected_value():
    assert get_correlation("BTCUSDT", "ETHUSDT") == 0.856


def test_known_stock_pair_returns_expected_value():
    assert get_correlation("NVDA", "AVGO") == 0.582


def test_crypto_and_stock_symbols_never_cross_contaminate():
    """Symbol namespaces never overlap ("BTCUSDT" vs "AAPL") — an
    unrecognized cross-asset-class pair must return 0.0, not raise or
    accidentally match something in the wrong matrix."""
    assert get_correlation("BTCUSDT", "AAPL") == 0.0
    assert get_correlation("AAPL", "ETHUSDT") == 0.0


def test_genuinely_unknown_pair_returns_zero():
    assert get_correlation("BTCUSDT", "NOTASYMBOL") == 0.0
