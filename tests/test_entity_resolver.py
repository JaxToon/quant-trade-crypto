"""
Tests for knowledge/entity_resolver.py.

Regression guard for a real coverage gap found while reconstructing
Macro_Event edges after a Neo4j data-loss incident: TOKEN_ALIAS_MAP only
covered 5 of the 16 actual watchlist symbols (config/settings.py's
CRYPTO_WATCHLIST + STOCK_WATCHLIST) -- ADA/AVAX/DOT and every stock ticker
were unresolvable, so news mentioning them silently produced zero edges
no matter how clearly the article named the token.
"""
from __future__ import annotations

from config import settings
from knowledge.entity_resolver import resolve_entity, resolve_protocol, resolve_token


class TestFullWatchlistCoverage:
    """Every symbol actually traded should be resolvable by at least one
    reasonable alias -- the bare ticker itself, lowercased."""

    def test_every_crypto_watchlist_symbol_has_some_alias(self):
        for symbol in settings.CRYPTO_WATCHLIST:
            base = symbol.replace("USDT", "").lower()
            assert resolve_token(base) == symbol, f"{symbol} has no resolvable alias for {base!r}"

    def test_every_stock_watchlist_symbol_resolves_via_bare_ticker(self):
        for symbol in settings.STOCK_WATCHLIST:
            assert resolve_token(symbol.lower()) == symbol, f"{symbol} has no resolvable alias"


class TestResolveToken:
    def test_previously_missing_crypto_symbols_now_resolve(self):
        assert resolve_token("avax") == "AVAXUSDT"
        assert resolve_token("$avax") == "AVAXUSDT"
        assert resolve_token("avalanche") == "AVAXUSDT"
        assert resolve_token("ada") == "ADAUSDT"
        assert resolve_token("cardano") == "ADAUSDT"
        assert resolve_token("dot") == "DOTUSDT"
        assert resolve_token("polkadot") == "DOTUSDT"

    def test_previously_missing_stock_symbols_now_resolve(self):
        assert resolve_token("aapl") == "AAPL"
        assert resolve_token("apple") == "AAPL"
        assert resolve_token("tsla") == "TSLA"
        assert resolve_token("tesla") == "TSLA"
        assert resolve_token("meta") == "META"
        assert resolve_token("facebook") == "META"

    def test_case_insensitive(self):
        assert resolve_token("AVAX") == "AVAXUSDT"
        assert resolve_token("Bitcoin") == "BTCUSDT"

    def test_unknown_mention_returns_none(self):
        assert resolve_token("dogecoin") is None

    def test_existing_aliases_unaffected(self):
        """Guard against the expansion accidentally changing pre-existing,
        already-working mappings."""
        assert resolve_token("bitcoin") == "BTCUSDT"
        assert resolve_token("btc") == "BTCUSDT"
        assert resolve_token("ethereum") == "ETHUSDT"
        assert resolve_token("xrp") == "XRPUSDT"
        assert resolve_token("ripple") == "XRPUSDT"


class TestResolveEntity:
    def test_resolves_token_type(self):
        assert resolve_entity("avax", "CRYPTO") == ("Token", "AVAXUSDT")

    def test_resolves_protocol_type(self):
        assert resolve_entity("uniswap", "ORG") == ("Protocol", "uniswap")

    def test_unresolvable_returns_none(self):
        assert resolve_entity("some random company", "ORG") is None
