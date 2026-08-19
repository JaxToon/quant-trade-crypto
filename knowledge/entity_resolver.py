"""
Maps raw entity mentions to canonical graph node IDs.
Handles aliases like "Bitcoin", "BTC", "$BTC" → "BTCUSDT".
"""
from __future__ import annotations

# Maps any alias (lowercase) → canonical symbol used in Neo4j Token nodes
#
# Was missing ADA/AVAX/DOT and every stock ticker entirely -- only 5 of the
# 16 actual watchlist symbols (config/settings.py's CRYPTO_WATCHLIST +
# STOCK_WATCHLIST) could ever resolve, so any news article mentioning e.g.
# "AVAX" or "AAPL" silently produced zero IMPACTS/COMPETES_WITH edges no
# matter how clearly it named the token -- not a bug in extraction, a gap
# in the lookup table itself. Found while reconstructing Macro_Event edges
# after a Neo4j data-loss incident; fixed here since it also affects all
# future live NLP extraction, not just the reconstruction.
TOKEN_ALIAS_MAP: dict[str, str] = {
    "bitcoin": "BTCUSDT", "btc": "BTCUSDT", "$btc": "BTCUSDT",
    "ethereum": "ETHUSDT", "eth": "ETHUSDT", "$eth": "ETHUSDT", "ether": "ETHUSDT",
    "solana": "SOLUSDT", "sol": "SOLUSDT", "$sol": "SOLUSDT",
    "bnb": "BNBUSDT", "binance coin": "BNBUSDT", "$bnb": "BNBUSDT",
    "xrp": "XRPUSDT", "$xrp": "XRPUSDT", "ripple": "XRPUSDT",
    "cardano": "ADAUSDT", "ada": "ADAUSDT", "$ada": "ADAUSDT",
    "avalanche": "AVAXUSDT", "avax": "AVAXUSDT", "$avax": "AVAXUSDT",
    "polkadot": "DOTUSDT", "dot": "DOTUSDT", "$dot": "DOTUSDT",
    # Stocks: ticker doubles as the canonical symbol already (see
    # config/settings.py's STOCK_WATCHLIST), so just alias common names.
    "apple": "AAPL", "aapl": "AAPL", "$aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "$msft": "MSFT",
    "alphabet": "GOOGL", "google": "GOOGL", "googl": "GOOGL", "$googl": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN", "$amzn": "AMZN",
    "nvidia": "NVDA", "nvda": "NVDA", "$nvda": "NVDA",
    "meta": "META", "meta platforms": "META", "facebook": "META", "$meta": "META",
    "tesla": "TSLA", "tsla": "TSLA", "$tsla": "TSLA",
    "broadcom": "AVGO", "avgo": "AVGO", "$avgo": "AVGO",
}

# Maps protocol/chain mentions → canonical Protocol pid
PROTOCOL_ALIAS_MAP: dict[str, str] = {
    "ethereum network": "ethereum-chain", "eth mainnet": "ethereum-chain",
    "solana network": "solana-chain", "solana blockchain": "solana-chain",
    "bnb chain": "bnb-chain", "binance smart chain": "bnb-chain", "bsc": "bnb-chain",
    "uniswap": "uniswap",
    "aave": "aave",
}


def resolve_token(mention: str) -> str | None:
    """Return canonical token symbol or None if not recognized."""
    return TOKEN_ALIAS_MAP.get(mention.lower().strip())


def resolve_protocol(mention: str) -> str | None:
    return PROTOCOL_ALIAS_MAP.get(mention.lower().strip())


def resolve_entity(mention: str, label: str) -> tuple[str, str] | None:
    """
    Returns (node_type, canonical_id) or None.
    label: spaCy entity label or custom label from NER.
    """
    token = resolve_token(mention)
    if token:
        return ("Token", token)
    proto = resolve_protocol(mention)
    if proto:
        return ("Protocol", proto)
    # Unknown named entities treated as Macro_Event candidates by caller
    return None
