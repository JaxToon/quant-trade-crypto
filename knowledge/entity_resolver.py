"""
Maps raw entity mentions to canonical graph node IDs.
Handles aliases like "Bitcoin", "BTC", "$BTC" → "BTCUSDT".
"""
from __future__ import annotations

# Maps any alias (lowercase) → canonical symbol used in Neo4j Token nodes
TOKEN_ALIAS_MAP: dict[str, str] = {
    "bitcoin": "BTCUSDT", "btc": "BTCUSDT", "$btc": "BTCUSDT",
    "ethereum": "ETHUSDT", "eth": "ETHUSDT", "$eth": "ETHUSDT", "ether": "ETHUSDT",
    "solana": "SOLUSDT", "sol": "SOLUSDT", "$sol": "SOLUSDT",
    "bnb": "BNBUSDT", "binance coin": "BNBUSDT", "$bnb": "BNBUSDT",
    "xrp": "XRPUSDT", "$xrp": "XRPUSDT", "ripple": "XRPUSDT",
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
