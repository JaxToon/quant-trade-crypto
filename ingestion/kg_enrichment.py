"""
External knowledge graph enrichment from CoinGecko and DeFiLlama.

Writes structured, curated edges into Neo4j to supplement NLP-discovered
relationships. Much higher quality than NLP because these are authoritative
structural facts (e.g. "Uniswap runs on Ethereum" is not a guess from text).

Runs at startup + every 6 hours via scheduler.
"""
from __future__ import annotations

import logging
import time

import requests

from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Binance symbol → CoinGecko coin ID
COINGECKO_IDS: dict[str, str] = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
}

# CoinGecko platform key → our Neo4j Protocol pid
PLATFORM_TO_PID: dict[str, str] = {
    "ethereum":             "ethereum-chain",
    "solana":               "solana-chain",
    "binance-smart-chain":  "bnb-chain",
    "bsc":                  "bnb-chain",
}

# Native token for each chain (for SUPPLIES edges)
CHAIN_NATIVE_TOKEN: dict[str, str] = {
    "ethereum-chain": "ETHUSDT",
    "solana-chain":   "SOLUSDT",
    "bnb-chain":      "BNBUSDT",
}

# DeFi protocols we track in Neo4j
TRACKED_PROTOCOLS: dict[str, str] = {
    "uniswap": "uniswap",
    "aave":    "aave",
}

_CG_BASE = "https://api.coingecko.com/api/v3"
_DL_BASE = "https://api.llama.fi"


# ── CoinGecko ──────────────────────────────────────────────────────────────

def _enrich_coingecko(db: Neo4jClient) -> int:
    edges = 0

    # 1. BUILT_ON edges: token → chain it runs on
    for symbol, coin_id in COINGECKO_IDS.items():
        try:
            resp = requests.get(
                f"{_CG_BASE}/coins/{coin_id}",
                params={
                    "localization": "false", "tickers": "false",
                    "market_data": "false", "community_data": "false",
                    "developer_data": "false",
                },
                timeout=10,
            )
            if resp.status_code == 429:
                logger.warning("CoinGecko rate-limited, sleeping 60s")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()

            for platform_key in data.get("platforms", {}):
                pid = PLATFORM_TO_PID.get(platform_key.lower())
                if pid:
                    db.upsert_edge(
                        "Token", "symbol", symbol,
                        "BUILT_ON",
                        "Protocol", "pid", pid,
                        confidence=0.98,
                        source="coingecko",
                    )
                    edges += 1

            time.sleep(1.5)  # free tier: ~40 req/min
        except Exception as exc:
            logger.warning("CoinGecko enrichment failed for %s: %s", symbol, exc)

    # 2. COMPETES_WITH edges: all L1 tokens compete with each other
    l1_tokens = list(COINGECKO_IDS.keys())
    for i in range(len(l1_tokens)):
        for j in range(i + 1, len(l1_tokens)):
            ta, tb = l1_tokens[i], l1_tokens[j]
            try:
                db.upsert_edge(
                    "Token", "symbol", ta,
                    "COMPETES_WITH",
                    "Token", "symbol", tb,
                    confidence=0.75,
                    source="coingecko_category",
                )
                edges += 1
            except Exception:
                pass

    return edges


# ── DeFiLlama ─────────────────────────────────────────────────────────────

def _enrich_defillama(db: Neo4jClient) -> int:
    edges = 0
    try:
        resp = requests.get(f"{_DL_BASE}/protocols", timeout=15)
        resp.raise_for_status()
        protocols = resp.json()
    except Exception as exc:
        logger.warning("DeFiLlama fetch failed: %s", exc)
        return 0

    for proto in protocols:
        name = proto.get("name", "").lower().strip()
        pid = TRACKED_PROTOCOLS.get(name)
        if not pid:
            continue

        tvl = proto.get("tvl", 0) or 0
        chains = proto.get("chains", [])

        for chain in chains:
            chain_lower = chain.lower()
            if chain_lower in ("ethereum", "eth"):
                chain_pid = "ethereum-chain"
            elif chain_lower in ("solana", "sol"):
                chain_pid = "solana-chain"
            elif chain_lower in ("bsc", "binance", "bnb", "bnb chain"):
                chain_pid = "bnb-chain"
            else:
                continue

            # Protocol DEPENDS_ON the chain it runs on
            confidence = min(0.97, 0.6 + tvl / 5e10) if tvl else 0.7
            try:
                db.upsert_edge(
                    "Protocol", "pid", pid,
                    "DEPENDS_ON",
                    "Protocol", "pid", chain_pid,
                    confidence=confidence,
                    source="defillama",
                    tvl_usd=int(tvl),
                )
                edges += 1
            except Exception:
                pass

            # Native token SUPPLIES the protocol (e.g. ETH supplies Uniswap)
            native = CHAIN_NATIVE_TOKEN.get(chain_pid)
            if native:
                try:
                    db.upsert_edge(
                        "Token", "symbol", native,
                        "SUPPLIES",
                        "Protocol", "pid", pid,
                        confidence=0.92,
                        source="defillama",
                    )
                    edges += 1
                except Exception:
                    pass

    return edges


# ── Public entry point ─────────────────────────────────────────────────────

def run_enrichment() -> None:
    """Run a full external KG enrichment cycle. Safe to call repeatedly."""
    logger.info("Starting external KG enrichment (CoinGecko + DeFiLlama)...")
    db = Neo4jClient.get()
    cg_edges = _enrich_coingecko(db)
    dl_edges = _enrich_defillama(db)
    logger.info(
        "KG enrichment done: +%d CoinGecko edges, +%d DeFiLlama edges",
        cg_edges, dl_edges,
    )
