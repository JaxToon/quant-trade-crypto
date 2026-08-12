from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase, Driver
from config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Thread-safe singleton wrapper around the Neo4j bolt driver."""

    _instance: "Neo4jClient | None" = None

    def __init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        logger.info("Connected to Neo4j at %s", settings.NEO4J_URI)

    @classmethod
    def get(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def session(self):
        with self._driver.session() as s:
            yield s

    # ── Generic read/write helpers ─────────────────────────────────────────

    def run_write(self, cypher: str, **params) -> list[dict]:
        with self.session() as s:
            result = s.execute_write(lambda tx: list(tx.run(cypher, **params)))
        return [dict(r) for r in result]

    def run_read(self, cypher: str, **params) -> list[dict]:
        with self.session() as s:
            result = s.execute_read(lambda tx: list(tx.run(cypher, **params)))
        return [dict(r) for r in result]

    # ── Node operations ────────────────────────────────────────────────────

    def upsert_token(self, symbol: str, name: str, **attrs) -> None:
        """Create or update a Token node (crypto asset)."""
        cypher = """
        MERGE (t:Token {symbol: $symbol})
        SET t.name = $name, t += $attrs, t.updated_at = timestamp()
        """
        self.run_write(cypher, symbol=symbol, name=name, attrs=attrs)

    def upsert_protocol(self, pid: str, name: str, **attrs) -> None:
        """Create or update a Protocol node (DeFi protocol, L1/L2 chain)."""
        cypher = """
        MERGE (p:Protocol {pid: $pid})
        SET p.name = $name, p += $attrs, p.updated_at = timestamp()
        """
        self.run_write(cypher, pid=pid, name=name, attrs=attrs)

    def upsert_macro_event(self, eid: str, description: str, **attrs) -> None:
        cypher = """
        MERGE (e:Macro_Event {eid: $eid})
        SET e.description = $description, e += $attrs, e.created_at = timestamp()
        """
        self.run_write(cypher, eid=eid, description=description, attrs=attrs)

    # ── Edge operations ────────────────────────────────────────────────────

    def upsert_edge(
        self,
        src_label: str,
        src_key: str,
        src_val: str,
        rel_type: str,
        dst_label: str,
        dst_key: str,
        dst_val: str,
        confidence: float = 1.0,
        **edge_attrs,
    ) -> None:
        """Generic upsert for any relationship type with a confidence score."""
        cypher = f"""
        MATCH (a:{src_label} {{{src_key}: $src_val}})
        MATCH (b:{dst_label} {{{dst_key}: $dst_val}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.confidence_score = $confidence,
            r.updated_at = timestamp(),
            r += $edge_attrs
        """
        self.run_write(
            cypher,
            src_val=src_val,
            dst_val=dst_val,
            confidence=confidence,
            edge_attrs=edge_attrs,
        )

    def apply_temporal_decay(self, gamma: float) -> int:
        """Decay all edge weights by e^(-γ * days_since_update). Returns edges updated."""
        cypher = """
        MATCH ()-[r]->()
        WHERE r.confidence_score IS NOT NULL AND r.updated_at IS NOT NULL
        WITH r, (timestamp() - r.updated_at) / 86400000.0 AS days_old
        SET r.confidence_score = r.confidence_score * exp(-$gamma * days_old)
        RETURN count(r) AS updated
        """
        rows = self.run_write(cypher, gamma=gamma)
        return rows[0]["updated"] if rows else 0

    # ── Node attribute updates (market data) ──────────────────────────────

    def update_token_market_data(self, symbol: str, **market_attrs) -> None:
        cypher = """
        MATCH (t:Token {symbol: $symbol})
        SET t += $attrs, t.updated_at = timestamp()
        """
        self.run_write(cypher, symbol=symbol, attrs=market_attrs)

    def update_token_smc_data(self, symbol: str, smc_context: dict) -> None:
        """Update Token node with Smart Money Concept analysis data."""
        # Flatten nested dict for Neo4j properties
        smc_attrs = {
            'smc_signal': smc_context.get('smc_signal', 0.0),
            'smc_market_structure': smc_context.get('market_structure', 'unknown'),
            'smc_active_obs': smc_context.get('active_order_blocks', 0),
            'smc_active_fvgs': smc_context.get('active_fvgs', 0),
            'smc_liquidity_zones': smc_context.get('liquidity_zones', 0),
            'smc_in_bullish_ob': smc_context.get('in_bullish_ob', False),
            'smc_in_bearish_ob': smc_context.get('in_bearish_ob', False),
            'smc_in_bullish_fvg': smc_context.get('in_bullish_fvg', False),
            'smc_in_bearish_fvg': smc_context.get('in_bearish_fvg', False),
            'smc_fvg_signal': smc_context.get('fvg_signal', 0.0),
            'smc_fvg_direction': smc_context.get('direction', 'neutral'),
            'smc_fvg_confidence': smc_context.get('confidence', 0.0),
            'smc_timestamp': smc_context.get('timestamp', ''),
        }

        cypher = """
        MATCH (t:Token {symbol: $symbol})
        SET t += $smc_attrs, t.smc_updated_at = timestamp()
        """
        self.run_write(cypher, symbol=symbol, smc_attrs=smc_attrs)

    # ── Graph reads for GNN feature extraction ────────────────────────────

    def get_all_tokens(self) -> list[dict]:
        return self.run_read("MATCH (t:Token) RETURN t")

    def get_neighbours(self, symbol: str, hops: int = 2) -> list[dict]:
        """Return all nodes within `hops` of a token (supply-web traversal)."""
        cypher = f"""
        MATCH path = (t:Token {{symbol: $symbol}})-[*1..{hops}]-(n)
        RETURN DISTINCT n, relationships(path) AS rels
        """
        return self.run_read(cypher, symbol=symbol)

    def get_edge_list(self) -> list[dict]:
        """Return full edge list for PyG graph construction."""
        cypher = """
        MATCH (a)-[r]->(b)
        RETURN
            elementId(a) AS src_id, labels(a)[0] AS src_type, a.symbol AS src_symbol,
            elementId(b) AS dst_id, labels(b)[0] AS dst_type, b.symbol AS dst_symbol,
            type(r) AS rel_type,
            coalesce(r.confidence_score, 1.0) AS confidence
        """
        return self.run_read(cypher)

    # ── Backtrack helpers ─────────────────────────────────────────────────

    def log_trade(self, trade: dict[str, Any]) -> None:
        cypher = """
        CREATE (t:Trade {
            symbol: $symbol, direction: $direction,
            entry_price: $entry_price, exit_price: $exit_price,
            pnl_pct: $pnl_pct, outcome: $outcome,
            timestamp: timestamp()
        })
        """
        self.run_write(cypher, **trade)

    def get_recent_losses(self, limit: int = 20) -> list[dict]:
        return self.run_read(
            "MATCH (t:Trade {outcome: 'loss'}) RETURN t ORDER BY coalesce(t.timestamp, 0) DESC LIMIT $limit",
            limit=limit,
        )
