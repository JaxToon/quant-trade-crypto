from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase, Driver
from config import settings

logger = logging.getLogger(__name__)

# The driver logs a WARNING via this logger whenever a query references a
# property key the DBMS hasn't seen on any node yet (e.g. Trade.timestamp
# before the first Trade node is ever created) — a schema-awareness notice,
# not a query error; run_read/run_write still succeed either way. Silenced
# here since it's expected and reappears on every dashboard poll until the
# first Trade node exists.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


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

    # ── Graph pruning (unbounded growth control) ───────────────────────────
    #
    # Macro_Event nodes have no cap or expiry otherwise: one new node per
    # qualifying news article, forever (see knowledge/graph_populator.py),
    # and apply_temporal_decay() above only ever weakens edges, never
    # removes them — at the configured TEMPORAL_DECAY_GAMMA (0.001/day),
    # decay alone is far too slow to matter (90 days only takes a 0.8
    # confidence down to ~0.73). These two methods are the actual
    # growth-control mechanism, meant to run on a slow (daily) schedule.

    def prune_old_macro_events(self, max_age_days: int) -> int:
        """
        Deletes Macro_Event nodes (and all their relationships, via
        DETACH DELETE) older than max_age_days. Uses
        coalesce(created_at, recovered_at.epochMillis) since nodes restored
        via forensic recovery (the 2026-08-19 Neo4j data-loss incident)
        have no created_at — recovered_at is a fair substitute "born"
        timestamp for pruning purposes, so recovered nodes aren't kept
        forever just because their true origin is unknown. A node with
        neither property is left alone (age unknown, not assumed stale).

        The .epochMillis is load-bearing, not decorative: created_at is
        stored as a plain epoch-millis INTEGER (via timestamp()), but
        recovered_at was written as a native Neo4j ZONED DATETIME by the
        forensic reconstruction script. Cypher's `<` on an INTEGER vs a
        DATETIME returns NULL (silently excluded from WHERE) rather than
        raising -- found live: coalesce(created_at, recovered_at) matched
        exactly the 113 nodes with created_at and NONE of the 273
        recovered nodes, even at max_age_days=0, making recovered nodes
        silently un-prunable forever -- the exact opposite of this
        method's documented intent. .epochMillis normalizes recovered_at
        to the same INTEGER type before compare.
        """
        cypher = """
        MATCH (e:Macro_Event)
        WITH e, coalesce(e.created_at, e.recovered_at.epochMillis) AS born_at_millis
        WHERE born_at_millis < timestamp() - ($max_age_days * 86400000)
        WITH collect(e) AS stale, count(e) AS n
        FOREACH (e IN stale | DETACH DELETE e)
        RETURN n AS deleted
        """
        rows = self.run_write(cypher, max_age_days=max_age_days)
        return rows[0]["deleted"] if rows else 0

    def prune_decayed_edges(self, min_confidence: float) -> int:
        """
        Drops relationships whose confidence_score has decayed (via
        apply_temporal_decay) below min_confidence -- functionally dead
        weight at that point, contributing negligible signal to anything
        that reads confidence_score. Secondary cleanup; at the current
        decay rate this rarely fires on its own (see module note above),
        prune_old_macro_events is the primary growth-control lever.
        """
        cypher = """
        MATCH ()-[r]->()
        WHERE r.confidence_score IS NOT NULL AND r.confidence_score < $min_confidence
        WITH collect(r) AS stale, count(r) AS n
        FOREACH (r IN stale | DELETE r)
        RETURN n AS deleted
        """
        rows = self.run_write(cypher, min_confidence=min_confidence)
        return rows[0]["deleted"] if rows else 0

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

    def get_all_tokens(self, asset_type: str | None = None) -> list[dict]:
        """asset_type filters to only crypto or stock tokens — pass None for all."""
        if asset_type:
            return self.run_read("MATCH (t:Token {asset_type: $asset_type}) RETURN t", asset_type=asset_type)
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
