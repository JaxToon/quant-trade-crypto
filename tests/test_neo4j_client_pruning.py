"""
Tests for graph/neo4j_client.py's pruning methods (prune_old_macro_events,
prune_decayed_edges) — the graph growth-control mechanism added after
confirming Macro_Event nodes have no cap or expiry otherwise (one new node
per qualifying news article, forever) and apply_temporal_decay() only ever
weakens edges, never removes them.

Never connects to a real Neo4j instance — Neo4jClient.__init__ opens a
real driver connection, so instances here are built via __new__ (bypassing
__init__) with run_write mocked directly, matching the project's
established pattern of patching at the network-I/O boundary rather than
faking the wire protocol.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from graph.neo4j_client import Neo4jClient


def _client_with_mocked_write(return_rows):
    client = object.__new__(Neo4jClient)
    client.run_write = MagicMock(return_value=return_rows)
    return client


class TestPruneOldMacroEvents:
    def test_returns_deleted_count(self):
        client = _client_with_mocked_write([{"deleted": 12}])
        assert client.prune_old_macro_events(90) == 12

    def test_passes_max_age_days_as_parameter(self):
        client = _client_with_mocked_write([{"deleted": 0}])
        client.prune_old_macro_events(90)
        _, kwargs = client.run_write.call_args
        assert kwargs["max_age_days"] == 90

    def test_query_uses_detach_delete(self):
        """Must remove the node's relationships too, not just the node —
        DELETE alone on a node with edges raises in Neo4j."""
        client = _client_with_mocked_write([{"deleted": 0}])
        client.prune_old_macro_events(90)
        cypher = client.run_write.call_args[0][0]
        assert "DETACH DELETE" in cypher

    def test_query_falls_back_to_recovered_at(self):
        """Forensically-recovered nodes have no created_at -- must not be
        exempt from pruning forever just because their true age is unknown."""
        client = _client_with_mocked_write([{"deleted": 0}])
        client.prune_old_macro_events(90)
        cypher = client.run_write.call_args[0][0]
        assert "coalesce(e.created_at, e.recovered_at)" in cypher

    def test_empty_result_returns_zero(self):
        client = _client_with_mocked_write([])
        assert client.prune_old_macro_events(90) == 0


class TestPruneDecayedEdges:
    def test_returns_deleted_count(self):
        client = _client_with_mocked_write([{"deleted": 7}])
        assert client.prune_decayed_edges(0.05) == 7

    def test_passes_min_confidence_as_parameter(self):
        client = _client_with_mocked_write([{"deleted": 0}])
        client.prune_decayed_edges(0.05)
        _, kwargs = client.run_write.call_args
        assert kwargs["min_confidence"] == 0.05

    def test_only_targets_edges_with_confidence_below_threshold(self):
        client = _client_with_mocked_write([{"deleted": 0}])
        client.prune_decayed_edges(0.05)
        cypher = client.run_write.call_args[0][0]
        assert "r.confidence_score < $min_confidence" in cypher
        assert "r.confidence_score IS NOT NULL" in cypher  # never touch untagged relationships

    def test_empty_result_returns_zero(self):
        client = _client_with_mocked_write([])
        assert client.prune_decayed_edges(0.05) == 0
