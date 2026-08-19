"""
Tests for graph/schema.py's concurrent-startup race-condition handling.

Regression guard for a real crash: every CONSTRAINT/INDEX statement
already says "IF NOT EXISTS", but that only makes a single call
idempotent -- it doesn't protect against two separate processes starting
up at once (crypto + stock concurrently, exactly the README's documented
"run two separate processes" pattern). Both check "does this exist yet"
before either commits, both try to create it, and the loser gets a real
EquivalentSchemaRuleAlreadyExists ClientError. Reproduced live: starting
crypto and stock ~10s apart crashed the crypto process with an unhandled
traceback during initialize_schema().
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import ClientError

from graph.schema import _run_schema_statement


def _client_error(code: str) -> ClientError:
    exc = ClientError("some message")
    exc._neo4j_code = code
    return exc


def test_already_exists_race_is_swallowed():
    client = MagicMock()
    client.run_write.side_effect = _client_error("Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists")

    _run_schema_statement(client, "CREATE INDEX token_updated IF NOT EXISTS FOR (t:Token) ON (t.updated_at)")
    # No exception raised -- that's the whole point.


def test_constraint_already_exists_variant_is_also_swallowed():
    client = MagicMock()
    client.run_write.side_effect = _client_error("Neo.ClientError.Schema.ConstraintAlreadyExists")

    _run_schema_statement(client, "CREATE CONSTRAINT token_symbol IF NOT EXISTS FOR (t:Token) REQUIRE t.symbol IS UNIQUE")


def test_unrelated_client_error_still_raises():
    """Only the AlreadyExists race is benign -- a real schema problem
    (bad syntax, permissions, etc.) must not be silently swallowed."""
    client = MagicMock()
    client.run_write.side_effect = _client_error("Neo.ClientError.Statement.SyntaxError")

    with pytest.raises(ClientError):
        _run_schema_statement(client, "CREATE INDEX broken FOR (t:Token) ON (t.foo)")


def test_successful_statement_runs_normally():
    client = MagicMock()
    _run_schema_statement(client, "CREATE INDEX token_updated IF NOT EXISTS FOR (t:Token) ON (t.updated_at)")
    client.run_write.assert_called_once_with("CREATE INDEX token_updated IF NOT EXISTS FOR (t:Token) ON (t.updated_at)")
