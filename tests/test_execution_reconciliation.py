"""
Tests for startup reconciliation between local position state and the
exchange (strategy/execution.py).

Why it exists: `_load_positions` restores from a JSON file, and nothing ever
compared that file to reality. The two drift for ordinary reasons — this
deployment restarted 39 times in one 25-day window, and a crash between an
order filling and the state being written is enough. The dangerous direction
is a position the file believes is guarded whose protective orders are gone:
naked, and silent.

Verified end-to-end on testnet 2026-09-08 (open -> cancel protection behind
the engine's back -> fresh engine -> protection restored). These tests pin
the behaviour so it cannot regress unnoticed.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from strategy.execution import ExecutionEngine, Position


class _ReconClient:
    def __init__(self, open_orders=None):
        self._open = open_orders or {}
        self.cancelled = []
        self.created = []

    def get_open_orders(self, symbol=None):
        if symbol is None:
            return [o for lst in self._open.values() for o in lst]
        return list(self._open.get(symbol, []))

    def cancel_order(self, symbol, orderId):
        self.cancelled.append((symbol, orderId))
        self._open[symbol] = [o for o in self._open.get(symbol, []) if o["orderId"] != orderId]

    def create_order(self, **kw):
        self.created.append(kw)
        return {"orderId": 1000 + len(self.created)}

    def get_symbol_info(self, symbol):
        return {"filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.00001"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "NOTIONAL", "minNotional": "5.0"},
        ]}


def _engine(client):
    eng = ExecutionEngine.__new__(ExecutionEngine)
    eng._client = client
    eng._filter_cache = {}
    eng._open_positions = {}
    return eng


def _pos(symbol="BTCUSDT", direction="LONG"):
    return Position(symbol=symbol, direction=direction, entry_price=80_000.0,
                    quantity=0.00247, sl_price=77_600.0, tp_price=84_800.0,
                    order_id="1", opened_at=datetime.utcnow())


def _prot(order_id, otype="STOP_LOSS_LIMIT", symbol="BTCUSDT"):
    # `symbol` matters: _known_order_symbols() reads it off the account-wide
    # open-orders call to find orphans belonging to no tracked position.
    # Omitting it made the stub silently unlike a real Binance order.
    return {"orderId": order_id, "type": otype, "side": "SELL", "symbol": symbol,
            "origQty": "0.00247", "price": "1", "stopPrice": "1", "status": "NEW"}


def test_naked_position_is_detected_and_reprotected():
    """The failure that motivated all of this: state says guarded, exchange
    says nothing is resting."""
    client = _ReconClient()                      # no open orders at all
    eng = _engine(client)
    eng._open_positions["BTCUSDT"] = _pos()

    report = eng.reconcile_with_exchange(auto_heal=True)

    assert report["unprotected"] == ["BTCUSDT"]
    assert report["reprotected"] == ["BTCUSDT"]
    types = {c["type"] for c in client.created}
    assert types == {"STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"}, \
        f"both protective legs must be restored, got {types}"


def test_reprotection_prices_snap_to_tick():
    """Re-placed orders go through the same tick-snapping as new ones; a raw
    float here would be rejected with -1013 exactly as it was on 2026-09-08."""
    client = _ReconClient()
    eng = _engine(client)
    eng._open_positions["BTCUSDT"] = _pos()

    eng.reconcile_with_exchange(auto_heal=True)

    for c in client.created:
        for field in ("stopPrice", "price"):
            v = c[field]
            assert isinstance(v, str) and "e" not in v.lower(), f"{field}={v!r}"
            assert abs(round(float(v) / 0.01) * 0.01 - float(v)) < 1e-6, f"{field}={v} off-tick"


def test_orphaned_protective_orders_are_cancelled():
    """Protective orders resting for a symbol we no longer track can fire
    later against an asset the system does not know it holds."""
    client = _ReconClient({"ETHUSDT": [_prot(7, symbol="ETHUSDT"),
                                   _prot(8, "TAKE_PROFIT_LIMIT", symbol="ETHUSDT")]})
    eng = _engine(client)                        # tracks nothing

    report = eng.reconcile_with_exchange(auto_heal=True)

    assert report["orphans"] == ["ETHUSDT"]
    assert sorted(o for _, o in client.cancelled) == [7, 8]


def test_healthy_state_changes_nothing():
    client = _ReconClient({"BTCUSDT": [_prot(1), _prot(2, "TAKE_PROFIT_LIMIT")]})
    eng = _engine(client)
    eng._open_positions["BTCUSDT"] = _pos()

    report = eng.reconcile_with_exchange(auto_heal=True)

    assert report["unprotected"] == [] and report["orphans"] == []
    assert client.created == [] and client.cancelled == []


def test_auto_heal_false_reports_without_touching_anything():
    """A read-only mode matters: an operator may want to see the divergence
    before anything acts on it."""
    client = _ReconClient()
    eng = _engine(client)
    eng._open_positions["BTCUSDT"] = _pos()

    report = eng.reconcile_with_exchange(auto_heal=False)

    assert report["unprotected"] == ["BTCUSDT"]
    assert report["reprotected"] == []
    assert client.created == []


def test_exchange_errors_are_collected_not_raised():
    """Reconciliation runs at startup; it must never prevent the process from
    coming up."""
    class Broken(_ReconClient):
        def get_open_orders(self, symbol=None):
            raise RuntimeError("network down")

    eng = _engine(Broken())
    eng._open_positions["BTCUSDT"] = _pos()

    report = eng.reconcile_with_exchange(auto_heal=True)
    assert report["errors"], "the failure should be reported, not swallowed silently"
