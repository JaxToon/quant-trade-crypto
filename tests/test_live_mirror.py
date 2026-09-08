"""
Tests for strategy/live_mirror.py — the layer that reproduces Path 1's paper
decisions as real orders.

The design contract is that the mirror DECIDES NOTHING. PaperTradingLedger is
the validated brain; the mirror only reproduces a PaperPosition it already
produced. Anything here that starts recomputing stops, targets or sizes would
mean the account is trading a different strategy than the one measured
(profitable 6/6 segments; live LONG n=82, +$166.85/trade).

The safety properties are the other half, and they exist because each one has
already failed once on testnet: a naked position after protection was
rejected, orders rejected for scientific-notation quantities, and a system
that could reach real money through a single forgotten flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from strategy.live_mirror import LiveOrderMirror


@dataclass
class _Pos:
    symbol: str = "BTCUSDT"
    direction: str = "long"
    entry_price: float = 80_000.0
    stop_loss: float = 78_000.0
    target: float = 84_000.0
    quantity: float = 0.0025


class _Client:
    API_URL = "https://testnet.binance.vision/api"

    def __init__(self, usdt=1_000_000.0, fail_protection=False, fail_entry=False):
        self.usdt = usdt
        self.fail_protection = fail_protection
        self.fail_entry = fail_entry
        self.created, self.oco, self.cancelled = [], [], []

    def get_account(self):
        return {"balances": [{"asset": "USDT", "free": str(self.usdt), "locked": "0"},
                             {"asset": "BTC", "free": "0", "locked": "0"}]}

    def get_symbol_info(self, symbol):
        return {"filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.00001"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "NOTIONAL", "minNotional": "5.0"},
        ]}

    def get_open_orders(self, symbol=None):
        return []

    def cancel_order(self, symbol, orderId):
        self.cancelled.append(orderId)

    def create_order(self, **kw):
        if self.fail_entry:
            raise RuntimeError("entry rejected")
        self.created.append(kw)
        return {"orderId": 1, "fills": [{"price": "80000.0", "qty": kw["quantity"]}]}

    def create_oco_order(self, **kw):
        if self.fail_protection:
            raise RuntimeError("APIError(code=-1013): Filter failure: PRICE_FILTER")
        self.oco.append(kw)
        return {"orderReports": [{"orderId": 2}, {"orderId": 3}]}


def _mirror(client):
    m = LiveOrderMirror.__new__(LiveOrderMirror)
    m.enabled = True
    m._client = client
    m._filters = {}
    from datetime import date
    m._day = date.today()
    m._day_start_balance = None
    m._halted = False
    m._halt_reason = ""
    return m


# ── the contract: mirror what the ledger decided, unchanged ───────────────

def test_places_exactly_the_ledgers_quantity_stop_and_target():
    c = _Client()
    m = _mirror(c)
    pos = _Pos()
    res = m.mirror_open(pos)

    assert res.ok, res.reason
    assert c.created[0]["quantity"] == "0.00250"
    o = c.oco[0]
    assert float(o["abovePrice"]) == pytest.approx(pos.target)
    assert float(o["belowStopPrice"]) == pytest.approx(pos.stop_loss)


def test_quantity_and_prices_are_fixed_point_strings():
    """A float would serialise as 8e-05 and be rejected with -1100."""
    c = _Client()
    m = _mirror(c)
    m.mirror_open(_Pos(quantity=0.00008))
    q = c.created[0]["quantity"]
    assert isinstance(q, str) and "e" not in q.lower(), q
    for k in ("abovePrice", "belowStopPrice", "belowPrice"):
        assert "e" not in c.oco[0][k].lower()


# ── safety ────────────────────────────────────────────────────────────────

def test_filled_entry_is_flattened_when_protection_fails():
    """The naked-position failure, observed for real on testnet 2026-09-08."""
    c = _Client(fail_protection=True)
    m = _mirror(c)
    res = m.mirror_open(_Pos())

    assert not res.ok
    closes = [o for o in c.created if o.get("side") == "SELL" and o.get("type") == "MARKET"]
    assert closes, "an entry that cannot be protected MUST be flattened"


def test_refuses_when_notional_exceeds_free_balance():
    """Spot cannot borrow. The ledger sizes with leverage (median 4.88x
    notional per $1), so this rejection is routine and must explain itself."""
    c = _Client(usdt=100.0)
    m = _mirror(c)
    res = m.mirror_open(_Pos())          # notional 0.0025 * 80000 = $200

    assert not res.ok
    assert "200" in res.reason and "100" in res.reason, res.reason


def test_refuses_when_the_stop_leg_would_fail_min_notional():
    c = _Client()
    m = _mirror(c)
    res = m.mirror_open(_Pos(quantity=0.00005, stop_loss=78_000.0))  # stop leg $3.90 < $5
    assert not res.ok and "minNotional" in res.reason


def test_shorts_are_refused_not_silently_sold():
    """A spot SELL for a 'short' would dump an asset the account may not hold."""
    m = _mirror(_Client())
    res = m.mirror_open(_Pos(direction="short"))
    assert not res.ok and "futures" in res.reason


def test_disabled_mirror_does_nothing():
    c = _Client()
    m = _mirror(c)
    m.enabled = False
    res = m.mirror_open(_Pos())
    assert not res.ok and c.created == []


# ── kill switch ───────────────────────────────────────────────────────────

def test_daily_loss_limit_halts_new_entries():
    """Path 1 has never had a loss limit; DAILY_DRAWDOWN_LIMIT lives only in
    execution.py (Path 2, disabled)."""
    c = _Client(usdt=1_000_000.0)
    m = _mirror(c)
    m.mirror_open(_Pos())                      # first call records the day's opening balance
    c.usdt = 1_000_000.0 * 0.90                # -10%, past the 5% default
    res = m.mirror_open(_Pos())
    assert not res.ok and "halted" in res.reason.lower()
