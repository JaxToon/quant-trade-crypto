"""
Regression tests for the order-formatting bugs found on Binance testnet,
2026-09-08 — the first time this project ever sent an order.

Three traps, all of which bite hardest on SMALL accounts, i.e. exactly the
configuration a first real-money deployment would use:

  1. Python renders small floats in scientific notation ("8e-05"), and
     Binance rejects that with -1100 "Illegal characters in parameter
     'quantity'". Passing a raw float meant every order on a small account
     would fail.
  2. Quantity was rounded to a hardcoded 5 decimal places ("simplified",
     per the old comment) rather than the symbol's real LOT_SIZE step.
  3. MIN_NOTIONAL was never checked against the PROTECTIVE leg. A quantity
     that clears the filter at entry can fail it on a stop-loss placed
     below entry — which opens the position and then silently fails to
     protect it. Refusing the trade is the only safe response.
"""
from __future__ import annotations

import pytest

from strategy.execution import ExecutionEngine


class _StubClient:
    """Minimal stand-in: only the symbol-info call _compute_quantity needs."""

    def __init__(self, step="0.00001", tick="0.01", min_notional="5.0"):
        self._f = {"step": step, "tick": tick, "mn": min_notional}
        self.calls = 0

    def get_symbol_info(self, symbol):
        self.calls += 1
        return {"filters": [
            {"filterType": "LOT_SIZE", "stepSize": self._f["step"]},
            {"filterType": "PRICE_FILTER", "tickSize": self._f["tick"]},
            {"filterType": "NOTIONAL", "minNotional": self._f["mn"]},
        ]}


def _engine(client):
    eng = ExecutionEngine.__new__(ExecutionEngine)   # skip __init__ (it builds a real Client)
    eng._client = client
    eng._filter_cache = {}
    return eng


# ── 1. scientific notation ────────────────────────────────────────────────

@pytest.mark.parametrize("value,increment,expected", [
    (8e-05, 1e-05, "0.00008"),     # the exact shape the exchange rejected
    (0.000123, 1e-05, "0.00012"),
    (0.5, 0.01, "0.50"),
    (123.0, 1.0, "123"),
])
def test_fmt_never_emits_scientific_notation(value, increment, expected):
    out = ExecutionEngine._fmt(value, increment)
    assert "e" not in out.lower(), f"{out} would be rejected with -1100"
    assert out == expected


def test_fmt_snaps_to_the_increment():
    # 0.000127 is not a multiple of 0.00001 -> must snap, not truncate blindly
    assert ExecutionEngine._fmt(0.000127, 1e-05) == "0.00013"


# ── 2. real LOT_SIZE, not a hardcoded 5dp ────────────────────────────────

def test_quantity_uses_the_symbols_actual_step_not_5dp():
    """A coarse-step symbol must not receive a 5-decimal quantity."""
    eng = _engine(_StubClient(step="1", tick="0.0001", min_notional="5.0"))
    q = eng._compute_quantity("DOGEUSDT", balance=10_000.0, price=0.25)
    assert "." not in q, f"stepSize=1 must yield a whole number, got {q}"


def test_symbol_filters_are_cached():
    client = _StubClient()
    eng = _engine(client)
    eng._compute_quantity("BTCUSDT", 10_000.0, 80_000.0)
    eng._compute_quantity("BTCUSDT", 10_000.0, 80_000.0)
    assert client.calls == 1, "filters should be fetched once per symbol"


# ── 3. the protective leg must clear MIN_NOTIONAL ────────────────────────

def test_refuses_when_the_stop_leg_would_fail_min_notional():
    """The trap that leaves a position naked: entry clears the filter, the
    stop-loss below it does not, so the entry fills and the protection
    silently cannot be placed."""
    eng = _engine(_StubClient(min_notional="5.0"))
    # Sized so entry notional is just over $5 but the stop leg falls under it.
    q = eng._compute_quantity("BTCUSDT", balance=260.0, price=80_000.0)
    assert q == "", "must refuse rather than open an unprotectable position"


def test_allows_when_both_legs_clear():
    eng = _engine(_StubClient(min_notional="5.0"))
    q = eng._compute_quantity("BTCUSDT", balance=10_000.0, price=80_000.0)
    assert q and float(q) > 0
    assert "e" not in q.lower()


def test_returns_empty_string_not_zero_when_unsizeable():
    """Callers check `if not quantity` — a 0.0 float would also be falsy, but
    the contract is a string, and mixing the two silently broke the guard."""
    eng = _engine(_StubClient(step="1", min_notional="5.0"))
    q = eng._compute_quantity("BTCUSDT", balance=0.01, price=80_000.0)
    assert q == "" and isinstance(q, str)


# ── 4. the naked-position bug ─────────────────────────────────────────────
# Observed live on testnet 2026-09-08: the entry market order FILLED, the
# protective order was rejected (-1013 PRICE_FILTER), the exception was
# swallowed, execute() returned None — and the account was left holding the
# asset with no stop, no take-profit, and no Position record. The system did
# not know it owned anything. With real money that is unbounded downside from
# a failure the caller never sees.


class _FailingProtectionClient(_StubClient):
    """Entry fills; every subsequent protective order is rejected."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.orders = []
        self.cancelled = []
        self._entry_done = False

    def get_open_orders(self, symbol):
        return []

    def cancel_order(self, symbol, orderId):
        self.cancelled.append(orderId)

    def create_order(self, **kw):
        self.orders.append(kw)
        if not self._entry_done:
            self._entry_done = True
            return {"orderId": 1, "fills": [{"price": "80000.0", "qty": kw["quantity"]}]}
        if kw.get("type") == "MARKET":          # the rollback close
            return {"orderId": 99, "fills": [{"price": "79990.0", "qty": kw["quantity"]}]}
        from binance.exceptions import BinanceAPIException
        raise BinanceAPIException(type("R", (), {"status_code": 400, "text": ""})(), 400,
                                  '{"code":-1013,"msg":"Filter failure: PRICE_FILTER"}')


def test_filled_entry_is_flattened_when_protection_fails():
    """The entry must not be left naked. If the stop cannot be placed, the
    position is closed immediately — accepting a small spread/fee loss,
    because the alternative has no bounded downside."""
    from strategy.consensus import TradeDecision
    from strategy.agents import AgentDebate

    client = _FailingProtectionClient()
    eng = _engine(client)
    eng._halted = False
    eng._open_positions = {}
    eng._daily_pnl = 0.0
    from datetime import date
    eng._trading_day = date.today()
    eng._get_account_balance = lambda: 10_000.0
    eng._get_price = lambda s: 80_000.0
    eng._save_positions = lambda: None

    d = TradeDecision(symbol="BTCUSDT", p_trade=0.99, execute=True, direction="LONG",
                      debate=AgentDebate(symbol="BTCUSDT", bull_score=0.9, bull_reasoning="x",
                                         bear_score=0.1, bear_reasoning="x", c_risk=0.0,
                                         critic_reasoning="x"),
                      gnn_signal=0.0)
    pos = eng.execute(d)

    assert pos is None, "execute() should report failure"
    closes = [o for o in client.orders if o.get("type") == "MARKET" and o.get("side") == "SELL"]
    assert closes, "a filled entry with failed protection MUST be flattened, not left naked"
    assert "BTCUSDT" not in eng._open_positions


def test_entry_price_averages_across_all_fills():
    """A market order that walks the book returns several fills. Reading only
    fills[0] understates the entry and corrupts every SL/TP distance and PnL
    figure derived from it."""
    class MultiFill(_StubClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def get_open_orders(self, symbol):
            return []

        def create_order(self, **kw):
            self.n += 1
            if self.n == 1:
                return {"orderId": 1, "fills": [
                    {"price": "80000.0", "qty": "0.001"},
                    {"price": "80100.0", "qty": "0.001"},
                ]}
            return {"orderId": self.n}

    from strategy.consensus import TradeDecision
    from strategy.agents import AgentDebate
    from datetime import date

    client = MultiFill()
    eng = _engine(client)
    eng._halted = False
    eng._open_positions = {}
    eng._daily_pnl = 0.0
    eng._trading_day = date.today()
    eng._get_account_balance = lambda: 10_000.0
    eng._get_price = lambda s: 80_000.0
    eng._save_positions = lambda: None

    d = TradeDecision(symbol="BTCUSDT", p_trade=0.99, execute=True, direction="LONG",
                      debate=AgentDebate(symbol="BTCUSDT", bull_score=0.9, bull_reasoning="x",
                                         bear_score=0.1, bear_reasoning="x", c_risk=0.0,
                                         critic_reasoning="x"),
                      gnn_signal=0.0)
    pos = eng.execute(d)
    assert pos is not None
    assert pos.entry_price == pytest.approx(80_050.0), \
        f"expected VWAP 80050, got {pos.entry_price} (reading fills[0] only?)"
