"""
Regression tests for backtrack/paper_trading.py — covering the sizing and
risk-cap behavior found and fixed this project's history:
  - conviction-scaled sizing was inert (confidence never reached maybe_open,
    see test_entry_insight.py) — here we test the sizing MATH itself, given
    a correct confidence input, actually scales risk/quantity as documented.
  - the portfolio correlation cap, added to stop several correlated
    positions silently stacking into one oversized aggregate bet.
"""
from __future__ import annotations

import pytest

from backtrack.paper_trading import PaperTradingLedger


def _rec(direction="long", confidence=0.0, entry_zone=(100.0, 100.0), stop_loss=95.0, target=110.0, leverage=1):
    return {
        "actionable": True,
        "direction": direction,
        "entry_zone": list(entry_zone),
        "stop_loss": stop_loss,
        "target": target,
        "confidence": confidence,
        "suggested_leverage": leverage,
    }


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "test_ledger.json")
    return PaperTradingLedger(path=path, starting_balance=10_000.0)


def test_zero_confidence_sizes_at_base_risk(ledger):
    """confidence=0 -> conviction_multiplier=1.0 -> risk_amount = 2% of
    starting balance ($200) -> quantity = 200 / stop_distance(5) = 40."""
    ledger.maybe_open("BTCUSDT", "spot", _rec(confidence=0.0), timeframe="1h")
    pos = ledger.open_positions[("BTCUSDT", "spot", "1h")]
    assert pos.quantity == pytest.approx(40.0, rel=1e-6)


def test_max_confidence_sizes_larger_than_zero_confidence(ledger):
    """This is the actual bug this project shipped with: confidence never
    reached here, so every trade — regardless of stated confidence — sized
    identically at the zero-confidence multiplier. A real fix must make
    high-confidence trades size strictly larger than low-confidence ones."""
    ledger.maybe_open("BTCUSDT", "spot", _rec(confidence=0.0), timeframe="1h")
    ledger.maybe_open("ETHUSDT", "spot", _rec(confidence=100.0), timeframe="1h")
    low_conf_qty = ledger.open_positions[("BTCUSDT", "spot", "1h")].quantity
    high_conf_qty = ledger.open_positions[("ETHUSDT", "spot", "1h")].quantity
    assert high_conf_qty > low_conf_qty
    # confidence=100 -> conviction_multiplier=2.5 -> risk=$500 -> qty=100
    assert high_conf_qty == pytest.approx(100.0, rel=1e-6)


def test_stop_too_tight_for_fees_is_rejected(ledger):
    """A stop closer than MIN_STOP_DISTANCE_PCT (~0.667%) should be
    rejected with a human-readable reason, not silently sized into a
    guaranteed-losing-after-fees trade."""
    tight_rec = _rec(confidence=50.0, entry_zone=(100.0, 100.0), stop_loss=99.95, target=110.0)  # 0.05% stop
    reason = ledger.maybe_open("BTCUSDT", "spot", tight_rec, timeframe="1h")
    assert reason is not None
    assert "Stop" in reason
    assert ("BTCUSDT", "spot", "1h") not in ledger.open_positions


def test_duplicate_open_is_a_noop(ledger):
    """Opening the same (symbol, market, timeframe) twice must not create
    a second position or resize the existing one."""
    ledger.maybe_open("BTCUSDT", "spot", _rec(confidence=50.0), timeframe="1h")
    original_qty = ledger.open_positions[("BTCUSDT", "spot", "1h")].quantity
    result = ledger.maybe_open("BTCUSDT", "spot", _rec(confidence=100.0), timeframe="1h")
    assert result is None
    assert len(ledger.open_positions) == 1
    assert ledger.open_positions[("BTCUSDT", "spot", "1h")].quantity == original_qty


def test_rejects_long_setup_already_through_its_own_stop(ledger):
    """Live bug (2026-08-19): entry_insight recommends off the SMC/FVG
    order-block zone without checking current price, so a stale-but-still-
    "actionable" signal (price already broke through the order block) would
    open and then get instantly stopped out by the very next check_and_close
    — always a loss, since the fill price was never actually reachable.
    NVDA hit this three times in ~25 minutes before the guard was added."""
    rec = _rec(direction="long", confidence=78.0, entry_zone=(219.084999, 219.084999),
                stop_loss=217.524002, target=221.314995)
    reason = ledger.maybe_open("NVDA", "spot", rec, timeframe="1h", current_price=217.524002)
    assert reason is not None
    assert "invalid" in reason.lower()
    assert ("NVDA", "spot", "1h") not in ledger.open_positions


def test_rejects_short_setup_already_through_its_own_stop(ledger):
    rec = _rec(direction="short", confidence=78.0, entry_zone=(100.0, 100.0),
                stop_loss=105.0, target=90.0)
    reason = ledger.maybe_open("NVDA", "derivative", rec, timeframe="1h", current_price=105.0)
    assert reason is not None
    assert ("NVDA", "derivative", "1h") not in ledger.open_positions


def test_still_opens_normally_when_current_price_is_between_entry_and_stop(ledger):
    """The guard must only reject already-invalidated setups, not ordinary
    entries where price has moved a little but hasn't reached the stop."""
    rec = _rec(direction="long", confidence=50.0, entry_zone=(100.0, 100.0),
                stop_loss=95.0, target=110.0)
    reason = ledger.maybe_open("BTCUSDT", "spot", rec, timeframe="1h", current_price=98.0)
    assert reason is None
    assert ("BTCUSDT", "spot", "1h") in ledger.open_positions


def test_current_price_omitted_preserves_old_behavior(ledger):
    """Backtests never pass current_price — must behave exactly as before
    (no new rejection path activates when it's None)."""
    rec = _rec(direction="long", confidence=50.0, entry_zone=(100.0, 100.0),
                stop_loss=95.0, target=110.0)
    reason = ledger.maybe_open("BTCUSDT", "spot", rec, timeframe="1h")
    assert reason is None
    assert ("BTCUSDT", "spot", "1h") in ledger.open_positions


def test_correlated_positions_can_trip_the_portfolio_cap(ledger):
    """BTCUSDT/ETHUSDT are correlated at 0.856 (well above the 0.5
    threshold) in strategy/correlation.py. Two max-confidence spot
    positions ($500 risk each) sum to $1000, above the $800 (8% of $10k)
    portfolio correlated-risk cap — the second must be rejected."""
    r1 = ledger.maybe_open("BTCUSDT", "spot", _rec(confidence=100.0), timeframe="1h")
    r2 = ledger.maybe_open("ETHUSDT", "spot", _rec(confidence=100.0), timeframe="1h")
    assert r1 is None  # first one opens fine
    assert r2 is not None
    assert "correlat" in r2.lower()
    assert ("ETHUSDT", "spot", "1h") not in ledger.open_positions


def test_uncorrelated_positions_dont_trip_the_cap(ledger):
    """ADAUSDT/AVAXUSDT are correlated at 0.459 — below the 0.5 threshold
    — so they shouldn't count against each other's cap check even at max
    confidence each."""
    r1 = ledger.maybe_open("ADAUSDT", "spot", _rec(confidence=100.0), timeframe="1h")
    r2 = ledger.maybe_open("AVAXUSDT", "spot", _rec(confidence=100.0), timeframe="1h")
    assert r1 is None
    assert r2 is None
    assert ("ADAUSDT", "spot", "1h") in ledger.open_positions
    assert ("AVAXUSDT", "spot", "1h") in ledger.open_positions
