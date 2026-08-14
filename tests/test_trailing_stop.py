"""
Tests for the ATR-based trailing stop (backtrack/paper_trading.py).
PAPER_TRADING_TRAILING_STOP_ENABLED defaults to False — these tests
explicitly monkeypatch it on where needed, and also verify the off-by-
default behavior is genuinely a no-op, matching the "test before deploying
anything risk-affecting" discipline used throughout this project.
"""
from __future__ import annotations

import pytest

from backtrack.paper_trading import PaperTradingLedger
from config import settings


def _rec(direction="long", confidence=50.0, entry_zone=(100.0, 100.0), stop_loss=95.0, target=200.0, atr=2.0, leverage=1):
    """target is set far away (200) by default specifically so tests can
    drive price up through several ATRs without ever hitting a FIXED
    target first — isolating the trailing-stop exit path from the
    ordinary fixed-target exit path."""
    return {
        "actionable": True,
        "direction": direction,
        "entry_zone": list(entry_zone),
        "stop_loss": stop_loss,
        "target": target,
        "confidence": confidence,
        "atr": atr,
        "suggested_leverage": leverage,
    }


@pytest.fixture
def ledger(tmp_path):
    return PaperTradingLedger(path=str(tmp_path / "test_ledger.json"), starting_balance=10_000.0)


def test_disabled_by_default_never_sets_trailing_stop(ledger, monkeypatch):
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_STOP_ENABLED", False)
    ledger.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    pos = ledger.open_positions[("BTCUSDT", "spot", "1h")]
    # entry=100, stop=95, risk=5 -> way more than 1R of profit
    ledger.check_and_close({"BTCUSDT": 130.0})
    assert pos.trailing_stop is None


def test_activates_only_after_profit_reaches_activation_r(ledger, monkeypatch):
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ACTIVATION_R", 1.0)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ATR_MULTIPLE", 2.0)
    ledger.maybe_open("BTCUSDT", "spot", _rec(entry_zone=(100.0, 100.0), stop_loss=95.0, atr=2.0), timeframe="1h")
    pos = ledger.open_positions[("BTCUSDT", "spot", "1h")]

    # risk = 5, so 1R profit = price 105. Below that, trailing must stay off.
    ledger.check_and_close({"BTCUSDT": 103.0})
    assert pos.trailing_stop is None

    # At/above 105 (1R), trailing activates: stop = price - 2*ATR = 110 - 4 = 106
    ledger.check_and_close({"BTCUSDT": 110.0})
    assert pos.trailing_stop == pytest.approx(106.0)


def test_trailing_stop_only_tightens_never_loosens(ledger, monkeypatch):
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ACTIVATION_R", 1.0)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ATR_MULTIPLE", 2.0)
    ledger.maybe_open("BTCUSDT", "spot", _rec(entry_zone=(100.0, 100.0), stop_loss=95.0, atr=2.0), timeframe="1h")
    pos = ledger.open_positions[("BTCUSDT", "spot", "1h")]

    ledger.check_and_close({"BTCUSDT": 120.0})  # trail -> 120-4=116
    first_trail = pos.trailing_stop
    assert first_trail == pytest.approx(116.0)

    ledger.check_and_close({"BTCUSDT": 110.0})  # pulls back, but still above the trail
    assert pos.trailing_stop == first_trail, "a pullback must never loosen an already-set trailing stop"


def test_short_position_trails_downward(ledger, monkeypatch):
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ACTIVATION_R", 1.0)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ATR_MULTIPLE", 2.0)
    ledger.maybe_open(
        "BTCUSDT", "derivative",
        _rec(direction="short", entry_zone=(100.0, 100.0), stop_loss=105.0, target=10.0, atr=2.0),
        timeframe="1h",
    )
    pos = ledger.open_positions[("BTCUSDT", "derivative", "1h")]

    # risk=5, 1R profit = price 95. At 80: trail = 80 + 2*2 = 84
    ledger.check_and_close({"BTCUSDT": 80.0})
    assert pos.trailing_stop == pytest.approx(84.0)

    ledger.check_and_close({"BTCUSDT": 90.0})  # pulls back up, must not loosen (raise) the trail
    assert pos.trailing_stop == pytest.approx(84.0)


def test_trailing_exit_is_recorded_as_win_when_above_breakeven(ledger, monkeypatch):
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ACTIVATION_R", 1.0)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ATR_MULTIPLE", 2.0)
    ledger.maybe_open("BTCUSDT", "spot", _rec(entry_zone=(100.0, 100.0), stop_loss=95.0, atr=2.0), timeframe="1h")

    ledger.check_and_close({"BTCUSDT": 120.0})   # trail activates at 116 (well above entry=100)
    ledger.check_and_close({"BTCUSDT": 116.0})   # price falls back and hits the trail

    assert len(ledger.closed_trades) == 1
    trade = ledger.closed_trades[0]
    assert trade["outcome"] == "win"
    assert trade["pnl_dollars"] > 0


def test_positions_without_atr_never_trail_even_if_enabled(ledger, monkeypatch):
    """Guards backward compatibility: positions persisted before this
    field existed (atr_at_entry=0.0 by dataclass default) must not
    activate trailing just because the setting is now on."""
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_STOP_ENABLED", True)
    rec_without_atr = _rec()
    del rec_without_atr["atr"]
    ledger.maybe_open("BTCUSDT", "spot", rec_without_atr, timeframe="1h")
    pos = ledger.open_positions[("BTCUSDT", "spot", "1h")]
    assert pos.atr_at_entry == 0.0

    ledger.check_and_close({"BTCUSDT": 150.0})
    assert pos.trailing_stop is None


def test_ohlc_backtest_mode_trailing_uses_favorable_extreme(ledger, monkeypatch):
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_STOP_ENABLED", True)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ACTIVATION_R", 1.0)
    monkeypatch.setattr(settings, "PAPER_TRADING_TRAILING_ATR_MULTIPLE", 2.0)
    ledger.maybe_open("BTCUSDT", "spot", _rec(entry_zone=(100.0, 100.0), stop_loss=95.0, atr=2.0), timeframe="1h")
    pos = ledger.open_positions[("BTCUSDT", "spot", "1h")]

    # Candle ranges from 108 to 130 — trail should use the HIGH (130) for a long.
    ledger.check_and_close_ohlc("BTCUSDT", low=108.0, high=130.0, closed_at="2026-01-01T00:00:00", timeframe="1h")
    assert pos.trailing_stop == pytest.approx(126.0)  # 130 - 2*2
