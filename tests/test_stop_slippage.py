"""
Tests for exit-type-aware slippage in backtrack/paper_trading.py.

Motivation: a stop becomes a MARKET order when breached and fills past its
trigger (measured 0.25-0.60% on real live SHORT stop-outs), while a target
is a LIMIT order that fills at its price or better. The pre-existing
`slippage_pct` is symmetric across every fill, so it cannot express that
difference — and any rule that converts target exits into stop exits
(breakeven, trailing) is therefore scored too generously by it.

The critical guarantee here is the DEFAULT: stop_slippage_pct=0.0 must
leave live and every existing backtest byte-identical to before.
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


def _ledger(tmp_path, **kwargs):
    return PaperTradingLedger(path=str(tmp_path / "sl.json"), starting_balance=10_000.0, **kwargs)


def test_default_is_zero_so_live_behaviour_is_unchanged(tmp_path):
    """The whole change must be inert unless a caller opts in — live
    (main.py) constructs the ledger with no arguments."""
    led = _ledger(tmp_path)
    assert led.stop_slippage_pct == 0.0

    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=94.0, high=101.0, closed_at="t", timeframe="1h")
    # Exit recorded exactly at the stop, no adverse adjustment.
    assert led.closed_trades[0]["exit_price"] == pytest.approx(95.0)


def test_stop_exit_fills_past_the_trigger(tmp_path):
    """A long stopped out sells BELOW its stop; the gap is the slippage."""
    led = _ledger(tmp_path, stop_slippage_pct=0.004)  # 0.4%
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=94.0, high=101.0, closed_at="t", timeframe="1h")

    trade = led.closed_trades[0]
    assert trade["exit_type"] == "stop"
    assert trade["exit_price"] == pytest.approx(95.0 * (1 - 0.004))


def test_short_stop_slips_the_other_way(tmp_path):
    """A short stopped out buys back ABOVE its stop — adverse in the
    opposite direction, which is the case actually measured live."""
    led = _ledger(tmp_path, stop_slippage_pct=0.004)
    led.maybe_open("BTCUSDT", "derivative",
                   _rec(direction="short", stop_loss=105.0, target=90.0), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=99.0, high=106.0, closed_at="t", timeframe="1h")

    trade = led.closed_trades[0]
    assert trade["exit_type"] == "stop"
    assert trade["exit_price"] == pytest.approx(105.0 * (1 + 0.004))


def test_target_exit_does_not_slip(tmp_path):
    """The point of the whole change: a limit order fills at its price.
    If this ever regresses to also slipping, the exit-mix comparison it
    exists to protect becomes meaningless again."""
    led = _ledger(tmp_path, stop_slippage_pct=0.004)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=99.0, high=111.0, closed_at="t", timeframe="1h")

    trade = led.closed_trades[0]
    assert trade["exit_type"] == "target"
    assert trade["exit_price"] == pytest.approx(110.0)


def test_stop_slippage_is_additive_to_symmetric_slippage(tmp_path):
    """The two knobs are independent: the symmetric one still applies to
    every fill, the stop one adds further cost on stop exits only."""
    led = _ledger(tmp_path, slippage_pct=0.001, stop_slippage_pct=0.004)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=94.0, high=101.0, closed_at="t", timeframe="1h")

    assert led.closed_trades[0]["exit_price"] == pytest.approx(95.0 * (1 - 0.005))


def test_live_path_also_applies_stop_slippage(tmp_path):
    """check_and_close (live/tick path) must not be left behind — it shares
    the same exit semantics as the OHLC path."""
    led = _ledger(tmp_path, stop_slippage_pct=0.004)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close({"BTCUSDT": 94.0})

    trade = led.closed_trades[0]
    assert trade["exit_type"] == "stop"
    assert trade["exit_price"] == pytest.approx(95.0 * (1 - 0.004))


def test_outcome_label_reflects_the_boundary_not_the_slipped_price(tmp_path):
    """win/loss should stay a statement about signal correctness. A target
    exit remains a "win" even though slippage elsewhere eats into PnL —
    otherwise the win rate silently becomes a cost metric."""
    led = _ledger(tmp_path, stop_slippage_pct=0.05)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=99.0, high=111.0, closed_at="t", timeframe="1h")
    assert led.closed_trades[0]["outcome"] == "win"
