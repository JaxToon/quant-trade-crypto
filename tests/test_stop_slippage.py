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


def test_live_path_stacks_modelled_slippage_on_the_observed_price(tmp_path):
    """check_and_close (live/tick path) must not be left behind — but it
    starts from a different base than the OHLC path, and deliberately so.

    Since the instrumentation change (2026-09-07) the live path books the
    OBSERVED price, which already contains the poll-gap component of real
    slippage. `stop_slippage_pct` then layers on the components paper
    trading can never see — order-book depth, market impact, exchange
    latency. Those are separate costs, so stacking them is additive, not
    double-counting. Base is 94.0 (observed), not 95.0 (trigger)."""
    led = _ledger(tmp_path, stop_slippage_pct=0.004)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close({"BTCUSDT": 94.0})

    trade = led.closed_trades[0]
    assert trade["exit_type"] == "stop"
    assert trade["exit_price"] == pytest.approx(94.0 * (1 - 0.004))


def test_outcome_label_reflects_the_boundary_not_the_slipped_price(tmp_path):
    """win/loss should stay a statement about signal correctness. A target
    exit remains a "win" even though slippage elsewhere eats into PnL —
    otherwise the win rate silently becomes a cost metric."""
    led = _ledger(tmp_path, stop_slippage_pct=0.05)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=99.0, high=111.0, closed_at="t", timeframe="1h")
    assert led.closed_trades[0]["outcome"] == "win"


# ── Live slippage instrumentation ──────────────────────────────────────────
# Added 2026-09-07. Until this, a live stop exit was booked at exactly its
# trigger and the overshoot was discarded, which is why the project has only
# 4 hand-reconstructed slippage data points despite ~90 live trades — and why
# it cannot currently say whether its own edge survives its own costs.


def test_live_stop_books_the_observed_price_not_the_trigger(tmp_path):
    """The core of the instrumentation: a breached stop is a market order,
    so the fill is where price actually was, not back at the trigger."""
    led = _ledger(tmp_path)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close({"BTCUSDT": 94.5})  # stop 95.0, price gapped to 94.5

    t = led.closed_trades[0]
    assert t["exit_price"] == pytest.approx(94.5)
    assert t["trigger_price"] == pytest.approx(95.0)
    assert t["observed_price"] == pytest.approx(94.5)


def test_slippage_is_positive_when_adverse_for_a_long(tmp_path):
    led = _ledger(tmp_path)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close({"BTCUSDT": 94.5})
    # 0.5 below a 95.0 stop = 0.5263% adverse
    assert led.closed_trades[0]["observed_slippage_pct"] == pytest.approx(0.5263, abs=1e-3)


def test_slippage_is_positive_when_adverse_for_a_short(tmp_path):
    """Sign convention must not flip with direction: positive = adverse,
    always. A short stopping out buys back ABOVE its stop."""
    led = _ledger(tmp_path)
    led.maybe_open("BTCUSDT", "derivative",
                   _rec(direction="short", stop_loss=105.0, target=90.0), timeframe="1h")
    led.check_and_close({"BTCUSDT": 105.5})
    t = led.closed_trades[0]
    assert t["exit_price"] == pytest.approx(105.5)
    assert t["observed_slippage_pct"] == pytest.approx(0.4762, abs=1e-3)


def test_target_still_books_at_the_limit_not_the_overshoot(tmp_path):
    """A limit order does not capture favourable overshoot. Booking the
    observed price here would silently invent profit that no real fill
    would have earned."""
    led = _ledger(tmp_path)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close({"BTCUSDT": 112.0})  # target 110.0, price ran past

    t = led.closed_trades[0]
    assert t["exit_price"] == pytest.approx(110.0)
    assert t["observed_price"] == pytest.approx(112.0)
    assert t["outcome"] == "win"


def test_outcome_still_reflects_the_boundary_when_slippage_crosses_entry(tmp_path):
    """A stop that slips far enough to land past entry is still a LOSS by
    signal correctness. If outcome were derived from the slipped fill the
    win rate would quietly become a cost metric."""
    led = _ledger(tmp_path)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close({"BTCUSDT": 80.0})
    assert led.closed_trades[0]["outcome"] == "loss"


def test_ohlc_backtest_records_no_observed_slippage(tmp_path):
    """OHLC has no observed price — only a bar. Leaving these None keeps
    backtest rows out of the live slippage statistics instead of polluting
    them with a number invented from a bar extreme."""
    led = _ledger(tmp_path)
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    led.check_and_close_ohlc("BTCUSDT", low=94.0, high=101.0, closed_at="t", timeframe="1h")

    t = led.closed_trades[0]
    assert t["observed_slippage_pct"] is None
    assert t["exit_price"] == pytest.approx(95.0)
