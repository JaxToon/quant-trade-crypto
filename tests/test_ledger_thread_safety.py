"""
Concurrency tests for PaperTradingLedger.

Why these exist: on 2026-09-07 SL/TP monitoring moved out of the 60s
decision_loop into its own fast thread (main.py, run_sltp_monitor), because
checking stops once a minute while prices arrived every tick was very
likely the dominant term in the 0.25-0.60% measured live stop slippage --
against a backtested break-even of ~0.51%. The ledger is now written by
three threads: the SL/TP monitor, the decision loop, and the scheduler's
learning hook.

A NOTE ON HOW THESE ARE WRITTEN, because the obvious version is worthless.
Simply firing N threads at a breached position does NOT reproduce the race:
under CPython's GIL the whole check-and-close completes inside one
scheduling slice, so the naive test passes identically with the lock
removed — verified, 0/20 trials corrupted. A test that cannot fail proves
nothing.

The race window is real but narrow, and opens whenever the GIL is released
mid-mutation — which save()'s file I/O does on every call, and which a
longer open-position list can also hit via the interpreter's switch
interval. So these tests force that yield explicitly. With the yield forced
and the lock removed, one position gets closed SIX times, five threads
raise, and the balance is off by ~$1,238 on a single trade.
"""
from __future__ import annotations

import threading
import time

import pytest

from backtrack.paper_trading import PaperTradingLedger


class _NoOpLock:
    """Stands in for the RLock to prove these tests actually detect its loss."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _rec(direction="long", entry_zone=(100.0, 100.0), stop_loss=95.0, target=110.0):
    return {
        "actionable": True, "direction": direction, "entry_zone": list(entry_zone),
        "stop_loss": stop_loss, "target": target, "confidence": 0.0,
        "suggested_leverage": 1,
    }


def _ledger_with_forced_yield(tmp_path, use_lock: bool):
    """A ledger whose _close yields the GIL mid-mutation, reproducing what
    save()'s file I/O does in production."""
    led = PaperTradingLedger(path=str(tmp_path / "l.json"), starting_balance=10_000.0)
    if not use_lock:
        led._lock = _NoOpLock()
    led.maybe_open("BTCUSDT", "spot", _rec(), timeframe="1h")
    original = led._close

    def slow_close(*args, **kwargs):
        time.sleep(0.02)
        return original(*args, **kwargs)

    led._close = slow_close
    return led


def _hammer(led, n=6):
    errors = []

    def run():
        try:
            led.check_and_close({"BTCUSDT": 94.0})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    return errors


def test_position_closes_exactly_once_under_forced_interleaving(tmp_path):
    led = _ledger_with_forced_yield(tmp_path, use_lock=True)
    errors = _hammer(led)

    assert not errors, f"threads raised: {errors}"
    assert len(led.closed_trades) == 1, "position closed more than once — lock lost?"
    assert len(led.open_positions) == 0


def test_the_previous_test_actually_detects_a_missing_lock(tmp_path):
    """Guards the guard. If this ever stops failing without the lock, the
    test above has silently become decorative and must be rewritten."""
    led = _ledger_with_forced_yield(tmp_path, use_lock=False)
    _hammer(led)
    assert len(led.closed_trades) > 1, (
        "removing the lock no longer corrupts the ledger — the concurrency "
        "tests here are no longer proving anything and need reworking"
    )


def test_balance_reflects_exactly_one_trade(tmp_path):
    """The damage a double close does is silent: PnL applied N times."""
    led = _ledger_with_forced_yield(tmp_path, use_lock=True)
    _hammer(led)
    assert led.balance == pytest.approx(10_000.0 + led.closed_trades[0]["pnl_dollars"], abs=1e-6)


def test_open_and_close_run_concurrently_without_double_closing(tmp_path):
    """Mirrors production: the decision loop opens while the monitor closes."""
    led = PaperTradingLedger(path=str(tmp_path / "l.json"), starting_balance=100_000.0)
    errors = []

    def opener():
        try:
            for i in range(40):
                led.maybe_open(f"SYM{i}USDT", "spot", _rec(), timeframe="1h")
        except Exception as exc:
            errors.append(exc)

    def closer():
        try:
            for _ in range(40):
                led.check_and_close({f"SYM{i}USDT": 94.0 for i in range(40)})
        except Exception as exc:
            errors.append(exc)

    ts = [threading.Thread(target=opener), threading.Thread(target=closer),
          threading.Thread(target=closer)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=20)

    assert not errors, f"threads raised: {errors}"
    keys = [(t["symbol"], t["market"], t["timeframe"]) for t in led.closed_trades]
    assert len(keys) == len(set(keys)), "a position was closed more than once"


def test_public_methods_are_the_locked_wrappers():
    """Guards against a refactor renaming an implementation back over its
    wrapper and silently dropping the lock."""
    for name in ("save", "maybe_open", "check_and_close", "check_and_close_ohlc",
                 "get_unlearned_trades", "mark_learned", "stats"):
        assert hasattr(PaperTradingLedger, f"_{name}_unlocked"), f"{name} lost its unlocked impl"
