"""The research engine must be honest before any result from it is believed:
no look-ahead, costs always charged, never leveraged."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import engine, metrics, strategies


def _frame(values, start="2024-01-01", name="AAAUSDT"):
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.DataFrame({name: values}, index=idx)


def _never_rebalance(index):
    return pd.Series(False, index=index)


def test_buy_and_hold_matches_price_without_costs():
    o = _frame([100.0, 110.0, 121.0, 133.1, 146.41])
    tgt = pd.DataFrame(1.0, index=o.index, columns=o.columns)
    res = engine.run(o, tgt, tgt, cost=0.0, rebalance_days=_never_rebalance(o.index))
    # target set at close of day 0 -> bought at open of day 1 (110) and held
    # open-to-open through day 3 (last return needs day 4's open).
    total = (1 + res.returns).prod() - 1
    assert total == pytest.approx(146.41 / 110.0 - 1)


def test_signal_cannot_capture_the_move_that_triggered_it():
    # Price jumps +50% from day 2's open to day 3's open. A signal that only
    # switches on at day 2's CLOSE trades at day 3's open, so it must miss it.
    o = _frame([100.0, 100.0, 100.0, 150.0, 150.0, 150.0])
    sig = pd.DataFrame([0, 0, 1, 1, 1, 1], index=o.index, columns=o.columns, dtype=float)
    res = engine.run(o, sig, sig, cost=0.0, rebalance_days=_never_rebalance(o.index))
    assert (1 + res.returns).prod() - 1 == pytest.approx(0.0)


def test_round_trip_pays_cost_on_both_sides():
    o = _frame([100.0] * 6)
    sig = pd.DataFrame([1, 0, 0, 0, 0, 0], index=o.index, columns=o.columns, dtype=float)
    res = engine.run(o, sig, sig, cost=0.004, rebalance_days=_never_rebalance(o.index))
    assert (1 + res.returns).prod() == pytest.approx((1 - 0.004) * (1 - 0.004))
    # costs are recorded as a fraction of equity at each trade
    assert res.costs.sum() == pytest.approx(0.008)


def test_never_leveraged_when_a_coin_becomes_eligible():
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    o = pd.DataFrame({"A": [100.0] * 6, "B": [50.0] * 6}, index=idx)
    # A alone at 100% first; B appears and both should be 50%.
    tgt = pd.DataFrame({"A": [1, 1, 0.5, 0.5, 0.5, 0.5], "B": [0, 0, 0.5, 0.5, 0.5, 0.5]},
                       index=idx, dtype=float)
    sig = (tgt > 0).astype(float)
    res = engine.run(o, tgt, sig, cost=0.0, rebalance_days=_never_rebalance(idx))
    assert res.exposure.max() <= 1.0 + 1e-9


def test_trade_ledger_records_wins_and_losses():
    # 1st entry: bought at day 1's open (100), sold at day 2's open (120).
    # 2nd entry: bought at day 4's open (120), sold at day 5's open (90).
    o = _frame([100.0, 100.0, 120.0, 120.0, 120.0, 90.0, 90.0])
    sig = pd.DataFrame([1, 0, 0, 1, 0, 0, 0], index=o.index, columns=o.columns, dtype=float)
    res = engine.run(o, sig, sig, cost=0.0, rebalance_days=_never_rebalance(o.index))
    t = res.trades
    assert len(t) == 2
    assert t.iloc[0]["pnl"] > 0 and t.iloc[1]["pnl"] < 0


def test_donchian_turns_on_in_an_uptrend_and_off_after_a_collapse():
    up = list(np.linspace(100, 200, 400))
    down = list(np.linspace(200, 60, 60))
    s = strategies.donchian_ensemble(pd.Series(up + down,
                                               index=pd.date_range("2020-01-01", periods=460, freq="D")))
    assert s.iloc[399] == pytest.approx(1.0)   # every lookback has broken out by the top
    assert s.iloc[-1] == pytest.approx(0.0)    # every stop hit on the way down


def test_weekly_signal_only_changes_on_mondays():
    idx = pd.date_range("2024-01-01", periods=28, freq="D", tz="UTC")  # 2024-01-01 is a Monday
    raw = pd.DataFrame({"A": np.random.default_rng(1).integers(0, 2, 28).astype(float)}, index=idx)
    w = strategies.weekly(raw)
    changes = w["A"].diff().fillna(0) != 0
    assert all(idx[changes].dayofweek == 0)


def test_stationary_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.001, 0.02, 1500))
    lo, hi = metrics.stationary_bootstrap_sharpe(r, n_boot=400)
    point = r.mean() / r.std(ddof=1) * np.sqrt(365)
    assert lo < point < hi
