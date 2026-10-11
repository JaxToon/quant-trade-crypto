"""Daily portfolio backtester with honest execution.

Timeline: a target decided on the CLOSE of day i-1 is traded at the OPEN of
day i, then held open-to-open until day i+1. Costs are charged on every unit of
notional traded, each side. Weights drift with prices between trades.

Trading rule (PREREGISTRATION.md): an asset is traded when its target turns on
or off, when its signal changes value, or on a rebalance day when the held
weight is more than `band` away from the target (relative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Result:
    returns: pd.Series                 # net daily return, indexed by the day the position is held
    exposure: pd.Series                # gross weight held that day
    turnover: pd.Series                # sum |dw| traded at that day's open
    costs: pd.Series                   # cost charged that day, fraction of equity
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(open_: pd.DataFrame, target: pd.DataFrame, signal: pd.DataFrame, cost: float,
        rebalance_days: pd.Series, band: float = 0.2, track_trades: bool = True) -> Result:
    dates = open_.index
    syms = list(open_.columns)
    O = open_.to_numpy(dtype=float)
    T = target.reindex(index=dates, columns=syms).fillna(0.0).to_numpy()
    S = signal.reindex(index=dates, columns=syms).fillna(0.0).to_numpy()
    reb = rebalance_days.reindex(dates).fillna(False).to_numpy(dtype=bool)
    n, m = O.shape

    R = np.zeros_like(O)
    with np.errstate(invalid="ignore", divide="ignore"):
        R[:-1] = O[1:] / O[:-1] - 1.0
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    w = np.zeros(m)
    last_sig = np.zeros(m)
    equity = 1.0
    out_r = np.zeros(n)
    out_x = np.zeros(n)
    out_t = np.zeros(n)
    out_c = np.zeros(n)

    open_ep: dict[int, dict] = {}
    episodes: list[dict] = []

    for i in range(1, n - 1):
        t, s = T[i - 1], S[i - 1]
        h = w
        exit_ = (t <= 0) & (h > 0)
        enter = (t > 0) & (h <= 0)
        sig_chg = (t > 0) & (h > 0) & (s != last_sig)
        drift = (t > 0) & (h > 0) & reb[i - 1] & (np.abs(t - h) > band * t)
        trade = exit_ | enter | sig_chg | drift
        new = np.where(trade, np.where(t > 0, t, 0.0), h)
        if new.sum() > 1.0 + 1e-9:
            # Partial trading would breach the no-leverage rule (e.g. a newly
            # eligible coin entering before the others are trimmed): rebalance
            # everything to target instead, which sums to at most 1.
            trade = (t > 0) | (h > 0)
            new = np.where(t > 0, t, 0.0)
            exit_ = (t <= 0) & (h > 0)
        last_sig = np.where(trade, s, last_sig)

        dw = new - h
        c = cost * np.abs(dw).sum()
        eq_before_trade = equity
        equity *= (1.0 - c)

        for k in (np.nonzero(enter)[0] if track_trades else ()):
            open_ep[k] = {"symbol": syms[k], "entry": dates[i], "pnl": 0.0,
                          "eq_entry": eq_before_trade, "cost": 0.0}
        for k in (np.nonzero(dw)[0] if track_trades else ()):
            if k in open_ep:
                open_ep[k]["cost"] += cost * abs(dw[k]) * eq_before_trade
                open_ep[k]["pnl"] -= cost * abs(dw[k]) * eq_before_trade
        for k in (np.nonzero(exit_)[0] if track_trades else ()):
            ep = open_ep.pop(k, None)
            if ep:
                ep["exit"] = dates[i]
                episodes.append(ep)

        w = new
        g = float((w * R[i]).sum())
        for k in (np.nonzero(w > 0)[0] if track_trades else ()):
            if k in open_ep:
                open_ep[k]["pnl"] += equity * w[k] * R[i, k]
        out_x[i] = w.sum()
        out_t[i] = np.abs(dw).sum()
        out_c[i] = c
        out_r[i] = (1.0 - c) * (1.0 + g) - 1.0
        equity *= (1.0 + g)
        if 1.0 + g != 0:
            w = w * (1.0 + R[i]) / (1.0 + g)

    for k, ep in open_ep.items():
        ep["exit"] = None  # still open at the end of the data
        episodes.append(ep)

    idx = dates
    tr = pd.DataFrame(episodes)
    if not tr.empty:
        tr["ret_on_equity"] = tr["pnl"] / tr["eq_entry"]
        tr["days"] = [(((e or dates[-1]) - s).days) for s, e in zip(tr["entry"], tr["exit"])]
    return Result(pd.Series(out_r, idx), pd.Series(out_x, idx), pd.Series(out_t, idx),
                  pd.Series(out_c, idx), tr)


def mondays(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.dayofweek == 0, index=index)


def month_starts(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.day == 1, index=index)
