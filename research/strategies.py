"""Candidate strategies, exactly as pre-registered in research/PREREGISTRATION.md.

Each strategy is a pair: a SIGNAL function (dates x symbols, values in [0, 1],
using only data up to that day's close) and a WEIGHT function that maps signals
to target portfolio weights. The engine trades the target at the NEXT day's
open. Keeping the two apart lets the random-timing null shift the signal and
re-apply the very same sizing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)
VOL_TARGET = 0.25
MIN_HISTORY = 365


# ---------------------------------------------------------------- helpers

def eligible(close: pd.DataFrame) -> pd.DataFrame:
    """True once a coin has at least MIN_HISTORY daily bars."""
    return (close.notna().cumsum() >= MIN_HISTORY) & close.notna()


def sigma90(close: pd.DataFrame) -> pd.DataFrame:
    r = close.pct_change(fill_method=None)
    return r.rolling(90).std() * np.sqrt(365)


def weekly(s: pd.DataFrame) -> pd.DataFrame:
    """Keep Monday decisions and carry them through the week."""
    monday = s.index.dayofweek == 0
    return s.where(pd.Series(monday, index=s.index), np.nan).ffill().fillna(0.0)


def _rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------- signals

def sig_sma200(close: pd.DataFrame) -> pd.DataFrame:
    return (close > close.rolling(200).mean()).astype(float)


def sig_tsmom4w(close: pd.DataFrame) -> pd.DataFrame:
    return weekly((close / close.shift(28) - 1 > 0).astype(float))


def donchian_ensemble(close: pd.Series) -> pd.Series:
    """Mean state of the 9 Donchian sub-models (Zarattini, Pagani & Barbon 2025).

    For each lookback L, hi/lo are the max/min of the previous L closes (today
    excluded), so each day's levels are known before that day's close. A
    sub-model turns on when the close breaks above hi; its stop starts at the
    channel midpoint and only ratchets up; from the day after entry it turns
    off when the close falls below the stop.
    """
    c = close.to_numpy(dtype=float)
    states = np.zeros((len(LOOKBACKS), len(c)))
    for j, L in enumerate(LOOKBACKS):
        hi = close.shift(1).rolling(L).max().to_numpy()
        lo = close.shift(1).rolling(L).min().to_numpy()
        on, stop = False, -np.inf
        for t in range(len(c)):
            if np.isnan(c[t]) or np.isnan(hi[t]) or np.isnan(lo[t]):
                on, stop = False, -np.inf
                continue
            mid = (hi[t] + lo[t]) / 2
            if not on:
                if c[t] > hi[t]:
                    on, stop = True, mid
            else:
                stop = max(stop, mid)
                if c[t] < stop:
                    on, stop = False, -np.inf
            states[j, t] = 1.0 if on else 0.0
    return pd.Series(states.mean(axis=0), index=close.index)


def sig_donchian(close: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({s: donchian_ensemble(close[s]) for s in close.columns})


def sig_xsmom3(close: pd.DataFrame) -> pd.DataFrame:
    ret = (close / close.shift(14) - 1).where(eligible(close))
    out = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for d in close.index[close.index.dayofweek == 0]:
        row = ret.loc[d].dropna()
        top = row[row > 0].sort_values(ascending=False).head(3).index
        out.loc[d, top] = 1.0
    monday = pd.Series(close.index.dayofweek == 0, index=close.index)
    return out.where(monday, np.nan).ffill().fillna(0.0)


def sig_rsi2(close: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for s in close.columns:
        c = close[s]
        rsi = _rsi_wilder(c, 2).to_numpy()
        sma200 = c.rolling(200).mean().to_numpy()
        sma5 = c.rolling(5).mean().to_numpy()
        cv = c.to_numpy()
        st = np.zeros(len(c))
        on = False
        for t in range(len(c)):
            if np.isnan(cv[t]) or np.isnan(sma200[t]) or np.isnan(rsi[t]):
                on = False
            elif not on and rsi[t] < 10 and cv[t] > sma200[t]:
                on = True
            elif on and cv[t] > sma5[t]:
                on = False
            st[t] = 1.0 if on else 0.0
        out[s] = st
    return pd.DataFrame(out, index=close.index)


# ---------------------------------------------------------------- weights

def _cap(w: pd.DataFrame) -> pd.DataFrame:
    gross = w.sum(axis=1)
    scale = (1.0 / gross).where(gross > 1.0, 1.0)
    return w.mul(scale, axis=0)


def w_equal(sig: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    el = eligible(close)
    n = el.sum(axis=1).replace(0, np.nan)
    return _cap(sig.where(el, 0.0).div(n, axis=0).fillna(0.0))


def w_voltarget(sig: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    el = eligible(close)
    n = el.sum(axis=1).replace(0, np.nan)
    scale = (VOL_TARGET / sigma90(close)).clip(upper=1.0)
    return _cap((sig * scale).where(el, 0.0).div(n, axis=0).fillna(0.0))


def w_btc_voltarget(sig: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    el = eligible(close)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    scale = (VOL_TARGET / sigma90(close[["BTCUSDT"]])).clip(upper=1.0)["BTCUSDT"]
    w["BTCUSDT"] = (sig["BTCUSDT"] * scale).where(el["BTCUSDT"], 0.0).fillna(0.0)
    return w


def w_top3(sig: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    return _cap((sig.where(eligible(close), 0.0) / 3.0).fillna(0.0))


# ---------------------------------------------------------------- registry

CANDIDATES = {
    "C1 SMA200": (sig_sma200, w_equal),
    "C2 TSMOM4W": (sig_tsmom4w, w_equal),
    "C3 DONCH-BTC": (sig_donchian, w_btc_voltarget),
    "C4 DONCH-VT": (sig_donchian, w_voltarget),
    "C5 DONCH-EW": (sig_donchian, w_equal),
    "C6 XSMOM3": (sig_xsmom3, w_top3),
    "C7 RSI2": (sig_rsi2, w_equal),
}
# Parameters per candidate, for the deflated-Sharpe tie-break ("fewer parameters").
N_PARAMS = {"C1 SMA200": 1, "C2 TSMOM4W": 1, "C3 DONCH-BTC": 3, "C4 DONCH-VT": 3,
            "C5 DONCH-EW": 2, "C6 XSMOM3": 2, "C7 RSI2": 4}


def bh_btc(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    el = eligible(close)
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    w["BTCUSDT"] = el["BTCUSDT"].astype(float)
    return w.copy(), w


def bh_equal(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    el = eligible(close).astype(float)
    return el, w_equal(el, close)
