"""
Causal (no-lookahead) technical-indicator series, shared by every AB/matrix
experiment script in this repo that needs more than the single-latest-value
helpers in train_gnn.py (_rsi, _ema, _macd, _bollinger only return the LAST
value; these return a full per-index series so a backtest replay can look
up "what would this indicator have read at candle t" without recomputing
from scratch each step).

Every function here takes full OHLCV series and returns a same-length list
where index i depends ONLY on data[0..i] — verified by construction (each
uses a trailing window or a strictly recursive/cumulative formula), so
replaying these index-by-index in a backtest cannot leak future information.
"""
from __future__ import annotations

import math

from train_gnn import _ema  # already causal: emas[i] depends only on data[0..i]


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    n = len(closes)
    rsi = [50.0] * n
    if n < period + 1:
        return rsi
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    for i in range(period, n):
        avg_gain = sum(gains[i - period:i]) / period
        avg_loss = sum(losses[i - period:i]) / period
        rsi[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def volume_avg_series(volumes: list[float], period: int = 20) -> list[float]:
    n = len(volumes)
    avg = [0.0] * n
    for i in range(n):
        window = volumes[max(0, i - period + 1):i + 1]
        avg[i] = sum(window) / len(window) if window else 0.0
    return avg


def macd_hist_series(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> list[float]:
    ema_fast, ema_slow = _ema(fast, closes), _ema(slow, closes)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(signal, macd_line)
    return [m - s for m, s in zip(macd_line, signal_line)]


def bollinger_bands_series(closes: list[float], period: int = 20, n_std: float = 2.0) -> tuple[list[float], list[float]]:
    """Returns (upper, lower) bands, one pair per index."""
    n = len(closes)
    upper, lower = [0.0] * n, [0.0] * n
    for i in range(n):
        window = closes[max(0, i - period + 1):i + 1]
        mean = sum(window) / len(window)
        std = math.sqrt(sum((c - mean) ** 2 for c in window) / len(window))
        upper[i] = mean + n_std * std
        lower[i] = mean - n_std * std
    return upper, lower


def stochastic_k_series(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    n = len(closes)
    k = [50.0] * n
    for i in range(n):
        hh = max(highs[max(0, i - period + 1):i + 1])
        ll = min(lows[max(0, i - period + 1):i + 1])
        k[i] = 50.0 if hh == ll else 100 * (closes[i] - ll) / (hh - ll)
    return k


def cci_series(highs: list[float], lows: list[float], closes: list[float], period: int = 20) -> list[float]:
    n = len(closes)
    typical = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]
    cci = [0.0] * n
    for i in range(n):
        window = typical[max(0, i - period + 1):i + 1]
        sma = sum(window) / len(window)
        mean_dev = sum(abs(t - sma) for t in window) / len(window)
        cci[i] = 0.0 if mean_dev == 0 else (typical[i] - sma) / (0.015 * mean_dev)
    return cci


def obv_series(closes: list[float], volumes: list[float]) -> list[float]:
    n = len(closes)
    obv = [0.0] * n
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def donchian_mid_series(highs: list[float], lows: list[float], period: int = 20) -> list[float]:
    n = len(highs)
    mid = [0.0] * n
    for i in range(n):
        hh = max(highs[max(0, i - period + 1):i + 1])
        ll = min(lows[max(0, i - period + 1):i + 1])
        mid[i] = (hh + ll) / 2.0
    return mid


def _wilder_smooth(values: list[float], period: int) -> list[float]:
    """Wilder's smoothing: first smoothed value is a simple average of the
    first `period` values, every value after is recursive (causal)."""
    n = len(values)
    out = [0.0] * n
    if n < period:
        return out
    out[period - 1] = sum(values[:period])
    for i in range(period, n):
        out[i] = out[i - 1] - out[i - 1] / period + values[i]
    return out


def atr_series(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    """Wilder ATR — causal, same smoothing as adx_series's internal TR handling."""
    n = len(closes)
    tr = [0.0] * n
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    smoothed = _wilder_smooth(tr, period)
    atr = [0.0] * n
    if n > period:
        atr[period] = smoothed[period] / period
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def adx_series(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float]:
    """Standard Wilder ADX. Trend-STRENGTH only (no direction) — used as a
    direction-agnostic "is this market trending enough to trade" gate."""
    n = len(closes)
    if n < period * 2:
        return [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)

    dx = [0.0] * n
    for i in range(period, n):
        if smoothed_tr[i] == 0:
            continue
        plus_di = 100 * smoothed_plus[i] / smoothed_tr[i]
        minus_di = 100 * smoothed_minus[i] / smoothed_tr[i]
        denom = plus_di + minus_di
        dx[i] = 0.0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom

    adx = [0.0] * n
    start = period * 2 - 1
    if start < n:
        adx[start] = sum(dx[period:start + 1]) / period
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx
