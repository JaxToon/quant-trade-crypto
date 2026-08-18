"""
yfinance-based near-real-time polling stream for stocks — Stage 1 of the
staged stock data-path validation (data source only, strategy logic in
strategy/smart_money.py, strategy/fvg_analyzer.py, strategy/entry_insight.py
untouched). Alternative to TradingViewStream, whose free-tier API
rate-limits after ~1 poll cycle regardless of pacing (documented,
unresolved external constraint — see ingestion/tradingview_stream.py).

Regular session only (prepost=False) — extended-hours 1m candles were
checked directly and are frequently zero-volume with the price still
ticking slightly (thin/illiquid quote noise), which would feed the
FVG/order-block analyzer spurious "imbalances" backed by no real
liquidity. Confirmed empirically that yfinance's regular-session index
already reflects real market-hours structure with no fabrication needed
on our side: a 5-day pull showed clean ~09:30-16:00 ET daily windows and
a correct ~65.5h Friday-close -> Monday-open gap (no fake weekend bars).

Market-hours awareness here is still explicit (not just relying on
yfinance naturally returning no new rows when closed) for two reasons:
avoids polling at full frequency for no reason while the market is shut,
and makes "why is nothing happening" legible in logs (CLOSED vs a dead
stream) instead of looking indistinguishable from a hung connection.

yfinance has no built-in indicators (unlike TradingView's analysis
object) — RSI/MACD/Bollinger are computed here from the polled window's
own closes, same formulas train_gnn.py and structured_stream.py already
use independently (duplicated locally rather than importing from
train_gnn.py, which pulls in torch/GNN as a side effect of two 10-line
functions).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time as dtime
from typing import Callable, Literal
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import settings
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

MarketType = Literal["crypto", "stock"]

EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# Seconds per interval — mirrors tradingview_stream.py's INTERVAL_SECONDS.
INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}

# How much history to pull each poll so RSI(14)/MACD(12,26,9)/BB(20) have
# enough lookback to be stable, not just "period exists" — same MIN_LOOKBACK
# reasoning train_gnn.py/structured_stream.py already apply. Comfortably
# inside yfinance's own per-interval limits for a short recent window.
FETCH_PERIOD = {"1m": "5d", "5m": "10d", "15m": "20d", "1h": "60d"}


def is_regular_session(now_et: datetime) -> bool:
    """Doesn't account for exchange holidays or early-close half-days —
    yfinance's own returned data (empty / stale index) is the real source
    of truth for whether a bar actually exists; this only gates polling
    cadence and log clarity, never whether data gets accepted."""
    if now_et.weekday() >= 5:  # Saturday/Sunday
        return False
    return MARKET_OPEN <= now_et.time() < MARKET_CLOSE


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    window = closes[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(window)):
        delta = window[i] - window[i - 1]
        (gains if delta > 0 else losses).append(abs(delta))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(period: int, data: list[float]) -> list[float]:
    k = 2 / (period + 1)
    emas = [data[0]]
    for p in data[1:]:
        emas.append(p * k + emas[-1] * (1 - k))
    return emas


def _macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    fast_ema = _ema(fast, closes)
    slow_ema = _ema(slow, closes)
    macd_line = [f - s for f, s in zip(fast_ema[-len(slow_ema):], slow_ema)]
    signal_line = _ema(signal, macd_line)
    macd_val, signal_val = macd_line[-1], signal_line[-1]
    return macd_val, signal_val, macd_val - signal_val


def _bollinger(closes: list[float], period: int = 20, n_std: float = 2.0) -> tuple[float, float, float]:
    if len(closes) < period:
        return 0.0, 0.0, 0.0
    window = closes[-period:]
    mean = sum(window) / period
    std = (sum((c - mean) ** 2 for c in window) / period) ** 0.5
    return mean + n_std * std, mean, mean - n_std * std


def build_tick_data(df: pd.DataFrame, interval_str: str, kline_closed: bool) -> dict:
    """
    Turns a yfinance OHLCV DataFrame (regular session, single symbol) into
    the same flat tick dict shape tradingview_stream.py's _build_tick_data
    produces — RSI/MACD/BB/etc keys downstream code (gnn/dataset.py,
    strategy/entry_insight.py) already expects. Extracted from the polling
    loop so it's unit-testable without a live network call or asyncio.
    """
    closes = df["Close"].tolist()
    last = df.iloc[-1]
    macd, macd_signal, macd_hist = _macd(closes)
    bb_upper, bb_mid, bb_lower = _bollinger(closes)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "interval": interval_str,
        "kline_closed": kline_closed,
        "price": float(last["Close"]),
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "volume": float(last["Volume"]),
        "change": float((last["Close"] - df.iloc[0]["Open"]) / df.iloc[0]["Open"]) if df.iloc[0]["Open"] else 0.0,
        "rsi": _rsi(closes),
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "ema_20": _ema(20, closes)[-1] if len(closes) >= 20 else float(last["Close"]),
        "ema_50": _ema(50, closes)[-1] if len(closes) >= 50 else float(last["Close"]),
        "sma_200": sum(closes[-200:]) / min(len(closes), 200),
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
    }


class YFinanceStream:
    """
    Polls yfinance for the latest CLOSED regular-session candle per
    symbol. "Closed" means the second-to-last row of the fetched window —
    yfinance's last row is frequently the still-forming current bar, same
    reasoning tradingview_stream.py's _is_new_bar time-bucket tracking
    exists for (a stream must never treat a forming candle as final).
    """

    def __init__(self, interval: str = "5m") -> None:
        self.interval = interval
        self.interval_str = interval
        self._running = False
        self._callbacks: list[Callable[[str, dict], None]] = []
        self._last_bar_time: dict[str, pd.Timestamp] = {}

    def add_callback(self, fn: Callable[[str, dict], None]) -> None:
        self._callbacks.append(fn)

    def stop(self) -> None:
        self._running = False

    async def start(self) -> None:
        self._running = True
        period = FETCH_PERIOD.get(self.interval, "10d")
        logger.info("yfinance stream started: %s @ %s (period=%s)", settings.WATCHLIST, self.interval, period)

        while self._running:
            now_et = datetime.now(EASTERN)
            if not is_regular_session(now_et):
                logger.debug("Market closed (%s ET) — skipping poll cycle", now_et.strftime("%a %H:%M"))
                await asyncio.sleep(60)
                continue

            for symbol in settings.WATCHLIST:
                await asyncio.sleep(0.5)
                try:
                    self._poll_symbol(symbol, period)
                except Exception as exc:
                    logger.error("yfinance poll error for %s: %s", symbol, exc)

            await asyncio.sleep(INTERVAL_SECONDS.get(self.interval, 300))

    def _poll_symbol(self, symbol: str, period: str) -> None:
        df = yf.download(symbol, period=period, interval=self.interval, progress=False, auto_adjust=True, prepost=False)
        if df.empty or len(df) < 2:
            return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        closed_row = df.iloc[:-1]  # everything up to (not including) the still-forming last bar
        closed_ts = closed_row.index[-1]

        is_new = symbol not in self._last_bar_time or closed_ts > self._last_bar_time[symbol]
        if not is_new:
            return
        self._last_bar_time[symbol] = closed_ts

        data = build_tick_data(closed_row, self.interval_str, kline_closed=True)
        try:
            persisted = {k: v for k, v in data.items() if k not in ("kline_closed", "interval")}
            Neo4jClient.get().update_token_market_data(symbol, **persisted)
        except Exception as exc:
            logger.warning("Neo4j write failed for %s: %s", symbol, exc)

        for cb in self._callbacks:
            try:
                cb(symbol, data)
            except Exception as exc:
                logger.warning("Callback error for %s: %s", symbol, exc)
