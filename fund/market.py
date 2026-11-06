"""Daily market data for the fund, from data-api.binance.vision.

Same host and same bar construction as research/data.py, but cached in its
own folder and refreshed each run: the research cache is pinned to the
pre-registered study window and must not move.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from research import data as rdata
from fund import config


class StaleData(RuntimeError):
    """The newest complete bar is not yesterday's: trading on it would be blind."""


def today_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")


def _load_symbol(symbol: str, end: pd.Timestamp) -> pd.DataFrame:
    """Complete daily bars strictly before `end`, cached and topped up."""
    os.makedirs(config.KLINE_DIR, exist_ok=True)
    path = os.path.join(config.KLINE_DIR, f"{symbol}_1d.parquet")
    end_iso = end.strftime("%Y-%m-%d")
    if os.path.exists(path):
        old = pd.read_parquet(path)
        if len(old) and old.index[-1] >= end - timedelta(days=1):
            return old[old.index < end]
    df = rdata.fetch(symbol, "1d", end=end_iso)
    df.to_parquet(path)
    return df


def history(symbols: list[str] = config.UNIVERSE, today: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """Wide open/close frames of complete bars up to and including yesterday.

    Raises StaleData if any symbol's newest complete bar is not yesterday.
    """
    today = today if today is not None else today_utc()
    frames = {s: _load_symbol(s, today) for s in symbols}
    yesterday = today - timedelta(days=1)
    stale = {s: str(f.index[-1].date()) if len(f) else "none"
             for s, f in frames.items() if not len(f) or f.index[-1] != yesterday}
    if stale:
        raise StaleData(f"newest complete daily bar is not {yesterday.date()}: {stale}")
    return {field: pd.DataFrame({s: f[field] for s, f in frames.items()}).sort_index()
            for field in ("open", "close")}


def todays_open(symbols: list[str] = config.UNIVERSE, today: pd.Timestamp | None = None) -> dict[str, float]:
    """Open of the current (in-progress) daily bar: the paper execution price."""
    today = today if today is not None else today_utc()
    out = {}
    for s in symbols:
        k = rdata._get(f"{rdata.BASE}?symbol={s}&interval=1d&limit=2")
        bar = next((b for b in k if pd.Timestamp(b[0], unit="ms", tz="UTC") == today), None)
        if bar is None:
            raise StaleData(f"{s}: no daily bar opened at {today.date()} yet")
        out[s] = float(bar[1])
    return out
