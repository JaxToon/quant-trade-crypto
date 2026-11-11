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
    if os.path.exists(path):
        old = pd.read_parquet(path)
        if len(old) and old.index[-1] >= end - timedelta(days=1):
            return old[old.index < end]
    # Always fetch up to the real today, so replaying a past day (catch-up) fills
    # the cache once instead of re-downloading the whole history for every day.
    df = rdata.fetch(symbol, "1d", end=max(end, today_utc()).strftime("%Y-%m-%d"))
    df.to_parquet(path)
    return df[df.index < end]


def history(symbols: list[str] = config.UNIVERSE, today: pd.Timestamp | None = None) -> dict:
    """Wide open/close frames of complete bars up to and including yesterday, plus
    "stale": the symbols whose newest complete bar is not yesterday's.

    A stale symbol is held as it is (no trade) while the rest of the fund trades,
    so one bad feed cannot block every exit. Raises StaleData when more than half
    the universe is stale: then the data source itself is suspect.
    """
    today = today if today is not None else today_utc()
    frames = {}
    for s in symbols:
        try:
            frames[s] = _load_symbol(s, today)
        except Exception:
            frames[s] = pd.DataFrame(columns=["open", "close"], dtype=float)
    yesterday = today - timedelta(days=1)
    stale = {s: str(f.index[-1].date()) if len(f) else "none"
             for s, f in frames.items() if not len(f) or f.index[-1] != yesterday}
    if len(stale) * 2 > len(symbols):
        raise StaleData(f"newest complete daily bar is not {yesterday.date()} for most coins: {stale}")
    out = {field: pd.DataFrame({s: f[field] for s, f in frames.items()}).sort_index()
           for field in ("open", "close")}
    out["stale"] = sorted(stale)
    return out


def todays_open(symbols: list[str] = config.UNIVERSE, today: pd.Timestamp | None = None) -> dict[str, float]:
    """Open of the daily bar that starts on `today`: the paper execution price.
    Works for a past day too (catch-up after downtime replays that day's real open).
    Symbols without that bar are left out; the caller values them at their last close."""
    today = today if today is not None else today_utc()
    ms = int(today.timestamp() * 1000)
    out = {}
    for s in symbols:
        try:
            k = rdata._get(f"{rdata.BASE}?symbol={s}&interval=1d&startTime={ms}&limit=1") or []
        except Exception:
            continue
        if k and k[0][0] == ms:
            out[s] = float(k[0][1])
    return out


def live_prices(symbols: list[str] = config.UNIVERSE) -> dict[str, float]:
    """Last traded prices in one request. Empty on any network error: callers
    must treat "no prices" as "unknown", never as zero."""
    import json
    import urllib.parse
    import urllib.request
    try:
        q = urllib.parse.quote(json.dumps(list(symbols), separators=(",", ":")))
        with urllib.request.urlopen(f"https://data-api.binance.vision/api/v3/ticker/price?symbols={q}", timeout=5) as r:
            return {d["symbol"]: float(d["price"]) for d in json.loads(r.read())}
    except Exception:
        return {}
