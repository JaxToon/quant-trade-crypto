"""Binance spot klines from data-api.binance.vision, cached as parquet.

data-api.binance.vision is Binance's public market-data host. It is reachable
from this network, unlike *.binance.com (see README), and it serves the full
history, not just the 60 days yfinance allows for 5m/15m.

Usage: python -m research.data [--intervals 1d 4h 1h] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

BASE = "https://data-api.binance.vision/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT"]
CACHE_DIR = os.path.join("data", "klines")
# Exclusive end: the last complete daily bar of the pre-registered study is
# 2026-10-10. Pinning it keeps every run on identical data.
END = "2026-10-11"


def _get(url: str):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch(symbol: str, interval: str, end: str = END) -> pd.DataFrame:
    end_ms = _ms(end)
    rows, start = [], 0
    while True:
        k = _get(f"{BASE}?symbol={symbol}&interval={interval}&startTime={start}"
                 f"&endTime={end_ms - 1}&limit=1000")
        if not k:
            break
        rows += k
        if len(k) < 1000:
            break
        start = k[-1][0] + 1
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "qv", "n", "tb", "tq", "ig"])
    df = df[df["close_time"] < end_ms]  # complete bars only
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def load(symbol: str, interval: str, refresh: bool = False) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")
    if os.path.exists(path) and not refresh:
        return pd.read_parquet(path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = fetch(symbol, interval)
    df.to_parquet(path)
    return df


def panel(interval: str = "1d", symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Wide frames (dates x symbols) for open/high/low/close; NaN before listing."""
    symbols = symbols or SYMBOLS
    frames = {s: load(s, interval) for s in symbols}
    out = {}
    for field in ("open", "high", "low", "close", "volume"):
        out[field] = pd.DataFrame({s: f[field] for s, f in frames.items()}).sort_index()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", nargs="+", default=["1d"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    for iv in a.intervals:
        for s in SYMBOLS:
            df = load(s, iv, refresh=a.refresh)
            print(f"{s:<9} {iv:<3} {len(df):>7} bars  {df.index[0].date()} -> {df.index[-1]}", flush=True)
