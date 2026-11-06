"""Binance USDT-M perpetual klines (15m, 1h) for the C4 coins, from data.binance.vision.

Usage: python -m research.intraday_data
"""
from __future__ import annotations

import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from research.data import SYMBOLS
from research.universe_data import _get

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
OUT = os.path.join("data", "klines_perp")
MONTHS = pd.period_range("2019-09", "2026-10", freq="M").astype(str)


def _month(sym: str, tf: str, ym: str) -> pd.DataFrame | None:
    b = _get(f"{BASE}/{sym}/{tf}/{sym}-{tf}-{ym}.zip", raw=True)
    if not b:
        return None
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(df.iloc[0, 0]).isdigit():   # newer files carry a header row
        df = df.iloc[1:]
    df = df.iloc[:, :6].astype(float)
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    t = df["t"].where(df["t"] < 1e14, df["t"] // 1000)
    df.index = pd.to_datetime(t, unit="ms", utc=True)
    return df[["open", "close", "volume"]]


def fetch(sym: str, tf: str) -> str:
    path = os.path.join(OUT, f"{sym}_{tf}.parquet")
    if os.path.exists(path):
        return f"{sym} {tf} cached"
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda ym: _month(sym, tf, ym), MONTHS) if p is not None]
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()]
    df = df[df.index < pd.Timestamp("2026-10-10", tz="UTC")]
    df.to_parquet(path)
    return f"{sym} {tf} {len(df)} bars {df.index[0]} -> {df.index[-1]}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for tf in ("1h", "15m"):
        for s in SYMBOLS:
            print(fetch(s, tf), flush=True)


if __name__ == "__main__":
    main()
