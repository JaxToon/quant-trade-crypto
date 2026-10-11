"""Descriptive look at funding-rate carry (long spot + short perpetual), per the
pre-registration's "also reported" section. Not a candidate: it needs a futures
venue alongside the spot one.

Source: Binance's public archive data.binance.vision (reachable from this
network), monthly fundingRate files for the USDT-margined perpetuals.

Usage: python -m research.carry_analysis
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
CACHE = os.path.join("data", "klines", "funding")
# Cost of one full cycle (open and later close BOTH legs), each side:
# spot at Tokocrypto 0.3822% + 0.05% slippage; perp at an ASSUMED 0.05% + 0.05%
# (Bittime's official futures fee could not be verified).
CYCLE_COST = 2 * (0.004322 + 0.001)


def _month_file(sym: str, ym: str) -> pd.DataFrame | None:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{sym}-{ym}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    url = f"{BASE}/{sym}/{sym}-fundingRate-{ym}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
    except Exception:
        return None
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f)
    df.to_csv(path, index=False)
    return df


def load(sym: str) -> pd.Series:
    frames = []
    for ym in pd.period_range("2019-09", "2026-09", freq="M").astype(str):
        df = _month_file(sym, ym)
        if df is not None and len(df):
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    tcol = "calc_time" if "calc_time" in df.columns else df.columns[0]
    rcol = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[-1]
    s = pd.Series(df[rcol].astype(float).to_numpy(),
                  index=pd.to_datetime(df[tcol], unit="ms", utc=True)).sort_index()
    return s[~s.index.duplicated()]


def main() -> None:
    print("FUNDING-RATE CARRY: long spot + short perpetual, funding collected per unit of notional")
    print(f"cycle cost assumed {CYCLE_COST*100:.2f}% (both legs in and out; perp fee is an assumption)\n")
    for sym in SYMBOLS:
        s = load(sym)
        by_year = s.groupby(s.index.year).agg(["sum", "count", lambda x: (x < 0).mean()])
        by_year.columns = ["funding_sum", "n", "share_negative"]
        print(f"{sym}: {len(s)} funding prints {s.index[0].date()} -> {s.index[-1].date()}")
        print(f"  {'year':<6}{'gross carry':>12}{'% prints < 0':>14}{'net of 1 cycle':>16}")
        for y, row in by_year.iterrows():
            print(f"  {y:<6}{row.funding_sum*100:>11.2f}%{row.share_negative*100:>13.1f}%"
                  f"{(row.funding_sum - CYCLE_COST)*100:>15.2f}%")
        m = s.resample("ME").sum()
        print(f"  worst month {m.min()*100:+.2f}% ({m.idxmin().date()}), "
              f"months negative {(m < 0).mean()*100:.0f}%, last 12 months {m.iloc[-12:].sum()*100:+.2f}%\n")
    print("Per unit of NOTIONAL. On capital it is lower: the short leg needs margin "
          "(at 2x margin, total capital is 1.5x the notional, so divide by 1.5).")


if __name__ == "__main__":
    main()
