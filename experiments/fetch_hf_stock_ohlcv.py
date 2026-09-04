"""
One-off data-engineering script: pulls our 8 STOCK_WATCHLIST tickers'
1-minute OHLCV history from the mito0o852/OHLCV-1m HuggingFace dataset
(https://huggingface.co/datasets/mito0o852/OHLCV-1m, Finnhub-sourced,
1992-2026, ~87.7GB across thousands of tickers) WITHOUT downloading the
full dataset -- duckdb queries each month's remote parquet file directly
over HTTP and filters to just our tickers before anything touches disk.

Always sets `TimeZone='UTC'` before any date-based filtering -- DuckDB
defaults to the session's LOCAL timezone for `CAST(timestamp AS DATE)`,
which silently shifted an earlier sanity check's date boundaries by 7
hours (see RESEARCH_NOTES.md / project_stock_ohlcv_dataset_prep memory
for the full story of how that got diagnosed and resolved 2026-09-03).
Verified clean against yfinance's own 1h bars for the same real day
before this script was written -- do not skip the TimeZone='UTC' line if
extending this script, or the same silent-shift bug will resurface.

Output: one parquet file per ticker under
data/stock_ohlcv_1m/<TICKER>.parquet, columns timestamp/open/high/low/
close/volume, sorted chronologically. Intended as the raw ingredient for
train_gnn.py's stock path to eventually replace/augment its current
1h/730d yfinance fallback (which undertrained badly -- see
RESEARCH_NOTES.md) with real 1-minute depth -- NOT wired in yet, this
script only builds the local cache.

Usage: python -m experiments.fetch_hf_stock_ohlcv [--start 2022-01] [--end 2026-03]
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date

import duckdb

from config import settings

OUT_DIR = "data/stock_ohlcv_1m"
BASE_URL = "https://huggingface.co/datasets/mito0o852/OHLCV-1m/resolve/main/data"
TICKERS = tuple(settings.STOCK_WATCHLIST)


def month_range(start: str, end: str) -> list[str]:
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01")
    parser.add_argument("--end", default="2026-03")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    months = month_range(args.start, args.end)
    print(f"Fetching {len(months)} months ({args.start}..{args.end}) for {len(TICKERS)} tickers: {TICKERS}")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET TimeZone='UTC';")  # critical -- see module docstring

    per_ticker_frames: dict[str, list] = {t: [] for t in TICKERS}
    total_rows = 0
    t0 = time.time()

    for i, month in enumerate(months):
        url = f"{BASE_URL}/ohlcv_{month}.parquet"
        step_t0 = time.time()
        try:
            df = con.execute(
                f"""
                SELECT ticker, timestamp, open, high, low, close, volume
                FROM read_parquet('{url}')
                WHERE ticker IN {TICKERS}
                ORDER BY ticker, timestamp
                """
            ).df()
        except duckdb.Error as exc:
            print(f"  [{i+1}/{len(months)}] {month}: FAILED ({exc}) -- skipping")
            continue

        elapsed = time.time() - step_t0
        total_rows += len(df)
        print(f"  [{i+1}/{len(months)}] {month}: {len(df)} rows in {elapsed:.1f}s")

        for ticker, group in df.groupby("ticker"):
            per_ticker_frames[ticker].append(group.drop(columns=["ticker"]))

    print(f"\nFetched {total_rows} total rows across {len(months)} months in {time.time()-t0:.0f}s.")
    print("Writing per-ticker parquet files...")

    import pandas as pd

    for ticker, frames in per_ticker_frames.items():
        out_path = os.path.join(OUT_DIR, f"{ticker}.parquet")
        new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Merge with whatever's already on disk instead of overwriting it --
        # this script is meant to be re-run for a single month (e.g. to
        # patch a failed fetch) without destroying previously-fetched
        # months. Dedup by timestamp, keeping the newly-fetched row on
        # conflict (freshest wins).
        if os.path.exists(out_path):
            existing = pd.read_parquet(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        if combined.empty:
            print(f"  {ticker}: no data (new or existing), skipping")
            continue

        combined = (
            combined.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        combined.to_parquet(out_path, index=False)
        print(f"  {ticker}: {len(combined)} rows -> {out_path} "
              f"({combined['timestamp'].min()} .. {combined['timestamp'].max()})")

    print("\nDone.")


if __name__ == "__main__":
    main()
