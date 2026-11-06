"""Daily bars (open, close, quote volume) for every USDT spot pair Binance ever listed.

Currently listed pairs come from data-api.binance.vision; delisted pairs, and
the early history of pairs that were delisted and relisted (LUNA), come from
the public archive at data.binance.vision. Without the dead coins the
universe study would only see survivors.

Usage: python -m research.universe_data
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from research.data import END, _ms

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE = "https://data.binance.vision"
API = "https://data-api.binance.vision/api/v3"
OUT = os.path.join("data", "klines_universe")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

STABLE = {"USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDS", "USDSB", "SUSD", "UST",
          "USTC", "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "UAH", "NGN", "BIDR", "IDRT", "BKRW",
          "EURI", "AEUR", "USDE", "USD1", "RLUSD", "XUSD", "BFUSD", "RWUSD", "USDD", "ZUSD"}
DUPLICATE = {"WBTC", "WBETH", "BNSOL", "BETH", "STETH", "PAXG", "XAUT", "BTTC"}
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")


def _get(url: str, raw: bool = False, tries: int = 5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            if i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(1.5 * 2 ** i)


def _s3_list(prefix: str, delimiter: bool) -> list[str]:
    out, marker = [], ""
    while True:
        url = f"{S3}?prefix={prefix}" + ("&delimiter=/" if delimiter else "") + (f"&marker={marker}" if marker else "")
        root = ET.fromstring(_get(url, raw=True))
        keys = [e.text for e in root.iterfind(".//s3:CommonPrefixes/s3:Prefix" if delimiter else ".//s3:Contents/s3:Key", NS)]
        out += keys
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true" or not keys:
            return out
        marker = root.findtext("s3:NextMarker", namespaces=NS) or keys[-1]


def pool() -> list[str]:
    syms = [p.rstrip("/").split("/")[-1] for p in _s3_list("data/spot/monthly/klines/", True)]
    keep = []
    for s in syms:
        if not s.endswith("USDT"):
            continue
        base = s[:-4]
        if base in STABLE or base in DUPLICATE or LEVERAGED.search(base) or not base:
            continue
        keep.append(s)
    return sorted(set(keep))


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame([r[:8] for r in rows], columns=["t", "open", "high", "low", "close", "volume", "ct", "qv"])
    t = pd.to_numeric(df["t"])
    t = t.where(t < 1e14, t // 1000)  # the archive switched to microseconds in 2025
    df["date"] = pd.to_datetime(t, unit="ms", utc=True)
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df[["open", "close", "qv"]].astype(float)


def _from_api(sym: str) -> pd.DataFrame:
    rows, start, end_ms = [], 0, _ms(END)
    while True:
        k = _get(f"{API}/klines?symbol={sym}&interval=1d&startTime={start}&endTime={end_ms - 1}&limit=1000")
        if not k:
            break
        rows += k
        if len(k) < 1000:
            break
        start = k[-1][0] + 1
    return _frame(rows) if rows else pd.DataFrame(columns=["open", "close", "qv"])


def _archive_months(sym: str) -> list[str]:
    keys = _s3_list(f"data/spot/monthly/klines/{sym}/1d/", False)
    return sorted(k for k in keys if k.endswith(".zip"))


def _from_archive(keys: list[str]) -> pd.DataFrame:
    rows = []
    for k in keys:
        b = _get(f"{ARCHIVE}/{k}", raw=True)
        if not b:
            continue
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            for name in z.namelist():
                for line in z.read(name).decode().splitlines():
                    parts = line.split(",")
                    if parts and parts[0].isdigit():
                        rows.append(parts)
    return _frame(rows) if rows else pd.DataFrame(columns=["open", "close", "qv"])


def fetch(sym: str) -> tuple[str, int, str]:
    path = os.path.join(OUT, f"{sym}_1d.parquet")
    if os.path.exists(path):
        return sym, len(pd.read_parquet(path)), "cached"
    api = _from_api(sym)
    months = _archive_months(sym)
    if len(api):
        first_api = api.index[0].strftime("%Y-%m")
        early = [k for k in months if re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) < first_api]
        df = pd.concat([_from_archive(early), api]) if early else api
        src = "api+archive" if early else "api"
    else:
        df, src = _from_archive(months), "archive"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index < pd.Timestamp(END, tz="UTC")]
    df.to_parquet(path)
    return sym, len(df), src


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    syms = pool()
    print(f"pool: {len(syms)} USDT pairs (stable/leveraged/duplicates excluded)", flush=True)
    done = 0
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(fetch, s): s for s in syms}
        for f in as_completed(futs):
            done += 1
            try:
                s, n, src = f.result()
                if done % 25 == 0 or src != "api":
                    print(f"[{done}/{len(syms)}] {s:<14} {n:>5} bars  {src}", flush=True)
            except Exception as e:
                print(f"[{done}/{len(syms)}] {futs[f]:<14} FAILED {type(e).__name__}: {e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
