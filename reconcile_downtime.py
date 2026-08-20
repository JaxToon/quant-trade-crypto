"""
Reconciles paper-ledger open positions against real historical price data
covering a shutdown-to-restart downtime window -- so a position whose
SL/TP was genuinely crossed while both processes were stopped doesn't sit
reported as "still open" until someone notices.

PRECONDITION: run this only while the live process(es) for the asset
type(s) being reconciled are actually stopped. This script writes
directly to the real paper_ledger_{crypto,stock}.json files; running it
while main.py is still running risks a save race (main.py's own periodic
save could silently overwrite what this script wrote, or vice versa).

check_and_close_ohlc() (backtrack/paper_trading.py) is the SAME method
the live system uses to resolve positions -- this replays real candles
through it for the downtime window specifically, so a reconciled close
uses the exact same SL/TP-priority logic (conservative SL-first
assumption when both fall in one candle) as live trading would have.

Downtime window: [ledger file's last-modified time, now]. The ledger
saves on every tick and on a graceful SIGTERM shutdown, so its mtime is a
reliable proxy for "last known good state before shutdown" -- confirmed
this empirically (2026-08-20): mtime matched the last real log line
almost exactly.

Data source: Binance klines first for crypto (matches what live trading
uses), yfinance fallback if Binance is unreachable (it has been flaky
this session -- SSL handshake failures, not just occasional timeouts).
yfinance only for stock, and skipped entirely if the whole downtime
window falls outside NASDAQ regular session (ingestion/yfinance_stream's
own is_regular_session(), reused here rather than reimplemented).

Backs up the ledger file before writing, to backtrack_logs/pre_reconcile_backup/.

Usage: python reconcile_downtime.py [--asset-type crypto|stock|both]
Default: both (whichever ledger files exist).
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone

import requests
import yfinance as yf

from backtest import CRYPTO_TICKERS
from backtrack.paper_trading import PaperTradingLedger
from config import settings
from ingestion.yfinance_stream import EASTERN, is_regular_session

LEDGER_DIR = "backtrack_logs"
BACKUP_DIR = os.path.join(LEDGER_DIR, "pre_reconcile_backup")


def _backup_ledger(path: str) -> None:
    if not os.path.exists(path):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(BACKUP_DIR, f"{os.path.basename(path)}.{stamp}.bak")
    shutil.copy2(path, dest)
    print(f"  backed up -> {dest}")


def _determine_window(ledger_path: str) -> tuple[datetime, datetime]:
    mtime = os.path.getmtime(ledger_path)
    shutdown = datetime.fromtimestamp(mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    return shutdown, now


def _fetch_crypto_window(symbol: str, start: datetime, end: datetime) -> list[dict]:
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1m", "startTime": start_ms, "endTime": end_ms, "limit": 1000},
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json()
        return [
            {"open_time": datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc), "high": float(r[2]), "low": float(r[3])}
            for r in rows
        ]
    except Exception as exc:
        print(f"    Binance klines failed for {symbol} ({exc}), trying yfinance")

    tickers = CRYPTO_TICKERS.get(symbol, [])
    for yf_ticker in tickers:
        try:
            df = yf.download(yf_ticker, period="7d", interval="1m", progress=False, auto_adjust=True)
            if df.empty:
                continue
            if df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
            window = df[(df.index >= start) & (df.index <= end)]
            return [
                {"open_time": ts.to_pydatetime(), "high": float(row["High"]), "low": float(row["Low"])}
                for ts, row in window.iterrows()
            ]
        except Exception as exc:
            print(f"    yfinance ({yf_ticker}) failed for {symbol}: {exc}")
    return []


def _fetch_stock_window(symbol: str, start: datetime, end: datetime) -> list[dict]:
    start_et, end_et = start.astimezone(EASTERN), end.astimezone(EASTERN)
    if not is_regular_session(start_et) and not is_regular_session(end_et) and start_et.date() == end_et.date():
        print(f"    {symbol}: downtime window entirely outside NASDAQ regular session, skipping fetch")
        return []
    try:
        df = yf.download(symbol, period="7d", interval="1m", progress=False, auto_adjust=True, prepost=False)
        if df.empty:
            return []
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        window = df[(df.index >= start) & (df.index <= end)]
        return [
            {"open_time": ts.to_pydatetime(), "high": float(row["High"]), "low": float(row["Low"])}
            for ts, row in window.iterrows()
        ]
    except Exception as exc:
        print(f"    yfinance failed for {symbol}: {exc}")
        return []


def reconcile(asset_type: str) -> None:
    ledger_path = os.path.join(LEDGER_DIR, f"paper_ledger_{asset_type}.json")
    if not os.path.exists(ledger_path):
        print(f"[{asset_type}] no ledger file at {ledger_path}, skipping")
        return

    start, end = _determine_window(ledger_path)
    minutes = (end - start).total_seconds() / 60
    print(f"\n[{asset_type}] downtime window: {start.isoformat()} -> {end.isoformat()} ({minutes:.1f} min)")

    _backup_ledger(ledger_path)
    ledger = PaperTradingLedger(path=ledger_path)
    if not ledger.open_positions:
        print(f"[{asset_type}] no open positions to reconcile")
        return

    symbols = sorted({pos.symbol for pos in ledger.open_positions.values()})
    print(f"[{asset_type}] open symbols: {symbols}")

    timeline: list[tuple[datetime, str, dict]] = []
    for symbol in symbols:
        candles = _fetch_crypto_window(symbol, start, end) if asset_type == "crypto" else _fetch_stock_window(symbol, start, end)
        print(f"    {symbol}: {len(candles)} candles fetched")
        for c in candles:
            timeline.append((c["open_time"], symbol, c))
    timeline.sort(key=lambda x: x[0])

    before_closed = len(ledger.closed_trades)
    for ts, symbol, c in timeline:
        ledger.check_and_close_ohlc(symbol, c["low"], c["high"], closed_at=ts.isoformat(), timeframe=None)

    new_trades = ledger.closed_trades[before_closed:]
    print(f"[{asset_type}] resolved during downtime: {len(new_trades)}")
    for t in new_trades:
        print(f"    {t['symbol']} {t['timeframe']} {t['direction']} -> {t['outcome'].upper()} @ {t['exit_price']:.6f} | PnL=${t['pnl_dollars']:.2f} | closed_at={t['closed_at']}")

    print(f"[{asset_type}] still open after reconciliation: {len(ledger.open_positions)}")
    for pos in ledger.open_positions.values():
        print(f"    {pos.symbol} {pos.market} {pos.direction} entry={pos.entry_price} SL={pos.stop_loss} TP={pos.target}")

    ledger.save()
    print(f"[{asset_type}] saved. balance=${ledger.balance:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile paper-ledger open positions against real price history for a shutdown-to-restart window.")
    parser.add_argument("--asset-type", choices=["crypto", "stock", "both"], default="both")
    args = parser.parse_args()

    print("PRECONDITION: this assumes the live process(es) being reconciled are currently stopped.")
    print("Running against a live, still-running process risks a ledger-save race.\n")

    asset_types = ["crypto", "stock"] if args.asset_type == "both" else [args.asset_type]
    for asset_type in asset_types:
        reconcile(asset_type)


if __name__ == "__main__":
    main()
