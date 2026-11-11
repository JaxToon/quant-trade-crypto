"""The fund's always-on process: one loop, three clocks.

    python -m fund.daemon                 # paper (default)
    python -m fund.daemon --mode live     # refuses unless armed, like run_daily

  every minute  watch():  mark the book to live prices, write a heartbeat,
                          warn at 20% drawdown, halt buys at the 30% limit
                          the moment it is crossed (not at the next daily run)
  every hour    hourly(): the daily run (idempotent per UTC day, replays missed
                          days), an equity snapshot, a price-feed check
  every day     the C4 trades happen inside the first hourly() after 00:00 UTC

Only C4 daily places orders. No intraday strategy is wired in because none has
passed its pre-registered test (docs/RISET_INTRADAY_2026-11.md); one that does
is added as another Job below.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from fund import config, ledger, market, notify, run_daily

log = logging.getLogger("fund")

HEARTBEAT = os.path.join(config.DATA_DIR, "fund_heartbeat.json")
SNAPSHOTS = os.path.join(config.DATA_DIR, "fund_equity_hourly.csv")
DD_WARN = 0.20                # tell the owner well before the 30% halt
FEED_DOWN_MINUTES = 10        # this many minutes without prices => alert once
DOWN_ALERT_SECONDS = 15 * 60  # a heartbeat gap longer than this is reported at start


@dataclass
class Job:
    name: str
    every: int                # seconds
    fn: Callable[["State"], None]
    offset: int = 0           # seconds past the period boundary
    next: float = 0.0


@dataclass
class State:
    mode: str = "paper"
    warned: bool = False      # 20% warning already sent for this drawdown
    feed_misses: int = 0
    last_error: str = ""      # dedupes the hourly failure alert
    last: dict | None = None  # newest valuation


def _path(mode: str) -> str:
    return config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_heartbeat() -> dict | None:
    try:
        with open(HEARTBEAT, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def valuation(mode: str, prices: dict[str, float]) -> dict | None:
    """The book at live prices. None when there is no book yet or a held coin has no price."""
    if not os.path.exists(_path(mode)):
        return None
    book = ledger.load(_path(mode), mode, config.PAPER_START_CAPITAL)
    held = {k: q for k, q in book.qty.items() if q > 0}
    if any(k not in prices for k in held):
        return None
    pos = sum(q * prices[k] for k, q in held.items())
    equity = book.cash + pos
    peak = max(book.peak_equity, 1e-12)
    return {"equity": equity, "drawdown": max(0.0, 1 - equity / peak),
            "exposure": pos / equity if equity > 0 else 0.0, "halted": book.halted}


def watch(st: State) -> None:
    prices = market.live_prices()
    v = valuation(st.mode, prices) if prices else None
    if not prices:
        st.feed_misses += 1
        if st.feed_misses == FEED_DOWN_MINUTES:
            notify.send(f"Fund {st.mode}: no live prices for {FEED_DOWN_MINUTES} minutes "
                        f"(network or data-api.binance.vision). The fund is flying blind.")
    else:
        if st.feed_misses >= FEED_DOWN_MINUTES:
            notify.send(f"Fund {st.mode}: live prices are back after {st.feed_misses} minutes.")
        st.feed_misses = 0
    if v:
        st.last = v
        dd = v["drawdown"]
        if dd >= config.MAX_DRAWDOWN_HALT and not v["halted"]:
            _halt(st.mode, f"intraday drawdown {dd:.1%} >= {config.MAX_DRAWDOWN_HALT:.0%} limit")
        elif dd >= DD_WARN and not st.warned:
            st.warned = True
            notify.send(f"Fund {st.mode}: drawdown {dd:.1%} (equity ${v['equity']:,.2f}). "
                        f"Buys stop automatically at {config.MAX_DRAWDOWN_HALT:.0%}.")
        elif dd < DD_WARN * 0.75:
            st.warned = False
    _write_json(HEARTBEAT, {"ts": time.time(), "at": _now(), "pid": os.getpid(), "mode": st.mode,
                            "prices_ok": bool(prices), **(v or {})})


def _halt(mode: str, reason: str) -> None:
    """Same switch the daily run sets: exits still happen, nothing is bought."""
    try:
        with run_daily.run_lock(config.LOCK_FILE):
            book = ledger.load(_path(mode), mode, config.PAPER_START_CAPITAL)
            if book.halted:
                return
            book.halted, book.halt_reason = True, reason
            ledger.save(book, _path(mode))
    except run_daily.RunLocked:
        return  # a daily run is in progress; the next minute tries again
    notify.send(f"Fund {mode}: HALTED, {reason}. No new buys until /lanjut.")


def hourly(st: State) -> None:
    try:
        run_daily.run(st.mode, catch_up=True)
        st.last_error = ""
    except run_daily.RunLocked as e:
        log.info("%s; skipping", e)
    except Exception as e:
        log.exception("fund run failed")
        err = f"{type(e).__name__}: {e}"
        if err != st.last_error:  # the retry an hour later must not repeat the alert
            notify.send(f"Fund {st.mode}: run FAILED (retrying hourly): {err}")
        st.last_error = err
    if st.last:
        new = not os.path.exists(SNAPSHOTS)
        with open(SNAPSHOTS, "a", encoding="utf-8") as f:
            if new:
                f.write("at,mode,equity,drawdown,exposure\n")
            f.write(f"{_now()},{st.mode},{st.last['equity']:.4f},{st.last['drawdown']:.6f},"
                    f"{st.last['exposure']:.6f}\n")


JOBS = [Job("watch", 60, watch), Job("hourly", 3600, hourly, offset=600)]


def _due(job: Job, now: float) -> float:
    """Next boundary of the job's period (hourly at hh:10, so the day's bar is closed)."""
    return ((now - job.offset) // job.every + 1) * job.every + job.offset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    a = ap.parse_args(argv)
    handlers = [logging.FileHandler(os.path.join(config.DATA_DIR, "fund.log"), encoding="utf-8")]
    if sys.stdout is not None:  # None under pythonw.exe (the scheduled task)
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)

    hb = read_heartbeat()
    if hb and hb.get("pid") != os.getpid() and time.time() - hb["ts"] < 150:
        log.info("daemon already running (pid %s); exiting", hb.get("pid"))
        return 0
    if hb and time.time() - hb["ts"] > DOWN_ALERT_SECONDS:
        notify.send(f"Fund {a.mode}: daemon is back. It was down {(time.time() - hb['ts']) / 3600:.1f} h "
                    f"(last heartbeat {hb.get('at')}). Missed days are replayed now.")
    st = State(mode=a.mode)
    log.info("fund daemon started, mode=%s pid=%s", a.mode, os.getpid())
    first = True
    while True:
        now = time.time()
        for job in JOBS:
            if first or now >= job.next:
                try:
                    job.fn(st)
                except Exception:
                    log.exception("job %s failed", job.name)
                job.next = _due(job, time.time())
        first = False
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
