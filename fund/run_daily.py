"""One trading day of the fund. Schedule it once a day, a few minutes after 00:00 UTC.

    python -m fund.run_daily              # paper (default)
    python -m fund.run_daily --dry-run    # show the decision, change nothing
    python -m fund.run_daily --mode live  # exchange; refuses unless armed

Idempotent: a second run on the same UTC day does nothing, and a lock file
keeps two runs (hourly loop + scheduled task) from overlapping.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import timedelta

import numpy as np
import pandas as pd

from fund import config, core, ledger, market, notify

log = logging.getLogger("fund")


class RunLocked(RuntimeError):
    pass


@contextmanager
def run_lock(path: str):
    """Exclusive lock file. A lock older than LOCK_STALE_SECONDS is a crashed run and is taken over."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - os.path.getmtime(path) > config.LOCK_STALE_SECONDS:
                os.remove(path)
                continue
            raise RunLocked(f"another fund run holds {path}")
    else:
        raise RunLocked(f"could not take {path}")
    try:
        os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
        os.close(fd)
        yield
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _summary(book: ledger.Book, run: dict) -> str:
    lines = [f"Fund {book.mode.upper()} {run['date']}",
             f"Equity ${run['equity']:,.2f} ({run['equity'] / book.start_capital - 1:+.2%} since start)",
             f"Drawdown {run['drawdown']:.1%} | exposure {run['exposure']:.0%} | fee ${run['fee']:.2f}"]
    for o in run["orders"]:
        lines.append(f"  {o['side']} {o['symbol']} ${o['notional']:,.2f}")
    if not run["orders"]:
        lines.append("  no trades today")
    if run.get("late_cost"):
        lines.append(f"  late-run cost vs open: ${run['late_cost']:+.2f}")
    if run.get("stale"):
        lines.append(f"  held without trading (stale data): {', '.join(run['stale'])}")
    if book.halted:
        lines.append(f"HALTED: {book.halt_reason}")
    return "\n".join(lines)


def _rebalance_due(book: ledger.Book, decision_day: pd.Timestamp) -> bool:
    """True if a Monday falls after the last run's decision day, up to this one.
    A missed Tuesday run no longer skips the week's drift rebalance."""
    if not book.runs:
        return decision_day.dayofweek == 0
    prev = book.runs[-1]
    last = (pd.Timestamp(prev["decision_day"], tz="UTC") if "decision_day" in prev
            else pd.Timestamp(prev["date"], tz="UTC") - timedelta(days=1))
    days = pd.date_range(last + timedelta(days=1), decision_day, freq="D")
    return bool((days.dayofweek == 0).any())


def run(mode: str = "paper", dry_run: bool = False, broker=None, today=None,
        catch_up: bool = False) -> dict | None:
    if dry_run:
        return _run(mode, True, broker, today)
    with run_lock(config.LOCK_FILE):
        if catch_up:
            _catch_up(mode, today if today is not None else market.today_utc())
        return _run(mode, False, broker, today)


def _catch_up(mode: str, today: pd.Timestamp) -> list[str]:
    """Replay the UTC days the bot was down for, oldest first.

    Paper only: a paper run fills at that day's real open, which is still known,
    so the replayed day is exactly what an on-time run would have booked (each
    is flagged "backfilled"). Live cannot trade in the past; it only reports the gap.
    """
    path = config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER
    book = ledger.load(path, mode, config.PAPER_START_CAPITAL)
    if not book.runs:
        return []
    last = pd.Timestamp(book.last_run_date, tz="UTC")
    missed = pd.date_range(last + timedelta(days=1), today - timedelta(days=1), freq="D")
    if not len(missed):
        return []
    span = f"{missed[0].date()}..{missed[-1].date()} ({len(missed)} day(s))"
    if mode != "paper":
        notify.send(f"Fund live: no run on {span}. Those days cannot be traded after the fact.")
        return []
    done = [str(d.date()) for d in missed if _run(mode, False, None, d, backfilled=True)]
    notify.send(f"Fund paper: bot was down {span}; replayed {len(done)} day(s) at each day's real open.")
    return done


def _run(mode: str, dry_run: bool, broker, today, backfilled: bool = False) -> dict | None:
    path = config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER
    if mode == "live" and not os.path.exists(path) and os.path.exists(config.LIVE_MARKER):
        raise RuntimeError(f"live ledger {path} is missing but live trading has started before "
                           f"({config.LIVE_MARKER}); restore it instead of starting over")
    book = ledger.load(path, mode, config.PAPER_START_CAPITAL)
    today = today if today is not None else market.today_utc()
    day = str(today.date())
    if book.last_run_date == day:
        log.info("already ran for %s; nothing to do", day)
        return None

    syms = config.UNIVERSE
    try:
        h = market.history(syms, today)
    except market.StaleData as e:
        notify.send(f"Fund {mode}: NO TRADES today, data is stale. {e}")
        raise
    stale = set(h.get("stale", []))
    if stale:
        notify.send(f"Fund {mode}: stale data for {sorted(stale)}; those are held without trading today")
    sig, tgt = core.targets(h["close"])
    decision_day = today - timedelta(days=1)
    t = tgt.reindex(columns=syms).loc[decision_day].fillna(0.0).to_numpy(dtype=float)
    s = sig.reindex(columns=syms).loc[decision_day].fillna(0.0).to_numpy(dtype=float)
    rebalance = _rebalance_due(book, decision_day)
    last_close = h["close"].ffill().iloc[-1]

    if mode == "paper":
        prices = market.todays_open(syms, today)
        for k in syms:  # no open yet (stale feed): value it at its last close
            if k not in prices:
                stale.add(k)
                prices[k] = float(last_close[k])
        qty, cash = dict(book.qty), book.cash
    else:
        if broker is None:
            from fund.broker import ExchangeBroker
            broker = ExchangeBroker()
        prices = broker.prices(syms)
        ex_qty, ex_cash = broker.positions(syms)
        if not book.runs:
            if config.LIVE_CAPITAL <= 0:
                raise RuntimeError("FUND_LIVE_CAPITAL is not set: refusing to guess how much money to manage")
            if ex_cash < config.LIVE_CAPITAL:
                raise RuntimeError(f"exchange holds {ex_cash:.2f} USDT < FUND_LIVE_CAPITAL {config.LIVE_CAPITAL:.2f}")
            book.cash, book.qty = config.LIVE_CAPITAL, {}
        tol = 1 - config.RECONCILE_TOLERANCE
        short = {k: (book.qty.get(k, 0.0), ex_qty[k]) for k in syms
                 if ex_qty[k] < book.qty.get(k, 0.0) * tol - 1e-12}
        if ex_cash < book.cash * tol - 1.0:
            short["USDT"] = (book.cash, ex_cash)
        if short:
            book.halted, book.halt_reason = True, f"exchange holds less than the book: {short}"
            ledger.save(book, path)
            notify.send(f"Fund live: HALTED, reconciliation failed {short}. No orders placed.")
            raise RuntimeError(book.halt_reason)
        qty, cash = dict(book.qty), book.cash

    px = np.array([prices[k] for k in syms])
    q = np.array([qty.get(k, 0.0) for k in syms])
    equity = cash + float((q * px).sum())
    if not book.runs:  # first day: the book starts from what is actually there
        book.start_capital = book.peak_equity = equity
    book.peak_equity = max(book.peak_equity, equity)
    dd = 1 - equity / book.peak_equity if book.peak_equity else 0.0
    held = q * px / equity if equity > 0 else np.zeros(len(syms))

    if os.path.exists(config.KILL_FILE) and not book.halted:
        book.halted, book.halt_reason = True, f"kill switch file {config.KILL_FILE} present"
    if dd >= config.MAX_DRAWDOWN_HALT and not book.halted:
        book.halted, book.halt_reason = True, f"drawdown {dd:.1%} >= {config.MAX_DRAWDOWN_HALT:.0%} limit"
    if book.halted:
        t = np.minimum(t, held)  # exits still happen; nothing is bought or added to

    last_sig = np.array([book.last_sig.get(k, 0.0) for k in syms])
    hold = np.array([k in stale for k in syms])
    if hold.any():  # stale coins keep their position and signal memory untouched
        t = np.where(hold, held, t)
        s = np.where(hold, last_sig, s)
    orders, new_q, new_cash, new_last, eq_before, fee = core.plan_orders(
        syms, qty, cash, prices, t, s, last_sig, rebalance,
        min_order=config.MIN_ORDER_USDT)
    orders = [o for o in orders if o.symbol not in stale]

    too_big = [o for o in orders if o.notional > config.MAX_ORDER_FRACTION * equity]
    if too_big:
        msg = f"Fund {mode}: ABORTED, order above {config.MAX_ORDER_FRACTION:.0%} of equity: {too_big}"
        notify.send(msg)
        raise RuntimeError(msg)

    run_rec = {"date": day, "decision_day": str(decision_day.date()), "rebalance": bool(rebalance),
               "equity": eq_before, "drawdown": dd, "fee": fee, "stale": sorted(stale),
               "target": dict(zip(syms, map(float, t))), "prices": prices,
               "exposure": float(sum(new_q[k] * prices[k] for k in syms) / eq_before) if eq_before else 0.0,
               "orders": [{"symbol": o.symbol, "side": o.side, "qty": o.qty, "price": o.price,
                           "notional": o.notional} for o in orders]}

    if backfilled:
        run_rec["backfilled"] = True
    elif mode == "paper" and today == market.today_utc():
        # What filling at the market right now, instead of at the open, would have
        # cost: the gap the live bot will actually pay for running late.
        mark = market.live_prices(syms)
        if mark:
            run_rec["mark_prices"] = mark
            run_rec["late_cost"] = float(sum(
                (1 if o.side == "BUY" else -1) * (mark[o.symbol] / o.price - 1) * o.notional
                for o in orders if o.symbol in mark))

    if dry_run:
        print(_summary(book, run_rec))
        return run_rec

    if mode == "paper":
        book.qty, book.cash = {k: float(v) for k, v in new_q.items()}, float(new_cash)
        fills = [dict(o, fee=o["notional"] * config.COST, date=day) for o in run_rec["orders"]]
    else:
        before_q, before_c = broker.positions(syms)
        fills, failure = [], None
        try:
            fills = [dict(f, date=day) for f in broker.execute(orders, cash_available=cash)]
        except Exception as e:  # an order failed mid-rebalance: book what DID happen, then stop
            failure = e
        finally:
            after_q, after_c = broker.positions(syms)
            # Book moves by what actually changed on the exchange (fees included),
            # attributable because the fund is the only thing trading during the run.
            book.qty = {k: max(0.0, qty.get(k, 0.0) + after_q[k] - before_q[k]) for k in syms}
            book.cash = cash + after_c - before_c
        if failure is not None:
            book.halted = True
            book.halt_reason = f"order failed mid-rebalance on {day}: {type(failure).__name__}: {failure}"
            run_rec["failed"] = book.halt_reason
        open(config.LIVE_MARKER, "a").close()
    book.last_sig = dict(zip(syms, map(float, new_last)))
    book.trades += fills
    book.runs.append(run_rec)
    ledger.save(book, path)
    if backfilled:
        log.info("BACKFILL: %s", _summary(book, run_rec).replace("\n", " | "))
        return run_rec
    notify.send(_summary(book, run_rec))
    if today.dayofweek == 0:  # weekly forward-test report (paper vs backtest vs BTC)
        try:
            from fund import report
            notify.send(report.text(report.compare(mode)))
        except Exception as e:
            log.warning("weekly report failed: %s", e)
    if mode == "live" and book.halted and run_rec.get("failed"):
        raise RuntimeError(run_rec["failed"])
    return run_rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    handlers = [logging.FileHandler(os.path.join(config.DATA_DIR, "fund.log"), encoding="utf-8")]
    if sys.stdout is not None:  # None under pythonw.exe (the scheduled task)
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
    try:
        rec = run(a.mode, a.dry_run, catch_up=True)
    except RunLocked as e:
        log.info("%s; skipping", e)
        return 0
    except Exception as e:
        log.exception("fund run failed")
        notify.send(f"Fund {a.mode}: run FAILED: {type(e).__name__}: {e}")
        return 1
    if rec and not a.dry_run:
        print(f"ok {rec['date']} equity={rec['equity']:.2f} orders={len(rec['orders'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
