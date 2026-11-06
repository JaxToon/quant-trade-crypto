"""One trading day of the fund. Schedule it once a day, a few minutes after 00:00 UTC.

    python -m fund.run_daily              # paper (default)
    python -m fund.run_daily --dry-run    # show the decision, change nothing
    python -m fund.run_daily --mode live  # exchange; refuses unless armed

Idempotent: a second run on the same UTC day does nothing.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta

import numpy as np

from fund import config, core, ledger, market, notify

log = logging.getLogger("fund")


def _summary(book: ledger.Book, run: dict) -> str:
    lines = [f"Fund {book.mode.upper()} {run['date']}",
             f"Equity ${run['equity']:,.2f} ({run['equity'] / book.start_capital - 1:+.2%} since start)",
             f"Drawdown {run['drawdown']:.1%} | exposure {run['exposure']:.0%} | fee ${run['fee']:.2f}"]
    for o in run["orders"]:
        lines.append(f"  {o['side']} {o['symbol']} ${o['notional']:,.2f}")
    if not run["orders"]:
        lines.append("  no trades today")
    if book.halted:
        lines.append(f"HALTED: {book.halt_reason}")
    return "\n".join(lines)


def run(mode: str = "paper", dry_run: bool = False, broker=None, today=None) -> dict | None:
    path = config.PAPER_LEDGER if mode == "paper" else config.LIVE_LEDGER
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
    sig, tgt = core.targets(h["close"])
    decision_day = today - timedelta(days=1)
    t = tgt.loc[decision_day, syms].to_numpy(dtype=float)
    s = sig.loc[decision_day, syms].to_numpy(dtype=float)
    rebalance = decision_day.dayofweek == 0

    if mode == "paper":
        prices = market.todays_open(syms, today)
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

    if os.path.exists(config.KILL_FILE) and not book.halted:
        book.halted, book.halt_reason = True, f"kill switch file {config.KILL_FILE} present"
    if dd >= config.MAX_DRAWDOWN_HALT and not book.halted:
        book.halted, book.halt_reason = True, f"drawdown {dd:.1%} >= {config.MAX_DRAWDOWN_HALT:.0%} limit"
    if book.halted:
        # Exits still happen; nothing is bought or added to.
        held = q * px / equity if equity > 0 else np.zeros(len(syms))
        t = np.minimum(t, held)

    last_sig = np.array([book.last_sig.get(k, 0.0) for k in syms])
    orders, new_q, new_cash, new_last, eq_before, fee = core.plan_orders(
        syms, qty, cash, prices, t, s, last_sig, rebalance,
        min_order=config.MIN_ORDER_USDT)

    too_big = [o for o in orders if o.notional > config.MAX_ORDER_FRACTION * equity]
    if too_big:
        msg = f"Fund {mode}: ABORTED, order above {config.MAX_ORDER_FRACTION:.0%} of equity: {too_big}"
        notify.send(msg)
        raise RuntimeError(msg)

    run_rec = {"date": day, "decision_day": str(decision_day.date()), "rebalance": bool(rebalance),
               "equity": eq_before, "drawdown": dd, "fee": fee,
               "target": dict(zip(syms, map(float, t))), "prices": prices,
               "exposure": float(sum(new_q[k] * prices[k] for k in syms) / eq_before) if eq_before else 0.0,
               "orders": [{"symbol": o.symbol, "side": o.side, "qty": o.qty, "price": o.price,
                           "notional": o.notional} for o in orders]}

    if dry_run:
        print(_summary(book, run_rec))
        return run_rec

    if mode == "paper":
        book.qty, book.cash = {k: float(v) for k, v in new_q.items()}, float(new_cash)
        fills = [dict(o, fee=o["notional"] * config.COST, date=day) for o in run_rec["orders"]]
    else:
        before_q, before_c = broker.positions(syms)
        fills = [dict(f, date=day) for f in broker.execute(orders, cash_available=cash)]
        after_q, after_c = broker.positions(syms)
        # Book moves by what actually changed on the exchange (fees included),
        # attributable because the fund is the only thing trading during the run.
        book.qty = {k: max(0.0, qty.get(k, 0.0) + after_q[k] - before_q[k]) for k in syms}
        book.cash = cash + after_c - before_c
    book.last_sig = dict(zip(syms, map(float, new_last)))
    book.trades += fills
    book.runs.append(run_rec)
    ledger.save(book, path)
    notify.send(_summary(book, run_rec))
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
        rec = run(a.mode, a.dry_run)
    except Exception as e:
        log.exception("fund run failed")
        notify.send(f"Fund {a.mode}: run FAILED: {type(e).__name__}: {e}")
        return 1
    if rec and not a.dry_run:
        print(f"ok {rec['date']} equity={rec['equity']:.2f} orders={len(rec['orders'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
