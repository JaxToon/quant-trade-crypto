"""Drive the fund's live path end to end on Binance TESTNET (play money).

Arms fund.broker.ExchangeBroker against testnet.binance.vision, runs one
fund day through fund.run_daily.run("live") with a throwaway ledger (so the
real data/fund_ledger_live.json stays untouched), then checks that the
exchange balances moved the way the plan said.

Usage: python -m checks.testnet_fund_check
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

from fund import broker as fbroker
from fund import config, run_daily


def fail_mid_rebalance() -> int:
    """Second order is rejected on purpose: the fills that happened must be booked,
    the book must match the exchange, and the fund must halt."""
    from fund import ledger
    b = fbroker.ExchangeBroker(venue="binance_testnet", enabled=True, confirm_real_money=False)
    real, n = b.client.create_order, {"i": 0}

    def flaky(**kw):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("simulated exchange rejection on order 2")
        return real(**kw)
    b.client.create_order = flaky
    syms = config.UNIVERSE
    q0, c0 = b.positions(syms)
    config.LIVE_LEDGER = os.path.join(tempfile.mkdtemp(), "fund_ledger_testnet.json")
    config.LIVE_MARKER = config.LIVE_LEDGER + ".marker"
    config.LOCK_FILE = config.LIVE_LEDGER + ".lock"
    config.LIVE_CAPITAL = 1000.0
    try:
        run_daily.run("live", broker=b)
        print("RESULT: FAIL, no error was raised")
        return 1
    except RuntimeError as e:
        print("raised as expected:", e)
    book = ledger.load(config.LIVE_LEDGER, "live", 1000)
    q1, c1 = b.positions(syms)
    moved = {k: q1[k] - q0[k] for k in syms if abs(q1[k] - q0[k]) > 0}
    ok = (book.halted and len(book.runs) == 1
          and all(abs(book.qty.get(k, 0.0) - v) < 1e-9 for k, v in moved.items())
          and abs(book.cash - (1000.0 + c1 - c0)) < 1e-6)
    print(f"exchange moved {moved}, USDT {c1 - c0:+.2f}; book qty {({k: v for k, v in book.qty.items() if v})}, "
          f"cash {book.cash:.2f}, halted={book.halted}: {book.halt_reason[:60]}")
    print("RESULT:", "OK, the fill was booked, book matches exchange, fund halted" if ok else "MISMATCH")
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    # A testnet drill must never page the owner's phone with "Fund LIVE ... HALTED".
    run_daily.notify.send = lambda text: logging.info("NOTIFY (muted for testnet): %s", text.replace("\n", " | ")) or False
    if "--fail-mid" in sys.argv:
        return fail_mid_rebalance()
    b = fbroker.ExchangeBroker(venue="binance_testnet", enabled=True, confirm_real_money=False)
    assert b.is_testnet, "refusing: not a testnet endpoint"
    syms = config.UNIVERSE
    listed = []
    for s in syms:
        try:
            b._filter(s)
            listed.append(s)
        except Exception as e:
            print(f"  {s}: not usable on testnet ({e})")
    q0, cash0 = b.positions(syms)
    print(f"before: USDT {cash0:,.2f}  " + "  ".join(f"{s}={q0[s]:g}" for s in syms))

    config.LIVE_LEDGER = os.path.join(tempfile.mkdtemp(), "fund_ledger_testnet.json")
    config.LIVE_CAPITAL = 1000.0  # the fund manages $1,000; the rest of the testnet account is not its money
    rec = run_daily.run("live", broker=b)
    print(f"plan: {len(rec['orders'])} orders, equity {rec['equity']:,.2f}")
    for o in rec["orders"]:
        print(f"  {o['side']} {o['symbol']} qty={o['qty']:.6f} ~${o['notional']:,.2f}")

    q1, cash1 = b.positions(syms)
    print(f"after:  USDT {cash1:,.2f}  " + "  ".join(f"{s}={q1[s]:g}" for s in syms))
    bad = []
    for o in rec["orders"]:
        moved = q1[o["symbol"]] - q0[o["symbol"]]
        want = o["qty"] if o["side"] == "BUY" else -o["qty"]
        step = b._filter(o["symbol"])[0]  # quantities are floored to the lot step
        if o["notional"] >= config.MIN_ORDER_USDT and abs(moved - want) > max(0.02 * abs(want), step * 1.01):
            bad.append((o["symbol"], want, moved))
    print("RESULT:", "OK, balances moved as planned" if not bad else f"MISMATCH {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
